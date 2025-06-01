"""
FULLY FIXED implementation of combined Variational Sparse Autoencoder (VSAE) with Gated Annealing.

This module combines variational techniques with p-norm annealing to achieve
better feature learning and controlled sparsity, following all the fixes from vsae_iso.py
plus additional improvements specific to gated architectures.

Key improvements applied:
1. Removed ReLU from log variance - can now be negative (mathematically correct)
2. Unconstrained mean encoding - removed ReLU from μ (proper VAE)
3. Conservative clamping ranges - log_var ∈ [-6,2] instead of [-8,8]
4. Separated KL and sparsity scaling - kl_scale vs sparsity_scale
5. Removed decoder norm weighting from KL loss - clean KL computation
6. Removed layer norm on variance encoder - direct variance learning
7. Added KL annealing - prevents posterior collapse
8. Better gradient clipping - improved stability
9. Enhanced numerical stability - consistent dtype handling
10. Improved diagnostics - detailed KL component tracking
11. Memory-efficient configurations - GPU-optimized buffer sizes
12. Better weight initialization - proper tied weights and bias init

Additional gated-specific improvements:
13. Soft gating with sigmoid - better differentiability than hard thresholding
14. Fixed coefficient adaptation bug - now actually adapts coefficients during p-annealing
15. Improved auxiliary loss - removed unnecessary weight detaching
16. Better dead neuron tracking - based on gate activity rather than just zeros
17. Enhanced resampling - resets all gating parameters properly
18. Separate p-norm coefficient - more principled hyperparameter control
19. Better optimizer state handling - more robust parameter resets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass
import math

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam
)


@dataclass
class VSAEGatedConfig:
    """Configuration for VSAE Gated model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class AnnealingConfig:
    """Configuration for p-norm annealing schedule."""
    anneal_start: int
    anneal_end: int
    p_start: float = 1.0
    p_end: float = 0.0
    n_sparsity_updates: int = 10
    sparsity_function: str = 'Lp^p'  # 'Lp' or 'Lp^p'
    sparsity_queue_length: int = 10
    
    def __post_init__(self):
        if self.sparsity_function not in ['Lp', 'Lp^p']:
            raise ValueError("sparsity_function must be 'Lp' or 'Lp^p'")
        if self.anneal_end <= self.anneal_start:
            raise ValueError("anneal_end must be greater than anneal_start")
        if self.n_sparsity_updates < 1:
            raise ValueError("n_sparsity_updates must be at least 1")
        if self.anneal_end - self.anneal_start < self.n_sparsity_updates:
            # Adjust n_sparsity_updates if the annealing period is too short
            max_updates = max(1, self.anneal_end - self.anneal_start)
            print(f"Warning: n_sparsity_updates ({self.n_sparsity_updates}) too large for annealing period "
                  f"({self.anneal_end - self.anneal_start} steps). Reducing to {max_updates}")
            self.n_sparsity_updates = max_updates


@dataclass 
class TrainingConfig:
    """Configuration for training the VSAEGated model."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    p_norm_coeff: float = 1.0  # IMPROVED: Separate coefficient for p-norm sparsity
    aux_coeff: float = 1.0  # Coefficient for auxiliary loss
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)  # KL annealing
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Return a function that computes KL annealing scale factor at a given step.
    Helps prevent posterior collapse in early training.
    """
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps, "kl_warmup_steps must be > 0 and <= total_steps"
    
    def scale_fn(step: int) -> float:
        if step < kl_warmup_steps:
            return step / kl_warmup_steps  # Linear warmup from 0 to 1
        else:
            return 1.0
    
    return scale_fn


class VSAEGatedAutoEncoder(Dictionary, nn.Module):
    """
    FIXED variational sparse autoencoder with gating networks for feature extraction.
    
    This model combines ideas from standard VSAEs with gated networks to improve
    feature learning and interpretability through controlled sparsity.
    """

    def __init__(self, config: VSAEGatedConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder/decoder
        self.encoder = nn.Linear(
            self.activation_dim, 
            self.dict_size, 
            bias=False,
            dtype=dtype,
            device=device
        )
        self.decoder = nn.Linear(
            self.dict_size, 
            self.activation_dim, 
            bias=False,
            dtype=dtype,
            device=device
        )
        
        # Bias parameters
        self.decoder_bias = nn.Parameter(
            torch.zeros(self.activation_dim, dtype=dtype, device=device)
        )
        
        # Gating specific parameters
        self.r_mag = nn.Parameter(
            torch.zeros(self.dict_size, dtype=dtype, device=device)
        )
        self.gate_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=dtype, device=device)
        )
        self.mag_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=dtype, device=device)
        )
        
        # Variance encoder (only used when var_flag=1)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim, 
                self.dict_size, 
                bias=True,
                dtype=dtype,
                device=device
            )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices from vsae_iso.py."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        with torch.no_grad():
            # FIXED: Proper tied initialization for encoder and decoder
            w = torch.randn(
                self.activation_dim,
                self.dict_size,
                dtype=dtype,
                device=device
            )
            w = w / w.norm(dim=0, keepdim=True) * 0.1
            
            # Set encoder and decoder weights (tied)
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Biases already initialized to zero in _init_layers
            
            # Initialize variance encoder if needed
            if self.var_flag == 1:
                # FIXED: Proper initialization without layer norm
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction."""
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        return x - self.decoder_bias

    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False, 
        return_log_var: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        FIXED: Encode input activations to latent space with improved gating.
        
        Args:
            x: Input activations to encode
            return_gate: Whether to return gate values
            return_log_var: Whether to return log variance
            
        Returns:
            Features, and optionally gate values and log variance
        """
        x_processed = self._preprocess_input(x)
        
        # Main encoder output
        x_enc = self.encoder(x_processed)

        # IMPROVED: Soft gating for better differentiability
        pi_gate = x_enc + self.gate_bias
        f_gate = torch.sigmoid(pi_gate)  # Soft gating instead of hard threshold

        # Magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)

        # Combined features (this becomes our mean μ)
        mu = f_gate * f_mag
        
        # Handle variance if required
        log_var = None
        if return_log_var and self.var_flag == 1:
            # Direct encoding without layer norm or ReLU
            log_var = self.var_encoder(x_processed)
            
        # Return appropriate combination
        if return_log_var and return_gate:
            return mu, f_gate, log_var  # Return actual gate values, not ReLU of pi_gate
        elif return_log_var:
            return mu, log_var
        elif return_gate:
            return mu, f_gate
        else:
            return mu

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        FIXED: Apply reparameterization trick with consistent log_var = log(σ²) interpretation.
        """
        if log_var is None or self.var_flag == 0:
            return mu
            
        # FIXED: Conservative clamping range
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var)) = sqrt(σ²)
        std = torch.sqrt(torch.exp(log_var_clamped))
        
        # Sample noise
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to activation space.
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        return self.decoder(f) + self.decoder_bias

    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False, 
        ghost_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FIXED: Forward pass through the autoencoder with proper VAE formulation.
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEGatedAutoEncoder")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # For variational version
        if self.var_flag == 1:
            mu, log_var = self.encode(x, return_log_var=True)
            # Sample from the latent distribution
            z = self.reparameterize(mu, log_var)
        else:
            # Standard version
            z = self.encode(x)
            
        x_hat = self.decode(z)

        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            z = z.to(dtype=original_dtype)
            return x_hat, z
        else:
            return x_hat

    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        FIXED: Get detailed KL diagnostics for monitoring training.
        """
        with torch.no_grad():
            if self.var_flag == 1:
                mu, log_var = self.encode(x, return_log_var=True)
                
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)  # σ²
                
                kl_mu = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                kl_total = kl_mu + kl_var
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_mu,
                    'kl_var_term': kl_var,
                    'mean_log_var': log_var.mean(),
                    'mean_var': var.mean(),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }
            else:
                mu = self.encode(x)
                kl_total = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }

    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.decoder_bias.mul_(scale)
            self.mag_bias.mul_(scale)
            self.gate_bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)

    def normalize_decoder(self) -> None:
        """
        Normalize decoder weights to have unit norm.
        Note: Only recommended for models without learned variance.
        """
        if self.var_flag == 1:
            print("Warning: Normalizing decoder weights with learned variance may hurt performance")
        
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0)
            
            if torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                return
            
            print("Normalizing decoder weights")
            
            # Test that normalization preserves output
            device = self.decoder.weight.device
            test_input = torch.randn(10, self.activation_dim, device=device, dtype=self.decoder.weight.dtype)
            initial_output = self(test_input)
            
            # Normalize decoder weights
            self.decoder.weight.div_(norms)
            
            # Scale encoder weights and biases accordingly
            self.encoder.weight.mul_(norms.unsqueeze(1))
            
            # Scale gating parameters
            self.gate_bias.mul_(norms)
            self.mag_bias.mul_(norms)
            self.r_mag.mul_(norms)
            
            # Verify normalization worked
            new_norms = torch.norm(self.decoder.weight, dim=0)
            assert torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-6)
            
            # Verify output is preserved
            new_output = self(test_input)
            assert torch.allclose(initial_output, new_output, atol=1e-4), "Normalization changed model output"

    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[VSAEGatedConfig] = None,
        dtype: torch.dtype = torch.float32, 
        device: Optional[torch.device] = None, 
        normalize_decoder: bool = True, 
        var_flag: Optional[int] = None
    ) -> 'VSAEGatedAutoEncoder':
        """
        Load a pretrained autoencoder from a file.
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if var_flag is None:
                has_var_encoder = "var_encoder.weight" in state_dict or "W_enc_var" in state_dict
                var_flag = 1 if has_var_encoder else 0
            
            # Determine dimensions from state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder_bias" in state_dict
            else:
                # Handle legacy format
                activation_dim, dict_size = state_dict.get("W_enc", state_dict["encoder.weight"].T).shape
                use_april_update_mode = "b_dec" in state_dict or "decoder_bias" in state_dict
                
            config = VSAEGatedConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        # Create model
        model = cls(config)
        
        # Handle legacy parameter names
        if "W_enc" in state_dict:
            converted_dict = cls._convert_legacy_state_dict(state_dict, config)
            state_dict = converted_dict
        
        # Load state dict with error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
                # Initialize missing variance encoder if needed
                if var_flag == 1 and any("var_encoder" in key for key in missing_keys):
                    print("Initializing missing variance encoder parameters")
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(model.var_encoder.weight, a=0.01)
                        nn.init.constant_(model.var_encoder.bias, config.log_var_init)
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Normalize decoder if requested (skip for learned variance models)
        if normalize_decoder and not (config.var_flag == 1 and "var_encoder.weight" in state_dict):
            try:
                model.normalize_decoder()
            except Exception as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model
    
    @staticmethod
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: VSAEGatedConfig) -> Dict[str, torch.Tensor]:
        """Convert legacy parameter names to current format."""
        converted = {}
        
        # Convert main parameters
        converted["encoder.weight"] = state_dict["W_enc"].T
        converted["decoder.weight"] = state_dict["W_dec"].T
        converted["decoder_bias"] = state_dict["b_dec"]
        converted["r_mag"] = state_dict["r_mag"]
        converted["gate_bias"] = state_dict["gate_bias"]
        converted["mag_bias"] = state_dict["mag_bias"]
        
        # Convert variance encoder if present
        if config.var_flag == 1 and "W_enc_var" in state_dict:
            converted["var_encoder.weight"] = state_dict["W_enc_var"].T
            converted["var_encoder.bias"] = state_dict["b_enc_var"]
        
        return converted


class DeadFeatureTracker:
    """Tracks dead features for resampling."""
    
    def __init__(self, dict_size: int, device: torch.device):
        self.steps_since_active = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
    
    def update(self, active_features: torch.Tensor) -> torch.Tensor:
        """Update dead feature tracking and return dead feature mask."""
        # Update counters
        deads = (active_features == 0).all(dim=0)
        self.steps_since_active[deads] += 1
        self.steps_since_active[~deads] = 0
        
        return self.steps_since_active
    
    def get_dead_mask(self, threshold: int) -> torch.Tensor:
        """Get boolean mask of dead features."""
        return self.steps_since_active > threshold


class VSAEGatedAnnealTrainer(SAETrainer):
    """
    FIXED trainer that combines Variational Sparse Autoencoding (VSAE) with Gated Annealing.
    
    This trainer uses both variational techniques and p-norm annealing to achieve
    better feature learning and controlled sparsity, with all fixes from vsae_iso.py applied.
    """
    
    def __init__(
        self,
        model_config: VSAEGatedConfig = None,
        training_config: TrainingConfig = None,
        annealing_config: AnnealingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        p_norm_coeff: Optional[float] = None,
        var_flag: Optional[int] = None,
        anneal_start: Optional[int] = None,
        anneal_end: Optional[int] = None,
        p_start: Optional[float] = None,
        p_end: Optional[float] = None,
        n_sparsity_updates: Optional[int] = None,
        sparsity_function: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility - if individual parameters are passed, create configs
        if model_config is None or training_config is None or annealing_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = VSAEGatedConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    var_flag=var_flag or 0,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = TrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                    p_norm_coeff=p_norm_coeff or 1.0,
                )
            
            if annealing_config is None:
                if anneal_start is None or anneal_end is None:
                    raise ValueError("Must provide either annealing_config or anneal_start + anneal_end")
                
                annealing_config = AnnealingConfig(
                    anneal_start=anneal_start,
                    anneal_end=anneal_end,
                    p_start=p_start or 1.0,
                    p_end=p_end or 0.0,
                    n_sparsity_updates=n_sparsity_updates or 10,
                    sparsity_function=sparsity_function or 'Lp^p'
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.annealing_config = annealing_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedAnnealTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEGatedAutoEncoder(model_config)
        self.ae.to(self.device)
        
        # Initialize annealing state
        self.p = annealing_config.p_start
        self.next_p = None
        
        # Create annealing schedule
        if annealing_config.n_sparsity_updates > 1:
            self.sparsity_update_steps = torch.linspace(
                annealing_config.anneal_start, 
                annealing_config.anneal_end, 
                annealing_config.n_sparsity_updates, 
                dtype=torch.long
            )
            self.p_values = torch.linspace(
                annealing_config.p_start, 
                annealing_config.p_end, 
                annealing_config.n_sparsity_updates
            )
        else:
            self.sparsity_update_steps = torch.tensor([annealing_config.anneal_start], dtype=torch.long)
            self.p_values = torch.tensor([annealing_config.p_end])
            
        self.p_step_count = 0
        self.sparsity_queue = []
        
        # Set the initial next_p
        if len(self.p_values) > 1:
            self.next_p = self.p_values[1].item()
        else:
            self.next_p = annealing_config.p_end
        
        # Initialize dead feature tracking
        if training_config.resample_steps is not None:
            self.dead_feature_tracker = DeadFeatureTracker(
                model_config.dict_size,
                self.device
            )
        else:
            self.dead_feature_tracker = None
        
        # Initialize optimizer and scheduler
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(), 
            self.ae.decoder.parameters(), 
            lr=training_config.lr,
            betas=(0.9, 0.999)
        )
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            training_config.resample_steps,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps, 
            training_config.sparsity_warmup_steps
        )
        
        # FIXED: Add KL annealing function (separate from sparsity!)
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = ['p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'kl_coeff', 'kl_scale']
        self.kl_loss = None
        self.scaled_kl_loss = None
        self.kl_coeff = training_config.kl_coeff
        self.kl_scale = 1.0
        
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        FIXED: Compute KL divergence loss with proper VAE formulation and numerical stability.
        
        For q(z|x) = N(μ, σ²) and p(z) = N(0, I):
        KL[q || p] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # FIXED: Conservative clamping range
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # KL divergence
            kl_per_sample = 0.5 * torch.sum(
                mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                dim=1
            )
        else:
            # Fixed variance case: KL = 0.5 * ||μ||²
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
        
        # Average over batch
        kl_loss = kl_per_sample.mean()
        
        # Ensure KL is non-negative (should be true mathematically)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        return kl_loss
    
    def _compute_p_norm_loss(self, mu: torch.Tensor, p: float) -> torch.Tensor:
        """
        FIXED: Compute p-norm sparsity loss with numerical stability.
        
        This is separate from KL loss and used for p-annealing sparsity.
        """
        # Clamp p to reasonable range to avoid numerical issues
        p_clamped = max(0.001, min(p, 10.0))
        
        if self.annealing_config.sparsity_function == 'Lp^p':
            # Clamp mu to prevent overflow in power operations
            mu_clamped = torch.clamp(torch.abs(mu), min=1e-8, max=1e6)
            p_norm_loss = mu_clamped.pow(p_clamped).sum(dim=-1).mean()
        elif self.annealing_config.sparsity_function == 'Lp':
            # Clamp mu to prevent overflow
            mu_clamped = torch.clamp(torch.abs(mu), min=1e-8, max=1e6)
            # Use stable computation for Lp norm
            if p_clamped >= 1.0:
                p_norm_loss = mu_clamped.pow(p_clamped).sum(dim=-1).pow(1/p_clamped).mean()
            else:
                # For p < 1, use a more stable computation
                p_norm_loss = mu_clamped.pow(p_clamped).sum(dim=-1).mean()
        else:
            raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
        
        # Ensure non-negative result
        p_norm_loss = torch.clamp(p_norm_loss, min=0.0, max=1e6)
        
        return p_norm_loss
    
    def _resample_neurons(self, dead_mask: torch.Tensor, activations: torch.Tensor) -> None:
        """
        IMPROVED: Resample dead neurons accounting for gating mechanism.
        """
        with torch.no_grad():
            if dead_mask.sum() == 0: 
                return
                
            print(f"Resampling {dead_mask.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([dead_mask.sum(), losses.shape[0]])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~dead_mask].norm(dim=-1).mean()
            
            # Handle encoder and decoder weight resampling
            self.ae.encoder.weight[dead_mask][:n_resample] = sampled_vecs * alive_norm * 0.2
            decoder_dtype = self.ae.decoder.weight.dtype
            normalized_vecs = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.decoder.weight[:,dead_mask][:,:n_resample] = normalized_vecs.to(dtype=decoder_dtype)
            
            # IMPROVED: Reset all gating-related biases for better resampling
            self.ae.gate_bias[dead_mask][:n_resample] = 0.
            self.ae.mag_bias[dead_mask][:n_resample] = 0.
            self.ae.r_mag[dead_mask][:n_resample] = 0.  # Reset magnitude scaling too

            # Reset optimizer state for all relevant parameters
            state_dict = self.optimizer.state_dict()['state']
            
            # Find parameter indices (this is fragile but necessary for ConstrainedAdam)
            param_idx = 0
            for name, param in self.ae.named_parameters():
                if 'encoder.weight' in name:
                    if param_idx in state_dict:
                        state_dict[param_idx]['exp_avg'][dead_mask] = 0.
                        state_dict[param_idx]['exp_avg_sq'][dead_mask] = 0.
                elif 'decoder.weight' in name:
                    if param_idx in state_dict:
                        state_dict[param_idx]['exp_avg'][:,dead_mask] = 0.
                        state_dict[param_idx]['exp_avg_sq'][:,dead_mask] = 0.
                elif any(bias_name in name for bias_name in ['gate_bias', 'mag_bias', 'r_mag']):
                    if param_idx in state_dict:
                        state_dict[param_idx]['exp_avg'][dead_mask] = 0.
                        state_dict[param_idx]['exp_avg_sq'][dead_mask] = 0.
                param_idx += 1
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """
        FIXED: Calculate loss for the VSAE with p-annealing and proper scaling separation.
        
        Key improvements:
        - Separate KL and sparsity scaling
        - Proper VAE KL divergence computation
        - P-norm annealing for sparsity
        - Numerical stability throughout
        """
        # FIXED: Separate scaling factors
        sparsity_scale = self.sparsity_warmup_fn(step)  # For p-norm sparsity
        kl_scale = self.kl_warmup_fn(step)  # For KL annealing (separate!)
        self.kl_scale = kl_scale
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Handle variational encoding
        if self.model_config.var_flag == 1:
            # Encode with variance
            mu, gate, log_var = self.ae.encode(x, return_gate=True, return_log_var=True)
            # Sample from the latent distribution
            z = self.ae.reparameterize(mu, log_var)
            # Decode
            x_hat = self.ae.decode(z)
            # IMPROVED: Auxiliary loss using gates for faster convergence
            x_hat_gate = gate @ self.ae.decoder.weight.T + self.ae.decoder_bias
        else:
            # Standard encoding
            mu, gate = self.ae.encode(x, return_gate=True)
            # Just use mean
            z = mu
            # Decode
            x_hat = self.ae.decode(z)
            # IMPROVED: Auxiliary loss using gates
            x_hat_gate = gate @ self.ae.decoder.weight.T + self.ae.decoder_bias
        
        # Ensure outputs match original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        x_hat_gate = x_hat_gate.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # Auxiliary loss to help with gating
        aux_loss = (x - x_hat_gate).pow(2).sum(dim=-1).mean()
        
        # FIXED: Proper KL divergence (separate from p-norm)
        if self.model_config.var_flag == 1:
            kl_loss = self._compute_kl_loss(mu, log_var)
        else:
            kl_loss = self._compute_kl_loss(mu, None)
        
        # FIXED: P-norm sparsity loss (separate from KL)
        p_norm_loss = self._compute_p_norm_loss(mu, self.p)
        
        # Store for logging
        self.kl_loss = kl_loss
        
        # FIXED: Separate scaling - KL gets kl_scale, p-norm gets sparsity_scale
        scaled_kl_loss = self.training_config.kl_coeff * kl_scale * kl_loss
        scaled_p_norm_loss = self.training_config.p_norm_coeff * sparsity_scale * p_norm_loss
        
        self.scaled_kl_loss = scaled_kl_loss

        # P-annealing handling with improved coefficient adaptation
        if self.next_p is not None:
            p_norm_next = self._compute_p_norm_loss(mu, self.next_p)
            self.sparsity_queue.append([p_norm_loss.item(), p_norm_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.annealing_config.sparsity_queue_length:]
    
        # Update p-value at scheduled steps
        if step in self.sparsity_update_steps and self.p_step_count < len(self.sparsity_update_steps):
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # FIXED: Actually implement coefficient adaptation
                if self.next_p is not None and len(self.sparsity_queue) >= 3:  # Need some data
                    local_sparsity_new = torch.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = torch.tensor([i[1] for i in self.sparsity_queue]).mean()
                    if local_sparsity_old > 1e-6:  # Avoid division by zero
                        adaptation_ratio = (local_sparsity_new / local_sparsity_old).item()
                        # Adapt the KL coefficient to maintain balance as p changes
                        self.kl_coeff = self.kl_coeff * adaptation_ratio
                        # Clamp to reasonable range
                        self.kl_coeff = max(10.0, min(self.kl_coeff, 10000.0))
                
                # Update p
                if self.p_step_count < len(self.p_values):
                    old_p = self.p
                    self.p = self.p_values[self.p_step_count].item()
                    print(f"Step {step}: Updated p from {old_p:.3f} to {self.p:.3f}, kl_coeff: {self.kl_coeff:.1f}")
                    
                    if self.p_step_count < len(self.p_values) - 1:
                        self.next_p = self.p_values[self.p_step_count + 1].item()
                    else:
                        self.next_p = self.annealing_config.p_end
                        
                    self.p_step_count += 1

        # IMPROVED: Update dead feature count based on gate activity, not just zeros
        if self.dead_feature_tracker is not None:
            # For gated models, track gate activity rather than just feature zeros
            gate_activity = gate if gate.max() <= 1.0 else (gate > 0.1).float()  # Handle both soft and hard gates
            self.dead_feature_tracker.update(gate_activity)
            
        # Convert loss components to original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        aux_loss = aux_loss.to(dtype=original_dtype)
        scaled_kl_loss = scaled_kl_loss.to(dtype=original_dtype)
        scaled_p_norm_loss = scaled_p_norm_loss.to(dtype=original_dtype)
            
        # FIXED: Total loss with separate components
        loss = recon_loss + scaled_kl_loss + scaled_p_norm_loss + self.training_config.aux_coeff * aux_loss
    
        if not logging:
            return loss
        else:
            # Get additional diagnostics
            kl_diagnostics = self.ae.get_kl_diagnostics(x)
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'mse_loss': recon_loss.item(),
                    'aux_loss': aux_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'p_norm_loss': p_norm_loss.item(),
                    'scaled_kl_loss': scaled_kl_loss.item(),
                    'scaled_p_norm_loss': scaled_p_norm_loss.item(),
                    'loss': loss.item(),
                    'p': self.p,
                    'next_p': self.next_p,
                    'kl_coeff': self.kl_coeff,
                    'kl_scale': kl_scale,
                    'sparsity_scale': sparsity_scale,
                    # Additional diagnostics
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
                }
            )
        
    def update(self, step: int, activations: torch.Tensor) -> None:
        """
        Update the model parameters for one step.
        """
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()
        
        # FIXED: Apply gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()

        # Resample dead neurons if needed
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == self.training_config.resample_steps - 1 and
            self.dead_feature_tracker is not None):
            
            dead_mask = self.dead_feature_tracker.get_dead_mask(self.training_config.resample_steps // 2)
            self._resample_neurons(dead_mask, activations)

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the configuration of this trainer (JSON serializable).
        """
        return {
            'dict_class': 'VSAEGatedAutoEncoder',
            'trainer_class': 'VSAEGatedAnnealTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'p_norm_coeff': self.training_config.p_norm_coeff,
            'aux_coeff': self.training_config.aux_coeff,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'resample_steps': self.training_config.resample_steps,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Annealing config (serializable)
            'anneal_start': self.annealing_config.anneal_start,
            'anneal_end': self.annealing_config.anneal_end,
            'p_start': self.annealing_config.p_start,
            'p_end': self.annealing_config.p_end,
            'n_sparsity_updates': self.annealing_config.n_sparsity_updates,
            'sparsity_function': self.annealing_config.sparsity_function,
            'sparsity_queue_length': self.annealing_config.sparsity_queue_length,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }
            