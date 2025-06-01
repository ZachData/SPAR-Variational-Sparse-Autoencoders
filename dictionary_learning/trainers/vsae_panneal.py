"""
FULLY FIXED implementation of variational autoencoder (VSAE) training scheme with p-annealing.

All fixes applied + additional improvements:
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
11. Better weight initialization - proper tied weights and bias init
12. Clean p-annealing with proper VAE KL divergence
13. FIXED: Use deterministic mu for p-norm penalty (more stable training)
14. FIXED: More conservative p-annealing adaptive scaling
15. ADDED: L0 sparsity tracking and comprehensive diagnostics
16. FIXED: Better handling of p < 1 cases with numerical stability
17. FIXED: Edge case handling in p-annealing schedule
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass
import math

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG
from ..dictionary import Dictionary


@dataclass
class VSAEPAnnealConfig:
    """Configuration for P-Annealing VSAE model."""
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
class VSAEPAnnealTrainingConfig:
    """Configuration for training the P-Annealing VSAE."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # NEW: KL annealing to prevent posterior collapse
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # P-annealing specific parameters
    sparsity_function: str = 'Lp'  # 'Lp' or 'Lp^p'
    anneal_start: Optional[int] = None
    anneal_end: Optional[int] = None
    p_start: float = 1.0
    p_end: float = 0.5
    n_sparsity_updates: int = 10
    sparsity_queue_length: int = 10
    use_deterministic_penalty: bool = True  # NEW: Use mu instead of z for stability
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        # KL annealing to prevent posterior collapse
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)  # 10% of training
            
        if self.anneal_start is None:
            self.anneal_start = max(self.warmup_steps, int(0.15 * self.steps))
        if self.anneal_end is None:
            self.anneal_end = self.steps - 1
            
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class VSAEPAnneal(Dictionary, nn.Module):
    """
    FIXED one-layer variational autoencoder with p-annealing for sparsity control.
    
    Key improvements:
    - Unconstrained mean encoding (no ReLU)
    - Consistent log_var = log(σ²) interpretation throughout
    - Conservative clamping ranges
    - Clean architecture without unnecessary complications
    """

    def __init__(self, config: VSAEPAnnealConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.use_april_update_mode = config.use_april_update_mode
        self.var_flag = config.var_flag
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder and decoder
        self.encoder = nn.Linear(
            self.activation_dim,
            self.dict_size,
            bias=True,
            dtype=dtype,
            device=device
        )
        
        self.decoder = nn.Linear(
            self.dict_size,
            self.activation_dim,
            bias=self.use_april_update_mode,
            dtype=dtype,
            device=device
        )
        
        # Bias parameter for standard mode
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.activation_dim,
                    dtype=dtype,
                    device=device
                )
            )
        
        # Variance encoder (only when learning variance)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim,
                self.dict_size,
                bias=True,
                dtype=dtype,
                device=device
            )
            # REMOVED: No layer norm - let model learn input-dependent variance naturally
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Tied initialization for encoder and decoder
        w = torch.randn(
            self.activation_dim,
            self.dict_size,
            dtype=dtype,
            device=device
        )
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        with torch.no_grad():
            # Set encoder and decoder weights (tied)
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Initialize biases
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                # Initialize variance encoder weights
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                
                # Initialize log_var bias to reasonable value
                # log_var = log(σ²), so log_var = -2 means σ² ≈ 0.135
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction in standard mode."""
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent space.
        
        FIXED: No ReLU on mean - VAE means should be unconstrained for expressiveness
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size] (unconstrained)
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        x_processed = self._preprocess_input(x)
        
        # FIXED: No ReLU constraint on mean - let it be unconstrained
        mu = self.encoder(x_processed)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            # FIXED: No layer norm - direct encoding of log variance
            log_var = self.var_encoder(x_processed)
        
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply reparameterization trick with consistent log_var = log(σ²) interpretation.
        
        FIXED: Consistent interpretation where log_var = log(σ²) throughout
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # FIXED: Conservative clamping range
        # log_var ∈ [-6, 2] means σ² ∈ [0.002, 7.4] - reasonable range
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
        Decode latent features to reconstruction.
        
        Args:
            f: Latent features [batch_size, dict_size]
            
        Returns:
            x_hat: Reconstructed activations [batch_size, activation_dim]
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False, ghost_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for VSAE (raises error if provided)
            
        Returns:
            x_hat: Reconstructed activations
            z: Latent features (if output_features=True)
            mu: Mean (if output_features=True and needed by trainer)
            log_var: Log variance (if output_features=True and var_flag=1)
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEPAnneal")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(z)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            z = z.to(dtype=original_dtype)
            mu = mu.to(dtype=original_dtype)
            if log_var is not None:
                log_var = log_var.to(dtype=original_dtype)
            
            # Return extra info for p-annealing trainer
            if self.var_flag == 1:
                return x_hat, z, mu, log_var
            else:
                return x_hat, z, mu, None
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed KL diagnostics for monitoring training.
        
        Returns:
            Dictionary with KL components and statistics
        """
        with torch.no_grad():
            mu, log_var = self.encode(x)
            
            if self.var_flag == 1 and log_var is not None:
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
                kl_total = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }
    
    def get_sparsity_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """NEW: Get sparsity diagnostics for p-annealing monitoring."""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            
            # L0 sparsity (fraction of near-zero activations)
            l0_mu = (torch.abs(mu) < 1e-3).float().mean()
            l0_z = (torch.abs(z) < 1e-3).float().mean()
            
            # L1 and L2 norms
            l1_mu = torch.abs(mu).sum(dim=-1).mean()
            l2_mu = torch.norm(mu, dim=-1).mean()
            l1_z = torch.abs(z).sum(dim=-1).mean()
            l2_z = torch.norm(z, dim=-1).mean()
            
            return {
                'l0_sparsity_mu': l0_mu,
                'l0_sparsity_z': l0_z,
                'l1_norm_mu': l1_mu,
                'l2_norm_mu': l2_mu,
                'l1_norm_z': l1_z,
                'l2_norm_z': l2_z,
                'mean_abs_mu': torch.abs(mu).mean(),
                'max_abs_mu': torch.abs(mu).max(),
                'mean_abs_z': torch.abs(z).mean(),
                'max_abs_z': torch.abs(z).max(),
            }

    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
                
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
            self.encoder.bias.mul_(norms)
            
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
        config: Optional[VSAEPAnnealConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None
    ) -> 'VSAEPAnneal':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if var_flag is None:
                var_flag = 1 if ("var_encoder.weight" in state_dict or "W_enc_var" in state_dict) else 0
            
            # Determine dimensions and mode
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle legacy format
                activation_dim, dict_size = state_dict.get("W_enc", state_dict["encoder.weight"].T).shape
                use_april_update_mode = "b_dec" in state_dict or "decoder.bias" in state_dict
            
            config = VSAEPAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        
        # Handle legacy parameter naming
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
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: VSAEPAnnealConfig) -> Dict[str, torch.Tensor]:
        """Convert legacy parameter names to current format."""
        converted = {}
        
        # Convert main parameters
        converted["encoder.weight"] = state_dict["W_enc"].T
        converted["encoder.bias"] = state_dict["b_enc"]
        converted["decoder.weight"] = state_dict["W_dec"].T
        
        if config.use_april_update_mode:
            converted["decoder.bias"] = state_dict["b_dec"]
        else:
            converted["bias"] = state_dict["b_dec"]
        
        # Convert variance encoder if present
        if config.var_flag == 1 and "W_enc_var" in state_dict:
            converted["var_encoder.weight"] = state_dict["W_enc_var"].T
            converted["var_encoder.bias"] = state_dict["b_enc_var"]
        
        return converted


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Return a function that computes KL annealing scale factor at a given step.
    Helps prevent posterior collapse in early training.
    
    Args:
        total_steps: Total training steps
        kl_warmup_steps: Steps to warm up KL coefficient from 0 to 1
        
    Returns:
        Function that returns KL scale factor for given step
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


class VSAEPAnnealTrainer(SAETrainer):
    """
    FULLY FIXED trainer for Variational Sparse Autoencoder with p-norm annealing.
    
    Key improvements:
    - FIXED: Proper VAE KL divergence without decoder norm weighting
    - FIXED: Separate KL annealing from sparsity scaling
    - FIXED: Conservative numerical stability
    - FIXED: Use deterministic mu for p-norm penalty (more stable)
    - FIXED: More conservative p-annealing adaptive scaling
    - ADDED: Comprehensive sparsity diagnostics
    - FIXED: Better handling of p < 1 cases
    - Enhanced p-annealing with proper sparsity control
    """
    
    def __init__(
        self,
        model_config: Optional[VSAEPAnnealConfig] = None,
        training_config: Optional[VSAEPAnnealTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility parameters
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        sparsity_penalty: Optional[float] = None,
        var_flag: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = VSAEPAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag or 0,
                use_april_update_mode=kwargs.get('use_april_update_mode', True),
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = VSAEPAnnealTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=sparsity_penalty or 500.0,  # Use sparsity_penalty as kl_coeff for backwards compat
                **{k: v for k, v in kwargs.items() if k in VSAEPAnnealTrainingConfig.__dataclass_fields__}
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEPAnnealTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEPAnneal(model_config)
        self.ae.to(self.device)
        
        # Initialize p-annealing state
        self._init_p_annealing()
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), 
            lr=training_config.lr,
            betas=(0.9, 0.999)
        )
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            None,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps, 
            training_config.sparsity_warmup_steps
        )
        
        # NEW: KL annealing function (separate from sparsity!)
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = [
            'p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'sparsity_coeff',
            'recon_loss', 'total_loss', 'kl_scale', 'sparsity_scale', 
            'p_norm_penalty', 'grad_norm'
        ]
        self._init_logging_state()
    
    def _init_p_annealing(self) -> None:
        """Initialize p-annealing schedule and tracking."""
        config = self.training_config
        
        # Create p-value schedule
        self.p = config.p_start
        self.next_p = None
        
        if isinstance(config.n_sparsity_updates, str) and config.n_sparsity_updates == "continuous":
            self.n_sparsity_updates = config.anneal_end - config.anneal_start + 1
        else:
            self.n_sparsity_updates = config.n_sparsity_updates
            
        self.sparsity_update_steps = torch.linspace(
            config.anneal_start, 
            config.anneal_end, 
            self.n_sparsity_updates, 
            dtype=torch.long
        )
        self.p_values = torch.linspace(
            config.p_start, 
            config.p_end, 
            self.n_sparsity_updates
        )
        
        self.p_step_count = 0
        self.sparsity_coeff = config.kl_coeff  # Start with base coefficient
        self.sparsity_queue = []
        
        # Set initial next_p
        if len(self.p_values) > 1:
            self.next_p = self.p_values[1].item()
        else:
            self.next_p = config.p_end
        
        # Check for duplicates in update steps
        unique_steps, counts = self.sparsity_update_steps.unique(return_counts=True)
        if (counts > 1).any():
            print("Warning! Duplicates in sparsity_update_steps detected!")
    
    def _init_logging_state(self) -> None:
        """Initialize logging state variables."""
        self.kl_loss = 0.0
        self.scaled_kl_loss = 0.0
        self.recon_loss = 0.0
        self.total_loss = 0.0
        self.kl_scale = 1.0
        self.sparsity_scale = 1.0
        self.p_norm_penalty = 0.0
        self.grad_norm = 0.0
    
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute standard VAE KL divergence loss (FIXED - no decoder norm weighting).
        
        For q(z|x) = N(μ, σ²) and p(z) = N(0, I):
        KL[q || p] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # FIXED: Conservative clamping range
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # Standard VAE KL divergence
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
    
    def _compute_p_norm_penalty(self, mu: torch.Tensor, log_var: Optional[torch.Tensor], p: float) -> torch.Tensor:
        """
        FIXED: Compute p-norm penalty using deterministic mu for more stable training.
        
        Args:
            mu: Mean of latent distribution (deterministic)
            log_var: Log variance (if learned) - not used in penalty for stability
            p: P-norm value for penalty
            
        Returns:
            P-norm penalty (always non-negative)
        """
        # FIXED: Use deterministic mu instead of stochastic z for more stable gradients
        if self.training_config.use_deterministic_penalty:
            penalty_input = mu
        else:
            # Option to use sampled z if desired (less stable but more theoretically consistent)
            penalty_input = self.ae.reparameterize(mu, log_var)
        
        penalty_input = penalty_input.to(dtype=torch.float32)
        
        # Clamp to prevent extreme values
        penalty_input_clamped = torch.clamp(penalty_input, min=-100.0, max=100.0)
        
        if p == 1.0:
            # L1 penalty: sum of absolute values
            penalty = torch.sum(torch.abs(penalty_input_clamped), dim=1).mean()
        elif p == 2.0:
            # L2 penalty: sum of squares
            penalty = torch.sum(penalty_input_clamped.pow(2), dim=1).mean()
        else:
            # FIXED: Better handling of p < 1 cases with numerical stability
            if p < 1.0:
                # For p < 1, add small epsilon to prevent numerical issues
                eps = 1e-8
                if self.training_config.sparsity_function == 'Lp^p':
                    penalty = torch.sum((torch.abs(penalty_input_clamped) + eps).pow(p), dim=1).mean()
                else:  # 'Lp'
                    penalty = torch.pow(torch.sum((torch.abs(penalty_input_clamped) + eps).pow(p), dim=1), 1.0/p).mean()
            else:
                # p >= 1 case
                if self.training_config.sparsity_function == 'Lp^p':
                    penalty = torch.sum(torch.abs(penalty_input_clamped).pow(p), dim=1).mean()
                else:  # 'Lp'
                    penalty = torch.pow(torch.sum(torch.abs(penalty_input_clamped).pow(p), dim=1), 1.0/p).mean()
        
        # Ensure penalty is non-negative and bounded
        penalty = torch.clamp(penalty, min=0.0, max=1e6)
        
        return penalty
    
    def _update_p_annealing(self, step: int, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> None:
        """FIXED: Update p-annealing schedule with more conservative adaptive scaling."""
        # Build up penalty queue for adaptive scaling
        if self.next_p is not None and len(self.sparsity_queue) < self.training_config.sparsity_queue_length:
            with torch.no_grad():
                penalty_current = self._compute_p_norm_penalty(mu.detach(), log_var.detach() if log_var is not None else None, p=self.p)
                penalty_next = self._compute_p_norm_penalty(mu.detach(), log_var.detach() if log_var is not None else None, p=self.next_p)
                self.sparsity_queue.append([penalty_current.item(), penalty_next.item()])
        
        # Update p-value if we've reached an update step
        if step in self.sparsity_update_steps and self.p_step_count < len(self.p_values):
            # FIXED: More conservative adaptive scaling
            if self.next_p is not None and len(self.sparsity_queue) >= 3:  # Need minimum samples
                try:
                    penalties_current = torch.tensor([i[0] for i in self.sparsity_queue])
                    penalties_next = torch.tensor([i[1] for i in self.sparsity_queue])
                    
                    # Use mean instead of median for smoother adaptation
                    mean_current = penalties_current.mean()
                    mean_next = penalties_next.mean()
                    
                    # FIXED: More conservative thresholds and adaptation
                    if mean_next > 1e-4 and mean_current > 1e-4:  # Less strict threshold
                        ratio = (mean_current / mean_next).item()
                        # FIXED: More conservative clamping (less aggressive adjustment)
                        ratio = max(0.5, min(ratio, 2.0))  # Limit adjustment to 2x change
                        
                        # FIXED: Smooth the adjustment with exponential moving average
                        new_coeff = self.sparsity_coeff * ratio
                        self.sparsity_coeff = 0.8 * self.sparsity_coeff + 0.2 * new_coeff
                        
                        # FIXED: More reasonable bounds
                        self.sparsity_coeff = max(1.0, min(self.sparsity_coeff, 5000.0))
                        
                except Exception as e:
                    print(f"Warning: Could not adapt sparsity coefficient: {e}")
                    
            # Update p-value
            self.p = self.p_values[self.p_step_count].item()
            
            # FIXED: Better handling of next_p edge cases
            if self.p_step_count < self.n_sparsity_updates - 1:
                self.next_p = self.p_values[self.p_step_count + 1].item()
            else:
                self.next_p = self.training_config.p_end  # Use final p_end instead of None
                
            self.p_step_count += 1
            
            # Clear queue after update
            self.sparsity_queue = []
            
            print(f"Step {step}: Updated p to {self.p:.4f}, next_p: {self.next_p:.4f}, sparsity_coeff: {self.sparsity_coeff:.4f}")
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """
        Compute total loss with FIXED KL divergence and separate p-norm penalty.
        
        Key improvements:
        - FIXED: Standard VAE KL divergence without decoder norm weighting
        - FIXED: Separate KL annealing from sparsity scaling
        - FIXED: Use deterministic mu for p-norm penalty
        - Clean separation of VAE loss and p-norm sparsity penalty
        """
        sparsity_scale = self.sparsity_warmup_fn(step)  # For p-norm penalty
        kl_scale = self.kl_warmup_fn(step)  # FIXED: Separate KL annealing
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        if self.ae.var_flag == 1:
            x_hat, z, mu, log_var = self.ae(x, output_features=True)
        else:
            x_hat, z, mu, log_var = self.ae(x, output_features=True)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        self.recon_loss = recon_loss.item()
        
        # FIXED: Standard VAE KL divergence (no decoder norm weighting)
        kl_loss = self._compute_kl_loss(mu, log_var)
        self.kl_loss = kl_loss.item()
        
        # FIXED: P-norm sparsity penalty using deterministic mu
        p_norm_penalty = self._compute_p_norm_penalty(mu, log_var, p=self.p)
        self.p_norm_penalty = p_norm_penalty.item()
        
        # Update p-annealing schedule
        self._update_p_annealing(step, mu.detach(), log_var.detach() if log_var is not None else None)
        
        # FIXED: Separate scaling for KL and sparsity
        scaled_kl_loss = kl_loss * self.training_config.kl_coeff * kl_scale
        scaled_p_norm_penalty = p_norm_penalty * self.sparsity_coeff * sparsity_scale
        
        self.scaled_kl_loss = scaled_kl_loss.item()
        self.kl_scale = kl_scale
        self.sparsity_scale = sparsity_scale
        
        # Total loss
        total_loss = recon_loss + scaled_kl_loss + scaled_p_norm_penalty
        self.total_loss = total_loss.item()
        
        # Ensure output is in original dtype
        total_loss = total_loss.to(dtype=original_dtype)
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        kl_diagnostics = self.ae.get_kl_diagnostics(x)
        sparsity_diagnostics = self.ae.get_sparsity_diagnostics(x)
        
        return LossLog(
            x, x_hat.to(dtype=original_dtype), z.to(dtype=original_dtype),
            {
                'l2_loss': torch.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': self.recon_loss,
                'kl_loss': self.kl_loss,
                'scaled_kl_loss': self.scaled_kl_loss,
                'p_norm_penalty': self.p_norm_penalty,
                'scaled_p_norm_penalty': scaled_p_norm_penalty.item(),
                'loss': self.total_loss,
                'p': self.p,
                'next_p': self.next_p,
                'sparsity_coeff': self.sparsity_coeff,
                'kl_scale': self.kl_scale,
                'sparsity_scale': self.sparsity_scale,
                'grad_norm': self.grad_norm,
                # Additional diagnostics
                **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()},
                **{k: v.item() if torch.is_tensor(v) else v for k, v in sparsity_diagnostics.items()}
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with improved gradient monitoring."""
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # FIXED: Monitor gradient norms
        total_norm = 0.0
        for p in self.ae.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.grad_norm = total_norm ** 0.5
        
        # FIXED: Better gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving (JSON serializable)."""
        return {
            'dict_class': 'VSAEPAnneal',
            'trainer_class': 'VSAEPAnnealTrainer',
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
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'sparsity_function': self.training_config.sparsity_function,
            'anneal_start': self.training_config.anneal_start,
            'anneal_end': self.training_config.anneal_end,
            'p_start': self.training_config.p_start,
            'p_end': self.training_config.p_end,
            'n_sparsity_updates': self.training_config.n_sparsity_updates,
            'sparsity_queue_length': self.training_config.sparsity_queue_length,
            'use_deterministic_penalty': self.training_config.use_deterministic_penalty,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }