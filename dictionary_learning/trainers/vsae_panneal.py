"""
Robust implementation of variational autoencoder (VSAE) training scheme with p-annealing.

Key improvements for robustness:
1. Simplified and more stable p-annealing logic
2. Better error handling and edge case management
3. Conservative numerical stability measures
4. Consistent patterns with vsae_iso.py
5. Comprehensive configuration validation
6. Improved documentation and type hints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
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
    """Enhanced training configuration for P-Annealing VSAE with validation."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    sparsity_coeff: float = 100.0  # Base sparsity coefficient for p-norm penalty
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # P-annealing specific parameters (simplified)
    sparsity_function: str = 'Lp'  # 'Lp' or 'Lp^p'
    anneal_start: Optional[int] = None
    anneal_end: Optional[int] = None
    p_start: float = 1.0
    p_end: float = 0.5
    n_sparsity_updates: int = 10
    use_adaptive_scaling: bool = False  # Simplified: disable complex adaptive scaling by default
    use_deterministic_penalty: bool = True  # Use mu instead of z for stability
    
    def __post_init__(self):
        """Set derived configuration values with validation."""
        # Basic validation
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.p_start < self.p_end:
            raise ValueError("p_start must be >= p_end for annealing")
        if self.sparsity_function not in ['Lp', 'Lp^p']:
            raise ValueError("sparsity_function must be 'Lp' or 'Lp^p'")
        
        # Set defaults
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)  # 10% of training
            
        # P-annealing schedule
        if self.anneal_start is None:
            self.anneal_start = max(self.warmup_steps, int(0.15 * self.steps))
        if self.anneal_end is None:
            self.anneal_end = min(int(0.9 * self.steps), self.steps - 1)
            
        # Ensure valid annealing range
        if self.anneal_start >= self.anneal_end:
            self.anneal_end = self.steps - 1
            print(f"Warning: Adjusted anneal_end to {self.anneal_end}")
            
        # Decay start
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= min_decay_start:
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class VSAEPAnneal(Dictionary, nn.Module):
    """
    Robust one-layer variational autoencoder with p-annealing for sparsity control.
    
    Follows the same robust patterns as VSAEIsoGaussian with added p-annealing functionality.
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
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction in standard mode."""
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent space.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size] (unconstrained)
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        x_processed = self._preprocess_input(x)
        
        # Unconstrained mean encoding
        mu = self.encoder(x_processed)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
        
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply reparameterization trick with consistent log_var = log(σ²) interpretation.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # Conservative clamping range
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var))
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
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False, ghost_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for VSAE (raises error if provided)
            
        Returns:
            x_hat: Reconstructed activations
            f: Latent features (if output_features=True)
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
            # Return x_hat and z (latent features) for compatibility with evaluation code
            return x_hat, z
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed KL diagnostics for monitoring training."""
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
        """Get sparsity diagnostics for p-annealing monitoring."""
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
        """Load pretrained model from checkpoint with robust error handling."""
        try:
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
        
        if config is None:
            # Auto-detect configuration from state dict
            try:
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
            except Exception as e:
                raise RuntimeError(f"Failed to auto-detect model configuration: {e}")
        
        model = cls(config)
        
        # Handle legacy parameter naming
        if "W_enc" in state_dict:
            try:
                converted_dict = cls._convert_legacy_state_dict(state_dict, config)
                state_dict = converted_dict
            except Exception as e:
                raise RuntimeError(f"Failed to convert legacy state dict: {e}")
        
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
        try:
            if device is not None or dtype != model.config.dtype:
                model = model.to(device=device, dtype=dtype)
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device/dtype: {e}")
            
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


def get_p_annealing_schedule(config: VSAEPAnnealTrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simplified, robust p-annealing schedule.
    
    Returns:
        Tuple of (update_steps, p_values)
    """
    if config.n_sparsity_updates <= 1:
        # No annealing, just return end values
        return torch.tensor([config.anneal_start]), torch.tensor([config.p_end])
    
    # Create linear schedule
    update_steps = torch.linspace(
        config.anneal_start, 
        config.anneal_end, 
        config.n_sparsity_updates, 
        dtype=torch.long
    )
    
    p_values = torch.linspace(
        config.p_start, 
        config.p_end, 
        config.n_sparsity_updates
    )
    
    return update_steps, p_values


class VSAEPAnnealTrainer(SAETrainer):
    """
    Robust trainer for Variational Sparse Autoencoder with simplified p-norm annealing.
    
    Key improvements:
    - Simplified p-annealing logic for better stability
    - Robust error handling and validation
    - Conservative numerical stability measures
    - Optional adaptive scaling (disabled by default)
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
                kl_coeff=kwargs.get('kl_coeff', 500.0),
                sparsity_coeff=sparsity_penalty or 100.0,
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
        
        # KL annealing function (separate from sparsity!)
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = [
            'p', 'current_sparsity_coeff', 'kl_loss', 'scaled_kl_loss', 
            'recon_loss', 'total_loss', 'kl_scale', 'sparsity_scale', 
            'p_norm_penalty', 'grad_norm'
        ]
        self._init_logging_state()
    
    def _init_p_annealing(self) -> None:
        """Initialize simplified p-annealing schedule."""
        config = self.training_config
        
        # Create p-value schedule
        self.update_steps, self.p_values = get_p_annealing_schedule(config)
        
        # Initialize state
        self.p = config.p_start
        self.p_step_count = 0
        self.current_sparsity_coeff = config.sparsity_coeff  # Base coefficient
        
        # Simple adaptive scaling state (if enabled)
        self.penalty_history = []
        
        print(f"P-annealing schedule: {len(self.p_values)} updates from {config.p_start} to {config.p_end}")
        print(f"Update steps: {self.update_steps.tolist()}")
    
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
        Compute standard VAE KL divergence loss.
        
        For q(z|x) = N(μ, σ²) and p(z) = N(0, I):
        KL[q || p] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # Conservative clamping range
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
        
        # Ensure KL is non-negative
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        return kl_loss
    
    def _compute_p_norm_penalty(self, mu: torch.Tensor, log_var: Optional[torch.Tensor], p: float) -> torch.Tensor:
        """
        Compute p-norm penalty with robust numerical stability.
        
        Args:
            mu: Mean of latent distribution (deterministic)
            log_var: Log variance (if learned) - not used for stability
            p: P-norm value for penalty
            
        Returns:
            P-norm penalty (always non-negative)
        """
        # Use deterministic mu for more stable gradients
        if self.training_config.use_deterministic_penalty:
            penalty_input = mu
        else:
            penalty_input = self.ae.reparameterize(mu, log_var)
        
        penalty_input = penalty_input.to(dtype=torch.float32)
        
        # Clamp to prevent extreme values
        penalty_input_clamped = torch.clamp(penalty_input, min=-50.0, max=50.0)
        
        try:
            if p == 1.0:
                # L1 penalty: sum of absolute values
                penalty = torch.sum(torch.abs(penalty_input_clamped), dim=1).mean()
            elif p == 2.0:
                # L2 penalty: sum of squares
                penalty = torch.sum(penalty_input_clamped.pow(2), dim=1).mean()
            else:
                # General p-norm with numerical stability
                abs_input = torch.abs(penalty_input_clamped)
                
                if p < 1.0:
                    # For p < 1, add small epsilon to prevent numerical issues
                    eps = 1e-8
                    abs_input = abs_input + eps
                
                if self.training_config.sparsity_function == 'Lp^p':
                    penalty = torch.sum(abs_input.pow(p), dim=1).mean()
                else:  # 'Lp'
                    if p < 1.0:
                        # For p < 1, compute carefully
                        penalty = torch.pow(torch.sum(abs_input.pow(p), dim=1) + 1e-8, 1.0/p).mean()
                    else:
                        penalty = torch.pow(torch.sum(abs_input.pow(p), dim=1), 1.0/p).mean()
            
            # Ensure penalty is non-negative and bounded
            penalty = torch.clamp(penalty, min=0.0, max=1e6)
            
        except Exception as e:
            print(f"Warning: Error computing p-norm penalty (p={p}): {e}")
            # Fallback to L1 penalty
            penalty = torch.sum(torch.abs(penalty_input_clamped), dim=1).mean()
            penalty = torch.clamp(penalty, min=0.0, max=1e6)
        
        return penalty
    
    def _update_p_annealing(self, step: int, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> None:
        """Update p-annealing schedule with simplified logic."""
        # Check if we need to update p
        if (self.p_step_count < len(self.update_steps) and 
            step >= self.update_steps[self.p_step_count]):
            
            # Update p-value
            old_p = self.p
            self.p = self.p_values[self.p_step_count].item()
            
            # Simple adaptive scaling (if enabled)
            if self.training_config.use_adaptive_scaling and len(self.penalty_history) > 5:
                try:
                    # Compute penalty ratio for scaling
                    recent_penalties = self.penalty_history[-5:]
                    avg_penalty = sum(recent_penalties) / len(recent_penalties)
                    
                    # Compute new penalty
                    with torch.no_grad():
                        new_penalty = self._compute_p_norm_penalty(mu.detach(), 
                                                                  log_var.detach() if log_var is not None else None, 
                                                                  self.p)
                    
                    if avg_penalty > 1e-6 and new_penalty > 1e-6:
                        ratio = avg_penalty / new_penalty.item()
                        # Conservative scaling
                        ratio = max(0.5, min(ratio, 2.0))
                        self.current_sparsity_coeff *= ratio
                        self.current_sparsity_coeff = max(1.0, min(self.current_sparsity_coeff, 1000.0))
                        
                except Exception as e:
                    print(f"Warning: Adaptive scaling failed: {e}")
            
            self.p_step_count += 1
            
            print(f"Step {step}: Updated p from {old_p:.4f} to {self.p:.4f}, "
                  f"sparsity_coeff: {self.current_sparsity_coeff:.4f}")
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """
        Compute total loss with clean VAE KL divergence and p-norm penalty.
        """
        sparsity_scale = self.sparsity_warmup_fn(step)  # For p-norm penalty
        kl_scale = self.kl_warmup_fn(step)  # Separate KL annealing
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Get intermediate values for loss computation
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        z = z.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        self.recon_loss = recon_loss.item()
        
        # Standard VAE KL divergence
        kl_loss = self._compute_kl_loss(mu, log_var)
        self.kl_loss = kl_loss.item()
        
        # P-norm sparsity penalty
        p_norm_penalty = self._compute_p_norm_penalty(mu, log_var, p=self.p)
        self.p_norm_penalty = p_norm_penalty.item()
        
        # Track penalty history for adaptive scaling
        if self.training_config.use_adaptive_scaling:
            self.penalty_history.append(self.p_norm_penalty)
            if len(self.penalty_history) > 20:  # Keep recent history
                self.penalty_history.pop(0)
        
        # Update p-annealing schedule
        self._update_p_annealing(step, mu.detach(), log_var.detach() if log_var is not None else None)
        
        # Separate scaling for KL and sparsity
        scaled_kl_loss = kl_loss * self.training_config.kl_coeff * kl_scale
        scaled_p_norm_penalty = p_norm_penalty * self.current_sparsity_coeff * sparsity_scale
        
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
                'current_sparsity_coeff': self.current_sparsity_coeff,
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
        
        # Monitor gradient norms
        total_norm = 0.0
        for p in self.ae.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.grad_norm = total_norm ** 0.5
        
        # Gradient clipping
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
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'sparsity_coeff': self.training_config.sparsity_coeff,
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
            'use_adaptive_scaling': self.training_config.use_adaptive_scaling,
            'use_deterministic_penalty': self.training_config.use_deterministic_penalty,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }
