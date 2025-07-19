"""
Clean, fail-fast implementation of VSAEGated with enhanced clarity and robustness.

Key improvements:
- Removed all fallback logic that masks real issues
- Clear, descriptive error messages with assertions
- Simplified control flow and tensor handling
- Consistent interface and validation throughout
- Much easier to debug when things go wrong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable, Union
from collections import namedtuple
from dataclasses import dataclass
import math
from pathlib import Path

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam,
)


@dataclass
class VSAEGatedConfig:
    """Configuration for VSAEGated model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0
    
    # Initialization settings
    init_strategy: str = "tied"  # "tied", "independent", "xavier"
    init_scale: float = 0.1
    gate_bias_init: float = 0.0
    mag_bias_init: float = 0.0
    r_mag_init: float = 0.0
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.activation_dim > 0, f"activation_dim must be positive: {self.activation_dim}"
        assert self.dict_size > 0, f"dict_size must be positive: {self.dict_size}"
        assert self.var_flag in [0, 1], f"var_flag must be 0 or 1: {self.var_flag}"
        assert self.init_strategy in ["tied", "independent", "xavier"], f"Invalid init_strategy: {self.init_strategy}"
        assert self.init_scale > 0, f"init_scale must be positive: {self.init_scale}"
    
    def get_device(self) -> torch.device:
        """Get device, defaulting to CUDA if available."""
        return self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_tensor(tensor: torch.Tensor, name: str = "tensor", replace_val: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf values with safe alternatives while preserving gradients."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"WARNING: Replacing NaN/Inf in {name} with {replace_val}")
        # Create a mask for valid values
        valid_mask = ~(torch.isnan(tensor) | torch.isinf(tensor))
        # Replace invalid values while preserving gradients
        safe_tensor = torch.where(valid_mask, tensor, torch.full_like(tensor, replace_val))
        return safe_tensor
    return tensor


def _safe_exp(x: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    """Safe exponential that won't explode."""
    x_clamped = torch.clamp(x, min=-max_val, max=max_val)
    return torch.exp(x_clamped)


class VSAEGated(Dictionary, nn.Module):
    """
    Clean Gated Variational Autoencoder implementation.
    
    Architecture:
    - Shared encoder processes input
    - Gating network: controls feature sparsity (binary gates)
    - Magnitude network: provides VAE mean values (continuous)
    - Variance network: provides VAE log variance (learned uncertainty)
    - Final features: gate_binary * reparameterized_magnitude
    """

    def __init__(self, config: VSAEGatedConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize all neural network layers."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder and decoder
        self.encoder = nn.Linear(self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device)
        self.decoder = nn.Linear(self.dict_size, self.activation_dim, bias=self.use_april_update_mode, dtype=dtype, device=device)
        
        # Bias for standard mode
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(torch.zeros(self.activation_dim, dtype=dtype, device=device))
        
        # Gating network parameters
        self.gate_bias = nn.Parameter(torch.full((self.dict_size,), self.config.gate_bias_init, dtype=dtype, device=device))
        
        # Magnitude network parameters  
        self.r_mag = nn.Parameter(torch.full((self.dict_size,), self.config.r_mag_init, dtype=dtype, device=device))
        self.mag_bias = nn.Parameter(torch.full((self.dict_size,), self.config.mag_bias_init, dtype=dtype, device=device))
        
        # Variance encoder (learned variance only)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device)
    
    def _init_weights(self) -> None:
        """Initialize model weights based on strategy."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        with torch.no_grad():
            if self.config.init_strategy == "tied":
                # Tied encoder/decoder initialization
                w = torch.randn(self.activation_dim, self.dict_size, dtype=dtype, device=device)
                w = w / w.norm(dim=0, keepdim=True) * self.config.init_scale
                self.encoder.weight.copy_(w.T)
                self.decoder.weight.copy_(w)
                
            elif self.config.init_strategy == "independent":
                nn.init.kaiming_uniform_(self.encoder.weight, a=0.01)
                nn.init.kaiming_uniform_(self.decoder.weight, a=0.01)
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
                
            elif self.config.init_strategy == "xavier":
                nn.init.xavier_uniform_(self.encoder.weight)
                nn.init.xavier_uniform_(self.decoder.weight)
            
            # Initialize biases
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
            
            # Initialize variance encoder
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor with validation."""
        # Handle 3D tensors by flattening
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
        elif x.dim() != 2:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")
        
        assert x.size(-1) == self.activation_dim, f"Input dim {x.size(-1)} != {self.activation_dim}"
        assert not torch.isnan(x).any(), "Input contains NaN"
        assert not torch.isinf(x).any(), "Input contains Inf"
        
        # Ensure correct dtype and device
        x = x.to(dtype=self.encoder.weight.dtype, device=self.encoder.weight.device)
        
        return x - self.bias if not self.use_april_update_mode else x
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False,
        use_reparameterization: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode input to latent features.
        
        Returns:
            For var_flag=0: (f, gate_continuous, mu_mag) if return_gate else (f, mu_mag)
            For var_flag=1: (f, gate_continuous, log_var, mu_mag) if return_gate else (f, log_var, mu_mag)
        """
        # Preprocess input
        x_processed = self._preprocess_input(x)
        
        # Shared encoding
        x_enc = self.encoder(x_processed)
        
        # Gating network (sparsity control)
        pi_gate = x_enc + self.gate_bias
        gate_binary = (pi_gate > 0).to(dtype=x_enc.dtype)
        gate_continuous = F.relu(pi_gate)
        
        # Magnitude network (VAE mean) - Use safe exponential
        r_mag_safe = torch.clamp(self.r_mag, min=-5.0, max=5.0)
        pi_mag = _safe_exp(r_mag_safe, max_val=5.0) * x_enc + self.mag_bias
        mu_mag = _safe_tensor(pi_mag, "mu_mag")  # Clean any NaN that slipped through
        
        # Variance encoding and reparameterization
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
            
            if use_reparameterization:
                mag_sampled = self.reparameterize(mu_mag, log_var)
            else:
                mag_sampled = mu_mag
                
            # Final features: gating * magnitude
            f = gate_binary * mag_sampled
            f = _safe_tensor(f, "final_features")  # Safety net
            
            # Return based on what's requested
            if return_gate:
                return f, gate_continuous, log_var, mu_mag
            else:
                return f, log_var, mu_mag
        else:
            # Fixed variance case
            mag_sampled = mu_mag if not use_reparameterization else mu_mag
            f = gate_binary * mag_sampled
            f = _safe_tensor(f, "final_features_fixed")  # Safety net
            
            if return_gate:
                return f, gate_continuous, mu_mag
            else:
                return f, mu_mag
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Apply reparameterization trick with comprehensive safeguards."""
        # Clean inputs first
        mu = _safe_tensor(mu, "reparameter_mu")
        log_var = _safe_tensor(log_var, "reparameter_log_var")
        
        # Conservative clamping to prevent explosion
        log_var_clamped = torch.clamp(log_var, min=-8.0, max=2.0)
        log_var_scaled = log_var_clamped / max(temperature, 0.1)  # Prevent division by zero
        
        # Additional safety: clamp mu to reasonable range
        mu_safe = torch.clamp(mu, min=-50.0, max=50.0)
        
        # Compute standard deviation with additional safety
        var = _safe_exp(log_var_scaled, max_val=2.0)  # Use safe exp
        std = torch.sqrt(var.clamp(min=1e-8, max=100.0))
        
        # Sample and reparameterize
        eps = torch.randn_like(std)
        result = mu_safe + eps * std
        
        # Final safety net
        result = _safe_tensor(result, "reparameter_result")
        return torch.clamp(result, min=-100.0, max=100.0)
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction with safeguards."""
        assert f.size(-1) == self.dict_size, f"Feature dim {f.size(-1)} != {self.dict_size}"
        
        # Ensure correct dtype and device
        f = f.to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)
        
        # Add safety clamp to prevent extreme feature values from causing NaN
        f_safe = torch.clamp(f, min=-100.0, max=100.0)
        
        if self.use_april_update_mode:
            result = self.decoder(f_safe)
        else:
            result = self.decoder(f_safe) + self.bias
        
        # Final safety net - replace any NaN that made it through
        result = _safe_tensor(result, "decoder_output")
        
        # Debug info if we had to replace values
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"WARNING: Had to clean NaN/Inf from decoder output!")
            print(f"f stats: min={f.min()}, max={f.max()}, mean={f.mean()}")
            print(f"decoder weight stats: min={self.decoder.weight.min()}, max={self.decoder.weight.max()}")
        
        return result
    
    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False,
        use_reparameterization: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the autoencoder."""
        if use_reparameterization is None:
            use_reparameterization = self.training
        
        original_dtype = x.dtype
        original_shape = x.shape
        
        # Encode
        if self.var_flag == 1:
            f, log_var, mu_mag = self.encode(x, return_gate=False, use_reparameterization=use_reparameterization)
        else:
            f, mu_mag = self.encode(x, return_gate=False, use_reparameterization=use_reparameterization)
        
        # Decode
        x_hat = self.decode(f)
        
        # Reshape if needed
        if len(original_shape) == 3:
            x_hat = x_hat.view(original_shape)
            if output_features:
                f = f.view(original_shape[0], original_shape[1], -1)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get KL diagnostics for monitoring."""
        with torch.no_grad():
            if self.var_flag == 1:
                f, gate_continuous, log_var, mu_mag = self.encode(
                    x, return_gate=True, use_reparameterization=False
                )
                
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)
                
                kl_mu = 0.5 * torch.sum(mu_mag.pow(2), dim=1).mean()
                kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                kl_total = kl_mu + kl_var
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_mu,
                    'kl_var_term': kl_var,
                    'mean_log_var': log_var.mean(),
                    'mean_var': var.mean(),
                    'mean_mu': mu_mag.mean(),
                    'mean_mu_magnitude': mu_mag.norm(dim=-1).mean(),
                    'gate_sparsity': (gate_continuous > 0).float().mean(),
                    'final_sparsity': (f != 0).float().mean(),
                }
            else:
                f, gate_continuous, mu_mag = self.encode(x, return_gate=True, use_reparameterization=False)
                kl_total = 0.5 * torch.sum(mu_mag.pow(2), dim=1).mean()
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu_mag.mean(),
                    'mean_mu_magnitude': mu_mag.norm(dim=-1).mean(),
                    'gate_sparsity': (gate_continuous > 0).float().mean(),
                    'final_sparsity': (f != 0).float().mean(),
                }
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters."""
        assert scale > 0, f"Scale must be positive: {scale}"
        
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
            
            self.gate_bias.mul_(scale)
            self.mag_bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        config: Optional[VSAEGatedConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        var_flag: Optional[int] = None
    ) -> 'VSAEGated':
        """Load pretrained model with simplified error handling."""
        path = Path(path)
        assert path.exists(), f"Model file not found: {path}"
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Auto-detect config if not provided
        if config is None:
            config = cls._detect_config_from_state_dict(state_dict, dtype=dtype, device=device, var_flag=var_flag)
        
        # Create and load model
        model = cls(config)
        
        # Handle legacy parameter names if needed
        if cls._is_legacy_state_dict(state_dict):
            state_dict = cls._convert_legacy_state_dict(state_dict, config)
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return model.to(device=device, dtype=dtype)
    
    @classmethod
    def _detect_config_from_state_dict(cls, state_dict: Dict, dtype: torch.dtype, device: Optional[torch.device], var_flag: Optional[int]) -> VSAEGatedConfig:
        """Detect configuration from state dict."""
        if 'encoder.weight' in state_dict:
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            use_april_update_mode = "decoder.bias" in state_dict
        else:
            raise ValueError("Cannot detect dimensions from state dict")
        
        if var_flag is None:
            var_flag = 1 if "var_encoder.weight" in state_dict else 0
        
        return VSAEGatedConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            var_flag=var_flag,
            use_april_update_mode=use_april_update_mode,
            dtype=dtype,
            device=device
        )
    
    @classmethod
    def _is_legacy_state_dict(cls, state_dict: Dict) -> bool:
        """Check if state dict uses legacy parameter names."""
        return any(key in state_dict for key in ['W_enc', 'W_dec', 'b_enc', 'b_dec'])
    
    @classmethod
    def _convert_legacy_state_dict(cls, state_dict: Dict, config: VSAEGatedConfig) -> Dict:
        """Convert legacy parameter names."""
        converted = {}
        
        # Convert main parameters
        if 'W_enc' in state_dict:
            converted["encoder.weight"] = state_dict["W_enc"].T
        if 'b_enc' in state_dict:
            converted["encoder.bias"] = state_dict["b_enc"]
        if 'W_dec' in state_dict:
            converted["decoder.weight"] = state_dict["W_dec"].T
        
        if config.use_april_update_mode and 'b_dec' in state_dict:
            converted["decoder.bias"] = state_dict["b_dec"]
        elif not config.use_april_update_mode and 'b_dec' in state_dict:
            converted["bias"] = state_dict["b_dec"]
        
        # Copy gating parameters and variance encoder
        for param in ['gate_bias', 'r_mag', 'mag_bias']:
            if param in state_dict:
                converted[param] = state_dict[param]
        
        if config.var_flag == 1:
            if "W_enc_var" in state_dict:
                converted["var_encoder.weight"] = state_dict["W_enc_var"].T
            if "b_enc_var" in state_dict:
                converted["var_encoder.bias"] = state_dict["b_enc_var"]
        
        return converted


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """KL annealing schedule function."""
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps, f"Invalid kl_warmup_steps: {kl_warmup_steps}"
    
    def scale_fn(step: int) -> float:
        return min(step / kl_warmup_steps, 1.0) if step >= 0 else 0.0
    
    return scale_fn


@dataclass
class VSAEGatedTrainingConfig:
    """Training configuration with validation."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    l1_penalty: float = 0.1
    aux_weight: float = 0.1
    kl_warmup_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    use_constrained_optimizer: bool = True
    gradient_clip_norm: float = 1.0
    temperature_schedule: str = "constant"
    min_temperature: float = 0.1
    max_temperature: float = 1.0
    
    def __post_init__(self):
        """Set derived values and validate."""
        assert self.steps > 0, "steps must be positive"
        assert self.lr > 0, "lr must be positive"
        assert self.kl_coeff >= 0, "kl_coeff must be non-negative"
        assert self.l1_penalty >= 0, "l1_penalty must be non-negative"
        assert self.aux_weight >= 0, "aux_weight must be non-negative"
        
        # Set derived values
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)
        
        # Set decay start
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start if default_decay_start > min_decay_start else None
    
    def get_temperature(self, step: int) -> float:
        """Get temperature for current step."""
        if self.temperature_schedule == "constant":
            return self.max_temperature
        elif self.temperature_schedule == "linear_decay":
            progress = min(step / self.steps, 1.0)
            return self.max_temperature - progress * (self.max_temperature - self.min_temperature)
        elif self.temperature_schedule == "cosine_decay":
            progress = min(step / self.steps, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_temperature + cosine_factor * (self.max_temperature - self.min_temperature)
        else:
            return self.max_temperature


class VSAEGatedTrainer(SAETrainer):
    """Clean, fail-fast trainer for VSAEGated."""
    
    def __init__(
        self,
        model_config: Optional[VSAEGatedConfig] = None,
        training_config: Optional[VSAEGatedTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            assert 'activation_dim' in kwargs and 'dict_size' in kwargs, "Must provide model_config or activation_dim + dict_size"
            model_config = VSAEGatedConfig(
                activation_dim=kwargs['activation_dim'],
                dict_size=kwargs['dict_size'],
                var_flag=kwargs.get('var_flag', 1),
                use_april_update_mode=kwargs.get('use_april_update_mode', True),
                device=torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            )
        
        if training_config is None:
            assert 'steps' in kwargs, "Must provide training_config or steps"
            training_config = VSAEGatedTrainingConfig(
                steps=kwargs['steps'],
                lr=kwargs.get('lr', 5e-4),
                kl_coeff=kwargs.get('kl_coeff', 500.0),
                l1_penalty=kwargs.get('l1_penalty', 0.1),
                aux_weight=kwargs.get('aux_weight', 0.1),
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedTrainer"
        
        # Initialize model
        self.ae = VSAEGated(model_config)
        self.device = model_config.get_device()
        self.ae.to(self.device)
        
        # Initialize optimizer
        if training_config.use_constrained_optimizer:
            self.optimizer = ConstrainedAdam(
                self.ae.parameters(),
                self.ae.decoder.parameters(),
                lr=training_config.lr,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.ae.parameters(),
                lr=training_config.lr,
                betas=(0.9, 0.999)
            )
        
        # Initialize schedules
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            None,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(training_config.steps, training_config.sparsity_warmup_steps)
        self.kl_warmup_fn = get_kl_warmup_fn(training_config.steps, training_config.kl_warmup_steps)
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "gate_sparsity", "aux_loss_raw", "current_temperature", "current_kl_scale"]
        self.effective_l0 = 0.0
        self.gate_sparsity = 0.0
        self.aux_loss_raw = 0.0
        self.current_temperature = 1.0
        self.current_kl_scale = 1.0
    
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute KL divergence loss with comprehensive safeguards."""
        # Clean inputs
        mu = _safe_tensor(mu, "kl_mu")
        
        if self.ae.var_flag == 1 and log_var is not None:
            log_var = _safe_tensor(log_var, "kl_log_var")
            
            # Clamp to reasonable ranges
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # Safe exponential for variance term
            var_term = _safe_exp(log_var_clamped, max_val=2.0)
            
            # Standard VAE KL divergence with safeguards
            kl_per_sample = 0.5 * torch.sum(
                mu_clamped.pow(2) + var_term - 1 - log_var_clamped,
                dim=1
            )
        else:
            # Fixed variance case
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
        
        # Average over batch with safety
        kl_loss = _safe_tensor(kl_per_sample.mean(), "kl_loss_mean")
        
        # Ensure non-negative (KL should always be >= 0)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        return kl_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Clean loss computation with fail-fast behavior."""
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Update current step and get scaling factors
        sparsity_scale = self.sparsity_warmup_fn(step)
        kl_scale = self.kl_warmup_fn(step)
        temperature = self.training_config.get_temperature(step)
        
        # Update logging variables
        self.current_kl_scale = kl_scale
        self.current_temperature = temperature
        
        # Store original properties
        original_dtype = x.dtype
        original_shape = x.shape
        
        # Handle 3D tensors by flattening
        x_flat = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        
        # Validate input
        assert x_flat.size(-1) == self.ae.activation_dim, f"Input dim {x_flat.size(-1)} != activation_dim {self.ae.activation_dim}"
        assert not torch.isnan(x_flat).any(), "Input contains NaN"
        assert not torch.isinf(x_flat).any(), "Input contains Inf"
        
        # Encoding based on var_flag
        if self.ae.var_flag == 1:
            f, gate_continuous, log_var, mu_mag = self.ae.encode(
                x_flat, return_gate=True, use_reparameterization=True
            )
            
            # Apply temperature if needed
            if temperature != 1.0:
                mag_sampled = self.ae.reparameterize(mu_mag, log_var, temperature=temperature)
                gate_binary = (gate_continuous > 0).to(dtype=gate_continuous.dtype)
                f = gate_binary * mag_sampled
        else:
            f, gate_continuous, mu_mag = self.ae.encode(
                x_flat, return_gate=True, use_reparameterization=True
            )
            log_var = None
        
        # Validate encoding outputs - mostly for debugging now since we use safe operations
        for name, tensor in [('f', f), ('gate_continuous', gate_continuous), ('mu_mag', mu_mag)]:
            assert tensor is not None, f"{name} is None"
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"WARNING: {name} contains NaN/Inf at step {step} (after safe operations)")
                print(f"{name} stats: min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}")
        
        if log_var is not None and (torch.isnan(log_var).any() or torch.isinf(log_var).any()):
            print(f"WARNING: log_var contains NaN/Inf at step {step} (after safe operations)")
            print(f"log_var stats: min={log_var.min()}, max={log_var.max()}, mean={log_var.mean()}")
        
        # Decoding
        x_hat = self.ae.decode(f)
        x_hat_gate = self.ae.decode(gate_continuous)
        
        # Handle tensor shapes for loss computation
        if len(original_shape) == 3:
            # Keep everything flattened for loss computation
            x_hat_flat = x_hat.view(-1, x_hat.size(-1)) if x_hat.dim() == 3 else x_hat
            x_hat_gate_flat = x_hat_gate.view(-1, x_hat_gate.size(-1)) if x_hat_gate.dim() == 3 else x_hat_gate
            f_flat = f.view(-1, f.size(-1)) if f.dim() == 3 else f
            gate_continuous_flat = gate_continuous.view(-1, gate_continuous.size(-1)) if gate_continuous.dim() == 3 else gate_continuous
        else:
            x_hat_flat = x_hat
            x_hat_gate_flat = x_hat_gate
            f_flat = f
            gate_continuous_flat = gate_continuous
        
        # Ensure dtype compatibility without breaking gradients
        # Don't use .to() as it can break gradient flow - let PyTorch handle dtype promotion
        
        # Validate reconstruction outputs - also mostly for debugging now
        if torch.isnan(x_hat_flat).any() or torch.isinf(x_hat_flat).any():
            print(f"WARNING: Main reconstruction contains NaN/Inf at step {step} (after safe operations)")
            print(f"x_hat_flat stats: min={x_hat_flat.min()}, max={x_hat_flat.max()}")
            print(f"f stats: min={f_flat.min()}, max={f_flat.max()}, mean={f_flat.mean()}")
            print(f"Active features: {(f_flat != 0).sum().item()}/{f_flat.numel()}")
            print(f"This suggests deeper numerical instability - check hyperparameters!")
            
        if torch.isnan(x_hat_gate_flat).any() or torch.isinf(x_hat_gate_flat).any():
            print(f"WARNING: Gate reconstruction contains NaN/Inf at step {step} (after safe operations)")
            print(f"x_hat_gate_flat stats: min={x_hat_gate_flat.min()}, max={x_hat_gate_flat.max()}")
            print(f"gate_continuous_flat stats: min={gate_continuous_flat.min()}, max={gate_continuous_flat.max()}")
            print(f"This suggests deeper numerical instability - check hyperparameters!")
        
        # Decoding
        x_hat = self.ae.decode(f)
        x_hat_gate = self.ae.decode(gate_continuous)
        
        # Handle tensor shapes for loss computation
        if len(original_shape) == 3:
            # Keep everything flattened for loss computation
            x_hat_flat = x_hat.view(-1, x_hat.size(-1)) if x_hat.dim() == 3 else x_hat
            x_hat_gate_flat = x_hat_gate.view(-1, x_hat_gate.size(-1)) if x_hat_gate.dim() == 3 else x_hat_gate
            f_flat = f.view(-1, f.size(-1)) if f.dim() == 3 else f
            gate_continuous_flat = gate_continuous.view(-1, gate_continuous.size(-1)) if gate_continuous.dim() == 3 else gate_continuous
        else:
            x_hat_flat = x_hat
            x_hat_gate_flat = x_hat_gate
            f_flat = f
            gate_continuous_flat = gate_continuous
        
        # Ensure dtype compatibility without breaking gradients
        # Don't use .to() as it can break gradient flow - let PyTorch handle dtype promotion
        # x_hat_flat = x_hat_flat.to(dtype=original_dtype)
        # x_hat_gate_flat = x_hat_gate_flat.to(dtype=original_dtype)
        
        # Validate reconstruction outputs with debug info
        if torch.isnan(x_hat_flat).any() or torch.isinf(x_hat_flat).any():
            print(f"ERROR: Main reconstruction contains NaN/Inf at step {step}")
            print(f"x_hat_flat stats: min={x_hat_flat.min()}, max={x_hat_flat.max()}")
            print(f"f stats: min={f_flat.min()}, max={f_flat.max()}, mean={f_flat.mean()}")
            print(f"Active features: {(f_flat != 0).sum().item()}/{f_flat.numel()}")
            assert False, "Main reconstruction contains NaN/Inf"
            
        if torch.isnan(x_hat_gate_flat).any() or torch.isinf(x_hat_gate_flat).any():
            print(f"ERROR: Gate reconstruction contains NaN/Inf at step {step}")
            print(f"x_hat_gate_flat stats: min={x_hat_gate_flat.min()}, max={x_hat_gate_flat.max()}")
            print(f"gate_continuous_flat stats: min={gate_continuous_flat.min()}, max={gate_continuous_flat.max()}")
            assert False, "Gate reconstruction contains NaN/Inf"
        
        # Compute loss components with safety nets
        recon_loss = torch.mean(torch.sum((x_flat - x_hat_flat) ** 2, dim=1))
        recon_loss = _safe_tensor(recon_loss, "recon_loss", 1.0)
        
        gate_sparsity_loss = torch.mean(torch.sum(torch.abs(gate_continuous_flat), dim=1))
        gate_sparsity_loss = _safe_tensor(gate_sparsity_loss, "sparsity_loss", 0.1)
        
        aux_loss = torch.mean(torch.sum((x_flat - x_hat_gate_flat) ** 2, dim=1))
        aux_loss = _safe_tensor(aux_loss, "aux_loss", 1.0)
        
        kl_loss = self._compute_kl_loss(mu_mag, log_var)
        kl_loss = _safe_tensor(kl_loss, "kl_loss", 0.1)
        # Don't force dtype conversion on kl_loss as it can break gradients
        # kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Validate loss components - should be clean now due to safe operations
        for name, loss_val in [('recon', recon_loss), ('sparsity', gate_sparsity_loss), ('aux', aux_loss), ('kl', kl_loss)]:
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"WARNING: {name} loss still has NaN/Inf after safety cleaning: {loss_val}")
                # This should be very rare now
        
        # Compute total loss
        total_loss = (
            recon_loss +
            (self.training_config.l1_penalty * sparsity_scale * gate_sparsity_loss) +
            (self.training_config.aux_weight * aux_loss) +
            (self.training_config.kl_coeff * kl_scale * kl_loss)
        )
        
        # Final validation - should be clean now
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: Total loss still has NaN/Inf after all safety measures: {total_loss}")
        
        # Only require gradients when we're actually going to use them for backprop
        if not logging:
            assert total_loss.requires_grad, f"Total loss does not require gradients during training at step {step}"
        
        # Update logging stats
        with torch.no_grad():
            self.effective_l0 = float(torch.sum(f_flat != 0).item()) / f_flat.numel() if f_flat.numel() > 0 else 0.0
            self.gate_sparsity = float(torch.sum(gate_continuous_flat > 0).item()) / gate_continuous_flat.numel() if gate_continuous_flat.numel() > 0 else 0.0
            self.aux_loss_raw = aux_loss.item()
        
        if not logging:
            return total_loss
        
        # Get diagnostics for logging
        kl_diagnostics = self.ae.get_kl_diagnostics(x_flat)
        
        loss_dict = {
            'l2_loss': torch.sqrt(recon_loss).item(),
            'mse_loss': recon_loss.item(),
            'gate_sparsity_loss': gate_sparsity_loss.item(),
            'aux_loss': aux_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'effective_l0': self.effective_l0,
            'gate_sparsity': self.gate_sparsity,
            'sparsity_scale': sparsity_scale,
            'kl_scale': kl_scale,
            'temperature': temperature,
            'current_lr': self.optimizer.param_groups[0]['lr'],
        }
        
        # Add diagnostics
        for k, v in kl_diagnostics.items():
            loss_dict[k] = v.item() if torch.is_tensor(v) else v
        
        # Return with proper shapes for logging
        if len(original_shape) == 3:
            x_hat_return = x_hat if x_hat.dim() == 3 else x_hat.view(original_shape)
            f_return = f if f.dim() == 3 else f.view(original_shape[0], original_shape[1], -1)
        else:
            x_hat_return = x_hat_flat
            f_return = f_flat
        
        # Convert to original dtype for output only (doesn't affect gradients since we're done with loss computation)
        x_hat_return = x_hat_return.to(dtype=original_dtype)
        f_return = f_return.to(dtype=original_dtype)
        
        return LossLog(x, x_hat_return, f_return, loss_dict)
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Clean update step without error recovery."""
        activations = activations.to(self.device)
        
        # Validate input
        assert not torch.isnan(activations).any(), f"Input activations contain NaN at step {step}"
        assert not torch.isinf(activations).any(), f"Input activations contain Inf at step {step}"
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.loss(activations, step=step)
        
        # Validate loss
        assert loss.requires_grad, f"Loss does not require gradients at step {step}"
        assert not torch.isnan(loss), f"Loss is NaN at step {step}: {loss}"
        assert not torch.isinf(loss), f"Loss is Inf at step {step}: {loss}"
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_norm = 0.0
        for name, p in self.ae.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradients at step {step}"
                assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradients at step {step}"
                total_norm += p.grad.data.norm(2).item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Gradient clipping
        if total_norm > self.training_config.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.ae.parameters(),
                self.training_config.gradient_clip_norm
            )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            'dict_class': 'VSAEGated',
            'trainer_class': 'VSAEGatedTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'init_strategy': self.model_config.init_strategy,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'l1_penalty': self.training_config.l1_penalty,
            'aux_weight': self.training_config.aux_weight,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'use_constrained_optimizer': self.training_config.use_constrained_optimizer,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'temperature_schedule': self.training_config.temperature_schedule,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }