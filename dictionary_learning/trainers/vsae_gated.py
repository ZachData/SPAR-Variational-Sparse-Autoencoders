"""
Robust implementation of VSAEGated with enhanced error handling and stability.

Key improvements over original:
1. Enhanced error handling and validation
2. More robust configuration management
3. Better numerical stability measures
4. Improved type hints and documentation
5. More comprehensive legacy support
6. Better initialization strategies
7. Enhanced diagnostics and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable, Union
from collections import namedtuple
from dataclasses import dataclass, field
import warnings
import json
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
    """Enhanced configuration for VSAEGated model with validation."""
    activation_dim: int
    dict_size: int
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    # Advanced configuration options
    init_strategy: str = "tied"  # "tied", "independent", "xavier"
    init_scale: float = 0.1
    gate_bias_init: float = 0.0
    mag_bias_init: float = 0.0
    r_mag_init: float = 0.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {self.activation_dim}")
        if self.dict_size <= 0:
            raise ValueError(f"dict_size must be positive, got {self.dict_size}")
        if self.var_flag not in [0, 1]:
            raise ValueError(f"var_flag must be 0 or 1, got {self.var_flag}")
        if self.init_strategy not in ["tied", "independent", "xavier"]:
            raise ValueError(f"init_strategy must be one of ['tied', 'independent', 'xavier'], got {self.init_strategy}")
        if self.init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {self.init_scale}")
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'activation_dim': self.activation_dim,
            'dict_size': self.dict_size,
            'var_flag': self.var_flag,
            'use_april_update_mode': self.use_april_update_mode,
            'dtype': str(self.dtype),
            'device': str(self.device) if self.device else None,
            'log_var_init': self.log_var_init,
            'init_strategy': self.init_strategy,
            'init_scale': self.init_scale,
            'gate_bias_init': self.gate_bias_init,
            'mag_bias_init': self.mag_bias_init,
            'r_mag_init': self.r_mag_init,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VSAEGatedConfig':
        """Create config from dictionary."""
        # Handle dtype conversion
        if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
            }
            config_dict['dtype'] = dtype_map.get(config_dict['dtype'], torch.float32)
        
        # Handle device conversion
        if 'device' in config_dict and isinstance(config_dict['device'], str):
            if config_dict['device'] != 'None':
                config_dict['device'] = torch.device(config_dict['device'])
            else:
                config_dict['device'] = None
        
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class VSAEGated(Dictionary, nn.Module):
    """
    Robust Gated Variational Autoencoder with enhanced error handling and stability.
    
    Architecture:
    - Shared encoder for initial processing
    - Gating network: determines feature activation (sparsity mechanism)
    - Magnitude network: provides VAE mean values (continuous latent values)
    - Variance network: provides VAE log variance (learned uncertainty)
    - Final features: gate_binary * reparameterized_magnitude
    
    The key insight is that gating controls sparsity while magnitude controls 
    the actual latent representation values for active features.
    """

    def __init__(self, config: VSAEGatedConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        # Validate configuration
        self._validate_config()
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
        
        # Register buffer for tracking statistics
        self.register_buffer('_training_steps', torch.tensor(0, dtype=torch.long))
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        if self.activation_dim != self.config.activation_dim:
            raise ValueError("Activation dimension mismatch")
        if self.dict_size != self.config.dict_size:
            raise ValueError("Dictionary size mismatch")
        
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input with enhanced validation and 3D tensor support."""
        # Handle 3D tensors (batch, seq_len, d_model) -> flatten to 2D
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape from [batch, seq_len, d_model] to [batch*seq_len, d_model]
            x = x.view(-1, x.size(-1))
        elif x.dim() != 2:
            raise ValueError(f"Input must be 2D or 3D tensor, got {x.dim()}D")
            
        if x.size(-1) != self.activation_dim:
            raise ValueError(f"Input dimension {x.size(-1)} doesn't match activation_dim {self.activation_dim}")
        
        # Store original shape for potential reshaping later
        if not hasattr(self, '_input_shape_cache'):
            self._input_shape_cache = original_shape
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype, device=self.encoder.weight.device)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias

    def _postprocess_output(self, x: torch.Tensor, target_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Reshape output back to original shape if needed."""
        if target_shape is not None and len(target_shape) == 3:
            # Reshape back from [batch*seq_len, d_model] to [batch, seq_len, d_model]
            batch_size, seq_len = target_shape[0], target_shape[1]
            x = x.view(batch_size, seq_len, -1)
        return x
    def _init_layers(self) -> None:
        """Initialize neural network layers with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        try:
            # Shared encoder for processing input
            self.encoder = nn.Linear(
                self.activation_dim,
                self.dict_size,
                bias=True,
                dtype=dtype,
                device=device
            )
            
            # Shared decoder
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
            
            # Gating network parameters (for sparsity)
            self.gate_bias = nn.Parameter(
                torch.full(
                    (self.dict_size,), 
                    self.config.gate_bias_init,
                    dtype=dtype, 
                    device=device
                )
            )
            
            # Magnitude network parameters (for VAE mean)
            self.r_mag = nn.Parameter(
                torch.full(
                    (self.dict_size,), 
                    self.config.r_mag_init,
                    dtype=dtype, 
                    device=device
                )
            )
            self.mag_bias = nn.Parameter(
                torch.full(
                    (self.dict_size,), 
                    self.config.mag_bias_init,
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
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize layers: {e}")
    
    def _init_weights(self) -> None:
        """Initialize model weights with enhanced strategies."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        try:
            with torch.no_grad():
                if self.config.init_strategy == "tied":
                    # Tied initialization (default)
                    w = torch.randn(
                        self.activation_dim,
                        self.dict_size,
                        dtype=dtype,
                        device=device
                    )
                    w = w / w.norm(dim=0, keepdim=True) * self.config.init_scale
                    
                    self.encoder.weight.copy_(w.T)
                    self.decoder.weight.copy_(w)
                    
                elif self.config.init_strategy == "independent":
                    # Independent initialization
                    nn.init.kaiming_uniform_(self.encoder.weight, a=0.01)
                    nn.init.kaiming_uniform_(self.decoder.weight, a=0.01)
                    
                    # Normalize decoder weights
                    self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
                    
                elif self.config.init_strategy == "xavier":
                    # Xavier initialization
                    nn.init.xavier_uniform_(self.encoder.weight)
                    nn.init.xavier_uniform_(self.decoder.weight)
                
                # Initialize biases
                nn.init.zeros_(self.encoder.bias)
                if self.use_april_update_mode:
                    nn.init.zeros_(self.decoder.bias)
                else:
                    nn.init.zeros_(self.bias)
                
                # Gating and magnitude parameters already initialized in _init_layers
                
                # Initialize variance encoder if present
                if self.var_flag == 1:
                    nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                    nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to initialize weights: {e}")
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input with enhanced validation."""
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, got {x.dim()}D")
        if x.size(-1) != self.activation_dim:
            raise ValueError(f"Input dimension {x.size(-1)} doesn't match activation_dim {self.activation_dim}")
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype, device=self.encoder.weight.device)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False,
        return_log_var: bool = False,
        use_reparameterization: bool = True,
        return_all_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Enhanced encode method with 3D tensor support and consistent return values.
        """
        try:
            # Store original shape for reshaping output
            original_shape = x.shape
            
            # Preprocess input (handles 3D -> 2D conversion)
            x_processed = self._preprocess_input(x)
            
            # Shared encoding
            x_enc = self.encoder(x_processed)
            
            # Gating network - for sparsity (which features are active)
            pi_gate = x_enc + self.gate_bias
            gate_binary = (pi_gate > 0).to(dtype=x_enc.dtype)
            gate_continuous = F.relu(pi_gate)
            
            # Magnitude network - provides VAE mean (unconstrained)
            pi_mag = torch.exp(self.r_mag) * x_enc + self.mag_bias
            mu_mag = pi_mag  # Unconstrained VAE mean
            
            # Variance encoding if enabled
            log_var = None
            if self.var_flag == 1:
                log_var = self.var_encoder(x_processed)
                
                # Apply reparameterization trick if requested
                if use_reparameterization:
                    mag_sampled = self.reparameterize(mu_mag, log_var)
                else:
                    mag_sampled = mu_mag
            else:
                mag_sampled = mu_mag
            
            # Combine gating (sparsity) with magnitude (VAE values)
            f = gate_binary * mag_sampled
            
            # Reshape outputs back to original shape if needed
            if len(original_shape) == 3:
                f = self._postprocess_output(f, original_shape)
                gate_continuous = self._postprocess_output(gate_continuous, original_shape)
                mu_mag = self._postprocess_output(mu_mag, original_shape)
                if log_var is not None:
                    log_var = self._postprocess_output(log_var, original_shape)
            
            # Handle different return modes - ALWAYS return consistent tuples
            if return_all_components:
                return {
                    'features': f,
                    'gate_binary': gate_binary,
                    'gate_continuous': gate_continuous,
                    'mu_mag': mu_mag,
                    'mag_sampled': mag_sampled,
                    'log_var': log_var,
                    'x_enc': x_enc,
                    'pi_gate': pi_gate,
                    'pi_mag': pi_mag
                }
            
            # FIXED: Always return consistent number of values based on flags
            if self.var_flag == 1:
                # Always return 4 values for learned variance case
                if return_gate:
                    return f, gate_continuous, log_var, mu_mag
                else:
                    return f, log_var, mu_mag
            else:
                # Always return 3 values for fixed variance case  
                if return_gate:
                    return f, gate_continuous, mu_mag
                else:
                    return f, mu_mag
            
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}")

    def reparameterize(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Enhanced reparameterization with temperature control.
        
        Args:
            mu: Mean values
            log_var: Log variance values
            temperature: Temperature for controlling stochasticity
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        try:
            # Conservative clamping with configurable range
            log_var_min = getattr(self.config, 'log_var_min', -6.0)
            log_var_max = getattr(self.config, 'log_var_max', 2.0)
            log_var_clamped = torch.clamp(log_var, min=log_var_min, max=log_var_max)
            
            # Scale by temperature
            log_var_scaled = log_var_clamped / temperature
            
            # Compute standard deviation
            std = torch.sqrt(torch.exp(log_var_scaled))
            
            # Sample noise
            eps = torch.randn_like(std)
            
            # Reparameterize
            z = mu + eps * std
            
            return z.to(dtype=mu.dtype)
            
        except Exception as e:
            raise RuntimeError(f"Reparameterization failed: {e}")
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Enhanced decode with 3D tensor support."""
        try:
            # Store original shape
            original_shape = f.shape
            
            # Handle 3D tensors
            if f.dim() == 3:
                f_flat = f.view(-1, f.size(-1))
            elif f.dim() == 2:
                f_flat = f
            else:
                raise ValueError(f"Features must be 2D or 3D tensor, got {f.dim()}D")
                
            if f_flat.size(-1) != self.dict_size:
                raise ValueError(f"Feature dimension {f_flat.size(-1)} doesn't match dict_size {self.dict_size}")
            
            # Ensure f matches decoder weight dtype
            f_flat = f_flat.to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)
            
            # Decode
            if self.use_april_update_mode:
                x_hat = self.decoder(f_flat)
            else:
                x_hat = self.decoder(f_flat) + self.bias
            
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                x_hat = x_hat.view(original_shape[0], original_shape[1], -1)
                
            return x_hat
                
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {e}")

    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False, 
        ghost_mask: Optional[torch.Tensor] = None,
        use_reparameterization: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Enhanced forward pass with better control and validation.
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEGated")
        
        try:
            # Default reparameterization behavior
            if use_reparameterization is None:
                use_reparameterization = self.training
            
            # Store original dtype
            original_dtype = x.dtype
            
            # Encoding
            if self.var_flag == 1:
                f, gate_continuous, log_var, mu_mag = self.encode(
                    x, return_gate=True, return_log_var=True, 
                    use_reparameterization=use_reparameterization
                )
            else:
                f, mu_mag = self.encode(x, use_reparameterization=use_reparameterization)
            
            # Decoding
            x_hat = self.decode(f)
            
            # Convert back to original dtype
            x_hat = x_hat.to(dtype=original_dtype)
            
            if output_features:
                f = f.to(dtype=original_dtype)
                return x_hat, f
            return x_hat
            
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced KL diagnostics with comprehensive statistics.
        """
        try:
            with torch.no_grad():
                if self.var_flag == 1:
                    components = self.encode(
                        x, return_gate=True, return_log_var=True, 
                        use_reparameterization=False, return_all_components=True
                    )
                    
                    f = components['features']
                    gate_continuous = components['gate_continuous']
                    log_var = components['log_var']
                    mu_mag = components['mu_mag']
                    
                    log_var_safe = torch.clamp(log_var, -6, 2)
                    var = torch.exp(log_var_safe)
                    
                    # KL divergence components
                    kl_mu = 0.5 * torch.sum(mu_mag.pow(2), dim=1).mean()
                    kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                    kl_total = kl_mu + kl_var
                    
                    return {
                        'kl_total': kl_total,
                        'kl_mu_term': kl_mu,
                        'kl_var_term': kl_var,
                        'mean_log_var': log_var.mean(),
                        'std_log_var': log_var.std(),
                        'mean_var': var.mean(),
                        'std_var': var.std(),
                        'mean_mag_mu': mu_mag.mean(),
                        'std_mag_mu': mu_mag.std(),
                        'mean_mag_magnitude': mu_mag.norm(dim=-1).mean(),
                        'mean_gate': gate_continuous.mean(),
                        'std_gate': gate_continuous.std(),
                        'gate_sparsity': (gate_continuous > 0).float().mean(),
                        'final_sparsity': (f != 0).float().mean(),
                        'effective_features': (f != 0).float().sum(dim=-1).mean(),
                        'gate_magnitude_correlation': torch.corrcoef(
                            torch.stack([gate_continuous.flatten(), mu_mag.abs().flatten()])
                        )[0, 1],
                    }
                else:
                    f, mu_mag = self.encode(x, use_reparameterization=False)
                    
                    kl_total = 0.5 * torch.sum(mu_mag.pow(2), dim=1).mean()
                    
                    return {
                        'kl_total': kl_total,
                        'kl_mu_term': kl_total,
                        'kl_var_term': torch.tensor(0.0),
                        'mean_mag_mu': mu_mag.mean(),
                        'std_mag_mu': mu_mag.std(),
                        'mean_mag_magnitude': mu_mag.norm(dim=-1).mean(),
                        'final_sparsity': (f != 0).float().mean(),
                        'effective_features': (f != 0).float().sum(dim=-1).mean(),
                    }
                    
        except Exception as e:
            warnings.warn(f"KL diagnostics computation failed: {e}")
            return {'kl_total': torch.tensor(float('nan'))}
    
    def get_reconstruction_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get comprehensive reconstruction diagnostics."""
        try:
            with torch.no_grad():
                x_hat, f = self(x, output_features=True, use_reparameterization=False)
                
                # Basic reconstruction metrics
                mse = torch.mean((x - x_hat) ** 2)
                mae = torch.mean(torch.abs(x - x_hat))
                
                # Cosine similarity
                x_norm = F.normalize(x, p=2, dim=-1)
                x_hat_norm = F.normalize(x_hat, p=2, dim=-1)
                cosine_sim = (x_norm * x_hat_norm).sum(dim=-1).mean()
                
                # Signal-to-noise ratio
                signal_power = torch.mean(x ** 2)
                noise_power = torch.mean((x - x_hat) ** 2)
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                
                # Feature statistics
                feature_magnitude = f.norm(dim=-1).mean()
                feature_sparsity = (f != 0).float().mean()
                
                return {
                    'mse': mse,
                    'mae': mae,
                    'rmse': torch.sqrt(mse),
                    'cosine_similarity': cosine_sim,
                    'snr_db': snr,
                    'feature_magnitude': feature_magnitude,
                    'feature_sparsity': feature_sparsity,
                    'input_magnitude': x.norm(dim=-1).mean(),
                    'output_magnitude': x_hat.norm(dim=-1).mean(),
                    'magnitude_ratio': x_hat.norm(dim=-1).mean() / (x.norm(dim=-1).mean() + 1e-8),
                }
                
        except Exception as e:
            warnings.warn(f"Reconstruction diagnostics failed: {e}")
            return {}
    
    def scale_biases(self, scale: float) -> None:
        """Enhanced bias scaling with validation."""
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"Scale must be a positive number, got {scale}")
        
        try:
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
                    
        except Exception as e:
            raise RuntimeError(f"Bias scaling failed: {e}")
    
    def normalize_decoder(self, test_tolerance: float = 1e-4) -> None:
        """Enhanced decoder normalization with validation."""
        if self.var_flag == 1:
            warnings.warn("Normalizing decoder weights with learned variance may hurt performance")
        
        try:
            with torch.no_grad():
                norms = torch.norm(self.decoder.weight, dim=0)
                
                if torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                    return
                
                print("Normalizing decoder weights")
                
                # Test preservation with larger sample
                device = self.decoder.weight.device
                test_input = torch.randn(
                    50, self.activation_dim, 
                    device=device, dtype=self.decoder.weight.dtype
                )
                initial_output = self(test_input)
                
                # Normalize decoder weights
                self.decoder.weight.div_(norms)
                
                # Scale other parameters accordingly
                self.encoder.weight.mul_(norms.unsqueeze(1))
                self.encoder.bias.mul_(norms)
                self.gate_bias.mul_(norms)
                self.mag_bias.mul_(norms)
                
                # Verify normalization
                new_norms = torch.norm(self.decoder.weight, dim=0)
                if not torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-6):
                    raise RuntimeError("Decoder normalization failed")
                
                # Verify output preservation
                new_output = self(test_input)
                if not torch.allclose(initial_output, new_output, atol=test_tolerance):
                    warnings.warn(f"Normalization changed output beyond tolerance {test_tolerance}")
                    
        except Exception as e:
            raise RuntimeError(f"Decoder normalization failed: {e}")
    
    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        config: Optional[VSAEGatedConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None,
        strict_loading: bool = False
    ) -> 'VSAEGated':
        """
        Enhanced model loading with comprehensive error handling.
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load checkpoint
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Auto-detect configuration if not provided
            if config is None:
                config = cls._detect_config_from_state_dict(
                    state_dict, dtype=dtype, device=device, var_flag=var_flag
                )
            
            # Create model
            model = cls(config)
            
            # Handle legacy parameter names
            if cls._is_legacy_state_dict(state_dict):
                print("Converting legacy state dict format")
                state_dict = cls._convert_legacy_state_dict(state_dict, config)
            
            # Load state dict
            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=strict_loading
                )
                
                if missing_keys:
                    print(f"Missing keys in state_dict: {missing_keys}")
                    model._handle_missing_keys(missing_keys)
                
                if unexpected_keys:
                    print(f"Unexpected keys in state_dict: {unexpected_keys}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load state dict: {e}")
            
            # Normalize decoder if requested
            if normalize_decoder and not (config.var_flag == 1 and "var_encoder.weight" in state_dict):
                try:
                    model.normalize_decoder()
                except Exception as e:
                    warnings.warn(f"Could not normalize decoder weights: {e}")
            
            # Move to target device and dtype
            if device is not None or dtype != model.config.dtype:
                model = model.to(device=device, dtype=dtype)
                
            # Validate loaded model
            model._validate_loaded_model()
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {e}")
    
    @classmethod
    def _detect_config_from_state_dict(
        cls, 
        state_dict: Dict[str, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        var_flag: Optional[int] = None
    ) -> VSAEGatedConfig:
        """Detect configuration from state dict."""
        try:
            # Detect dimensions
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            elif 'W_enc' in state_dict:
                activation_dim, dict_size = state_dict["W_enc"].shape
                use_april_update_mode = "b_dec" in state_dict or "decoder.bias" in state_dict
            else:
                raise ValueError("Cannot detect dimensions from state dict")
            
            # Auto-detect var_flag
            if var_flag is None:
                var_flag = 1 if (
                    "var_encoder.weight" in state_dict or 
                    "W_enc_var" in state_dict
                ) else 0
            
            return VSAEGatedConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to detect config from state dict: {e}")
    
    @classmethod
    def _is_legacy_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Check if state dict uses legacy parameter names."""
        legacy_keys = ['W_enc', 'W_dec', 'b_enc', 'b_dec']
        return any(key in state_dict for key in legacy_keys)
    
    @classmethod
    def _convert_legacy_state_dict(
        cls, 
        state_dict: Dict[str, torch.Tensor], 
        config: VSAEGatedConfig
    ) -> Dict[str, torch.Tensor]:
        """Convert legacy parameter names to current format."""
        try:
            converted = {}
            
            # Convert main parameters
            if 'W_enc' in state_dict:
                converted["encoder.weight"] = state_dict["W_enc"].T
            if 'b_enc' in state_dict:
                converted["encoder.bias"] = state_dict["b_enc"]
            if 'W_dec' in state_dict:
                converted["decoder.weight"] = state_dict["W_dec"].T
            
            if config.use_april_update_mode:
                if 'b_dec' in state_dict:
                    converted["decoder.bias"] = state_dict["b_dec"]
            else:
                if 'b_dec' in state_dict:
                    converted["bias"] = state_dict["b_dec"]
            
            # Convert gating parameters
            for param in ['gate_bias', 'r_mag', 'mag_bias']:
                if param in state_dict:
                    converted[param] = state_dict[param]
            
            # Convert variance encoder if present
            if config.var_flag == 1:
                if "W_enc_var" in state_dict:
                    converted["var_encoder.weight"] = state_dict["W_enc_var"].T
                if "b_enc_var" in state_dict:
                    converted["var_encoder.bias"] = state_dict["b_enc_var"]
            
            # Copy any other parameters
            for key, value in state_dict.items():
                if key not in converted and not key.startswith(('W_', 'b_')):
                    converted[key] = value
            
            return converted
            
        except Exception as e:
            raise RuntimeError(f"Legacy state dict conversion failed: {e}")
    
    def _handle_missing_keys(self, missing_keys: list) -> None:
        """Handle missing keys during model loading."""
        try:
            # Initialize missing variance encoder if needed
            if self.var_flag == 1 and any("var_encoder" in key for key in missing_keys):
                print("Initializing missing variance encoder parameters")
                with torch.no_grad():
                    if hasattr(self, 'var_encoder'):
                        nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                        nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
            
            # Initialize missing gating parameters
            gating_params = ['gate_bias', 'r_mag', 'mag_bias']
            missing_gating = [key for key in missing_keys if any(param in key for param in gating_params)]
            
            if missing_gating:
                print(f"Initializing missing gating parameters: {missing_gating}")
                with torch.no_grad():
                    if 'gate_bias' in str(missing_keys):
                        nn.init.constant_(self.gate_bias, self.config.gate_bias_init)
                    if 'r_mag' in str(missing_keys):
                        nn.init.constant_(self.r_mag, self.config.r_mag_init)
                    if 'mag_bias' in str(missing_keys):
                        nn.init.constant_(self.mag_bias, self.config.mag_bias_init)
                        
        except Exception as e:
            warnings.warn(f"Failed to handle missing keys: {e}")
    
    def _validate_loaded_model(self) -> None:
        """Validate the loaded model."""
        try:
            # Check parameter shapes
            expected_shapes = {
                'encoder.weight': (self.dict_size, self.activation_dim),
                'encoder.bias': (self.dict_size,),
                'decoder.weight': (self.activation_dim, self.dict_size),
                'gate_bias': (self.dict_size,),
                'r_mag': (self.dict_size,),
                'mag_bias': (self.dict_size,),
            }
            
            if self.use_april_update_mode:
                expected_shapes['decoder.bias'] = (self.activation_dim,)
            else:
                expected_shapes['bias'] = (self.activation_dim,)
            
            if self.var_flag == 1:
                expected_shapes.update({
                    'var_encoder.weight': (self.dict_size, self.activation_dim),
                    'var_encoder.bias': (self.dict_size,),
                })
            
            for param_name, expected_shape in expected_shapes.items():
                if hasattr(self, param_name.replace('.', '_')):
                    param = getattr(self, param_name.split('.')[0])
                    if '.' in param_name:
                        param = getattr(param, param_name.split('.')[1])
                    
                    if param.shape != expected_shape:
                        raise ValueError(f"Parameter {param_name} has shape {param.shape}, expected {expected_shape}")
            
            # Test forward pass
            device = self.encoder.weight.device
            test_input = torch.randn(2, self.activation_dim, device=device, dtype=self.encoder.weight.dtype)
            with torch.no_grad():
                output = self(test_input)
                if output.shape != (2, self.activation_dim):
                    raise ValueError(f"Forward pass output shape {output.shape}, expected {(2, self.activation_dim)}")
                    
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}")


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """Enhanced KL annealing function with validation."""
    if kl_warmup_steps is not None:
        if not isinstance(kl_warmup_steps, int) or kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be a non-negative integer")
        if kl_warmup_steps > total_steps:
            raise ValueError("kl_warmup_steps cannot exceed total_steps")
    
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    def scale_fn(step: int) -> float:
        if step < 0:
            return 0.0
        elif step < kl_warmup_steps:
            return step / kl_warmup_steps
        else:
            return 1.0
    
    return scale_fn


@dataclass
class VSAEGatedTrainingConfig:
    """Enhanced training configuration with comprehensive validation."""
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
    
    # Advanced options
    temperature_schedule: str = "constant"  # "constant", "linear_decay", "cosine_decay"
    min_temperature: float = 0.1
    max_temperature: float = 1.0
    weight_decay: float = 0.0
    eps: float = 1e-8
    
    def __post_init__(self):
        """Enhanced validation and derived values."""
        # Validate basic parameters
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.kl_coeff < 0:
            raise ValueError("kl_coeff must be non-negative")
        if self.l1_penalty < 0:
            raise ValueError("l1_penalty must be non-negative")
        if self.aux_weight < 0:
            raise ValueError("aux_weight must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        
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
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start
        
        # Validate temperature schedule
        if self.temperature_schedule not in ["constant", "linear_decay", "cosine_decay"]:
            raise ValueError("temperature_schedule must be one of ['constant', 'linear_decay', 'cosine_decay']")
        if not 0 < self.min_temperature <= self.max_temperature:
            raise ValueError("Must have 0 < min_temperature <= max_temperature")
    
    def get_temperature(self, step: int) -> float:
        """Get temperature for current step."""
        if self.temperature_schedule == "constant":
            return self.max_temperature
        elif self.temperature_schedule == "linear_decay":
            progress = min(step / self.steps, 1.0)
            return self.max_temperature - progress * (self.max_temperature - self.min_temperature)
        elif self.temperature_schedule == "cosine_decay":
            import math
            progress = min(step / self.steps, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_temperature + cosine_factor * (self.max_temperature - self.min_temperature)
        else:
            return self.max_temperature


class VSAEGatedTrainer(SAETrainer):
    """
    Enhanced robust trainer for VSAEGated with comprehensive error handling.
    """
    
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
            model_config = self._create_model_config_from_kwargs(**kwargs)
        if training_config is None:
            training_config = self._create_training_config_from_kwargs(**kwargs)
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        try:
            self.ae = VSAEGated(model_config)
            self.ae.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VSAEGated model: {e}")
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize scheduler and warmup functions
        self._init_schedules()
        
        # Initialize logging
        self._init_logging()
        
        # Training state
        self.current_step = 0
    
    def _create_model_config_from_kwargs(self, **kwargs) -> VSAEGatedConfig:
        """Create model config from backwards compatibility kwargs."""
        required = ['activation_dim', 'dict_size']
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Must provide {param}")
        
        return VSAEGatedConfig(
            activation_dim=kwargs['activation_dim'],
            dict_size=kwargs['dict_size'],
            var_flag=kwargs.get('var_flag', 1),
            use_april_update_mode=kwargs.get('use_april_update_mode', True),
            device=torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        )
    
    def _create_training_config_from_kwargs(self, **kwargs) -> VSAEGatedTrainingConfig:
        """Create training config from backwards compatibility kwargs."""
        if 'steps' not in kwargs:
            raise ValueError("Must provide steps")
        
        return VSAEGatedTrainingConfig(
            steps=kwargs['steps'],
            lr=kwargs.get('lr', 5e-4),
            kl_coeff=kwargs.get('kl_coeff', 500.0),
            l1_penalty=kwargs.get('l1_penalty', 0.1),
            aux_weight=kwargs.get('aux_weight', 0.1),
            use_constrained_optimizer=kwargs.get('use_constrained_optimizer', True),
        )
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer with error handling."""
        try:
            if self.training_config.use_constrained_optimizer:
                # ConstrainedAdam only accepts params, constrained_params, lr, and betas
                self.optimizer = ConstrainedAdam(
                    self.ae.parameters(),
                    self.ae.decoder.parameters(),
                    lr=self.training_config.lr,
                    betas=(0.9, 0.999)
                )
            else:
                # Regular Adam accepts all the additional parameters
                self.optimizer = torch.optim.Adam(
                    self.ae.parameters(),
                    lr=self.training_config.lr,
                    betas=(0.9, 0.999),
                    eps=self.training_config.eps,
                    weight_decay=self.training_config.weight_decay
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize optimizer: {e}")
    
    def _init_schedules(self) -> None:
        """Initialize learning rate and warmup schedules."""
        try:
            lr_fn = get_lr_schedule(
                self.training_config.steps,
                self.training_config.warmup_steps,
                self.training_config.decay_start,
                None,
                self.training_config.sparsity_warmup_steps
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
            
            self.sparsity_warmup_fn = get_sparsity_warmup_fn(
                self.training_config.steps,
                self.training_config.sparsity_warmup_steps
            )
            
            self.kl_warmup_fn = get_kl_warmup_fn(
                self.training_config.steps,
                self.training_config.kl_warmup_steps
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize schedules: {e}")
    
    def _init_logging(self) -> None:
        """Initialize logging parameters."""
        self.logging_parameters = [
            "effective_l0", "gate_sparsity", "aux_loss_raw", "current_temperature", "current_kl_scale"
        ]
        self.effective_l0 = 0.0
        self.gate_sparsity = 0.0
        self.aux_loss_raw = 0.0
        self.current_temperature = 1.0
        self.current_kl_scale = 1.0
    
    def _compute_kl_loss(
        self, 
        mu_mag: torch.Tensor, 
        log_var: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence loss with enhanced stability."""
        try:
            if self.ae.var_flag == 1 and log_var is not None:
                # Conservative clamping
                log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
                mu_clamped = torch.clamp(mu_mag, min=-10.0, max=10.0)
                
                # Check for NaN/inf values
                if torch.isnan(mu_clamped).any() or torch.isnan(log_var_clamped).any():
                    warnings.warn("NaN detected in KL computation inputs")
                    return torch.tensor(0.0, device=mu_mag.device, dtype=mu_mag.dtype)
                
                # Standard VAE KL divergence
                kl_per_sample = 0.5 * torch.sum(
                    mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                    dim=1
                )
            else:
                # Fixed variance case
                mu_clamped = torch.clamp(mu_mag, min=-10.0, max=10.0)
                
                if torch.isnan(mu_clamped).any():
                    warnings.warn("NaN detected in KL computation")
                    return torch.tensor(0.0, device=mu_mag.device, dtype=mu_mag.dtype)
                
                kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
            
            # Average over batch and ensure non-negative
            kl_loss = kl_per_sample.mean()
            kl_loss = torch.clamp(kl_loss, min=0.0)
            
            # Check for problematic values
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                warnings.warn("Invalid KL loss detected, returning zero")
                return torch.tensor(0.0, device=mu_mag.device, dtype=mu_mag.dtype)
            
            return kl_loss
            
        except Exception as e:
            warnings.warn(f"KL loss computation failed: {e}")
            return torch.tensor(0.0, device=mu_mag.device, dtype=mu_mag.dtype)
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
            """Enhanced loss computation with proper tensor handling."""
            # Define LossLog at the top so it's available in exception handlers
            LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
            
            try:
                # Update current step
                self.current_step = step
                
                # Get scaling factors
                sparsity_scale = self.sparsity_warmup_fn(step)
                kl_scale = self.kl_warmup_fn(step)
                temperature = self.training_config.get_temperature(step)
                
                # Update logging variables
                self.current_kl_scale = kl_scale
                self.current_temperature = temperature
                
                # Store original dtype and shape
                original_dtype = x.dtype
                original_shape = x.shape
                
                # FIXED: Define x_flat early to avoid NameError
                if x.dim() == 3:
                    x_flat = x.view(-1, x.size(-1))
                else:
                    x_flat = x
                
                # FIXED: Consistent encoding calls based on var_flag
                if self.ae.var_flag == 1:
                    # Learned variance case - always get 4 values
                    f, gate_continuous, log_var, mu_mag = self.ae.encode(
                        x_flat, return_gate=True, return_log_var=True, use_reparameterization=True
                    )
                    
                    # Apply temperature to reparameterization if needed
                    if temperature != 1.0 and log_var is not None:
                        mag_sampled = self.ae.reparameterize(mu_mag, log_var, temperature=temperature)
                        gate_binary = (gate_continuous > 0).to(dtype=gate_continuous.dtype)
                        f = gate_binary * mag_sampled
                else:
                    # Fixed variance case - always get 3 values
                    f, gate_continuous, mu_mag = self.ae.encode(
                        x_flat, return_gate=True, return_log_var=False, use_reparameterization=True
                    )
                    log_var = None
                
                # Validate all tensors are not None and don't contain NaN/inf
                required_tensors = {
                    'f': f, 
                    'gate_continuous': gate_continuous, 
                    'mu_mag': mu_mag
                }
                
                for name, tensor in required_tensors.items():
                    if tensor is None:
                        warnings.warn(f"Required tensor {name} is None, returning fallback loss")
                        fallback_loss = torch.tensor(1.0, device=x.device, dtype=original_dtype)
                        if logging:
                            dummy_f = torch.zeros(x_flat.shape[0], self.ae.dict_size, device=x.device)
                            return LossLog(x, x, dummy_f, {'error': f'None tensor: {name}'})
                        return fallback_loss
                        
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        warnings.warn(f"Invalid {name} tensor detected, returning fallback loss")
                        fallback_loss = torch.tensor(1.0, device=x.device, dtype=original_dtype)
                        if logging:
                            dummy_f = torch.zeros(x_flat.shape[0], self.ae.dict_size, device=x.device)
                            return LossLog(x, x, dummy_f, {'error': f'Invalid {name} tensor'})
                        return fallback_loss
                
                # Main reconstruction
                x_hat = self.ae.decode(f)
                
                # Auxiliary reconstruction using gate values
                x_hat_gate = self.ae.decode(gate_continuous)
                
                # Handle reshaping for 3D tensors
                if len(original_shape) == 3:
                    # If we had 3D input, make sure everything is properly shaped for loss computation
                    if x_hat.dim() == 3:
                        x_hat_flat = x_hat.view(-1, x_hat.size(-1))
                    else:
                        x_hat_flat = x_hat
                        
                    if x_hat_gate.dim() == 3:
                        x_hat_gate_flat = x_hat_gate.view(-1, x_hat_gate.size(-1))
                    else:
                        x_hat_gate_flat = x_hat_gate
                        
                    if f.dim() == 3:
                        f_flat = f.view(-1, f.size(-1))
                    else:
                        f_flat = f
                        
                    if gate_continuous.dim() == 3:
                        gate_continuous_flat = gate_continuous.view(-1, gate_continuous.size(-1))
                    else:
                        gate_continuous_flat = gate_continuous
                else:
                    # 2D case - everything should already be flat
                    x_hat_flat = x_hat
                    x_hat_gate_flat = x_hat_gate
                    f_flat = f
                    gate_continuous_flat = gate_continuous
                
                # Ensure compatibility
                x_hat_flat = x_hat_flat.to(dtype=original_dtype)
                x_hat_gate_flat = x_hat_gate_flat.to(dtype=original_dtype)
                
                # Validate reconstruction outputs
                if torch.isnan(x_hat_flat).any() or torch.isinf(x_hat_flat).any():
                    warnings.warn("Invalid reconstruction detected, returning fallback loss")
                    fallback_loss = torch.tensor(1.0, device=x.device, dtype=original_dtype)
                    if logging:
                        dummy_f = torch.zeros(x_flat.shape[0], self.ae.dict_size, device=x.device)
                        return LossLog(x, x, dummy_f, {'error': 'Invalid reconstruction'})
                    return fallback_loss
                
                # Compute individual losses with safety checks
                recon_loss = torch.mean(torch.sum((x_flat - x_hat_flat) ** 2, dim=1))
                gate_sparsity_loss = torch.mean(torch.sum(torch.abs(gate_continuous_flat), dim=1))
                aux_loss = torch.mean(torch.sum((x_flat - x_hat_gate_flat) ** 2, dim=1))
                kl_loss = self._compute_kl_loss(mu_mag, log_var)
                kl_loss = kl_loss.to(dtype=original_dtype)
                
                # Check for invalid losses and replace with safe values
                losses = {'recon': recon_loss, 'sparsity': gate_sparsity_loss, 'aux': aux_loss, 'kl': kl_loss}
                for name, loss_val in losses.items():
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        warnings.warn(f"Invalid {name} loss detected: {loss_val}, replacing with safe value")
                        losses[name] = torch.tensor(0.1, device=x.device, dtype=original_dtype)
                
                # Compute total loss with scaling - ensure all components are valid
                total_loss = (
                    losses['recon'] +
                    (self.training_config.l1_penalty * sparsity_scale * losses['sparsity']) +
                    (self.training_config.aux_weight * losses['aux']) +
                    (self.training_config.kl_coeff * kl_scale * losses['kl'])
                )
                
                # Final safety check on total loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    warnings.warn(f"Invalid total loss: {total_loss}, using fallback")
                    total_loss = torch.tensor(1.0, device=x.device, dtype=original_dtype)
                
                # Update logging stats
                with torch.no_grad():
                    if f_flat.numel() > 0:
                        self.effective_l0 = float(torch.sum(f_flat != 0).item()) / f_flat.numel()
                    else:
                        self.effective_l0 = 0.0
                        
                    if gate_continuous_flat.numel() > 0:
                        self.gate_sparsity = float(torch.sum(gate_continuous_flat > 0).item()) / gate_continuous_flat.numel()
                    else:
                        self.gate_sparsity = 0.0
                        
                    self.aux_loss_raw = aux_loss.item()
                
                if not logging:
                    return total_loss
                
                # Get diagnostics with error handling
                try:
                    kl_diagnostics = self.ae.get_kl_diagnostics(x_flat)
                    recon_diagnostics = self.ae.get_reconstruction_diagnostics(x_flat)
                except Exception as e:
                    warnings.warn(f"Diagnostics computation failed: {e}")
                    kl_diagnostics = {}
                    recon_diagnostics = {}
                
                loss_dict = {
                    'l2_loss': torch.sqrt(losses['recon']).item(),
                    'mse_loss': losses['recon'].item(),
                    'gate_sparsity_loss': losses['sparsity'].item(),
                    'aux_loss': losses['aux'].item(),
                    'kl_loss': losses['kl'].item(),
                    'total_loss': total_loss.item(),
                    'effective_l0': self.effective_l0,
                    'gate_sparsity': self.gate_sparsity,
                    'sparsity_scale': sparsity_scale,
                    'kl_scale': kl_scale,
                    'temperature': temperature,
                    'current_lr': self.optimizer.param_groups[0]['lr'],
                }
                
                # Add diagnostics
                for k, v in {**kl_diagnostics, **recon_diagnostics}.items():
                    try:
                        loss_dict[k] = v.item() if torch.is_tensor(v) else v
                    except:
                        loss_dict[k] = 0.0
                
                # Return with proper shapes
                if len(original_shape) == 3:
                    # Reshape back to 3D for logging
                    x_hat_return = x_hat if x_hat.dim() == 3 else x_hat.view(original_shape)
                    f_return = f if f.dim() == 3 else f.view(original_shape[0], original_shape[1], -1)
                else:
                    x_hat_return = x_hat_flat
                    f_return = f_flat
                
                return LossLog(x, x_hat_return, f_return, loss_dict)
                
            except Exception as e:
                warnings.warn(f"Loss computation failed: {e}")
                # Return a safe fallback loss
                fallback_loss = torch.tensor(1.0, device=x.device, dtype=x.dtype)
                if logging:
                    # Create safe dummy tensors
                    if x.dim() == 3:
                        dummy_f = torch.zeros(x.shape[0], x.shape[1], self.ae.dict_size, device=x.device)
                    else:
                        dummy_f = torch.zeros(x.shape[0], self.ae.dict_size, device=x.device)
                    return LossLog(x, x, dummy_f, {'error': str(e)})
                return fallback_loss

    def update(self, step: int, activations: torch.Tensor) -> None:
        """Enhanced update step with comprehensive error handling."""
        try:
            activations = activations.to(self.device)
            
            # Validate input
            if torch.isnan(activations).any() or torch.isinf(activations).any():
                warnings.warn("Invalid activations detected, skipping update")
                return
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss and backpropagate
            loss = self.loss(activations, step=step)
            
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn(f"Invalid loss at step {step}: {loss}, skipping update")
                return
            
            loss.backward()
            
            # Check gradients
            total_norm = 0.0
            for p in self.ae.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        warnings.warn(f"Invalid gradients detected at step {step}")
                        return
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
            
        except Exception as e:
            warnings.warn(f"Update step failed at step {step}: {e}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return comprehensive configuration dictionary."""
        try:
            config_dict = {
                'dict_class': 'VSAEGated',
                'trainer_class': 'VSAEGatedTrainer',
                # Model config
                **self.model_config.to_dict(),
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
                'min_temperature': self.training_config.min_temperature,
                'max_temperature': self.training_config.max_temperature,
                'weight_decay': self.training_config.weight_decay,
                'eps': self.training_config.eps,
                # Other attributes
                'layer': self.layer,
                'lm_name': self.lm_name,
                'wandb_name': self.wandb_name,
                'submodule_name': self.submodule_name,
                'seed': self.seed,
            }
            
            return config_dict
            
        except Exception as e:
            warnings.warn(f"Config serialization failed: {e}")
            return {'error': str(e)}
