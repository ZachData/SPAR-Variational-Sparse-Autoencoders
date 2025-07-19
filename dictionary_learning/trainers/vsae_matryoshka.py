"""
Enhanced MatryoshkaVSAE with Batch Top-K selection mechanism.

This combines the variational autoencoder approach with manual batch-level top-k 
feature selection, giving you fine-grained control over sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from collections import namedtuple
from dataclasses import dataclass
import math

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)


@dataclass
class MatryoshkaVSAEConfig:
    """Configuration for Matryoshka VSAE with Batch Top-K selection."""
    activation_dim: int
    dict_size: int
    k: int  # NEW: Number of features to keep active per batch
    group_fractions: List[float]
    group_weights: Optional[List[float]] = None
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0
    # NEW: Batch top-k specific parameters
    use_batch_topk: bool = True  # Whether to use batch top-k or group masking
    topk_mode: str = "absolute"  # "absolute" or "magnitude" - how to select top-k
    
    def __post_init__(self):
        # Validate group fractions
        if not math.isclose(sum(self.group_fractions), 1.0, rel_tol=1e-6):
            raise ValueError(f"group_fractions must sum to 1.0, got {sum(self.group_fractions)}")
        
        # Calculate group sizes
        self.group_sizes = [int(f * self.dict_size) for f in self.group_fractions[:-1]]
        self.group_sizes.append(self.dict_size - sum(self.group_sizes))
        
        # Default group weights and validate
        if self.group_weights is None:
            self.group_weights = [1.0 / len(self.group_sizes)] * len(self.group_sizes)
        else:
            weight_sum = sum(self.group_weights)
            if not math.isclose(weight_sum, 1.0, rel_tol=1e-6):
                print(f"Warning: group_weights sum to {weight_sum:.6f}, normalizing to 1.0")
                self.group_weights = [w / weight_sum for w in self.group_weights]
        
        # Validate k parameter
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.k > self.dict_size:
            print(f"Warning: k ({self.k}) > dict_size ({self.dict_size}), will be clamped")
            self.k = self.dict_size
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class MatryoshkaVSAEIso(Dictionary, nn.Module):
    """
    Matryoshka VSAE with Batch Top-K feature selection.
    
    This model combines:
    1. Variational autoencoder framework (μ, σ² parameterization)
    2. Hierarchical group structure (Matryoshka)
    3. Batch-level top-k sparsity control
    
    Key features:
    - Manual sparsity control via k parameter
    - Applies top-k to mean values, then masks both μ and log_var
    - Supports both absolute value and magnitude-based top-k selection
    - Maintains VAE properties with proper KL divergence
    """

    def __init__(self, config: MatryoshkaVSAEConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        self.k = config.k
        self.use_batch_topk = config.use_batch_topk
        self.topk_mode = config.topk_mode
        
        # Register group configuration as buffers
        self.register_buffer("group_sizes", torch.tensor(config.group_sizes, dtype=torch.long))
        self.register_buffer("group_weights", torch.tensor(config.group_weights, dtype=torch.float32))
        
        # Create group indices for efficient slicing
        group_cumsum = torch.cumsum(torch.tensor([0] + config.group_sizes), dim=0)
        self.register_buffer("group_indices", group_cumsum)
        
        # Number of active groups (can be modified during training)
        self.active_groups = len(config.group_sizes)
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
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
                torch.zeros(self.activation_dim, dtype=dtype, device=device)
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
        w = torch.randn(self.activation_dim, self.dict_size, dtype=dtype, device=device)
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
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction in standard mode."""
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def _apply_batch_topk_mask(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply batch top-k selection to mean values and mask both μ and log_var.
        
        Args:
            mu: Mean values [..., dict_size]
            log_var: Log variance values [..., dict_size] (optional)
            
        Returns:
            Masked mu and log_var tensors
        """
        if not self.use_batch_topk:
            # Fall back to group-based masking
            return self._apply_group_mask(mu, log_var)
        
        # Store original shape for reshaping later
        original_shape = mu.shape
        
        # Flatten all dimensions except the last one
        mu_flat = mu.view(-1, mu.shape[-1])  # [batch_total, dict_size]
        
        if log_var is not None:
            log_var_flat = log_var.view(-1, log_var.shape[-1])
        else:
            log_var_flat = None
        
        # Choose values for top-k selection
        if self.topk_mode == "magnitude":
            # Select based on absolute magnitude of μ
            selection_values = torch.abs(mu_flat).flatten()
        else:  # "absolute"
            # Select based on actual μ values (can select negative values too)
            selection_values = mu_flat.flatten()
        
        # Perform batch top-k selection
        batch_size = mu_flat.shape[0]
        k_total = self.k * batch_size
        
        if k_total > 0 and len(selection_values) > 0:
            k_actual = min(k_total, len(selection_values))
            
            # Get top-k indices
            _, topk_indices = selection_values.topk(k_actual, sorted=False, dim=-1)
            
            # Create mask for selected features
            mask = torch.zeros_like(selection_values)
            mask[topk_indices] = 1.0
            
            # Reshape mask back to feature dimensions
            mask = mask.reshape(mu_flat.shape)  # [batch_total, dict_size]
            
            # Apply mask to mu
            mu_masked = mu_flat * mask
            
            # Apply mask to log_var if present
            if log_var_flat is not None:
                log_var_masked = log_var_flat * mask
            else:
                log_var_masked = None
        else:
            # If k is 0 or no features, return zeros
            mu_masked = torch.zeros_like(mu_flat)
            log_var_masked = torch.zeros_like(log_var_flat) if log_var_flat is not None else None
        
        # Reshape back to original shape
        mu_final = mu_masked.view(original_shape)
        log_var_final = log_var_masked.view(original_shape) if log_var_masked is not None else None
        
        return mu_final, log_var_final
    
    def _apply_group_mask(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply group-based masking (fallback when batch top-k is disabled).
        """
        # Get the feature dimension (last dimension)
        dict_size = mu.shape[-1]
        max_active_idx = self.group_indices[self.active_groups].item()
        
        # Create mask using arange (differentiable)
        feature_indices = torch.arange(dict_size, device=mu.device, dtype=torch.long)
        mask = (feature_indices < max_active_idx).float()
        
        # Expand mask to match all dimensions except the last one
        for _ in range(len(mu.shape) - 1):
            mask = mask.unsqueeze(0)
        
        # Expand to match the input tensor shape
        mask = mask.expand_as(mu)
        
        # Apply mask
        mu_masked = mu * mask
        log_var_masked = log_var * mask if log_var is not None else None
        
        return mu_masked, log_var_masked
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent space with batch top-k selection.
        
        Args:
            x: Input activations [..., activation_dim]
            
        Returns:
            mu: Mean of latent distribution [..., dict_size] (after top-k selection)
            log_var: Log variance [..., dict_size] (after top-k selection, None if var_flag=0)
        """
        x_processed = self._preprocess_input(x)
        
        # Compute full means (unconstrained - no ReLU)
        mu_full = self.encoder(x_processed)
        
        # Compute full variances if learning them
        log_var_full = None
        if self.var_flag == 1:
            # No ReLU on log_var - can be negative
            log_var_full = self.var_encoder(x_processed)
        
        # Apply batch top-k selection to both μ and log_var
        mu, log_var = self._apply_batch_topk_mask(mu_full, log_var_full)
        
        return mu, log_var
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.
        
        Args:
            f: Latent features [..., dict_size]
            
        Returns:
            x_hat: Reconstructed activations [..., activation_dim]
        """
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply reparameterization trick.
        
        Args:
            mu: Mean of latent distribution [..., dict_size]
            log_var: Log variance = log(σ²) [..., dict_size]
            
        Returns:
            z: Sampled latent features [..., dict_size]
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False, 
        ghost_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [..., activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for VSAE
            
        Returns:
            x_hat: Reconstructed activations [..., activation_dim]
            z: Latent features (if output_features=True) [..., dict_size]
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for MatryoshkaVSAEIso")
        
        # Store original dtype and shape
        original_dtype = x.dtype
        original_shape = x.shape
        
        # Encode with batch top-k selection
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(z)
        
        # Ensure output shape matches input shape
        assert x_hat.shape == original_shape, f"Output shape {x_hat.shape} doesn't match input shape {original_shape}"
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            z = z.to(dtype=original_dtype)
            return x_hat, z
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed KL diagnostics for monitoring training."""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            
            # Flatten all dimensions except the last one for computing means
            mu_flat = mu.view(-1, mu.shape[-1])
            
            if self.var_flag == 1 and log_var is not None:
                log_var_flat = log_var.view(-1, log_var.shape[-1])
                log_var_safe = torch.clamp(log_var_flat, -6, 2)
                var = torch.exp(log_var_safe)  # σ²
                
                kl_mu = 0.5 * torch.sum(mu_flat.pow(2), dim=1).mean()
                kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                kl_total = kl_mu + kl_var
                
                # Additional batch top-k specific diagnostics
                active_features = (mu_flat != 0).float().sum(dim=-1).mean()
                sparsity_ratio = active_features / self.dict_size
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_mu,
                    'kl_var_term': kl_var,
                    'mean_log_var': log_var_flat.mean(),
                    'mean_var': var.mean(),
                    'mean_mu': mu_flat.mean(),
                    'mean_mu_magnitude': mu_flat.norm(dim=-1).mean(),
                    'mu_std': mu_flat.std(),
                    'active_features': active_features,
                    'sparsity_ratio': sparsity_ratio,
                    'target_k': float(self.k),
                }
            else:
                kl_total = 0.5 * torch.sum(mu_flat.pow(2), dim=1).mean()
                active_features = (mu_flat != 0).float().sum(dim=-1).mean()
                sparsity_ratio = active_features / self.dict_size
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu_flat.mean(),
                    'mean_mu_magnitude': mu_flat.norm(dim=-1).mean(),
                    'mu_std': mu_flat.std(),
                    'active_features': active_features,
                    'sparsity_ratio': sparsity_ratio,
                    'target_k': float(self.k),
                }
    
    def set_k(self, new_k: int) -> None:
        """Dynamically change the k parameter during training."""
        if new_k <= 0:
            raise ValueError("k must be positive")
        if new_k > self.dict_size:
            print(f"Warning: k ({new_k}) > dict_size ({self.dict_size}), will be clamped")
            new_k = self.dict_size
        
        self.k = new_k
        print(f"Set k to {new_k} (sparsity: {new_k/self.dict_size:.3f})")
    
    def set_topk_mode(self, mode: str) -> None:
        """Change the top-k selection mode."""
        if mode not in ["absolute", "magnitude"]:
            raise ValueError("topk_mode must be 'absolute' or 'magnitude'")
        self.topk_mode = mode
        print(f"Set top-k mode to '{mode}'")
    
    def toggle_batch_topk(self, enabled: bool) -> None:
        """Enable or disable batch top-k selection."""
        self.use_batch_topk = enabled
        mode_str = "batch top-k" if enabled else "group masking"
        print(f"Switched to {mode_str} selection")
    
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
    
    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[MatryoshkaVSAEConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'MatryoshkaVSAEIso':
        """Load a pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle legacy format
                activation_dim, dict_size = state_dict.get("W_enc", state_dict["encoder.weight"].T).shape
                use_april_update_mode = "b_dec" in state_dict or "decoder.bias" in state_dict
            
            # Auto-detect other parameters
            var_flag = 1 if ("var_encoder.weight" in state_dict or "W_enc_var" in state_dict) else 0
            k = kwargs.get('k', 64)  # Default k
            
            # Extract group configuration from state dict if available
            if "group_sizes" in state_dict:
                group_sizes = state_dict["group_sizes"].tolist()
                group_fractions = [s / dict_size for s in group_sizes]
            else:
                group_fractions = kwargs.get('group_fractions', [0.25, 0.25, 0.25, 0.25])
            
            if "group_weights" in state_dict:
                group_weights = state_dict["group_weights"].tolist()
            else:
                group_weights = kwargs.get('group_weights', None)
            
            config = MatryoshkaVSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                group_fractions=group_fractions,
                group_weights=group_weights,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                device=device,
                **{k: v for k, v in kwargs.items() if k in MatryoshkaVSAEConfig.__dataclass_fields__}
            )
        
        model = cls(config)
        
        # Load state dict with error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        if device is not None:
            model = model.to(device=device)
            
        return model


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """Return a function that computes KL annealing scale factor at a given step."""
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps, "kl_warmup_steps must be > 0 and <= total_steps"
    
    def scale_fn(step: int) -> float:
        if step < kl_warmup_steps:
            return step / kl_warmup_steps  # Linear warmup from 0 to 1
        else:
            return 1.0
    
    return scale_fn


@dataclass 
class MatryoshkaVSAETrainingConfig:
    """Training configuration for Matryoshka VSAE with Batch Top-K."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)

        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class MatryoshkaVSAEIsoTrainer(SAETrainer):
    """
    Trainer for MatryoshkaVSAE with Batch Top-K selection.
    
    Features manual sparsity control via k parameter while maintaining
    all the benefits of the variational framework.
    """
    
    def __init__(
        self,
        model_config: Optional[MatryoshkaVSAEConfig] = None,
        training_config: Optional[MatryoshkaVSAETrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility
        **kwargs
    ):
        super().__init__(seed)
        
        if model_config is None:
            raise ValueError("Must provide model_config")
        if training_config is None:
            raise ValueError("Must provide training_config")
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "MatryoshkaVSAEIsoTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = MatryoshkaVSAEIso(model_config)
        self.ae.to(self.device)
        
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
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "target_k"]
        self.effective_l0 = 0.0
        self.target_k = float(model_config.k)
    
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute clean KL divergence loss."""
        total_kl_loss = torch.tensor(0.0, dtype=mu.dtype, device=mu.device)
        
        # Flatten all dimensions except the last one
        mu_flat = mu.view(-1, mu.shape[-1])
        if log_var is not None:
            log_var_flat = log_var.view(-1, log_var.shape[-1])
        else:
            log_var_flat = None
        
        # Group-wise KL computation
        for group_idx in range(self.ae.active_groups):
            start_idx = self.ae.group_indices[group_idx].item()
            end_idx = self.ae.group_indices[group_idx + 1].item()
            
            mu_group = mu_flat[:, start_idx:end_idx]
            
            if self.ae.var_flag == 1 and log_var_flat is not None:
                log_var_group = log_var_flat[:, start_idx:end_idx]
                log_var_clamped = torch.clamp(log_var_group, min=-6.0, max=2.0)
                mu_clamped = torch.clamp(mu_group, min=-10.0, max=10.0)
                
                kl_per_sample = 0.5 * torch.sum(
                    torch.exp(log_var_clamped) + mu_clamped.pow(2) - 1 - log_var_clamped,
                    dim=1
                )
            else:
                mu_clamped = torch.clamp(mu_group, min=-10.0, max=10.0)
                kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
            
            kl_group = kl_per_sample.mean()
            kl_group = torch.clamp(kl_group, min=0.0)
            
            weighted_kl = kl_group * self.model_config.group_weights[group_idx]
            total_kl_loss += weighted_kl
        
        return total_kl_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with batch top-k sparsity control."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        kl_scale = self.kl_warmup_fn(step)
        
        # Store original dtype and shape
        original_dtype = x.dtype
        original_shape = x.shape
        
        # Forward pass
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        assert x_hat.shape == original_shape, f"Reconstruction shape mismatch: {x_hat.shape} vs {original_shape}"
        
        # Reconstruction loss
        x_flat = x.view(-1, x.shape[-1])
        x_hat_flat = x_hat.view(-1, x_hat.shape[-1])
        recon_loss = torch.mean(torch.sum((x_flat - x_hat_flat) ** 2, dim=1))
        
        # KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Total loss with separate scaling
        total_loss = recon_loss + self.training_config.kl_coeff * kl_scale * kl_loss
        
        # Update logging stats
        mu_flat = mu.view(-1, mu.shape[-1])
        active_features = torch.sum(mu_flat != 0).item()
        total_features = mu_flat.numel()
        self.effective_l0 = active_features / total_features if total_features > 0 else 0.0
        
        if not logging:
            return total_loss
        
        # Return detailed loss information with diagnostics
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        kl_diagnostics = self.ae.get_kl_diagnostics(x)
        
        return LossLog(
            x, x_hat, z,
            {
                'l2_loss': torch.norm(x_flat - x_hat_flat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'kl_scale': kl_scale,
                'effective_l0': self.effective_l0,
                'target_k': self.target_k,
                **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step."""
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
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
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'MatryoshkaVSAEIso',
            'trainer_class': 'MatryoshkaVSAEIsoTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'group_fractions': self.model_config.group_fractions,
            'group_weights': self.model_config.group_weights,
            'var_flag': self.model_config.var_flag,
            'use_batch_topk': self.model_config.use_batch_topk,
            'topk_mode': self.model_config.topk_mode,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }


# Example usage:
if __name__ == "__main__":
    # Create config with manual k parameter
    config = MatryoshkaVSAEConfig(
        activation_dim=2048,
        dict_size=8192,
        k=64,  # Manual sparsity control!
        group_fractions=[0.25, 0.25, 0.25, 0.25],
        var_flag=1,  # Enable learned variance
        use_batch_topk=True,
        topk_mode="magnitude"  # Select by absolute magnitude
    )
    
    # Create model
    model = MatryoshkaVSAEIso(config)
    
    # Test with some data
    x = torch.randn(32, 2048)  # batch_size=32
    
    # Forward pass
    x_hat, z = model(x, output_features=True)
    
    # Check sparsity
    active_features = (z != 0).float().sum()
    target_active = config.k * x.size(0)  # k per sample * batch_size
    
    print(f"Active features: {active_features.item():.0f}")
    print(f"Target active: {target_active}")
    print(f"Sparsity ratio: {active_features.item() / z.numel():.3f}")
    
    # Dynamically change k during training
    model.set_k(32)  # Make it sparser
    
    # Toggle between batch top-k and group masking
    model.toggle_batch_topk(False)  # Use group masking
    model.toggle_batch_topk(True)   # Back to batch top-k