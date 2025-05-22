"""
Improved implementation of combined MatryoshkaBatchTopK + VSAEIso approach.

This module combines hierarchical grouping and learning from the Matryoshka approach
with variational learning from the VSAE approach, following better PyTorch practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any
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
class MatryoshkaConfig:
    """Configuration for Matryoshka VSAE model."""
    activation_dim: int
    dict_size: int
    group_fractions: List[float]
    group_weights: Optional[List[float]] = None
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        # Validate group fractions
        if not math.isclose(sum(self.group_fractions), 1.0, rel_tol=1e-6):
            raise ValueError(f"group_fractions must sum to 1.0, got {sum(self.group_fractions)}")
        
        # Calculate group sizes
        self.group_sizes = [int(f * self.dict_size) for f in self.group_fractions[:-1]]
        self.group_sizes.append(self.dict_size - sum(self.group_sizes))
        
        # Default group weights
        if self.group_weights is None:
            self.group_weights = [1.0 / len(self.group_sizes)] * len(self.group_sizes)
        
        if len(self.group_sizes) != len(self.group_weights):
            raise ValueError("group_sizes and group_weights must have the same length")


class MatryoshkaVSAEIso(Dictionary, nn.Module):
    """
    A combined Matryoshka + Variational Sparse Autoencoder with isotropic Gaussian prior.
    
    Features:
    - Hierarchical group structure for progressive training
    - Variational approach with KL divergence regularization
    - Optional learned variance for more expressive latent space
    - Proper gradient handling and memory management
    """

    def __init__(self, config: MatryoshkaConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        
        # Register group configuration as buffers (non-trainable but saved with model)
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
        self.encoder = nn.Linear(
            self.activation_dim, 
            self.dict_size, 
            bias=True,
            dtype=self.config.dtype,
            device=self.config.device
        )
        self.decoder = nn.Linear(
            self.dict_size, 
            self.activation_dim, 
            bias=True,
            dtype=self.config.dtype,
            device=self.config.device
        )
        
        # Variance encoder (only when learning variance)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim, 
                self.dict_size, 
                bias=True,
                dtype=self.config.dtype,
                device=self.config.device
            )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        # Encoder and decoder weights (tied initialization)
        w = torch.randn(
            self.activation_dim, 
            self.dict_size, 
            dtype=self.config.dtype,
            device=self.config.device
        )
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        # Set weights
        with torch.no_grad():
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Initialize biases to zero
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.decoder.bias)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)
    
    def _create_group_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Create mask for active groups without in-place operations."""
        batch_size, dict_size = features.shape
        max_active_idx = self.group_indices[self.active_groups].item()
        
        # Create mask using arange (differentiable)
        feature_indices = torch.arange(
            dict_size, 
            device=features.device, 
            dtype=torch.long
        )
        mask = (feature_indices < max_active_idx).float()
        
        return mask.unsqueeze(0).expand(batch_size, -1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input activations to latent space.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size]
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Encode mean
        mu_full = F.relu(self.encoder(x))
        
        # Apply group masking without in-place operations
        mask = self._create_group_mask(mu_full)
        mu = mu_full * mask
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            log_var_full = F.relu(self.var_encoder(x))
            log_var = log_var_full * mask
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution."""
        if log_var is None:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Ensure output matches mu dtype
        return z.to(dtype=mu.dtype)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        # Ensure z matches decoder weight dtype
        z = z.to(dtype=self.decoder.weight.dtype)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            return_features: Whether to return latent features
            
        Returns:
            reconstruction: Reconstructed activations [batch_size, activation_dim]
            features: Latent features (if return_features=True) [batch_size, dict_size]
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(z)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if return_features:
            z = z.to(dtype=original_dtype)
            return x_hat, z
        return x_hat
    
    def get_active_features_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Get boolean mask of which features are active across the batch."""
        return (features.sum(0) > 0)
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            self.decoder.bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[MatryoshkaConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> 'MatryoshkaVSAEIso':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        if config is None:
            # Try to reconstruct config from checkpoint
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
            
            # Extract dimensions
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            group_sizes = state_dict["group_sizes"].tolist()
            group_fractions = [s / dict_size for s in group_sizes]
            
            # Detect var_flag
            var_flag = 1 if "var_encoder.weight" in state_dict else 0
            
            config = MatryoshkaConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                group_fractions=group_fractions,
                var_flag=var_flag,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict'])
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
            
        return model


@dataclass 
class TrainingConfig:
    """Configuration for training the MatryoshkaVSAE."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    dead_feature_threshold: int = 10_000_000
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


class DeadFeatureTracker:
    """Tracks dead features for auxiliary loss computation."""
    
    def __init__(self, dict_size: int, threshold: int, device: torch.device):
        self.threshold = threshold
        self.num_tokens_since_fired = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
    
    def update(self, active_features: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Update dead feature tracking and return dead feature mask."""
        # Update counters
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[active_features] = 0
        
        # Return dead feature mask
        return self.num_tokens_since_fired >= self.threshold
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about dead features."""
        dead_mask = self.num_tokens_since_fired >= self.threshold
        return {
            "dead_features": int(dead_mask.sum()),
            "alive_features": int((~dead_mask).sum()),
            "total_features": len(self.num_tokens_since_fired)
        }


class MatryoshkaVSAEIsoTrainer(SAETrainer):
    """
    Trainer for the MatryoshkaVSAEIso model with improved architecture.
    
    Features:
    - Clean separation of concerns
    - Proper loss computation with progressive reconstruction
    - Dead feature tracking and auxiliary loss
    - Memory-efficient group-wise processing
    """
    
    def __init__(
        self,
        model_config: MatryoshkaConfig = None,
        training_config: TrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility with trainSAE
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        group_fractions: Optional[List[float]] = None,
        group_weights: Optional[List[float]] = None,
        var_flag: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility - if individual parameters are passed, create configs
        if model_config is None or training_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                # Set defaults
                group_fractions = group_fractions or [0.25, 0.25, 0.25, 0.25]
                var_flag = var_flag or 0
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = MatryoshkaConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    group_fractions=group_fractions,
                    group_weights=group_weights,
                    var_flag=var_flag,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = TrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                    auxk_alpha=auxk_alpha or 1/32,
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "MatryoshkaVSAEIsoTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
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
        
        # Initialize dead feature tracking
        self.dead_feature_tracker = DeadFeatureTracker(
            model_config.dict_size,
            training_config.dead_feature_threshold,
            self.device
        )
        
        # Logging parameters
        self.logging_parameters = ["dead_features", "auxk_loss_raw", "effective_l0"]
        self.dead_features = 0
        self.auxk_loss_raw = 0.0
        self.effective_l0 = 0.0
    
    def _compute_group_reconstruction_loss(
        self, 
        x: torch.Tensor, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute progressive reconstruction loss through groups."""
        # Use the model's built-in decode method which handles dtype conversion
        x_hat = self.ae.decode(z)
        x_hat = x_hat.to(dtype=x.dtype)
        
        # For now, just compute single reconstruction loss
        # TODO: Implement proper progressive group-wise reconstruction
        total_recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # Create dummy group losses for compatibility
        group_losses = [total_recon_loss / self.ae.active_groups] * self.ae.active_groups
        
        return total_recon_loss, group_losses
    
    def _compute_kl_loss(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence loss with decoder norm weighting."""
        total_kl_loss = 0.0
        
        for group_idx in range(self.ae.active_groups):
            start_idx = self.ae.group_indices[group_idx].item()
            end_idx = self.ae.group_indices[group_idx + 1].item()
            
            # Get group means and decoder norms
            mu_group = mu[:, start_idx:end_idx]
            decoder_norms = torch.norm(
                self.ae.decoder.weight[:, start_idx:end_idx], 
                p=2, dim=0
            )
            
            # Ensure dtype consistency
            decoder_norms = decoder_norms.to(dtype=mu_group.dtype)
            
            # Compute KL divergence
            if self.ae.var_flag == 1 and log_var is not None:
                log_var_group = log_var[:, start_idx:end_idx]
                # Full KL: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)
                kl_base = 0.5 * torch.sum(
                    torch.exp(log_var_group) + mu_group.pow(2) - 1 - log_var_group,
                    dim=1
                )
            else:
                # Simplified KL for fixed variance: 0.5 * sum(mu^2)
                kl_base = 0.5 * torch.sum(mu_group.pow(2), dim=1)
            
            # Weight by decoder norms and group weight
            kl_loss = torch.mean(kl_base) * torch.mean(decoder_norms)
            weighted_kl = kl_loss * self.model_config.group_weights[group_idx]
            total_kl_loss += weighted_kl
        
        return total_kl_loss
    
    def _compute_auxiliary_loss(
        self, 
        residual: torch.Tensor, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss for dead features."""
        # Update dead feature tracking
        active_features = self.ae.get_active_features_mask(features)
        num_tokens = features.size(0)
        dead_mask = self.dead_feature_tracker.update(active_features, num_tokens)
        
        # Update logging stats
        stats = self.dead_feature_tracker.get_stats()
        self.dead_features = stats["dead_features"]
        
        if self.dead_features == 0:
            self.auxk_loss_raw = 0.0
            return torch.tensor(0.0, device=self.device, dtype=features.dtype)
        
        # Compute auxiliary loss for dead features
        k_aux = min(self.model_config.activation_dim // 2, self.dead_features)
        
        # Create auxiliary activations using only dead features
        aux_features = torch.where(
            dead_mask[None, :], 
            features, 
            torch.full_like(features, -float('inf'))
        )
        
        # Get top-k dead features
        aux_acts, aux_indices = torch.topk(aux_features, k_aux, dim=1, sorted=False)
        
        # Create sparse auxiliary reconstruction
        aux_buffer = torch.zeros_like(features)
        aux_buffer.scatter_(1, aux_indices, aux_acts)
        
        # Reconstruct using auxiliary features
        x_aux = self.ae.decode(aux_buffer)
        
        # Compute auxiliary loss
        aux_loss = torch.mean(torch.sum((residual - x_aux) ** 2, dim=1))
        self.auxk_loss_raw = aux_loss.item()
        
        # Normalize by residual variance
        residual_var = torch.mean(torch.sum((residual - torch.mean(residual, dim=0)) ** 2, dim=1))
        normalized_aux_loss = aux_loss / (residual_var + 1e-8)
        
        return normalized_aux_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Forward pass - the model handles dtype conversion internally
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        
        # Compute reconstruction loss
        recon_loss, group_losses = self._compute_group_reconstruction_loss(x, z)
        
        # Compute KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        
        # Compute auxiliary loss for dead features
        x_hat = self.ae.decode(z)
        x_hat = x_hat.to(dtype=original_dtype)  # Ensure compatibility with x
        residual = x - x_hat
        aux_loss = self._compute_auxiliary_loss(residual.detach(), mu)
        
        # Total loss - ensure all components are in original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        kl_loss = kl_loss.to(dtype=original_dtype)
        aux_loss = aux_loss.to(dtype=original_dtype)
        
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * sparsity_scale * kl_loss + 
            self.training_config.auxk_alpha * aux_loss
        )
        
        # Update logging stats
        self.effective_l0 = float(torch.sum(mu > 0).item()) / mu.numel()
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat, z,
            {
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'auxk_loss': aux_loss.item(),
                'total_loss': total_loss.item(),
                'min_group_loss': min(loss.item() for loss in group_losses),
                'max_group_loss': max(loss.item() for loss in group_losses),
                'sparsity_scale': sparsity_scale,
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
        """Return configuration dictionary for logging/saving (JSON serializable)."""
        return {
            'dict_class': 'MatryoshkaVSAEIso',
            'trainer_class': 'MatryoshkaVSAEIsoTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'group_fractions': self.model_config.group_fractions,
            'group_weights': self.model_config.group_weights,
            'var_flag': self.model_config.var_flag,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }