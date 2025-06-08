"""
Robust Matryoshka Batch Top-K SAE implementation with improved architecture.

Key improvements:
- Dataclass-based configuration management
- Better error handling and validation
- Comprehensive type hints
- Robust weight initialization
- Enhanced numerical stability
- Proper device management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable, List
from collections import namedtuple
from dataclasses import dataclass
from math import isclose

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


def apply_temperature(probabilities: List[float], temperature: float) -> List[float]:
    """
    Apply temperature scaling to a list of probabilities using PyTorch.

    Args:
        probabilities: Initial probability distribution
        temperature: Temperature parameter (> 0)

    Returns:
        Scaled and normalized probabilities
    """
    probs_tensor = torch.tensor(probabilities, dtype=torch.float32)
    logits = torch.log(torch.clamp(probs_tensor, min=1e-8))
    scaled_logits = logits / temperature
    scaled_probs = torch.nn.functional.softmax(scaled_logits, dim=0)
    return scaled_probs.tolist()


@dataclass
class MatryoshkaConfig:
    """Configuration for Matryoshka Batch Top-K SAE model."""
    activation_dim: int
    dict_size: int
    k: int  # Number of features to keep active per batch
    group_fractions: List[float]  # Fractions of dict_size for each group
    group_weights: Optional[List[float]] = None  # Weights for each group in loss
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient
    threshold_beta: float = 0.999  # EMA coefficient for threshold updates
    threshold_start_step: int = 1000  # When to start threshold updates
    dead_feature_threshold: int = 10_000_000  # Steps before feature considered dead
    top_k_aux_fraction: float = 0.5  # Fraction of activation_dim for auxiliary k
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        """Validate and set derived configuration values."""
        if not isclose(sum(self.group_fractions), 1.0, abs_tol=1e-6):
            raise ValueError(f"group_fractions must sum to 1.0, got {sum(self.group_fractions)}")
        
        if any(f <= 0 for f in self.group_fractions):
            raise ValueError("All group fractions must be positive")
            
        if self.k <= 0:
            raise ValueError("k must be positive")
            
        if self.activation_dim <= 0 or self.dict_size <= 0:
            raise ValueError("activation_dim and dict_size must be positive")
            
        # Calculate group sizes
        group_sizes = [int(f * self.dict_size) for f in self.group_fractions[:-1]]
        group_sizes.append(self.dict_size - sum(group_sizes))
        
        if any(size <= 0 for size in group_sizes):
            raise ValueError("All computed group sizes must be positive")
            
        self.group_sizes = group_sizes
        
        # Set default group weights
        if self.group_weights is None:
            self.group_weights = [1.0 / len(group_sizes)] * len(group_sizes)
        elif len(self.group_weights) != len(group_sizes):
            raise ValueError(f"group_weights length ({len(self.group_weights)}) must match number of groups ({len(group_sizes)})")
            
        # Validate group weights
        if any(w < 0 for w in self.group_weights):
            raise ValueError("All group weights must be non-negative")
            
        if sum(self.group_weights) == 0:
            raise ValueError("At least one group weight must be positive")
            
        # Set auxiliary top-k
        self.top_k_aux = max(1, int(self.activation_dim * self.top_k_aux_fraction))
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class MatryoshkaBatchTopKSAE(Dictionary, nn.Module):
    """
    Robust Matryoshka Batch Top-K Sparse Autoencoder.
    
    This SAE uses a hierarchical "matryoshka doll" structure where the dictionary
    is divided into groups of different importance levels. Features are selected
    using batch-level top-k selection rather than per-sample selection.
    
    Key features:
    - Hierarchical group structure for progressive feature selection
    - Batch-level top-k for stability
    - Adaptive threshold updates with EMA
    - Dead feature revival through auxiliary loss
    - Robust numerical stability
    """
    
    def __init__(self, config: MatryoshkaConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        
        # Initialize layers and buffers
        self._init_parameters()
        self._init_buffers()
    
    def _init_parameters(self) -> None:
        """Initialize model parameters with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder and decoder weights
        self.W_enc = nn.Parameter(
            torch.empty(
                self.activation_dim, 
                self.dict_size,
                dtype=dtype,
                device=device
            )
        )
        
        self.b_enc = nn.Parameter(
            torch.zeros(
                self.dict_size,
                dtype=dtype,
                device=device
            )
        )
        
        self.W_dec = nn.Parameter(
            torch.empty(
                self.dict_size, 
                self.activation_dim,
                dtype=dtype,
                device=device
            )
        )
        
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.activation_dim,
                dtype=dtype,
                device=device
            )
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_buffers(self) -> None:
        """Initialize non-trainable buffers."""
        device = self.config.get_device()
        
        self.register_buffer("k", torch.tensor(self.config.k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("group_sizes", torch.tensor(self.config.group_sizes))
        
        # Calculate group indices for efficient slicing
        group_indices = [0] + list(torch.cumsum(torch.tensor(self.config.group_sizes), dim=0))
        self.group_indices = group_indices
        self.active_groups = len(self.config.group_sizes)
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        with torch.no_grad():
            # Initialize decoder with Kaiming uniform
            nn.init.kaiming_uniform_(self.W_dec, a=0.01)
            
            # Normalize decoder weights to unit norm
            self.W_dec.data = set_decoder_norm_to_unit_norm(
                self.W_dec.T, self.activation_dim, self.dict_size
            ).T
            
            # Tie encoder weights to decoder
            self.W_enc.data = self.W_dec.data.clone().T
            
            # Initialize biases to zero
            nn.init.zeros_(self.b_enc)
            nn.init.zeros_(self.b_dec)
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_active: bool = False, 
        use_threshold: bool = True
    ) -> torch.Tensor:
        """
        Encode input to latent space using batch top-k selection.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            return_active: Whether to return additional activation info
            use_threshold: Whether to use threshold-based or top-k based selection
            
        Returns:
            Encoded features, optionally with additional info
        """
        # Ensure input matches model dtype
        x = x.to(dtype=self.W_enc.dtype)
        
        # Compute pre-activation features
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        post_relu_acts = F.relu(pre_acts)
        
        if use_threshold and self.threshold >= 0:
            # Threshold-based selection
            encoded_acts = post_relu_acts * (post_relu_acts > self.threshold)
        else:
            # Batch top-k selection
            flattened_acts = post_relu_acts.flatten()
            k_total = self.k * x.size(0)
            
            if k_total > 0 and len(flattened_acts) > 0:
                k_actual = min(k_total, len(flattened_acts))
                post_topk = flattened_acts.topk(k_actual, sorted=False, dim=-1)
                
                encoded_acts = (
                    torch.zeros_like(flattened_acts)
                    .scatter_(-1, post_topk.indices, post_topk.values)
                    .reshape(post_relu_acts.shape)
                )
            else:
                encoded_acts = torch.zeros_like(post_relu_acts)
        
        # Apply group masking (only keep active groups)
        max_act_index = self.group_indices[self.active_groups]
        if max_act_index < self.dict_size:
            encoded_acts = encoded_acts.clone()
            encoded_acts[:, max_act_index:] = 0
        
        if return_active:
            active_features = encoded_acts.sum(0) > 0
            return encoded_acts, active_features, post_relu_acts
        else:
            return encoded_acts
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.
        
        Args:
            f: Latent features [batch_size, dict_size]
            
        Returns:
            Reconstructed activations [batch_size, activation_dim]
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.W_dec.dtype)
        return f @ self.W_dec + self.b_dec
    
    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False,
        ghost_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for this SAE type
            
        Returns:
            x_hat: Reconstructed activations
            f: Latent features (if output_features=True)
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for MatryoshkaBatchTopKSAE")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode and decode
        f = self.encode(x)
        x_hat = self.decode(f)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        return x_hat
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.b_enc.mul_(scale)
            self.b_dec.mul_(scale)
            if self.threshold >= 0:
                self.threshold.mul_(scale)
    
    def update_threshold(self, f: torch.Tensor) -> None:
        """Update the activation threshold using exponential moving average."""
        device_type = "cuda" if f.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = f[f > 0]
            
            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=torch.float32)
            
            if self.threshold < 0:
                self.threshold.data = min_activation
            else:
                self.threshold.data = (
                    self.config.threshold_beta * self.threshold + 
                    (1 - self.config.threshold_beta) * min_activation
                )
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[MatryoshkaConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'MatryoshkaBatchTopKSAE':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            device: Device to load model to
            **kwargs: Additional arguments for backwards compatibility
            
        Returns:
            Loaded autoencoder
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            try:
                activation_dim, dict_size = state_dict["W_enc"].shape
                k = state_dict["k"].item() if "k" in state_dict else kwargs.get("k", 64)
                group_sizes = state_dict["group_sizes"].tolist() if "group_sizes" in state_dict else [dict_size]
                
                # Calculate group fractions
                group_fractions = [size / dict_size for size in group_sizes]
                
                # Ensure group fractions sum to 1.0
                if not isclose(sum(group_fractions), 1.0, abs_tol=1e-6):
                    # Normalize to sum to 1.0
                    total = sum(group_fractions)
                    group_fractions = [f / total for f in group_fractions]
                
                config = MatryoshkaConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    k=k,
                    group_fractions=group_fractions,
                    device=device,
                    **{k: v for k, v in kwargs.items() if k in MatryoshkaConfig.__dataclass_fields__}
                )
            except Exception as e:
                raise ValueError(f"Could not auto-detect configuration from state dict: {e}")
        
        # Create model
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
        
        # Move to target device
        if device is not None:
            model = model.to(device=device)
        
        return model


@dataclass
class MatryoshkaTrainingConfig:
    """Training configuration for Matryoshka Batch Top-K SAE."""
    steps: int
    lr: Optional[float] = None  # Will be auto-calculated if None
    warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.02 * self.steps))
        
        min_decay_start = self.warmup_steps + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= self.warmup_steps:
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class MatryoshkaBatchTopKTrainer(SAETrainer):
    """
    Robust trainer for Matryoshka Batch Top-K SAE.
    
    Features:
    - Group-based hierarchical training
    - Auxiliary loss for dead feature revival
    - Adaptive threshold updates
    - Proper numerical stability
    - Geometric median initialization
    """
    
    def __init__(
        self,
        model_config: Optional[MatryoshkaConfig] = None,
        training_config: Optional[MatryoshkaTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        group_fractions: Optional[List[float]] = None,
        lr: Optional[float] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use MatryoshkaBatchTopKSAE
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None or k is None or group_fractions is None:
                raise ValueError("Must provide either model_config or all required parameters")
            
            device_obj = torch.device(device) if device else None
            model_config = MatryoshkaConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                group_fractions=group_fractions,
                device=device_obj,
                **{k: v for k, v in kwargs.items() if k in MatryoshkaConfig.__dataclass_fields__}
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = MatryoshkaTrainingConfig(
                steps=steps,
                lr=lr,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "MatryoshkaBatchTopKTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = MatryoshkaBatchTopKSAE(model_config)
        self.ae.to(self.device)
        
        # Auto-calculate learning rate if not provided
        if training_config.lr is None:
            # Auto-select LR using 1 / sqrt(d) scaling law
            scale = model_config.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5
        else:
            self.lr = training_config.lr
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            None
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Initialize tracking variables
        self.num_tokens_since_fired = torch.zeros(
            model_config.dict_size, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
    
    def get_auxiliary_loss(self, residual: torch.Tensor, post_relu_acts: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss to revive dead features.
        
        Args:
            residual: Reconstruction residual
            post_relu_acts: Post-ReLU feature activations
            
        Returns:
            Normalized auxiliary loss
        """
        dead_features = self.num_tokens_since_fired >= self.model_config.dead_feature_threshold
        self.dead_features = int(dead_features.sum())
        
        if self.dead_features > 0:
            k_aux = min(self.model_config.top_k_aux, self.dead_features)
            
            # Create auxiliary latents (only from dead features)
            auxk_latents = torch.where(dead_features[None], post_relu_acts, -torch.inf)
            
            # Top-k dead latents
            if k_aux > 0:
                auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
                
                auxk_buffer = torch.zeros_like(post_relu_acts)
                auxk_acts_normalized = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
                
                # Reconstruct using only auxiliary features (no bias)
                x_reconstruct_aux = auxk_acts_normalized @ self.ae.W_dec
                l2_loss_aux = (
                    (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
                )
                
                self.pre_norm_auxk_loss = l2_loss_aux.item()
                
                # Normalization from OpenAI implementation
                residual_mu = residual.mean(dim=0)[None, :].broadcast_to(residual.shape)
                loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
                
                # Avoid division by zero
                loss_denom = torch.clamp(loss_denom, min=1e-8)
                normalized_auxk_loss = l2_loss_aux / loss_denom
                
                return normalized_auxk_loss.nan_to_num(0.0)
        
        self.pre_norm_auxk_loss = -1
        return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with group-based reconstruction and auxiliary loss."""
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode with detailed outputs
        f, active_indices, post_relu_acts = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        
        # Update threshold if past start step
        if step > self.model_config.threshold_start_step:
            self.ae.update_threshold(f)
        
        # Group-based reconstruction
        x_reconstruct = torch.zeros_like(x) + self.ae.b_dec
        total_l2_loss = 0.0
        l2_losses = []
        
        # Split weights and features by groups
        W_dec_chunks = torch.split(self.ae.W_dec, self.model_config.group_sizes, dim=0)
        f_chunks = torch.split(f, self.model_config.group_sizes, dim=1)
        
        for i in range(self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice
            
            # Compute weighted group loss
            group_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean()
            weighted_loss = group_loss * self.model_config.group_weights[i]
            total_l2_loss += weighted_loss
            l2_losses.append(group_loss)
        
        # Convert losses to tensor for statistics
        l2_losses_tensor = torch.stack(l2_losses) if l2_losses else torch.tensor([0.0])
        
        # Update feature firing tracking
        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        # Compute auxiliary loss
        residual = (x - x_reconstruct).detach()
        auxk_loss = self.get_auxiliary_loss(residual, post_relu_acts)
        
        # Total loss
        loss = total_l2_loss + self.model_config.auxk_alpha * auxk_loss
        
        # Set effective L0 (simplified for batch top-k)
        self.effective_l0 = self.model_config.k
        
        if not logging:
            return loss
        
        # Return detailed loss information
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        return LossLog(
            x, x_reconstruct, f,
            {
                'l2_loss': total_l2_loss.item(),
                'auxk_loss': auxk_loss.item(),
                'loss': loss.item(),
                'min_l2_loss': l2_losses_tensor.min().item(),
                'max_l2_loss': l2_losses_tensor.max().item(),
                'mean_l2_loss': l2_losses_tensor.mean().item(),
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with proper geometric median initialization."""
        # Initialize decoder bias with geometric median on first step
        if step == 0:
            median = self.geometric_median(activations)
            self.ae.b_dec.data = median.to(dtype=self.ae.b_dec.dtype)
        
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Remove gradients parallel to decoder directions
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, 
            self.ae.W_dec.grad.T, 
            self.ae.activation_dim, 
            self.ae.dict_size
        ).T
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(),
            self.training_config.gradient_clip_norm
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Normalize decoder weights
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, 
            self.ae.activation_dim, 
            self.ae.dict_size
        ).T
    
    @staticmethod
    def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
        """
        Compute geometric median of points using Weiszfeld's algorithm.
        
        Args:
            points: Input points [n_points, dim]
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Geometric median
        """
        guess = points.mean(dim=0)
        prev = torch.zeros_like(guess)
        
        for _ in range(max_iter):
            prev = guess.clone()
            
            # Compute distances and weights
            distances = torch.norm(points - guess, dim=1)
            weights = 1 / torch.clamp(distances, min=1e-8)
            weights /= weights.sum()
            
            # Update guess
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            
            # Check convergence
            if torch.norm(guess - prev) < tol:
                break
        
        return guess
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'MatryoshkaBatchTopKSAE',
            'trainer_class': 'MatryoshkaBatchTopKTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'group_fractions': self.model_config.group_fractions,
            'group_sizes': self.model_config.group_sizes,
            'group_weights': self.model_config.group_weights,
            'auxk_alpha': self.model_config.auxk_alpha,
            'threshold_beta': self.model_config.threshold_beta,
            'threshold_start_step': self.model_config.threshold_start_step,
            'dead_feature_threshold': self.model_config.dead_feature_threshold,
            'top_k_aux_fraction': self.model_config.top_k_aux_fraction,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.lr,
            'warmup_steps': self.training_config.warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }