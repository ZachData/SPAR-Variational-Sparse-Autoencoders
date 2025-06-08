"""
Enhanced BatchTopK SAE implementation with robustness improvements.

Key improvements:
- Proper configuration management with dataclasses
- Enhanced error handling and validation
- Better dtype and device handling
- Comprehensive diagnostics and logging
- Improved from_pretrained method
- Better documentation and type hints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass
import warnings

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


@dataclass
class BatchTopKConfig:
    """Configuration for BatchTopK SAE model."""
    activation_dim: int
    dict_size: int
    k: int  # Number of top-k features to keep
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k must be a positive integer, got {self.k}")
        if self.k > self.dict_size:
            raise ValueError(f"k ({self.k}) cannot be larger than dict_size ({self.dict_size})")
        if self.activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {self.activation_dim}")
        if self.dict_size <= 0:
            raise ValueError(f"dict_size must be positive, got {self.dict_size}")
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class BatchTopKSAE(Dictionary, nn.Module):
    """
    Enhanced BatchTopK Sparse Autoencoder.
    
    This implementation uses batch-level top-k selection rather than per-sample
    top-k, which can be more efficient and provides different sparsity patterns.
    
    Key features:
    - Adaptive threshold mechanism
    - Auxiliary loss for dead feature recovery
    - Unit norm decoder constraints
    - Geometric median bias initialization
    """

    def __init__(self, config: BatchTopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.k = config.k
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
        
        # Initialize threshold and k as buffers (will be saved/loaded with model)
        self.register_buffer("k_buffer", torch.tensor(config.k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
    
    def _init_layers(self) -> None:
        """Initialize neural network layers with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Decoder with unit norm constraint (no bias)
        self.decoder = nn.Linear(
            self.dict_size,
            self.activation_dim,
            bias=False,
            dtype=dtype,
            device=device
        )
        
        # Encoder
        self.encoder = nn.Linear(
            self.activation_dim,
            self.dict_size,
            bias=True,
            dtype=dtype,
            device=device
        )
        
        # Decoder bias parameter
        if self.use_april_update_mode:
            # In April update mode, use a separate bias parameter
            self.b_dec = nn.Parameter(
                torch.zeros(self.activation_dim, dtype=dtype, device=device)
            )
        else:
            # Legacy mode - decoder bias integrated differently
            self.bias = nn.Parameter(
                torch.zeros(self.activation_dim, dtype=dtype, device=device)
            )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        with torch.no_grad():
            # Initialize decoder with unit norm
            self.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.decoder.weight, self.activation_dim, self.dict_size
            )
            
            # Initialize encoder weights as transpose of decoder
            self.encoder.weight.data = self.decoder.weight.T.clone()
            
            # Initialize encoder bias to zero
            nn.init.zeros_(self.encoder.bias)
            
            # Initialize decoder bias to zero (will be set via geometric median later)
            if self.use_april_update_mode:
                nn.init.zeros_(self.b_dec)
            else:
                nn.init.zeros_(self.bias)
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_active: bool = False, 
        use_threshold: bool = True
    ) -> torch.Tensor:
        """
        Encode input to latent space using BatchTopK selection.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            return_active: Whether to return additional activation info
            use_threshold: Whether to use adaptive threshold (vs pure top-k)
            
        Returns:
            Encoded features (and optionally additional info)
        """
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Subtract decoder bias
        if self.use_april_update_mode:
            x_centered = x - self.b_dec
        else:
            x_centered = x - self.bias
        
        # Apply encoder and ReLU
        pre_relu_acts = self.encoder(x_centered)
        post_relu_acts = F.relu(pre_relu_acts)
        
        if use_threshold and self.threshold >= 0:
            # Use adaptive threshold
            encoded_acts = post_relu_acts * (post_relu_acts > self.threshold)
        else:
            # Use batch-level top-k selection
            flattened_acts = post_relu_acts.flatten()
            batch_size = x.size(0)
            total_k = self.k * batch_size
            
            # Get top-k across entire batch
            if total_k < flattened_acts.numel():
                topk_values, topk_indices = flattened_acts.topk(
                    total_k, sorted=False, dim=-1
                )
                
                # Create sparse tensor
                encoded_acts = torch.zeros_like(flattened_acts)
                encoded_acts.scatter_(-1, topk_indices, topk_values)
                encoded_acts = encoded_acts.reshape(post_relu_acts.shape)
            else:
                # If k is too large, just use all positive activations
                encoded_acts = post_relu_acts
        
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
        f = f.to(dtype=self.decoder.weight.dtype)
        
        # Apply decoder
        reconstruction = self.decoder(f)
        
        # Add decoder bias
        if self.use_april_update_mode:
            reconstruction = reconstruction + self.b_dec
        else:
            reconstruction = reconstruction + self.bias
        
        return reconstruction
    
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
            ghost_mask: Not implemented for BatchTopK (raises error if provided)
            
        Returns:
            x_hat: Reconstructed activations
            f: Latent features (if output_features=True)
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for BatchTopKSAE")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode
        f = self.encode(x, return_active=False, use_threshold=True)
        
        # Decode
        x_hat = self.decode(f)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        return x_hat
    
    def get_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed diagnostics for monitoring training.
        
        Returns:
            Dictionary with various metrics and statistics
        """
        with torch.no_grad():
            f, active_features, post_relu_acts = self.encode(x, return_active=True)
            x_hat = self.decode(f)
            
            # Basic metrics
            l0 = (f != 0).float().sum(dim=-1).mean()
            l2_loss = torch.norm(x - x_hat, dim=-1).mean()
            
            # Sparsity metrics
            frac_active = active_features.float().mean()
            effective_l0 = f.norm(p=0, dim=-1).mean()
            
            # Activation statistics
            mean_activation = post_relu_acts[post_relu_acts > 0].mean() if (post_relu_acts > 0).any() else torch.tensor(0.0)
            max_activation = post_relu_acts.max()
            
            return {
                'l0': l0,
                'l2_loss': l2_loss,
                'frac_active_features': frac_active,
                'effective_l0': effective_l0,
                'mean_positive_activation': mean_activation,
                'max_activation': max_activation,
                'threshold': self.threshold,
                'num_active_features': active_features.sum(),
            }
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            
            if self.use_april_update_mode:
                self.b_dec.mul_(scale)
            else:
                self.bias.mul_(scale)
            
            # Scale threshold if it's been initialized
            if self.threshold >= 0:
                self.threshold.mul_(scale)
    
    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
        with torch.no_grad():
            print("Normalizing decoder weights")
            
            # Test that normalization preserves output
            device = self.decoder.weight.device
            test_input = torch.randn(
                10, self.activation_dim, 
                device=device, 
                dtype=self.decoder.weight.dtype
            )
            initial_output = self(test_input)
            
            # Normalize decoder weights
            self.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.decoder.weight, self.activation_dim, self.dict_size
            )
            
            # Verify output is preserved (approximately)
            new_output = self(test_input)
            if not torch.allclose(initial_output, new_output, atol=1e-3):
                warnings.warn("Decoder normalization significantly changed model output")
    
    @staticmethod
    def geometric_median(
        points: torch.Tensor, 
        max_iter: int = 100, 
        tol: float = 1e-5
    ) -> torch.Tensor:
        """
        Compute geometric median of a set of points.
        More robust than arithmetic mean for initialization.
        
        Args:
            points: Tensor of shape [n_points, dim]
            max_iter: Maximum iterations for convergence
            tol: Tolerance for convergence
            
        Returns:
            Geometric median point
        """
        if points.numel() == 0:
            raise ValueError("Cannot compute geometric median of empty tensor")
        
        # Initialize with arithmetic mean
        guess = points.mean(dim=0)
        prev = torch.zeros_like(guess)
        
        for _ in range(max_iter):
            prev = guess.clone()
            
            # Compute distances (with small epsilon for numerical stability)
            distances = torch.norm(points - guess, dim=1) + 1e-8
            weights = 1.0 / distances
            weights = weights / weights.sum()
            
            # Update guess
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            
            # Check convergence
            if torch.norm(guess - prev) < tol:
                break
        
        return guess
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[BatchTopKConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        k: Optional[int] = None
    ) -> 'BatchTopKSAE':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            k: Override k value detection
            
        Returns:
            Loaded autoencoder
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            
            # Get k value
            if k is not None:
                k_value = k
            elif "k" in state_dict:
                k_value = state_dict["k"].item()
            elif "k_buffer" in state_dict:
                k_value = state_dict["k_buffer"].item()
            else:
                raise ValueError("Could not determine k value. Please provide k parameter.")
            
            # Detect April update mode
            use_april_update_mode = "b_dec" in state_dict
            
            config = BatchTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k_value,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        # Validate k if provided
        if k is not None and "k_buffer" in state_dict:
            if k != state_dict["k_buffer"].item():
                raise ValueError(f"Provided k={k} doesn't match saved k={state_dict['k_buffer'].item()}")
        
        # Create model
        model = cls(config)
        
        # Handle legacy parameter names if needed
        converted_state_dict = cls._convert_legacy_state_dict(state_dict, config)
        
        # Load state dict with error handling
        try:
            missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Normalize decoder if requested
        if normalize_decoder:
            try:
                model.normalize_decoder()
            except Exception as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model
    
    @staticmethod
    def _convert_legacy_state_dict(
        state_dict: Dict[str, torch.Tensor], 
        config: BatchTopKConfig
    ) -> Dict[str, torch.Tensor]:
        """Convert legacy parameter names to current format if needed."""
        converted = {}
        
        # Handle different bias naming conventions
        for key, value in state_dict.items():
            if key == "k" and "k_buffer" not in state_dict:
                # Convert old k to k_buffer
                converted["k_buffer"] = value
            else:
                converted[key] = value
        
        return converted

@dataclass
class BatchTopKTrainingConfig:
    """Enhanced training configuration for BatchTopK."""
    steps: int
    lr: Optional[float] = None  # Will auto-compute if None
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient
    warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    threshold_beta: float = 0.999  # EMA coefficient for threshold updates
    threshold_start_step: int = 1000  # When to start threshold updates
    gradient_clip_norm: float = 1.0
    dead_feature_threshold: int = 10_000_000  # Steps before feature considered dead

    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            # FIXED: Ensure warmup_steps is always < steps
            warmup_candidate = max(200, int(0.02 * self.steps))
            self.warmup_steps = min(warmup_candidate, self.steps - 1)
        
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)
            if self.decay_start <= self.warmup_steps:
                self.decay_start = None  # Disable decay

class BatchTopKTrainer(SAETrainer):
    """
    Enhanced trainer for BatchTopK SAE with improved robustness.
    
    Key improvements:
    - Better configuration management
    - Enhanced diagnostics and logging
    - Improved error handling
    - Automatic learning rate scaling
    """
    
    def __init__(
        self,
        model_config: Optional[BatchTopKConfig] = None,
        training_config: Optional[BatchTopKTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility parameters
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        lr: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None or k is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
            
            device_obj = torch.device(device) if device else None
            model_config = BatchTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = BatchTopKTrainingConfig(
                steps=steps,
                lr=lr,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "BatchTopKTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = BatchTopKSAE(model_config)
        self.ae.to(self.device)
        
        # Set learning rate with automatic scaling if not provided
        if training_config.lr is None:
            # Auto-select LR using 1 / sqrt(d) scaling law from the paper
            scale = model_config.dict_size / (2**14)
            self.lr = 2e-4 / (scale**0.5)
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
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Initialize tracking variables
        self.num_tokens_since_fired = torch.zeros(
            model_config.dict_size, dtype=torch.long, device=self.device
        )
        self.top_k_aux = model_config.activation_dim // 2  # Heuristic from paper
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
    
    def get_auxiliary_loss(
        self, 
        residual: torch.Tensor, 
        post_relu_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for dead feature recovery.
        
        Args:
            residual: Reconstruction error
            post_relu_acts: Post-ReLU activations before top-k
            
        Returns:
            Normalized auxiliary loss
        """
        # Identify dead features
        dead_features = self.num_tokens_since_fired >= self.training_config.dead_feature_threshold
        self.dead_features = int(dead_features.sum())
        
        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())
            
            # Get activations for dead features only
            auxk_latents = torch.where(
                dead_features[None], 
                post_relu_acts, 
                -torch.inf
            )
            
            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            
            # Create sparse activation tensor
            auxk_buffer = torch.zeros_like(post_relu_acts)
            auxk_acts_sparse = auxk_buffer.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )
            
            # Compute auxiliary reconstruction (decoder only, no bias)
            x_reconstruct_aux = self.ae.decoder(auxk_acts_sparse)
            l2_loss_aux = (
                (residual.float() - x_reconstruct_aux.float())
                .pow(2).sum(dim=-1).mean()
            )
            
            self.pre_norm_auxk_loss = l2_loss_aux.item()
            
            # Normalization from OpenAI implementation
            residual_mu = residual.mean(dim=0)[None, :].broadcast_to(residual.shape)
            loss_denom = (
                (residual.float() - residual_mu.float())
                .pow(2).sum(dim=-1).mean()
            )
            
            if loss_denom > 1e-8:
                normalized_auxk_loss = l2_loss_aux / loss_denom
            else:
                normalized_auxk_loss = torch.tensor(0.0, device=residual.device)
            
            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = 0.0
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
    
    def update_threshold(self, f: torch.Tensor) -> None:
        """
        Update adaptive threshold using exponential moving average.
        
        Args:
            f: Feature activations
        """
        device_type = "cuda" if f.is_cuda else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = f[f > 0]
            
            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=torch.float32)
            
            if self.ae.threshold < 0:
                # First time initialization
                self.ae.threshold.data = torch.tensor(min_activation)
            else:
                # EMA update
                beta = self.training_config.threshold_beta
                self.ae.threshold.data = (
                    beta * self.ae.threshold + 
                    (1 - beta) * min_activation
                )
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with comprehensive logging."""
        # Encode with all information
        f, active_indices, post_relu_acts = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        
        # Update threshold if past start step
        if step > self.training_config.threshold_start_step:
            self.update_threshold(f)
        
        # Decode
        x_hat = self.ae.decode(f)
        
        # Compute reconstruction error
        residual = x - x_hat
        
        # Track effective L0
        self.effective_l0 = self.model_config.k
        
        # Update dead feature tracking
        batch_size = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += batch_size
        self.num_tokens_since_fired[did_fire] = 0
        
        # Compute losses
        l2_loss = residual.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(residual.detach(), post_relu_acts)
        total_loss = l2_loss + self.training_config.auxk_alpha * auxk_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        diagnostics = self.ae.get_diagnostics(x)
        
        return LossLog(
            x, x_hat, f,
            {
                'l2_loss': l2_loss.item(),
                'auxk_loss': auxk_loss.item(),
                'loss': total_loss.item(),
                # Additional diagnostics
                **{k: v.item() if torch.is_tensor(v) else v for k, v in diagnostics.items()}
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with geometric median initialization."""
        activations = activations.to(self.device)
        
        # Initialize decoder bias with geometric median on first step
        if step == 0:
            median = self.ae.geometric_median(activations)
            median = median.to(dtype=self.ae.b_dec.dtype)
            if self.ae.use_april_update_mode:
                self.ae.b_dec.data = median
            else:
                self.ae.bias.data = median
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Remove gradients parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(),
            self.training_config.gradient_clip_norm
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Ensure decoder weights remain unit norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, 
            self.ae.activation_dim, 
            self.ae.dict_size
        )
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'BatchTopKSAE',
            'trainer_class': 'BatchTopKTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.lr,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'decay_start': self.training_config.decay_start,
            'threshold_beta': self.training_config.threshold_beta,
            'threshold_start_step': self.training_config.threshold_start_step,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            'top_k_aux': self.top_k_aux,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }
