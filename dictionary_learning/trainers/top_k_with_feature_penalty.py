"""
Robust Top-K SAE implementation based on https://arxiv.org/abs/2406.04093
Enhanced with better configuration management and error handling.
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..config import DEBUG
from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5) -> t.Tensor:
    """
    Compute the geometric median of points. Used for robust decoder bias initialization.
    More robust than simple mean, especially with outliers.
    
    Args:
        points: Input points [batch_size, feature_dim]
        max_iter: Maximum iterations for convergence
        tol: Tolerance for convergence
        
    Returns:
        Geometric median of the points
    """
    if len(points) == 0:
        raise ValueError("Cannot compute geometric median of empty tensor")
    
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device, dtype=points.dtype)

    for iteration in range(max_iter):
        prev = guess.clone()

        # Compute distances from guess to all points
        distances = t.norm(points - guess, dim=1)
        
        # Handle the case where a point coincides with the current guess
        # (distance would be 0, leading to infinite weight)
        distances = t.clamp(distances, min=1e-8)
        
        # Compute the weights (inverse of distances)
        weights = 1.0 / distances

        # Normalize the weights
        weights = weights / weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


@dataclass
class TopKConfig:
    """Configuration for Top-K SAE model."""
    activation_dim: int
    dict_size: int
    k: int
    dtype: t.dtype = t.bfloat16
    device: Optional[t.device] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.k >= self.dict_size:
            raise ValueError(f"k ({self.k}) must be less than dict_size ({self.dict_size})")
        if self.activation_dim <= 0:
            raise ValueError(f"activation_dim must be positive, got {self.activation_dim}")
        if self.dict_size <= 0:
            raise ValueError(f"dict_size must be positive, got {self.dict_size}")
    
    def get_device(self) -> t.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return t.device("cuda" if t.cuda.is_available() else "cpu")
        return self.device


@dataclass 
class TopKTrainingConfig:
    """Configuration for Top-K SAE training."""
    steps: int
    lr: Optional[float] = None  # Auto-computed if None
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient
    warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    threshold_beta: float = 0.999  # EMA coefficient for threshold updates
    threshold_start_step: int = 1000  # When to start threshold updates
    gradient_clip_norm: float = 1.0
    dead_feature_threshold: int = 1_000  # Steps before considering feature dead  #changed! was 10k
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
            
        # Set decay start conservatively
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class AutoEncoderTopK(Dictionary, nn.Module):
    """
    Robust Top-K autoencoder with enhanced error handling and configuration management.
    Based on the architecture from https://arxiv.org/abs/2406.04093
    """

    def __init__(self, config: TopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        
        device = config.get_device()
        dtype = config.dtype
        
        # Register k as a buffer (not a parameter)
        self.register_buffer("k", t.tensor(config.k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        # Initialize decoder with unit norm constraint
        self.decoder = nn.Linear(config.dict_size, config.activation_dim, bias=False, dtype=dtype, device=device)
        with t.no_grad():
            self.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.decoder.weight, config.activation_dim, config.dict_size
            )

        # Initialize encoder (tied to decoder)
        self.encoder = nn.Linear(config.activation_dim, config.dict_size, dtype=dtype, device=device)
        with t.no_grad():
            self.encoder.weight.data = self.decoder.weight.T.clone()
            self.encoder.bias.data.zero_()

        # Decoder bias (to be initialized with geometric median)
        self.b_dec = nn.Parameter(t.zeros(config.activation_dim, dtype=dtype, device=device))

    def encode(self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False) -> t.Tensor:
        """
        Encode input using Top-K sparsification.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            return_topk: Whether to return detailed top-k information
            use_threshold: Whether to use threshold-based sparsification
            
        Returns:
            Encoded sparse activations, optionally with top-k details
        """
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Apply encoder with bias subtraction
        post_relu_feat_acts_BF = t.relu(self.encoder(x - self.b_dec))

        if use_threshold and self.threshold >= 0:
            # Threshold-based sparsification
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return encoded_acts_BF, post_topk.values, post_topk.indices, post_relu_feat_acts_BF
            else:
                return encoded_acts_BF

        # Top-k sparsification
        try:
            post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
        except RuntimeError as e:
            if "k not in range" in str(e):
                raise ValueError(f"k={self.k} is too large for feature dimension {post_relu_feat_acts_BF.shape[-1]}")
            raise

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        # Create sparse representation
        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        """Decode sparse features to reconstruction."""
        x = x.to(dtype=self.decoder.weight.dtype)
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, ghost_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations
            output_features: Whether to return features
            ghost_mask: Not implemented for Top-K SAE
            
        Returns:
            Reconstructed activations, optionally with features
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for Top-K SAE")
            
        original_dtype = x.dtype
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        x_hat_BD = x_hat_BD.to(dtype=original_dtype)
        
        if output_features:
            encoded_acts_BF = encoded_acts_BF.to(dtype=original_dtype)
            return x_hat_BD, encoded_acts_BF
        else:
            return x_hat_BD

    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with t.no_grad():
            self.encoder.bias.mul_(scale)
            self.b_dec.mul_(scale)
            if self.threshold >= 0:
                self.threshold.mul_(scale)

    def initialize_decoder_bias(self, data_batch: t.Tensor) -> None:
        """Initialize decoder bias using geometric median of data."""
        with t.no_grad():
            if len(data_batch.shape) == 3:
                # Flatten batch and sequence dimensions
                data_batch = data_batch.reshape(-1, data_batch.shape[-1])
            
            try:
                median = geometric_median(data_batch)
                median = median.to(dtype=self.b_dec.dtype, device=self.b_dec.device)
                self.b_dec.data.copy_(median)
            except Exception as e:
                print(f"Warning: Could not compute geometric median, using mean: {e}")
                mean = data_batch.mean(dim=0)
                mean = mean.to(dtype=self.b_dec.dtype, device=self.b_dec.device)
                self.b_dec.data.copy_(mean)

    def get_sparsity_diagnostics(self, x: t.Tensor) -> Dict[str, t.Tensor]:
        """Get sparsity diagnostics for monitoring."""
        with t.no_grad():
            encoded_acts, top_acts, top_indices, pre_relu = self.encode(x, return_topk=True)
            
            return {
                'l0': (encoded_acts != 0).float().sum(dim=-1).mean(),
                'mean_activation': encoded_acts[encoded_acts != 0].mean() if (encoded_acts != 0).any() else t.tensor(0.0),
                'min_top_activation': top_acts.min(dim=-1).values.mean(),
                'max_top_activation': top_acts.max(dim=-1).values.mean(),
                'threshold': self.threshold,
            }

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[TopKConfig] = None,
        dtype: t.dtype = t.float32,
        device: Optional[t.device] = None,
        k: Optional[int] = None
    ) -> 'AutoEncoderTopK':
        """
        Load a pretrained Top-K autoencoder.
        
        Args:
            path: Path to saved model
            config: Model configuration (auto-detected if None)
            dtype: Target dtype
            device: Target device
            k: Override k value
            
        Returns:
            Loaded model
        """
        checkpoint = t.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            
            # Get k from state dict or parameter
            if k is None:
                if "k" in state_dict:
                    k = state_dict["k"].item()
                else:
                    raise ValueError("k not found in state_dict and not provided")
            
            config = TopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                dtype=dtype,
                device=device
            )
        
        # Create model and load state
        model = cls(config)
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Move to target device and dtype
        model = model.to(device=device, dtype=dtype)
        
        return model


class TopKTrainer(SAETrainer):
    """
    Enhanced Top-K SAE trainer with robust configuration management.
    """

    def __init__(
        self,
        model_config: Optional[TopKConfig] = None,
        training_config: Optional[TopKTrainingConfig] = None,
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
        dict_class=None,  # Ignored, always use AutoEncoderTopK
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None or k is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
            
            device_obj = t.device(device) if device else None
            model_config = TopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = TopKTrainingConfig(
                steps=steps,
                lr=lr,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "TopKTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = AutoEncoderTopK(model_config)
        self.ae.to(self.device)
        
        # Auto-compute learning rate if not provided
        if training_config.lr is None:
            # Use scaling law from the paper: LR ~ 1/sqrt(dict_size)
            scale = model_config.dict_size / (2**14)
            self.lr = 2e-4 / (scale**0.5)
        else:
            self.lr = training_config.lr
        
        # Initialize optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Sparsity warmup (though Top-K doesn't need traditional sparsity penalties)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps,
            training_config.sparsity_warmup_steps
        )
        
        # Initialize tracking variables
        self.top_k_aux = model_config.activation_dim // 2  # Heuristic from paper
        self.num_tokens_since_fired = t.zeros(
            model_config.dict_size, 
            dtype=t.long, 
            device=self.device
        )
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        
        # Initialize decoder bias flag
        self._decoder_bias_initialized = False

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor) -> t.Tensor:
        """
        Compute auxiliary loss to resurrect dead features.
        
        Args:
            residual_BD: Reconstruction error
            post_relu_acts_BF: Pre-TopK activations
            
        Returns:
            Auxiliary loss for dead feature resurrection
        """
        dead_features = self.num_tokens_since_fired >= self.training_config.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # Select only dead features for auxiliary loss
            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False, dim=-1)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Reconstruction using only auxiliary features
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()

            self.pre_norm_auxk_loss = l2_loss_aux.item()

            # Normalization from OpenAI implementation
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            
            if loss_denom > 1e-8:
                normalized_auxk_loss = l2_loss_aux / loss_denom
            else:
                normalized_auxk_loss = t.tensor(0.0, device=residual_BD.device)

            return normalized_auxk_loss.clamp(min=0.0)
        else:
            self.pre_norm_auxk_loss = 0.0
            return t.tensor(0.0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor) -> None:
        """Update the activation threshold using exponential moving average."""
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            # Find minimum activation in each sample's top-k
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            if self.ae.threshold < 0:
                # First update
                self.ae.threshold.data = min_activation
            else:
                # EMA update
                beta = self.training_config.threshold_beta
                self.ae.threshold.data = (beta * self.ae.threshold + (1 - beta) * min_activation)

    def loss(self, x: t.Tensor, step: int, logging: bool = False):
        """Compute Top-K SAE loss with auxiliary dead feature resurrection."""
        # Ensure proper dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)
        
        # Run the SAE with detailed outputs
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        # Update threshold after warmup period
        if step > self.training_config.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)
        
        # Ensure output matches input dtype
        x_hat = x_hat.to(dtype=x.dtype)

        # Compute reconstruction error
        e = x - x_hat

        # Update effective L0 (should be k)
        self.effective_l0 = top_acts_BK.size(1)

        # Update feature firing statistics
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        # Compute losses
        l2_loss = e.pow(2).sum(dim=-1).mean()
        
        auxk_loss = t.tensor(0.0, device=x.device)
        if self.training_config.auxk_alpha > 0:
            auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        
        activation_cost = 0.01 * post_relu_acts_BF.pow(2).sum(dim=-1).mean() #added loss term

        total_loss = l2_loss + self.training_config.auxk_alpha * auxk_loss + activation_cost

        if not logging:
            return total_loss
        else:
            LossLog = namedtuple("LossLog", ["x", "x_hat", "f", "losses"])
            return LossLog(
                x, x_hat, f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item() if t.is_tensor(auxk_loss) else auxk_loss,
                    "loss": total_loss.item(),
                    "effective_l0": self.effective_l0,
                    "dead_features": self.dead_features,
                }
            )

    def update(self, step: int, activations: t.Tensor) -> None:
        """Perform one training step."""
        activations = activations.to(self.device)
        
        # Initialize decoder bias on first step
        if step == 0 and not self._decoder_bias_initialized:
            self.ae.initialize_decoder_bias(activations)
            self._decoder_bias_initialized = True

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()

        # Gradient processing specific to Top-K SAE
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        
        # Gradient clipping
        t.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        # Maintain unit norm constraint on decoder
        with t.no_grad():
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, 
                self.ae.activation_dim, 
                self.ae.dict_size
            )

    def get_logging_parameters(self) -> Dict[str, Any]:
        """Get additional logging parameters."""
        base_params = super().get_logging_parameters()
        base_params.update({
            "threshold": self.ae.threshold.item() if self.ae.threshold >= 0 else -1,
            "lr_current": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.lr,
        })
        return base_params

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'AutoEncoderTopK',
            'trainer_class': 'TopKTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.lr,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'decay_start': self.training_config.decay_start,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'threshold_beta': self.training_config.threshold_beta,
            'threshold_start_step': self.training_config.threshold_start_step,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }
