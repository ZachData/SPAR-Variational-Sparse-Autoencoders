"""
Robust JumpReLU SAE implementation following vsae_iso.py patterns.

Key improvements:
- Separate model and trainer classes
- Proper configuration dataclasses
- Better device/dtype handling
- Comprehensive from_pretrained support
- Numerical stability improvements
- Enhanced error handling and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


# Custom autograd functions (enhanced for stability)
class RectangleFunction(autograd.Function):
    """Differentiable rectangle function for jump ReLU gradients."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    """Differentiable Jump ReLU activation function."""
    
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        
        # Clamp bandwidth to avoid division by zero
        bandwidth = max(bandwidth, 1e-8)
        
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class StepFunction(autograd.Function):
    """Differentiable step function for L0 computation."""
    
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        
        # Clamp bandwidth to avoid division by zero
        bandwidth = max(bandwidth, 1e-8)
        
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) 
            * RectangleFunction.apply((x - threshold) / bandwidth) 
            * grad_output
        )
        return x_grad, threshold_grad, None


@dataclass
class JumpReluConfig:
    """Configuration for JumpReLU SAE model."""
    activation_dim: int
    dict_size: int
    bandwidth: float = 0.001
    threshold_init: float = 0.001  # Initial threshold value
    apply_b_dec_to_input: bool = False  # SAE-lens compatibility
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class JumpReluSAE(Dictionary, nn.Module):
    """
    Jump ReLU Sparse Autoencoder implementation.
    
    Based on the paper: "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
    Uses jump ReLU activations with learnable thresholds for sparsity.
    """

    def __init__(self, config: JumpReluConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.bandwidth = config.bandwidth
        self.apply_b_dec_to_input = config.apply_b_dec_to_input
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Encoder and decoder weights (as parameters for easier gradient manipulation)
        self.W_enc = nn.Parameter(
            torch.empty(self.activation_dim, self.dict_size, device=device, dtype=dtype)
        )
        self.W_dec = nn.Parameter(
            torch.empty(self.dict_size, self.activation_dim, device=device, dtype=dtype)
        )
        
        # Biases
        self.b_enc = nn.Parameter(
            torch.zeros(self.dict_size, device=device, dtype=dtype)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.activation_dim, device=device, dtype=dtype)
        )
        
        # Learnable thresholds for jump ReLU
        self.threshold = nn.Parameter(
            torch.ones(self.dict_size, device=device, dtype=dtype) * self.config.threshold_init
        )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        with torch.no_grad():
            # Initialize decoder weights with Kaiming uniform and normalize
            nn.init.kaiming_uniform_(self.W_dec, a=0.01)
            self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
            
            # Tie encoder weights to decoder (transposed)
            self.W_enc.data = self.W_dec.data.T.clone()
            
            # Initialize biases to zero
            nn.init.zeros_(self.b_enc)
            nn.init.zeros_(self.b_dec)
            
            # Threshold already initialized in _init_layers
    
    def encode(self, x: torch.Tensor, output_pre_jump: bool = False) -> torch.Tensor:
        """
        Encode input using jump ReLU activation.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_pre_jump: Whether to return pre-activation values
            
        Returns:
            f: Sparse features [batch_size, dict_size]
            pre_jump: Pre-activation values (if output_pre_jump=True)
        """
        # Ensure input matches model dtype
        x = x.to(dtype=self.W_enc.dtype)
        
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        
        # Linear transformation
        pre_jump = x @ self.W_enc + self.b_enc
        
        # Apply jump ReLU with clamped threshold
        threshold_clamped = torch.clamp(self.threshold, min=1e-6)
        f = JumpReLUFunction.apply(pre_jump, threshold_clamped, self.bandwidth)
        
        if output_pre_jump:
            return f, pre_jump
        return f
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to reconstruction.
        
        Args:
            f: Sparse features [batch_size, dict_size]
            
        Returns:
            x_hat: Reconstructed activations [batch_size, activation_dim]
        """
        f = f.to(dtype=self.W_dec.dtype)
        return f @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor, output_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return sparse features
            
        Returns:
            x_hat: Reconstructed activations
            f: Sparse features (if output_features=True)
        """
        original_dtype = x.dtype
        
        # Encode
        f = self.encode(x)
        
        # Decode
        x_hat = self.decode(f)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        return x_hat
    
    def get_l0(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L0 sparsity (number of active features)."""
        with torch.no_grad():
            pre_jump = x @ self.W_enc + self.b_enc
            threshold_clamped = torch.clamp(self.threshold, min=1e-6)
            l0_per_sample = StepFunction.apply(pre_jump, threshold_clamped, self.bandwidth).sum(dim=-1)
            return l0_per_sample.mean()
    
    def get_feature_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed feature diagnostics for monitoring."""
        with torch.no_grad():
            f, pre_jump = self.encode(x, output_pre_jump=True)
            
            # Feature activation statistics
            feature_activations = f.sum(dim=0)  # Total activation per feature
            active_features = (feature_activations > 0).sum()
            
            # Threshold statistics
            mean_threshold = self.threshold.mean()
            threshold_std = self.threshold.std()
            
            # Pre-jump statistics
            mean_pre_jump = pre_jump.mean()
            pre_jump_std = pre_jump.std()
            
            return {
                'active_features': active_features,
                'frac_active_features': active_features / self.dict_size,
                'mean_threshold': mean_threshold,
                'threshold_std': threshold_std,
                'mean_pre_jump': mean_pre_jump,
                'pre_jump_std': pre_jump_std,
                'mean_feature_activation': feature_activations.mean(),
                'feature_activation_std': feature_activations.std(),
            }
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.b_enc.mul_(scale)
            self.b_dec.mul_(scale)
            self.threshold.mul_(scale)
    
    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
        with torch.no_grad():
            norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            
            if torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                return
            
            print("Normalizing decoder weights")
            
            # Test that normalization preserves output
            device = self.W_dec.device
            test_input = torch.randn(10, self.activation_dim, device=device, dtype=self.W_dec.dtype)
            initial_output = self(test_input)
            
            # Normalize decoder weights
            self.W_dec.div_(norms)
            
            # Scale encoder weights and biases accordingly
            self.W_enc.mul_(norms.T)
            self.b_enc.mul_(norms.squeeze())
            
            # Verify normalization worked
            new_norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            assert torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-6)
            
            # Verify output is preserved
            new_output = self(test_input)
            assert torch.allclose(initial_output, new_output, atol=1e-4)
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[JumpReluConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        **kwargs
    ) -> 'JumpReluSAE':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            
        Returns:
            Loaded autoencoder
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if 'W_enc' in state_dict:
                activation_dim, dict_size = state_dict["W_enc"].shape
            else:
                raise ValueError("Could not determine model dimensions from state dict")
            
            # Try to detect bandwidth and other parameters
            bandwidth = kwargs.get('bandwidth', 0.001)
            apply_b_dec_to_input = kwargs.get('apply_b_dec_to_input', False)
            
            config = JumpReluConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                bandwidth=bandwidth,
                apply_b_dec_to_input=apply_b_dec_to_input,
                dtype=dtype,
                device=device
            )
        
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


@dataclass
class JumpReluTrainingConfig:
    """Training configuration for JumpReLU SAE."""
    steps: int
    lr: float = 7e-5
    bandwidth: float = 0.001
    sparsity_penalty: float = 1.0
    target_l0: float = 20.0
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    dead_feature_threshold: int = 10_000_000
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = max(2000, int(0.1 * self.steps))
        
        # Set decay start conservatively
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if self.decay_start is None or self.decay_start < min_decay_start:
            if default_decay_start > min_decay_start:
                self.decay_start = default_decay_start
            else:
                self.decay_start = None  # Disable decay


class JumpReluTrainer(SAETrainer):
    """
    Robust trainer for JumpReLU SAE with proper separation of concerns.
    """
    
    def __init__(
        self,
        model_config: Optional[JumpReluConfig] = None,
        training_config: Optional[JumpReluTrainingConfig] = None,
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
        bandwidth: Optional[float] = None,
        sparsity_penalty: Optional[float] = None,
        target_l0: Optional[float] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use JumpReluSAE
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = JumpReluConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                bandwidth=bandwidth or 0.001,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = JumpReluTrainingConfig(
                steps=steps,
                lr=lr or 7e-5,
                bandwidth=bandwidth or 0.001,
                sparsity_penalty=sparsity_penalty or 1.0,
                target_l0=target_l0 or 20.0,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "JumpReluTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = JumpReluSAE(model_config)
        self.ae.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(),
            lr=training_config.lr,
            betas=(0.0, 0.999),
            eps=1e-8
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
        
        # Dead feature tracking
        self.num_tokens_since_fired = torch.zeros(
            model_config.dict_size, 
            dtype=torch.long, 
            device=self.device
        )
        self.dead_features = -1
        self.logging_parameters = ["dead_features"]
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with proper L0-based sparsity penalty."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(device=self.device, dtype=self.ae.W_enc.dtype)
        
        # Forward pass with pre-jump values for L0 computation
        f, pre_jump = self.ae.encode(x, output_pre_jump=True)
        x_hat = self.ae.decode(f)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # L0 sparsity computation
        threshold_clamped = torch.clamp(self.ae.threshold, min=1e-6)
        l0 = StepFunction.apply(f, threshold_clamped, self.training_config.bandwidth).sum(dim=-1).mean()
        
        # L0-based sparsity loss (target-based)
        sparsity_loss = (
            self.training_config.sparsity_penalty 
            * ((l0 / self.training_config.target_l0) - 1).pow(2) 
            * sparsity_scale
        )
        
        total_loss = recon_loss + sparsity_loss
        
        # Update dead feature tracking
        active_indices = f.sum(0) > 0
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            self.num_tokens_since_fired > self.training_config.dead_feature_threshold
        ).sum().item()
        
        if not logging:
            return total_loss
        
        # Return detailed loss information with diagnostics
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        feature_diagnostics = self.ae.get_feature_diagnostics(x)
        
        return LossLog(
            x, x_hat, f,
            {
                'l2_loss': torch.norm(x - x_hat, dim=-1).mean().item(),
                'recon_loss': recon_loss.item(),
                'sparsity_loss': sparsity_loss.item(),
                'loss': total_loss.item(),
                'l0': l0.item(),
                'target_l0': self.training_config.target_l0,
                'sparsity_scale': sparsity_scale,
                'dead_features': self.dead_features,
                'mean_threshold': self.ae.threshold.mean().item(),
                # Additional diagnostics
                **{k: v.item() if torch.is_tensor(v) else v for k, v in feature_diagnostics.items()}
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with proper gradient handling."""
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient modifications for decoder normalization
        if self.ae.W_dec.grad is not None:
            self.ae.W_dec.grad.data = remove_gradient_parallel_to_decoder_directions(
                self.ae.W_dec.data.T, 
                self.ae.W_dec.grad.data.T, 
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
        with torch.no_grad():
            self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
                self.ae.W_dec.data.T, 
                self.ae.activation_dim, 
                self.ae.dict_size
            ).T
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'JumpReluSAE',
            'trainer_class': 'JumpReluTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'bandwidth': self.model_config.bandwidth,
            'threshold_init': self.model_config.threshold_init,
            'apply_b_dec_to_input': self.model_config.apply_b_dec_to_input,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'sparsity_penalty': self.training_config.sparsity_penalty,
            'target_l0': self.training_config.target_l0,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }
