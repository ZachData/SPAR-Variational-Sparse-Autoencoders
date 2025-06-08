"""
Robust Standard SAE implementation with enhanced configuration and error handling.
Based on vsae_iso.py architecture but implementing standard SAE training schemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import AutoEncoder
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam,
)


@dataclass
class StandardSAEConfig:
    """Configuration for Standard SAE model."""
    activation_dim: int
    dict_size: int
    use_april_update_mode: bool = False  # True for April update, False for original Towards Monosemanticity
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class StandardSAETrainingConfig:
    """Enhanced training configuration for Standard SAE."""
    steps: int
    lr: float = 1e-3
    l1_penalty: float = 1e-1
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    resample_steps: Optional[int] = None  # Set to enable neuron resampling
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        
        # Set reasonable decay start
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class StandardSAE(AutoEncoder):
    """
    Robust Standard SAE implementation with enhanced initialization and error handling.
    """
    
    def __init__(self, config: StandardSAEConfig):
        # Call the parent AutoEncoder constructor
        super().__init__(
            config.activation_dim, 
            config.dict_size, 
            use_april_update_mode=config.use_april_update_mode
        )
        
        self.config = config
        
        # Move to correct device and dtype
        device = config.get_device()
        self.to(device=device, dtype=config.dtype)
        
        # Re-initialize weights with better defaults
        self._robust_init_weights()
    
    def _robust_init_weights(self) -> None:
        """Initialize weights with robust defaults."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        with torch.no_grad():
            # Initialize encoder and decoder with tied weights
            w = torch.randn(
                self.activation_dim,
                self.dict_size,
                dtype=dtype,
                device=device
            )
            w = w / w.norm(dim=0, keepdim=True) * 0.1
            
            # Set weights
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Initialize biases
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
    
    @classmethod
    def from_config(cls, config: StandardSAEConfig) -> 'StandardSAE':
        """Create a StandardSAE from configuration."""
        return cls(config)
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[StandardSAEConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
    ) -> 'StandardSAE':
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
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            use_april_update_mode = "decoder.bias" in state_dict
            
            config = StandardSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        # Create model and load state
        model = cls(config)
        
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


class StandardTrainer(SAETrainer):
    """
    Standard SAE trainer with enhanced error handling and configuration.
    """
    
    def __init__(
        self,
        model_config: Optional[StandardSAEConfig] = None,
        training_config: Optional[StandardSAETrainingConfig] = None,
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
        l1_penalty: Optional[float] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use StandardSAE
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = StandardSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else False,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = StandardSAETrainingConfig(
                steps=steps,
                lr=lr or 1e-3,
                l1_penalty=l1_penalty or 1e-1,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "StandardTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = StandardSAE(model_config)
        self.ae.to(self.device)
        
        # Initialize neuron resampling if enabled
        if training_config.resample_steps is not None:
            self.steps_since_active = torch.zeros(
                self.ae.dict_size, 
                dtype=torch.int, 
                device=self.device
            )
        else:
            self.steps_since_active = None
        
        # Initialize optimizer
        if model_config.use_april_update_mode:
            # April update uses standard Adam
            self.optimizer = torch.optim.Adam(
                self.ae.parameters(),
                lr=training_config.lr,
                betas=(0.9, 0.999)
            )
        else:
            # Original approach uses ConstrainedAdam
            self.optimizer = ConstrainedAdam(
                self.ae.parameters(),
                self.ae.decoder.parameters(),
                lr=training_config.lr
            )
        
        # Initialize scheduler
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            training_config.resample_steps,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Initialize sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps,
            training_config.sparsity_warmup_steps
        )
    
    def resample_neurons(self, deads: torch.Tensor, activations: torch.Tensor) -> None:
        """Resample dead neurons to prevent feature death."""
        with torch.no_grad():
            if deads.sum() == 0:
                return
            
            print(f"Resampling {deads.sum().item()} dead neurons")
            
            # Compute reconstruction losses for sampling
            losses = (activations - self.ae(activations)).norm(dim=-1)
            
            # Sample activations proportional to their reconstruction loss
            n_resample = min(deads.sum().item(), losses.shape[0])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]
            
            # Get norm of living neurons for proper scaling
            alive_mask = ~deads
            if alive_mask.any():
                alive_norm = self.ae.encoder.weight[alive_mask].norm(dim=-1).mean()
            else:
                alive_norm = torch.tensor(1.0, device=self.device)
            
            # Resample encoder weights
            dead_indices = deads.nonzero(as_tuple=True)[0][:n_resample]
            self.ae.encoder.weight[dead_indices] = sampled_vecs * alive_norm * 0.2
            
            # Resample decoder weights (ensure proper dtype)
            decoder_dtype = self.ae.decoder.weight.dtype
            normalized_vecs = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.decoder.weight[:, dead_indices] = normalized_vecs.to(dtype=decoder_dtype)
            
            # Reset encoder biases
            self.ae.encoder.bias[dead_indices] = 0.0
            
            # Reset optimizer state for resampled neurons
            self._reset_optimizer_state_for_neurons(dead_indices)
            
            # Reset activity tracking
            if self.steps_since_active is not None:
                self.steps_since_active[dead_indices] = 0
    
    def _reset_optimizer_state_for_neurons(self, neuron_indices: torch.Tensor) -> None:
        """Reset Adam optimizer state for specific neurons."""
        try:
            state_dict = self.optimizer.state_dict()['state']
            
            # Reset encoder weight states
            if 1 in state_dict:  # encoder.weight is typically param 1
                state_dict[1]['exp_avg'][neuron_indices] = 0.0
                state_dict[1]['exp_avg_sq'][neuron_indices] = 0.0
            
            # Reset encoder bias states
            if 2 in state_dict:  # encoder.bias is typically param 2
                state_dict[2]['exp_avg'][neuron_indices] = 0.0
                state_dict[2]['exp_avg_sq'][neuron_indices] = 0.0
            
            # Reset decoder weight states
            if 3 in state_dict:  # decoder.weight is typically param 3
                state_dict[3]['exp_avg'][:, neuron_indices] = 0.0
                state_dict[3]['exp_avg_sq'][:, neuron_indices] = 0.0
                
        except Exception as e:
            print(f"Warning: Could not reset optimizer state: {e}")
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with proper sparsity scaling."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        x_hat, f = self.ae(x, output_features=True)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        f = f.to(dtype=original_dtype)
        
        # Reconstruction loss
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # Sparsity loss (different for April update vs original)
        if self.model_config.use_april_update_mode:
            # April update: weight L1 by decoder norms
            l1_loss = (f * self.ae.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()
        else:
            # Original: simple L1
            l1_loss = f.norm(p=1, dim=-1).mean()
        
        # Update neuron activity tracking
        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        # Total loss
        total_loss = recon_loss + self.training_config.l1_penalty * sparsity_scale * l1_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        return LossLog(
            x, x_hat, f,
            {
                'l2_loss': l2_loss.item(),
                'mse_loss': recon_loss.item(),
                'sparsity_loss': l1_loss.item(),
                'loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'n_dead_neurons': (self.steps_since_active > self.training_config.resample_steps // 2).sum().item() if self.steps_since_active is not None else 0,
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with optional neuron resampling."""
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
        
        # Neuron resampling
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == 0 and 
            step > 0):
            
            dead_threshold = self.training_config.resample_steps // 2
            dead_mask = self.steps_since_active > dead_threshold
            self.resample_neurons(dead_mask, activations)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'StandardSAE',
            'trainer_class': 'StandardTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'l1_penalty': self.training_config.l1_penalty,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'resample_steps': self.training_config.resample_steps,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }


# Backwards compatibility aliases
StandardTrainerAprilUpdate = lambda **kwargs: StandardTrainer(
    use_april_update_mode=True, **kwargs
)