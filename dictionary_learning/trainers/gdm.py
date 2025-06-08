"""
Enhanced Gated Dictionary Model (GDM) implementation with improved robustness.

Based on: https://arxiv.org/abs/2404.16014
Key improvements:
- Modern configuration management with dataclasses
- Enhanced error handling and numerical stability
- Better device and dtype management
- Comprehensive logging and diagnostics
- Gradient clipping and modern training practices
- Proper initialization and normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import GatedAutoEncoder as BaseGatedAutoEncoder
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam,
)


@dataclass
class GDMConfig:
    """Configuration for Gated Dictionary Model."""
    activation_dim: int
    dict_size: int
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    initialization: str = "default"  # Can be customized for different init strategies
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class GDMTrainingConfig:
    """Enhanced training configuration for GDM."""
    steps: int
    lr: float = 5e-5
    l1_penalty: float = 1e-1
    aux_loss_weight: float = 1.0  # Weight for auxiliary loss
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = max(2000, int(0.05 * self.steps))
        
        # Set decay start if not provided
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class GatedAutoEncoder(BaseGatedAutoEncoder):
    """
    Enhanced version of GatedAutoEncoder with better initialization and diagnostics.
    Inherits from the base GatedAutoEncoder in dictionary.py but adds robustness features.
    """
    
    def __init__(self, config: GDMConfig):
        # Initialize with the base class
        super().__init__(
            activation_dim=config.activation_dim,
            dict_size=config.dict_size,
            initialization=config.initialization,
            device=config.get_device()
        )
        self.config = config
        
        # Convert to desired dtype
        self.to(dtype=config.dtype, device=config.get_device())
    
    def get_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed diagnostics for monitoring training.
        
        Returns:
            Dictionary with various diagnostic metrics
        """
        with torch.no_grad():
            f, f_gate = self.encode(x, return_gate=True)
            x_hat = self.decode(f)
            x_hat_gate = f_gate @ self.decoder.weight.detach().T + self.decoder_bias.detach()
            
            # Basic statistics
            gate_l0 = (f_gate > 0).float().sum(dim=-1).mean()
            feature_l0 = (f != 0).float().sum(dim=-1).mean()
            
            # Reconstruction quality
            recon_error = torch.norm(x - x_hat, dim=-1).mean()
            aux_recon_error = torch.norm(x - x_hat_gate, dim=-1).mean()
            
            # Gate statistics
            gate_magnitude = f_gate.norm(dim=-1).mean()
            feature_magnitude = f.norm(dim=-1).mean()
            
            # Fraction of features that are "alive" (used in any sample)
            frac_alive_gates = (f_gate.abs().max(dim=0)[0] > 1e-6).float().mean()
            frac_alive_features = (f.abs().max(dim=0)[0] > 1e-6).float().mean()
            
            return {
                'gate_l0': gate_l0,
                'feature_l0': feature_l0,
                'recon_error': recon_error,
                'aux_recon_error': aux_recon_error,
                'gate_magnitude': gate_magnitude,
                'feature_magnitude': feature_magnitude,
                'frac_alive_gates': frac_alive_gates,
                'frac_alive_features': frac_alive_features,
                'gate_max': f_gate.max(),
                'gate_mean': f_gate.mean(),
                'feature_max': f.max(),
                'feature_mean': f.mean(),
            }
    
    def forward_with_gates(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning all intermediate values for loss computation.
        
        Returns:
            Tuple of (x_hat, x_hat_gate, f, f_gate)
        """
        # Store original dtype
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.decoder_bias.dtype)
        
        # Encode
        f, f_gate = self.encode(x, return_gate=True)
        
        # Decode
        x_hat = self.decode(f)
        
        # Auxiliary reconstruction from gates
        x_hat_gate = f_gate @ self.decoder.weight.detach().T + self.decoder_bias.detach()
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        x_hat_gate = x_hat_gate.to(dtype=original_dtype)
        f = f.to(dtype=original_dtype)
        f_gate = f_gate.to(dtype=original_dtype)
        
        return x_hat, x_hat_gate, f, f_gate
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[GDMConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'GatedAutoEncoder':
        """
        Load a pretrained model with enhanced configuration support.
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            activation_dim, dict_size = state_dict["decoder.weight"].shape
            config = GDMConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                dtype=dtype,
                device=device
            )
        
        # Create model
        model = cls(config)
        
        # Load state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model


class GatedSAETrainer(SAETrainer):
    """
    Enhanced Gated SAE trainer with modern training practices.
    
    Key improvements:
    - Better configuration management
    - Enhanced numerical stability
    - Comprehensive logging and diagnostics
    - Gradient clipping and proper optimization
    - Better error handling
    """
    
    def __init__(
        self,
        model_config: Optional[GDMConfig] = None,
        training_config: Optional[GDMTrainingConfig] = None,
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
        warmup_steps: Optional[int] = None,
        sparsity_warmup_steps: Optional[int] = None,
        decay_start: Optional[int] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use EnhancedGatedAutoEncoder
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = GDMConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = GDMTrainingConfig(
                steps=steps,
                lr=lr or 5e-5,
                l1_penalty=l1_penalty or 1e-1,
                warmup_steps=warmup_steps,
                sparsity_warmup_steps=sparsity_warmup_steps,
                decay_start=decay_start,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "GDMTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = GatedAutoEncoder(model_config)
        self.ae.to(self.device)
        
        # Initialize optimizer with ConstrainedAdam
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            self.ae.decoder.parameters(),  # Constrain decoder weights
            lr=training_config.lr,
            betas=(0.0, 0.999),  # β1=0 as in original paper
        )
        
        # Initialize scheduler
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
    
    def _compute_gated_loss(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor, 
        x_hat_gate: torch.Tensor, 
        f: torch.Tensor, 
        f_gate: torch.Tensor,
        sparsity_scale: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the three-component GDM loss with numerical stability.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Ensure all tensors have the same dtype
        dtype = x.dtype
        x_hat = x_hat.to(dtype=dtype)
        x_hat_gate = x_hat_gate.to(dtype=dtype)
        f_gate = f_gate.to(dtype=dtype)
        
        # L_recon: Main reconstruction loss
        L_recon = torch.mean(torch.sum((x - x_hat) ** 2, dim=-1))
        
        # L_sparse: Sparsity penalty on gates (L1 norm)
        L_sparse = torch.mean(torch.norm(f_gate, p=1, dim=-1))
        
        # L_aux: Auxiliary loss - gates should reconstruct input
        L_aux = torch.mean(torch.sum((x - x_hat_gate) ** 2, dim=-1))
        
        # Ensure all losses are non-negative
        L_recon = torch.clamp(L_recon, min=0.0)
        L_sparse = torch.clamp(L_sparse, min=0.0)
        L_aux = torch.clamp(L_aux, min=0.0)
        
        # Combine losses
        total_loss = (
            L_recon + 
            self.training_config.l1_penalty * sparsity_scale * L_sparse + 
            self.training_config.aux_loss_weight * L_aux
        )
        
        loss_components = {
            'recon_loss': L_recon,
            'sparsity_loss': L_sparse,
            'aux_loss': L_aux,
            'total_loss': total_loss,
        }
        
        return total_loss, loss_components
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute GDM loss with enhanced diagnostics."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        x_hat, x_hat_gate, f, f_gate = self.ae.forward_with_gates(x)
        
        # Compute loss
        total_loss, loss_components = self._compute_gated_loss(
            x, x_hat, x_hat_gate, f, f_gate, sparsity_scale
        )
        
        if not logging:
            return total_loss
        
        # Return detailed loss information with diagnostics
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        diagnostics = self.ae.get_diagnostics(x)
        
        # Combine loss components and diagnostics
        log_dict = {}
        for k, v in loss_components.items():
            log_dict[k] = v.item() if torch.is_tensor(v) else v
        
        for k, v in diagnostics.items():
            log_dict[k] = v.item() if torch.is_tensor(v) else v
        
        # Add training-specific metrics
        log_dict.update({
            'sparsity_scale': sparsity_scale,
            'lr': self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.training_config.lr,
            'step': step,
        })
        
        return LossLog(x, x_hat, f, log_dict)
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with enhanced error handling."""
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        
        # Check for NaN/Inf losses
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected at step {step}: {loss}")
            return
        
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
            'dict_class': 'GatedAutoEncoder',
            'trainer_class': 'GDMTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            'initialization': self.model_config.initialization,
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'l1_penalty': self.training_config.l1_penalty,
            'aux_loss_weight': self.training_config.aux_loss_weight,
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
