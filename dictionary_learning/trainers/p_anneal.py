"""
Robust P-Annealing SAE Trainer with enhanced architecture and error handling.

Key improvements:
- Dataclass-based configuration
- Enhanced typing and error handling
- Better device/dtype management
- Comprehensive diagnostics
- Cleaner separation of concerns
- Numerical stability improvements
- FIXED: Logging parameters properly initialized as instance variables
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass
from collections import namedtuple
import logging

from ..dictionary import AutoEncoder
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam
)


@dataclass
class PAnnealConfig:
    """Configuration for P-Annealing training."""
    # Model configuration
    activation_dim: int
    dict_size: int
    dict_class: type = AutoEncoder
    
    # P-Annealing specific parameters
    sparsity_function: str = 'Lp'  # 'Lp' or 'Lp^p'
    initial_sparsity_penalty: float = 1e-1
    p_start: float = 1.0  # Starting value of p (L1 norm)
    p_end: float = 0.1  # Ending value of p (near L0)
    
    # Annealing schedule
    anneal_start: int = 15000  # Step to start annealing p
    anneal_end: Optional[int] = None  # Step to stop annealing (defaults to total steps)
    n_sparsity_updates: Union[int, str] = 10  # Number of p updates or "continuous"
    sparsity_queue_length: int = 10  # History length for adaptive penalty
    
    # System configuration
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.sparsity_function not in ['Lp', 'Lp^p']:
            raise ValueError("sparsity_function must be 'Lp' or 'Lp^p'")
        
        if not 0 < self.p_end <= self.p_start <= 2:
            raise ValueError("p values must satisfy 0 < p_end <= p_start <= 2")
        
        if self.initial_sparsity_penalty <= 0:
            raise ValueError("initial_sparsity_penalty must be positive")
        
        if self.anneal_start < 0:
            raise ValueError("anneal_start must be non-negative")
            
        if isinstance(self.n_sparsity_updates, int) and self.n_sparsity_updates <= 0:
            raise ValueError("n_sparsity_updates must be positive")
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class PAnnealTrainingConfig:
    """Training configuration for P-Annealing."""
    steps: int
    lr: float = 1e-3
    warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = min(1000, max(100, int(0.05 * self.steps)))
        
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = min(2000, max(200, int(0.1 * self.steps)))
        
        # Ensure decay_start is after warmup phases
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= min_decay_start:
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class PAnnealTrainer(SAETrainer):
    """
    Robust P-Annealing SAE trainer with enhanced error handling and diagnostics.
    
    Implements progressive sparsity regularization that transitions from L1 to near-L0 penalties,
    with adaptive coefficient adjustment to maintain consistent sparsity levels.
    """
    
    def __init__(
        self,
        model_config: Optional[PAnnealConfig] = None,
        training_config: Optional[PAnnealTrainingConfig] = None,
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
        sparsity_function: Optional[str] = None,
        initial_sparsity_penalty: Optional[float] = None,
        p_start: Optional[float] = None,
        p_end: Optional[float] = None,
        anneal_start: Optional[int] = None,
        anneal_end: Optional[int] = None,
        n_sparsity_updates: Optional[Union[int, str]] = None,
        device: Optional[str] = None,
        dict_class: Optional[type] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = PAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                dict_class=dict_class or AutoEncoder,
                sparsity_function=sparsity_function or 'Lp',
                initial_sparsity_penalty=initial_sparsity_penalty or 1e-1,
                p_start=p_start or 1.0,
                p_end=p_end or 0.1,
                anneal_start=anneal_start or 15000,
                anneal_end=anneal_end,
                n_sparsity_updates=n_sparsity_updates or 10,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = PAnnealTrainingConfig(
                steps=steps,
                lr=lr or 1e-3,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "PAnnealTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = model_config.dict_class(model_config.activation_dim, model_config.dict_size)
        self.ae.to(device=self.device, dtype=model_config.dtype)
        
        # Initialize P-annealing parameters
        self._setup_p_annealing()
        
        # FIXED: Initialize logging variables as instance attributes
        self.annealing_active = False
        self.lp_loss = torch.tensor(0.0, device=self.device)
        self.scaled_lp_loss = torch.tensor(0.0, device=self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            self.ae.decoder.parameters(),
            lr=training_config.lr
        )
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            training_config.resample_steps,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps,
            training_config.sparsity_warmup_steps
        )
        
        # Initialize resampling tracking
        if training_config.resample_steps is not None:
            self.steps_since_active = torch.zeros(
                model_config.dict_size, 
                dtype=torch.int, 
                device=self.device
            )
        else:
            self.steps_since_active = None
        
        # FIXED: Setup logging with properly initialized parameters
        self.logging_parameters = [
            'p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'sparsity_coeff',
            'p_step_count', 'annealing_active'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_p_annealing(self) -> None:
        """Initialize P-annealing schedule and tracking variables."""
        config = self.model_config
        training_config = self.training_config
        
        # Set annealing end if not specified
        if config.anneal_end is None:
            self.anneal_end = training_config.steps - 1
        else:
            self.anneal_end = config.anneal_end
        
        # Current p value and tracking
        self.p = config.p_start
        self.next_p = None
        self.p_step_count = 0
        self.sparsity_coeff = config.initial_sparsity_penalty
        
        # Calculate p update schedule
        if config.n_sparsity_updates == "continuous":
            self.n_sparsity_updates = self.anneal_end - config.anneal_start + 1
        else:
            self.n_sparsity_updates = config.n_sparsity_updates
        
        if self.n_sparsity_updates > 0:
            self.sparsity_update_steps = torch.linspace(
                config.anneal_start, 
                self.anneal_end, 
                self.n_sparsity_updates, 
                dtype=torch.int
            )
            self.p_values = torch.linspace(
                config.p_start, 
                config.p_end, 
                self.n_sparsity_updates
            )
        else:
            self.sparsity_update_steps = torch.tensor([], dtype=torch.int)
            self.p_values = torch.tensor([])
        
        # Check for duplicate update steps
        if len(self.sparsity_update_steps) > 0:
            unique_steps, counts = self.sparsity_update_steps.unique(return_counts=True)
            if (counts > 1).any():
                self.logger.warning("Duplicate steps detected in sparsity update schedule!")
        
        # Sparsity tracking queue for adaptive coefficient
        self.sparsity_queue: List[Tuple[float, float]] = []
        
        # Initialize next_p if we have updates scheduled
        if len(self.p_values) > 1:
            self.next_p = self.p_values[1].item()
    
    def lp_norm(self, f: torch.Tensor, p: float) -> torch.Tensor:
        """
        Compute Lp norm with numerical stability.
        
        Args:
            f: Feature activations [batch_size, dict_size]
            p: Norm parameter
            
        Returns:
            Lp norm value
        """
        # Ensure p is positive for numerical stability
        p = max(p, 1e-8)
        
        # Compute p-th power with numerical safety
        f_abs = torch.abs(f)
        f_pow_p = torch.pow(f_abs + 1e-8, p)  # Add small epsilon to avoid pow(0, p) issues
        norm_p = f_pow_p.sum(dim=-1)
        
        if self.model_config.sparsity_function == 'Lp^p':
            return norm_p.mean()
        elif self.model_config.sparsity_function == 'Lp':
            # For Lp norm, take the p-th root
            if p != 1.0:
                norm = torch.pow(norm_p + 1e-8, 1.0 / p)
            else:
                norm = norm_p  # L1 case
            return norm.mean()
        else:
            raise ValueError(f"Unknown sparsity function: {self.model_config.sparsity_function}")
    
    def _update_p_schedule(self, step: int, f: torch.Tensor) -> None:
        """Update p value and sparsity coefficient based on schedule."""
        if step not in self.sparsity_update_steps:
            return
        
        # Make sure we don't update on repeated steps
        if step < self.sparsity_update_steps[self.p_step_count]:
            return
        
        # Adapt sparsity penalty coefficient if we have next_p
        if self.next_p is not None and len(self.sparsity_queue) > 0:
            try:
                # Calculate average sparsity for current and next p
                current_sparsities = [entry[0] for entry in self.sparsity_queue]
                next_sparsities = [entry[1] for entry in self.sparsity_queue]
                
                avg_current = torch.tensor(current_sparsities).mean()
                avg_next = torch.tensor(next_sparsities).mean()
                
                # Avoid division by zero
                if avg_next > 1e-8:
                    adaptation_ratio = (avg_current / avg_next).item()
                    # Clamp adaptation ratio to reasonable range
                    adaptation_ratio = torch.clamp(torch.tensor(adaptation_ratio), 0.1, 10.0).item()
                    self.sparsity_coeff *= adaptation_ratio
                    
                    self.logger.info(
                        f"Step {step}: Adapted sparsity coefficient by {adaptation_ratio:.3f} "
                        f"(new coeff: {self.sparsity_coeff:.6f})"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to adapt sparsity coefficient: {e}")
        
        # Update p value
        old_p = self.p
        self.p = self.p_values[self.p_step_count].item()
        
        # Set next_p for queue tracking
        if self.p_step_count < self.n_sparsity_updates - 1:
            self.next_p = self.p_values[self.p_step_count + 1].item()
        else:
            self.next_p = self.model_config.p_end
        
        self.p_step_count += 1
        
        self.logger.info(f"Step {step}: Updated p from {old_p:.3f} to {self.p:.3f}")
    
    def _update_sparsity_queue(self, f: torch.Tensor) -> None:
        """Update the sparsity tracking queue for adaptive coefficient."""
        if self.next_p is None:
            return
        
        try:
            current_lp = self.lp_norm(f, self.p).item()
            next_lp = self.lp_norm(f, self.next_p).item()
            
            self.sparsity_queue.append((current_lp, next_lp))
            
            # Keep queue at specified length
            if len(self.sparsity_queue) > self.model_config.sparsity_queue_length:
                self.sparsity_queue = self.sparsity_queue[-self.model_config.sparsity_queue_length:]
                
        except Exception as e:
            self.logger.warning(f"Failed to update sparsity queue: {e}")
    
    def resample_neurons(self, dead_mask: torch.Tensor, activations: torch.Tensor) -> None:
        """
        Resample dead neurons with enhanced error handling.
        
        Args:
            dead_mask: Boolean mask indicating dead neurons
            activations: Current batch of activations for resampling
        """
        if dead_mask.sum() == 0:
            return
        
        with torch.no_grad():
            try:
                n_dead = dead_mask.sum().item()
                self.logger.info(f"Resampling {n_dead} dead neurons")
                
                # Compute reconstruction losses for sampling
                x_hat = self.ae(activations)
                losses = (activations - x_hat).norm(dim=-1)
                
                # Sample activations proportional to loss
                n_resample = min(n_dead, losses.shape[0])
                if n_resample == 0:
                    return
                
                indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
                sampled_vecs = activations[indices]
                
                # Reset weights for dead neurons
                alive_neurons = ~dead_mask
                if alive_neurons.sum() > 0:
                    alive_norm = self.ae.encoder.weight[alive_neurons].norm(dim=-1).mean()
                else:
                    alive_norm = 1.0
                
                # Reset encoder weights
                self.ae.encoder.weight.data[dead_mask][:n_resample] = (
                    sampled_vecs * alive_norm * 0.2
                )
                
                # Reset decoder weights (normalized)
                normalized_vecs = sampled_vecs / (sampled_vecs.norm(dim=-1, keepdim=True) + 1e-8)
                self.ae.decoder.weight.data[:, dead_mask][:, :n_resample] = normalized_vecs.T
                
                # Reset encoder bias
                self.ae.encoder.bias.data[dead_mask][:n_resample] = 0.0
                
                # Reset optimizer state
                self._reset_optimizer_state(dead_mask, n_resample)
                
            except Exception as e:
                self.logger.error(f"Failed to resample neurons: {e}")
    
    def _reset_optimizer_state(self, dead_mask: torch.Tensor, n_resample: int) -> None:
        """Reset Adam optimizer state for resampled neurons."""
        try:
            state_dict = self.optimizer.state_dict()['state']
            
            # Find parameter indices (this is fragile and depends on parameter order)
            param_indices = list(state_dict.keys())
            
            if len(param_indices) >= 3:
                # Encoder weight (usually index 0 or 1)
                encoder_weight_idx = param_indices[1]  # May need adjustment
                if 'exp_avg' in state_dict[encoder_weight_idx]:
                    state_dict[encoder_weight_idx]['exp_avg'][dead_mask][:n_resample] = 0.0
                    state_dict[encoder_weight_idx]['exp_avg_sq'][dead_mask][:n_resample] = 0.0
                
                # Encoder bias
                encoder_bias_idx = param_indices[2]  # May need adjustment
                if 'exp_avg' in state_dict[encoder_bias_idx]:
                    state_dict[encoder_bias_idx]['exp_avg'][dead_mask][:n_resample] = 0.0
                    state_dict[encoder_bias_idx]['exp_avg_sq'][dead_mask][:n_resample] = 0.0
                
                # Decoder weight
                decoder_weight_idx = param_indices[3]  # May need adjustment
                if 'exp_avg' in state_dict[decoder_weight_idx]:
                    state_dict[decoder_weight_idx]['exp_avg'][:, dead_mask][:, :n_resample] = 0.0
                    state_dict[decoder_weight_idx]['exp_avg_sq'][:, dead_mask][:, :n_resample] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Failed to reset optimizer state: {e}")
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with P-annealing sparsity penalty."""
        # Get sparsity warmup scale
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Forward pass
        x_hat, f = self.ae(x, output_features=True)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # P-annealed sparsity loss
        lp_loss = self.lp_norm(f, self.p)
        scaled_lp_loss = lp_loss * self.sparsity_coeff * sparsity_scale
        
        # FIXED: Update instance variables for logging
        self.lp_loss = lp_loss.detach()
        self.scaled_lp_loss = scaled_lp_loss.detach()
        self.annealing_active = step >= self.model_config.anneal_start
        
        # Update sparsity queue for adaptive coefficient
        self._update_sparsity_queue(f)
        
        # Update p schedule
        self._update_p_schedule(step, f)
        
        # Update dead neuron tracking
        if self.steps_since_active is not None:
            with torch.no_grad():
                dead_features = (f == 0).all(dim=0)
                self.steps_since_active[dead_features] += 1
                self.steps_since_active[~dead_features] = 0
        
        total_loss = recon_loss + scaled_lp_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        return LossLog(
            x, x_hat, f,
            {
                'recon_loss': recon_loss.item(),
                'lp_loss': lp_loss.item(),
                'scaled_lp_loss': scaled_lp_loss.item(),
                'loss': total_loss.item(),
                'p': self.p,
                'next_p': self.next_p,
                'sparsity_coeff': self.sparsity_coeff,
                'p_step_count': self.p_step_count,
                'annealing_active': self.annealing_active,
                'sparsity_scale': sparsity_scale,
                'queue_length': len(self.sparsity_queue),
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step with resampling."""
        activations = activations.to(device=self.device, dtype=self.model_config.dtype)
        
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
        
        # Resample dead neurons if needed
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == self.training_config.resample_steps - 1):
            
            dead_threshold = self.training_config.resample_steps // 2
            dead_mask = self.steps_since_active > dead_threshold
            self.resample_neurons(dead_mask, activations)
    
    def get_p_diagnostics(self) -> Dict[str, Any]:
        """Get detailed P-annealing diagnostics."""
        return {
            'current_p': self.p,
            'next_p': self.next_p,
            'p_step_count': self.p_step_count,
            'total_p_updates': self.n_sparsity_updates,
            'sparsity_coeff': self.sparsity_coeff,
            'annealing_progress': self.p_step_count / max(self.n_sparsity_updates, 1),
            'queue_length': len(self.sparsity_queue),
            'sparsity_function': self.model_config.sparsity_function,
            'anneal_start': self.model_config.anneal_start,
            'anneal_end': self.anneal_end,
        }
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'trainer_class': 'PAnnealTrainer',
            'dict_class': self.model_config.dict_class.__name__,
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'sparsity_function': self.model_config.sparsity_function,
            'initial_sparsity_penalty': self.model_config.initial_sparsity_penalty,
            'p_start': self.model_config.p_start,
            'p_end': self.model_config.p_end,
            'anneal_start': self.model_config.anneal_start,
            'anneal_end': self.anneal_end,
            'n_sparsity_updates': self.n_sparsity_updates,
            'sparsity_queue_length': self.model_config.sparsity_queue_length,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'warmup_steps': self.training_config.warmup_steps,
            'decay_start': self.training_config.decay_start,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'resample_steps': self.training_config.resample_steps,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }