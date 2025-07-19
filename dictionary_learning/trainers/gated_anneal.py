"""
Enhanced Gated SAE with P-Annealing implementation - Robust version with bias handling fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable, List
from collections import namedtuple, deque
from dataclasses import dataclass

from ..dictionary import Dictionary, GatedAutoEncoder
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam,
)


@dataclass
class GatedAnnealConfig:
    """Configuration for Gated SAE with P-Annealing."""
    activation_dim: int
    dict_size: int
    sparsity_function: str = 'Lp^p'  # 'Lp' or 'Lp^p'
    initial_sparsity_penalty: float = 1e-1
    anneal_start: int = 15000
    anneal_end: Optional[int] = None  # Defaults to total steps - 1
    p_start: float = 1.0
    p_end: float = 0.0
    n_sparsity_updates: int = 10
    sparsity_queue_length: int = 10
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class GatedAnnealTrainingConfig:
    """Training configuration for Gated SAE with P-Annealing."""
    steps: int
    lr: float = 3e-4
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = min(max(200, int(0.02 * self.steps)), self.steps - 1)
        if self.sparsity_warmup_steps is None:
            # Ensure sparsity warmup steps is less than total steps
            target_sparsity_warmup = max(500, int(0.05 * self.steps))
            self.sparsity_warmup_steps = min(target_sparsity_warmup, self.steps - 1)

        # Handle mutual exclusivity: decay_start and resample_steps cannot both be set
        if self.resample_steps is not None:
            # If resample_steps is set, disable decay_start
            self.decay_start = None
        elif self.decay_start is None:
            # Only set decay_start if resample_steps is not set
            min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
            default_decay_start = int(0.8 * self.steps)
            
            if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
                self.decay_start = None  # Disable decay
            else:
                self.decay_start = default_decay_start


class EnhancedGatedAutoEncoder(GatedAutoEncoder):
    """Enhanced Gated AutoEncoder with better initialization and utilities."""
    
    def __init__(self, config: GatedAnnealConfig):
        # Initialize with the standard GatedAutoEncoder parameters
        super().__init__(
            activation_dim=config.activation_dim,
            dict_size=config.dict_size,
            device=config.get_device()
        )
        self.config = config
        
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[GatedAnnealConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> 'EnhancedGatedAutoEncoder':
        """Load a pretrained gated autoencoder."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            config = GatedAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                dtype=dtype,
                device=device
            )
        
        # Create model and load state
        model = cls(config)
        model.load_state_dict(state_dict)
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model

    def get_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed diagnostics for monitoring training."""
        with torch.no_grad():
            f, f_gate = self.encode(x, return_gate=True)
            x_hat = self.decode(f)
            x_hat_gate = f_gate @ self.decoder.weight.detach().T + self.decoder_bias.detach()
            
            return {
                'mean_activation': f.mean(),
                'mean_gate_activation': f_gate.mean(),
                'sparsity_l0': (f > 0).float().mean(),
                'gate_sparsity_l0': (f_gate > 0).float().mean(),
                'reconstruction_error': (x - x_hat).norm(dim=-1).mean(),
                'gate_reconstruction_error': (x - x_hat_gate).norm(dim=-1).mean(),
                'feature_density': (f > 0).float().sum(dim=-1).mean(),
                'gate_density': (f_gate > 0).float().sum(dim=-1).mean(),
            }


class GatedAnnealTrainer(SAETrainer):
    """
    Enhanced Gated SAE trainer with P-Annealing and improved robustness.
    
    Key improvements:
    - Proper configuration management
    - Enhanced numerical stability
    - Better error handling
    - Comprehensive logging
    - Fixed bias handling for GatedAutoEncoder
    """
    
    def __init__(
        self,
        model_config: Optional[GatedAnnealConfig] = None,
        training_config: Optional[GatedAnnealTrainingConfig] = None,
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
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = GatedAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                device=device_obj,
                **{k: v for k, v in kwargs.items() if k in GatedAnnealConfig.__dataclass_fields__}
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = GatedAnnealTrainingConfig(
                steps=steps,
                lr=lr or 3e-4,
                **{k: v for k, v in kwargs.items() if k in GatedAnnealTrainingConfig.__dataclass_fields__}
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "GatedAnnealTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = EnhancedGatedAutoEncoder(model_config)
        self.ae.to(self.device)
        
        # Initialize P-annealing parameters
        self._setup_p_annealing()
        
        # Initialize optimizer and scheduler
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            self.ae.decoder.parameters(),
            lr=training_config.lr,
            betas=(0.0, 0.999)
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
        
        # Initialize resampling if enabled
        if training_config.resample_steps is not None:
            self.steps_since_active = torch.zeros(
                model_config.dict_size, 
                dtype=torch.long,
                device=self.device
            )
        else:
            self.steps_since_active = None
        
        # Logging parameters
        self.logging_parameters = [
            'p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'sparsity_coeff'
        ]
        
    def _setup_p_annealing(self) -> None:
        """Setup P-annealing schedule and tracking."""
        config = self.model_config
        training_config = self.training_config
        
        # Set anneal_end if not provided
        anneal_end = config.anneal_end or (training_config.steps - 1)
        
        # Create annealing schedule
        if isinstance(config.n_sparsity_updates, str) and config.n_sparsity_updates == "continuous":
            n_updates = anneal_end - config.anneal_start + 1
        else:
            n_updates = config.n_sparsity_updates
            
        self.sparsity_update_steps = torch.linspace(
            config.anneal_start, 
            anneal_end, 
            n_updates, 
            dtype=torch.long
        )
        self.p_values = torch.linspace(
            config.p_start, 
            config.p_end, 
            n_updates
        )
        
        # Initialize tracking variables
        self.p = config.p_start
        self.next_p = self.p_values[1].item() if len(self.p_values) > 1 else config.p_end
        self.p_step_count = 0
        self.sparsity_coeff = config.initial_sparsity_penalty
        self.sparsity_queue = deque(maxlen=config.sparsity_queue_length)
        
        # Logging variables
        self.lp_loss = None
        self.scaled_lp_loss = None
    
    def lp_norm(self, f: torch.Tensor, p: float) -> torch.Tensor:
        """Compute Lp norm with numerical stability."""
        # Clamp p to avoid numerical issues
        p = max(p, 1e-6)
        
        if self.model_config.sparsity_function == 'Lp^p':
            return f.pow(p).sum(dim=-1).mean()
        elif self.model_config.sparsity_function == 'Lp':
            norm_p = f.pow(p).sum(dim=-1)
            return norm_p.pow(1/p).mean()
        else:
            raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
    
    def _compute_loss_components(
        self, 
        x: torch.Tensor, 
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all loss components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Forward pass
        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()
        
        # Loss components
        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()
        
        # Sparsity loss
        lp_loss = self.lp_norm(f_gate, self.p)
        scaled_lp_loss = lp_loss * self.sparsity_coeff * sparsity_scale
        
        # Store for logging
        self.lp_loss = lp_loss
        self.scaled_lp_loss = scaled_lp_loss
        
        return L_recon, L_aux, lp_loss, scaled_lp_loss
    
    def _update_p_annealing(self, step: int, f_gate: torch.Tensor) -> None:
        """Update P-annealing schedule and sparsity coefficient."""
        # Add to sparsity queue for next update
        if self.next_p is not None:
            lp_loss_next = self.lp_norm(f_gate, self.next_p)
            self.sparsity_queue.append([self.lp_loss.item(), lp_loss_next.item()])
        
        # Check if it's time to update (with bounds checking)
        if (self.p_step_count < len(self.sparsity_update_steps) and 
            step in self.sparsity_update_steps and 
            step >= self.sparsity_update_steps[self.p_step_count]):
            
            # Adapt sparsity penalty
            if len(self.sparsity_queue) > 0 and self.next_p is not None:
                queue_array = torch.tensor(list(self.sparsity_queue))
                local_sparsity_new = queue_array[:, 0].mean()
                local_sparsity_old = queue_array[:, 1].mean()
                
                # Avoid division by zero
                if local_sparsity_old > 1e-8:
                    ratio = (local_sparsity_new / local_sparsity_old).item()
                    # Clamp ratio to reasonable range
                    ratio = torch.clamp(torch.tensor(ratio), min=0.1, max=10.0).item()
                    self.sparsity_coeff *= ratio
            
            # Update p (with bounds checking)
            if self.p_step_count < len(self.p_values):
                self.p = self.p_values[self.p_step_count].item()
            
            # Set next p (with bounds checking)
            if self.p_step_count < len(self.p_values) - 1:
                self.next_p = self.p_values[self.p_step_count + 1].item()
            else:
                self.next_p = self.model_config.p_end
            
            self.p_step_count += 1
    
    def _update_resampling(self, f: torch.Tensor) -> None:
        """Update dead neuron tracking for resampling."""
        if self.steps_since_active is not None:
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
    
    def resample_neurons(self, deads: torch.Tensor, activations: torch.Tensor) -> None:
        """Resample dead neurons with improved stability and bias handling."""
        if deads.sum() == 0:
            return
            
        print(f"Resampling {deads.sum().item()} neurons")
        
        with torch.no_grad():
            # Compute reconstruction losses
            x_hat = self.ae(activations)
            losses = (activations - x_hat).norm(dim=-1)
            
            # Sample inputs for new weights
            n_resample = min(deads.sum().item(), losses.shape[0])
            if n_resample == 0:
                return
                
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]
            
            # FIXED: Get the indices of dead neurons (not boolean mask)
            dead_indices = deads.nonzero().flatten()[:n_resample]
            
            # FIXED: Ensure dtype consistency with model weights
            model_dtype = self.ae.encoder.weight.dtype
            device = self.ae.encoder.weight.device
            
            # Reset encoder/decoder weights using indices with proper dtype
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()
            new_encoder_weights = (sampled_vecs * alive_norm * 0.2).to(dtype=model_dtype, device=device)
            new_decoder_weights = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T.to(dtype=model_dtype, device=device)
            
            self.ae.encoder.weight[dead_indices] = new_encoder_weights
            self.ae.decoder.weight[:, dead_indices] = new_decoder_weights
            
            # FIXED: Reset encoder bias only if it exists (GatedAutoEncoder has bias=False)
            if self.ae.encoder.bias is not None:
                self.ae.encoder.bias[dead_indices] = 0.0
            
            # Reset optimizer state with indices
            self._reset_optimizer_state(dead_indices)
    
    def _reset_optimizer_state(self, dead_indices: torch.Tensor) -> None:
        """Reset Adam optimizer state for resampled neurons with proper indexing."""
        state_dict = self.optimizer.state_dict()['state']
        
        # Find the parameter indices (this is fragile and might need adjustment)
        param_indices = {id(param): i for i, param in enumerate(self.optimizer.param_groups[0]['params'])}
        
        # Reset encoder weight
        encoder_weight_idx = param_indices.get(id(self.ae.encoder.weight))
        if encoder_weight_idx is not None and encoder_weight_idx in state_dict:
            state_dict[encoder_weight_idx]['exp_avg'][dead_indices] = 0.0
            state_dict[encoder_weight_idx]['exp_avg_sq'][dead_indices] = 0.0
        
        # FIXED: Reset encoder bias only if it exists (GatedAutoEncoder has bias=False)
        if self.ae.encoder.bias is not None:
            encoder_bias_idx = param_indices.get(id(self.ae.encoder.bias))
            if encoder_bias_idx is not None and encoder_bias_idx in state_dict:
                state_dict[encoder_bias_idx]['exp_avg'][dead_indices] = 0.0
                state_dict[encoder_bias_idx]['exp_avg_sq'][dead_indices] = 0.0
        
        # Reset decoder weight
        decoder_weight_idx = param_indices.get(id(self.ae.decoder.weight))
        if decoder_weight_idx is not None and decoder_weight_idx in state_dict:
            state_dict[decoder_weight_idx]['exp_avg'][:, dead_indices] = 0.0
            state_dict[decoder_weight_idx]['exp_avg_sq'][:, dead_indices] = 0.0
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with enhanced error handling."""
        # Store original dtype
        original_dtype = x.dtype
        
        # Ensure input is on correct device and dtype
        x = x.to(device=self.device, dtype=self.ae.encoder.weight.dtype)
        
        # Compute loss components
        L_recon, L_aux, lp_loss, scaled_lp_loss = self._compute_loss_components(x, step)
        
        # Forward pass for p-annealing updates
        f, f_gate = self.ae.encode(x, return_gate=True)
        
        # Update p-annealing
        self._update_p_annealing(step, f_gate)
        
        # Update resampling tracking
        self._update_resampling(f)
        
        # Total loss
        total_loss = L_recon + L_aux + scaled_lp_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        x_hat = self.ae.decode(f)
        
        # Get additional diagnostics
        diagnostics = self.ae.get_diagnostics(x)
        
        return LossLog(
            x.to(dtype=original_dtype), 
            x_hat.to(dtype=original_dtype), 
            f.to(dtype=original_dtype),
            {
                'mse_loss': L_recon.item(),
                'aux_loss': L_aux.item(),
                'loss': total_loss.item(),
                'p': self.p,
                'next_p': self.next_p,
                'lp_loss': lp_loss.item(),
                'sparsity_loss': scaled_lp_loss.item(),
                'sparsity_coeff': self.sparsity_coeff,
                **{k: v.item() if torch.is_tensor(v) else v for k, v in diagnostics.items()}
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
        
        # Resample neurons if needed
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == self.training_config.resample_steps - 1):
            dead_threshold = self.training_config.resample_steps // 2
            deads = self.steps_since_active > dead_threshold
            self.resample_neurons(deads, activations)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'GatedAutoEncoder',
            'trainer_class': 'GatedAnnealTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'sparsity_function': self.model_config.sparsity_function,
            'initial_sparsity_penalty': self.model_config.initial_sparsity_penalty,
            'anneal_start': self.model_config.anneal_start,
            'anneal_end': self.model_config.anneal_end,
            'p_start': self.model_config.p_start,
            'p_end': self.model_config.p_end,
            'n_sparsity_updates': self.model_config.n_sparsity_updates,
            'sparsity_queue_length': self.model_config.sparsity_queue_length,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
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