"""
ROBUST implementation of Variational Sparse Autoencoder (VSAE) with GatedAnneal Architecture.

This module combines variational techniques with GatedAnneal networks to achieve
better feature learning and controlled sparsity, following the robust patterns
established in vsae_iso.py.

Key improvements applied:
1. Simplified configuration management with proper validation
2. Conservative numerical clamping and stability improvements
3. Clean separation of KL and sparsity scaling
4. Robust from_pretrained method with auto-detection
5. Simplified p-annealing without complex adaptation
6. Better error handling and dtype consistency
7. Memory-efficient defaults
8. Comprehensive diagnostics
9. Cleaner gating mechanism with soft gates
10. Simplified resampling logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam
)


@dataclass
class VSAEGatedAnnealConfig:
    """Configuration for VSAE Gated model with proper validation."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.activation_dim <= 0:
            raise ValueError("activation_dim must be positive")
        if self.dict_size <= 0:
            raise ValueError("dict_size must be positive")
        if self.var_flag not in [0, 1]:
            raise ValueError("var_flag must be 0 or 1")
        if self.log_var_init < -10 or self.log_var_init > 0:
            raise ValueError("log_var_init should be in range [-10, 0] for numerical stability")
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class VSAEGatedAnnealTrainingConfig:
    """Training configuration with proper defaults and validation."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 300.0  # Lower default for GatedAnneal models
    sparsity_coeff: float = 0.1  # Separate sparsity coefficient
    kl_warmup_steps: Optional[int] = None  # KL annealing
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # P-annealing parameters (simplified)
    p_start: float = 1.0
    p_end: float = 0.5
    anneal_start_frac: float = 0.2  # Start annealing at 20% of training
    anneal_end_frac: float = 0.8   # End annealing at 80% of training

    def __post_init__(self):
        """Set derived configuration values with validation."""
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.kl_coeff < 0:
            raise ValueError("kl_coeff must be non-negative")
        if self.sparsity_coeff < 0:
            raise ValueError("sparsity_coeff must be non-negative")
            
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)
            
        # Validate annealing fractions
        if not (0 <= self.anneal_start_frac < self.anneal_end_frac <= 1):
            raise ValueError("Must have 0 <= anneal_start_frac < anneal_end_frac <= 1")
        if not (0 <= self.p_end < self.p_start <= 2):
            raise ValueError("Must have 0 <= p_end < p_start <= 2")


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """Return KL annealing function to prevent posterior collapse."""
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps, "kl_warmup_steps must be > 0 and <= total_steps"
    
    def scale_fn(step: int) -> float:
        if step < kl_warmup_steps:
            return step / kl_warmup_steps
        else:
            return 1.0
    
    return scale_fn


def get_p_annealing_fn(
    total_steps: int, 
    p_start: float, 
    p_end: float, 
    anneal_start_frac: float, 
    anneal_end_frac: float
) -> Callable[[int], float]:
    """Return p-annealing function for sparsity control."""
    anneal_start = int(anneal_start_frac * total_steps)
    anneal_end = int(anneal_end_frac * total_steps)
    
    def p_fn(step: int) -> float:
        if step < anneal_start:
            return p_start
        elif step >= anneal_end:
            return p_end
        else:
            # Linear interpolation
            progress = (step - anneal_start) / (anneal_end - anneal_start)
            return p_start + progress * (p_end - p_start)
    
    return p_fn


class VSAEGatedAnneal(Dictionary, nn.Module):
    """
    Robust variational sparse autoencoder with GatedAnneal architecture.
    
    Combines VAE techniques with GatedAnneal networks for improved feature learning
    and interpretability through controlled sparsity.
    """

    def __init__(self, config: VSAEGatedAnnealConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers with proper configuration."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder/decoder (no bias for cleaner tied weights)
        self.encoder = nn.Linear(
            self.activation_dim, 
            self.dict_size, 
            bias=False,
            dtype=dtype,
            device=device
        )
        self.decoder = nn.Linear(
            self.dict_size, 
            self.activation_dim, 
            bias=False,
            dtype=dtype,
            device=device
        )
        
        # Bias parameters
        self.decoder_bias = nn.Parameter(
            torch.zeros(self.activation_dim, dtype=dtype, device=device)
        )
        
        # Gating parameters (simplified compared to original)
        self.gate_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=dtype, device=device)
        )
        self.mag_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=dtype, device=device)
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
        
        with torch.no_grad():
            # Tied initialization for encoder and decoder
            w = torch.randn(
                self.activation_dim,
                self.dict_size,
                dtype=dtype,
                device=device
            )
            w = w / w.norm(dim=0, keepdim=True) * 0.1
            
            # Set encoder and decoder weights (tied)
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Biases already initialized to zero
            
            # Initialize variance encoder if needed
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction."""
        x = x.to(dtype=self.encoder.weight.dtype)
        return x - self.decoder_bias

    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode input activations to latent space with gating.
        
        Args:
            x: Input activations
            return_gate: Whether to return gate values
            
        Returns:
            Mean (and optionally log_var), and optionally gate values
        """
        x_processed = self._preprocess_input(x)
        
        # Main encoder output
        x_enc = self.encoder(x_processed)

        # Simplified gating mechanism
        gate_logits = x_enc + self.gate_bias
        gate = torch.sigmoid(gate_logits)  # Soft gating
        
        # Magnitude with gating
        magnitude = F.relu(x_enc + self.mag_bias)
        
        # Combined features (this becomes our mean μ)
        mu = gate * magnitude
        
        # Handle variance if required
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
            
        # Return appropriate combination
        if self.var_flag == 1 and return_gate:
            return mu, log_var, gate
        elif self.var_flag == 1:
            return mu, log_var
        elif return_gate:
            return mu, gate
        else:
            return mu

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply reparameterization trick with conservative clamping."""
        if log_var is None or self.var_flag == 0:
            return mu
            
        # Conservative clamping range
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Standard reparameterization
        std = torch.sqrt(torch.exp(log_var_clamped))
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode features back to activation space."""
        f = f.to(dtype=self.decoder.weight.dtype)
        return self.decoder(f) + self.decoder_bias

    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False, 
        ghost_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEGatedAnneal")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode
        if self.var_flag == 1:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
        else:
            z = self.encode(x)
            
        # Decode
        x_hat = self.decode(z)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            z = z.to(dtype=original_dtype)
            return x_hat, z
        return x_hat

    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed KL diagnostics for monitoring."""
        with torch.no_grad():
            if self.var_flag == 1:
                mu, log_var = self.encode(x)
                
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)
                
                kl_mu = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                kl_total = kl_mu + kl_var
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_mu,
                    'kl_var_term': kl_var,
                    'mean_log_var': log_var.mean(),
                    'mean_var': var.mean(),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }
            else:
                mu = self.encode(x)
                kl_total = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }

    def get_gating_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get gating-specific diagnostics."""
        with torch.no_grad():
            if self.var_flag == 1:
                mu, log_var, gate = self.encode(x, return_gate=True)
            else:
                mu, gate = self.encode(x, return_gate=True)
            
            return {
                'mean_gate_activation': gate.mean(),
                'gate_sparsity': (gate < 0.1).float().mean(),  # Fraction nearly off
                'gate_std': gate.std(),
                'active_features': (gate > 0.1).float().sum(dim=-1).mean(),  # Features per sample
            }

    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.decoder_bias.mul_(scale)
            self.gate_bias.mul_(scale)
            self.mag_bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)

    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
        if self.var_flag == 1:
            print("Warning: Normalizing decoder weights with learned variance may hurt performance")
        
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0)
            
            if torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
                return
            
            print("Normalizing decoder weights")
            
            # Test that normalization preserves output
            device = self.decoder.weight.device
            test_input = torch.randn(10, self.activation_dim, device=device, dtype=self.decoder.weight.dtype)
            initial_output = self(test_input)
            
            # Normalize decoder weights
            self.decoder.weight.div_(norms)
            
            # Scale encoder weights and biases accordingly
            self.encoder.weight.mul_(norms.unsqueeze(1))
            self.gate_bias.mul_(norms)
            self.mag_bias.mul_(norms)
            
            # Verify normalization worked
            new_norms = torch.norm(self.decoder.weight, dim=0)
            assert torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-6)
            
            # Verify output is preserved
            new_output = self(test_input)
            assert torch.allclose(initial_output, new_output, atol=1e-4), "Normalization changed model output"

    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[VSAEGatedAnnealConfig] = None,
        dtype: torch.dtype = torch.float32, 
        device: Optional[torch.device] = None, 
        normalize_decoder: bool = True, 
        var_flag: Optional[int] = None
    ) -> 'VSAEGatedAnneal':
        """Load a pretrained autoencoder from a file."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if var_flag is None:
                has_var_encoder = "var_encoder.weight" in state_dict
                var_flag = 1 if has_var_encoder else 0
            
            # Determine dimensions from state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder_bias" in state_dict
            else:
                # Handle potential legacy formats
                raise ValueError("Could not auto-detect model dimensions from state_dict")
                
            config = VSAEGatedAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
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
                # Initialize missing variance encoder if needed
                if var_flag == 1 and any("var_encoder" in key for key in missing_keys):
                    print("Initializing missing variance encoder parameters")
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(model.var_encoder.weight, a=0.01)
                        nn.init.constant_(model.var_encoder.bias, config.log_var_init)
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Normalize decoder if requested
        if normalize_decoder and not (config.var_flag == 1):
            try:
                model.normalize_decoder()
            except Exception as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model


class VSAEGatedAnnealTrainer(SAETrainer):
    """
    Robust trainer for VSAE with GatedAnneal architecture and p-annealing.
    
    Simplified compared to the original with better stability and cleaner logic.
    """
    
    def __init__(
        self,
        model_config: Optional[VSAEGatedAnnealConfig] = None,
        training_config: Optional[VSAEGatedAnnealTrainingConfig] = None,
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
        kl_coeff: Optional[float] = None,
        var_flag: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = VSAEGatedAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag or 0,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = VSAEGatedAnnealTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=kl_coeff or 300.0,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedAnnealTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEGatedAnneal(model_config)
        self.ae.to(self.device)
        
        # Initialize p-annealing function
        self.p_annealing_fn = get_p_annealing_fn(
            training_config.steps,
            training_config.p_start,
            training_config.p_end,
            training_config.anneal_start_frac,
            training_config.anneal_end_frac
        )
        
        # Initialize dead feature tracking
        if training_config.resample_steps is not None:
            self.steps_since_active = torch.zeros(
                model_config.dict_size, dtype=torch.long, device=self.device
            )
        else:
            self.steps_since_active = None
        
        # Initialize optimizer and scheduler
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(), 
            self.ae.decoder.parameters(), 
            lr=training_config.lr,
            betas=(0.9, 0.999)
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
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = ['current_p', 'kl_loss', 'sparsity_loss', 'kl_scale', 'sparsity_scale']
        self.current_p = training_config.p_start
        self.kl_loss = None
        self.sparsity_loss = None
        self.kl_scale = 1.0
        self.sparsity_scale = 1.0
        
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute KL divergence loss with proper VAE formulation."""
        if self.ae.var_flag == 1 and log_var is not None:
            # Conservative clamping
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # KL divergence
            kl_per_sample = 0.5 * torch.sum(
                mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                dim=1
            )
        else:
            # Fixed variance case
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
        
        kl_loss = kl_per_sample.mean()
        return torch.clamp(kl_loss, min=0.0)
    
    def _compute_sparsity_loss(self, mu: torch.Tensor, p: float) -> torch.Tensor:
        """Compute p-norm sparsity loss with numerical stability."""
        p_clamped = max(0.01, min(p, 2.0))  # Conservative range
        mu_abs = torch.clamp(torch.abs(mu), min=1e-8, max=1e6)
        
        # Simple Lp norm
        sparsity_loss = mu_abs.pow(p_clamped).sum(dim=-1).mean()
        return torch.clamp(sparsity_loss, min=0.0, max=1e6)
    
    def _resample_neurons(self, dead_mask: torch.Tensor, activations: torch.Tensor) -> None:
        """Simplified resampling for dead neurons."""
        with torch.no_grad():
            if dead_mask.sum() == 0: 
                return
                
            print(f"Resampling {dead_mask.sum().item()} neurons")

            # Compute reconstruction errors
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample vectors for resampling
            n_resample = min([dead_mask.sum(), losses.shape[0]])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Get norm of living neurons
            alive_norm = self.ae.encoder.weight[~dead_mask].norm(dim=-1).mean()
            
            # Resample weights
            self.ae.encoder.weight[dead_mask][:n_resample] = sampled_vecs * alive_norm * 0.2
            normalized_vecs = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.decoder.weight[:,dead_mask][:,:n_resample] = normalized_vecs.to(dtype=self.ae.decoder.weight.dtype)
            
            # Reset biases
            self.ae.gate_bias[dead_mask][:n_resample] = 0.
            self.ae.mag_bias[dead_mask][:n_resample] = 0.

            # Reset optimizer state (simplified)
            state_dict = self.optimizer.state_dict()['state']
            for param_state in state_dict.values():
                if 'exp_avg' in param_state and param_state['exp_avg'].shape == dead_mask.shape:
                    param_state['exp_avg'][dead_mask] = 0.
                    param_state['exp_avg_sq'][dead_mask] = 0.
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Calculate loss with clean separation of components."""
        # Get scaling factors
        sparsity_scale = self.sparsity_warmup_fn(step)
        kl_scale = self.kl_warmup_fn(step)
        current_p = self.p_annealing_fn(step)
        
        # Store for logging
        self.kl_scale = kl_scale
        self.sparsity_scale = sparsity_scale
        self.current_p = current_p
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        if self.model_config.var_flag == 1:
            mu, log_var, gate = self.ae.encode(x, return_gate=True)
            z = self.ae.reparameterize(mu, log_var)
        else:
            mu, gate = self.ae.encode(x, return_gate=True)
            z = mu
            log_var = None
            
        x_hat = self.ae.decode(z)
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        self.kl_loss = kl_loss
        
        # Sparsity loss with p-annealing
        sparsity_loss = self._compute_sparsity_loss(mu, current_p)
        self.sparsity_loss = sparsity_loss
        
        # Update dead feature tracking based on gate activity
        if self.steps_since_active is not None:
            active = (gate > 0.1).any(dim=0)  # Features active in any sample
            self.steps_since_active[~active] += 1
            self.steps_since_active[active] = 0
        
        # Total loss with separate scaling
        scaled_kl_loss = self.training_config.kl_coeff * kl_scale * kl_loss
        scaled_sparsity_loss = self.training_config.sparsity_coeff * sparsity_scale * sparsity_loss
        
        total_loss = recon_loss + scaled_kl_loss + scaled_sparsity_loss
        
        if not logging:
            return total_loss
        else:
            # Get diagnostics
            kl_diagnostics = self.ae.get_kl_diagnostics(x)
            gating_diagnostics = self.ae.get_gating_diagnostics(x)
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'sparsity_loss': sparsity_loss.item(),
                    'scaled_kl_loss': scaled_kl_loss.item(),
                    'scaled_sparsity_loss': scaled_sparsity_loss.item(),
                    'loss': total_loss.item(),
                    'current_p': current_p,
                    'kl_scale': kl_scale,
                    'sparsity_scale': sparsity_scale,
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()},
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in gating_diagnostics.items()}
                }
            )
        
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Update model parameters for one step."""
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()

        # Resample dead neurons if needed
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == self.training_config.resample_steps - 1 and
            self.steps_since_active is not None):
            
            dead_mask = self.steps_since_active > (self.training_config.resample_steps // 2)
            if dead_mask.any():
                self._resample_neurons(dead_mask, activations)

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'VSAEGatedAnneal',
            'trainer_class': 'VSAEGatedAnnealTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'sparsity_coeff': self.training_config.sparsity_coeff,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'resample_steps': self.training_config.resample_steps,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'p_start': self.training_config.p_start,
            'p_end': self.training_config.p_end,
            'anneal_start_frac': self.training_config.anneal_start_frac,
            'anneal_end_frac': self.training_config.anneal_end_frac,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }