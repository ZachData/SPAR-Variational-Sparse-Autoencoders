"""
VSAEIso implementation
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
)


@dataclass
class VSAEIsoConfig:
    """Configuration for VSAEIso model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class VSAEIsoGaussian(Dictionary, nn.Module):
    """
    improvements:
    - Unconstrained mean encoding (no ReLU)
    - Consistent log_var = log(σ²) interpretation throughout
    - Conservative clamping ranges
    - Proper separation of KL and sparsity scaling
    - Clean architecture without unnecessary complications
    """

    def __init__(self, config: VSAEIsoConfig):
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
        
        # Main encoder and decoder
        self.encoder = nn.Linear(
            self.activation_dim,
            self.dict_size,
            bias=True,
            dtype=dtype,
            device=device
        )
        
        self.decoder = nn.Linear(
            self.dict_size,
            self.activation_dim,
            bias=self.use_april_update_mode,
            dtype=dtype,
            device=device
        )
        
        # Bias parameter for standard mode
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.activation_dim,
                    dtype=dtype,
                    device=device
                )
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
            # REMOVED: No layer norm - let model learn input-dependent variance naturally
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Tied initialization for encoder and decoder
        w = torch.randn(
            self.activation_dim,
            self.dict_size,
            dtype=dtype,
            device=device
        )
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        with torch.no_grad():
            # Set encoder and decoder weights (tied)
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Initialize biases
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                # Initialize variance encoder weights
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                
                # Initialize log_var bias to reasonable value
                # log_var = log(σ²), so log_var = -2 means σ² ≈ 0.135
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction in standard mode."""
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent space.
        
        FIXED: No ReLU on mean - VAE means should be unconstrained for expressiveness
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size] (unconstrained)
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        x_processed = self._preprocess_input(x)
        
        # FIXED: No ReLU constraint on mean - let it be unconstrained
        mu = self.encoder(x_processed)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            # FIXED: No layer norm - direct encoding of log variance
            log_var = self.var_encoder(x_processed)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply reparameterization trick with consistent log_var = log(σ²) interpretation.
        
        FIXED: Consistent interpretation where log_var = log(σ²) throughout
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # FIXED: More conservative clamping range
        # log_var ∈ [-6, 2] means σ² ∈ [0.002, 7.4] - reasonable range
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var)) = sqrt(σ²)
        std = torch.sqrt(torch.exp(log_var_clamped))
        
        # Sample noise
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.
        
        Args:
            z: Latent features [batch_size, dict_size]
            
        Returns:
            x_hat: Reconstructed activations [batch_size, activation_dim]
        """
        # Ensure z matches decoder weight dtype
        z = z.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(z)
        else:
            return self.decoder(z) + self.bias
    
    def forward(self, x: torch.Tensor, output_features: bool = False, ghost_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for VSAE (raises error if provided)
            
        Returns:
            x_hat: Reconstructed activations
            z: Latent features (if output_features=True)
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEIsoGaussian")
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(z)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            z = z.to(dtype=original_dtype)
            return x_hat, z
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed KL diagnostics for monitoring training.
        
        Returns:
            Dictionary with KL components and statistics
        """
        with torch.no_grad():
            mu, log_var = self.encode(x)
            
            if self.var_flag == 1 and log_var is not None:
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)  # σ²
                
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
                kl_total = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                }
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    def normalize_decoder(self) -> None:
        """
        Normalize decoder weights to have unit norm.
        Note: Only recommended for models without learned variance.
        """
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
            self.encoder.bias.mul_(norms)
            
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
        config: Optional[VSAEIsoConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None
    ) -> 'VSAEIsoGaussian':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Override var_flag detection
            
        Returns:
            Loaded autoencoder
        """
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Auto-detect configuration from state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle legacy format
                activation_dim, dict_size = state_dict.get("W_enc", state_dict["encoder.weight"].T).shape
                use_april_update_mode = "b_dec" in state_dict or "decoder.bias" in state_dict
            
            # Auto-detect var_flag
            if var_flag is None:
                var_flag = 1 if ("var_encoder.weight" in state_dict or "W_enc_var" in state_dict) else 0
            
            config = VSAEIsoConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        # Create model
        model = cls(config)
        
        # Handle legacy parameter names
        if "W_enc" in state_dict:
            converted_dict = cls._convert_legacy_state_dict(state_dict, config)
            state_dict = converted_dict
        
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
        
        # Normalize decoder if requested (skip for learned variance models)
        if normalize_decoder and not (config.var_flag == 1 and "var_encoder.weight" in state_dict):
            try:
                model.normalize_decoder()
            except Exception as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model
    
    @staticmethod
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: VSAEIsoConfig) -> Dict[str, torch.Tensor]:
        """Convert legacy parameter names to current format."""
        converted = {}
        
        # Convert main parameters
        converted["encoder.weight"] = state_dict["W_enc"].T
        converted["encoder.bias"] = state_dict["b_enc"]
        converted["decoder.weight"] = state_dict["W_dec"].T
        
        if config.use_april_update_mode:
            converted["decoder.bias"] = state_dict["b_dec"]
        else:
            converted["bias"] = state_dict["b_dec"]
        
        # Convert variance encoder if present
        if config.var_flag == 1 and "W_enc_var" in state_dict:
            converted["var_encoder.weight"] = state_dict["W_enc_var"].T
            converted["var_encoder.bias"] = state_dict["b_enc_var"]
        
        return converted


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Return a function that computes KL annealing scale factor at a given step.
    Helps prevent posterior collapse in early training.
    
    Args:
        total_steps: Total training steps
        kl_warmup_steps: Steps to warm up KL coefficient from 0 to 1
        
    Returns:
        Function that returns KL scale factor for given step
    """
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps, "kl_warmup_steps must be > 0 and <= total_steps"
    
    def scale_fn(step: int) -> float:
        if step < kl_warmup_steps:
            return step / kl_warmup_steps  # Linear warmup from 0 to 1
        else:
            return 1.0
    
    return scale_fn


@dataclass
class VSAEIsoTrainingConfig:
    """Enhanced training configuration with proper scaling separation."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None  # For any actual sparsity penalties
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        # KL annealing to prevent posterior collapse
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)  # 10% of training

        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


class VSAEIsoTrainer(SAETrainer):
    """
    Actually Fixed trainer for VSAEIso with proper scaling separation.
    
    Key improvements:
    - Correct KL divergence computation (was already correct!)
    - FIXED: Separate KL annealing from sparsity scaling
    - Better numerical stability
    - Enhanced logging and diagnostics
    """
    
    def __init__(
        self,
        model_config: Optional[VSAEIsoConfig] = None,
        training_config: Optional[VSAEIsoTrainingConfig] = None,
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
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use VSAEIsoGaussian
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = VSAEIsoConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag or 0,
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else True,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = VSAEIsoTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=kl_coeff or 500.0,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEIsoTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEIsoGaussian(model_config)
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
        # KL annealing function (separate from sparsity!)
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
    
    def _compute_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence loss 
        
        For q(z|x) = N(μ, σ²) and p(z) = N(0, I):
        KL[q || p] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # FIXED: Conservative clamping range
            log_var_clamped = torch.clamp(log_var, min=-8.0, max=4.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # KL divergence (term order doesn't matter due to commutativity)
            kl_per_sample = 0.5 * torch.sum(
                mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                dim=1
            )
        else:
            # Fixed variance case: KL = 0.5 * ||μ||²
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
        
        # Average over batch
        kl_loss = kl_per_sample.mean()
        
        # Ensure KL is non-negative (should be true mathematically)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        return kl_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with proper scaling separation."""
        sparsity_scale = self.sparsity_warmup_fn(step)  # For any L1 penalties
        kl_scale = self.kl_warmup_fn(step)  # FIXED: Separate KL annealing
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # FIXED: Separate scaling - KL gets kl_scale, sparsity would get sparsity_scale
        total_loss = recon_loss + self.training_config.kl_coeff * kl_scale * kl_loss
        
        # If we had L1 sparsity penalty, it would use sparsity_scale:
        # l1_loss = torch.mean(torch.abs(z))
        # total_loss += l1_coeff * sparsity_scale * l1_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information with diagnostics
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        kl_diagnostics = self.ae.get_kl_diagnostics(x)
        
        return LossLog(
            x, x_hat, z,
            {
                'l2_loss': torch.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'kl_scale': kl_scale,  # Separate from sparsity scaling
                # Additional diagnostics
                **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
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
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'VSAEIsoGaussian',
            'trainer_class': 'VSAEIsoTrainer',
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
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
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