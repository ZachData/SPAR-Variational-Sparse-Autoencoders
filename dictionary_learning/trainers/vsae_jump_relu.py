"""
Improved implementation of VSAEJumpReLU with better PyTorch practices.

This module combines variational learning from the VSAE approach
with JumpReLU activation function, following better PyTorch practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.autograd as autograd
from typing import Optional, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import dataclass
import math

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)


class RectangleFunction(autograd.Function):
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
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


@dataclass
class VSAEJumpReLUConfig:
    """Configuration for VSAEJumpReLU model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    bandwidth: float = 0.001  # Bandwidth parameter for JumpReLU
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.var_flag not in [0, 1]:
            raise ValueError("var_flag must be 0 (fixed variance) or 1 (learned variance)")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")


class VSAEJumpReLU(Dictionary, nn.Module):
    """
    A hybrid model combining VSAEIsoGaussian with JumpReLU activation.
    
    This model uses a variational approach for the encoder, combined with 
    a JumpReLU activation function to encourage sparsity in feature activations.
    
    Features:
    - Variational encoding with optional learned variance
    - JumpReLU activation for sparsity
    - Proper gradient handling and memory management
    - Configurable April update mode
    """

    def __init__(self, config: VSAEJumpReLUConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.bandwidth = config.bandwidth
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        self.encoder = nn.Linear(
            self.activation_dim,
            self.dict_size,
            bias=True,
            dtype=self.config.dtype,
            device=self.config.device
        )
        
        self.decoder = nn.Linear(
            self.dict_size,
            self.activation_dim,
            bias=self.use_april_update_mode,
            dtype=self.config.dtype,
            device=self.config.device
        )
        
        # Threshold parameters for JumpReLU
        self.threshold = nn.Parameter(
            torch.ones(self.dict_size, dtype=self.config.dtype, device=self.config.device) * 0.001
        )
        
        # Bias parameter for standard mode
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(
                torch.zeros(self.activation_dim, dtype=self.config.dtype, device=self.config.device)
            )
            
        # Variance encoder (only when learning variance)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim,
                self.dict_size,
                bias=True,
                dtype=self.config.dtype,
                device=self.config.device
            )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        # Encoder and decoder weights (tied initialization)
        w = torch.randn(
            self.activation_dim,
            self.dict_size,
            dtype=self.config.dtype,
            device=self.config.device
        )
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        # Set weights
        with torch.no_grad():
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            # Initialize biases
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
            
            # Initialize threshold
            nn.init.constant_(self.threshold, 0.001)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input activations to latent space with JumpReLU activation.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size]
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Encode mean
        if self.use_april_update_mode:
            pre_activation = self.encoder(x)
        else:
            pre_activation = self.encoder(x - self.bias)
        
        # Apply JumpReLU activation
        mu = JumpReLUFunction.apply(pre_activation, self.threshold, self.bandwidth)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            if self.use_april_update_mode:
                log_var_pre = self.var_encoder(x)
            else:
                log_var_pre = self.var_encoder(x - self.bias)
            
            # Apply JumpReLU to log_var as well for consistency
            log_var = JumpReLUFunction.apply(log_var_pre, self.threshold, self.bandwidth)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution."""
        if log_var is None:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Ensure output matches mu dtype
        return z.to(dtype=mu.dtype)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        # Ensure z matches decoder weight dtype
        z = z.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(z)
        else:
            return self.decoder(z) + self.bias
    
    def forward(self, x: torch.Tensor, output_features: bool = False) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            
        Returns:
            reconstruction: Reconstructed activations [batch_size, activation_dim]
            features: Latent features (if output_features=True) [batch_size, dict_size]
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
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
    
    def get_active_features_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Get boolean mask of which features are active across the batch."""
        return (features.sum(0) > 0)
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            self.threshold.mul_(scale)
            
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
        norms = torch.norm(self.decoder.weight, dim=0).to(
            dtype=self.decoder.weight.dtype, device=self.decoder.weight.device
        )

        if torch.allclose(norms, torch.ones_like(norms)):
            return
        
        print("Normalizing decoder weights")

        # Create test input on the same device as the model parameters
        device = self.decoder.weight.device
        test_input = torch.randn(10, self.activation_dim, device=device, dtype=self.config.dtype)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = torch.norm(self.decoder.weight, dim=0)
        assert torch.allclose(new_norms, torch.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms
        
        if self.var_flag == 1:
            self.var_encoder.weight.data *= norms[:, None]
            self.var_encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert torch.allclose(initial_output, new_output, atol=1e-4)
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[VSAEJumpReLUConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
    ) -> 'VSAEJumpReLU':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
        
        if config is None:
            # Try to reconstruct config from checkpoint
            
            # Extract dimensions
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle older format
                activation_dim, dict_size = state_dict["W_enc"].shape
                use_april_update_mode = "b_dec" in state_dict
            
            # Detect var_flag
            var_flag = 1 if ("var_encoder.weight" in state_dict or "W_enc_var" in state_dict) else 0
            
            # Extract bandwidth if available (default to 0.001)
            bandwidth = 0.001
            
            config = VSAEJumpReLUConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                bandwidth=bandwidth,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        
        # Handle legacy parameter names
        if 'W_enc' in state_dict:
            # Convert old parameter names to new ones
            converted_dict = {}
            converted_dict["encoder.weight"] = state_dict["W_enc"].T
            converted_dict["encoder.bias"] = state_dict["b_enc"]
            converted_dict["decoder.weight"] = state_dict["W_dec"].T
            
            if config.use_april_update_mode:
                converted_dict["decoder.bias"] = state_dict["b_dec"]
            else:
                converted_dict["bias"] = state_dict["b_dec"]
            
            if "threshold" in state_dict:
                converted_dict["threshold"] = state_dict["threshold"]
            else:
                converted_dict["threshold"] = torch.ones(config.dict_size) * 0.001
            
            if config.var_flag == 1 and "W_enc_var" in state_dict:
                converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
            
            state_dict = converted_dict
        
        # Filter state_dict to only include keys that are in the model
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Add missing threshold if not present
        if "threshold" not in filtered_state_dict:
            filtered_state_dict["threshold"] = torch.ones(config.dict_size, device=device) * 0.001
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Check for missing keys and initialize them
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if config.var_flag == 1 and "var_encoder.weight" in missing_keys:
                print("Initializing missing variance encoder parameters")
                with torch.no_grad():
                    nn.init.kaiming_uniform_(model.var_encoder.weight)
                    nn.init.zeros_(model.var_encoder.bias)
        
        # Normalize decoder if requested
        if normalize_decoder and not (config.var_flag == 1 and ("var_encoder.weight" in state_dict or "W_enc_var" in state_dict)):
            try:
                model.normalize_decoder()
            except (AssertionError, RuntimeError) as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
        
        return model


@dataclass
class TrainingConfig:
    """Configuration for training the VSAEJumpReLU."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    l0_coeff: float = 1.0
    target_l0: float = 20.0
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


class VSAEJumpReLUTrainer(SAETrainer):
    """
    Trainer for the VSAEJumpReLU model with improved architecture.
    
    Features:
    - Clean separation of concerns
    - Proper loss computation with KL divergence and L0 regularization
    - Memory-efficient processing
    - Configurable training parameters
    """
    
    def __init__(
        self,
        model_config: VSAEJumpReLUConfig = None,
        training_config: TrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility with trainSAE
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        l0_coeff: Optional[float] = None,
        target_l0: Optional[float] = None,
        var_flag: Optional[int] = None,
        bandwidth: Optional[float] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None or training_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                # Set defaults
                var_flag = var_flag or 1
                bandwidth = bandwidth or 0.001
                use_april_update_mode = use_april_update_mode if use_april_update_mode is not None else True
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = VSAEJumpReLUConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    var_flag=var_flag,
                    bandwidth=bandwidth,
                    use_april_update_mode=use_april_update_mode,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = TrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                    l0_coeff=l0_coeff or 1.0,
                    target_l0=target_l0 or 20.0,
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEJumpReLUTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAEJumpReLU(model_config)
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
        
        # Logging parameters
        self.logging_parameters = ["l0_loss", "kl_loss", "mse_loss", "effective_l0"]
        self.l0_loss = 0.0
        self.kl_loss = 0.0
        self.mse_loss = 0.0
        self.effective_l0 = 0.0
    
    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        log_var: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence loss with decoder norm weighting."""
        # Compute KL divergence
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)
            kl_base = 0.5 * torch.sum(
                torch.exp(log_var) + mu.pow(2) - 1 - log_var,
                dim=1
            )
        else:
            # Simplified KL for fixed variance: 0.5 * sum(mu^2)
            kl_base = 0.5 * torch.sum(mu.pow(2), dim=1)
        
        # Calculate decoder norms
        decoder_norms = torch.norm(self.ae.decoder.weight, p=2, dim=0)
        decoder_norms = decoder_norms.to(dtype=mu.dtype)
        
        # Weight by decoder norms (April update approach)
        kl_loss = torch.mean(kl_base) * torch.mean(decoder_norms)
        
        return kl_loss
    
    def _compute_l0_loss(self, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute L0 sparsity regularization loss."""
        # Compute L0 (number of active features)
        l0 = StepFunction.apply(mu, self.ae.threshold, self.ae.bandwidth).sum(dim=-1).mean()
        
        # L0 target regularization
        l0_loss = self.training_config.l0_coeff * ((l0 / self.training_config.target_l0) - 1).pow(2)
        
        return l0_loss, l0
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Forward pass
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure x_hat matches x dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss (MSE)
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        
        # L0 sparsity regularization
        l0_loss, l0 = self._compute_l0_loss(mu)
        
        # Ensure all components are in original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        kl_loss = kl_loss.to(dtype=original_dtype)
        l0_loss = l0_loss.to(dtype=original_dtype)
        
        # Total loss
        total_loss = (
            recon_loss +
            self.training_config.kl_coeff * sparsity_scale * kl_loss +
            sparsity_scale * l0_loss
        )
        
        # Update logging stats
        self.mse_loss = recon_loss.item()
        self.kl_loss = kl_loss.item()
        self.l0_loss = l0_loss.item()
        self.effective_l0 = l0.item()
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat, mu,
            {
                'l2_loss': torch.linalg.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'l0_loss': l0_loss.item(),
                'l0': l0.item(),
                'loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
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
        """Return configuration dictionary for logging/saving (JSON serializable)."""
        return {
            'dict_class': 'VSAEJumpReLU',
            'trainer_class': 'VSAEJumpReLUTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'bandwidth': self.model_config.bandwidth,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'l0_coeff': self.training_config.l0_coeff,
            'target_l0': self.training_config.target_l0,
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