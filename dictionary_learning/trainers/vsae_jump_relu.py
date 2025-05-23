"""
Improved implementation of Variational JumpReLU SAE with better PyTorch practices.

This module implements a JumpReLU-based variational autoencoder with proper
configuration management, type hints, and robust error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import namedtuple
from dataclasses import dataclass
import math

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)


@dataclass
class JumpReLUConfig:
    """Configuration for JumpReLU VSAE model."""
    activation_dim: int
    dict_size: int
    threshold: float = 0.001  # JumpReLU threshold parameter
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True  # Use April update with decoder bias
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")


class VSAEJumpReLU(Dictionary, nn.Module):
    """
    A Variational Sparse Autoencoder with JumpReLU activation and optional learned variance.
    
    Features:
    - JumpReLU activation function with learnable threshold
    - Variational approach with KL divergence regularization
    - Optional learned variance for more expressive latent space
    - Proper gradient handling and memory management
    - April update mode for better performance
    """

    def __init__(self, config: JumpReLUConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
        
        # Apply bias to input flag (for compatibility with older models)
        self.apply_b_dec_to_input = False
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        # Encoder
        self.W_enc = nn.Parameter(torch.empty(
            self.activation_dim, 
            self.dict_size,
            dtype=self.config.dtype,
            device=self.config.device
        ))
        self.b_enc = nn.Parameter(torch.zeros(
            self.dict_size,
            dtype=self.config.dtype,
            device=self.config.device
        ))
        
        # Decoder
        self.W_dec = nn.Parameter(torch.empty(
            self.dict_size, 
            self.activation_dim,
            dtype=self.config.dtype,
            device=self.config.device
        ))
        
        # Decoder bias (always present in improved version)
        self.b_dec = nn.Parameter(torch.zeros(
            self.activation_dim,
            dtype=self.config.dtype,
            device=self.config.device
        ))
        
        # JumpReLU threshold (learnable parameter)
        self.threshold = nn.Parameter(torch.full(
            (self.dict_size,),
            self.config.threshold,
            dtype=self.config.dtype,
            device=self.config.device
        ))
        
        # Variance encoder (only when learning variance)
        if self.var_flag == 1:
            self.W_enc_var = nn.Parameter(torch.empty(
                self.activation_dim,
                self.dict_size,
                dtype=self.config.dtype,
                device=self.config.device
            ))
            self.b_enc_var = nn.Parameter(torch.zeros(
                self.dict_size,
                dtype=self.config.dtype,
                device=self.config.device
            ))
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        # Initialize decoder weights as unit vectors
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec)
            self.W_dec.data = self.W_dec.data / self.W_dec.norm(dim=1, keepdim=True)
            
            # Initialize encoder weights to match decoder
            self.W_enc.data = self.W_dec.data.clone().T
            
            # Initialize biases to zero
            nn.init.zeros_(self.b_enc)
            nn.init.zeros_(self.b_dec)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.W_enc_var)
                nn.init.zeros_(self.b_enc_var)
    
    def jump_relu(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Apply JumpReLU activation: ReLU(x) if x > threshold, else 0.
        
        Args:
            x: Input tensor
            threshold: Threshold tensor (per-feature)
            
        Returns:
            JumpReLU activated tensor
        """
        return F.relu(x) * (x > threshold).float()
    
    def encode(self, x: torch.Tensor, output_pre_jump: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Encode input activations to latent space.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_pre_jump: Whether to return pre-activation values
            
        Returns:
            If output_pre_jump=False: encoded features
            If output_pre_jump=True: (features, pre_jump, log_var)
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.W_enc.dtype)
        
        # Apply decoder bias to input if needed (legacy compatibility)
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        
        # Encode mean
        pre_jump = x @ self.W_enc + self.b_enc
        mu = self.jump_relu(pre_jump, self.threshold)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            pre_jump_var = x @ self.W_enc_var + self.b_enc_var
            log_var = F.relu(pre_jump_var)  # Log variance should be non-negative
        
        if output_pre_jump:
            return mu, pre_jump, log_var
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution."""
        if log_var is None or self.var_flag == 0:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Ensure output matches mu dtype
        return z.to(dtype=mu.dtype)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        # Ensure z matches decoder weight dtype
        z = z.to(dtype=self.W_dec.dtype)
        return z @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor, output_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.b_enc.mul_(scale)
            self.b_dec.mul_(scale)
            self.threshold.mul_(scale)
            
            if self.var_flag == 1:
                self.b_enc_var.mul_(scale)
    
    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[JumpReLUConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        load_from_sae_lens: bool = False,
        **kwargs
    ) -> 'VSAEJumpReLU':
        """Load pretrained model from checkpoint."""
        if load_from_sae_lens:
            # Handle SAE-lens loading
            from sae_lens import SAE
            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            
            # Extract configuration
            activation_dim = cfg_dict["d_in"]
            dict_size = cfg_dict["d_sae"]
            
            config = JumpReLUConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=0,  # SAE-lens models don't have learned variance
                dtype=dtype,
                device=device
            )
            
            model = cls(config)
            model.load_state_dict(sae.state_dict())
            model.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]
            
        else:
            # Handle our format loading
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
            
            if config is None:
                # Try to reconstruct config from checkpoint
                activation_dim, dict_size = state_dict["W_enc"].shape
                
                # Detect var_flag
                var_flag = 1 if "W_enc_var" in state_dict else 0
                
                # Detect threshold value
                threshold = state_dict.get("threshold", torch.tensor(0.001)).mean().item()
                
                config = JumpReLUConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    threshold=threshold,
                    var_flag=var_flag,
                    dtype=dtype,
                    device=device
                )
            
            model = cls(config)
            
            # Handle parameter name mapping if needed
            model_state = model.state_dict()
            filtered_state = {}
            
            for key, value in state_dict.items():
                if key in model_state:
                    filtered_state[key] = value
                else:
                    print(f"Warning: Skipping parameter {key} not found in model")
            
            model.load_state_dict(filtered_state, strict=False)
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
            
        return model


@dataclass 
class JumpReLUTrainingConfig:
    """Configuration for training the VSAEJumpReLU."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    aux_weight: float = 0.1  # Weight for auxiliary reconstruction loss
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
    - Proper loss computation with KL divergence
    - Memory-efficient processing
    - Auxiliary loss for better convergence
    """
    
    def __init__(
        self,
        model_config: JumpReLUConfig = None,
        training_config: JumpReLUTrainingConfig = None,
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
        aux_weight: Optional[float] = None,
        threshold: Optional[float] = None,
        var_flag: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility - if individual parameters are passed, create configs
        if model_config is None or training_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                # Set defaults
                threshold = threshold or 0.001
                var_flag = var_flag or 0
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = JumpReLUConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    threshold=threshold,
                    var_flag=var_flag,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = JumpReLUTrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                    aux_weight=aux_weight or 0.1,
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
        self.logging_parameters = ["effective_l0", "threshold_mean"]
        self.effective_l0 = 0.0
        self.threshold_mean = 0.0
    
    def _compute_kl_loss(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence loss with decoder norm weighting."""
        # Get decoder norms
        decoder_norms = torch.norm(self.ae.W_dec, p=2, dim=1)
        decoder_norms = decoder_norms.to(dtype=mu.dtype)
        
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
        
        # Weight by decoder norms
        kl_loss = torch.mean(kl_base) * torch.mean(decoder_norms)
        
        return kl_loss
    
    def _compute_auxiliary_loss(
        self, 
        x: torch.Tensor,
        mu: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary reconstruction loss for better convergence."""
        # Create auxiliary reconstruction using just the encoded features
        x_aux = self.ae.decode(mu)
        
        # Compute auxiliary loss
        aux_loss = torch.mean(torch.sum((x - x_aux) ** 2, dim=1))
        
        return aux_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Forward pass - the model handles dtype conversion internally
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure all tensors are in the same dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Compute reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # Compute KL divergence loss
        kl_loss = self._compute_kl_loss(mu, log_var)
        
        # Compute auxiliary loss
        aux_loss = self._compute_auxiliary_loss(x, mu)
        
        # Total loss - ensure all components are in original dtype
        kl_loss = kl_loss.to(dtype=original_dtype)
        aux_loss = aux_loss.to(dtype=original_dtype)
        
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * sparsity_scale * kl_loss + 
            self.training_config.aux_weight * aux_loss
        )
        
        # Update logging stats
        self.effective_l0 = float(torch.sum(mu > 0).item()) / mu.numel()
        self.threshold_mean = float(torch.mean(self.ae.threshold).item())
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat, z,
            {
                'l2_loss': torch.sqrt(recon_loss).item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'aux_loss': aux_loss.item(),
                'total_loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'threshold_mean': self.threshold_mean,
                'effective_l0': self.effective_l0,
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
            'threshold': self.model_config.threshold,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'aux_weight': self.training_config.aux_weight,
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