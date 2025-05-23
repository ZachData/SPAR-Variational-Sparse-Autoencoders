"""
Improved implementation of VSAEGated with better practices and architecture.

This module combines gated networks with variational learning, following 
better PyTorch practices and improved software architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam,
)


@dataclass
class GatedVSAEConfig:
    """Configuration for Gated VSAE model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


@dataclass
class GatedTrainingConfig:
    """Configuration for training the Gated VSAE."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    l1_penalty: float = 0.1
    aux_weight: float = 0.1
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    use_constrained_optimizer: bool = True
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


class VSAEGated(Dictionary, nn.Module):
    """
    A gated variational autoencoder with improved architecture.
    
    Features:
    - Gating network for feature selection
    - Variational component for magnitude modeling
    - Proper gradient handling and memory management
    - Clean separation of gating and magnitude networks
    """

    def __init__(self, config: GatedVSAEConfig):
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
        """Initialize neural network layers."""
        # Shared decoder bias
        self.decoder_bias = nn.Parameter(
            torch.zeros(self.activation_dim, dtype=self.config.dtype, device=self.config.device)
        )
        
        # Shared encoder for processing input
        self.encoder = nn.Linear(
            self.activation_dim,
            self.dict_size,
            bias=True,
            dtype=self.config.dtype,
            device=self.config.device
        )
        
        # Gating network parameters
        self.gate_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        
        # Magnitude network parameters
        self.r_mag = nn.Parameter(
            torch.zeros(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        self.mag_bias = nn.Parameter(
            torch.zeros(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        
        # Shared decoder
        self.decoder = nn.Linear(
            self.dict_size,
            self.activation_dim,
            bias=False,
            dtype=self.config.dtype,
            device=self.config.device
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
        # Initialize decoder weights as random unit vectors
        with torch.no_grad():
            dec_weight = torch.randn_like(self.decoder.weight)
            dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
            self.decoder.weight.copy_(dec_weight)
            
            # Initialize encoder weights to match decoder (transposed)
            self.encoder.weight.copy_(dec_weight.T)
            
            # Initialize all biases to zero
            nn.init.zeros_(self.decoder_bias)
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.gate_bias)
            nn.init.zeros_(self.mag_bias)
            nn.init.zeros_(self.r_mag)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False, 
        return_log_var: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode input tensor to get latent features.
        
        Args:
            x: Input tensor [batch_size, activation_dim]
            return_gate: If True, return gating values
            return_log_var: If True, return log variance values
            
        Returns:
            Tuple containing features and optionally gate values and log variance
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Compute shared encoding
        x_enc = self.encoder(x - self.decoder_bias)
        
        # Gating network
        pi_gate = x_enc + self.gate_bias
        gate = (pi_gate > 0).to(dtype=x_enc.dtype)  # Binary gate
        f_gate = F.relu(pi_gate)  # Continuous for gradients
        
        # Magnitude network
        pi_mag = torch.exp(self.r_mag) * x_enc + self.mag_bias
        mu_mag = F.relu(pi_mag)  # Mean magnitude
        
        # Variational component
        log_var = None
        if self.var_flag == 1 and return_log_var:
            log_var = F.relu(self.var_encoder(x - self.decoder_bias))
            # Apply reparameterization trick
            mag = self.reparameterize(mu_mag, log_var)
        else:
            mag = mu_mag
        
        # Combine gating and magnitude
        f = gate * mag
        
        # Return requested outputs
        outputs = [f]
        if return_gate:
            outputs.append(f_gate)
        if return_log_var:
            outputs.append(log_var)
        
        return tuple(outputs) if len(outputs) > 1 else f
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for variational sampling."""
        if log_var is None:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        return self.decoder(f) + self.decoder_bias
    
    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch_size, activation_dim]
            output_features: Whether to return features along with reconstruction
            
        Returns:
            Reconstruction or tuple of (reconstruction, features)
        """
        # Store original dtype
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Forward pass
        if self.var_flag == 1:
            f, f_gate, log_var = self.encode(x, return_gate=True, return_log_var=True)
        else:
            f, f_gate = self.encode(x, return_gate=True)
        
        x_hat = self.decode(f)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        return x_hat
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.decoder_bias.mul_(scale)
            self.encoder.bias.mul_(scale)
            self.gate_bias.mul_(scale)
            self.mag_bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[GatedVSAEConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True
    ) -> 'VSAEGated':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
        
        if config is None:
            # Auto-detect configuration from state_dict
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            var_flag = 1 if "var_encoder.weight" in state_dict else 0
            
            config = GatedVSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        
        # Filter state_dict to only include existing keys
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Handle missing keys
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
        
        # Normalize decoder if requested
        if normalize_decoder:
            with torch.no_grad():
                norms = torch.norm(model.decoder.weight, dim=0)
                if not torch.allclose(norms, torch.ones_like(norms)):
                    print("Normalizing decoder weights")
                    model.decoder.weight.div_(norms.unsqueeze(0))
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
        
        return model


class VSAEGatedTrainer(SAETrainer):
    """
    Improved trainer for the VSAEGated model with better architecture.
    
    Features:
    - Clean separation of concerns
    - Proper loss computation with multiple components
    - Memory-efficient processing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model_config: GatedVSAEConfig = None,
        training_config: GatedTrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility parameters
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        l1_penalty: Optional[float] = None,
        aux_weight: Optional[float] = None,
        var_flag: Optional[int] = None,
        use_constrained_optimizer: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None or training_config is None:
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                device_obj = torch.device(device) if device else (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                )
                
                model_config = GatedVSAEConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    var_flag=var_flag or 1,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = GatedTrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                    l1_penalty=l1_penalty or 0.1,
                    aux_weight=aux_weight or 0.1,
                    use_constrained_optimizer=use_constrained_optimizer or True,
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAEGated(model_config)
        self.ae.to(self.device)
        
        # Initialize optimizer
        if training_config.use_constrained_optimizer:
            self.optimizer = ConstrainedAdam(
                self.ae.parameters(),
                self.ae.decoder.parameters(),
                lr=training_config.lr,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.ae.parameters(),
                lr=training_config.lr,
                betas=(0.9, 0.999)
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
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "gate_sparsity", "aux_loss_raw"]
        self.effective_l0 = 0.0
        self.gate_sparsity = 0.0
        self.aux_loss_raw = 0.0
    
    def _compute_kl_loss(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor],
        f_gate: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence loss with decoder norm weighting."""
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for learned variance
            kl_base = -0.5 * torch.sum(
                1 + log_var - f_gate.pow(2) - log_var.exp(), 
                dim=1
            )
        else:
            # Simplified KL for fixed variance
            kl_base = 0.5 * torch.sum(f_gate.pow(2), dim=1)
        
        # Weight by decoder norms
        decoder_norms = torch.norm(self.ae.decoder.weight, p=2, dim=0)
        decoder_norms = decoder_norms.to(dtype=kl_base.dtype)
        
        kl_loss = torch.mean(kl_base) * torch.mean(decoder_norms)
        return kl_loss
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)
        
        # Forward pass
        if self.ae.var_flag == 1:
            f, f_gate, log_var = self.ae.encode(x, return_gate=True, return_log_var=True)
        else:
            f, f_gate = self.ae.encode(x, return_gate=True)
            log_var = None
        
        # Main reconstruction
        x_hat = self.ae.decode(f)
        
        # Auxiliary reconstruction from gate network only
        x_hat_gate = self.ae.decoder(f_gate) + self.ae.decoder_bias
        
        # Compute losses
        # 1. Main reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # 2. Sparsity loss for gating network
        gate_sparsity_loss = torch.mean(torch.sum(torch.abs(f_gate), dim=1))
        
        # 3. Auxiliary reconstruction loss
        aux_loss = torch.mean(torch.sum((x - x_hat_gate) ** 2, dim=1))
        
        # 4. KL divergence loss
        kl_loss = self._compute_kl_loss(f, log_var, f_gate)
        
        # Convert all losses to original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        gate_sparsity_loss = gate_sparsity_loss.to(dtype=original_dtype)
        aux_loss = aux_loss.to(dtype=original_dtype)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Combine losses
        total_loss = (
            recon_loss +
            (self.training_config.l1_penalty * sparsity_scale * gate_sparsity_loss) +
            (self.training_config.aux_weight * aux_loss) +
            (self.training_config.kl_coeff * sparsity_scale * kl_loss)
        )
        
        # Update logging stats
        self.effective_l0 = float(torch.sum(f > 0).item()) / f.numel()
        self.gate_sparsity = float(torch.sum(f_gate > 0).item()) / f_gate.numel()
        self.aux_loss_raw = aux_loss.item()
        
        if not logging:
            return total_loss
        
        # Return detailed loss information
        x_hat = x_hat.to(dtype=original_dtype)
        f = f.to(dtype=original_dtype)
        
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat, f,
            {
                'l2_loss': torch.sqrt(recon_loss).item(),
                'mse_loss': recon_loss.item(),
                'gate_sparsity_loss': gate_sparsity_loss.item(),
                'aux_loss': aux_loss.item(),
                'kl_loss': kl_loss.item(),
                'total_loss': total_loss.item(),
                'effective_l0': self.effective_l0,
                'gate_sparsity': self.gate_sparsity,
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
        """Return configuration dictionary for logging/saving."""
        return {
            'dict_class': 'VSAEGated',
            'trainer_class': 'VSAEGatedTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'l1_penalty': self.training_config.l1_penalty,
            'aux_weight': self.training_config.aux_weight,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'use_constrained_optimizer': self.training_config.use_constrained_optimizer,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }