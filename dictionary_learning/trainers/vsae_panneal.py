"""
Improved implementation of variational autoencoder (VSAE) training scheme with p-annealing.

This module combines the benefits of variational autoencoders with p-norm annealing
to provide more control over sparsity and potentially discover better features.
Follows improved PyTorch practices and better error handling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import dataclass
import math

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG
from ..dictionary import Dictionary


@dataclass
class PAnnealConfig:
    """Configuration for P-Annealing VSAE model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


@dataclass 
class PAnnealTrainingConfig:
    """Configuration for training the P-Annealing VSAE."""
    steps: int
    lr: float = 5e-4
    sparsity_penalty: float = 500.0
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # P-annealing specific parameters
    sparsity_function: str = 'Lp'  # 'Lp' or 'Lp^p'
    anneal_start: Optional[int] = None
    anneal_end: Optional[int] = None
    p_start: float = 1.0
    p_end: float = 0.5
    n_sparsity_updates: int = 10
    sparsity_queue_length: int = 10
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)
        if self.anneal_start is None:
            self.anneal_start = max(self.warmup_steps, int(0.15 * self.steps))
        if self.anneal_end is None:
            self.anneal_end = self.steps - 1


class VSAEPAnneal(Dictionary, nn.Module):
    """
    A one-layer variational autoencoder with p-annealing for sparsity control.
    
    Combines the VAE approach with p-annealing technique to provide flexible 
    control over the sparsity of the latent representation.
    """

    def __init__(self, config: PAnnealConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.use_april_update_mode = config.use_april_update_mode
        self.var_flag = config.var_flag
        
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
        
        # In standard mode, use separate bias parameter
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(torch.zeros(
                self.activation_dim, 
                dtype=self.config.dtype,
                device=self.config.device
            ))
            
        # Variance network (only used when var_flag=1)
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
        # Initialize encoder and decoder weights (tied initialization)
        w = torch.randn(
            self.activation_dim, 
            self.dict_size, 
            dtype=self.config.dtype,
            device=self.config.device
        )
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        with torch.no_grad():
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
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input activations to latent space.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size]
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            mu = F.relu(self.encoder(x))
        else:
            mu = F.relu(self.encoder(x - self.bias))
            
        log_var = None
        if self.var_flag == 1:
            if self.use_april_update_mode:
                log_var = F.relu(self.var_encoder(x))
            else:
                log_var = F.relu(self.var_encoder(x - self.bias))
                
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution."""
        if log_var is None:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstruction."""
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False, **kwargs) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            
        Returns:
            reconstruction: Reconstructed activations [batch_size, activation_dim]
            features: Latent features (if output_features=True) [batch_size, dict_size]
        """
        # Store original dtype
        original_dtype = x.dtype
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        f = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.decode(f)
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            if self.var_flag == 1:
                # Return extra info for p-annealing trainer
                return x_hat, f, mu.to(dtype=original_dtype), log_var.to(dtype=original_dtype) if log_var is not None else None
            else:
                return x_hat, f
        return x_hat

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
        """Normalize decoder weights to have unit norm."""
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0)
            
            if torch.allclose(norms, torch.ones_like(norms)):
                return
                
            print("Normalizing decoder weights")
            
            # Create test input on the same device
            device = self.decoder.weight.device
            test_input = torch.randn(10, self.activation_dim, device=device, dtype=self.decoder.weight.dtype)
            initial_output = self(test_input)
            
            # Normalize decoder
            self.decoder.weight.div_(norms)
            
            # Adjust encoder accordingly
            self.encoder.weight.mul_(norms[:, None])
            self.encoder.bias.mul_(norms)
            
            # Verify normalization worked
            new_output = self(test_input)
            assert torch.allclose(initial_output, new_output, atol=1e-4)

    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[PAnnealConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None
    ) -> 'VSAEPAnneal':
        """Load pretrained model from checkpoint."""
        state_dict = torch.load(path, map_location=device)
        
        if config is None:
            # Auto-detect configuration from state dict
            if var_flag is None:
                var_flag = 1 if "var_encoder.weight" in state_dict else 0
            
            # Determine dimensions and mode
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle legacy format
                activation_dim, dict_size = state_dict.get("W_enc", state_dict["encoder.weight"].T).shape
                use_april_update_mode = "b_dec" in state_dict or "decoder.bias" in state_dict
            
            config = PAnnealConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        
        # Handle legacy parameter naming
        if "W_enc" in state_dict:
            converted_dict = {}
            converted_dict["encoder.weight"] = state_dict["W_enc"].T
            converted_dict["encoder.bias"] = state_dict["b_enc"]
            converted_dict["decoder.weight"] = state_dict["W_dec"].T
            
            if "b_dec" in state_dict:
                if config.use_april_update_mode:
                    converted_dict["decoder.bias"] = state_dict["b_dec"]
                else:
                    converted_dict["bias"] = state_dict["b_dec"]
                    
            if config.var_flag == 1 and "W_enc_var" in state_dict:
                converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                
            state_dict = converted_dict
        
        # Load compatible parameters
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Initialize missing parameters
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys and config.var_flag == 1:
            print(f"Initializing missing variance encoder parameters: {missing_keys}")
            with torch.no_grad():
                if hasattr(model, 'var_encoder'):
                    nn.init.kaiming_uniform_(model.var_encoder.weight)
                    nn.init.zeros_(model.var_encoder.bias)
        
        # Normalize decoder if requested
        if normalize_decoder and not (config.var_flag == 1 and "var_encoder.weight" in state_dict):
            try:
                model.normalize_decoder()
            except AssertionError:
                print("Warning: Could not normalize decoder weights. Skipping normalization.")
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
            
        return model


class VSAEPAnnealTrainer(SAETrainer):
    """
    Improved trainer for Variational Sparse Autoencoder with p-norm annealing.
    
    Features:
    - Clean separation of concerns
    - Proper p-annealing schedule with adaptive sparsity coefficient
    - Memory-efficient processing
    - Better error handling and logging
    """
    
    def __init__(
        self,
        model_config: PAnnealConfig = None,
        training_config: PAnnealTrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        sparsity_penalty: Optional[float] = None,
        var_flag: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None or training_config is None:
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = PAnnealConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    var_flag=var_flag or 0,
                    use_april_update_mode=kwargs.get('use_april_update_mode', True),
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = PAnnealTrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    sparsity_penalty=sparsity_penalty or 500.0,
                    **{k: v for k, v in kwargs.items() if k in PAnnealTrainingConfig.__dataclass_fields__}
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEPAnnealTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAEPAnneal(model_config)
        self.ae.to(self.device)
        
        # Initialize p-annealing state
        self._init_p_annealing()
        
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
        self.logging_parameters = [
            'p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'sparsity_coeff',
            'recon_loss', 'total_loss'
        ]
        self._init_logging_state()
    
    def _init_p_annealing(self) -> None:
        """Initialize p-annealing schedule and tracking."""
        config = self.training_config
        
        # Create p-value schedule
        self.p = config.p_start
        self.next_p = None
        
        if config.n_sparsity_updates == "continuous":
            self.n_sparsity_updates = config.anneal_end - config.anneal_start + 1
        else:
            self.n_sparsity_updates = config.n_sparsity_updates
            
        self.sparsity_update_steps = torch.linspace(
            config.anneal_start, 
            config.anneal_end, 
            self.n_sparsity_updates, 
            dtype=torch.long
        )
        self.p_values = torch.linspace(
            config.p_start, 
            config.p_end, 
            self.n_sparsity_updates
        )
        
        self.p_step_count = 0
        self.sparsity_coeff = config.sparsity_penalty
        self.sparsity_queue = []
        
        # Check for duplicates in update steps
        unique_steps, counts = self.sparsity_update_steps.unique(return_counts=True)
        if (counts > 1).any():
            print("Warning! Duplicates in sparsity_update_steps detected!")
    
    def _init_logging_state(self) -> None:
        """Initialize logging state variables."""
        self.kl_loss = 0.0
        self.scaled_kl_loss = 0.0
        self.recon_loss = 0.0
        self.total_loss = 0.0
    
    def _compute_lp_kl_loss(self, mu: torch.Tensor, log_var: Optional[torch.Tensor], p: float) -> torch.Tensor:
        """
        Compute the p-norm regularized KL divergence loss.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution (only used when var_flag=1)
            p: The p-norm value to use
            
        Returns:
            The p-norm regularized KL divergence loss
        """
        # Get decoder norms for scaling (from April update)
        decoder_norms = torch.norm(self.ae.decoder.weight, p=2, dim=0)
        decoder_norms = decoder_norms.to(dtype=mu.dtype)
        
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for Gaussian with learned variance
            kl_per_latent = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            if p == 1.0:
                kl_loss = (kl_per_latent.sum(dim=-1) * decoder_norms.mean()).mean()
            else:
                if self.training_config.sparsity_function == 'Lp^p':
                    kl_loss = (kl_per_latent.pow(p).sum(dim=-1) * decoder_norms.mean()).mean()
                else:  # 'Lp'
                    kl_loss = (kl_per_latent.pow(p).sum(dim=-1).pow(1/p) * decoder_norms.mean()).mean()
        else:
            # Simplified KL for fixed variance
            if p == 1.0:
                kl_loss = (mu.abs().sum(dim=-1) * decoder_norms.mean()).mean()
            elif p == 2.0:
                kl_loss = (mu.pow(2).sum(dim=-1) * decoder_norms.mean()).mean() * 0.5
            else:
                if self.training_config.sparsity_function == 'Lp^p':
                    kl_loss = (mu.abs().pow(p).sum(dim=-1) * decoder_norms.mean()).mean()
                else:  # 'Lp'
                    kl_loss = (mu.abs().pow(p).sum(dim=-1).pow(1/p) * decoder_norms.mean()).mean()
        
        return kl_loss
    
    def _update_p_annealing(self, step: int, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> None:
        """Update p-annealing schedule and sparsity coefficient."""
        # Check if we need to compute the next p-value's loss for adaptive scaling
        if self.next_p is not None:
            kl_loss_current = self._compute_lp_kl_loss(mu, log_var, p=self.p)
            kl_loss_next = self._compute_lp_kl_loss(mu, log_var, p=self.next_p)
            self.sparsity_queue.append([kl_loss_current.item(), kl_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.training_config.sparsity_queue_length:]
        
        # Update p-value if needed
        if step in self.sparsity_update_steps:
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt sparsity penalty
                if self.next_p is not None and len(self.sparsity_queue) > 0:
                    local_sparsity_current = torch.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_next = torch.tensor([i[1] for i in self.sparsity_queue]).mean()
                    if local_sparsity_next > 0:
                        self.sparsity_coeff = self.sparsity_coeff * (local_sparsity_current / local_sparsity_next).item()
                        
                # Update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates - 1:
                    self.next_p = self.p_values[self.p_step_count + 1].item()
                else:
                    self.next_p = self.training_config.p_end
                self.p_step_count += 1
                
                print(f"Step {step}: Updated p to {self.p:.4f}, next_p: {self.next_p:.4f}, sparsity_coeff: {self.sparsity_coeff:.4f}")
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        if self.ae.var_flag == 1:
            x_hat, f, mu, log_var = self.ae(x, output_features=True)
        else:
            x_hat, f = self.ae(x, output_features=True)
            mu = f
            log_var = None
        
        # Reconstruction loss
        recon_loss = torch.pow(x - x_hat, 2).sum(dim=-1).mean()
        self.recon_loss = recon_loss.item()
        
        # KL divergence loss with p-norm regularization
        kl_loss = self._compute_lp_kl_loss(mu, log_var, p=self.p)
        self.kl_loss = kl_loss.item()
        
        # Update p-annealing schedule
        self._update_p_annealing(step, mu.detach(), log_var.detach() if log_var is not None else None)
        
        # Scale KL loss
        scaled_kl_loss = kl_loss * self.sparsity_coeff * sparsity_scale
        self.scaled_kl_loss = scaled_kl_loss.item()
        
        # Total loss
        total_loss = recon_loss + scaled_kl_loss
        self.total_loss = total_loss.item()
        
        # Ensure output is in original dtype
        total_loss = total_loss.to(dtype=original_dtype)
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat.to(dtype=original_dtype), f.to(dtype=original_dtype),
            {
                'p': self.p,
                'next_p': self.next_p,
                'kl_loss': self.kl_loss,
                'scaled_kl_loss': self.scaled_kl_loss,
                'sparsity_coeff': self.sparsity_coeff,
                'recon_loss': self.recon_loss,
                'total_loss': self.total_loss,
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
            'dict_class': 'VSAEPAnneal',
            'trainer_class': 'VSAEPAnnealTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'sparsity_penalty': self.training_config.sparsity_penalty,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            'sparsity_function': self.training_config.sparsity_function,
            'anneal_start': self.training_config.anneal_start,
            'anneal_end': self.training_config.anneal_end,
            'p_start': self.training_config.p_start,
            'p_end': self.training_config.p_end,
            'n_sparsity_updates': self.training_config.n_sparsity_updates,
            'sparsity_queue_length': self.training_config.sparsity_queue_length,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }