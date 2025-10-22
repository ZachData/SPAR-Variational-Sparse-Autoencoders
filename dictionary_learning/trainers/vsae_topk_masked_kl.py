"""
VSAETopK: Variational Sparse Autoencoder with Top-K Selection
MODIFIED: KL loss only applied to top-k selected features
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
from collections import namedtuple
import math


@dataclass
class VSAETopKConfig:
    """Configuration for VSAETopK model."""
    activation_dim: int
    dict_size: int
    k: int
    var_flag: int = 0
    use_april_update_mode: bool = True
    log_var_init: float = -2.0
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        if self.device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self.device


class VSAETopK(nn.Module):
    """
    Variational Sparse Autoencoder with Top-K selection.
    MODIFIED: KL loss computation now supports masking for top-k features only.
    """
    
    def __init__(self, config: VSAETopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.k = nn.Parameter(torch.tensor(config.k), requires_grad=False)
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        device = config.get_device()
        dtype = config.dtype
        
        # Encoder: activation_dim -> dict_size (mean)
        self.encoder = nn.Linear(self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device)
        
        # Variance encoder (if learned variance)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device)
        
        # Decoder: dict_size -> activation_dim
        self.decoder = nn.Linear(self.dict_size, self.activation_dim, bias=False, dtype=dtype, device=device)
        
        # Bias handling
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(torch.zeros(self.activation_dim, dtype=dtype, device=device))
        else:
            self.decoder.bias = nn.Parameter(torch.zeros(self.activation_dim, dtype=dtype, device=device))
        
        self._initialize_weights()

    def scale_biases(self, scale: float) -> None:
        """Scale biases by a factor (for activation normalization)."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
                
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    def _initialize_weights(self):
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Tied initialization for encoder and decoder
        w = torch.randn(self.activation_dim, self.dict_size, dtype=dtype, device=device)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        
        with torch.no_grad():
            self.encoder.weight.copy_(w.T)
            self.decoder.weight.copy_(w)
            
            nn.init.zeros_(self.encoder.bias)
            if self.use_april_update_mode:
                nn.init.zeros_(self.decoder.bias)
            else:
                nn.init.zeros_(self.bias)
            
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight, a=0.01)
                nn.init.constant_(self.var_encoder.bias, self.config.log_var_init)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.encoder.weight.dtype)
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_topk: bool = False,
        training: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode with VAE + Top-K selection.
        
        Returns:
            sparse_features, latent_z, mu, log_var, [top_indices, selected_vals]
        """
        x_processed = self._preprocess_input(x)
        
        # Encode to latent distribution parameters
        mu = self.encoder(x_processed)
        
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
        
        # Sample from latent distribution
        if training and self.var_flag == 1 and log_var is not None:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        
        # Apply Top-K sparsity on absolute values
        z_abs = torch.abs(z)
        top_vals_abs, top_indices = z_abs.topk(self.k.item(), sorted=False, dim=-1)
        
        # Create sparse feature vector using original latent values
        sparse_features = torch.zeros_like(z)
        selected_vals = torch.gather(z, dim=-1, index=top_indices)
        sparse_features = sparse_features.scatter_(dim=-1, index=top_indices, src=selected_vals)
        
        if return_topk:
            return sparse_features, z, mu, log_var, top_indices, selected_vals
        else:
            return sparse_features, z, mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        if log_var is None:
            return mu
        
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        std = torch.exp(0.5 * log_var_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, sparse_features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstruction."""
        if self.use_april_update_mode:
            return self.decoder(sparse_features)
        else:
            return self.decoder(sparse_features) + self.bias
    
    def forward(self, x: torch.Tensor, output_features: bool = False):
        sparse_features, _, _, _ = self.encode(x, training=self.training)
        x_hat = self.decode(sparse_features)
        
        if output_features:
            return x_hat, sparse_features
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        """Get detailed KL divergence diagnostics."""
        with torch.no_grad():
            _, latent_z, mu, log_var, _, _ = self.encode(x, return_topk=True, training=False)
            
            diagnostics = {
                'latent_z_mean': latent_z.mean().item(),
                'latent_z_std': latent_z.std().item(),
                'mu_mean': mu.mean().item(),
                'mu_std': mu.std().item(),
            }
            
            if self.var_flag == 1 and log_var is not None:
                diagnostics.update({
                    'log_var_mean': log_var.mean().item(),
                    'log_var_std': log_var.std().item(),
                    'variance_mean': torch.exp(log_var).mean().item(),
                })
            
            return diagnostics
    
    def get_topk_analysis(self, x: torch.Tensor) -> Dict[str, float]:
        """Get detailed Top-K selection analysis."""
        with torch.no_grad():
            sparse_features, latent_z, mu, log_var, top_indices, selected_vals = self.encode(
                x, return_topk=True, training=False
            )
            
            z_abs = torch.abs(latent_z)
            sparsity_ratio = (sparse_features != 0).float().mean()
            
            return {
                'batch_size': x.shape[0],
                'k_value': self.k.item(),
                'actual_sparsity_ratio': sparsity_ratio.item(),
                'theoretical_sparsity_ratio': self.k.item() / self.dict_size,
                'latent_z_mean': latent_z.mean().item(),
                'latent_z_std': latent_z.std().item(),
                'selected_vals_mean': selected_vals.mean().item(),
                'selected_vals_std': selected_vals.std().item(),
                'negative_selected_ratio': (selected_vals < 0).float().mean().item(),
                'topk_threshold': z_abs.topk(self.k.item(), dim=-1)[0][:, -1].mean().item(),
            }


@dataclass
class VSAETopKTrainingConfig:
    """Training configuration for VSAETopK."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 1.0
    auxk_alpha: float = 0
    
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    kl_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    
    dead_feature_threshold: int = 10_000
    
    def __post_init__(self):
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)
        
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start


def get_lr_schedule(total_steps, warmup_steps, decay_start, decay_end, sparsity_warmup_steps):
    """Learning rate schedule with warmup and optional decay."""
    def lr_fn(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif decay_start is not None and step >= decay_start:
            if decay_end is None:
                progress = (step - decay_start) / (total_steps - decay_start)
            else:
                progress = min(1.0, (step - decay_start) / (decay_end - decay_start))
            return 1.0 - progress
        else:
            return 1.0
    return lr_fn


def get_sparsity_warmup_fn(total_steps, sparsity_warmup_steps):
    """Sparsity penalty warmup schedule."""
    def warmup_fn(step):
        if sparsity_warmup_steps is None or sparsity_warmup_steps == 0:
            return 1.0
        return min(1.0, step / sparsity_warmup_steps)
    return warmup_fn


def get_kl_warmup_fn(total_steps, kl_warmup_steps):
    """KL divergence warmup schedule to prevent posterior collapse."""
    def warmup_fn(step):
        if kl_warmup_steps is None or kl_warmup_steps == 0:
            return 1.0
        return min(1.0, step / kl_warmup_steps)
    return warmup_fn


class DeadFeatureTracker:
    """Tracks dead features for auxiliary loss computation."""
    
    def __init__(self, dict_size: int, threshold: int, device: torch.device):
        self.threshold = threshold
        self.num_tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    
    def update(self, active_features: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Update dead feature tracking and return dead feature mask."""
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[active_features] = 0
        return self.num_tokens_since_fired >= self.threshold
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about dead features."""
        dead_mask = self.num_tokens_since_fired >= self.threshold
        return {
            "dead_features": int(dead_mask.sum()),
            "alive_features": int((~dead_mask).sum()),
            "total_features": len(self.num_tokens_since_fired)
        }


class VSAETopKTrainer:
    """
    Trainer for VSAETopK model.
    MODIFIED: KL loss now only applied to top-k selected features.
    """
    
    def __init__(
        self,
        model_config: Optional[VSAETopKConfig] = None,
        training_config: Optional[VSAETopKTrainingConfig] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        var_flag: Optional[int] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
    ):
        if model_config is None:
            if activation_dim is None or dict_size is None or k is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
            
            device_obj = torch.device(device) if device else None
            model_config = VSAETopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                var_flag=var_flag or 0,
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else True,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = VSAETopKTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=kl_coeff or 500.0,
                auxk_alpha=auxk_alpha or 1/32,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAETopKMasked"
        
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAETopK(model_config)
        self.ae.to(self.device)
        
        # Top-K specific tracking
        self.top_k_aux = model_config.activation_dim // 2
        
        # Initialize dead feature tracking
        self.dead_feature_tracker = DeadFeatureTracker(
            model_config.dict_size,
            training_config.dead_feature_threshold,
            self.device
        )
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        
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
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )

    @property
    def config(self):
        """Return config dict for wandb logging."""
        return {
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'wandb_name': self.wandb_name,
        }
    
    def get_logging_parameters(self) -> Dict[str, float]:
        """Return current logging parameters as a dictionary."""
        return {
            "effective_l0": self.effective_l0,
            "dead_features": self.dead_features,
            "pre_norm_auxk_loss": self.pre_norm_auxk_loss,
        }
    
    def _compute_kl_loss(
        self, 
        latent_z: torch.Tensor, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MODIFIED: KL divergence computation with optional masking for top-k features only.
        
        Args:
            latent_z: The actual latent variables
            mu: Mean parameters from encoder
            log_var: Log variance parameters (None for fixed variance)
            mask: Boolean mask indicating which features to include in KL (top-k selected)
            
        Returns:
            KL divergence loss
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for learned variance
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # KL[q(z|x) || p(z)] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
            kl_per_dim = 0.5 * (
                mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped
            )
        else:
            # Fixed variance case: KL = 0.5 * ||latent_z||²
            latent_z_clamped = torch.clamp(latent_z, min=-10.0, max=10.0)
            kl_per_dim = 0.5 * latent_z_clamped.pow(2)
        
        # MODIFIED: Apply mask before summing if provided
        if mask is not None:
            kl_per_dim = kl_per_dim * mask.float()
        
        kl_per_sample = kl_per_dim.sum(dim=1)
        kl_loss = kl_per_sample.mean()
        
        return torch.clamp(kl_loss, min=0.0)
    
    def get_auxiliary_loss(
        self, 
        residual_BD: torch.Tensor, 
        sparse_features_BF: torch.Tensor, 
        latent_z_BF: torch.Tensor
    ):
        """Auxiliary loss for reviving dead features."""
        active_features = (sparse_features_BF.sum(0) > 0)
        num_tokens = sparse_features_BF.size(0)
        dead_features = self.dead_feature_tracker.update(active_features, num_tokens)
        
        self.dead_features = int(dead_features.sum())
        
        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)
            
            auxk_latents = torch.where(dead_features[None], torch.abs(latent_z_BF), -torch.inf)
            auxk_abs_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_original_vals = torch.gather(latent_z_BF, dim=-1, index=auxk_indices)
            
            auxk_buffer_BF = torch.zeros_like(latent_z_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_original_vals)
            
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )
            
            self.pre_norm_auxk_loss = l2_loss_aux.item()
            
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / (loss_denom + 1e-8)
            
            return normalized_auxk_loss
        else:
            self.pre_norm_auxk_loss = 0.0
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """
        MODIFIED: Loss computation with KL applied only to top-k selected features.
        
        The loss includes:
        1. Reconstruction error (MSE)
        2. KL divergence on top-k selected features only (with separate annealing)
        3. Auxiliary loss for reviving dead sparse features
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        kl_scale = self.kl_warmup_fn(step)
        
        original_dtype = x.dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)
        
        # Encode and get top-k indices
        sparse_features, latent_z, mu, log_var, top_indices, selected_vals = self.ae.encode(
            x, return_topk=True, training=self.ae.training
        )
        
        # MODIFIED: Create mask from top_indices
        topk_mask = torch.zeros_like(latent_z, dtype=torch.bool)
        topk_mask.scatter_(dim=-1, index=top_indices, value=True)
        
        # Decode and calculate reconstruction error
        x_hat = self.ae.decode(sparse_features)
        residual = x - x_hat
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l2_loss = torch.linalg.norm(residual, dim=-1).mean()
        
        self.effective_l0 = self.ae.k.item()
        
        # MODIFIED: KL divergence loss with mask for top-k features only
        kl_loss = self._compute_kl_loss(latent_z, mu, log_var, mask=topk_mask)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Auxiliary loss
        auxk_loss = self.get_auxiliary_loss(residual.detach(), sparse_features, latent_z) if self.training_config.auxk_alpha > 0 else 0
        
        # Total loss
        recon_loss = recon_loss.to(dtype=original_dtype)
        if isinstance(auxk_loss, torch.Tensor):
            auxk_loss = auxk_loss.to(dtype=original_dtype)
        else:
            auxk_loss = torch.tensor(auxk_loss, dtype=original_dtype, device=x.device)
        
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * kl_scale * kl_loss +
            self.training_config.auxk_alpha * auxk_loss
        )
        
        if not logging:
            return total_loss
        else:
            x_hat = x_hat.to(dtype=original_dtype)
            sparse_features = sparse_features.to(dtype=original_dtype)
            
            kl_diagnostics = self.ae.get_kl_diagnostics(x.to(dtype=original_dtype))
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x.to(dtype=original_dtype),
                x_hat,
                sparse_features,
                {
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    'l2_loss': l2_loss.item(),
                    **kl_diagnostics
                }
            )
    
    def update(self, step: int, x: torch.Tensor):
        """Perform single training step."""
        self.optimizer.zero_grad()
        loss = self.loss(x, step, logging=False)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def get_state(self):
        """Get trainer state for checkpointing."""
        return {
            'model_state_dict': self.ae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dead_feature_tracker': self.dead_feature_tracker.num_tokens_since_fired,
        }
    
    def load_state(self, state):
        """Load trainer state from checkpoint."""
        self.ae.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.dead_feature_tracker.num_tokens_since_fired = state['dead_feature_tracker']
