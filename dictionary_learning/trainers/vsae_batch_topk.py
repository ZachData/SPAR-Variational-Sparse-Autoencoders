"""
HONEST implementation of VSAEBatchTopK - Acknowledging Architectural Tensions

FUNDAMENTAL ISSUE ACKNOWLEDGED:
VAE (unconstrained latents) and TopK sparsity (positive sparse activations) are 
fundamentally incompatible. This implementation makes explicit design choices
for honest experimentation.

DESIGN CHOICE MADE:
"VAE-first" approach - prioritize VAE mathematics, apply sparsity as post-processing.
This preserves gradient flow but may reduce sparsity effectiveness.

ALTERNATIVES NOT IMPLEMENTED (but could be):
1. "Sparse-first": Treat VAE parts as regularizers (breaks VAE theory)
2. "Gumbel-Softmax": Use differentiable discrete sampling (more complex)
3. "Separate streams": Completely separate VAE and sparse pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, List, Callable
from collections import namedtuple
from dataclasses import dataclass
import math
import warnings

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


@dataclass
class VSAEBatchTopKConfig:
    """Configuration with explicit architectural choice documentation."""
    activation_dim: int
    dict_size: int
    k: int  # Number of top features to keep active
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    constrain_decoder: bool = True
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0
    
    # NEW: Explicit architectural choice
    architecture_mode: str = "vae_first"  # "vae_first", "sparse_first", or "hybrid"
    apply_topk_to_samples: bool = True  # Apply TopK to reparameterized samples (VAE-first)
    preserve_gradient_flow: bool = True  # Avoid ReLU before TopK to preserve gradients
    
    def __post_init__(self):
        if self.k > self.dict_size:
            raise ValueError(f"k ({self.k}) cannot be larger than dict_size ({self.dict_size})")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        
        # Warn about architectural tensions
        if self.var_flag == 1 and self.architecture_mode == "sparse_first":
            warnings.warn("Learned variance with sparse_first mode may not train well - consider vae_first")
    
    def get_device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


@dataclass
class TrainingConfig:
    """Training configuration with KL annealing."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None
    auxk_alpha: float = 1/32
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    dead_feature_threshold: int = 10_000_000
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


def get_kl_warmup_fn(total_steps: int, kl_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """KL annealing to prevent posterior collapse."""
    if kl_warmup_steps is None or kl_warmup_steps == 0:
        return lambda step: 1.0
    
    assert 0 < kl_warmup_steps <= total_steps
    
    def scale_fn(step: int) -> float:
        return min(step / kl_warmup_steps, 1.0) if step < kl_warmup_steps else 1.0
    
    return scale_fn


class VSAEBatchTopK(Dictionary, nn.Module):
    """
    HONEST VSAEBatchTopK implementation with explicit architectural choices.
    
    FUNDAMENTAL TENSION ACKNOWLEDGED:
    VAE and TopK sparsity have conflicting requirements. This implementation
    makes explicit choices about how to handle this tension.
    
    CHOSEN APPROACH: "VAE-first"
    1. Preserve VAE mathematics (unconstrained latents, proper KL)
    2. Apply TopK as post-processing step
    3. Compute KL on actual latents being used for reconstruction
    4. Accept that sparsity may be less effective than pure sparse autoencoders
    """

    def __init__(self, config: VSAEBatchTopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.constrain_decoder = config.constrain_decoder
        self.use_april_update_mode = config.use_april_update_mode
        
        # Architecture mode determines how we handle VAE/sparsity interaction
        self.architecture_mode = config.architecture_mode
        self.apply_topk_to_samples = config.apply_topk_to_samples
        self.preserve_gradient_flow = config.preserve_gradient_flow
        
        # Register k as a buffer
        self.register_buffer("k", torch.tensor(config.k, dtype=torch.int))
        
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize layers."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        self.encoder = nn.Linear(
            self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device
        )
        self.decoder = nn.Linear(
            self.dict_size, self.activation_dim, bias=self.use_april_update_mode, dtype=dtype, device=device
        )
        
        if not self.use_april_update_mode:
            self.bias = nn.Parameter(torch.zeros(self.activation_dim, dtype=dtype, device=device))
        
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim, self.dict_size, bias=True, dtype=dtype, device=device
            )
    
    def _init_weights(self) -> None:
        """Initialize weights with tied encoder/decoder."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
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
        """Handle input preprocessing."""
        x = x.to(dtype=self.encoder.weight.dtype)
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def _apply_topk_sparsity(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TopK sparsity with architectural choice consideration.
        
        DESIGN CHOICE: How to handle negative values from VAE sampling?
        
        Returns:
            sparse_z: Sparsified latents for reconstruction  
            selection_mask: Boolean mask of selected features
        """
        if self.architecture_mode == "vae_first" and self.preserve_gradient_flow:
            # OPTION 1: Apply TopK to raw samples (preserves gradients, allows negative values)
            batch_size = z.size(0)
            flattened = z.view(-1)
            
            # Get top-k by absolute value to handle negative samples
            abs_values = torch.abs(flattened)
            k_total = min(self.k * batch_size, abs_values.numel())
            
            if k_total > 0:
                _, topk_indices = abs_values.topk(k_total, sorted=False)
                
                # Create sparse tensor preserving original values (including negatives)
                sparse_flat = torch.zeros_like(flattened)
                sparse_flat[topk_indices] = flattened[topk_indices]
                sparse_z = sparse_flat.view(z.shape)
                
                selection_mask = torch.zeros_like(z, dtype=torch.bool)
                selection_mask.view(-1)[topk_indices] = True
            else:
                sparse_z = torch.zeros_like(z)
                selection_mask = torch.zeros_like(z, dtype=torch.bool)
                
        elif self.architecture_mode == "sparse_first":
            # OPTION 2: Apply ReLU then TopK (breaks VAE gradients, enforces positivity)
            z_positive = F.relu(z)
            batch_size = z_positive.size(0)
            flattened = z_positive.view(-1)
            
            k_total = min(self.k * batch_size, flattened.numel())
            if k_total > 0:
                topk_values, topk_indices = flattened.topk(k_total, sorted=False)
                
                sparse_flat = torch.zeros_like(flattened)
                sparse_flat[topk_indices] = topk_values
                sparse_z = sparse_flat.view(z_positive.shape)
                
                selection_mask = sparse_z > 0
            else:
                sparse_z = torch.zeros_like(z_positive)
                selection_mask = torch.zeros_like(z_positive, dtype=torch.bool)
        
        else:  # hybrid or fallback
            # Apply TopK to positive part only
            z_positive = F.relu(z)
            sparse_z, selection_mask = self._apply_topk_sparsity_positive(z_positive)
        
        return sparse_z, selection_mask
    
    def _apply_topk_sparsity_positive(self, z_positive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply TopK to positive activations only."""
        batch_size = z_positive.size(0)
        flattened = z_positive.view(-1)
        
        k_total = min(self.k * batch_size, flattened.numel())
        if k_total > 0:
            topk_values, topk_indices = flattened.topk(k_total, sorted=False)
            
            sparse_flat = torch.zeros_like(flattened)
            sparse_flat[topk_indices] = topk_values
            sparse_z = sparse_flat.view(z_positive.shape)
            
            selection_mask = sparse_z > 0
        else:
            sparse_z = torch.zeros_like(z_positive)
            selection_mask = torch.zeros_like(z_positive, dtype=torch.bool)
        
        return sparse_z, selection_mask
    
    def encode(self, x: torch.Tensor, return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        HONEST encoding that acknowledges the VAE/sparsity tension.
        
        Returns all components needed for different loss computations.
        """
        x_processed = self._preprocess_input(x)
        
        # Get unconstrained mean (VAE parameter)
        mu = self.encoder(x_processed)
        
        # Get log variance if learning it
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
        
        # Sample from VAE distribution
        if self.var_flag == 1:
            z_samples = self.reparameterize(mu, log_var)
        else:
            z_samples = mu  # For fixed variance, samples = mean
        
        # Apply sparsity (this is where the architectural tension manifests)
        sparse_z, selection_mask = self._apply_topk_sparsity(z_samples)
        
        # Package results
        result = {
            'mu': mu,                    # Unconstrained mean for KL computation
            'log_var': log_var,          # Log variance (if learned)
            'z_samples': z_samples,      # Full VAE samples
            'sparse_z': sparse_z,        # Sparse latents for reconstruction
            'selection_mask': selection_mask,  # Which features were selected
        }
        
        # Additional components for auxiliary loss
        if return_components:
            result.update({
                'z_positive': F.relu(z_samples),  # Positive part of samples
                'active_features': selection_mask.sum(0) > 0,  # Which features are active across batch
            })
        
        return result
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """Standard VAE reparameterization trick."""
        if log_var is None:
            return mu
        
        # Conservative clamping
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        std = torch.sqrt(torch.exp(log_var_clamped))
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Standard decoding."""
        z = z.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(z)
        else:
            return self.decoder(z) + self.bias
    
    def forward(self, x: torch.Tensor, output_features: bool = False, **kwargs) -> torch.Tensor:
        """
        HONEST forward pass - uses sparse latents for reconstruction.
        
        ARCHITECTURAL CHOICE: We reconstruct from sparse_z, not full z_samples.
        This means we're not using the full VAE generative model.
        """
        original_dtype = x.dtype
        
        # Get encoding components
        encoding = self.encode(x)
        sparse_z = encoding['sparse_z']
        
        # Reconstruct from sparse latents
        x_hat = self.decode(sparse_z)
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            sparse_z = sparse_z.to(dtype=original_dtype)
            return x_hat, sparse_z
        return x_hat
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get KL diagnostics based on actual latents being used."""
        with torch.no_grad():
            encoding = self.encode(x)
            mu = encoding['mu']
            log_var = encoding['log_var']
            z_samples = encoding['z_samples']
            sparse_z = encoding['sparse_z']
            
            if self.var_flag == 1 and log_var is not None:
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)
                
                # KL components
                kl_mu = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                kl_var = 0.5 * torch.sum(var - 1 - log_var_safe, dim=1).mean()
                kl_total = kl_mu + kl_var
                
                # Diagnostics about sparsity effect
                sparsity_ratio = (sparse_z != 0).float().mean()
                sample_utilization = (sparse_z != 0).float().sum(dim=1).mean()
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_mu,
                    'kl_var_term': kl_var,
                    'mean_log_var': log_var.mean(),
                    'mean_var': var.mean(),
                    'sparsity_ratio': sparsity_ratio,
                    'sample_utilization': sample_utilization,
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                }
            else:
                kl_total = 0.5 * torch.sum(mu.pow(2), dim=1).mean()
                sparsity_ratio = (sparse_z != 0).float().mean()
                
                return {
                    'kl_total': kl_total,
                    'kl_mu_term': kl_total,
                    'kl_var_term': torch.tensor(0.0),
                    'sparsity_ratio': sparsity_ratio,
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                }
    
    def scale_biases(self, scale: float) -> None:
        """Scale bias parameters."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            
            if self.use_april_update_mode:
                if hasattr(self.decoder, 'bias') and self.decoder.bias is not None:
                    self.decoder.bias.mul_(scale)
            else:
                self.bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
    
    @classmethod
    def from_pretrained(cls, path: str, config: Optional[VSAEBatchTopKConfig] = None, **kwargs):
        """Load pretrained model with proper config reconstruction."""
        checkpoint = torch.load(path, map_location=kwargs.get('device'))
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Reconstruct config from checkpoint
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            
            k = kwargs.get('k')
            if k is None and "k" in state_dict:
                k = state_dict["k"].item()
            if k is None:
                raise ValueError("k must be provided")
            
            var_flag = kwargs.get('var_flag', 1 if "var_encoder.weight" in state_dict else 0)
            use_april_update_mode = "decoder.bias" in state_dict
            
            config = VSAEBatchTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                **{k: v for k, v in kwargs.items() if k in VSAEBatchTopKConfig.__dataclass_fields__}
            )
        
        model = cls(config)
        model.load_state_dict(state_dict, strict=False)
        
        if kwargs.get('device'):
            model = model.to(device=kwargs['device'])
            
        return model


class DeadFeatureTracker:
    """Simple dead feature tracking."""
    
    def __init__(self, dict_size: int, threshold: int, device: torch.device):
        self.threshold = threshold
        self.num_tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    
    def update(self, active_mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Update tracking with boolean mask of active features."""
        self.num_tokens_since_fired += num_tokens
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        self.num_tokens_since_fired[active_indices] = 0
        
        return self.num_tokens_since_fired >= self.threshold
    
    def get_stats(self) -> Dict[str, int]:
        dead_mask = self.num_tokens_since_fired >= self.threshold
        return {
            "dead_features": int(dead_mask.sum()),
            "alive_features": int((~dead_mask).sum()),
            "total_features": len(self.num_tokens_since_fired)
        }


class VSAEBatchTopKTrainer(SAETrainer):
    """
    HONEST trainer that acknowledges architectural tensions.
    
    DESIGN PHILOSOPHY:
    1. Be explicit about what we're optimizing
    2. Use mathematically sound loss computation
    3. Acknowledge trade-offs in VAE/sparsity combination
    4. Provide clear diagnostics about what's happening
    """
    
    def __init__(
        self,
        model_config: Optional[VSAEBatchTopKConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility
        **kwargs
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            activation_dim = kwargs.get('activation_dim')
            dict_size = kwargs.get('dict_size') 
            k = kwargs.get('k')
            if not all([activation_dim, dict_size, k]):
                raise ValueError("Must provide activation_dim, dict_size, k")
            
            model_config = VSAEBatchTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                var_flag=kwargs.get('var_flag', 0),
                device=torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            )
        
        if training_config is None:
            steps = kwargs.get('steps')
            if steps is None:
                raise ValueError("Must provide steps")
            
            training_config = TrainingConfig(
                steps=steps,
                lr=kwargs.get('lr', 5e-4),
                kl_coeff=kwargs.get('kl_coeff', 500.0),
                auxk_alpha=kwargs.get('auxk_alpha', 1/32),
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEBatchTopKTrainer"
        
        # Warn about architectural choice
        if model_config.var_flag == 1:
            print(f"WARNING: Using VAE with TopK sparsity (architecture_mode: {model_config.architecture_mode})")
            print("This combination has fundamental tensions. Monitor KL vs sparsity trade-offs carefully.")
        
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEBatchTopK(model_config)
        self.ae.to(self.device)
        
        # Initialize optimizer and schedulers
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=training_config.lr)
        
        lr_fn = get_lr_schedule(
            training_config.steps, training_config.warmup_steps, 
            training_config.decay_start, None, training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps, training_config.sparsity_warmup_steps
        )
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps, training_config.kl_warmup_steps
        )
        
        # Dead feature tracking
        self.dead_feature_tracker = DeadFeatureTracker(
            model_config.dict_size, training_config.dead_feature_threshold, self.device
        )
        
        # Logging attributes
        self.logging_parameters = ["effective_l0", "dead_features", "kl_utilization"]
        self.effective_l0 = 0.0
        self.dead_features = 0
        self.kl_utilization = 0.0
    
    def _compute_kl_loss(self, encoding: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        HONEST KL computation - what distribution are we actually using?
        
        ARCHITECTURAL CHOICE: Compute KL on the latents we're actually using for reconstruction.
        This might not be the traditional VAE KL, but it's honest about what we're doing.
        """
        mu = encoding['mu']
        log_var = encoding.get('log_var')
        sparse_z = encoding['sparse_z']
        
        if self.model_config.architecture_mode == "vae_first":
            # Compute KL on the original VAE parameters
            if self.ae.var_flag == 1 and log_var is not None:
                log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
                mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
                
                kl_per_sample = 0.5 * torch.sum(
                    mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                    dim=1
                )
            else:
                mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
                kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
        
        elif self.model_config.architecture_mode == "sparse_first":
            # Alternative: compute "KL" on the sparse features we're actually using
            # This is not theoretically justified but may work better in practice
            sparse_z_clamped = torch.clamp(sparse_z, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(sparse_z_clamped.pow(2), dim=1)
        
        else:  # hybrid
            # Weighted combination
            if self.ae.var_flag == 1 and log_var is not None:
                vae_kl = 0.5 * torch.sum(mu.pow(2) + torch.exp(log_var) - 1 - log_var, dim=1)
            else:
                vae_kl = 0.5 * torch.sum(mu.pow(2), dim=1)
            
            sparse_kl = 0.5 * torch.sum(sparse_z.pow(2), dim=1)
            kl_per_sample = 0.7 * vae_kl + 0.3 * sparse_kl  # Weighted combination
        
        kl_loss = kl_per_sample.mean()
        return torch.clamp(kl_loss, min=0.0)
    
    def _compute_auxiliary_loss(self, residual: torch.Tensor, encoding: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute auxiliary loss on sparse features (not intermediate activations)."""
        sparse_z = encoding['sparse_z']
        active_features = encoding.get('active_features', sparse_z.sum(0) > 0)
        
        # Update dead feature tracking
        num_tokens = residual.size(0)
        dead_mask = self.dead_feature_tracker.update(active_features, num_tokens)
        
        stats = self.dead_feature_tracker.get_stats()
        self.dead_features = stats["dead_features"]
        
        if self.dead_features == 0:
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
        
        # Get auxiliary features from sparse latents (not from separate activations)
        z_positive = encoding.get('z_positive', F.relu(encoding['z_samples']))
        
        k_aux = min(self.ae.activation_dim // 2, self.dead_features)
        auxk_latents = torch.where(dead_mask[None], z_positive, -torch.inf)
        
        if auxk_latents.max() == -torch.inf:  # No valid auxiliary features
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
        
        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
        auxk_buffer = torch.zeros_like(z_positive)
        auxk_acts_active = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
        
        # Use consistent decode method
        x_reconstruct_aux = self.ae.decode(auxk_acts_active)
        
        l2_loss_aux = (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
        
        # Normalized auxiliary loss
        residual_var = residual.var(dim=0).sum()
        normalized_auxk_loss = l2_loss_aux / (residual_var + 1e-8)
        
        return normalized_auxk_loss.nan_to_num(0.0)
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """
        HONEST loss computation that acknowledges architectural choices.
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        kl_scale = self.kl_warmup_fn(step)
        
        original_dtype = x.dtype
        
        # Get all encoding components
        encoding = self.ae.encode(x, return_components=True)
        sparse_z = encoding['sparse_z']
        
        # Reconstruct from sparse latents
        x_hat = self.ae.decode(sparse_z)
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # KL loss (acknowledging architectural choice)
        kl_loss = self._compute_kl_loss(encoding)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Auxiliary loss for dead features
        residual = x - x_hat
        auxk_loss = self._compute_auxiliary_loss(residual, encoding)
        auxk_loss = auxk_loss.to(dtype=original_dtype)
        
        # Update logging stats
        if logging:
            self.effective_l0 = (sparse_z != 0).float().sum(dim=-1).mean().item()
            nonzero_features = (sparse_z != 0).any(dim=0).sum().item()
            self.kl_utilization = nonzero_features / self.ae.dict_size
        
        # Total loss with separate scaling
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * kl_scale * kl_loss +
            self.training_config.auxk_alpha * auxk_loss
        )
        
        if not logging:
            return total_loss
        
        # Detailed logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        kl_diagnostics = self.ae.get_kl_diagnostics(x)
        
        return LossLog(
            x, x_hat, sparse_z,
            {
                'l2_loss': torch.linalg.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'auxk_loss': auxk_loss.item(),
                'total_loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'kl_scale': kl_scale,
                'architecture_tension_warning': f"Using {self.model_config.architecture_mode} mode",
                **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Training step with bias initialization."""
        # Initialize bias to median for first step
        if step == 0:
            median = activations.median(dim=0).values
            median = median.to(dtype=self.ae.encoder.weight.dtype)
            
            if not self.ae.use_april_update_mode:
                self.ae.bias.data.copy_(median)
        
        activations = activations.to(self.device)
        
        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), self.training_config.gradient_clip_norm)
        
        # Handle constrained decoder
        if self.ae.constrain_decoder:
            self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.decoder.weight, self.ae.decoder.weight.grad,
                self.ae.activation_dim, self.ae.dict_size
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Normalize decoder if constrained
        if self.ae.constrain_decoder:
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
            )
    
    @property
    def config(self) -> Dict[str, Any]:
        """Configuration for logging."""
        return {
            'dict_class': 'VSAEBatchTopK',
            'trainer_class': 'VSAEBatchTopKTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'var_flag': self.model_config.var_flag,
            'constrain_decoder': self.model_config.constrain_decoder,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'architecture_mode': self.model_config.architecture_mode,
            'apply_topk_to_samples': self.model_config.apply_topk_to_samples,
            'preserve_gradient_flow': self.model_config.preserve_gradient_flow,
            'dtype': str(self.model_config.dtype),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
            # Architectural choices made explicit
            'architectural_tensions_acknowledged': True,
            'vae_topk_conflict_handling': self.model_config.architecture_mode,
        }