"""
Improved implementation of hybrid variational sparse autoencoder with batch top-k feature selection.

This module combines the variational approach from VSAEIso with batch top-k selection,
following better PyTorch practices and architecture patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, Dict, Any, List
from collections import namedtuple
from dataclasses import dataclass
import math

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
    """Configuration for VSAEBatchTopK model."""
    activation_dim: int
    dict_size: int
    k: int  # Number of top features to keep active
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    constrain_decoder: bool = True  # Whether to constrain decoder weights to unit norm
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        if self.k > self.dict_size:
            raise ValueError(f"k ({self.k}) cannot be larger than dict_size ({self.dict_size})")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")


@dataclass
class TrainingConfig:
    """Configuration for training the VSAEBatchTopK."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    dead_feature_threshold: int = 10_000_000
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


class VSAEBatchTopK(Dictionary, nn.Module):
    """
    A hybrid variational autoencoder with batch top-k feature selection.
    
    Combines the variational approach from VSAEIsoGaussian with the more efficient
    batch top-k feature selection mechanism, following improved architecture patterns.
    """

    def __init__(self, config: VSAEBatchTopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.constrain_decoder = config.constrain_decoder
        
        # Register k as a buffer so it's saved with the model
        self.register_buffer("k", torch.tensor(config.k, dtype=torch.int))
        
        # Adaptive threshold for top-k feature selection
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        
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
            bias=False,
            dtype=self.config.dtype,
            device=self.config.device
        )
        self.b_dec = nn.Parameter(
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
            nn.init.zeros_(self.b_dec)
            
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)
    
    def encode(
        self, 
        x: torch.Tensor, 
        return_active: bool = False, 
        use_threshold: bool = False,
        output_log_var: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode a batch of activations.
        
        Args:
            x: Input activations of shape [batch_size, activation_dim]
            return_active: Whether to return indices of active features
            use_threshold: Whether to use the adaptive threshold instead of top-k
            output_log_var: Whether to output log variance (only used when var_flag=1)
            
        Returns:
            Encoded features and optionally log variances and active indices
        """
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Compute pre-activations
        pre_acts = self.encoder(x - self.b_dec)
        
        # For variational case, compute log variance if requested
        log_var = None
        if self.var_flag == 1 and output_log_var:
            log_var = F.relu(self.var_encoder(x - self.b_dec))
        
        # Apply ReLU activation
        post_relu_feat_acts = F.relu(pre_acts)
        
        # Apply feature selection (threshold or top-k)
        if use_threshold and self.threshold >= 0:
            # Use adaptive threshold
            encoded_acts = post_relu_feat_acts * (post_relu_feat_acts > self.threshold)
        else:
            # Apply batch top-k selection
            flattened_acts = post_relu_feat_acts.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)
            
            encoded_acts = (
                torch.zeros_like(post_relu_feat_acts.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts.shape)
            )
        
        # Build return tuple based on requested outputs
        outputs = [encoded_acts]
        
        if self.var_flag == 1 and output_log_var:
            outputs.append(log_var)
        
        if return_active:
            active_mask = encoded_acts.sum(0) > 0
            outputs.append(active_mask)
            outputs.append(post_relu_feat_acts)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply reparameterization trick for sampling from latent distribution."""
        if log_var is None:
            return mu
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Ensure output matches mu dtype
        return z.to(dtype=mu.dtype)
    
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode a batch of features."""
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        return self.decoder(f) + self.b_dec
    
    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False,
        use_threshold: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Input activations
            output_features: Whether to return features alongside reconstruction
            use_threshold: Whether to use threshold-based feature selection
            
        Returns:
            Reconstructed activations and optionally features
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Encode inputs
        if self.var_flag == 1:
            mu, log_var = self.encode(x, output_log_var=True, use_threshold=use_threshold)
            # Sample latents using reparameterization
            f = self.reparameterize(mu, log_var)
        else:
            f = self.encode(x, use_threshold=use_threshold)
            
        # Decode
        x_hat = self.decode(f)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        else:
            return x_hat
    
    def get_active_features_mask(self, features: torch.Tensor) -> torch.Tensor:
        """Get boolean mask of which features are active across the batch."""
        return (features.sum(0) > 0)
    
    def scale_biases(self, scale: float) -> None:
        """Scale all bias parameters by a given factor."""
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            self.b_dec.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
                
            # Also scale threshold if it's being used
            if self.threshold >= 0:
                self.threshold.mul_(scale)
    
    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm (only when constrain_decoder=True)."""
        if not self.constrain_decoder:
            return
            
        norms = torch.norm(self.decoder.weight, dim=0).to(
            dtype=self.decoder.weight.dtype, 
            device=self.decoder.weight.device
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
            return
            
        print("Normalizing decoder weights")

        # Test that normalization preserves output
        test_input = torch.randn(10, self.activation_dim, 
                                dtype=self.decoder.weight.dtype,
                                device=self.decoder.weight.device)
        initial_output = self(test_input)

        with torch.no_grad():
            self.decoder.weight.div_(norms)
            
            new_norms = torch.norm(self.decoder.weight, dim=0)
            assert torch.allclose(new_norms, torch.ones_like(new_norms), atol=1e-6)

            self.encoder.weight.mul_(norms[:, None])
            self.encoder.bias.mul_(norms)

        new_output = self(test_input)
        assert torch.allclose(initial_output, new_output, atol=1e-4)
    
    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[VSAEBatchTopKConfig] = None,
        k: Optional[int] = None,
        var_flag: Optional[int] = None,
        constrain_decoder: Optional[bool] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> 'VSAEBatchTopK':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        
        if config is None:
            # Try to reconstruct config from checkpoint or provided parameters
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            
            # Get k from parameter or checkpoint
            if k is not None:
                k_val = k
            elif "k" in state_dict:
                k_val = state_dict["k"].item()
            else:
                raise ValueError("k must be provided either as parameter or in checkpoint")
            
            # Auto-detect other parameters if not provided
            if var_flag is None:
                var_flag = 1 if "var_encoder.weight" in state_dict else 0
            
            if constrain_decoder is None:
                decoder_norms = torch.norm(state_dict["decoder.weight"], dim=0)
                constrain_decoder = torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-3)
            
            config = VSAEBatchTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k_val,
                var_flag=var_flag,
                constrain_decoder=constrain_decoder,
                dtype=dtype,
                device=device
            )
        
        model = cls(config)
        
        # Filter state_dict to only include keys that are in the model
        model_keys = set(model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Handle missing keys (e.g., var_encoder when loading var_flag=0 model as var_flag=1)
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if config.var_flag == 1 and "var_encoder.weight" in missing_keys:
                print("Initializing missing variance encoder parameters")
                with torch.no_grad():
                    nn.init.kaiming_uniform_(model.var_encoder.weight)
                    nn.init.zeros_(model.var_encoder.bias)
        
        if device is not None:
            model = model.to(device=device, dtype=dtype)
            
        return model


class DeadFeatureTracker:
    """Tracks dead features for auxiliary loss computation."""
    
    def __init__(self, dict_size: int, threshold: int, device: torch.device):
        self.threshold = threshold
        self.num_tokens_since_fired = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
    
    def update(self, active_features: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Update dead feature tracking and return dead feature mask."""
        # Update counters
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[active_features] = 0
        
        # Return dead feature mask
        return self.num_tokens_since_fired >= self.threshold
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about dead features."""
        dead_mask = self.num_tokens_since_fired >= self.threshold
        return {
            "dead_features": int(dead_mask.sum()),
            "alive_features": int((~dead_mask).sum()),
            "total_features": len(self.num_tokens_since_fired)
        }


class VSAEBatchTopKTrainer(SAETrainer):
    """
    Improved trainer for the VSAEBatchTopK model.
    
    Features:
    - Clean separation of concerns
    - Proper loss computation with KL divergence and auxiliary loss
    - Dead feature tracking and reactivation
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        model_config: VSAEBatchTopKConfig = None,
        training_config: TrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        var_flag: Optional[int] = None,
        constrain_decoder: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None or training_config is None:
            if model_config is None:
                if activation_dim is None or dict_size is None or k is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
                
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = VSAEBatchTopKConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    k=k,
                    var_flag=var_flag or 0,
                    constrain_decoder=constrain_decoder if constrain_decoder is not None else True,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = TrainingConfig(
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
        self.wandb_name = wandb_name or "VSAEBatchTopKTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAEBatchTopK(model_config)
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
        
        # Initialize dead feature tracking
        self.dead_feature_tracker = DeadFeatureTracker(
            model_config.dict_size,
            training_config.dead_feature_threshold,
            self.device
        )
        
        # Auxiliary loss parameters
        self.top_k_aux = model_config.activation_dim // 2  # Heuristic from BatchTopK paper
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "auxk_loss_raw", "threshold_value"]
        self.effective_l0 = 0.0
        self.dead_features = 0
        self.auxk_loss_raw = 0.0
        self.threshold_value = -1.0
    
    def _update_threshold(self, f: torch.Tensor) -> None:
        """Update the adaptive threshold based on the minimum non-zero activation."""
        with torch.no_grad():
            active = f[f > 0]
            
            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=torch.float32)
                
            if self.ae.threshold < 0:
                self.ae.threshold.copy_(min_activation)
            else:
                beta = self.training_config.threshold_beta
                self.ae.threshold.mul_(beta).add_(min_activation, alpha=(1 - beta))
                
            self.threshold_value = self.ae.threshold.item()
    
    def _compute_auxiliary_loss(
        self, 
        residual: torch.Tensor, 
        post_relu_acts: torch.Tensor,
        active_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss to help reactivate dead features."""
        # Update dead feature tracking
        num_tokens = residual.size(0)
        dead_mask = self.dead_feature_tracker.update(active_features, num_tokens)
        
        # Update logging stats
        stats = self.dead_feature_tracker.get_stats()
        self.dead_features = stats["dead_features"]
        
        if self.dead_features == 0:
            self.auxk_loss_raw = 0.0
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
        
        # Compute auxiliary loss for dead features
        k_aux = min(self.top_k_aux, self.dead_features)
        
        # Filter to just dead features
        auxk_latents = torch.where(dead_mask[None], post_relu_acts, -torch.inf)
        
        # Top-k dead latents
        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
        
        auxk_buffer = torch.zeros_like(post_relu_acts)
        auxk_acts_active = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)
        
        # Reconstruct using auxiliary features (use decoder directly, not decode method)
        x_reconstruct_aux = self.ae.decoder(auxk_acts_active)
        l2_loss_aux = (
            (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
        )
        
        self.auxk_loss_raw = l2_loss_aux.item()
        
        # Normalization from OpenAI implementation
        residual_mu = residual.mean(dim=0, keepdim=True)
        loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
        normalized_auxk_loss = l2_loss_aux / (loss_denom + 1e-8)
        
        return normalized_auxk_loss.nan_to_num(0.0)
    
    def _geometric_median(self, points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
        """Compute the geometric median of a set of points."""
        guess = points.mean(dim=0)
        prev = torch.zeros_like(guess)
        weights = torch.ones(len(points), device=points.device)
        
        for _ in range(max_iter):
            prev.copy_(guess)
            diffs = points - guess
            norms = torch.norm(diffs, dim=1)
            # Avoid division by zero
            weights = 1 / (norms + 1e-8)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if torch.norm(guess - prev) < tol:
                break
                
        return guess
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
        """Compute total loss with all components."""
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Encode inputs
        if self.ae.var_flag == 1:
            # Get mean, log variance, and active indices
            if logging:
                mu, log_var, active_features, post_relu_acts = self.ae.encode(
                    x, 
                    output_log_var=True, 
                    return_active=True,
                    use_threshold=(step > self.training_config.threshold_start_step)
                )
            else:
                mu, log_var = self.ae.encode(
                    x,
                    output_log_var=True,
                    use_threshold=(step > self.training_config.threshold_start_step)
                )
                
            # Sample using reparameterization trick
            f = self.ae.reparameterize(mu, log_var)
            
            # Calculate KL divergence (full KL for learned variance)
            kl_base = 0.5 * (torch.exp(log_var) + mu.pow(2) - 1 - log_var).sum(dim=-1)
        else:
            # Get features and active indices with fixed variance
            if logging:
                f, active_features, post_relu_acts = self.ae.encode(
                    x,
                    return_active=True,
                    use_threshold=(step > self.training_config.threshold_start_step)
                )
            else:
                f = self.ae.encode(
                    x, 
                    use_threshold=(step > self.training_config.threshold_start_step)
                )
                
            # Simplified KL divergence for fixed unit variance
            kl_base = 0.5 * f.pow(2).sum(dim=-1)
        
        # Update adaptive threshold if past threshold_start_step
        if step > self.training_config.threshold_start_step:
            self._update_threshold(f)
        
        # Decode to get reconstruction
        x_hat = self.ae.decode(f)
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Compute reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # Calculate KL loss, weighted by decoder norms if not constraining
        if self.ae.constrain_decoder:
            kl_loss = kl_base.mean()
        else:
            # Weight by decoder norms (April update approach)
            decoder_norms = torch.norm(self.ae.decoder.weight, p=2, dim=0)
            decoder_norms = decoder_norms.to(dtype=kl_base.dtype)
            kl_loss = kl_base.mean() * decoder_norms.mean()
        
        # Compute auxiliary loss for dead features
        if logging:
            residual = x - x_hat
            auxk_loss = self._compute_auxiliary_loss(residual, post_relu_acts, active_features)
            # Update effective L0
            self.effective_l0 = (f != 0).float().sum(dim=-1).mean().item()
        else:
            residual = x - x_hat
            # Get post_relu_acts for auxiliary loss
            _, _, post_relu_acts = self.ae.encode(x, return_active=True, use_threshold=False)
            active_features = self.ae.get_active_features_mask(f)
            auxk_loss = self._compute_auxiliary_loss(residual, post_relu_acts, active_features)
        
        # Convert all losses to original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        kl_loss = kl_loss.to(dtype=original_dtype)
        auxk_loss = auxk_loss.to(dtype=original_dtype)
        
        # Combine all losses
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * sparsity_scale * kl_loss + 
            self.training_config.auxk_alpha * auxk_loss
        )
        
        if not logging:
            return total_loss
        
        # Return detailed loss information for logging
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        return LossLog(
            x, x_hat, f,
            {
                'l2_loss': torch.linalg.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'auxk_loss': auxk_loss.item(),
                'total_loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
            }
        )
    
    def update(self, step: int, activations: torch.Tensor) -> None:
        """Perform one training step."""
        # Special case for first step: initialize bias to geometric median
        if step == 0:
            median = self._geometric_median(activations)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data.copy_(median)
        
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Handle constrained decoder weights
        if self.ae.constrain_decoder:
            self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.decoder.weight,
                self.ae.decoder.weight.grad,
                self.ae.activation_dim,
                self.ae.dict_size
            )
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Normalize decoder weights if constrained
        if self.ae.constrain_decoder:
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight,
                self.ae.activation_dim,
                self.ae.dict_size
            )
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving (JSON serializable)."""
        return {
            'dict_class': 'VSAEBatchTopK',
            'trainer_class': 'VSAEBatchTopKTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'var_flag': self.model_config.var_flag,
            'constrain_decoder': self.model_config.constrain_decoder,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'threshold_beta': self.training_config.threshold_beta,
            'threshold_start_step': self.training_config.threshold_start_step,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }