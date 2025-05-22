"""
Improved implementation of hybrid Variational Sparse Autoencoder with Top-K activation mechanism.

This module combines the variational approach from VSAEIso with the structured sparsity of TopK,
following better PyTorch practices similar to the Matryoshka implementation.
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
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions
)
from ..config import DEBUG


@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


@dataclass
class VSAETopKConfig:
    """Configuration for VSAETopK model."""
    activation_dim: int
    dict_size: int
    k: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


class VSAETopK(Dictionary, nn.Module):
    """
    A hybrid dictionary that combines the variational approach from VSAEIso 
    with the structured Top-K sparsity mechanism.
    
    This model uses:
    1. The variational sampling approach for learning feature distributions
    2. The Top-K activation mechanism to enforce structured sparsity
    3. Support for both fixed and learned variance
    """

    def __init__(self, config: VSAETopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.use_april_update_mode = config.use_april_update_mode
        
        # Register k as a buffer so it's saved with the model
        self.register_buffer("k", torch.tensor(config.k, dtype=torch.int))
        # Threshold for activation filtering (used as alternative to explicit top-k)
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        # Initialize encoder and decoder
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
        
        # In standard mode, we use a separate bias parameter
        if not self.use_april_update_mode:
            self.b_dec = nn.Parameter(
                torch.zeros(
                    self.activation_dim, 
                    dtype=self.config.dtype,
                    device=self.config.device
                )
            )
            
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
        # Initialize weights
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
                # b_dec is already initialized to zeros in _init_layers
                pass
                
            # Initialize variance encoder if present
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)

    def encode(
        self, 
        x: torch.Tensor, 
        output_log_var: bool = False,
        return_topk: bool = False,
        use_threshold: bool = False,
        training: bool = True
    ):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input activation tensor
            output_log_var: Whether to return log variance
            return_topk: Whether to return top-k indices and values
            use_threshold: Whether to use threshold for filtering instead of explicit top-k
            training: Whether in training mode (affects sampling)
            
        Returns:
            Encoded features and optionally log variance, top-k indices, and pre-filter activations
        """
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        # Get the bias term for subtraction
        if self.use_april_update_mode:
            bias_term = 0
        else:
            bias_term = self.b_dec
            
        # Compute the mean activations
        mu = self.encoder(x - bias_term)
        
        # Compute the log variance if needed
        if output_log_var or self.var_flag == 1:
            if self.var_flag == 1:
                log_var = self.var_encoder(x - bias_term)
            else:
                log_var = torch.zeros_like(mu)
                
        # Apply ReLU to mean activations
        post_relu_mu = F.relu(mu)
            
        # If in training mode and using variational approach, sample from distribution
        if training and self.var_flag == 1:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            # Sample using reparameterization trick
            z = mu + eps * std
            post_relu_z = F.relu(z)
        else:
            post_relu_z = post_relu_mu
            
        # Apply top-k or threshold filtering
        if use_threshold:
            encoded_acts = post_relu_z * (post_relu_z > self.threshold)
            if return_topk:
                post_topk = post_relu_z.topk(self.k.item(), sorted=False, dim=-1)
                return encoded_acts, post_topk.values, post_topk.indices, post_relu_z
            else:
                return encoded_acts
        else:
            # Explicit top-k selection
            post_topk = post_relu_z.topk(self.k.item(), sorted=False, dim=-1)
            tops_acts = post_topk.values
            top_indices = post_topk.indices

            # Create sparse feature vector with only top-k activations
            buffer = torch.zeros_like(post_relu_z)
            encoded_acts = buffer.scatter_(dim=-1, index=top_indices, src=tops_acts)
            
            if return_topk:
                if output_log_var:
                    return encoded_acts, log_var, tops_acts, top_indices, post_relu_z
                else:
                    return encoded_acts, tops_acts, top_indices, post_relu_z
            else:
                if output_log_var:
                    return encoded_acts, log_var
                else:
                    return encoded_acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode a dictionary vector f.
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False, training: bool = True):
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Input tensor
            output_features: Whether to return features as well
            training: Whether in training mode (affects sampling behavior)
            
        Returns:
            Reconstructed tensor and optionally features
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        encoded_acts = self.encode(x, training=training)
        x_hat = self.decode(encoded_acts)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if not output_features:
            return x_hat
        else:
            encoded_acts = encoded_acts.to(dtype=original_dtype)
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        """
        Scale all bias parameters by a given factor.
        """
        with torch.no_grad():
            self.encoder.bias.mul_(scale)
            if self.use_april_update_mode:
                self.decoder.bias.mul_(scale)
            else:
                self.b_dec.mul_(scale)
                
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)
                
            if self.threshold >= 0:
                self.threshold.mul_(scale)

    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[VSAETopKConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = False,  # Default to False for safety
    ) -> 'VSAETopK':
        """Load pretrained model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        if config is None:
            # Try to reconstruct config from checkpoint
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint['state_dict']
            
            # Auto-detect var_flag from state_dict
            has_var_encoder = "var_encoder.weight" in state_dict or "W_enc_var" in state_dict
            var_flag = 1 if has_var_encoder else 0
            
            # Determine dimensions, mode, and k value based on state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
                use_april_update_mode = "decoder.bias" in state_dict
            else:
                # Handle older format with W_enc, W_dec parameters
                activation_dim, dict_size = state_dict["W_enc"].shape if "W_enc" in state_dict else state_dict["encoder.weight"].T.shape
                use_april_update_mode = "b_dec" in state_dict
                
                # Convert parameter names if needed
                if "W_enc" in state_dict:
                    converted_dict = {}
                    converted_dict["encoder.weight"] = state_dict["W_enc"].T
                    converted_dict["encoder.bias"] = state_dict["b_enc"]
                    converted_dict["decoder.weight"] = state_dict["W_dec"].T
                    
                    if use_april_update_mode:
                        converted_dict["decoder.bias"] = state_dict["b_dec"]
                    else:
                        converted_dict["b_dec"] = state_dict["b_dec"]
                        
                    if var_flag == 1 and "W_enc_var" in state_dict:
                        converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                        converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                        
                    state_dict = converted_dict
            
            # Get k value from state_dict or use default
            k = state_dict["k"].item() if "k" in state_dict else max(1, dict_size // 10)
            
            config = VSAETopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                var_flag=var_flag,
                use_april_update_mode=use_april_update_mode,
                dtype=dtype,
                device=device
            )
        
        # Create model
        model = cls(config)
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Move to target device/dtype
        if device is not None:
            model = model.to(device=device, dtype=dtype)
            
        return model


@dataclass 
class VSAETopKTrainingConfig:
    """Configuration for training the VSAETopK."""
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


class VSAETopKTrainer(SAETrainer):
    """
    Trainer for the hybrid VSAETopK model with improved architecture.
    
    Features:
    - Clean separation of concerns
    - Proper loss computation with KL divergence and auxiliary loss
    - Dead feature tracking and threshold management
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        model_config: VSAETopKConfig = None,
        training_config: VSAETopKTrainingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility with trainSAE
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        var_flag: Optional[int] = None,
        use_april_update_mode: Optional[bool] = None,
        threshold_beta: Optional[float] = None,
        threshold_start_step: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility - if individual parameters are passed, create configs
        if model_config is None or training_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None or k is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
                
                # Set defaults
                var_flag = var_flag or 0
                use_april_update_mode = use_april_update_mode if use_april_update_mode is not None else True
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = VSAETopKConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    k=k,
                    var_flag=var_flag,
                    use_april_update_mode=use_april_update_mode,
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
                    threshold_beta=threshold_beta or 0.999,
                    threshold_start_step=threshold_start_step or 1000,
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAETopKTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAETopK(model_config)
        self.ae.to(self.device)
        
        # Top-K specific tracking
        self.top_k_aux = model_config.activation_dim // 2  # Heuristic from TopK paper
        
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
        
    def update_threshold(self, top_acts_BK: torch.Tensor):
        """Update the dynamic threshold for activation filtering."""
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=torch.float32)
            min_activation = min_activations.mean()

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.training_config.threshold_beta * self.ae.threshold) + (
                    (1 - self.training_config.threshold_beta) * min_activation
                )
                
    def get_auxiliary_loss(self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor):
        """Calculate auxiliary loss to resurrect dead features."""
        # Update dead feature tracking
        active_features = (post_relu_acts_BF.sum(0) > 0)
        num_tokens = post_relu_acts_BF.size(0)
        dead_features = self.dead_feature_tracker.update(active_features, num_tokens)
        
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = torch.where(dead_features[None], post_relu_acts_BF, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux.item()

            # Normalization from OpenAI implementation
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / (loss_denom + 1e-8)

            return normalized_auxk_loss
        else:
            self.pre_norm_auxk_loss = 0.0
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)
        
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """
        Calculate the hybrid loss combining KL divergence and reconstruction error.
        
        The loss includes:
        1. Reconstruction error (MSE)
        2. KL divergence for variational component
        3. Auxiliary loss for reviving dead features
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)

        # Encode with the hybrid model
        if self.ae.var_flag == 1:
            f, log_var, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                x, output_log_var=True, return_topk=True, use_threshold=False
            )
            # KL divergence components from variational sampling
            mu = post_relu_acts_BF
        else:
            f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                x, return_topk=True, use_threshold=False
            )
            mu = post_relu_acts_BF
            log_var = None
            
        # Update threshold if past start step
        if step > self.training_config.threshold_start_step:
            self.update_threshold(top_acts_BK)

        # Decode and calculate reconstruction error
        x_hat = self.ae.decode(f)
        residual = x - x_hat
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l2_loss = torch.linalg.norm(residual, dim=-1).mean()

        # Update the effective L0 (should just be K)
        self.effective_l0 = top_acts_BK.size(1)
        
        # Get decoder norms for KL weighting
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
        
        # KL divergence loss (using activated features)
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for learned variance
            kl_base = 0.5 * (torch.exp(log_var) + mu.pow(2) - 1 - log_var).sum(dim=-1)
        else:
            # Simplified KL divergence for fixed variance
            kl_base = 0.5 * (mu.pow(2)).sum(dim=-1)
        
        kl_loss = kl_base.mean() * decoder_norms.mean()
        
        # Auxiliary loss to resurrect dead features
        auxk_loss = self.get_auxiliary_loss(residual.detach(), post_relu_acts_BF) if self.training_config.auxk_alpha > 0 else 0
        
        # Total loss - ensure all components are in original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        kl_loss = kl_loss.to(dtype=original_dtype)
        if isinstance(auxk_loss, torch.Tensor):
            auxk_loss = auxk_loss.to(dtype=original_dtype)
        else:
            auxk_loss = torch.tensor(auxk_loss, dtype=original_dtype, device=x.device)
        
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * sparsity_scale * kl_loss + 
            self.training_config.auxk_alpha * auxk_loss
        )

        if not logging:
            return total_loss
        else:
            # Convert x_hat back to original dtype for logging
            x_hat = x_hat.to(dtype=original_dtype)
            f = f.to(dtype=original_dtype)
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x.to(dtype=original_dtype), x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    'loss': total_loss.item()
                }
            )
        
    def update(self, step: int, activations: torch.Tensor):
        """Perform a single training update."""
        activations = activations.to(self.device)
        
        # Initialize decoder bias with geometric median on first step
        if step == 0 and not self.ae.use_april_update_mode:
            median = geometric_median(activations)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # Zero gradients
        self.optimizer.zero_grad()
        
        # Calculate loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        # Step optimizer and scheduler
        self.optimizer.step()
        self.scheduler.step()
        
        # If using the original approach (not April update), normalize decoder
        if not self.ae.use_april_update_mode:
            with torch.no_grad():
                self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                    self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
                )

    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration dictionary for logging/saving (JSON serializable)."""
        return {
            'dict_class': 'VSAETopK',
            'trainer_class': 'VSAETopKTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'var_flag': self.model_config.var_flag,
            'use_april_update_mode': self.model_config.use_april_update_mode,
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