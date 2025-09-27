"""
Hybrid Variational Sparse Autoencoder with Top-K activation mechanism.

This module combines the variational approach from VSAEIso with the structured sparsity of TopK.
The architecture maintains proper VAE properties while enforcing structured sparsity through
Top-K selection on latent activations.

Key architectural principles:
- Latent variables remain unconstrained throughout VAE computation
- Top-K selection based on absolute values while preserving original sign for gradients  
- Clean separation between VAE latent space and sparse selection
- KL computation on actual latent variables being used
- Auxiliary loss mechanism for dead feature resurrection

Flow: x → encoder → μ,σ² → reparameterize → z → Top-K(|z|) → sparse_z → decode → x̂
                                                ↳ KL(z)     ↳ L2(x,x̂)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable
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
class VSAETopKConfig:
    """Configuration for VSAETopK model."""
    activation_dim: int
    dict_size: int
    k: int
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


class VSAETopK(Dictionary, nn.Module):
    """
    Hybrid dictionary that combines variational autoencoders with Top-K sparsity mechanism.
    
    This architecture maintains proper VAE properties while enforcing structured sparsity.
    The key insight is to apply Top-K selection on absolute values of latent variables
    while preserving the original signed values for gradient computation.
    
    Key features:
    - Unconstrained VAE latent space throughout computation
    - Top-K selection preserves gradient flow
    - Clean separation between VAE properties and sparsity constraints
    - Enhanced numerical stability and dtype handling
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

    def encode(
        self, 
        x: torch.Tensor, 
        return_topk: bool = False,
        training: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode input through VAE latent space with Top-K sparsity selection.
        
        The encoding process maintains proper VAE properties by keeping latent variables
        unconstrained throughout computation. Top-K selection is applied on absolute values
        to preserve gradient flow while enforcing structured sparsity.
        
        Args:
            x: Input activation tensor
            return_topk: Whether to return top-k indices and values
            training: Whether in training mode (affects sampling)
            
        Returns:
            sparse_features, latent_z, mu, log_var, [top_indices, selected_vals]
        """
        x_processed = self._preprocess_input(x)
        
        # Step 1: Encode to latent distribution parameters
        mu = self.encoder(x_processed)
        mu = F.relu(mu)  # Ensure positive mean for stability
        
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)
        
        # Step 2: Sample from latent distribution (reparameterization trick)
        if training and self.var_flag == 1 and log_var is not None:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu  # Deterministic latents
            
        # Step 3: Apply Top-K sparsity while preserving gradients
        # Key insight: select based on magnitude, but keep original values
        z_abs = torch.abs(z)
        top_vals_abs, top_indices = z_abs.topk(self.k.item(), sorted=False, dim=-1)
        
        # Step 4: Create sparse feature vector using original latent values
        sparse_features = torch.zeros_like(z)
        # Gather the original values (not absolute values) for the selected indices
        selected_vals = torch.gather(z, dim=-1, index=top_indices)
        sparse_features = sparse_features.scatter_(dim=-1, index=top_indices, src=selected_vals)
        
        if return_topk:
            return sparse_features, z, mu, log_var, top_indices, selected_vals
        else:
            return sparse_features, z, mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Reparameterization trick with consistent log_var = log(σ²) interpretation.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # Conservative clamping range: log_var ∈ [-6, 2] means σ² ∈ [0.002, 7.4]
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var)) = sqrt(σ²)
        std = torch.sqrt(torch.exp(log_var_clamped))
        
        # Sample noise and reparameterize
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.
        
        Args:
            f: Latent features [batch_size, dict_size]
            
        Returns:
            x_hat: Reconstructed activations [batch_size, activation_dim]
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias

    def forward(self, x: torch.Tensor, output_features: bool = False, training: bool = True):
        """
        Forward pass through the hybrid VAE + Top-K architecture.
        
        Args:
            x: Input tensor
            output_features: Whether to return features as well
            training: Whether in training mode (affects sampling behavior)
            
        Returns:
            Reconstructed tensor and optionally features
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Encode: get sparse features from Top-K selection on VAE latents
        sparse_features, latent_z, mu, log_var = self.encode(x, training=training)
        
        # Decode using sparse features
        x_hat = self.decode(sparse_features)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if not output_features:
            return x_hat
        else:
            sparse_features = sparse_features.to(dtype=original_dtype)
            return x_hat, sparse_features

    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed KL diagnostics for monitoring training.
        
        Returns:
            Dictionary with KL components and statistics
        """
        with torch.no_grad():
            _, latent_z, mu, log_var = self.encode(x, training=False)
            
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
                    'latent_z_mean': latent_z.mean(),
                    'latent_z_std': latent_z.std(),
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
                    'latent_z_mean': latent_z.mean(),
                    'latent_z_std': latent_z.std(),
                }

    def analyze_sparse_pattern(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the sparse activation pattern for debugging and understanding.
        
        Returns detailed information about how Top-K selection works on this input.
        """
        with torch.no_grad():
            sparse_features, latent_z, mu, log_var, top_indices, selected_vals = self.encode(
                x, return_topk=True, training=False
            )
            
            # Compute various statistics
            z_abs = torch.abs(latent_z)
            sparsity_ratio = (sparse_features != 0).float().mean()
            
            analysis = {
                # Basic statistics
                'batch_size': x.shape[0],
                'k_value': self.k.item(),
                'actual_sparsity_ratio': sparsity_ratio.item(),
                'theoretical_sparsity_ratio': self.k.item() / self.dict_size,
                
                # Latent space statistics
                'latent_z_mean': latent_z.mean().item(),
                'latent_z_std': latent_z.std().item(),
                'latent_z_abs_mean': z_abs.mean().item(),
                'mu_mean': mu.mean().item(),
                'mu_std': mu.std().item(),
                
                # Selection statistics
                'selected_vals_mean': selected_vals.mean().item(),
                'selected_vals_std': selected_vals.std().item(),
                'selected_vals_min': selected_vals.min().item(),
                'selected_vals_max': selected_vals.max().item(),
                'negative_selected_ratio': (selected_vals < 0).float().mean().item(),
                
                # Top-K threshold (minimum absolute value selected)
                'topk_threshold': z_abs.topk(self.k.item(), dim=-1)[0][:, -1].mean().item(),
                
                # Feature activation patterns
                'active_features_per_sample': (sparse_features != 0).sum(dim=-1).float().mean().item(),
                'features_used_in_batch': (sparse_features.sum(dim=0) != 0).sum().item(),
            }
            
            if self.var_flag == 1 and log_var is not None:
                analysis.update({
                    'log_var_mean': log_var.mean().item(),
                    'log_var_std': log_var.std().item(),
                    'variance_mean': torch.exp(log_var).mean().item(),
                })
            
            return analysis

    def scale_biases(self, scale: float):
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
        config: Optional[VSAETopKConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None
    ) -> 'VSAETopK':
        """
        Load a pretrained model with robust error handling and auto-detection.
        
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
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: VSAETopKConfig) -> Dict[str, torch.Tensor]:
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
        
        # Convert other buffers
        if "k" in state_dict:
            converted["k"] = state_dict["k"]
        
        return converted


@dataclass 
class VSAETopKTrainingConfig:
    """Training configuration with proper scaling separation and clean architecture."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient for dead feature resurrection
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None  # For any actual sparsity penalties (unused in this model)
    decay_start: Optional[int] = None
    dead_feature_threshold: int = 10_000_000  # Steps before considering a feature "dead"
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
    Trainer for the hybrid VSAETopK model with improved stability and monitoring.
    
    Features:
    - Separate KL and sparsity annealing schedules
    - Clean KL divergence computation on actual latent variables
    - Proper reparameterization trick handling
    - Enhanced numerical stability and dtype handling
    - Better gradient clipping and optimization
    - Detailed KL and sparsity diagnostics
    """
    
    def __init__(
        self,
        model_config: Optional[VSAETopKConfig] = None,
        training_config: Optional[VSAETopKTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility parameters
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        k: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        var_flag: Optional[int] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
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
        self.wandb_name = wandb_name or "VSAETopKTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
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
        # Separate KL annealing function
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )

    def _compute_kl_loss(self, latent_z: torch.Tensor, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        KL divergence computation using actual latent variables.
        
        Computes KL[q(z|x) || p(z)] where q(z|x) = N(μ, σ²) and p(z) = N(0, I)
        
        Args:
            latent_z: The actual latent variables being used (from reparameterization)
            mu: Mean parameters from encoder
            log_var: Log variance parameters (None for fixed variance)
            
        Returns:
            KL divergence loss
        """
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for learned variance
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
            
            # KL[q(z|x) || p(z)] = 0.5 * Σ[μ² + σ² - 1 - log(σ²)]
            kl_per_sample = 0.5 * torch.sum(
                mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                dim=1
            )
        else:
            # Fixed variance case: KL = 0.5 * ||latent_z||² 
            # Use actual latent variables for more accurate gradients
            latent_z_clamped = torch.clamp(latent_z, min=-10.0, max=10.0)
            kl_per_sample = 0.5 * torch.sum(latent_z_clamped.pow(2), dim=1)
        
        # Average over batch
        kl_loss = kl_per_sample.mean()
        
        # Ensure KL is non-negative
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        return kl_loss

    def get_auxiliary_loss(self, residual_BD: torch.Tensor, sparse_features_BF: torch.Tensor, latent_z_BF: torch.Tensor):
        """
        Auxiliary loss computation using actual sparse features.
        
        Args:
            residual_BD: Reconstruction residual (x - x_hat)
            sparse_features_BF: The sparse features being used for reconstruction
            latent_z_BF: The full latent variables (before Top-K selection)
        """
        # Update dead feature tracking based on sparse features (what's actually being used)
        active_features = (sparse_features_BF.sum(0) > 0)
        num_tokens = sparse_features_BF.size(0)
        dead_features = self.dead_feature_tracker.update(active_features, num_tokens)
        
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # Select auxiliary features from the full latent space
            # This allows dead features to potentially be resurrected with negative values
            auxk_latents = torch.where(dead_features[None], torch.abs(latent_z_BF), -torch.inf)

            # Top-k dead latents by absolute value
            auxk_abs_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            
            # Get the original values (preserving sign) for the selected indices
            auxk_original_vals = torch.gather(latent_z_BF, dim=-1, index=auxk_indices)

            # Create auxiliary feature vector using original values
            auxk_buffer_BF = torch.zeros_like(latent_z_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_original_vals)

            # Decode auxiliary features directly
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
        Loss computation with proper VAE + Top-K architecture.
        
        The loss includes:
        1. Reconstruction error (MSE) 
        2. KL divergence on actual latent variables (with separate annealing)
        3. Auxiliary loss for reviving dead sparse features
        """
        sparsity_scale = self.sparsity_warmup_fn(step)  # For any L1 penalties (unused here)
        kl_scale = self.kl_warmup_fn(step)  # Separate KL annealing
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)

        # Use the cleaner encode method
        sparse_features, latent_z, mu, log_var, top_indices, selected_vals = self.ae.encode(
            x, return_topk=True, training=self.ae.training
        )

        # Decode using sparse features and calculate reconstruction error
        x_hat = self.ae.decode(sparse_features)
        residual = x - x_hat
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l2_loss = torch.linalg.norm(residual, dim=-1).mean()

        # Update the effective L0 (should be exactly K)
        self.effective_l0 = self.ae.k.item()
        
        # KL divergence loss using actual latent variables
        kl_loss = self._compute_kl_loss(latent_z, mu, log_var)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # Auxiliary loss using sparse features and full latent space
        auxk_loss = self.get_auxiliary_loss(residual.detach(), sparse_features, latent_z) if self.training_config.auxk_alpha > 0 else 0
        
        # Total loss with proper scaling
        recon_loss = recon_loss.to(dtype=original_dtype)
        if isinstance(auxk_loss, torch.Tensor):
            auxk_loss = auxk_loss.to(dtype=original_dtype)
        else:
            auxk_loss = torch.tensor(auxk_loss, dtype=original_dtype, device=x.device)
        
        total_loss = (
            recon_loss + 
            self.training_config.kl_coeff * kl_scale * kl_loss +  # Separate KL scaling
            self.training_config.auxk_alpha * auxk_loss
        )

        if not logging:
            return total_loss
        else:
            # Convert outputs back to original dtype for logging
            x_hat = x_hat.to(dtype=original_dtype)
            sparse_features = sparse_features.to(dtype=original_dtype)
            
            # Get additional diagnostics
            kl_diagnostics = self.ae.get_kl_diagnostics(x.to(dtype=original_dtype))
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x.to(dtype=original_dtype), x_hat, sparse_features,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    'loss': total_loss.item(),
                    'sparsity_scale': sparsity_scale,
                    'kl_scale': kl_scale,  # Separate from sparsity scaling
                    'effective_l0': self.effective_l0,
                    # Additional VAE + Top-K diagnostics
                    'sparse_feature_norm': sparse_features.norm(dim=-1).mean().item(),
                    'latent_z_norm': latent_z.norm(dim=-1).mean().item(),
                    'selected_vals_mean': selected_vals.mean().item(),
                    'selected_vals_std': selected_vals.std().item(),
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
                }
            )
        
    def update(self, step: int, activations: torch.Tensor):
        """Training update with improved stability and clean architecture."""
        activations = activations.to(self.device)
        
        # Initialize decoder bias with geometric median on first step
        if step == 0 and not self.ae.use_april_update_mode:
            median = geometric_median(activations)
            median = median.to(self.ae.bias.dtype)
            self.ae.bias.data = median

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
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
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
            'auxk_alpha': self.training_config.auxk_alpha,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'dead_feature_threshold': self.training_config.dead_feature_threshold,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
            # Architecture metadata
            'architecture_version': 'vae_topk_v2',
            'key_features': [
                'unconstrained_latent_space',
                'topk_on_absolute_values', 
                'proper_kl_computation',
                'clean_vae_sparsity_separation',
                'auxiliary_loss_mechanism',
                'enhanced_gradient_flow'
            ]
        }