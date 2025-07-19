"""
Block Diagonal Laplacian Top-K Sparse Autoencoder (BDL-SAE-TopK) - NO VAE Components

This module implements a standard (non-variational) sparse autoencoder that combines:
1. Standard autoencoder architecture (NO reparameterization, NO KL loss)
2. Top-K structured sparsity mechanism  
3. Block diagonal Laplacian regularization for structured smoothness

The key difference from the VAE version is simplicity:
- NO variance encoder
- NO reparameterization trick  
- NO KL divergence loss
- JUST: reconstruction + Top-K sparsity + Laplacian smoothness

This allows isolating the effect of Laplacian regularization without VAE complexity.

Mathematical Framework:
- Standard encoding: z = encoder(x) (deterministic)
- Top-K sparsity: Select k largest |z| values while preserving signs
- Laplacian regularization: R(z) = Σᵢ zᵢᵀ Lᵢ zᵢ
- Total loss: MSE + λ₁·R(z) + λ₂·AuxLoss (NO KL term)

Ablation Study Position:
- Pure TopK SAE: ✓ (baseline)
- VSAE TopK: ✓ (adds VAE)
- Laplacian TopK VAE: ✓ (adds VAE + Laplacian)  
- Laplacian TopK SAE: ← THIS (adds just Laplacian)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
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

# Import the BlockDiagonalLaplacian from our VAE version
from .laplace_topk import BlockDiagonalLaplacian


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
class LaplacianTopKSAEConfig:
    """Configuration for Block Diagonal Laplacian TopK SAE (NO VAE components)."""
    activation_dim: int
    dict_size: int
    k: int
    
    # Block diagonal Laplacian parameters
    block_sizes: Optional[List[int]] = None  # Auto-computed if None
    laplacian_type: str = "chain"  # 'chain', 'complete', 'ring'
    
    # Standard autoencoder parameters (NO VAE)
    use_april_update_mode: bool = True
    
    # System parameters
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class LaplacianTopKSAE(Dictionary, nn.Module):
    """
    Block Diagonal Laplacian Top-K Sparse Autoencoder (NO VAE).
    
    Combines standard autoencoders with Top-K sparsity and block diagonal
    Laplacian regularization for structured, interpretable representations.
    
    Architecture (SIMPLIFIED compared to VAE version):
    1. Standard encoder: x → z (deterministic, no sampling)
    2. Top-K selection: sparse_z = TopK(|z|) but preserve signs  
    3. Laplacian regularization: R(z) = Σᵢ zᵢᵀLᵢzᵢ (applied to full z)
    4. Standard decoder: sparse_z → x̂
    
    NO VAE COMPONENTS:
    - No variance encoder
    - No reparameterization trick
    - No KL divergence loss
    """

    def __init__(self, config: LaplacianTopKSAEConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.use_april_update_mode = config.use_april_update_mode
        
        # Register k as a buffer so it's saved with the model
        self.register_buffer("k", torch.tensor(config.k, dtype=torch.int))
        
        # Initialize block diagonal Laplacian
        self.laplacian = BlockDiagonalLaplacian(
            dict_size=config.dict_size,
            block_sizes=config.block_sizes,
            laplacian_type=config.laplacian_type,
            device=config.get_device(),
            dtype=config.dtype
        )
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers (SIMPLIFIED - no variance encoder)."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # Main encoder and decoder (NO variance encoder)
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
        return_topk: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        SIMPLIFIED encode: Standard AE + Top-K (NO VAE components).
        
        Args:
            x: Input activation tensor
            return_topk: Whether to return top-k indices and values
            
        Returns:
            sparse_features, latent_z, [top_indices, selected_vals]
        """
        x_processed = self._preprocess_input(x)
        
        # Step 1: Standard deterministic encoding (NO sampling)
        z = self.encoder(x_processed)  # Just deterministic features
            
        # Step 2: Apply Top-K sparsity on absolute values (preserves gradients)
        z_abs = torch.abs(z)  # Use absolute values for selection
        top_vals_abs, top_indices = z_abs.topk(self.k.item(), sorted=False, dim=-1)
        
        # Step 3: Create sparse feature vector using original latent values
        sparse_features = torch.zeros_like(z)
        # Gather the original values (not absolute values) for the selected indices
        selected_vals = torch.gather(z, dim=-1, index=top_indices)
        sparse_features = sparse_features.scatter_(dim=-1, index=top_indices, src=selected_vals)
        
        if return_topk:
            return sparse_features, z, top_indices, selected_vals
        else:
            return sparse_features, z

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

    def forward(self, x: torch.Tensor, output_features: bool = False):
        """
        SIMPLIFIED forward pass: Standard AE + Top-K + Laplacian.
        
        Args:
            x: Input tensor
            output_features: Whether to return features as well
            
        Returns:
            Reconstructed tensor and optionally features
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Encode: get sparse features from Top-K selection
        sparse_features, latent_z = self.encode(x)
        
        # Decode using sparse features
        x_hat = self.decode(sparse_features)
        
        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if not output_features:
            return x_hat
        else:
            sparse_features = sparse_features.to(dtype=original_dtype)
            return x_hat, sparse_features

    def compute_laplacian_regularization(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute block diagonal Laplacian regularization on latent features.
        
        This encourages smoothness within feature blocks while maintaining
        independence between blocks.
        
        Args:
            z: Latent features [batch_size, dict_size]
            
        Returns:
            Laplacian regularization loss
        """
        return self.laplacian.compute_regularization(z)

    def get_laplacian_diagnostics(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get detailed Laplacian regularization diagnostics."""
        with torch.no_grad():
            sparse_features, latent_z = self.encode(x)
            
            # Compute regularization on full latent space
            laplacian_reg = self.compute_laplacian_regularization(latent_z)
            
            # Block-wise analysis
            block_info = self.laplacian.get_block_info()
            block_stats = {}
            
            for i, (start, size) in enumerate(zip(self.laplacian.block_starts, self.laplacian.block_sizes)):
                if size > 1:  # Skip single-feature blocks
                    z_block = latent_z[:, start:start + size]
                    block_stats[f'block_{i}_mean'] = z_block.mean().item()
                    block_stats[f'block_{i}_std'] = z_block.std().item()
                    block_stats[f'block_{i}_sparsity'] = (z_block != 0).float().mean().item()
            
            return {
                'laplacian_reg': laplacian_reg.item(),
                'block_info': block_info,
                **block_stats
            }

    def analyze_block_structure(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze how features are organized within blocks.
        
        Returns detailed information about block-wise activation patterns.
        """
        with torch.no_grad():
            sparse_features, latent_z, top_indices, selected_vals = self.encode(
                x, return_topk=True
            )
            
            analysis = {
                'total_features': self.dict_size,
                'k_value': self.k.item(),
                'n_blocks': self.laplacian.n_blocks,
                'block_sizes': self.laplacian.block_sizes,
                'laplacian_type': self.laplacian.laplacian_type
            }
            
            # Analyze Top-K selection within each block
            block_selection_stats = {}
            for i, (start, size) in enumerate(zip(self.laplacian.block_starts, self.laplacian.block_sizes)):
                # Count how many features from this block were selected
                block_mask = (top_indices >= start) & (top_indices < start + size)
                selected_from_block = block_mask.sum(dim=1).float()  # [batch_size]
                
                block_selection_stats[f'block_{i}'] = {
                    'size': size,
                    'start_idx': start,
                    'avg_selected': selected_from_block.mean().item(),
                    'max_selected': selected_from_block.max().item(),
                    'selection_rate': (selected_from_block.mean() / size).item()
                }
            
            analysis['block_selection_stats'] = block_selection_stats
            
            # Overall statistics
            z_abs = torch.abs(latent_z)
            analysis.update({
                'sparsity_ratio': (sparse_features != 0).float().mean().item(),
                'latent_z_mean': latent_z.mean().item(),
                'latent_z_std': latent_z.std().item(),
                'topk_threshold': z_abs.topk(self.k.item(), dim=-1)[0][:, -1].mean().item(),
                'laplacian_reg': self.compute_laplacian_regularization(latent_z).item()
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

    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
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
        config: Optional[LaplacianTopKSAEConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True
    ) -> 'LaplacianTopKSAE':
        """
        Load model with robust error handling and auto-detection.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            
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
            
            # Get k value from state_dict or use default
            k = state_dict["k"].item() if "k" in state_dict else max(1, dict_size // 10)
            
            config = LaplacianTopKSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
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
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict: {e}")
        
        # Normalize decoder if requested
        if normalize_decoder:
            try:
                model.normalize_decoder()
            except Exception as e:
                print(f"Warning: Could not normalize decoder weights: {e}")
        
        # Move to target device and dtype
        if device is not None or dtype != model.config.dtype:
            model = model.to(device=device, dtype=dtype)
        
        return model
    
    @staticmethod
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: LaplacianTopKSAEConfig) -> Dict[str, torch.Tensor]:
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
        
        # Convert other buffers
        if "k" in state_dict:
            converted["k"] = state_dict["k"]
        
        return converted


@dataclass 
class LaplacianTopKSAETrainingConfig:
    """Training configuration for Block Diagonal Laplacian TopK SAE (NO VAE)."""
    steps: int
    lr: float = 5e-4
    laplacian_coeff: float = 1.0  # Block diagonal Laplacian regularization coefficient
    laplacian_warmup_steps: Optional[int] = None  # Warmup for Laplacian regularization
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient for dead feature resurrection
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    dead_feature_threshold: int = 10_000_000  # Steps before considering a feature "dead"
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Set derived configuration values."""
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        # Laplacian warmup
        if self.laplacian_warmup_steps is None:
            self.laplacian_warmup_steps = int(0.15 * self.steps)  # 15% of training

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


class LaplacianTopKSAETrainer(SAETrainer):
    """
    Trainer for Block Diagonal Laplacian Top-K Sparse Autoencoder (NO VAE).
    
    SIMPLIFIED compared to VAE version - optimizes only:
    1. Reconstruction loss (MSE)
    2. Block diagonal Laplacian regularization 
    3. Auxiliary loss for dead feature resurrection
    
    NO VAE COMPONENTS:
    - No KL divergence loss
    - No variance-related parameters or annealing
    """
    
    def __init__(
        self,
        model_config: Optional[LaplacianTopKSAEConfig] = None,
        training_config: Optional[LaplacianTopKSAETrainingConfig] = None,
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
        laplacian_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        block_sizes: Optional[List[int]] = None,
        laplacian_type: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None or k is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size + k")
            
            device_obj = torch.device(device) if device else None
            model_config = LaplacianTopKSAEConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                block_sizes=block_sizes,
                laplacian_type=laplacian_type or "chain",
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else True,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = LaplacianTopKSAETrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                laplacian_coeff=laplacian_coeff or 1.0,
                auxk_alpha=auxk_alpha or 1/32,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "LaplacianTopKSAETrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = LaplacianTopKSAE(model_config)
        self.ae.to(self.device)
        
        # Top-K specific tracking
        self.top_k_aux = model_config.activation_dim // 2  # Heuristic from TopK paper
        
        # Initialize dead feature tracking
        self.dead_feature_tracker = DeadFeatureTracker(
            model_config.dict_size,
            training_config.dead_feature_threshold,
            self.device
        )
        
        # Logging parameters (SIMPLIFIED - no KL terms)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss", "laplacian_reg"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.laplacian_reg = -1

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
        # Laplacian annealing function (reuse existing implementation)
        from .laplace_topk import get_kl_warmup_fn
        self.laplacian_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.laplacian_warmup_steps
        )

    def get_auxiliary_loss(self, residual_BD: torch.Tensor, sparse_features_BF: torch.Tensor, latent_z_BF: torch.Tensor):
        """
        Auxiliary loss computation using actual sparse features.
        """
        # Update dead feature tracking based on sparse features
        active_features = (sparse_features_BF.sum(0) > 0)
        num_tokens = sparse_features_BF.size(0)
        dead_features = self.dead_feature_tracker.update(active_features, num_tokens)
        
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # Select auxiliary features from the FULL latent space
            auxk_latents = torch.where(dead_features[None], torch.abs(latent_z_BF), -torch.inf)

            # Top-k dead latents by absolute value
            auxk_abs_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            
            # Get the original values (preserving sign) for the selected indices
            auxk_original_vals = torch.gather(latent_z_BF, dim=-1, index=auxk_indices)

            # Create auxiliary feature vector using original values
            auxk_buffer_BF = torch.zeros_like(latent_z_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_original_vals)

            # Decode auxiliary features
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
        SIMPLIFIED loss computation (NO KL loss):
        
        The loss includes:
        1. Reconstruction error (MSE) 
        2. Block diagonal Laplacian regularization (with annealing)
        3. Auxiliary loss for reviving dead sparse features
        
        NO VAE COMPONENTS - much simpler than VAE version!
        """
        sparsity_scale = self.sparsity_warmup_fn(step)  # For any L1 penalties (unused here)
        laplacian_scale = self.laplacian_warmup_fn(step)  # Laplacian annealing
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)

        # SIMPLIFIED encode (no VAE complexity)
        sparse_features, latent_z, top_indices, selected_vals = self.ae.encode(
            x, return_topk=True
        )

        # Decode using sparse features and calculate reconstruction error
        x_hat = self.ae.decode(sparse_features)
        residual = x - x_hat
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l2_loss = torch.linalg.norm(residual, dim=-1).mean()

        # Update the effective L0 (should be exactly K)
        self.effective_l0 = self.ae.k.item()
        
        # Block diagonal Laplacian regularization
        laplacian_reg = self.ae.compute_laplacian_regularization(latent_z)
        laplacian_reg = laplacian_reg.to(dtype=original_dtype)
        self.laplacian_reg = laplacian_reg.item()
        
        # Auxiliary loss using sparse features and full latent space
        auxk_loss = self.get_auxiliary_loss(residual.detach(), sparse_features, latent_z) if self.training_config.auxk_alpha > 0 else 0
        
        # Total loss with proper scaling (SIMPLIFIED - no KL term)
        recon_loss = recon_loss.to(dtype=original_dtype)
        if isinstance(auxk_loss, torch.Tensor):
            auxk_loss = auxk_loss.to(dtype=original_dtype)
        else:
            auxk_loss = torch.tensor(auxk_loss, dtype=original_dtype, device=x.device)
        
        total_loss = (
            recon_loss + 
            self.training_config.laplacian_coeff * laplacian_scale * laplacian_reg +  # Laplacian regularization
            self.training_config.auxk_alpha * auxk_loss
            # NO KL TERM - this is the key simplification!
        )

        if not logging:
            return total_loss
        else:
            # Convert outputs back to original dtype for logging
            x_hat = x_hat.to(dtype=original_dtype)
            sparse_features = sparse_features.to(dtype=original_dtype)
            
            # Get additional diagnostics
            laplacian_diagnostics = self.ae.get_laplacian_diagnostics(x.to(dtype=original_dtype))
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x.to(dtype=original_dtype), x_hat, sparse_features,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'laplacian_reg': laplacian_reg.item(),
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    'loss': total_loss.item(),
                    'sparsity_scale': sparsity_scale,
                    'laplacian_scale': laplacian_scale,
                    'effective_l0': self.effective_l0,
                    # Additional diagnostics (SIMPLIFIED - no KL)
                    'sparse_feature_norm': sparse_features.norm(dim=-1).mean().item(),
                    'latent_z_norm': latent_z.norm(dim=-1).mean().item(),
                    'selected_vals_mean': selected_vals.mean().item(),
                    'selected_vals_std': selected_vals.std().item(),
                    **{k: v if not torch.is_tensor(v) else v.item() for k, v in laplacian_diagnostics.items() if k != 'block_info'}
                }
            )
        
    def update(self, step: int, activations: torch.Tensor):
        """Training update with improved stability and cleaner architecture."""
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
            'dict_class': 'LaplacianTopKSAE',
            'trainer_class': 'LaplacianTopKSAETrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'block_sizes': self.model_config.block_sizes,
            'laplacian_type': self.model_config.laplacian_type,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (SIMPLIFIED - no KL terms)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'laplacian_coeff': self.training_config.laplacian_coeff,
            'laplacian_warmup_steps': self.training_config.laplacian_warmup_steps,
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
            # Architecture notes
            'architecture_version': 'block_diagonal_laplacian_topk_sae_v1',
            'features': [
                'block_diagonal_laplacian_regularization',
                'topk_sparsity',
                'standard_autoencoder',  # NO VAE
                'structured_sparsity_within_blocks',
                'hierarchical_feature_organization',
                'laplacian_annealing'
            ],
            'simplifications': [
                'no_vae_components',
                'no_kl_divergence',
                'no_reparameterization',
                'no_variance_encoder'
            ]
        }
