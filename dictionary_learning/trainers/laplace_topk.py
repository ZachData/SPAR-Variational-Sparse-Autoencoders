"""
Block Diagonal Laplacian Top-K Variational Sparse Autoencoder (BDL-VSAE-TopK)

This module implements a hybrid variational sparse autoencoder that combines:
1. Variational autoencoder framework with reparameterization
2. Top-K structured sparsity mechanism  
3. Block diagonal Laplacian regularization for structured smoothness

The block diagonal Laplacian encourages smoothness within feature blocks while
maintaining independence between blocks, creating hierarchical feature organization.

Mathematical Framework:
- VAE latent space: z ~ N(μ, σ²)
- Top-K sparsity: Select k largest |z| values  
- Laplacian regularization: R(z) = Σᵢ zᵢᵀ Lᵢ zᵢ
- Block structure: Features partitioned into semantic blocks

Key Improvements over Standard VSAE-TopK:
✅ Block diagonal Laplacian regularization for structured sparsity
✅ Hierarchical feature organization within and across blocks
✅ Configurable block sizes and Laplacian structures
✅ Enhanced interpretability through block-wise feature grouping
✅ Improved feature diversity and reduced redundancy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from collections import namedtuple
from dataclasses import dataclass
import math

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


class BlockDiagonalLaplacian:
    """
    Implements block diagonal Laplacian regularization for structured sparsity.
    
    Mathematical formulation:
    L = block_diag(L₁, L₂, ..., Lₖ)
    R(z) = zᵀLz = Σᵢ zᵢᵀLᵢzᵢ
    
    Each block Lᵢ encourages smoothness within that feature block.
    """
    
    def __init__(
        self, 
        dict_size: int,
        block_sizes: Optional[List[int]] = None,
        laplacian_type: str = "chain",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize block diagonal Laplacian.
        
        Args:
            dict_size: Total dictionary size
            block_sizes: List of block sizes (must sum to dict_size)
            laplacian_type: Type of Laplacian ('chain', 'complete', 'ring')
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.dict_size = dict_size
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        self.laplacian_type = laplacian_type
        
        # Set default block sizes if not provided
        if block_sizes is None:
            # Create roughly equal blocks of size ~64
            block_size = max(8, min(64, dict_size // 8))
            n_full_blocks = dict_size // block_size
            remainder = dict_size % block_size
            
            self.block_sizes = [block_size] * n_full_blocks
            if remainder > 0:
                self.block_sizes.append(remainder)
        else:
            assert sum(block_sizes) == dict_size, f"Block sizes {block_sizes} must sum to dict_size {dict_size}"
            self.block_sizes = block_sizes
        
        self.n_blocks = len(self.block_sizes)
        
        # Create block start indices
        self.block_starts = [0]
        for size in self.block_sizes[:-1]:
            self.block_starts.append(self.block_starts[-1] + size)
        
        # Precompute Laplacian matrices for each block
        self.laplacians = self._create_laplacians()
    
    def _create_laplacians(self) -> List[torch.Tensor]:
        """Create Laplacian matrix for each block."""
        laplacians = []
        
        for block_size in self.block_sizes:
            if block_size == 1:
                # Single feature block - no regularization needed
                L = torch.zeros(1, 1, device=self.device, dtype=self.dtype)
            elif self.laplacian_type == "chain":
                L = self._create_chain_laplacian(block_size)
            elif self.laplacian_type == "complete":
                L = self._create_complete_laplacian(block_size)
            elif self.laplacian_type == "ring":
                L = self._create_ring_laplacian(block_size)
            else:
                raise ValueError(f"Unknown Laplacian type: {self.laplacian_type}")
            
            laplacians.append(L)
        
        return laplacians
    
    def _create_chain_laplacian(self, n: int) -> torch.Tensor:
        """
        Create chain graph Laplacian: features connected in sequence.
        Encourages smoothness along the chain.
        
        Adjacency: A[i,i+1] = A[i+1,i] = 1
        Degree: D[i,i] = number of neighbors
        Laplacian: L = D - A
        """
        A = torch.zeros(n, n, device=self.device, dtype=self.dtype)
        
        # Connect adjacent features
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        
        # Degree matrix
        D = torch.diag(A.sum(dim=1))
        
        return D - A
    
    def _create_complete_laplacian(self, n: int) -> torch.Tensor:
        """
        Create complete graph Laplacian: all features connected.
        Encourages all features in block to have similar magnitudes.
        
        Adjacency: A[i,j] = 1 for i ≠ j
        Degree: D[i,i] = n-1
        """
        A = torch.ones(n, n, device=self.device, dtype=self.dtype)
        A.fill_diagonal_(0.0)  # No self-connections
        
        D = torch.diag(A.sum(dim=1))
        
        return D - A
    
    def _create_ring_laplacian(self, n: int) -> torch.Tensor:
        """
        Create ring graph Laplacian: features connected in circle.
        Like chain but with wraparound connection.
        """
        if n < 3:
            return self._create_chain_laplacian(n)
        
        A = torch.zeros(n, n, device=self.device, dtype=self.dtype)
        
        # Connect adjacent features
        for i in range(n):
            A[i, (i + 1) % n] = 1.0
            A[(i + 1) % n, i] = 1.0
        
        D = torch.diag(A.sum(dim=1))
        
        return D - A
    
    def compute_regularization(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute block diagonal Laplacian regularization.
        
        Args:
            z: Feature tensor [batch_size, dict_size]
            
        Returns:
            Regularization loss scalar
        """
        total_reg = 0.0
        batch_size = z.size(0)
        
        for i, (start, size, L) in enumerate(zip(self.block_starts, self.block_sizes, self.laplacians)):
            if size == 1:
                continue  # Skip single-feature blocks
            
            # Extract block features
            z_block = z[:, start:start + size]  # [batch_size, block_size]
            
            # Compute quadratic form: z_block^T L z_block for each sample
            # L is [block_size, block_size], z_block is [batch_size, block_size]
            reg_per_sample = torch.sum(z_block * torch.matmul(z_block, L.T), dim=1)  # [batch_size]
            total_reg += reg_per_sample.mean()  # Average over batch
        
        return total_reg
    
    def get_block_info(self) -> Dict[str, Any]:
        """Get information about the block structure."""
        return {
            'n_blocks': self.n_blocks,
            'block_sizes': self.block_sizes,
            'block_starts': self.block_starts,
            'laplacian_type': self.laplacian_type,
            'total_features': self.dict_size
        }


@dataclass
class LaplacianTopKConfig:
    """Configuration for Block Diagonal Laplacian TopK model."""
    activation_dim: int
    dict_size: int
    k: int
    
    # Block diagonal Laplacian parameters
    block_sizes: Optional[List[int]] = None  # Auto-computed if None
    laplacian_type: str = "chain"  # 'chain', 'complete', 'ring'
    
    # VAE parameters
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    # System parameters
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device


class LaplacianTopK(Dictionary, nn.Module):
    """
    Block Diagonal Laplacian Top-K Variational Sparse Autoencoder.
    
    Combines variational autoencoders with Top-K sparsity and block diagonal
    Laplacian regularization for structured, interpretable representations.
    
    Architecture:
    1. VAE encoder: x → μ, σ² (or just μ for fixed variance)
    2. Reparameterization: z ~ N(μ, σ²) 
    3. Top-K selection: sparse_z = TopK(|z|) but preserve signs
    4. Laplacian regularization: R(z) = Σᵢ zᵢᵀLᵢzᵢ (applied to full z)
    5. VAE decoder: sparse_z → x̂
    """

    def __init__(self, config: LaplacianTopKConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
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
        Encode with VAE + Top-K + block diagonal Laplacian structure.
        
        Args:
            x: Input activation tensor
            return_topk: Whether to return top-k indices and values
            training: Whether in training mode (affects sampling)
            
        Returns:
            sparse_features, latent_z, mu, log_var, [top_indices, selected_vals]
        """
        x_processed = self._preprocess_input(x)
        
        # Step 1: Encode to latent distribution parameters (unconstrained)
        mu = self.encoder(x_processed)  # Unconstrained mean
        
        log_var = None
        if self.var_flag == 1:
            log_var = self.var_encoder(x_processed)  # Direct log variance encoding
        
        # Step 2: Sample from latent distribution (reparameterization trick)
        if training and self.var_flag == 1 and log_var is not None:
            z = self.reparameterize(mu, log_var)  # Sampled latents (unconstrained)
        else:
            z = mu  # Deterministic latents
            
        # Step 3: Apply Top-K sparsity on absolute values (preserves gradients)
        z_abs = torch.abs(z)  # Use absolute values for selection
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
            mu: Mean of latent distribution (unconstrained)
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features (unconstrained)
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # Conservative clamping range
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var)) = sqrt(σ²)
        std = torch.sqrt(torch.exp(log_var_clamped))
        
        # Sample noise
        eps = torch.randn_like(std)
        
        # Reparameterize
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
        Forward pass with block diagonal Laplacian Top-K VAE.
        
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

    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed KL diagnostics for monitoring training."""
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

    def get_laplacian_diagnostics(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get detailed Laplacian regularization diagnostics."""
        with torch.no_grad():
            sparse_features, latent_z, mu, log_var = self.encode(x, training=False)
            
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
            sparse_features, latent_z, mu, log_var, top_indices, selected_vals = self.encode(
                x, return_topk=True, training=False
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
                
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)

    def normalize_decoder(self) -> None:
        """Normalize decoder weights to have unit norm."""
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
        config: Optional[LaplacianTopKConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None
    ) -> 'LaplacianTopK':
        """
        Load model with robust error handling and auto-detection.
        
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
            
            config = LaplacianTopKConfig(
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
        
        # Normalize decoder if requested
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
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: LaplacianTopKConfig) -> Dict[str, torch.Tensor]:
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
class LaplacianTopKTrainingConfig:
    """Training configuration for Block Diagonal Laplacian TopK."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
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
        # KL annealing to prevent posterior collapse
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.steps)
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


class LaplacianTopKTrainer(SAETrainer):
    """
    Trainer for Block Diagonal Laplacian Top-K Variational Sparse Autoencoder.
    
    Optimizes the combination of:
    1. Reconstruction loss (MSE)
    2. KL divergence loss (with separate annealing)
    3. Block diagonal Laplacian regularization (with separate annealing)
    4. Auxiliary loss for dead feature resurrection
    """
    
    def __init__(
        self,
        model_config: Optional[LaplacianTopKConfig] = None,
        training_config: Optional[LaplacianTopKTrainingConfig] = None,
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
        laplacian_coeff: Optional[float] = None,
        auxk_alpha: Optional[float] = None,
        var_flag: Optional[int] = None,
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
            model_config = LaplacianTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                block_sizes=block_sizes,
                laplacian_type=laplacian_type or "chain",
                var_flag=var_flag or 0,
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else True,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = LaplacianTopKTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=kl_coeff or 500.0,
                laplacian_coeff=laplacian_coeff or 1.0,
                auxk_alpha=auxk_alpha or 1/32,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "LaplacianTopKTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = LaplacianTopK(model_config)
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
        # Separate annealing functions
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        self.laplacian_warmup_fn = get_kl_warmup_fn(  # Reuse the same function
            training_config.steps,
            training_config.laplacian_warmup_steps
        )

    def _compute_kl_loss(self, latent_z: torch.Tensor, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence loss using actual latent variables.
        
        Computes KL[q(z|x) || p(z)] where q(z|x) = N(μ, σ²) and p(z) = N(0, I)
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
        Complete loss computation with block diagonal Laplacian regularization.
        
        The loss includes:
        1. Reconstruction error (MSE) 
        2. KL divergence on actual latent variables (with separate annealing)
        3. Block diagonal Laplacian regularization (with separate annealing)
        4. Auxiliary loss for reviving dead sparse features
        """
        sparsity_scale = self.sparsity_warmup_fn(step)  # For any L1 penalties (unused here)
        kl_scale = self.kl_warmup_fn(step)  # Separate KL annealing
        laplacian_scale = self.laplacian_warmup_fn(step)  # Separate Laplacian annealing
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.ae.encoder.weight.dtype)

        # Encode with the new cleaner method
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
        
        # NEW: Block diagonal Laplacian regularization
        laplacian_reg = self.ae.compute_laplacian_regularization(latent_z)
        laplacian_reg = laplacian_reg.to(dtype=original_dtype)
        self.laplacian_reg = laplacian_reg.item()
        
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
            self.training_config.laplacian_coeff * laplacian_scale * laplacian_reg +  # NEW: Laplacian regularization
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
            laplacian_diagnostics = self.ae.get_laplacian_diagnostics(x.to(dtype=original_dtype))
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x.to(dtype=original_dtype), x_hat, sparse_features,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'laplacian_reg': laplacian_reg.item(),  # NEW
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, torch.Tensor) else auxk_loss,
                    'loss': total_loss.item(),
                    'sparsity_scale': sparsity_scale,
                    'kl_scale': kl_scale,  # Separate from sparsity scaling
                    'laplacian_scale': laplacian_scale,  # NEW: Separate Laplacian scaling
                    'effective_l0': self.effective_l0,
                    # Additional VAE + Top-K + Laplacian diagnostics
                    'sparse_feature_norm': sparse_features.norm(dim=-1).mean().item(),
                    'latent_z_norm': latent_z.norm(dim=-1).mean().item(),
                    'selected_vals_mean': selected_vals.mean().item(),
                    'selected_vals_std': selected_vals.std().item(),
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()},
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
            'dict_class': 'LaplacianTopK',
            'trainer_class': 'LaplacianTopKTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'k': self.model_config.k,
            'block_sizes': self.model_config.block_sizes,
            'laplacian_type': self.model_config.laplacian_type,
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
            'architecture_version': 'block_diagonal_laplacian_topk_v1',
            'features': [
                'block_diagonal_laplacian_regularization',
                'unconstrained_latent_space',
                'topk_on_absolute_values', 
                'proper_kl_computation',
                'structured_sparsity_within_blocks',
                'hierarchical_feature_organization',
                'separate_kl_laplacian_annealing'
            ]
        }
