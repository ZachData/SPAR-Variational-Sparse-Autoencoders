"""
Fixed VSAEMultiGaussian implementation with all improvements from VSAEIso.

Key improvements:
- Unconstrained mean encoding (no ReLU on μ)
- Proper log variance handling (no ReLU, can be negative)
- Conservative clamping ranges for numerical stability
- Separate KL and sparsity scaling with KL annealing
- Clean KL divergence computation without decoder norm weighting
- Enhanced numerical stability and dtype consistency
- Better initialization and configuration management
- Optimized multivariate Gaussian prior handling
- Memory efficiency optimizations for large dict_size
- Robust correlation matrix validation and computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, List, Dict, Any, Callable
from collections import namedtuple
from dataclasses import dataclass

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
)


@dataclass
class VSAEMultiConfig:
    """Configuration for VSAEMultiGaussian model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    corr_rate: float = 0.5  # Correlation rate for building default correlation matrix
    corr_matrix: Optional[torch.Tensor] = None  # Custom correlation matrix
    use_april_update_mode: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def get_device(self) -> torch.device:
        """Get the device, defaulting to CUDA if available."""
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.corr_rate is not None and abs(self.corr_rate) >= 1.0:
            raise ValueError(f"corr_rate must be in (-1, 1), got {self.corr_rate}")
        
        if self.var_flag not in [0, 1]:
            raise ValueError(f"var_flag must be 0 or 1, got {self.var_flag}")
        
        if self.dict_size > 20000:
            print(f"Warning: Very large dict_size ({self.dict_size}) may cause memory issues")
        
        if self.corr_matrix is not None:
            if self.corr_matrix.shape != (self.dict_size, self.dict_size):
                raise ValueError(f"corr_matrix shape {self.corr_matrix.shape} doesn't match dict_size {self.dict_size}")
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        # Encoder + decoder weights
        main_weights = 2 * self.activation_dim * self.dict_size
        
        # Variance encoder if used
        var_weights = self.activation_dim * self.dict_size if self.var_flag == 1 else 0
        
        # Correlation matrix + precision matrix (only if not independent)
        correlation_weights = 2 * self.dict_size ** 2 if abs(self.corr_rate) > 1e-6 else 0
        
        total_params = main_weights + var_weights + correlation_weights
        
        # Assuming float32 (4 bytes per parameter)
        return (total_params * 4) / (1024 ** 2)


class VSAEMultiGaussian(Dictionary, nn.Module):
    """
    Fixed Variational Sparse Autoencoder with multivariate Gaussian prior.
    
    Key improvements:
    - Unconstrained mean encoding (no ReLU on μ)
    - Proper log variance handling (no ReLU, can be negative)  
    - Conservative clamping for numerical stability
    - Clean KL divergence computation
    - Enhanced multivariate Gaussian prior support
    - Memory optimization for independent case
    - Robust correlation matrix handling
    """

    def __init__(self, config: VSAEMultiConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        self.corr_rate = config.corr_rate
        self.use_april_update_mode = config.use_april_update_mode
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
        
        # Initialize correlation structure
        self._init_correlation_matrix()
        
        # Validate configuration and warn about potential issues
        self.validate_config()
    
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
    
    def _init_correlation_matrix(self) -> None:
        """Initialize the correlation matrix for the multivariate Gaussian prior."""
        if self.config.corr_matrix is not None:
            self.corr_matrix = self.config.corr_matrix.to(
                device=self.config.get_device(),
                dtype=self.config.dtype
            )
        else:
            self.corr_matrix = self._build_correlation_matrix(self.corr_rate)
        
        # Validate correlation matrix
        self._validate_correlation_matrix()
        
        # Pre-compute precision matrix and log determinant for efficiency
        self._precompute_prior_quantities()
    
    def _build_correlation_matrix(self, corr_rate: float) -> torch.Tensor:
        """Build correlation matrix based on the given correlation rate."""
        device = self.config.get_device()
        dtype = self.config.dtype
        
        # FIXED: Better handling of edge cases and memory efficiency
        if abs(corr_rate) < 1e-6:
            # Independent case - just return identity (more efficient)
            return torch.eye(self.dict_size, dtype=dtype, device=device)
        
        if abs(corr_rate) >= 0.99:
            # Near-singular case - clamp to avoid numerical issues
            corr_rate = 0.99 * (1 if corr_rate > 0 else -1)
            print(f"Warning: Correlation rate too close to ±1, clamped to {corr_rate}")
        
        # Memory warning for large matrices
        if self.dict_size > 10000:
            memory_mb = (self.dict_size ** 2 * 4) / (1024 ** 2)
            print(f"Warning: Large dict_size ({self.dict_size}) will use {memory_mb:.1f} MB for correlation matrix")
        
        # Create correlation matrix with specified off-diagonal correlation
        corr_matrix = torch.full(
            (self.dict_size, self.dict_size), 
            corr_rate, 
            dtype=dtype, 
            device=device
        )
        # Diagonal elements are 1
        torch.diagonal(corr_matrix)[:] = 1.0
        
        return corr_matrix
    
    def _validate_correlation_matrix(self) -> None:
        """Validate and fix the correlation matrix if needed."""
        with torch.no_grad():
            # Check for reasonable size
            if self.dict_size > 20000:
                raise ValueError(f"dict_size {self.dict_size} too large for dense correlation matrix. Consider using corr_rate=0.0 for independence.")
            
            # Ensure symmetry
            if not torch.allclose(self.corr_matrix, self.corr_matrix.t(), atol=1e-6):
                print("Warning: Correlation matrix not symmetric, symmetrizing it")
                self.corr_matrix = 0.5 * (self.corr_matrix + self.corr_matrix.t())
            
            # Check diagonal is 1
            diagonal = torch.diagonal(self.corr_matrix)
            if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6):
                print("Warning: Correlation matrix diagonal not 1, fixing it")
                torch.diagonal(self.corr_matrix)[:] = 1.0
            
            # Check off-diagonal values are reasonable
            off_diag = self.corr_matrix.clone()
            off_diag.fill_diagonal_(0)
            max_corr = off_diag.abs().max()
            if max_corr >= 0.99:
                print(f"Warning: Very high correlation {max_corr:.3f} detected, may cause numerical issues")
            
            # Ensure positive definiteness with robust approach
            try:
                eigenvals = torch.linalg.eigvals(self.corr_matrix).real
                min_eigenval = eigenvals.min()
                
                if min_eigenval <= 1e-6:
                    # Calculate required jitter more carefully
                    jitter_amount = max(1e-6 - min_eigenval + 1e-8, 1e-6)
                    print(f"Warning: Adding jitter {jitter_amount:.2e} to ensure positive definiteness")
                    self.corr_matrix += torch.eye(
                        self.dict_size, 
                        dtype=self.corr_matrix.dtype, 
                        device=self.corr_matrix.device
                    ) * jitter_amount
                    
                    # Verify it worked
                    new_eigenvals = torch.linalg.eigvals(self.corr_matrix).real
                    if new_eigenvals.min() <= 0:
                        raise ValueError("Could not make correlation matrix positive definite")
                        
            except Exception as e:
                print(f"Warning: Could not compute eigenvalues: {e}")
                # Fallback: add more aggressive jitter
                jitter = torch.eye(
                    self.dict_size,
                    dtype=self.corr_matrix.dtype,
                    device=self.corr_matrix.device
                ) * 1e-3
                self.corr_matrix += jitter
                print("Applied fallback jitter of 1e-3")
    
    def _precompute_prior_quantities(self) -> None:
        """Precompute precision matrix and log determinant for efficiency."""
        # Check if this is effectively an independent case
        identity = torch.eye(self.dict_size, device=self.corr_matrix.device, dtype=self.corr_matrix.dtype)
        
        if torch.allclose(self.corr_matrix, identity, atol=1e-6):
            # Independent case - much more efficient
            self.is_independent = True
            self.prior_precision = identity
            self.prior_log_det = torch.tensor(0.0, device=self.corr_matrix.device, dtype=self.corr_matrix.dtype)
            print("Using efficient independent prior (correlation matrix is identity)")
            return
        
        self.is_independent = False
        
        try:
            # For small matrices, direct inverse is fine
            if self.dict_size <= 1000:
                self.prior_precision = torch.linalg.inv(self.corr_matrix)
                self.prior_log_det = torch.logdet(self.corr_matrix)
            else:
                # For larger matrices, use more stable computation
                print(f"Computing inverse for large correlation matrix (size {self.dict_size})")
                # Use Cholesky decomposition for numerical stability
                try:
                    L = torch.linalg.cholesky(self.corr_matrix)
                    self.prior_precision = torch.cholesky_inverse(L)
                    # log det = 2 * sum(log(diag(L)))
                    self.prior_log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
                except Exception as e:
                    print(f"Cholesky failed: {e}, falling back to direct inverse")
                    self.prior_precision = torch.linalg.inv(self.corr_matrix)
                    self.prior_log_det = torch.logdet(self.corr_matrix)
                    
        except Exception as e:
            print(f"Warning: Could not compute prior quantities, adding more jitter: {e}")
            # Add more jitter and try again
            jitter = torch.eye(
                self.dict_size,
                dtype=self.corr_matrix.dtype,
                device=self.corr_matrix.device
            ) * 1e-3
            
            self.corr_matrix += jitter
            self.prior_precision = torch.linalg.inv(self.corr_matrix)
            self.prior_log_det = torch.logdet(self.corr_matrix)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input to handle bias subtraction in standard mode."""
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        if self.use_april_update_mode:
            return x
        else:
            return x - self.bias
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input to latent space.
        
        FIXED: No ReLU on mean - VAE means should be unconstrained for expressiveness
        
        Args:
            x: Input activations [batch_size, activation_dim]
            
        Returns:
            mu: Mean of latent distribution [batch_size, dict_size] (unconstrained)
            log_var: Log variance (None if var_flag=0) [batch_size, dict_size]
        """
        x_processed = self._preprocess_input(x)
        
        # FIXED: No ReLU constraint on mean - let it be unconstrained
        mu = self.encoder(x_processed)
        
        # Encode variance if learning it
        log_var = None
        if self.var_flag == 1:
            # FIXED: No ReLU on log_var - it can be negative
            log_var = self.var_encoder(x_processed)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply reparameterization trick with consistent log_var = log(σ²) interpretation.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance = log(σ²) (None for fixed variance)
            
        Returns:
            z: Sampled latent features
        """
        if log_var is None or self.var_flag == 0:
            return mu
        
        # FIXED: Conservative clamping range for numerical stability
        log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
        
        # Since log_var = log(σ²), we have σ = sqrt(exp(log_var))
        std = torch.sqrt(torch.exp(log_var_clamped))
        
        # Sample noise
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z.to(dtype=mu.dtype)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstruction.
        
        Args:
            z: Latent features [batch_size, dict_size]
            
        Returns:
            x_hat: Reconstructed activations [batch_size, activation_dim]
        """
        # Ensure z matches decoder weight dtype
        z = z.to(dtype=self.decoder.weight.dtype)
        
        if self.use_april_update_mode:
            return self.decoder(z)
        else:
            return self.decoder(z) + self.bias
    
    def forward(self, x: torch.Tensor, output_features: bool = False, ghost_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, activation_dim]
            output_features: Whether to return latent features
            ghost_mask: Not implemented for VSAEMultiGaussian (raises error if provided)
            
        Returns:
            x_hat: Reconstructed activations
            z: Latent features (if output_features=True)
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEMultiGaussian")
        
        # Store original dtype
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
    
    def compute_kl_divergence(self, mu: torch.Tensor, log_var: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence between q(z|x) and multivariate Gaussian prior p(z).
        
        FIXED: Optimized for independent case and better numerical stability
        
        Args:
            mu: Mean of approximate posterior [batch_size, dict_size]
            log_var: Log variance of approximate posterior [batch_size, dict_size]
            
        Returns:
            kl_loss: KL divergence [scalar]
        """
        batch_size = mu.shape[0]
        
        # Clamp mu to prevent numerical issues
        mu_clamped = torch.clamp(mu, min=-10.0, max=10.0)
        
        # FIXED: Efficient handling of independent case
        if hasattr(self, 'is_independent') and self.is_independent:
            # Much more efficient computation when prior is independent
            if self.var_flag == 1 and log_var is not None:
                log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
                # For independent case: KL = 0.5 * sum(μ² + σ² - 1 - log(σ²))
                kl_per_sample = 0.5 * torch.sum(
                    mu_clamped.pow(2) + torch.exp(log_var_clamped) - 1 - log_var_clamped,
                    dim=1
                )
            else:
                # Fixed variance case: KL = 0.5 * ||μ||²
                kl_per_sample = 0.5 * torch.sum(mu_clamped.pow(2), dim=1)
            
            kl_loss = torch.clamp(kl_per_sample.mean(), min=0.0)
            return kl_loss
        
        # General multivariate case
        # Ensure precision matrix has correct dtype and device
        precision = self.prior_precision.to(dtype=mu.dtype, device=mu.device)
        prior_log_det = self.prior_log_det.to(dtype=mu.dtype, device=mu.device)
        
        if self.var_flag == 1 and log_var is not None:
            # FIXED: Conservative clamping and better numerical handling
            log_var_clamped = torch.clamp(log_var, min=-6.0, max=2.0)
            var = torch.exp(log_var_clamped)  # σ²
            
            # For multivariate KL: KL[q||p] = 0.5 * [trace(Σp^-1 * Σq) + μ^T * Σp^-1 * μ - k + log(|Σp|/|Σq|)]
            
            # Trace term: trace(Σp^-1 * Σq) - since Σq is diagonal, this is sum over diagonal
            trace_term = torch.sum(precision.diagonal().unsqueeze(0) * var, dim=1)
            
            # Quadratic term: μ^T * Σp^-1 * μ for each sample in batch
            # More numerically stable computation
            precision_mu = torch.matmul(mu_clamped, precision)
            quad_term = torch.sum(mu_clamped * precision_mu, dim=1)
            
            # Log determinant term: log(|Σp|) - log(|Σq|)
            # |Σq| = prod(σ²) since it's diagonal, so log(|Σq|) = sum(log(σ²)) = sum(log_var)
            log_det_q = torch.sum(log_var_clamped, dim=1)
            log_det_term = prior_log_det - log_det_q
            
            # Combine all terms
            kl_per_sample = 0.5 * (trace_term + quad_term - self.dict_size + log_det_term)
            
        else:
            # Fixed variance case: σ² = 1, so Σq = I
            # KL = 0.5 * [trace(Σp^-1) + μ^T * Σp^-1 * μ - k + log(|Σp|)]
            
            trace_term = precision.diagonal().sum().expand(batch_size)
            precision_mu = torch.matmul(mu_clamped, precision)
            quad_term = torch.sum(mu_clamped * precision_mu, dim=1)
            log_det_term = prior_log_det.expand(batch_size)
            
            kl_per_sample = 0.5 * (trace_term + quad_term - self.dict_size + log_det_term)
        
        # Average over batch and ensure non-negative
        kl_loss = torch.clamp(kl_per_sample.mean(), min=0.0)
        
        # Sanity check for unreasonable KL values
        if kl_loss > 1000.0:
            print(f"Warning: Very large KL loss {kl_loss:.2f}, may indicate numerical issues")
        
        return kl_loss
    
    def get_kl_diagnostics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed KL diagnostics for monitoring training.
        
        Returns:
            Dictionary with KL components and statistics
        """
        with torch.no_grad():
            mu, log_var = self.encode(x)
            
            # Handle independent case more efficiently
            if hasattr(self, 'is_independent') and self.is_independent:
                if self.var_flag == 1 and log_var is not None:
                    log_var_safe = torch.clamp(log_var, -6, 2)
                    var = torch.exp(log_var_safe)
                    
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
                        'correlation_effect': torch.tensor(0.0),  # No correlation effect for independent case
                        'prior_type': 'independent',
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
                        'correlation_effect': torch.tensor(0.0),
                        'prior_type': 'independent',
                    }
            
            # Multivariate case
            if self.var_flag == 1 and log_var is not None:
                log_var_safe = torch.clamp(log_var, -6, 2)
                var = torch.exp(log_var_safe)
                
                # For multivariate case, compute individual terms
                precision = self.prior_precision.to(dtype=mu.dtype, device=mu.device)
                
                trace_term = torch.sum(precision.diagonal().unsqueeze(0) * var, dim=1).mean()
                quad_term = torch.sum(mu * (mu @ precision.T), dim=1).mean()
                log_det_term = (self.prior_log_det - torch.sum(log_var_safe, dim=1)).mean()
                
                kl_total = 0.5 * (trace_term + quad_term - self.dict_size + log_det_term)
                
                # Calculate correlation effect
                independent_trace = self.dict_size  # trace of identity
                correlation_effect = (trace_term - independent_trace) / independent_trace
                
                return {
                    'kl_total': kl_total,
                    'kl_trace_term': 0.5 * trace_term,
                    'kl_quad_term': 0.5 * quad_term,
                    'kl_const_term': -0.5 * self.dict_size,
                    'kl_logdet_term': 0.5 * log_det_term,
                    'mean_log_var': log_var.mean(),
                    'mean_var': var.mean(),
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                    'correlation_effect': correlation_effect,
                    'prior_type': 'multivariate',
                }
            else:
                precision = self.prior_precision.to(dtype=mu.dtype, device=mu.device)
                
                trace_term = precision.diagonal().sum()
                quad_term = torch.sum(mu * (mu @ precision.T), dim=1).mean()
                log_det_term = self.prior_log_det
                
                kl_total = 0.5 * (trace_term + quad_term - self.dict_size + log_det_term)
                
                # Calculate correlation effect
                correlation_effect = (trace_term - self.dict_size) / self.dict_size
                
                return {
                    'kl_total': kl_total,
                    'kl_trace_term': 0.5 * trace_term,
                    'kl_quad_term': 0.5 * quad_term,
                    'kl_const_term': -0.5 * self.dict_size,
                    'kl_logdet_term': 0.5 * log_det_term,
                    'mean_mu': mu.mean(),
                    'mean_mu_magnitude': mu.norm(dim=-1).mean(),
                    'mu_std': mu.std(),
                    'correlation_effect': correlation_effect,
                    'prior_type': 'multivariate',
                }
    
    def validate_config(self) -> None:
        """Validate the configuration and warn about potential issues."""
        # Memory usage warning
        memory_mb = self.config.estimate_memory_mb()
        if memory_mb > 1000:
            print(f"Warning: Model will use approximately {memory_mb:.1f} MB for parameters")
        
        # Numerical stability warnings
        if abs(self.corr_rate) > 0.95:
            print(f"Warning: High correlation rate {self.corr_rate} may cause numerical instability")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get estimated memory usage breakdown in MB."""
        def tensor_mb(tensor):
            return tensor.numel() * tensor.element_size() / (1024 ** 2)
        
        memory_usage = {
            'encoder_weights': tensor_mb(self.encoder.weight),
            'decoder_weights': tensor_mb(self.decoder.weight),
        }
        
        if hasattr(self, 'corr_matrix'):
            memory_usage['correlation_matrix'] = tensor_mb(self.corr_matrix)
        
        if hasattr(self, 'prior_precision'):
            memory_usage['precision_matrix'] = tensor_mb(self.prior_precision)
        
        if self.var_flag == 1:
            memory_usage['var_encoder_weights'] = tensor_mb(self.var_encoder.weight)
        
        memory_usage['total'] = sum(memory_usage.values())
        return memory_usage
    
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
        config: Optional[VSAEMultiConfig] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        normalize_decoder: bool = True,
        var_flag: Optional[int] = None,
        corr_rate: float = 0.5,
        corr_matrix: Optional[torch.Tensor] = None
    ) -> 'VSAEMultiGaussian':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (will auto-detect if None)
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Override var_flag detection
            corr_rate: Correlation rate for building correlation matrix
            corr_matrix: Custom correlation matrix
            
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
            
            config = VSAEMultiConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                corr_rate=corr_rate,
                corr_matrix=corr_matrix,
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
    def _convert_legacy_state_dict(state_dict: Dict[str, torch.Tensor], config: VSAEMultiConfig) -> Dict[str, torch.Tensor]:
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
        
        return converted


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
class VSAEMultiTrainingConfig:
    """Enhanced training configuration with proper scaling separation."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None  # For any actual sparsity penalties
    decay_start: Optional[int] = None
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


class VSAEMultiGaussianTrainer(SAETrainer):
    """
    Fixed trainer for VSAEMultiGaussian with proper scaling separation.
    
    Key improvements:
    - Correct multivariate KL divergence computation
    - Separate KL annealing from sparsity scaling
    - Better numerical stability
    - Enhanced logging and diagnostics
    - Clean KL computation without decoder norm weighting
    """
    
    def __init__(
        self,
        model_config: Optional[VSAEMultiConfig] = None,
        training_config: Optional[VSAEMultiTrainingConfig] = None,
        layer: Optional[int] = None,
        lm_name: Optional[str] = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Backwards compatibility parameters
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        var_flag: Optional[int] = None,
        corr_rate: Optional[float] = None,
        corr_matrix: Optional[torch.Tensor] = None,
        use_april_update_mode: Optional[bool] = None,
        device: Optional[str] = None,
        dict_class=None,  # Ignored, always use VSAEMultiGaussian
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility
        if model_config is None:
            if activation_dim is None or dict_size is None:
                raise ValueError("Must provide either model_config or activation_dim + dict_size")
            
            device_obj = torch.device(device) if device else None
            model_config = VSAEMultiConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag or 0,
                corr_rate=corr_rate or 0.5,
                corr_matrix=corr_matrix,
                use_april_update_mode=use_april_update_mode if use_april_update_mode is not None else True,
                device=device_obj
            )
        
        if training_config is None:
            if steps is None:
                raise ValueError("Must provide either training_config or steps")
            
            training_config = VSAEMultiTrainingConfig(
                steps=steps,
                lr=lr or 5e-4,
                kl_coeff=kl_coeff or 500.0,
            )
        
        self.model_config = model_config
        self.training_config = training_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEMultiGaussianTrainer"
        
        # Set device
        self.device = model_config.get_device()
        
        # Initialize model
        self.ae = VSAEMultiGaussian(model_config)
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
        # KL annealing function (separate from sparsity!)
        self.kl_warmup_fn = get_kl_warmup_fn(
            training_config.steps,
            training_config.kl_warmup_steps
        )
        
        # Add logging parameters
        self.logging_parameters = ["kl_coeff", "var_flag", "corr_rate"]
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """Compute loss with proper scaling separation."""
        sparsity_scale = self.sparsity_warmup_fn(step)  # For any L1 penalties
        kl_scale = self.kl_warmup_fn(step)  # FIXED: Separate KL annealing
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Forward pass
        mu, log_var = self.ae.encode(x)
        z = self.ae.reparameterize(mu, log_var)
        x_hat = self.ae.decode(z)
        
        # Ensure compatibility
        x_hat = x_hat.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
        
        # FIXED: Clean KL divergence computation without decoder norm weighting
        kl_loss = self.ae.compute_kl_divergence(mu, log_var)
        kl_loss = kl_loss.to(dtype=original_dtype)
        
        # FIXED: Separate scaling - KL gets kl_scale, sparsity would get sparsity_scale
        total_loss = recon_loss + self.training_config.kl_coeff * kl_scale * kl_loss
        
        # If we had L1 sparsity penalty, it would use sparsity_scale:
        # l1_loss = torch.mean(torch.abs(z))
        # total_loss += l1_coeff * sparsity_scale * l1_loss
        
        if not logging:
            return total_loss
        
        # Return detailed loss information with diagnostics
        LossLog = namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])
        
        # Get additional diagnostics
        kl_diagnostics = self.ae.get_kl_diagnostics(x)
        
        return LossLog(
            x, x_hat, z,
            {
                'l2_loss': torch.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'loss': total_loss.item(),
                'sparsity_scale': sparsity_scale,
                'kl_scale': kl_scale,  # Separate from sparsity scaling
                # Additional diagnostics
                **{k: v.item() if torch.is_tensor(v) else v for k, v in kl_diagnostics.items()}
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
            'dict_class': 'VSAEMultiGaussian',
            'trainer_class': 'VSAEMultiGaussianTrainer',
            # Model config
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'corr_rate': self.model_config.corr_rate,
            'use_april_update_mode': self.model_config.use_april_update_mode,
            'log_var_init': self.model_config.log_var_init,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'kl_warmup_steps': self.training_config.kl_warmup_steps,
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