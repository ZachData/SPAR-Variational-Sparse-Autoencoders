"""
Implements a Variational Sparse Autoencoder with multivariate Gaussian prior.
This implementation is designed to handle general correlation structures in the latent space.
"""
import torch as t
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEMultiGaussian(Dictionary, t.nn.Module):
    """
    Variational Sparse Autoencoder with multivariate Gaussian prior
    Designed to handle general correlation structures in the latent space
    """
    
    def __init__(self, activation_dim, dict_size, device="cuda"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        
        # Main parameters
        self.W_enc = t.nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.W_dec = t.nn.Parameter(t.empty(dict_size, activation_dim, device=device))
        self.b_enc = t.nn.Parameter(t.zeros(dict_size, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(activation_dim, device=device))
        
        # Initialize parameters with Kaiming uniform
        t.nn.init.kaiming_uniform_(self.W_enc)
        t.nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space.
        """
        x_cent = x - self.b_dec
        z = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            log_var = t.zeros_like(z)  # Fixed variance when var_flag=0
            return z, log_var
        return z
    
    def decode(self, f):
        """
        Decode a dictionary vector f
        """
        return f @ self.W_dec + self.b_dec
    
    def forward(self, x, output_features=False):
        """
        Forward pass through the autoencoder.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    @t.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = t.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def scale_biases(self, scale: float):
        """Scale biases by a factor"""
        self.b_dec.data *= scale
        self.b_enc.data *= scale
    
    @classmethod
    def from_pretrained(cls, path, device=None):
        """Load a pretrained autoencoder from a file."""
        state_dict = t.load(path)
        activation_dim, dict_size = state_dict["W_enc"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class VSAEMultiGaussianLearned(VSAEMultiGaussian):
    """
    VSAE with multivariate Gaussian prior and learned variance
    """
    
    def __init__(self, activation_dim, dict_size, device="cuda"):
        super().__init__(activation_dim, dict_size, device)
        
        # Variance parameters
        self.W_enc_var = t.nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc_var = t.nn.Parameter(t.zeros(dict_size, device=device))
        
        # Initialize variance parameters
        t.nn.init.kaiming_uniform_(self.W_enc_var)
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space with learned variance
        """
        x_cent = x - self.b_dec
        z = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
            return z, log_var
        return z
    
    def scale_biases(self, scale: float):
        """Scale biases by a factor"""
        super().scale_biases(scale)
        self.b_enc_var.data *= scale


class VSAEMultiGaussianTrainer(SAETrainer):
    """
    Trainer for VSAE with multivariate Gaussian prior
    
    This model uses a multivariate Gaussian prior with a full covariance matrix
    to capture complex correlation patterns between latent variables.
    
    Incorporates improvements from Anthropic's April update:
    - Unconstrained decoder weights (no unit norm constraint)
    - Modified sparsity penalty that includes decoder norms
    - Better initialization
    - Gradient clipping
    """
    
    def __init__(
        self,
        steps: int,                           # total number of steps to train for
        activation_dim: int,                  # dimension of input activations
        dict_size: int,                       # dictionary size
        layer: int,                           # layer to train on
        lm_name: str,                         # language model name
        corr_rate: float = 0.5,               # correlation rate for prior
        corr_matrix: Optional[t.Tensor] = None, # custom correlation matrix
        var_flag: int = 0,                    # whether to use fixed (0) or learned (1) variance
        lr: float = 5e-5,                     # learning rate (recommended in April update)
        kl_coeff: float = 5.0,                # KL coefficient (default from April update)
        warmup_steps: int = 1000,             # LR warmup steps
        sparsity_warmup_steps: Optional[int] = None, # sparsity warmup steps (5% of steps in April update)
        decay_start: Optional[int] = None,    # when to start LR decay (80% of steps in April update)
        seed: Optional[int] = None,
        device = None,
        wandb_name: Optional[str] = 'VSAEMultiGaussianTrainerApril',
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)
            
        # Use the April update defaults if not specified
        if sparsity_warmup_steps is None:
            sparsity_warmup_steps = int(0.05 * steps)  # 5% of steps
        
        if decay_start is None:
            decay_start = int(0.8 * steps)  # Start decay at 80% of training
        
        # Setup general parameters
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Setup correlation matrix for prior
        self.corr_rate = corr_rate
        self.corr_matrix = corr_matrix
        
        # Initialize model parameters (April update style)
        # Initialize decoder weights to random unit vectors with norm of 0.1
        self.W_enc = t.nn.Parameter(t.empty(activation_dim, dict_size, device=self.device))
        self.b_enc = t.nn.Parameter(t.zeros(dict_size, device=self.device))
        
        self.W_dec = t.nn.Parameter(t.empty(dict_size, activation_dim, device=self.device))
        self.b_dec = t.nn.Parameter(t.zeros(activation_dim, device=self.device))
        
        # Initialize with random directions but fixed norm of 0.1
        t.nn.init.xavier_normal_(self.W_dec)
        norm = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm * 0.1
        
        # Initialize encoder as transpose of decoder
        self.W_enc.data = self.W_dec.data.t().clone()
        
        # For variance learning if needed
        if self.var_flag == 1:
            self.W_enc_var = t.nn.Parameter(t.empty(activation_dim, dict_size, device=self.device))
            self.b_enc_var = t.nn.Parameter(t.zeros(dict_size, device=self.device))
            
            # Initialize with Xavier/Kaiming
            t.nn.init.kaiming_uniform_(self.W_enc_var)
        
        # Build prior covariance matrix and compute derived quantities
        self.prior_covariance = self._build_prior_covariance()
        self.prior_precision = self._compute_prior_precision()
        self.prior_cov_logdet = self._compute_prior_logdet()
        
        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Setup learning rate and sparsity schedules
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add logging parameters
        self.logging_parameters = ["kl_coeff", "var_flag", "corr_rate"]
    
    def _build_prior_covariance(self):
        """Build the prior covariance matrix"""
        d_hidden = self.dict_size
        
        if self.corr_matrix is not None and self.corr_matrix.shape[0] == d_hidden:
            corr_matrix = self.corr_matrix
        elif self.corr_rate is not None:
            # Create a matrix with uniform correlation
            corr_matrix = t.full((d_hidden, d_hidden), self.corr_rate, device=self.device)
            t.diagonal(corr_matrix)[:] = 1.0
        else:
            # Default to identity (no correlation)
            return t.eye(d_hidden, device=self.device)
        
        # Ensure the correlation matrix is valid
        if not t.allclose(corr_matrix, corr_matrix.t()):
            print("Warning: Correlation matrix not symmetric, symmetrizing it")
            corr_matrix = 0.5 * (corr_matrix + corr_matrix.t())
        
        # Add small jitter to ensure positive definiteness
        corr_matrix = corr_matrix + t.eye(d_hidden, device=self.device) * 1e-4
        
        return corr_matrix.to(self.device)
    
    def _compute_prior_precision(self):
        """Compute the precision matrix (inverse of covariance)"""
        try:
            return t.linalg.inv(self.prior_covariance)
        except:
            # Add jitter for numerical stability
            jitter = t.eye(self.dict_size, device=self.device) * 1e-4
            return t.linalg.inv(self.prior_covariance + jitter)
    
    def _compute_prior_logdet(self):
        """Compute log determinant of prior covariance"""
        return t.logdet(self.prior_covariance)
    
    def encode(self, x, deterministic=True):
        """
        Encode inputs to sparse features
        
        Args:
            x: Input tensor
            deterministic: If True, return mean without sampling
            
        Returns:
            Encoded features
        """
        # Compute mean of latent distribution
        mu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        
        if deterministic:
            return mu
        
        # Get log variance 
        if self.var_flag == 1:
            log_var = F.relu((x - self.b_dec) @ self.W_enc_var + self.b_enc_var)
        else:
            log_var = t.zeros_like(mu)
        
        # Sample from the latent distribution
        return self.reparameterize(mu, log_var)
    
    def decode(self, z):
        """
        Decode sparse features back to inputs
        
        Args:
            z: Sparse features
            
        Returns:
            Reconstructed inputs
        """
        return z @ self.W_dec + self.b_dec
    
    def reparameterize(self, mu, log_var):
        """
        Apply the reparameterization trick: z = mu + eps * sigma, where eps ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent variable z
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def _compute_kl_divergence(self, mu, log_var, decoder_norms):
        """
        Compute KL divergence between approximate posterior and prior
        
        Args:
            mu: Mean of approximate posterior
            log_var: Log variance of approximate posterior
            decoder_norms: Norms of decoder columns for April update style penalty
            
        Returns:
            KL divergence
        """
        # For efficiency, compute on batch mean
        mu_avg = mu.mean(0)  # [dict_size]
        var_avg = log_var.exp().mean(0)  # [dict_size]
        
        # Trace term: tr(Σp^-1 * Σq)
        trace_term = (self.prior_precision.diagonal() * var_avg).sum()
        
        # Quadratic term: μ^T * Σp^-1 * μ
        quad_term = mu_avg @ self.prior_precision @ mu_avg
        
        # Log determinant term: ln(|Σp|/|Σq|)
        log_det_q = log_var.sum(1).mean()
        log_det_term = self.prior_cov_logdet - log_det_q
        
        # Combine terms
        kl = 0.5 * (trace_term + quad_term - self.dict_size + log_det_term)
        
        # Ensure non-negative
        kl = t.clamp(kl, min=0.0)
        
        # Apply the April update style penalty incorporating decoder norms
        # For multivariate case, we apply a weighted average based on decoder norms
        kl = kl * decoder_norms.mean()
        
        return kl
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with multivariate Gaussian prior
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            step: Current training step
            logging: Whether to return detailed loss components
            
        Returns:
            Loss value (or loss details if logging=True)
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Compute mean of latent distribution
        mu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        
        # Get log variance
        if self.var_flag == 1:
            log_var = F.relu((x - self.b_dec) @ self.W_enc_var + self.b_enc_var)
        else:
            log_var = t.zeros_like(mu)
        
        # Sample from the latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode to get reconstruction
        x_hat = z @ self.W_dec + self.b_dec
        
        # Get decoder column norms for April update style penalty
        decoder_norms = t.norm(self.W_dec, dim=1)
        
        # Compute losses
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        kl_loss = self._compute_kl_divergence(mu, log_var, decoder_norms)
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(), 
                    'loss': loss.item()
                }
            )
    
    def update(self, step, activations):
        """
        Update the autoencoder with a batch of activations
        
        Args:
            step: Current training step
            activations: Batch of activations
        """
        activations = activations.to(self.device)
        
        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient clipping (April update)
        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
    
    def forward(self, x, output_features=False):
        """
        Forward pass through the VAE
        
        Args:
            x: Input activations
            output_features: Whether to return features
            
        Returns:
            Reconstructed activations (and features if requested)
        """
        # Encode to get mean (deterministic forward pass)
        f = self.encode(x, deterministic=True)
        
        # Decode
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    @property
    def config(self):
        """Return configuration for logging"""
        return {
            'dict_class': 'VSAEMultiGaussian',
            'trainer_class': 'VSAEMultiGaussianTrainerApril',
            'activation_dim': self.activation_dim,
            'dict_size': self.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'corr_rate': self.corr_rate,
            'var_flag': self.var_flag,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }