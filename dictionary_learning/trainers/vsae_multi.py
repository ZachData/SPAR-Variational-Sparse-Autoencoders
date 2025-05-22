"""
Implements a Variational Sparse Autoencoder with multivariate Gaussian prior.
This implementation is designed to handle general correlation structures in the latent space.
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, List, Dict, Any
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEMultiGaussian(Dictionary, nn.Module):
    """
    Variational Sparse Autoencoder with multivariate Gaussian prior
    Designed to handle general correlation structures in the latent space
    """
    
    def __init__(self, 
                 activation_dim, 
                 dict_size, 
                 use_april_update_mode=True,
                 var_flag=0,
                 corr_rate=0.5,
                 corr_matrix=None,
                 device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        self.var_flag = var_flag
        self.corr_rate = corr_rate
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=use_april_update_mode)
        
        # Initialize weights
        w = t.randn(activation_dim, dict_size, device=device)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())
        
        # Initialize biases
        init.zeros_(self.encoder.bias)
        if use_april_update_mode:
            init.zeros_(self.decoder.bias)
        else:
            # In standard mode, we use a separate bias parameter
            self.bias = nn.Parameter(t.zeros(activation_dim, device=device))
            
        # Variance network (only used when var_flag=1)
        if var_flag == 1:
            self.var_encoder = nn.Linear(activation_dim, dict_size, bias=True)
            init.kaiming_uniform_(self.var_encoder.weight)
            init.zeros_(self.var_encoder.bias)
        
        # Initialize correlation structure
        if corr_matrix is not None:
            self.corr_matrix = corr_matrix
        else:
            # Create default correlation matrix based on corr_rate
            self.corr_matrix = self._build_correlation_matrix(corr_rate)
        
        # Move to the specified device
        if device is not None:
            self.to(device)
        
    def _build_correlation_matrix(self, corr_rate):
        """Build correlation matrix based on the given correlation rate"""
        d_hidden = self.dict_size
        corr_matrix = t.full((d_hidden, d_hidden), corr_rate)
        t.diagonal(corr_matrix)[:] = 1.0  # Diagonal elements are 1
        
        # Ensure the correlation matrix is valid
        if not t.allclose(corr_matrix, corr_matrix.t()):
            print("Warning: Correlation matrix not symmetric, symmetrizing it")
            corr_matrix = 0.5 * (corr_matrix + corr_matrix.t())
        
        # Add small jitter to ensure positive definiteness
        corr_matrix = corr_matrix + t.eye(d_hidden) * 1e-4
        
        return corr_matrix
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space.
        Returns mean (and log variance if requested).
        """
        if self.use_april_update_mode:
            mu = F.relu(self.encoder(x))
        else:
            mu = F.relu(self.encoder(x - self.bias))
            
        if output_log_var:
            if self.var_flag == 1:
                if self.use_april_update_mode:
                    log_var = F.relu(self.var_encoder(x))
                else:
                    log_var = F.relu(self.var_encoder(x - self.bias))
            else:
                log_var = t.zeros_like(mu)
            return mu, log_var
        return mu
    
    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization trick: z = mu + eps * sigma
        where eps ~ N(0, 1)
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def decode(self, f):
        """
        Decode a dictionary vector f.
        """
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        
        Args:
            x: activations to be autoencoded
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEMultiGaussian")
            
        # Encode and get mu (and log_var if var_flag=1)
        if self.var_flag == 1:
            mu, log_var = self.encode(x, output_log_var=True)
            # Sample from the latent distribution
            f = self.reparameterize(mu, log_var)
        else:
            mu = self.encode(x)
            f = mu  # Without variance, just use mu directly
            
        # Decode
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    def scale_biases(self, scale: float):
        """
        Scale all bias parameters by a given factor.
        """
        self.encoder.bias.data *= scale
        if self.use_april_update_mode:
            self.decoder.bias.data *= scale
        else:
            self.bias.data *= scale
            
        if self.var_flag == 1:
            self.var_encoder.bias.data *= scale

    def normalize_decoder(self):
        """
        Normalize decoder weights to have unit norm.
        """
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = t.norm(self.decoder.weight, dim=0)
        assert t.allclose(new_norms, t.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=None, corr_rate=0.5):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
                     If None, will auto-detect from state_dict
            corr_rate: Correlation rate for prior
            
        Returns:
            Loaded autoencoder
        """
        state_dict = t.load(path)
        
        # Auto-detect var_flag from state_dict if not explicitly provided
        if var_flag is None:
            # Check if the state dict contains variance encoder weights
            has_var_encoder = "var_encoder.weight" in state_dict or "W_enc_var" in state_dict
            var_flag = 1 if has_var_encoder else 0
        
        # Determine dimensions and mode based on state dict
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
                    converted_dict["bias"] = state_dict["b_dec"]
                    
                if var_flag == 1 and "W_enc_var" in state_dict:
                    converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                    converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                    
                state_dict = converted_dict
        
        # Create model with detected parameters
        autoencoder = cls(
            activation_dim, 
            dict_size, 
            use_april_update_mode=use_april_update_mode, 
            var_flag=var_flag, 
            corr_rate=corr_rate,
            device=device
        )
        
        # Filter state_dict to only include keys that are in the model
        model_keys = set(autoencoder.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Load the filtered state dictionary
        autoencoder.load_state_dict(filtered_state_dict, strict=False)
        
        # Check for missing keys
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
            # Initialize missing parameters to default values
            # This is typically just needed for var_encoder when loading a var_flag=0 model as var_flag=1
            if var_flag == 1 and "var_encoder.weight" in missing_keys:
                print("Initializing missing variance encoder parameters with default values")
                # Set var_encoder to small random values
                init.kaiming_uniform_(autoencoder.var_encoder.weight)
                init.zeros_(autoencoder.var_encoder.bias)
        
        # Skip normalization if loaded model had learned variances, as it might break the model
        if normalize_decoder and not (var_flag == 1 and "var_encoder.weight" in state_dict):
            try:
                autoencoder.normalize_decoder()
            except AssertionError:
                print("Warning: Could not normalize decoder weights. Skipping normalization.")
        
        if device is not None:
            autoencoder.to(dtype=dtype, device=device)
        
        return autoencoder


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
        dict_class=VSAEMultiGaussian,         # dictionary class to use
        corr_rate: float = 0.5,               # correlation rate for prior
        corr_matrix: Optional[t.Tensor] = None, # custom correlation matrix
        var_flag: int = 0,                    # whether to use fixed (0) or learned (1) variance
        lr: float = 5e-5,                     # learning rate (recommended in April update)
        kl_coeff: float = 5.0,                # KL coefficient (default from April update)
        warmup_steps: int = 1000,             # LR warmup steps
        sparsity_warmup_steps: Optional[int] = None, # sparsity warmup steps (5% of steps in April update)
        decay_start: Optional[int] = None,    # when to start LR decay (80% of steps in April update)
        use_april_update_mode: bool = True,   # whether to use April update mode
        seed: Optional[int] = None,
        device = None,
        wandb_name: Optional[str] = 'VSAEMultiGaussianTrainer',
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
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.corr_rate = corr_rate
        self.use_april_update_mode = use_april_update_mode
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize dictionary using the provided dictionary class
        self.ae = dict_class(
            activation_dim, 
            dict_size,
            use_april_update_mode=use_april_update_mode,
            var_flag=var_flag,
            corr_rate=corr_rate,
            corr_matrix=corr_matrix,
            device=self.device
        )
        self.ae.to(self.device)
        
        # Compute derived quantities for the prior
        self.prior_precision = self._compute_prior_precision()
        self.prior_cov_logdet = self._compute_prior_logdet()
        
        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Setup learning rate and sparsity schedules
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add logging parameters
        self.logging_parameters = ["kl_coeff", "var_flag", "corr_rate"]
    
    def _compute_prior_precision(self):
        """Compute the precision matrix (inverse of covariance)"""
        try:
            # Get the current autocast dtype to ensure consistency
            dtype = self.ae.encoder.weight.dtype
            return t.linalg.inv(self.ae.corr_matrix.to(self.device).to(dtype))
        except:
            # Add jitter for numerical stability
            jitter = t.eye(self.ae.dict_size, device=self.device, dtype=self.ae.encoder.weight.dtype) * 1e-4
            return t.linalg.inv(self.ae.corr_matrix.to(self.device).to(self.ae.encoder.weight.dtype) + jitter)

    def _compute_prior_logdet(self):
        """Compute log determinant of prior covariance"""
        # Get the current autocast dtype to ensure consistency
        dtype = self.ae.encoder.weight.dtype
        return t.logdet(self.ae.corr_matrix.to(self.device).to(dtype))
    
    def _compute_kl_divergence(self, mu, log_var, decoder_norms):
        """
        Compute KL divergence between approximate posterior and multivariate Gaussian prior
        
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
        
        # Get the current dtype for consistency
        dtype = mu.dtype
        
        # Cast the precision matrix to match mu's dtype
        prior_precision = self.prior_precision.to(dtype)
        
        # Trace term: tr(Σp^-1 * Σq)
        trace_term = (prior_precision.diagonal() * var_avg).sum()
        
        # Quadratic term: μ^T * Σp^-1 * μ
        quad_term = mu_avg @ prior_precision @ mu_avg
        
        # Log determinant term: ln(|Σp|/|Σq|)
        log_det_q = log_var.sum(1).mean()
        log_det_term = self.prior_cov_logdet - log_det_q
        
        # Combine terms
        kl = 0.5 * (trace_term + quad_term - self.ae.dict_size + log_det_term)
        
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
            x: Input tensor
            step: Current training step
            logging: Whether to return detailed loss components
            
        Returns:
            Loss value (or loss details if logging=True)
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Get mean and log variance from encoder
        if self.ae.var_flag == 1:
            mu, log_var = self.ae.encode(x, output_log_var=True)
            # Sample from the latent distribution
            f = self.ae.reparameterize(mu, log_var)
        else:
            mu = self.ae.encode(x)
            f = mu  # Without variance, just use mu directly
            log_var = t.zeros_like(mu)  # For KL calculation
        
        # Decode to get reconstruction
        x_hat = self.ae.decode(f)
        
        # Get decoder column norms for April update style penalty
        decoder_norms = t.norm(self.ae.decoder.weight, dim=0)
        
        # Compute losses
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        kl_loss = self._compute_kl_divergence(mu, log_var, decoder_norms)
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
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
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
    
    @property
    def config(self):
        """Return configuration for logging"""
        return {
            'dict_class': 'VSAEMultiGaussian',
            'trainer_class': 'VSAEMultiGaussianTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'corr_rate': self.corr_rate,
            'var_flag': self.var_flag,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'use_april_update_mode': self.use_april_update_mode,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }