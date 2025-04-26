"""
Implements a Variational Sparse Autoencoder with multivariate Gaussian prior.
This implementation is designed to handle general correlation structures in the latent space.
"""
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, List, Dict, Any
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEMultiGaussian(Dictionary, nn.Module):
    """
    A one-layer variational autoencoder with multivariate Gaussian prior.
    
    Can be configured in two ways:
    1. Standard mode: Uses bias in the encoder input (old approach from Towards Monosemanticity)
    2. April update mode: Uses bias in both encoder and decoder (newer approach)
    
    This implementation is designed to handle general correlation structures in the 
    latent space through a custom covariance matrix for the prior distribution.
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a 
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    """

    def __init__(self, 
                 activation_dim, 
                 dict_size, 
                 use_april_update_mode=False, 
                 var_flag=0,
                 corr_rate=0.0,
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
            
        # Setup correlation structure
        self.setup_correlation(corr_matrix, corr_rate, device)

    def setup_correlation(self, corr_matrix, corr_rate, device):
        """
        Setup correlation matrix for the prior distribution
        
        Args:
            corr_matrix: Custom correlation matrix (if None, use corr_rate)
            corr_rate: Default correlation rate for all pairs
            device: Device to store tensors on
        """
        if device is None:
            device = self.encoder.weight.device
            
        # Build prior covariance matrix
        if corr_matrix is not None and corr_matrix.shape[0] == self.dict_size:
            self.prior_covariance = corr_matrix.to(device)
        elif corr_rate is not None:
            # Create a matrix with uniform correlation
            self.prior_covariance = t.full((self.dict_size, self.dict_size), corr_rate, device=device)
            t.diagonal(self.prior_covariance)[:] = 1.0
        else:
            # Default to identity (no correlation)
            self.prior_covariance = t.eye(self.dict_size, device=device)
        
        # Ensure the correlation matrix is valid
        if not t.allclose(self.prior_covariance, self.prior_covariance.t()):
            print("Warning: Correlation matrix not symmetric, symmetrizing it")
            self.prior_covariance = 0.5 * (self.prior_covariance + self.prior_covariance.t())
        
        # Add small jitter to ensure positive definiteness
        self.prior_covariance = self.prior_covariance + t.eye(self.dict_size, device=device) * 1e-4
        
        # Pre-compute precision matrix and log determinant for efficiency
        try:
            self.prior_precision = t.linalg.inv(self.prior_covariance)
        except:
            # Add jitter for numerical stability
            jitter = t.eye(self.dict_size, device=device) * 1e-4
            self.prior_precision = t.linalg.inv(self.prior_covariance + jitter)
            
        self.prior_cov_logdet = t.logdet(self.prior_covariance)

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
            f = mu  # Without variance, just use mu
            
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
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=0, 
                        corr_rate=0.0, corr_matrix=None):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
            corr_rate: Default correlation rate for all pairs
            corr_matrix: Custom correlation matrix
            
        Returns:
            Loaded autoencoder
        """
        state_dict = t.load(path)
        
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
        
        autoencoder = cls(
            activation_dim, 
            dict_size, 
            use_april_update_mode=use_april_update_mode, 
            var_flag=var_flag,
            corr_rate=corr_rate,
            corr_matrix=corr_matrix,
            device=device
        )
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class VSAEMultiGaussianTrainer(SAETrainer):
    """
    Trainer for VSAE with multivariate Gaussian prior
    
    This model uses a multivariate Gaussian prior with a full covariance matrix
    to capture complex correlation patterns between latent variables.
    
    Can use two different training approaches:
    1. Standard mode with constrained decoder norms and separate bias
    2. April update mode with unconstrained decoder weights, modified penalties
       (incorporating improvements from Anthropic's April update)
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a 
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    """
    
    def __init__(
        self,
        steps: int,                           # total number of steps to train for
        activation_dim: int,                  # dimension of input activations
        dict_size: int,                       # dictionary size
        layer: int,                           # layer to train on
        lm_name: str,                         # language model name
        dict_class=VSAEMultiGaussian,         # dictionary class to use
        lr: float = 5e-5,                     # learning rate (recommended in April update)
        kl_coeff: float = 5.0,                # coefficient for KL divergence term
        warmup_steps: int = 1000,             # lr warmup period at start of training
        sparsity_warmup_steps: Optional[int] = None, # sparsity warmup period
        decay_start: Optional[int] = None,    # decay learning rate after this many steps
        resample_steps: Optional[int] = None, # how often to resample neurons (None for no resampling)
        var_flag: int = 0,                    # whether to learn variance (0: fixed, 1: learned)
        corr_rate: float = 0.0,               # correlation rate for prior
        corr_matrix: Optional[t.Tensor] = None, # custom correlation matrix
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

        # Use the April update defaults if not specified and in April mode
        if sparsity_warmup_steps is None and use_april_update_mode:
            sparsity_warmup_steps = int(0.05 * steps)  # 5% of steps
        elif sparsity_warmup_steps is None:
            sparsity_warmup_steps = 2000  # Standard default
        
        if decay_start is None and use_april_update_mode:
            decay_start = int(0.8 * steps)  # Start decay at 80% of training for April mode

        # initialize dictionary
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Create the autoencoder model
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

        # Save all parameters
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.resample_steps = resample_steps
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.corr_rate = corr_rate
        self.use_april_update_mode = use_april_update_mode

        # For dead neuron detection and resampling
        if self.resample_steps is not None:
            # How many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(dict_size, dtype=t.long).to(self.device)
        else:
            self.steps_since_active = None

        # Choose optimizer based on mode
        if use_april_update_mode:
            # April update uses Adam without constrained weights
            self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))
        else:
            # Standard mode uses constrained Adam to maintain unit decoder norms
            self.optimizer = ConstrainedAdam(
                self.ae.parameters(), 
                [self.ae.decoder.weight], 
                lr=lr
            )

        # Create learning rate scheduler
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Create sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add tracking metrics for logging
        self.logging_parameters = ["kl_coeff", "var_flag", "corr_rate", "use_april_update_mode"]
        
    def compute_kl_divergence(self, mu, log_var, decoder_norms=None):
        """
        Compute KL divergence between approximate posterior and multivariate Gaussian prior
        
        Args:
            mu: Mean of approximate posterior [batch_size, dict_size]
            log_var: Log variance of approximate posterior [batch_size, dict_size]
            decoder_norms: Optional norms of decoder columns for April update style penalty
            
        Returns:
            KL divergence (scalar)
        """
        # For efficiency, compute on batch mean
        mu_avg = mu.mean(0)  # [dict_size]
        var_avg = log_var.exp().mean(0)  # [dict_size]
        
        # Trace term: tr(Σp^-1 * Σq)
        trace_term = (self.ae.prior_precision.diagonal() * var_avg).sum()
        
        # Quadratic term: μ^T * Σp^-1 * μ
        quad_term = mu_avg @ self.ae.prior_precision @ mu_avg
        
        # Log determinant term: ln(|Σp|/|Σq|)
        log_det_q = log_var.sum(1).mean()
        log_det_term = self.ae.prior_cov_logdet - log_det_q
        
        # Combine terms (standard KL formula for multivariate Gaussians)
        kl = 0.5 * (trace_term + quad_term - self.ae.dict_size + log_det_term)
        
        # Ensure non-negative
        kl = t.clamp(kl, min=0.0)
        
        # Apply the April update style penalty incorporating decoder norms
        if self.use_april_update_mode and decoder_norms is not None:
            # For multivariate case, we apply a weighted average based on decoder norms
            kl = kl * decoder_norms.mean()
        
        return kl
        
    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons with high loss activations
        
        Args:
            deads: Boolean tensor indicating which neurons are dead
            activations: Batch of activations to sample from
        """
        with t.no_grad():
            if deads.sum() == 0: 
                return
                
            print(f"resampling {deads.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Get norm of living neurons
            alive_mask = ~deads
            alive_norm = self.ae.encoder.weight[alive_mask].norm(dim=-1).mean()

            # Resample first n_resample dead neurons (limit by n_resample)
            resample_idx = deads.nonzero().squeeze()
            if len(resample_idx) > n_resample:
                excess_idx = resample_idx[n_resample:]
                deads[excess_idx] = False
                
            # Process inputs for resampling
            if self.use_april_update_mode:
                # For April update mode
                sampled_centered = sampled_vecs
            else:
                # For standard mode, center by bias
                sampled_centered = sampled_vecs - self.ae.bias
                
            # Update encoder weights and bias
            self.ae.encoder.weight.data[deads] = sampled_centered * alive_norm * 0.2
            self.ae.encoder.bias.data[deads] = 0.
            
            # Update decoder weights
            normalized_samples = sampled_centered / sampled_centered.norm(dim=-1, keepdim=True)
            self.ae.decoder.weight.data[:, deads] = normalized_samples.T
            
            # Also reset variance params if needed
            if self.var_flag == 1:
                self.ae.var_encoder.weight.data[deads] = 0.
                self.ae.var_encoder.bias.data[deads] = 0.

            # Reset optimizer state for resampled neurons
            self._reset_optimizer_stats(deads)
    
    def _reset_optimizer_stats(self, dead_mask):
        """
        Reset optimizer state for resampled neurons
        
        Args:
            dead_mask: Boolean tensor indicating which neurons are dead
        """
        state_dict = self.optimizer.state_dict()['state']
        param_mapping = {}
        
        # Build mapping from param objects to their state indices
        for param_idx, param in enumerate(self.optimizer.param_groups[0]['params']):
            if param_idx in state_dict:
                param_mapping[param] = param_idx
        
        # Reset stats for each parameter group
        if self.ae.encoder.weight in param_mapping:
            idx = param_mapping[self.ae.encoder.weight]
            state_dict[idx]['exp_avg'][dead_mask] = 0.
            state_dict[idx]['exp_avg_sq'][dead_mask] = 0.
            
        if self.ae.encoder.bias in param_mapping:
            idx = param_mapping[self.ae.encoder.bias]
            state_dict[idx]['exp_avg'][dead_mask] = 0.
            state_dict[idx]['exp_avg_sq'][dead_mask] = 0.
            
        if self.ae.decoder.weight in param_mapping:
            idx = param_mapping[self.ae.decoder.weight]
            state_dict[idx]['exp_avg'][:, dead_mask] = 0.
            state_dict[idx]['exp_avg_sq'][:, dead_mask] = 0.
            
        if self.var_flag == 1 and hasattr(self.ae, 'var_encoder'):
            if self.ae.var_encoder.weight in param_mapping:
                idx = param_mapping[self.ae.var_encoder.weight]
                state_dict[idx]['exp_avg'][dead_mask] = 0.
                state_dict[idx]['exp_avg_sq'][dead_mask] = 0.
                
            if self.ae.var_encoder.bias in param_mapping:
                idx = param_mapping[self.ae.var_encoder.bias]
                state_dict[idx]['exp_avg'][dead_mask] = 0.
                state_dict[idx]['exp_avg_sq'][dead_mask] = 0.
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with multivariate Gaussian prior
        
        Args:
            x: Input activations [batch_size, activation_dim]
            step: Current training step
            logging: Whether to return extended information for logging
            
        Returns:
            Loss value or namedtuple with extended information
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Encode and get mu (and log_var if var_flag=1)
        if self.var_flag == 1:
            mu, log_var = self.ae.encode(x, output_log_var=True)
            # Sample from the latent distribution
            f = self.ae.reparameterize(mu, log_var)
        else:
            mu = self.ae.encode(x)
            log_var = t.zeros_like(mu)  # Fixed variance when var_flag=0
            f = mu  # Without variance, just use mu directly
            
        # Decode
        x_hat = self.ae.decode(f)
        
        # Get decoder column norms for April update style penalty
        decoder_norms = t.norm(self.ae.decoder.weight, dim=0) if self.use_april_update_mode else None
        
        # Compute losses
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        kl_loss = self.compute_kl_divergence(mu, log_var, decoder_norms)
        
        # Track active features for resampling
        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
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
        Perform a single training step
        
        Args:
            step: Current training step
            activations: Input activations
        """
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass and compute loss
        loss = self.loss(activations, step=step)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping in April update mode
        if self.use_april_update_mode:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Resample dead neurons if needed
        if self.resample_steps is not None and step > 0 and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps // 2, activations)

    @property
    def config(self):
        """Return configuration for logging and saving"""
        return {
            'dict_class': 'VSAEMultiGaussian',
            'trainer_class': 'VSAEMultiGaussianTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'resample_steps': self.resample_steps,
            'var_flag': self.var_flag,
            'corr_rate': self.corr_rate,
            'use_april_update_mode': self.use_april_update_mode,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }