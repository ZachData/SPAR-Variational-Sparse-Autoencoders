"""
Implements a Variational Sparse Autoencoder with Gaussian mixture prior.
This implementation is designed to handle setwise correlations in the latent space.
"""
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple, List, Dict, Any
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEMixtureGaussian(Dictionary, nn.Module):
    """
    A one-layer variational autoencoder with Gaussian mixture prior.
    
    Can be configured in two ways:
    1. Standard mode: Uses bias in the encoder input (old approach from Towards Monosemanticity)
    2. April update mode: Uses bias in both encoder and decoder (newer approach)
    
    This implementation is designed to handle setwise correlations in the latent space
    by using a mixture of Gaussian distributions as the prior, specifically for
    correlated and anticorrelated feature pairs.
    
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
                 n_correlated_pairs=0,
                 n_anticorrelated_pairs=0,
                 device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        self.var_flag = var_flag
        self.n_correlated_pairs = n_correlated_pairs
        self.n_anticorrelated_pairs = n_anticorrelated_pairs
        
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
        
        # Initialize prior means based on correlation structure
        self.prior_means = self.initialize_prior_means(device)
        self.prior_std = 1.0  # Standard deviation for each Gaussian in the mixture
        
        self.to(device if device is not None else self.encoder.weight.device)

    def initialize_prior_means(self, device=None):
        """
        Initialize the prior means based on the correlation structure
        
        Args:
            device: Device to store tensors on
            
        Returns:
            Tensor of prior means with shape [dict_size]
        """
        means = []
        
        # For correlated pairs, we use the same mean (e.g., [1.0, 1.0])
        for _ in range(self.n_correlated_pairs):
            means.extend([1.0, 1.0])
            
        # For anticorrelated pairs, we use opposite means (e.g., [1.0, -1.0])
        for _ in range(self.n_anticorrelated_pairs):
            means.extend([1.0, -1.0])
            
        # Remaining features get zero mean (standard Gaussian)
        remaining = self.dict_size - 2 * (self.n_correlated_pairs + self.n_anticorrelated_pairs)
        means.extend([0.0] * remaining)
        
        device = device if device is not None else self.encoder.weight.device if hasattr(self, 'encoder') else None
        return t.tensor(means, dtype=t.float32, device=device)

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
        Forward pass of the variational autoencoder.
        
        Args:
            x: activations to be autoencoded
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEMixtureGaussian")
            
        # Encode and get mu (and log_var if var_flag=1)
        if self.var_flag == 1:
            mu, log_var = self.encode(x, output_log_var=True)
            # Sample from the latent distribution
            f = self.reparameterize(mu, log_var)
        else:
            mu = self.encode(x)
            f = mu  # Without variance, just use mu directly
            log_var = t.zeros_like(mu)  # For KL calculation
            
        # Decode
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def compute_kl_divergence(self, mu, log_var):
        """
        Compute the KL divergence between the approximate posterior and the mixture prior.
        
        Args:
            mu: Mean of approximate posterior [batch_size, n_instances, dict_size]
            log_var: Log variance of approximate posterior [batch_size, n_instances, dict_size]
            
        Returns:
            KL divergence [batch_size, n_instances]
        """
        # Compute KL divergence to a mixture of Gaussians
        # For each feature, we use its corresponding prior mean from prior_means
        
        # Get variance from log_var
        var = t.exp(log_var)
        
        # Compute KL divergence for a Gaussian with mean mu and variance var
        # to a Gaussian with mean prior_mean and variance 1 (prior_std^2)
        # KL(N(mu, var) || N(prior_mean, 1)) = 
        #   0.5 * (var + (mu - prior_mean)^2 - 1 - log(var))
        
        prior_means = self.prior_means.to(mu.device)
        kl_div = 0.5 * (var + (mu - prior_means)**2 / (self.prior_std**2) - 1 - log_var)
        
        # Sum over features
        return kl_div.sum(dim=-1)  # [batch_size, n_instances]

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
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=None):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
                    If None, will auto-detect from state_dict
            
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
        
        # Try to extract correlation parameters from state_dict metadata
        n_correlated_pairs = 0
        n_anticorrelated_pairs = 0
        
        # Create model with detected parameters
        autoencoder = cls(
            activation_dim, 
            dict_size, 
            use_april_update_mode=use_april_update_mode, 
            var_flag=var_flag,
            n_correlated_pairs=n_correlated_pairs,
            n_anticorrelated_pairs=n_anticorrelated_pairs,
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


class VSAEMixtureTrainer(SAETrainer):
    """
    Trainer for VSAE with Gaussian mixture prior
    
    This model uses a mixture of Gaussian distributions as the prior to account
    for correlated and anticorrelated feature pairs in the latent space.
    
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
        dict_class=VSAEMixtureGaussian,       # dictionary class to use
        lr: float = 5e-5,                     # learning rate (recommended in April update)
        kl_coeff: float = 5.0,                # coefficient for KL divergence term
        warmup_steps: int = 1000,             # lr warmup period at start of training
        sparsity_warmup_steps: Optional[int] = None, # sparsity warmup period
        decay_start: Optional[int] = None,    # decay learning rate after this many steps
        resample_steps: Optional[int] = None, # how often to resample neurons (None for no resampling)
        var_flag: int = 0,                    # whether to learn variance (0: fixed, 1: learned)
        n_correlated_pairs: int = 0,          # number of correlated feature pairs
        n_anticorrelated_pairs: int = 0,      # number of anticorrelated feature pairs
        use_april_update_mode: bool = True,   # whether to use April update mode
        seed: Optional[int] = None,
        device = None,
        wandb_name: Optional[str] = 'VSAEMixtureTrainer',
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

        # Initialize dictionary
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
            n_correlated_pairs=n_correlated_pairs,
            n_anticorrelated_pairs=n_anticorrelated_pairs,
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
        self.n_correlated_pairs = n_correlated_pairs
        self.n_anticorrelated_pairs = n_anticorrelated_pairs
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
        self.logging_parameters = [
            "kl_coeff", 
            "var_flag", 
            "n_correlated_pairs", 
            "n_anticorrelated_pairs", 
            "use_april_update_mode"
        ]
        
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with Gaussian mixture prior
        
        Args:
            x: Input activations [batch_size, activation_dim]
            step: Current training step
            logging: Whether to return extended information for logging
            
        Returns:
            Loss value or namedtuple with extended information
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Encode and get mu (and log_var if var_flag=1)
        if self.ae.var_flag == 1:
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
        if self.use_april_update_mode:
            decoder_norms = t.norm(self.ae.decoder.weight, dim=0)
        else:
            decoder_norms = None
        
        # Compute losses
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        
        # Compute KL divergence with mixture prior
        kl_loss = self.ae.compute_kl_divergence(mu, log_var)
        
        # Apply April update style weighting if in that mode
        if self.use_april_update_mode and decoder_norms is not None:
            # Weight KL divergence by average decoder norm
            kl_loss = kl_loss * decoder_norms.mean()
        
        # Average KL loss over batch
        kl_loss = kl_loss.mean()
        
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
    
    def resample_neurons(
        self,
        h: t.Tensor,
        frac_active_in_window: t.Tensor,
        neuron_resample_scale: float,
    ) -> Tuple[List[List[str]], str]:
        """
        Resample neurons that have been dead for a specified window period
        
        Args:
            h: Input activations [batch_size, n_instances, n_hidden]
            frac_active_in_window: Fraction of active neurons [window, n_instances, n_hidden_ae]
            neuron_resample_scale: Scaling factor for resampled neurons
            
        Returns:
            Tuple of (colors, title) for visualization
        """
        _, l2_loss, _, _, _ = self.ae.forward(h)

        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_neurons_mask = t.empty((self.ae.dict_size,), dtype=t.bool, device=self.ae.encoder.weight.device)

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = (frac_active_in_window.sum(0) < 1e-8)
        dead_neurons_mask = is_dead
        dead_neurons = t.nonzero(is_dead).squeeze(-1)
        alive_neurons = t.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()
        
        if n_dead > 0:
            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                pass  # If we have zero reconstruction loss, we don't need to resample neurons
            else:
                # Use the t.multinomial distribution for weighted sampling
                from torch.distributions import Categorical
                
                # Create distribution proportional to l2_loss
                probs = l2_loss_instance / l2_loss_instance.sum()
                distn = Categorical(probs=probs)
                
                # Sample indices based on loss
                replacement_indices = distn.sample((n_dead,))  # shape [n_dead]
                
                # Index into the batch of hidden activations to get our replacement values
                replacement_values = (h - self.ae.b_dec)[replacement_indices]  # shape [n_dead, n_input_ae]
                
                # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
                W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.ae.encoder.weight[alive_neurons].norm(dim=-1).mean().item()
                
                # Use this to renormalize the replacement values
                replacement_values = (replacement_values / (replacement_values.norm(dim=-1, keepdim=True) + 1e-8)) * W_enc_norm_alive_mean * neuron_resample_scale
                
                # Lastly, set the new weights & biases
                self.ae.encoder.weight.data[dead_neurons] = replacement_values
                self.ae.encoder.bias.data[dead_neurons] = 0.0

        # Return data for visualizing the resampling process
        colors = ["red" if dead else "black" for dead in dead_neurons_mask]
        title = f"resampling {dead_neurons_mask.sum()}/{dead_neurons_mask.numel()} neurons (shown in red)"
        return colors, title
    
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
            'dict_class': 'VSAEMixtureGaussian',
            'trainer_class': 'VSAEMixtureTrainer',
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
            'n_correlated_pairs': self.n_correlated_pairs,
            'n_anticorrelated_pairs': self.n_anticorrelated_pairs,
            'use_april_update_mode': self.use_april_update_mode,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }