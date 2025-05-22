"""
Implements a variational autoencoder (VSAE) training scheme with p-annealing.

This trainer combines the benefits of variational autoencoders with p-norm annealing
to provide more control over sparsity and potentially discover better features.
"""
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEPAnneal(Dictionary, nn.Module):
    """
    A one-layer variational autoencoder with p-annealing for sparsity control.
    
    Combines the VAE approach from VSAEIsoGaussian and the p-annealing technique
    to provide flexible control over the sparsity of the latent representation.
    """

    def __init__(self, activation_dim, dict_size, use_april_update_mode=True, var_flag=0, device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        self.var_flag = var_flag
        
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
            raise NotImplementedError("Ghost mode not implemented for VSAEPAnneal")
            
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
            if self.var_flag == 1:
                return x_hat, f, mu, log_var
            else:
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

        # Create test input on the same device as the model parameters
        device = self.decoder.weight.device
        test_input = t.randn(10, self.activation_dim, device=device)
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
        
        # Create model with detected parameters
        autoencoder = cls(activation_dim, dict_size, use_april_update_mode=use_april_update_mode, var_flag=var_flag, device=device)
        
        # Filter state_dict to only include keys that are in the model
        model_keys = set(autoencoder.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Load the filtered state dictionary
        autoencoder.load_state_dict(filtered_state_dict, strict=False)
        
        # MOVE TO DEVICE FIRST, BEFORE DOING ANYTHING ELSE
        if device is not None:
            autoencoder.to(dtype=dtype, device=device)
        
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
        
        return autoencoder


class VSAEPAnnealTrainer(SAETrainer):
    """
    Trainer for Variational Sparse Autoencoder with p-norm annealing.
    
    This trainer combines the benefits of:
    1. Variational autoencoders (KL divergence based regularization)
    2. p-norm annealing (adjustable sparsity norm)
    
    Key features:
    - Can learn both mean and variance of latent distributions (var_flag=1)
    - Gradually anneals the p-norm from p_start to p_end
    - Uses adaptive sparsity coefficient to maintain consistent regularization
    - Unconstrained decoder weights (from the April update)
    """
    
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=VSAEPAnneal,
                 lr:float=5e-5, # Learning rate (from April update)
                 sparsity_penalty:float=5.0, # Initial KL coefficient (lambda)
                 warmup_steps:int=1000, # lr warmup period at start of training
                 sparsity_warmup_steps:Optional[int]=None, # sparsity warmup period (5% of steps in April update)
                 decay_start:Optional[int]=None, # decay learning rate after this many steps (80% of steps in April update)
                 var_flag:int=0, # whether to learn variance (0: fixed, 1: learned)
                 use_april_update_mode:bool=True, # whether to use April update mode
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='VSAEPAnnealTrainer',
                 submodule_name:Optional[str]=None,
                 # P-annealing specific parameters
                 sparsity_function: str = 'Lp', # Lp or Lp^p
                 anneal_start: int = 15000, # step at which to start annealing p
                 anneal_end: Optional[int] = None, # step at which to stop annealing, defaults to steps-1
                 p_start: float = 1, # starting value of p (constant throughout warmup)
                 p_end: float = 0.5, # annealing p_start to p_end linearly after warmup_steps
                 n_sparsity_updates: int | str = 10, # number of times to update the sparsity penalty
                 sparsity_queue_length: int = 10, # number of recent sparsity loss terms to track
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
            
        if anneal_end is None:
            anneal_end = steps - 1

        # initialize dictionary
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ae = dict_class(
            activation_dim, 
            dict_size, 
            use_april_update_mode=use_april_update_mode,
            var_flag=var_flag,
            device=self.device
        )
        self.ae.to(self.device)

        # Base VAE parameters
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.use_april_update_mode = use_april_update_mode
        
        # P-annealing parameters
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start
        self.next_p = None
        
        if n_sparsity_updates == "continuous":
            self.n_sparsity_updates = self.anneal_end - anneal_start + 1
        else:
            self.n_sparsity_updates = n_sparsity_updates
            
        self.sparsity_update_steps = t.linspace(anneal_start, self.anneal_end, self.n_sparsity_updates, dtype=int)
        self.p_values = t.linspace(p_start, p_end, self.n_sparsity_updates)
        self.p_step_count = 0
        self.sparsity_coeff = sparsity_penalty
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []
        
        # For logging
        self.logging_parameters = [
            'p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'sparsity_coeff',
            'recon_loss', 'total_loss'
        ]
        self.kl_loss = 0
        self.scaled_kl_loss = 0
        self.recon_loss = 0
        self.total_loss = 0

        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Check for duplicates in update steps
        if (self.sparsity_update_steps.unique(return_counts=True)[1] > 1).any():
            print("Warning! Duplicates in self.sparsity_update_steps detected!")
        
    def lp_kl_loss(self, mu, log_var=None, p=1.0):
        """
        Compute the p-norm regularized KL divergence loss.
        
        When p=1, this is equivalent to L1 regularization (or standard KL for Gaussian prior).
        When p<1, it promotes more sparse solutions.
        When p=2, it's equivalent to L2 regularization.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution (only used when var_flag=1)
            p: The p-norm value to use
            
        Returns:
            The p-norm regularized KL divergence loss
        """
        # Get decoder norms for scaling (from April update)
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)  # [dict_size]
        
        if self.ae.var_flag == 1 and log_var is not None:
            # Full KL divergence for Gaussian with learned variance
            kl_per_latent = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            if p == 1.0:
                # Standard case - just sum the KL terms
                kl_loss = (kl_per_latent.sum(dim=-1) * decoder_norms.mean()).mean()
            else:
                # Apply Lp or Lp^p normalization to the KL terms
                if self.sparsity_function == 'Lp^p':
                    kl_loss = (kl_per_latent.pow(p).sum(dim=-1) * decoder_norms.mean()).mean()
                else:  # 'Lp'
                    kl_loss = (kl_per_latent.pow(p).sum(dim=-1).pow(1/p) * decoder_norms.mean()).mean()
        else:
            # Simplified KL for fixed variance - effectively just regularizing mu
            if p == 1.0:
                # L1 regularization
                kl_loss = (mu.abs().sum(dim=-1) * decoder_norms.mean()).mean()
            elif p == 2.0:
                # L2 regularization
                kl_loss = (mu.pow(2).sum(dim=-1) * decoder_norms.mean()).mean() * 0.5
            else:
                # Lp or Lp^p regularization
                if self.sparsity_function == 'Lp^p':
                    kl_loss = (mu.abs().pow(p).sum(dim=-1) * decoder_norms.mean()).mean()
                else:  # 'Lp'
                    kl_loss = (mu.abs().pow(p).sum(dim=-1).pow(1/p) * decoder_norms.mean()).mean()
        
        return kl_loss
        
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the loss for training the variational autoencoder with p-annealing.
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            step: Current training step
            logging: Whether to return detailed logging information
            
        Returns:
            If logging=False: Total loss tensor
            If logging=True: Tuple of (x, x_hat, f, log_dict)
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Forward pass through the model
        if self.ae.var_flag == 1:
            x_hat, f, mu, log_var = self.ae(x, output_features=True)
        else:
            x_hat, f = self.ae(x, output_features=True)
            mu = f
            log_var = None
        
        # Reconstruction loss 
        recon_loss = t.pow(x - x_hat, 2).sum(dim=-1).mean()
        self.recon_loss = recon_loss.item()
        
        # KL divergence loss with p-norm regularization
        kl_loss = self.lp_kl_loss(mu, log_var, p=self.p)
        self.kl_loss = kl_loss.item()
        
        # Scale KL loss by sparsity coefficient and warmup factor
        scaled_kl_loss = kl_loss * self.sparsity_coeff * sparsity_scale
        self.scaled_kl_loss = scaled_kl_loss.item()
        
        # Total loss
        total_loss = recon_loss + scaled_kl_loss
        self.total_loss = total_loss.item()
        
        # Check if we need to compute the next p-value's loss for adaptive scaling
        if self.next_p is not None:
            kl_loss_next = self.lp_kl_loss(mu, log_var, p=self.next_p)
            self.sparsity_queue.append([kl_loss.item(), kl_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]
    
        # Update p-value if needed
        if step in self.sparsity_update_steps:
            # Check to make sure we don't update on repeat step
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt sparsity penalty alpha
                if self.next_p is not None and len(self.sparsity_queue) > 0:
                    local_sparsity_current = t.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_next = t.tensor([i[1] for i in self.sparsity_queue]).mean()
                    if local_sparsity_next > 0:  # Avoid division by zero
                        self.sparsity_coeff = self.sparsity_coeff * (local_sparsity_current / local_sparsity_next).item()
                        
                # Update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates-1:
                    self.next_p = self.p_values[self.p_step_count+1].item()
                else:
                    self.next_p = self.p_end
                self.p_step_count += 1
                
                # Log the update
                print(f"Step {step}: Updated p to {self.p:.4f}, next_p: {self.next_p:.4f}, sparsity_coeff: {self.sparsity_coeff:.4f}")
        
        # Return appropriate values based on logging flag
        if not logging:
            return total_loss
        else:
            log_dict = {
                'p': self.p,
                'next_p': self.next_p,
                'kl_loss': self.kl_loss,
                'scaled_kl_loss': self.scaled_kl_loss,
                'sparsity_coeff': self.sparsity_coeff,
                'recon_loss': self.recon_loss,
                'total_loss': self.total_loss,
            }
            return x, x_hat, f, log_dict
        
    def update(self, step, activations):
        """
        Perform a single training step.
        
        Args:
            step: Current training step
            activations: Batch of activations to train on
        """
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient clipping (from April update)
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        """Return the configuration of this trainer for logging purposes."""
        return {
            'dict_class': 'VSAEPAnneal',
            'trainer_class': 'VSAEPAnnealTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'sparsity_coeff': self.sparsity_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'var_flag': self.var_flag,
            'use_april_update_mode': self.use_april_update_mode,
            'sparsity_function': self.sparsity_function,
            'anneal_start': self.anneal_start,
            'anneal_end': self.anneal_end,
            'p_start': self.p_start,
            'p_end': self.p_end,
            'n_sparsity_updates': self.n_sparsity_updates,
            'sparsity_queue_length': self.sparsity_queue_length,
        }