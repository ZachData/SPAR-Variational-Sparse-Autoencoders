"""
Implements the variational autoencoder (VSAE) training scheme with isotropic Gaussian prior.
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


class VSAEIsoGaussian(Dictionary, nn.Module):
    """
    A one-layer variational autoencoder with isotropic Gaussian prior.
    
    Can be configured in two ways:
    1. Standard mode: Uses bias in the encoder input (old approach from Towards Monosemanticity)
    2. April update mode: Uses bias in both encoder and decoder (newer approach)
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    """

    def __init__(self, activation_dim, dict_size, use_april_update_mode=False, var_flag=0, device=None):
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
            raise NotImplementedError("Ghost mode not implemented for VSAEIsoGaussian")
            
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
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=0):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
            
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
        
        autoencoder = cls(activation_dim, dict_size, use_april_update_mode=use_april_update_mode, var_flag=var_flag)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class VSAEIsoTrainer(SAETrainer):
    """
    Variational Sparse Autoencoder training scheme with isotropic Gaussian prior.
    
    This trainer uses KL divergence instead of L1 penalty to encourage sparsity.
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a 
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    
    Incorporates improvements from Anthropic's April update:
    - Unconstrained decoder weights (no unit norm constraint)
    - Modified sparsity penalty that includes decoder norms
    - Better initialization
    - Gradient clipping
    """
    
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=VSAEIsoGaussian,
                 lr:float=5e-5, # recommended in April update
                 kl_coeff:float=5.0, # default lambda value from April update
                 warmup_steps:int=1000, # lr warmup period at start of training
                 sparsity_warmup_steps:Optional[int]=None, # sparsity warmup period (5% of steps in April update)
                 decay_start:Optional[int]=None, # decay learning rate after this many steps (80% of steps in April update)
                 var_flag:int=0, # whether to learn variance (0: fixed, 1: learned)
                 use_april_update_mode:bool=True, # whether to use April update mode
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='VSAEIsoTrainer',
                 submodule_name:Optional[str]=None,
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

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.use_april_update_mode = use_april_update_mode

        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Get both mean and log variance when var_flag=1
        if self.ae.var_flag == 1:
            mu, log_var = self.ae.encode(x, output_log_var=True)
            # Sample from the latent distribution using reparameterization
            z = self.ae.reparameterize(mu, log_var)
        else:
            # Just use mean if variance is fixed
            mu = self.ae.encode(x)
            z = mu
        
        # Decode
        x_hat = self.ae.decode(z)
        
        # Reconstruction loss
        recon_loss = t.pow(x - x_hat, 2).sum(dim=-1).mean()
        
        # KL divergence loss
        if self.ae.var_flag == 1:
            # Full KL divergence for learned variance: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)
            kl_base = 0.5 * (t.exp(log_var) + mu.pow(2) - 1 - log_var).sum(dim=-1)
        else:
            # Simplified KL divergence for fixed unit variance: 0.5 * sum(mu^2)
            kl_base = 0.5 * mu.pow(2).sum(dim=-1)
        
        # Calculate decoder norms
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
        
        # Weight KL divergence by decoder norms (April update approach)
        kl_loss = kl_base.mean() * decoder_norms.mean()
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        if not logging:
            return loss
        else:
            return x, x_hat, z, {
                'l2_loss': t.linalg.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'loss': loss.item()
            }
        
    def update(self, step, activations):
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
        return {
            'dict_class': 'VSAEIsoGaussian',
            'trainer_class': 'VSAEIsoTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
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
        }