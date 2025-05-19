"""
Implements a variational autoencoder (VSAE) training scheme with gated components.
This implementation combines the benefits of both VSAEIsoGaussian and GatedAutoEncoder.
"""
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Union
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEGated(Dictionary, nn.Module):
    """
    A gated variational autoencoder that combines elements from both 
    GatedAutoEncoder and VSAEIsoGaussian.
    
    This model has:
    1. A gating network as in the GatedSAE
    2. A variational component for magnitude as in VSAEIso
    
    This approach aims to benefit from the interpretability of the gating network
    while leveraging the probabilistic nature of variational autoencoders.
    """

    def __init__(self, activation_dim, dict_size, use_april_update_mode=True, var_flag=1, device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        self.var_flag = var_flag
        
        # Decoder bias (shared)
        self.decoder_bias = nn.Parameter(t.zeros(activation_dim, device=device))
        
        # Encoder for shared processing
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True, device=device)
        
        # Gating network parameters
        self.gate_bias = nn.Parameter(t.zeros(dict_size, device=device))
        
        # Magnitude network parameters
        self.r_mag = nn.Parameter(t.zeros(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.zeros(dict_size, device=device))
        
        # Decoder (shared)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        
        # For variational approach, add variance encoder if needed
        if var_flag == 1:
            self.var_encoder = nn.Linear(activation_dim, dict_size, bias=True, device=device)
            init.zeros_(self.var_encoder.bias)
        
        # Initialize weights with careful scaling
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with appropriate scaling."""
        # Initialize decoder weights as random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        
        # Initialize encoder weights to match decoder weights (transposed)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)
        
        # Initialize biases to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.encoder.bias)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)
        init.zeros_(self.r_mag)
        
        # Initialize variance encoder if present
        if self.var_flag == 1 and hasattr(self, 'var_encoder'):
            init.kaiming_uniform_(self.var_encoder.weight)
            init.zeros_(self.var_encoder.bias)
    
    def encode(self, x: t.Tensor, return_gate: bool = False, return_log_var: bool = False) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        """
        Encode input tensor to get latent features.
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            return_gate: If True, return gating values 
            return_log_var: If True, return log variance values (for VAE)
            
        Returns:
            Features tensor or tuple of tensors depending on flags
        """
        # Compute shared encoding
        x_enc = self.encoder(x - self.decoder_bias) 
        
        # Gating network
        pi_gate = x_enc + self.gate_bias
        gate = (pi_gate > 0).to(self.encoder.weight.dtype)  # Binary gate
        f_gate = F.relu(pi_gate)  # Continuous relaxation for gradients
        
        # Magnitude network with variational component
        pi_mag = t.exp(self.r_mag) * x_enc + self.mag_bias
        mu_mag = F.relu(pi_mag)  # Mean
        
        # For variational case, compute log variance
        if self.var_flag == 1 and return_log_var:
            log_var = F.relu(self.var_encoder(x - self.decoder_bias))
            mag = self.reparameterize(mu_mag, log_var)
        else:
            mag = mu_mag
            log_var = None
        
        # Combine gating and magnitude
        f = gate * mag
        
        # Return requested outputs
        if return_gate and return_log_var:
            return f, f_gate, log_var
        elif return_gate:
            return f, f_gate
        elif return_log_var:
            return f, log_var
        else:
            return f
    
    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """Apply reparameterization trick for variational sampling."""
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def decode(self, f: t.Tensor) -> t.Tensor:
        """Decode latent features."""
        return self.decoder(f) + self.decoder_bias
    
    def forward(self, x: t.Tensor, output_features: bool = False) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            output_features: Whether to return features along with reconstructions
            
        Returns:
            Reconstructed tensor or tuple of (reconstructed, features)
        """
        if self.var_flag == 1:
            f, f_gate, log_var = self.encode(x, return_gate=True, return_log_var=True)
        else:
            f, f_gate = self.encode(x, return_gate=True)
            log_var = None
        
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    def scale_biases(self, scale: float):
        """Scale all bias parameters by given factor."""
        self.decoder_bias.data *= scale
        self.encoder.bias.data *= scale
        self.gate_bias.data *= scale
        self.mag_bias.data *= scale
        if self.var_flag == 1 and hasattr(self, 'var_encoder'):
            self.var_encoder.bias.data *= scale
    
    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=1):
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
        state_dict = t.load(path, map_location=device)
        
        # Determine dimensions from state_dict
        if 'encoder.weight' in state_dict:
            dict_size, activation_dim = state_dict["encoder.weight"].shape
        else:
            raise ValueError("Unexpected state_dict format, cannot determine dimensions")
            
        autoencoder = cls(
            activation_dim=activation_dim, 
            dict_size=dict_size, 
            use_april_update_mode=True,
            var_flag=var_flag,
            device=device
        )
        
        # Try to load, may fail if state_dict doesn't match exactly
        try:
            autoencoder.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load state_dict exactly: {e}")
            print("Attempting parameter-by-parameter loading...")
            
            # Manual loading of parameters that exist in both models
            for name, param in autoencoder.named_parameters():
                if name in state_dict:
                    try:
                        param.data.copy_(state_dict[name])
                        print(f"Loaded parameter: {name}")
                    except Exception as e2:
                        print(f"Could not load parameter {name}: {e2}")

        if device is not None:
            autoencoder = autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class VSAEGatedTrainer(SAETrainer):
    """
    Trainer for the VSAEGated model.
    
    This trainer combines approaches from both GatedSAETrainer and VSAEIsoTrainer
    to leverage the strengths of both models.
    """
    
    def __init__(self,
                 steps: int,  # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=VSAEGated,
                 lr: float = 5e-5,  # recommended in April update
                 kl_coeff: float = 5.0,  # KL divergence regularization strength
                 l1_penalty: float = 0.01,  # L1 regularization for gating network
                 aux_weight: float = 0.1,  # Weight for auxiliary loss
                 warmup_steps: int = 1000,  # lr warmup period
                 sparsity_warmup_steps: Optional[int] = None,  # sparsity warmup period
                 decay_start: Optional[int] = None,  # decay learning rate after this many steps
                 var_flag: int = 1,  # whether to learn variance (0: fixed, 1: learned)
                 use_constrained_optimizer: bool = True,  # whether to use constrained optimizer
                 use_april_update_mode: bool = True,  # whether to use April update mode 
                 seed: Optional[int] = None,
                 device=None,
                 wandb_name: Optional[str] = 'VSAEGatedTrainer',
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

        # Initialize device
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize dictionary
        self.ae = dict_class(
            activation_dim, 
            dict_size, 
            use_april_update_mode=use_april_update_mode,
            var_flag=var_flag,
            device=self.device
        )
        self.ae.to(self.device)

        # Save hyperparameters
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.l1_penalty = l1_penalty
        self.aux_weight = aux_weight
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.use_april_update_mode = use_april_update_mode
        self.use_constrained_optimizer = use_constrained_optimizer

        # Initialize optimizer
        if use_constrained_optimizer:
            self.optimizer = ConstrainedAdam(
                self.ae.parameters(),
                self.ae.decoder.parameters(),
                lr=lr,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = t.optim.Adam(
                self.ae.parameters(), 
                lr=lr, 
                betas=(0.9, 0.999)
            )

        # Initialize learning rate scheduler
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Initialize sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
    def kl_divergence(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Compute KL divergence between N(mu, var) and N(0, 1).
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            KL divergence tensor
        """
        return -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
    def loss(self, x: t.Tensor, step: int, logging: bool = False, **kwargs):
        """
        Compute combined loss function.
        
        Args:
            x: Input tensor
            step: Current training step
            logging: Whether to return detailed logs
            
        Returns:
            Loss tensor or named tuple with detailed metrics
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Get encodings with all auxiliary outputs
        if self.var_flag == 1:
            f, f_gate, log_var = self.ae.encode(x, return_gate=True, return_log_var=True)
        else:
            f, f_gate = self.ae.encode(x, return_gate=True)
            log_var = None
        
        # Main reconstruction
        x_hat = self.ae.decode(f)
        
        # Create auxiliary reconstruction from gate network only
        x_hat_gate = self.ae.decoder(f_gate) + self.ae.decoder_bias
        
        # Compute losses
        # 1. Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # 2. Sparsity loss for gating network
        gate_sparsity_loss = t.linalg.norm(f_gate, ord=1, dim=-1).mean()
        
        # 3. Auxiliary loss (helps train gate network)
        aux_loss = (x - x_hat_gate).pow(2).sum(dim=-1).mean()
        
        # 4. KL divergence for variational component
        if self.var_flag == 1 and log_var is not None:
            # Modified KL loss with decoder norm scaling per GVSAE paper
            kl_base = -0.5 * t.sum(1 + log_var - f_gate.pow(2) - log_var.exp(), dim=1)
            decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
            kl_loss = (kl_base * decoder_norms.mean()).mean()
        else:
            kl_loss = t.tensor(0.0, device=self.device)
        
        # Combine losses with appropriate weights
        total_loss = (
            recon_loss + 
            (self.l1_penalty * sparsity_scale * gate_sparsity_loss) + 
            (self.aux_weight * aux_loss) + 
            (self.kl_coeff * sparsity_scale * kl_loss)
        )
        
        if not logging:
            return total_loss
        else:
            l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
            l0 = (f != 0).float().sum(dim=-1).mean()
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'gate_sparsity_loss': gate_sparsity_loss.item(),
                    'aux_loss': aux_loss.item(),
                    'kl_loss': kl_loss.item() if isinstance(kl_loss, t.Tensor) else kl_loss,
                    'l0': l0.item(),
                    'loss': total_loss.item()
                }
            )
        
    def update(self, step, activations):
        """Perform a single update step."""
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient clipping
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        """Return configuration dictionary."""
        return {
            'dict_class': 'VSAEGated',
            'trainer_class': 'VSAEGatedTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'l1_penalty': self.l1_penalty,
            'aux_weight': self.aux_weight,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'use_constrained_optimizer': self.use_constrained_optimizer,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'var_flag': self.var_flag,
            'use_april_update_mode': self.use_april_update_mode,
        }
