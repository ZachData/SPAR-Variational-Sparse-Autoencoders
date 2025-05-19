"""
Implements a hybrid Variational Sparse Autoencoder with JumpReLU activation.
This combines the variational approach from VSAEIsoGaussian with the JumpReLU activation
function to potentially improve training performance.
"""
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torch.autograd as autograd
from typing import Optional, List
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..dictionary import Dictionary


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, t.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, t.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = t.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class VSAEJumpReLU(Dictionary, nn.Module):
    """
    A hybrid model combining VSAEIsoGaussian with JumpReLU activation.
    
    This model uses a variational approach for the encoder, combined with 
    a JumpReLU activation function to encourage sparsity in feature activations.
    
    Args:
        activation_dim: Dimension of the input activations
        dict_size: Number of features in the dictionary
        use_april_update_mode: Whether to use the April update approach (bias in decoder)
        var_flag: Whether to learn variance (0: fixed, 1: learned)
        bandwidth: The bandwidth parameter for JumpReLU
        device: Device to place the model on
    """

    def __init__(self, activation_dim, dict_size, use_april_update_mode=True, 
                 var_flag=1, bandwidth=0.001, device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        self.var_flag = var_flag
        self.bandwidth = bandwidth
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=use_april_update_mode)
        
        # Initialize thresholds for JumpReLU
        self.threshold = nn.Parameter(t.ones(dict_size, device=device) * 0.001)
        
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
            pre_activation = self.encoder(x)
        else:
            pre_activation = self.encoder(x - self.bias)
            
        # Apply JumpReLU activation
        mu = JumpReLUFunction.apply(pre_activation, self.threshold, self.bandwidth)
            
        if output_log_var:
            if self.var_flag == 1:
                if self.use_april_update_mode:
                    log_var_pre = self.var_encoder(x)
                else:
                    log_var_pre = self.var_encoder(x - self.bias)
                    
                # Apply JumpReLU to log_var as well for consistency
                log_var = JumpReLUFunction.apply(log_var_pre, self.threshold, self.bandwidth)
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
        Forward pass of the autoencoder.
        
        Args:
            x: activations to be autoencoded
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: not implemented for this model
        """
        if ghost_mask is not None:
            raise NotImplementedError("Ghost mode not implemented for VSAEJumpReLU")
            
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
        self.threshold.data *= scale
        
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
        
        if self.var_flag == 1:
            self.var_encoder.weight.data *= norms[:, None]
            self.var_encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True, var_flag=1, bandwidth=0.001):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
            bandwidth: Bandwidth parameter for JumpReLU
            
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
                converted_dict["threshold"] = state_dict.get("threshold", t.ones(dict_size) * 0.001)
                
                if use_april_update_mode:
                    converted_dict["decoder.bias"] = state_dict["b_dec"]
                else:
                    converted_dict["bias"] = state_dict["b_dec"]
                    
                if var_flag == 1 and "W_enc_var" in state_dict:
                    converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                    converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                    
                state_dict = converted_dict
        
        autoencoder = cls(activation_dim, dict_size, use_april_update_mode=use_april_update_mode, 
                          var_flag=var_flag, bandwidth=bandwidth, device=device)
        
        # Load state dict, handling missing threshold parameter
        if "threshold" not in state_dict:
            state_dict["threshold"] = autoencoder.threshold
            
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class VSAEJumpReLUTrainer(SAETrainer):
    """
    Trainer for the VSAEJumpReLU model, combining variational techniques with JumpReLU activation.
    
    This trainer uses KL divergence as in VSAE plus a target L0 loss from JumpReLU to encourage 
    appropriate levels of sparsity in the learned features.
    """
    
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=VSAEJumpReLU,
                 lr: float=5e-5, # recommended in April update
                 kl_coeff: float=5.0, # default lambda value from April update
                 l0_coeff: float=1.0, # coefficient for target L0 regularization
                 target_l0: float=20.0, # target L0 value (average number of active features)
                 warmup_steps: int=1000, # lr warmup period at start of training
                 sparsity_warmup_steps: Optional[int]=None, # sparsity warmup period 
                 decay_start: Optional[int]=None, # decay learning rate after this many steps
                 var_flag: int=1, # whether to learn variance (0: fixed, 1: learned)
                 bandwidth: float=0.001, # bandwidth parameter for JumpReLU
                 use_april_update_mode: bool=True, # whether to use April update mode
                 seed: Optional[int]=None,
                 device=None,
                 wandb_name: Optional[str]='VSAEJumpReLUTrainer',
                 submodule_name: Optional[str]=None,
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
            bandwidth=bandwidth,
            device=self.device
        )
        self.ae.to(self.device)

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.l0_coeff = l0_coeff
        self.target_l0 = target_l0
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.bandwidth = bandwidth
        self.use_april_update_mode = use_april_update_mode

        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # For logging
        self.logging_parameters = ["l0_loss", "kl_loss", "mse_loss", "loss"]
        self.l0_loss = 0
        self.kl_loss = 0
        self.mse_loss = 0
        self.loss = 0
        
    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Encode to get mu (and log_var if var_flag=1)
        if self.ae.var_flag == 1:
            mu, log_var = self.ae.encode(x, output_log_var=True)
            # Sample from the latent distribution
            z = self.ae.reparameterize(mu, log_var)
        else:
            mu = self.ae.encode(x)
            z = mu  # Without variance, just use mu
        
        # Decode
        x_hat = self.ae.decode(z)
        
        # Reconstruction loss (MSE)
        recon_loss = t.pow(x - x_hat, 2).sum(dim=-1).mean()
        
        # KL divergence loss
        kl_base = 0.5 * (mu.pow(2)).sum(dim=-1)  # [batch_size]
        
        # Calculate decoder norms
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)  # [dict_size]
        
        # KL loss with decoder norms (as in VSAE)
        kl_loss = kl_base.mean() * decoder_norms.mean()
        
        # L0 sparsity regularization (as in JumpReLU)
        l0 = StepFunction.apply(mu, self.ae.threshold, self.ae.bandwidth).sum(dim=-1).mean()
        l0_loss = self.l0_coeff * ((l0 / self.target_l0) - 1).pow(2)
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss + sparsity_scale * l0_loss
        
        # Store for logging
        self.mse_loss = recon_loss.item()
        self.kl_loss = kl_loss.item()
        self.l0_loss = l0_loss.item()
        self.loss = loss.item()
        
        if not logging:
            return loss
        else:
            return x, x_hat, mu, {
                'l2_loss': t.linalg.norm(x - x_hat, dim=-1).mean().item(),
                'mse_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'l0_loss': l0_loss.item(),
                'l0': l0.item(),
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
            'dict_class': 'VSAEJumpReLU',
            'trainer_class': 'VSAEJumpReLUTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'l0_coeff': self.l0_coeff,
            'target_l0': self.target_l0,
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
            'bandwidth': self.bandwidth,
            'use_april_update_mode': self.use_april_update_mode,
        }
