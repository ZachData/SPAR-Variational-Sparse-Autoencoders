"""
Implements a combined MatryoshkaBatchTopK + VSAEIso approach.
This combines the hierarchical grouping and learning from the Matryoshka approach
with the variational learning from the VSAE approach.
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import einops
from collections import namedtuple
from typing import Optional, List
from math import isclose

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)

class MatryoshkaVSAEIso(Dictionary, nn.Module):
    """
    A combined Matryoshka + Variational Sparse Autoencoder with isotropic Gaussian prior.
    
    This combines:
    1. Hierarchical group structure from Matryoshka for progressive training
    2. Variational approach with KL divergence regularization
    3. Optional learned variance for more expressive latent space
    
    The model can be used with fixed or learned variance, and with configurable
    group sizes for progressive training.
    """

    def __init__(
        self, 
        activation_dim: int, 
        dict_size: int, 
        group_sizes: list[int],
        var_flag: int = 0,  # 0: fixed variance, 1: learned variance 
        device=None
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.var_flag = var_flag
        
        # Validate group sizes
        assert sum(group_sizes) == dict_size, "group sizes must sum to dict_size"
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"
        
        # Set up group structure
        self.active_groups = len(group_sizes)
        group_indices = [0] + list(t.cumsum(t.tensor(group_sizes), dim=0))
        self.group_indices = group_indices
        self.register_buffer("group_sizes", t.tensor(group_sizes))
        
        # Initialize encoder, decoder, and biases
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)
        
        # Initialize variance network (only used when var_flag=1)
        if var_flag == 1:
            self.var_encoder = nn.Linear(activation_dim, dict_size, bias=True)
            init.kaiming_uniform_(self.var_encoder.weight)
            init.zeros_(self.var_encoder.bias)
        
        # Initialize weights
        w = t.randn(activation_dim, dict_size, device=device)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())
        
        # Initialize biases
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x, output_log_var=False, return_active=False):
        """
        Encode a vector x in the activation space.
        Returns mean (and log variance if requested).
        """
        # Calculate encoder activations
        mu = F.relu(self.encoder(x))
        
        # Limit to active groups
        max_act_index = self.group_indices[self.active_groups]
        mu[:, max_act_index:] = 0
        
        if output_log_var:
            if self.var_flag == 1:
                log_var = F.relu(self.var_encoder(x))
                # Limit to active groups
                log_var[:, max_act_index:] = 0
            else:
                log_var = t.zeros_like(mu)
                
            if return_active:
                return mu, log_var, (mu.sum(0) > 0)
            return mu, log_var
            
        if return_active:
            return mu, (mu.sum(0) > 0)
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
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of the autoencoder.
        """
        # Encode
        if self.var_flag == 1:
            mu, log_var = self.encode(x, output_log_var=True)
            # Sample from the latent distribution
            f = self.reparameterize(mu, log_var)
        else:
            mu = self.encode(x)
            f = mu  # Without learned variance, just use mu
            
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
        self.decoder.bias.data *= scale
            
        if self.var_flag == 1:
            self.var_encoder.bias.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, var_flag=0):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        
        # Determine dimensions from state dict
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        
        # Load group sizes
        group_sizes = state_dict["group_sizes"].tolist()
        
        # Create model instance
        autoencoder = cls(
            activation_dim, 
            dict_size, 
            group_sizes=group_sizes,
            var_flag=var_flag,
            device=device
        )
        
        autoencoder.load_state_dict(state_dict)
        
        if device is not None:
            autoencoder.to(device)
            
        return autoencoder


class MatryoshkaVSAEIsoTrainer(SAETrainer):
    """
    Trainer for the combined MatryoshkaVSAEIso model.
    
    This trainer combines:
    1. Progressive group activation from Matryoshka
    2. Variational training with KL divergence
    3. Group-specific loss weighting
    4. Optional auxiliary loss for dead features
    """
    
    def __init__(self,
                steps: int,  # total number of steps to train for
                activation_dim: int,
                dict_size: int,
                layer: int,
                lm_name: str,
                group_fractions: list[float],
                group_weights: Optional[list[float]] = None,
                dict_class=MatryoshkaVSAEIso,
                lr: float = 5e-5,  # recommended in April update
                kl_coeff: float = 5.0,  # default lambda value
                auxk_alpha: float = 1 / 32,  # weight for auxiliary loss
                warmup_steps: int = 1000,  # lr warmup period
                sparsity_warmup_steps: Optional[int] = None,  # sparsity warmup period
                decay_start: Optional[int] = None,  # decay learning rate after this many steps
                var_flag: int = 0,  # whether to learn variance (0: fixed, 1: learned)
                seed: Optional[int] = None,
                device=None,
                wandb_name: Optional[str] = 'MatryoshkaVSAEIsoTrainer',
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
            
        # Validate group fractions
        assert isclose(sum(group_fractions), 1.0), "group_fractions must sum to 1.0"
        # Calculate group sizes
        group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
        # Put remainder in the last group
        group_sizes.append(dict_size - sum(group_sizes))

        if group_weights is None:
            group_weights = [(1.0 / len(group_sizes))] * len(group_sizes)

        assert len(group_sizes) == len(group_weights), (
            "group_sizes and group_weights must have the same length"
        )

        self.group_fractions = group_fractions
        self.group_sizes = group_sizes
        self.group_weights = group_weights

        # Initialize dictionary
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ae = dict_class(
            activation_dim, 
            dict_size, 
            group_sizes=group_sizes,
            var_flag=var_flag,
            device=self.device
        )
        self.ae.to(self.device)

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.auxk_alpha = auxk_alpha
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        
        # Tracking dead features
        self.dead_feature_threshold = 10_000_000
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.top_k_aux = activation_dim // 2  # Heuristic for auxiliary loss

        # April update uses Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # For logging
        self.logging_parameters = ["dead_features", "pre_norm_auxk_loss"]
        self.dead_features = 0
        self.pre_norm_auxk_loss = 0
        
    def get_auxiliary_loss(self, residual: t.Tensor, post_relu_acts: t.Tensor):
        """
        Get auxiliary loss for dead features, similar to Matryoshka's approach
        """
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            # Mark dead latents
            auxk_latents = t.where(dead_features[None], post_relu_acts, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer = t.zeros_like(post_relu_acts)
            auxk_acts_encoded = auxk_buffer.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # We don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_encoded)
            l2_loss_aux = (
                (residual.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # Normalize loss
            residual_mu = residual.mean(dim=0)[None, :].broadcast_to(residual.shape)
            loss_denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = 0
            return t.tensor(0, dtype=residual.dtype, device=residual.device)
        
    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Initialize reconstruction with the bias
        x_reconstruct = t.zeros_like(x)
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        recon_losses = t.tensor([]).to(self.device)
        
        # Split weights and activations by group
        W_dec_chunks = []
        start_idx = 0
        for size in self.group_sizes[:self.ae.active_groups]:
            end_idx = start_idx + size
            W_dec_chunks.append(self.ae.decoder.weight[start_idx:end_idx])
            start_idx = end_idx
            
        # Encode with mean and variance if using var_flag=1
        if self.var_flag == 1:
            mu, log_var, active_indices = self.ae.encode(x, output_log_var=True, return_active=True)
            z = self.ae.reparameterize(mu, log_var)
        else:
            mu, active_indices = self.ae.encode(x, return_active=True)
            log_var = t.zeros_like(mu)  # Dummy for KL calculation
            z = mu  # Without learned variance, just use mu
            
        # Split z by group
        z_chunks = []
        start_idx = 0
        for size in self.group_sizes:
            end_idx = start_idx + size
            z_chunks.append(z[:, start_idx:end_idx])
            start_idx = end_idx
            
        # Progressive reconstruction through groups
        for i in range(self.ae.active_groups):
            if i == 0:
                # First group includes bias
                x_reconstruct = x_reconstruct + self.ae.decoder.bias + z_chunks[i] @ W_dec_chunks[i]
            else:
                # Subsequent groups add their contribution
                x_reconstruct = x_reconstruct + z_chunks[i] @ W_dec_chunks[i]
                
            # Calculate reconstruction loss for this level
            recon_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean() * self.group_weights[i]
            total_recon_loss += recon_loss
            recon_losses = t.cat([recon_losses, recon_loss.unsqueeze(0)])
            
            # Calculate KL divergence - use decoder norm weights as in VSAEIso
            start_idx = self.ae.group_indices[i]
            end_idx = self.ae.group_indices[i+1]
            decoder_norms = self.ae.decoder.weight[start_idx:end_idx].norm(p=2, dim=0)
            
            # KL divergence term: 0.5 * sum(exp(log_var) - log_var - 1 + mu^2)
            # For fixed variance (var_flag=0), this simplifies to 0.5 * sum(mu^2)
            kl_base = 0.5 * (mu[:, start_idx:end_idx].pow(2)).sum(dim=-1)
            kl_loss = kl_base.mean() * decoder_norms.mean() * self.group_weights[i]
            total_kl_loss += kl_loss
        
        # Update stats for dead features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        # Get auxiliary loss for dead features
        auxk_loss = self.get_auxiliary_loss((x - x_reconstruct).detach(), mu)
        
        # Total loss with all components
        loss = total_recon_loss + self.kl_coeff * sparsity_scale * total_kl_loss + self.auxk_alpha * auxk_loss
        
        if not logging:
            return loss
        else:
            min_recon_loss = recon_losses.min().item()
            max_recon_loss = recon_losses.max().item()
            mean_recon_loss = recon_losses.mean().item()
            
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_reconstruct, z,
                {
                    'mse_loss': total_recon_loss.item(),
                    'kl_loss': total_kl_loss.item(),
                    'auxk_loss': auxk_loss.item(),
                    'min_recon_loss': min_recon_loss,
                    'max_recon_loss': max_recon_loss,
                    'mean_recon_loss': mean_recon_loss,
                    'loss': loss.item()
                }
            )
        
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
            'dict_class': 'MatryoshkaVSAEIso',
            'trainer_class': 'MatryoshkaVSAEIsoTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'auxk_alpha': self.auxk_alpha,
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
            'group_fractions': self.group_fractions,
            'group_weights': self.group_weights,
            'group_sizes': self.group_sizes,
        }
