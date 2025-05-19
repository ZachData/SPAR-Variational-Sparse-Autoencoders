"""
Implements a combination of Variational Sparse Autoencoder (VSAE) with Gated Annealing.
This combines techniques from vsae_iso.py and gated_anneal.py.
"""

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEGatedAutoEncoder(Dictionary, nn.Module):
    """
    A variational sparse autoencoder with gating networks for feature extraction.
    
    This model combines ideas from standard VSAEs with gated networks to improve
    feature learning and interpretability through controlled sparsity.
    
    Args:
        activation_dim (int): Dimension of input activations
        dict_size (int): Size of the learned dictionary
        var_flag (int): Determines variance behavior (0: fixed, 1: learned)
        initialization (str): Weight initialization strategy
        device (torch.device): Device to place model on
    """

    def __init__(self, activation_dim, dict_size, var_flag=0, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.var_flag = var_flag
        
        # Initialize decoder bias and main components
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        
        # Gating specific parameters
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        
        # Variance encoder (only used when var_flag=1)
        if var_flag == 1:
            self.var_encoder = nn.Linear(activation_dim, dict_size, bias=True, device=device)
            init.zeros_(self.var_encoder.bias)
        
        # Initialize parameters
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing weights and biases.
        """
        # Biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)
        
        # Decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)
        
        # Initialize variance encoder if needed
        if self.var_flag == 1:
            init.kaiming_uniform_(self.var_encoder.weight)

    def encode(self, x: t.Tensor, return_gate=False, normalize_decoder=False, return_log_var=False):
        """
        Encode input activations to latent space.
        
        Args:
            x: Input activations to encode
            return_gate: Whether to return gate values
            normalize_decoder: Whether to apply decoder normalization
            return_log_var: Whether to return log variance
            
        Returns:
            Features, and optionally gate values and log variance
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # Gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # Magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        # Combined features
        f = f_gate * f_mag

        if normalize_decoder:
            # Normalizing to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
        
        # Handle variance if required
        if return_log_var and self.var_flag == 1:
            log_var = F.relu(self.var_encoder(x - self.decoder_bias))
            
            if return_gate:
                return f, nn.ReLU()(pi_gate), log_var
            return f, log_var
            
        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def reparameterize(self, mu, log_var):
        """
        Apply reparameterization trick for variational sampling.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent code
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std

    def decode(self, f: t.Tensor, normalize_decoder=False):
        """
        Decode features back to activation space.
        
        Args:
            f: Features to decode
            normalize_decoder: Whether to apply decoder normalization
            
        Returns:
            Reconstructed activations
        """
        if normalize_decoder:
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x: t.Tensor, output_features=False, normalize_decoder=False):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations
            output_features: Whether to return features along with reconstructions
            normalize_decoder: Whether to apply decoder normalization
            
        Returns:
            Reconstructed activations and optionally features
        """
        # For variational version
        if self.var_flag == 1:
            mu, log_var = self.encode(x, return_log_var=True)
            # Sample from the latent distribution
            f = self.reparameterize(mu, log_var)
        else:
            # Standard version
            f = self.encode(x)
            
        x_hat = self.decode(f)

        if normalize_decoder:
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        """
        Scale all bias parameters by a given factor.
        
        Args:
            scale: Scale factor to apply
        """
        self.decoder_bias.data *= scale
        self.mag_bias.data *= scale
        self.gate_bias.data *= scale
        
        if self.var_flag == 1:
            self.var_encoder.bias.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, var_flag=0):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            var_flag: Variance flag to use
            
        Returns:
            Loaded autoencoder
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict.get("encoder.weight", state_dict.get("W_enc", None)).shape
        
        # Convert from old format if necessary
        if "W_enc" in state_dict:
            converted_dict = {}
            converted_dict["encoder.weight"] = state_dict["W_enc"].T
            converted_dict["decoder.weight"] = state_dict["W_dec"].T
            converted_dict["decoder_bias"] = state_dict["b_dec"]
            converted_dict["r_mag"] = state_dict["r_mag"]
            converted_dict["gate_bias"] = state_dict["gate_bias"]
            converted_dict["mag_bias"] = state_dict["mag_bias"]
            
            if var_flag == 1 and "W_enc_var" in state_dict:
                converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                
            state_dict = converted_dict
        
        autoencoder = cls(activation_dim, dict_size, var_flag=var_flag, device=device)
        autoencoder.load_state_dict(state_dict)
        
        if device is not None:
            autoencoder.to(device)
            
        return autoencoder


class VSAEGatedAnnealTrainer(SAETrainer):
    """
    A trainer that combines Variational Sparse Autoencoding (VSAE) with Gated Annealing.
    
    This trainer uses both variational techniques and p-norm annealing to achieve
    better feature learning and controlled sparsity.
    
    Args:
        steps: Total number of training steps
        activation_dim: Dimension of input activations
        dict_size: Size of the learned dictionary
        layer: Layer index in the model
        lm_name: Language model name
        dict_class: Dictionary class to use
        lr: Learning rate
        kl_coeff: KL divergence penalty coefficient
        anneal_start: Step to start annealing p
        anneal_end: Step to end annealing p
        p_start: Initial p value
        p_end: Final p value
        var_flag: Variance flag (0: fixed, 1: learned)
        sparsity_function: Type of sparsity function (Lp or Lp^p)
        n_sparsity_updates: Number of sparsity updates during annealing
        sparsity_queue_length: Length of sparsity queue for adaptive updates
        warmup_steps: Learning rate warmup steps
        sparsity_warmup_steps: Sparsity warmup steps
        decay_start: Learning rate decay start step
        resample_steps: Steps between neuron resampling
        device: Device to use
        seed: Random seed
        wandb_name: W&B experiment name
    """
    
    def __init__(self,
                 steps: int,
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=VSAEGatedAutoEncoder,
                 lr: float = 5e-5,
                 kl_coeff: float = 5e4,
                 anneal_start: int = 15000,
                 anneal_end: Optional[int] = None,
                 p_start: float = 1,
                 p_end: float = 0,
                 var_flag: int = 0,
                 sparsity_function: str = 'Lp^p',
                 n_sparsity_updates: int | str = 10,
                 sparsity_queue_length: int = 10,
                 warmup_steps: int = 1000,
                 sparsity_warmup_steps: Optional[int] = None,
                 decay_start: Optional[int] = None,
                 resample_steps: Optional[int] = None,
                 device=None,
                 seed: Optional[int] = 42,
                 wandb_name: Optional[str] = 'VSAEGatedAnnealTrainer',
                 submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.var_flag = var_flag

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Use the defaults from both methods if not specified
        if sparsity_warmup_steps is None:
            sparsity_warmup_steps = int(0.05 * steps)  # 5% of steps (from VSAE)
        
        if decay_start is None:
            decay_start = int(0.8 * steps)  # Start decay at 80% of training (from VSAE)
            
        anneal_end = anneal_end if anneal_end is not None else steps

        # Initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ae = dict_class(
            activation_dim=activation_dim, 
            dict_size=dict_size, 
            var_flag=var_flag,
            device=self.device
        )
        self.ae.to(self.device)

        # Core parameters
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        
        # Annealing parameters
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start  # p is set in self.loss()
        self.next_p = None  # set in self.loss()
        self.kl_loss = None  # set in self.loss()
        self.scaled_kl_loss = None  # set in self.loss()
        
        if n_sparsity_updates == "continuous":
            self.n_sparsity_updates = self.anneal_end - anneal_start + 1
        else:
            self.n_sparsity_updates = n_sparsity_updates
            
        self.sparsity_update_steps = t.linspace(anneal_start, self.anneal_end, self.n_sparsity_updates, dtype=int)
        self.p_values = t.linspace(p_start, p_end, self.n_sparsity_updates)
        self.p_step_count = 0
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []
        
        # Resampling for dead neurons
        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # How many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        # Optimization setup
        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr, betas=(0.9, 0.999))
        
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Logging parameters
        self.logging_parameters = ['p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'kl_coeff']
        
    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons to prevent wasted capacity.
        
        Args:
            deads: Boolean tensor indicating dead neurons
            activations: Batch of activations to sample from
        """
        with t.no_grad():
            if deads.sum() == 0: 
                return
                
            print(f"Resampling {deads.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()
            
            # Handle encoder and decoder weight resampling
            self.ae.encoder.weight[deads][:n_resample] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads][:,:n_resample] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            
            # Reset biases
            self.ae.gate_bias[deads][:n_resample] = 0.
            self.ae.mag_bias[deads][:n_resample] = 0.

            # Reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            
            for param_idx, param in enumerate(self.ae.parameters()):
                if param_idx in state_dict and param.shape == self.ae.encoder.weight.shape:
                    state_dict[param_idx]['exp_avg'][deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads] = 0.
                elif param_idx in state_dict and param.shape == self.ae.decoder.weight.shape:
                    state_dict[param_idx]['exp_avg'][:,deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][:,deads] = 0.
                elif param_idx in state_dict and param.shape == self.ae.gate_bias.shape:
                    state_dict[param_idx]['exp_avg'][deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads] = 0.
                elif param_idx in state_dict and param.shape == self.ae.mag_bias.shape:
                    state_dict[param_idx]['exp_avg'][deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads] = 0.
    
    def kl_divergence_p(self, mu, log_var=None, p=1.0):
        """
        Calculate KL divergence with p-norm regularization.
        
        For standard VAE when p=2, this is a standard KL divergence term.
        As p approaches 0, we get more sparse activations.
        
        Args:
            mu: Mean values of the latent distribution
            log_var: Log variance values (if None, assume fixed variance)
            p: p-norm parameter value
            
        Returns:
            KL divergence loss
        """
        if p == 2.0:
            # Standard KL divergence for VAE with Gaussian prior
            if log_var is not None:
                kl = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
            else:
                kl = 0.5 * mu.pow(2).sum(dim=-1)
                
            return kl.mean()
        else:
            # Modified p-norm version for sparsity
            if self.sparsity_function == 'Lp^p':
                return mu.pow(p).sum(dim=-1).mean()
            elif self.sparsity_function == 'Lp':
                return mu.pow(p).sum(dim=-1).pow(1/p).mean()
            else:
                raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
        
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Calculate loss for the VSAE with p-annealing.
        
        Args:
            x: Input activations
            step: Current training step
            logging: Whether to return logging info
            
        Returns:
            Loss value, or loss components if logging=True
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Handle variational encoding
        if self.var_flag == 1:
            # Encode with variance
            mu, gate, log_var = self.ae.encode(x, return_gate=True, return_log_var=True)
            # Sample from the latent distribution
            z = self.ae.reparameterize(mu, log_var)
            # Decode
            x_hat = self.ae.decode(z)
            # For auxiliary loss
            x_hat_gate = gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()
            
            # Feature activations for sparsity
            fs = gate
        else:
            # Standard encoding
            mu, gate = self.ae.encode(x, return_gate=True)
            # Just use mean
            z = mu
            # Decode
            x_hat = self.ae.decode(z)
            # For auxiliary loss
            x_hat_gate = gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()
            
            # Feature activations for sparsity
            fs = gate
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # Auxiliary loss to help with gating
        aux_loss = (x - x_hat_gate).pow(2).sum(dim=-1).mean()
        
        # KL divergence with p-annealing
        if self.var_flag == 1:
            kl_loss = self.kl_divergence_p(fs, log_var, self.p)
        else:
            kl_loss = self.kl_divergence_p(fs, None, self.p)
            
        # Apply decoder norm scaling for better regularization
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
        scaled_kl_loss = (kl_loss * decoder_norms.mean()) * self.kl_coeff * sparsity_scale
        
        self.kl_loss = kl_loss
        self.scaled_kl_loss = scaled_kl_loss

        # P-annealing handling
        if self.next_p is not None:
            if self.var_flag == 1:
                kl_next = self.kl_divergence_p(fs, log_var, self.next_p)
            else:
                kl_next = self.kl_divergence_p(fs, None, self.next_p)
                
            self.sparsity_queue.append([self.kl_loss.item(), kl_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]
    
        # Update p-value at scheduled steps
        if step in self.sparsity_update_steps:
            # Check to make sure we don't update on repeat step
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt KL coefficient
                if self.next_p is not None and len(self.sparsity_queue) > 0:
                    local_sparsity_new = t.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = t.tensor([i[1] for i in self.sparsity_queue]).mean()
                    if local_sparsity_old > 0:  # Avoid division by zero
                        self.kl_coeff = self.kl_coeff * (local_sparsity_new / local_sparsity_old).item()
                
                # Update p
                self.p = self.p_values[self.p_step_count].item()
                if self.p_step_count < self.n_sparsity_updates-1:
                    self.next_p = self.p_values[self.p_step_count+1].item()
                else:
                    self.next_p = self.p_end
                self.p_step_count += 1

        # Update dead feature count
        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (z == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0       
            
        # Total loss
        loss = recon_loss + scaled_kl_loss + aux_loss
    
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'mse_loss': recon_loss.item(),
                    'aux_loss': aux_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'scaled_kl_loss': scaled_kl_loss.item(),
                    'loss': loss.item(),
                    'p': self.p,
                    'next_p': self.next_p,
                    'kl_coeff': self.kl_coeff,
                }
            )
        
    def update(self, step, activations):
        """
        Update the model parameters for one step.
        
        Args:
            step: Current training step
            activations: Batch of activations
        """
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step, logging=False)
        loss.backward()
        
        # Apply gradient clipping (recommended for stability)
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

        # Resample dead neurons if needed
        if self.resample_steps is not None and step % self.resample_steps == self.resample_steps - 1:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        """
        Return the configuration of this trainer.
        """
        return {
            'dict_class': 'VSAEGatedAutoEncoder',
            'trainer_class': 'VSAEGatedAnnealTrainer',
            'activation_dim': self.activation_dim,
            'dict_size': self.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'sparsity_function': self.sparsity_function,
            'p_start': self.p_start,
            'p_end': self.p_end,
            'var_flag': self.var_flag,
            'anneal_start': self.anneal_start,
            'anneal_end': self.anneal_end,
            'sparsity_queue_length': self.sparsity_queue_length,
            'n_sparsity_updates': self.n_sparsity_updates,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'resample_steps': self.resample_steps,
            'decay_start': self.decay_start,
            'steps': self.steps,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }
