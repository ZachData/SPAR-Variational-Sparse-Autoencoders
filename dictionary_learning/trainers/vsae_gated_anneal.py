"""
Improved implementation of combined Variational Sparse Autoencoder (VSAE) with Gated Annealing.

This module combines variational techniques with p-norm annealing to achieve
better feature learning and controlled sparsity, following better PyTorch practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, List, Tuple, Dict, Any
from collections import namedtuple
from dataclasses import dataclass
import math

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    ConstrainedAdam
)


@dataclass
class VSAEGatedConfig:
    """Configuration for VSAE Gated model."""
    activation_dim: int
    dict_size: int
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    dtype: torch.dtype = torch.float32
    device: Optional[torch.device] = None


@dataclass
class AnnealingConfig:
    """Configuration for p-norm annealing schedule."""
    anneal_start: int
    anneal_end: int
    p_start: float = 1.0
    p_end: float = 0.0
    n_sparsity_updates: int = 10
    sparsity_function: str = 'Lp^p'  # 'Lp' or 'Lp^p'
    sparsity_queue_length: int = 10
    
    def __post_init__(self):
        if self.sparsity_function not in ['Lp', 'Lp^p']:
            raise ValueError("sparsity_function must be 'Lp' or 'Lp^p'")
        if self.anneal_end <= self.anneal_start:
            raise ValueError("anneal_end must be greater than anneal_start")
        if self.n_sparsity_updates < 1:
            raise ValueError("n_sparsity_updates must be at least 1")
        if self.anneal_end - self.anneal_start < self.n_sparsity_updates:
            # Adjust n_sparsity_updates if the annealing period is too short
            max_updates = max(1, self.anneal_end - self.anneal_start)
            print(f"Warning: n_sparsity_updates ({self.n_sparsity_updates}) too large for annealing period "
                  f"({self.anneal_end - self.anneal_start} steps). Reducing to {max_updates}")
            self.n_sparsity_updates = max_updates


@dataclass 
class TrainingConfig:
    """Configuration for training the VSAEGated model."""
    steps: int
    lr: float = 5e-4
    kl_coeff: float = 500.0
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        # Set defaults based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(1000, int(0.05 * self.steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.steps)
        if self.decay_start is None:
            self.decay_start = int(0.8 * self.steps)


class VSAEGatedAutoEncoder(Dictionary, nn.Module):
    """
    A variational sparse autoencoder with gating networks for feature extraction.
    
    This model combines ideas from standard VSAEs with gated networks to improve
    feature learning and interpretability through controlled sparsity.
    """

    def __init__(self, config: VSAEGatedConfig):
        super().__init__()
        self.config = config
        self.activation_dim = config.activation_dim
        self.dict_size = config.dict_size
        self.var_flag = config.var_flag
        
        # Initialize layers
        self._init_layers()
        self._init_weights()
    
    def _init_layers(self) -> None:
        """Initialize neural network layers."""
        # Main encoder/decoder
        self.encoder = nn.Linear(
            self.activation_dim, 
            self.dict_size, 
            bias=False,
            dtype=self.config.dtype,
            device=self.config.device
        )
        self.decoder = nn.Linear(
            self.dict_size, 
            self.activation_dim, 
            bias=False,
            dtype=self.config.dtype,
            device=self.config.device
        )
        
        # Bias parameters
        self.decoder_bias = nn.Parameter(
            torch.empty(self.activation_dim, dtype=self.config.dtype, device=self.config.device)
        )
        
        # Gating specific parameters
        self.r_mag = nn.Parameter(
            torch.empty(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        self.gate_bias = nn.Parameter(
            torch.empty(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        self.mag_bias = nn.Parameter(
            torch.empty(self.dict_size, dtype=self.config.dtype, device=self.config.device)
        )
        
        # Variance encoder (only used when var_flag=1)
        if self.var_flag == 1:
            self.var_encoder = nn.Linear(
                self.activation_dim, 
                self.dict_size, 
                bias=True,
                dtype=self.config.dtype,
                device=self.config.device
            )
    
    def _init_weights(self) -> None:
        """Initialize model weights following best practices."""
        # Biases are initialized to zero
        with torch.no_grad():
            nn.init.zeros_(self.decoder_bias)
            nn.init.zeros_(self.r_mag)
            nn.init.zeros_(self.gate_bias)
            nn.init.zeros_(self.mag_bias)
            
            # Decoder weights are initialized to random unit vectors
            dec_weight = torch.randn_like(self.decoder.weight)
            dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
            self.decoder.weight.copy_(dec_weight)
            self.encoder.weight.copy_(dec_weight.clone().T)
            
            # Initialize variance encoder if needed
            if self.var_flag == 1:
                nn.init.kaiming_uniform_(self.var_encoder.weight)
                nn.init.zeros_(self.var_encoder.bias)

    def encode(
        self, 
        x: torch.Tensor, 
        return_gate: bool = False, 
        normalize_decoder: bool = False, 
        return_log_var: bool = False
    ) -> Tuple[torch.Tensor, ...]:
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
        # Ensure input matches encoder dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
        x_enc = self.encoder(x - self.decoder_bias)

        # Gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # Magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)

        # Combined features
        f = f_gate * f_mag

        if normalize_decoder:
            # Normalizing to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
        
        # Handle variance if required
        if return_log_var and self.var_flag == 1:
            log_var = F.relu(self.var_encoder(x - self.decoder_bias))
            
            if return_gate:
                return f, F.relu(pi_gate), log_var
            return f, log_var
            
        if return_gate:
            return f, F.relu(pi_gate)

        return f

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply reparameterization trick for variational sampling.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            Sampled latent code
        """
        if log_var is None:
            return mu
            
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Ensure output matches mu dtype
        return z.to(dtype=mu.dtype)

    def decode(self, f: torch.Tensor, normalize_decoder: bool = False) -> torch.Tensor:
        """
        Decode features back to activation space.
        
        Args:
            f: Features to decode
            normalize_decoder: Whether to apply decoder normalization
            
        Returns:
            Reconstructed activations
        """
        # Ensure f matches decoder weight dtype
        f = f.to(dtype=self.decoder.weight.dtype)
        
        if normalize_decoder:
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(
        self, 
        x: torch.Tensor, 
        output_features: bool = False, 
        normalize_decoder: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations
            output_features: Whether to return features along with reconstructions
            normalize_decoder: Whether to apply decoder normalization
            
        Returns:
            Reconstructed activations and optionally features
        """
        # Store original dtype to return output in same format
        original_dtype = x.dtype
        
        # Ensure input matches model dtype
        x = x.to(dtype=self.encoder.weight.dtype)
        
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

        # Convert back to original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        
        if output_features:
            f = f.to(dtype=original_dtype)
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float) -> None:
        """
        Scale all bias parameters by a given factor.
        
        Args:
            scale: Scale factor to apply
        """
        with torch.no_grad():
            self.decoder_bias.mul_(scale)
            self.mag_bias.mul_(scale)
            self.gate_bias.mul_(scale)
            
            if self.var_flag == 1:
                self.var_encoder.bias.mul_(scale)

    @classmethod
    def from_pretrained(
        cls, 
        path: str, 
        config: Optional[VSAEGatedConfig] = None,
        dtype: torch.dtype = torch.float32, 
        device: Optional[torch.device] = None, 
        normalize_decoder: bool = True, 
        var_flag: Optional[int] = None
    ) -> 'VSAEGatedAutoEncoder':
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            config: Model configuration (if None, will auto-detect from state_dict)
            dtype: Data type to convert model to
            device: Device to load model on
            normalize_decoder: Whether to normalize decoder weights
            var_flag: Whether to load with variance encoding (0: fixed, 1: learned)
                    If None, will auto-detect from state_dict
            
        Returns:
            Loaded autoencoder
        """
        state_dict = torch.load(path, map_location=device)
        
        if config is None:
            # Auto-detect config from state_dict
            if var_flag is None:
                # Check if the state dict contains variance encoder weights
                has_var_encoder = "var_encoder.weight" in state_dict or "W_enc_var" in state_dict
                var_flag = 1 if has_var_encoder else 0
            
            # Determine dimensions from state dict
            if 'encoder.weight' in state_dict:
                dict_size, activation_dim = state_dict["encoder.weight"].shape
            else:
                # Handle older format with W_enc, W_dec parameters
                activation_dim, dict_size = state_dict["W_enc"].shape if "W_enc" in state_dict else state_dict["encoder.weight"].T.shape
                
                # Convert parameter names if needed
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
            
            config = VSAEGatedConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=var_flag,
                dtype=dtype,
                device=device
            )
        
        # Create model with detected parameters
        autoencoder = cls(config)
        
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
            if var_flag == 1 and "var_encoder.weight" in missing_keys:
                print("Initializing missing variance encoder parameters with default values")
                with torch.no_grad():
                    nn.init.kaiming_uniform_(autoencoder.var_encoder.weight)
                    nn.init.zeros_(autoencoder.var_encoder.bias)
        
        if device is not None:
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        
        return autoencoder


class DeadFeatureTracker:
    """Tracks dead features for resampling."""
    
    def __init__(self, dict_size: int, device: torch.device):
        self.steps_since_active = torch.zeros(
            dict_size, dtype=torch.long, device=device
        )
    
    def update(self, active_features: torch.Tensor) -> torch.Tensor:
        """Update dead feature tracking and return dead feature mask."""
        # Update counters
        deads = (active_features == 0).all(dim=0)
        self.steps_since_active[deads] += 1
        self.steps_since_active[~deads] = 0
        
        return self.steps_since_active
    
    def get_dead_mask(self, threshold: int) -> torch.Tensor:
        """Get boolean mask of dead features."""
        return self.steps_since_active > threshold


class VSAEGatedAnnealTrainer(SAETrainer):
    """
    A trainer that combines Variational Sparse Autoencoding (VSAE) with Gated Annealing.
    
    This trainer uses both variational techniques and p-norm annealing to achieve
    better feature learning and controlled sparsity.
    """
    
    def __init__(
        self,
        model_config: VSAEGatedConfig = None,
        training_config: TrainingConfig = None,
        annealing_config: AnnealingConfig = None,
        layer: int = None,
        lm_name: str = None,
        submodule_name: Optional[str] = None,
        wandb_name: Optional[str] = None,
        seed: Optional[int] = None,
        # Alternative parameters for backwards compatibility
        steps: Optional[int] = None,
        activation_dim: Optional[int] = None,
        dict_size: Optional[int] = None,
        lr: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        var_flag: Optional[int] = None,
        anneal_start: Optional[int] = None,
        anneal_end: Optional[int] = None,
        p_start: Optional[float] = None,
        p_end: Optional[float] = None,
        n_sparsity_updates: Optional[int] = None,
        sparsity_function: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs  # Catch any other parameters
    ):
        super().__init__(seed)
        
        # Handle backwards compatibility - if individual parameters are passed, create configs
        if model_config is None or training_config is None or annealing_config is None:
            # Create configs from individual parameters
            if model_config is None:
                if activation_dim is None or dict_size is None:
                    raise ValueError("Must provide either model_config or activation_dim + dict_size")
                
                device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                model_config = VSAEGatedConfig(
                    activation_dim=activation_dim,
                    dict_size=dict_size,
                    var_flag=var_flag or 0,
                    device=device_obj
                )
            
            if training_config is None:
                if steps is None:
                    raise ValueError("Must provide either training_config or steps")
                
                training_config = TrainingConfig(
                    steps=steps,
                    lr=lr or 5e-4,
                    kl_coeff=kl_coeff or 500.0,
                )
            
            if annealing_config is None:
                if anneal_start is None or anneal_end is None:
                    raise ValueError("Must provide either annealing_config or anneal_start + anneal_end")
                
                annealing_config = AnnealingConfig(
                    anneal_start=anneal_start,
                    anneal_end=anneal_end,
                    p_start=p_start or 1.0,
                    p_end=p_end or 0.0,
                    n_sparsity_updates=n_sparsity_updates or 10,
                    sparsity_function=sparsity_function or 'Lp^p'
                )
        
        self.model_config = model_config
        self.training_config = training_config
        self.annealing_config = annealing_config
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name or "VSAEGatedAnnealTrainer"
        
        # Set device
        self.device = model_config.device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        # Initialize model
        self.ae = VSAEGatedAutoEncoder(model_config)
        self.ae.to(self.device)
        
        # Initialize annealing state
        self.p = annealing_config.p_start
        self.next_p = None
        
        # Create annealing schedule - ensure we have the right number of steps
        if annealing_config.n_sparsity_updates > 1:
            self.sparsity_update_steps = torch.linspace(
                annealing_config.anneal_start, 
                annealing_config.anneal_end, 
                annealing_config.n_sparsity_updates, 
                dtype=torch.long
            )
            self.p_values = torch.linspace(
                annealing_config.p_start, 
                annealing_config.p_end, 
                annealing_config.n_sparsity_updates
            )
        else:
            # Handle edge case where we only have 1 update
            self.sparsity_update_steps = torch.tensor([annealing_config.anneal_start], dtype=torch.long)
            self.p_values = torch.tensor([annealing_config.p_end])
            
        self.p_step_count = 0
        self.sparsity_queue = []
        
        # Set the initial next_p if we have more than one update
        if len(self.p_values) > 1:
            self.next_p = self.p_values[1].item()
        else:
            self.next_p = annealing_config.p_end
        
        # Initialize dead feature tracking
        if training_config.resample_steps is not None:
            self.dead_feature_tracker = DeadFeatureTracker(
                model_config.dict_size,
                self.device
            )
        else:
            self.dead_feature_tracker = None
        
        # Initialize optimizer and scheduler
        self.optimizer = ConstrainedAdam(
            self.ae.parameters(), 
            self.ae.decoder.parameters(), 
            lr=training_config.lr,
            betas=(0.9, 0.999)
        )
        
        lr_fn = get_lr_schedule(
            training_config.steps,
            training_config.warmup_steps,
            training_config.decay_start,
            training_config.resample_steps,
            training_config.sparsity_warmup_steps
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(
            training_config.steps, 
            training_config.sparsity_warmup_steps
        )
        
        # Logging parameters
        self.logging_parameters = ['p', 'next_p', 'kl_loss', 'scaled_kl_loss', 'kl_coeff']
        self.kl_loss = None
        self.scaled_kl_loss = None
        self.kl_coeff = training_config.kl_coeff
        
    def _compute_kl_divergence_p(
        self, 
        mu: torch.Tensor, 
        log_var: Optional[torch.Tensor] = None, 
        p: float = 1.0
    ) -> torch.Tensor:
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
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
            else:
                kl = 0.5 * mu.pow(2).sum(dim=-1)
                
            return kl.mean()
        else:
            # Modified p-norm version for sparsity
            if self.annealing_config.sparsity_function == 'Lp^p':
                return mu.pow(p).sum(dim=-1).mean()
            elif self.annealing_config.sparsity_function == 'Lp':
                return mu.pow(p).sum(dim=-1).pow(1/p).mean()
            else:
                raise ValueError("Sparsity function must be 'Lp' or 'Lp^p'")
    
    def _resample_neurons(self, dead_mask: torch.Tensor, activations: torch.Tensor) -> None:
        """
        Resample dead neurons to prevent wasted capacity.
        
        Args:
            dead_mask: Boolean tensor indicating dead neurons
            activations: Batch of activations to sample from
        """
        with torch.no_grad():
            if dead_mask.sum() == 0: 
                return
                
            print(f"Resampling {dead_mask.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([dead_mask.sum(), losses.shape[0]])
            indices = torch.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~dead_mask].norm(dim=-1).mean()
            
            # Handle encoder and decoder weight resampling
            self.ae.encoder.weight[dead_mask][:n_resample] = sampled_vecs * alive_norm * 0.2
            decoder_dtype = self.ae.decoder.weight.dtype
            normalized_vecs = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.decoder.weight[:,dead_mask][:,:n_resample] = normalized_vecs.to(dtype=decoder_dtype)
            
            # Reset biases
            self.ae.gate_bias[dead_mask][:n_resample] = 0.
            self.ae.mag_bias[dead_mask][:n_resample] = 0.

            # Reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            
            # Reset for encoder weight (parameter index 1)
            if 1 in state_dict:
                state_dict[1]['exp_avg'][dead_mask] = 0.
                state_dict[1]['exp_avg_sq'][dead_mask] = 0.
            
            # Reset for decoder weight (parameter index 3) 
            if 3 in state_dict:
                state_dict[3]['exp_avg'][:,dead_mask] = 0.
                state_dict[3]['exp_avg_sq'][:,dead_mask] = 0.
    
    def loss(self, x: torch.Tensor, step: int, logging: bool = False) -> torch.Tensor:
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
        
        # Store original dtype for final output
        original_dtype = x.dtype
        
        # Handle variational encoding
        if self.model_config.var_flag == 1:
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
        
        # Ensure outputs match original dtype
        x_hat = x_hat.to(dtype=original_dtype)
        x_hat_gate = x_hat_gate.to(dtype=original_dtype)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # Auxiliary loss to help with gating
        aux_loss = (x - x_hat_gate).pow(2).sum(dim=-1).mean()
        
        # KL divergence with p-annealing
        if self.model_config.var_flag == 1:
            kl_loss = self._compute_kl_divergence_p(fs, log_var, self.p)
        else:
            kl_loss = self._compute_kl_divergence_p(fs, None, self.p)
            
        # Apply decoder norm scaling for better regularization
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
        scaled_kl_loss = (kl_loss * decoder_norms.mean()) * self.kl_coeff * sparsity_scale
        
        # Store for logging
        self.kl_loss = kl_loss
        self.scaled_kl_loss = scaled_kl_loss

        # P-annealing handling
        if self.next_p is not None:
            if self.model_config.var_flag == 1:
                kl_next = self._compute_kl_divergence_p(fs, log_var, self.next_p)
            else:
                kl_next = self._compute_kl_divergence_p(fs, None, self.next_p)
                
            self.sparsity_queue.append([self.kl_loss.item(), kl_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.annealing_config.sparsity_queue_length:]
    
        # Update p-value at scheduled steps
        if step in self.sparsity_update_steps and self.p_step_count < len(self.sparsity_update_steps):
            # Check to make sure we don't update on repeat step
            if step >= self.sparsity_update_steps[self.p_step_count]:
                # Adapt KL coefficient
                if self.next_p is not None and len(self.sparsity_queue) > 0:
                    local_sparsity_new = torch.tensor([i[0] for i in self.sparsity_queue]).mean()
                    local_sparsity_old = torch.tensor([i[1] for i in self.sparsity_queue]).mean()
                    if local_sparsity_old > 0:  # Avoid division by zero
                        self.kl_coeff = self.kl_coeff * (local_sparsity_new / local_sparsity_old).item()
                
                # Update p only if we haven't exceeded the number of updates
                if self.p_step_count < len(self.p_values):
                    self.p = self.p_values[self.p_step_count].item()
                    
                    # Set next_p for the queue
                    if self.p_step_count < len(self.p_values) - 1:
                        self.next_p = self.p_values[self.p_step_count + 1].item()
                    else:
                        self.next_p = self.annealing_config.p_end
                        
                    self.p_step_count += 1

        # Update dead feature count
        if self.dead_feature_tracker is not None:
            self.dead_feature_tracker.update(z)
            
        # Convert loss components to original dtype
        recon_loss = recon_loss.to(dtype=original_dtype)
        aux_loss = aux_loss.to(dtype=original_dtype)
        scaled_kl_loss = scaled_kl_loss.to(dtype=original_dtype)
            
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
        
    def update(self, step: int, activations: torch.Tensor) -> None:
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
        torch.nn.utils.clip_grad_norm_(
            self.ae.parameters(), 
            self.training_config.gradient_clip_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()

        # Resample dead neurons if needed
        if (self.training_config.resample_steps is not None and 
            step % self.training_config.resample_steps == self.training_config.resample_steps - 1 and
            self.dead_feature_tracker is not None):
            
            dead_mask = self.dead_feature_tracker.get_dead_mask(self.training_config.resample_steps // 2)
            self._resample_neurons(dead_mask, activations)

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the configuration of this trainer (JSON serializable).
        """
        return {
            'dict_class': 'VSAEGatedAutoEncoder',
            'trainer_class': 'VSAEGatedAnnealTrainer',
            # Model config (serializable)
            'activation_dim': self.model_config.activation_dim,
            'dict_size': self.model_config.dict_size,
            'var_flag': self.model_config.var_flag,
            'dtype': str(self.model_config.dtype),
            'device': str(self.model_config.device),
            # Training config (serializable)
            'steps': self.training_config.steps,
            'lr': self.training_config.lr,
            'kl_coeff': self.training_config.kl_coeff,
            'warmup_steps': self.training_config.warmup_steps,
            'sparsity_warmup_steps': self.training_config.sparsity_warmup_steps,
            'decay_start': self.training_config.decay_start,
            'resample_steps': self.training_config.resample_steps,
            'gradient_clip_norm': self.training_config.gradient_clip_norm,
            # Annealing config (serializable)
            'anneal_start': self.annealing_config.anneal_start,
            'anneal_end': self.annealing_config.anneal_end,
            'p_start': self.annealing_config.p_start,
            'p_end': self.annealing_config.p_end,
            'n_sparsity_updates': self.annealing_config.n_sparsity_updates,
            'sparsity_function': self.annealing_config.sparsity_function,
            'sparsity_queue_length': self.annealing_config.sparsity_queue_length,
            # Other attributes
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'seed': self.seed,
        }