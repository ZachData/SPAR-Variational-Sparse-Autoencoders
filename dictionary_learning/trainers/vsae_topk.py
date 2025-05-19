"""
Implements a hybrid Variational Sparse Autoencoder with Top-K activation mechanism.
This combines the variational approach from VSAEIso with the structured sparsity of TopK.
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional, Tuple
from collections import namedtuple

from ..trainers.trainer import (
    SAETrainer, 
    get_lr_schedule, 
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions
)
from ..dictionary import Dictionary
from ..config import DEBUG


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class VSAETopK(Dictionary, nn.Module):
    """
    A hybrid dictionary that combines the variational approach from VSAEIso 
    with the structured Top-K sparsity mechanism.
    
    This model uses:
    1. The variational sampling approach for learning feature distributions
    2. The Top-K activation mechanism to enforce structured sparsity
    3. Support for both fixed and learned variance
    
    Args:
        activation_dim: Dimension of the activation vectors
        dict_size: Number of features in the dictionary
        k: Number of top features to keep active
        var_flag: Whether to learn variance (0: fixed, 1: learned)
        use_april_update_mode: Whether to use bias in both encoder and decoder
        device: Device to initialize tensors on
    """

    def __init__(
        self, 
        activation_dim: int, 
        dict_size: int, 
        k: int,
        var_flag: int = 0,
        use_april_update_mode: bool = True,
        device=None
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.var_flag = var_flag
        self.use_april_update_mode = use_april_update_mode
        
        # Register k as a buffer so it's saved with the model
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        # Threshold for activation filtering (used as alternative to explicit top-k)
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))
        
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
            self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
            
        # Variance network (only used when var_flag=1)
        if var_flag == 1:
            self.var_encoder = nn.Linear(activation_dim, dict_size, bias=True)
            init.kaiming_uniform_(self.var_encoder.weight)
            init.zeros_(self.var_encoder.bias)

    def encode(
        self, 
        x: t.Tensor, 
        output_log_var: bool = False,
        return_topk: bool = False,
        use_threshold: bool = False,
        training: bool = True
    ):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input activation tensor
            output_log_var: Whether to return log variance
            return_topk: Whether to return top-k indices and values
            use_threshold: Whether to use threshold for filtering instead of explicit top-k
            training: Whether in training mode (affects sampling)
            
        Returns:
            Encoded features and optionally log variance, top-k indices, and pre-filter activations
        """
        # Get the bias term for subtraction
        if self.use_april_update_mode:
            bias_term = 0
        else:
            bias_term = self.b_dec
            
        # Compute the mean activations
        mu = self.encoder(x - bias_term)
        
        # Compute the log variance if needed
        if output_log_var or self.var_flag == 1:
            if self.var_flag == 1:
                log_var = self.var_encoder(x - bias_term)
            else:
                log_var = t.zeros_like(mu)
                
        # Apply ReLU to mean activations
        post_relu_mu = F.relu(mu)
            
        # If in training mode and using variational approach, sample from distribution
        if training and self.var_flag == 1:
            std = t.exp(0.5 * log_var)
            eps = t.randn_like(std)
            # Sample using reparameterization trick
            z = mu + eps * std
            post_relu_z = F.relu(z)
        else:
            post_relu_z = post_relu_mu
            
        # Apply top-k or threshold filtering
        if use_threshold:
            encoded_acts = post_relu_z * (post_relu_z > self.threshold)
            if return_topk:
                post_topk = post_relu_z.topk(self.k, sorted=False, dim=-1)
                return encoded_acts, post_topk.values, post_topk.indices, post_relu_z
            else:
                return encoded_acts
        else:
            # Explicit top-k selection
            post_topk = post_relu_z.topk(self.k, sorted=False, dim=-1)
            tops_acts = post_topk.values
            top_indices = post_topk.indices

            # Create sparse feature vector with only top-k activations
            buffer = t.zeros_like(post_relu_z)
            encoded_acts = buffer.scatter_(dim=-1, index=top_indices, src=tops_acts)
            
            if return_topk:
                if output_log_var:
                    return encoded_acts, log_var, tops_acts, top_indices, post_relu_z
                else:
                    return encoded_acts, tops_acts, top_indices, post_relu_z
            else:
                if output_log_var:
                    return encoded_acts, log_var
                else:
                    return encoded_acts

    def decode(self, f: t.Tensor) -> t.Tensor:
        """
        Decode a dictionary vector f.
        """
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False, training: bool = True):
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Input tensor
            output_features: Whether to return features as well
            training: Whether in training mode (affects sampling behavior)
            
        Returns:
            Reconstructed tensor and optionally features
        """
        encoded_acts = self.encode(x, training=training)
        x_hat = self.decode(encoded_acts)
        
        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        """
        Scale all bias parameters by a given factor.
        """
        self.encoder.bias.data *= scale
        if self.use_april_update_mode:
            self.decoder.bias.data *= scale
        else:
            self.b_dec.data *= scale
            
        if self.var_flag == 1:
            self.var_encoder.bias.data *= scale
            
        if self.threshold >= 0:
            self.threshold *= scale

    def normalize_decoder(self):
        """
        Normalize decoder weights to have unit norm.
        """
        norms = t.norm(self.decoder.weight, dim=0).to(
            dtype=self.decoder.weight.dtype, 
            device=self.decoder.weight.device
        )

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim, device=self.decoder.weight.device)
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
    def from_pretrained(
        cls, 
        path, 
        k: Optional[int] = None, 
        dtype=t.float, 
        device=None, 
        normalize_decoder=True, 
        var_flag=0
    ):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            k: Number of top features to keep active (can override saved value)
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
                    converted_dict["b_dec"] = state_dict["b_dec"]
                    
                if var_flag == 1 and "W_enc_var" in state_dict:
                    converted_dict["var_encoder.weight"] = state_dict["W_enc_var"].T
                    converted_dict["var_encoder.bias"] = state_dict["b_enc_var"]
                    
                state_dict = converted_dict
                
        # Get k value
        if k is None:
            if "k" in state_dict:
                k = state_dict["k"].item()
            else:
                # Default to 10% of dict_size if not specified
                k = max(1, dict_size // 10)
        elif "k" in state_dict and k != state_dict["k"].item():
            print(f"Warning: Overriding saved k={state_dict['k'].item()} with provided k={k}")
        
        autoencoder = cls(
            activation_dim, 
            dict_size, 
            k=k,
            use_april_update_mode=use_april_update_mode, 
            var_flag=var_flag,
            device=device
        )
        
        # Load compatible parameters
        model_dict = autoencoder.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        autoencoder.load_state_dict(model_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class VSAETopKTrainer(SAETrainer):
    """
    Trainer for the hybrid VSAETopK model that combines variational and top-k approaches.
    
    This trainer:
    1. Uses KL divergence for the variational component
    2. Incorporates the auxiliary loss mechanism from TopKTrainer for dead features
    3. Maintains a dynamic activation threshold as an alternative to explicit top-k
    4. Supports both fixed and learned variance modes
    
    Core features:
    - KL regularization weighted by decoder norms (from VSAEIso)
    - Dynamic threshold for activations (from TopK)
    - Auxiliary loss for reviviing dead features (from TopK)
    - Optional learned variance
    """
    
    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class=VSAETopK,
        lr: float = 5e-5,
        kl_coeff: float = 5.0,
        auxk_alpha: float = 1/32,
        warmup_steps: int = 1000,
        sparsity_warmup_steps: Optional[int] = None,
        decay_start: Optional[int] = None,
        var_flag: int = 0,
        use_april_update_mode: bool = True,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device = None,
        wandb_name: Optional[str] = 'VSAETopKTrainer',
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

        # Initialize dictionary
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ae = dict_class(
            activation_dim, 
            dict_size, 
            k=k,
            var_flag=var_flag,
            use_april_update_mode=use_april_update_mode,
            device=self.device
        )
        self.ae.to(self.device)

        # Store parameters
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.auxk_alpha = auxk_alpha
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.use_april_update_mode = use_april_update_mode
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.k = k
        
        # Top-K specific tracking
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from TopK paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        
        # Logging parameters
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Optimizer for Adam without constrained weights
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.9, 0.999))

        # Learning rate scheduler with warmup and decay
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, None, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # Sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
    def update_threshold(self, top_acts_BK: t.Tensor):
        """Update the dynamic threshold for activation filtering."""
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )
                
    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        """Calculate auxiliary loss to resurrect dead features."""
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # Normalization from OpenAI implementation
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)
        
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Calculate the hybrid loss combining KL divergence and reconstruction error.
        
        The loss includes:
        1. Reconstruction error (MSE)
        2. KL divergence for variational component
        3. Auxiliary loss for reviving dead features
        """
        sparsity_scale = self.sparsity_warmup_fn(step)

        # Encode with the hybrid model
        if self.var_flag == 1:
            f, log_var, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                x, output_log_var=True, return_topk=True, use_threshold=False
            )
            # KL divergence components from variational sampling
            mu = post_relu_acts_BF
        else:
            f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
                x, return_topk=True, use_threshold=False
            )
            mu = post_relu_acts_BF
            
        # Update threshold if past start step
        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        # Decode and calculate reconstruction error
        x_hat = self.ae.decode(f)
        residual = x - x_hat
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l2_loss = t.linalg.norm(residual, dim=-1).mean()

        # Update the effective L0 (should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each feature
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        # Get decoder norms for KL weighting
        decoder_norms = self.ae.decoder.weight.norm(p=2, dim=0)
        
        # KL divergence loss (using activated features)
        # Modified KL loss calculation for Top-K style features
        kl_base = 0.5 * (mu.pow(2)).sum(dim=-1)
        kl_loss = kl_base.mean() * decoder_norms.mean()
        
        # Auxiliary loss to resurrect dead features
        auxk_loss = self.get_auxiliary_loss(residual, post_relu_acts_BF) if self.auxk_alpha > 0 else 0
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'auxk_loss': auxk_loss.item() if isinstance(auxk_loss, t.Tensor) else auxk_loss,
                    'loss': loss.item()
                }
            )
        
    def update(self, step, activations):
        """Perform a single training update."""
        activations = activations.to(self.device)
        
        # Initialize decoder bias with geometric median on first step
        if step == 0 and not self.use_april_update_mode:
            median = geometric_median(activations)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # Zero gradients
        self.optimizer.zero_grad()
        
        # Calculate loss and backpropagate
        loss = self.loss(activations, step=step)
        loss.backward()
        
        # Apply gradient clipping (April update)
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        
        # Step optimizer and scheduler
        self.optimizer.step()
        self.scheduler.step()
        
        # If using the original approach (not April update), normalize decoder
        if not self.use_april_update_mode:
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
            )

    @property
    def config(self):
        """Return configuration for wandb logging."""
        return {
            'dict_class': 'VSAETopK',
            'trainer_class': 'VSAETopKTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'k': self.k,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'auxk_alpha': self.auxk_alpha,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'threshold_beta': self.threshold_beta,
            'threshold_start_step': self.threshold_start_step,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'var_flag': self.var_flag,
            'use_april_update_mode': self.use_april_update_mode,
        }
