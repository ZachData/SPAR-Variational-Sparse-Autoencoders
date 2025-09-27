"""
Unified interface for SAE models - works with both VSAETopK and AutoEncoderTopK
FIXED: Correct decoder weight matrix orientation and dict_size detection
"""
import torch
from typing import Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class SAEOutput:
    """Clean output structure for SAE encode operations"""
    sparse_features: torch.Tensor      # [batch, seq, dict_size] - sparse activations
    top_values: torch.Tensor          # [batch, seq, k] - top-k activation values  
    top_indices: torch.Tensor         # [batch, seq, k] - indices of top-k features
    pre_activation: torch.Tensor      # [batch, seq, dict_size] - pre-sparsity activations


class UnifiedSAEInterface:
    """
    Unified interface that works with both VSAETopK and AutoEncoderTopK models.
    Handles the differences in their APIs and return formats.
    """
    
    def __init__(self, sae_model):
        self.sae = sae_model
        self.model_type = self._detect_model_type()
        
        # Extract key properties with proper orientation handling
        if hasattr(sae_model, 'dict_size'):
            self.dict_size = sae_model.dict_size
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'dict_size'):
            self.dict_size = sae_model.config.dict_size
        else:
            # Try to infer from decoder weight shape
            # PyTorch Linear decoder: nn.Linear(dict_size, activation_dim)
            # So decoder.weight.shape = [activation_dim, dict_size]
            if hasattr(sae_model, 'decoder'):
                decoder_weight = sae_model.decoder.weight
                if decoder_weight.shape[0] < decoder_weight.shape[1]:
                    # [activation_dim, dict_size] format - dict_size is larger
                    self.dict_size = decoder_weight.shape[1]
                else:
                    # [dict_size, activation_dim] format - unusual but possible
                    self.dict_size = decoder_weight.shape[0]
            else:
                self.dict_size = 1000  # fallback
        
        if hasattr(sae_model, 'activation_dim'):
            self.activation_dim = sae_model.activation_dim
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'activation_dim'):
            self.activation_dim = sae_model.config.activation_dim  
        else:
            # Try to infer from decoder weight shape
            if hasattr(sae_model, 'decoder'):
                decoder_weight = sae_model.decoder.weight
                if decoder_weight.shape[0] < decoder_weight.shape[1]:
                    # [activation_dim, dict_size] format
                    self.activation_dim = decoder_weight.shape[0]
                else:
                    # [dict_size, activation_dim] format
                    self.activation_dim = decoder_weight.shape[1]
            else:
                self.activation_dim = 512  # fallback
            
        # Get k value
        if hasattr(sae_model, 'k'):
            self.k = sae_model.k.item() if torch.is_tensor(sae_model.k) else sae_model.k
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'k'):
            self.k = sae_model.config.k
        else:
            self.k = 32  # reasonable default
        
        print(f"SAE Interface initialized: dict_size={self.dict_size}, activation_dim={self.activation_dim}, k={self.k}")
    
    def _detect_model_type(self) -> str:
        """Detect whether this is VSAETopK, AutoEncoderTopK, or other"""
        class_name = type(self.sae).__name__
        if 'VSAE' in class_name:
            return 'vsae_topk'
        elif 'AutoEncoder' in class_name or 'TopK' in class_name:
            return 'autoencoder_topk'
        else:
            return 'unknown'
    
    def encode(self, x: torch.Tensor) -> SAEOutput:
        """
        Unified encode method that works with both model types.
        Always returns topk information in a standard format.
        """
        x = x.to(dtype=next(self.sae.parameters()).dtype)
        
        if self.model_type == 'vsae_topk':
            # VSAETopK returns: sparse_features, z, mu, log_var, top_indices, selected_vals
            result = self.sae.encode(x, return_topk=True, training=False)
            sparse_features, _, _, _, top_indices, selected_vals = result
            
            # For VSAETopK, we need to reconstruct the full pre-activation tensor
            # This is an approximation since we only have the top-k values
            pre_activation = torch.zeros_like(sparse_features)
            pre_activation.scatter_(dim=-1, index=top_indices, src=selected_vals.abs())
            
        elif self.model_type == 'autoencoder_topk':
            # AutoEncoderTopK returns: sparse_features, top_values, top_indices, pre_activation
            sparse_features, selected_vals, top_indices, pre_activation = self.sae.encode(x, return_topk=True)
            
        else:
            # Unknown model type - try to call encode and see what we get
            try:
                result = self.sae.encode(x, return_topk=True)
                if len(result) == 4:
                    # Assume AutoEncoderTopK format
                    sparse_features, selected_vals, top_indices, pre_activation = result
                elif len(result) == 6:
                    # Assume VSAETopK format  
                    sparse_features, _, _, _, top_indices, selected_vals = result
                    pre_activation = torch.zeros_like(sparse_features)
                    pre_activation.scatter_(dim=-1, index=top_indices, src=selected_vals.abs())
                else:
                    raise ValueError(f"Unexpected encode output format: {len(result)} elements")
            except:
                # Fallback - just call encode normally
                sparse_features = self.sae.encode(x)
                # Create dummy topk info
                top_values, top_indices = sparse_features.topk(min(self.k, sparse_features.shape[-1]), dim=-1)
                selected_vals = top_values
                pre_activation = sparse_features
        
        return SAEOutput(
            sparse_features=sparse_features,
            top_values=selected_vals,
            top_indices=top_indices, 
            pre_activation=pre_activation
        )
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Unified decode method"""
        features = features.to(dtype=next(self.sae.parameters()).dtype)
        return self.sae.decode(features)
    
    @property
    def decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix in standard format [dict_size, activation_dim]"""
        if hasattr(self.sae, 'decoder'):
            weight = self.sae.decoder.weight
            # PyTorch Linear layers store weights as [out_features, in_features]
            # For decoder: nn.Linear(dict_size, activation_dim) -> weight.shape = [activation_dim, dict_size]
            # We want [dict_size, activation_dim], so transpose if needed
            if weight.shape[0] == self.activation_dim and weight.shape[1] == self.dict_size:
                return weight.T  # Transpose to [dict_size, activation_dim]
            else:
                return weight  # Already in correct format
        elif hasattr(self.sae, 'W_dec'):
            return self.sae.W_dec
        else:
            raise AttributeError("Cannot find decoder weights")
    
    @property
    def encoder_weights(self) -> torch.Tensor:
        """Get encoder weight matrix in standard format [activation_dim, dict_size]"""
        if hasattr(self.sae, 'encoder'):
            weight = self.sae.encoder.weight
            # For encoder: nn.Linear(activation_dim, dict_size) -> weight.shape = [dict_size, activation_dim]
            # We want [activation_dim, dict_size], so transpose if needed
            if weight.shape[0] == self.dict_size and weight.shape[1] == self.activation_dim:
                return weight.T  # Transpose to [activation_dim, dict_size]
            else:
                return weight  # Already in correct format
        elif hasattr(self.sae, 'W_enc'):
            return self.sae.W_enc
        else:
            raise AttributeError("Cannot find encoder weights")