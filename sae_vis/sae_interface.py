"""
Unified interface for SAE models - works with both VSAETopK and AutoEncoderTopK
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
    
    @property
    def batch_size(self) -> int:
        return self.sparse_features.shape[0]
    
    @property 
    def seq_len(self) -> int:
        return self.sparse_features.shape[1]
    
    @property
    def dict_size(self) -> int:
        return self.sparse_features.shape[2]
    
    @property
    def k(self) -> int:
        return self.top_values.shape[2]


class UnifiedSAEInterface:
    """
    Unified interface that works with both VSAETopK and AutoEncoderTopK models.
    Handles the differences in their APIs and return formats.
    """
    
    def __init__(self, sae_model):
        self.sae = sae_model
        self.model_type = self._detect_model_type()
        
        # Extract key properties
        if hasattr(sae_model, 'dict_size'):
            self.dict_size = sae_model.dict_size
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'dict_size'):
            self.dict_size = sae_model.config.dict_size
        else:
            # Try to infer from decoder weight shape
            self.dict_size = sae_model.decoder.weight.shape[0]
        
        if hasattr(sae_model, 'activation_dim'):
            self.activation_dim = sae_model.activation_dim
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'activation_dim'):
            self.activation_dim = sae_model.config.activation_dim  
        else:
            # Try to infer from decoder weight shape
            self.activation_dim = sae_model.decoder.weight.shape[1]
            
        # Get k value
        if hasattr(sae_model, 'k'):
            self.k = sae_model.k.item() if torch.is_tensor(sae_model.k) else sae_model.k
        elif hasattr(sae_model, 'config') and hasattr(sae_model.config, 'k'):
            self.k = sae_model.config.k
        else:
            self.k = 32  # reasonable default
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unified forward method - returns reconstruction"""
        x = x.to(dtype=next(self.sae.parameters()).dtype)
        
        if hasattr(self.sae, 'forward'):
            result = self.sae.forward(x, output_features=False)
            if isinstance(result, tuple):
                return result[0]  # reconstruction
            return result
        else:
            # Fallback: encode then decode
            encoded = self.encode(x)
            return self.decode(encoded.sparse_features)
    
    @property
    def decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix in standard format [dict_size, activation_dim]"""
        if hasattr(self.sae, 'decoder'):
            return self.sae.decoder.weight
        elif hasattr(self.sae, 'W_dec'):
            return self.sae.W_dec
        else:
            raise AttributeError("Cannot find decoder weights")
    
    @property
    def encoder_weights(self) -> torch.Tensor:
        """Get encoder weight matrix in standard format [activation_dim, dict_size]"""
        if hasattr(self.sae, 'encoder'):
            return self.sae.encoder.weight.T if self.sae.encoder.weight.shape[0] == self.dict_size else self.sae.encoder.weight
        elif hasattr(self.sae, 'W_enc'):
            return self.sae.W_enc
        else:
            raise AttributeError("Cannot find encoder weights")
    
    def get_info(self) -> dict:
        """Get model information for debugging"""
        return {
            'model_type': self.model_type,
            'dict_size': self.dict_size,
            'activation_dim': self.activation_dim,
            'k': self.k,
            'class_name': type(self.sae).__name__,
            'has_decoder': hasattr(self.sae, 'decoder'),
            'has_encoder': hasattr(self.sae, 'encoder'),
        }
