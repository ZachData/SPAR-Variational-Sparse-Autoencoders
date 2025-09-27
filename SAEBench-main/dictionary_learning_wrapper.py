import sys
from pathlib import Path
import torch
import json

# Add the parent directory to path to import dictionary_learning
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.trainers.vsae_topk import VSAETopK
    from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKConfig
except ImportError as e:
    print(f"Error importing dictionary_learning: {e}")
    sys.exit(1)

# Import SAEBench base class
sys.path.append(str(Path(__file__).parent.parent))
from sae_bench.custom_saes.base_sae import BaseSAE

class DictionaryLearningSAEWrapper(BaseSAE):
    """
    Wrapper to make dictionary_learning SAEs compatible with SAEBench.
    Inherits from BaseSAE to get all the required interface methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        
        # Check if required files exist
        # ae_path = self.model_path / "ae.pt"``
        ae_path = self.model_path / "ae_10000.pt"
        config_path = self.model_path / "config.json"
        
        if not ae_path.exists():
            raise FileNotFoundError(f"ae.pt not found at {ae_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        
        # Try loading with load_dictionary first
        try:
            sae, config = load_dictionary(str(model_path), device=device)
        except Exception as e:
            # Fallback: try loading directly
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                trainer_config = saved_config["trainer"]
                
                # Create TopK config
                topk_config = TopKConfig(
                    activation_dim=trainer_config["activation_dim"],
                    dict_size=trainer_config["dict_size"],
                    k=trainer_config["k"],
                    device=torch.device(device)
                )
                
                # Load directly
                sae = AutoEncoderTopK.from_pretrained(
                    str(ae_path),
                    config=topk_config,
                    device=device
                )
                config = saved_config
                
            except Exception as e2:
                raise RuntimeError(f"Both load_dictionary and direct loading failed. load_dictionary: {e}, direct: {e2}")
        
        # Verify SAE loaded properly
        if sae is None:
            raise ValueError("SAE loading returned None")
        
        # Get dimensions from config
        trainer_config = config["trainer"]
        d_in = trainer_config["activation_dim"]
        d_sae = trainer_config["dict_size"]
        
        # Initialize BaseSAE with proper parameters FIRST
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            model_name="pythia-70m-deduped",  # Fixed for your case
            hook_layer=3,
            device=torch.device(device),
            dtype=torch.bfloat16,
            hook_name="blocks.3.hook_resid_post"
        )
        
        # NOW assign the SAE and config after super().__init__()
        self.sae = sae
        self.config = config
        
        # Copy weight matrices from the loaded SAE
        self._copy_weights_from_sae()
        
    def _copy_weights_from_sae(self):
        """Copy weight matrices from the underlying SAE to BaseSAE format."""
        
        with torch.no_grad():
            if hasattr(self.sae, 'encoder') and hasattr(self.sae, 'decoder'):
                # Standard autoencoder structure
                
                # Check if weights are None before cloning
                if self.sae.encoder.weight is None:
                    raise ValueError("Encoder weight is None")
                if self.sae.decoder.weight is None:
                    raise ValueError("Decoder weight is None")
                
                # For Top-K SAE:
                # encoder: [dict_size, activation_dim] -> need [activation_dim, dict_size] for W_enc
                # decoder: [activation_dim, dict_size] -> need [dict_size, activation_dim] for W_dec
                
                self.W_enc.data = self.sae.encoder.weight.T.clone()  # [dict_size, activation_dim] -> [activation_dim, dict_size]
                self.W_dec.data = self.sae.decoder.weight.T.clone()  # [activation_dim, dict_size] -> [dict_size, activation_dim]
                
                # Copy bias terms
                if hasattr(self.sae.encoder, 'bias') and self.sae.encoder.bias is not None:
                    self.b_enc.data = self.sae.encoder.bias.clone()
                else:
                    self.b_enc.data.zero_()
                
                # Handle decoder bias - Top-K SAE uses b_dec parameter
                if hasattr(self.sae, 'b_dec') and self.sae.b_dec is not None:
                    self.b_dec.data = self.sae.b_dec.clone()
                elif hasattr(self.sae.decoder, 'bias') and self.sae.decoder.bias is not None:
                    self.b_dec.data = self.sae.decoder.bias.clone()
                else:
                    self.b_dec.data.zero_()
                    
            elif hasattr(self.sae, 'W_enc') and hasattr(self.sae, 'W_dec'):
                # Already in the right format
                self.W_enc.data = self.sae.W_enc.clone()
                self.W_dec.data = self.sae.W_dec.clone()
                if hasattr(self.sae, 'b_enc'):
                    self.b_enc.data = self.sae.b_enc.clone()
                if hasattr(self.sae, 'b_dec'):
                    self.b_dec.data = self.sae.b_dec.clone()
            else:
                # Print all attributes to debug
                sae_attrs = [attr for attr in dir(self.sae) if not attr.startswith('_')]
                raise ValueError(f"SAE doesn't have expected encoder/decoder or W_enc/W_dec structure. Available attributes: {sae_attrs}")
        
        # Normalize decoder weights to unit norm (SAEBench requirement)
        with torch.no_grad():
            self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=1)
    
    def encode(self, x):
        """Encode activations to features using the underlying SAE."""
        if hasattr(self.sae, 'encode'):
            # For VSAETopK, get just the sparse features (first element of tuple)
            if 'VSAETopK' in str(type(self.sae)):
                result = self.sae.encode(x, training=False)  # Set training=False for evaluation
                sparse_features = result[0]  # First element is sparse_features
                return sparse_features
            # For AutoEncoderTopK, also returns tuple but different structure
            elif hasattr(self.sae, 'k') and hasattr(self.sae, 'encode'):
                result = self.sae.encode(x)
                if isinstance(result, tuple):
                    return result[0]  # First element should be the sparse features
                else:
                    return result
            else:
                # Standard SAE that returns tensor directly
                return self.sae.encode(x)
        else:
            raise NotImplementedError("SAE does not have encode method")
    
    def decode(self, f):
        """Decode features back to activations using the underlying SAE."""
        if hasattr(self.sae, 'decode'):
            return self.sae.decode(f)
        else:
            raise NotImplementedError("SAE does not have decode method")
    
    def forward(self, x, output_features=False):
        """Forward pass through the SAE."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        if output_features:
            return reconstruction, features
        else:
            return reconstruction
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda"):
        """Load a pretrained model."""
        return cls(model_path, device)

# Export the wrapper for SAEBench to use
SAE = DictionaryLearningSAEWrapper