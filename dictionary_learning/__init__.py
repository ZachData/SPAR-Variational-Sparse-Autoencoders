"""
Dictionary Learning implementations for Sparse Autoencoders (SAE) and variants.
"""

# Core Dictionary classes
from .dictionary import Dictionary, AutoEncoder, IdentityDict, GatedAutoEncoder, JumpReluAutoEncoder

# Core functionality
from .buffer import (
    BaseActivationBuffer, 
    ActivationBuffer, 
    TransformerLensActivationBuffer, 
    create_activation_buffer
)
from .evaluation import evaluate, loss_recovered
from .interp import feature_effect, examine_dimension, feature_umap, feature_tsne
from .training import trainSAE

# Utility functions
from .utils import (
    hf_dataset_to_generator, 
    zst_to_generator, 
    get_nested_folders, 
    load_dictionary, 
    get_submodule
)

# Gradient pursuit for inference
from .grad_pursuit import grad_pursuit

__all__ = [
    # Dictionary classes
    "Dictionary",
    "AutoEncoder",
    "IdentityDict",
    "GatedAutoEncoder",
    "JumpReluAutoEncoder",
    
    # Core functionality
    "BaseActivationBuffer",
    "ActivationBuffer", 
    "TransformerLensActivationBuffer",
    "create_activation_buffer",
    "evaluate", 
    "loss_recovered",
    "feature_effect", 
    "examine_dimension", 
    "feature_umap",
    "feature_tsne",
    "trainSAE",
    
    # Utility functions
    "hf_dataset_to_generator",
    "zst_to_generator",
    "get_nested_folders",
    "load_dictionary",
    "get_submodule",
    
    # Inference methods
    "grad_pursuit"
]