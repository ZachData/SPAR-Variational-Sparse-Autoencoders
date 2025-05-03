"""
Trainer implementations for Sparse Autoencoders (SAE) and its variants.
"""

# Core trainers
from .trainer import SAETrainer, ConstrainedAdam
from .standard import StandardTrainer, StandardTrainerAprilUpdate

# Specialized trainers
from .batch_top_k import BatchTopKTrainer, BatchTopKSAE
from .top_k import TopKTrainer, AutoEncoderTopK
from .matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE
from .jumprelu import JumpReluTrainer
from .gdm import GatedSAETrainer
from .gated_anneal import GatedAnnealTrainer
from .p_anneal import PAnnealTrainer

# # Variational autoencoder trainers
from .vsae_iso import VSAEIsoTrainer
# from .vsae_mixture import VSAEMixtureTrainer
# from .vsae_multi import VSAEMultiGaussianTrainer

__all__ = [
    # Core
    "SAETrainer", 
    "ConstrainedAdam",
    "StandardTrainer",
    "StandardTrainerAprilUpdate",
    
    # Specialized
    "BatchTopKTrainer",
    "BatchTopKSAE",
    "TopKTrainer",
    "AutoEncoderTopK",
    "MatryoshkaBatchTopKTrainer",
    "MatryoshkaBatchTopKSAE",
    "JumpReluTrainer",
    "GatedSAETrainer",
    "GatedAnnealTrainer",
    "PAnnealTrainer",
    
    # Variational
    "VSAEIsoTrainer",
    "VSAEMixtureTrainer",
    "VSAEMultiGaussianTrainer"
]