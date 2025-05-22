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

# Variational autoencoder trainers
from .vsae_iso import VSAEIsoTrainer
from .vsae_gated import VSAEGated, VSAEGatedTrainer
from .vsae_gated_anneal import VSAEGatedAutoEncoder, VSAEGatedAnnealTrainer
from .vsae_jump_relu import VSAEJumpReLU, VSAEJumpReLUTrainer
from .vsae_matryoshka import MatryoshkaVSAEIso, MatryoshkaVSAEIsoTrainer
from .vsae_panneal import VSAEPAnneal, VSAEPAnnealTrainer
from .vsae_topk import VSAETopK, VSAETopKTrainer
from .vsae_multi import VSAEMultiGaussian, VSAEMultiGaussianTrainer

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
    "VSAEGated",
    "VSAEGatedTrainer",
    "VSAEGatedAutoEncoder",
    "VSAEGatedAnnealTrainer",
    "VSAEJumpReLU",
    "VSAEJumpReLUTrainer",
    "MatryoshkaVSAEIso",
    "MatryoshkaVSAEIsoTrainer",
    "VSAEPAnneal",
    "VSAEPAnnealTrainer",
    "VSAETopK",
    "VSAETopKTrainer",
    "VSAEMultiGaussian",
    "VSAEMultiGaussianTrainer"
]