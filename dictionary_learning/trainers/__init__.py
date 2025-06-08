"""
Trainer implementations for Sparse Autoencoders (SAE) and its variants.
"""

# Core trainers
from .trainer import SAETrainer, ConstrainedAdam
from .standard import StandardTrainer, StandardTrainerAprilUpdate

# Specialized trainers
from .jumprelu import JumpReluTrainer, JumpReluSAE, JumpReluConfig, JumpReluTrainingConfig
from .batch_top_k import BatchTopKTrainer, BatchTopKSAE
from .top_k import TopKTrainer, AutoEncoderTopK
from .matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE
from .gated_anneal import GatedAnnealTrainer, GatedAnnealConfig, GatedAnnealTrainingConfig
from .p_anneal import PAnnealTrainer
from .gdm import GatedSAETrainer, GDMConfig, GDMTrainingConfig, GatedSAETrainer
from .matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer, 
    MatryoshkaConfig,
    MatryoshkaTrainingConfig,
    apply_temperature
)
from .p_anneal import PAnnealTrainer, PAnnealConfig, PAnnealTrainingConfig

# Variational autoencoder trainers
from .vsae_iso import VSAEIsoTrainer
try:
    from .vsae_gated import VSAEGated, VSAEGatedConfig, VSAEGatedTrainingConfig, VSAEGatedTrainer
except ImportError:
    VSAEGated = VSAEGatedConfig = VSAEGatedTrainingConfig = VSAEGatedTrainer = None
from .vsae_gated_anneal import VSAEGatedAnneal, VSAEGatedAnnealTrainer
from .vsae_jump_relu import VSAEJumpReLU, VSAEJumpReLUTrainer

from .vsae_panneal import VSAEPAnnealTrainer, VSAEPAnneal, VSAEPAnnealConfig, VSAEPAnnealTrainingConfig
from .vsae_topk import VSAETopK, VSAETopKTrainer
from .vsae_multi import VSAEMultiGaussian, VSAEMultiGaussianTrainer, VSAEMultiConfig, VSAEMultiTrainingConfig

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
    "MatryoshkaBatchTopKSAE",
    "MatryoshkaBatchTopKTrainer", 
    "MatryoshkaConfig",
    "MatryoshkaTrainingConfig",
    "apply_temperature",
    "JumpReluTrainer",
    "GatedSAETrainer",
    "GatedAnnealTrainer",
    "PAnnealTrainer",
    "GatedAnnealConfig",
    "GatedAnnealTrainingConfig",
    "JumpReluSAE",
    "JumpReluConfig",
    "JumpReluTrainingConfig",
    "GDMConfig",
    "GDMTrainingConfig",
    "PAnnealConfig", 
    "PAnnealTrainingConfig",
    
    # Variational
    "VSAEIsoTrainer",
    'VSAEGated',
    'VSAEGatedConfig', 
    'VSAEGatedTrainingConfig',
    'VSAEGatedTrainer',
    "VSAEJumpReLU",
    "VSAEJumpReLUTrainer",
    "MatryoshkaVSAEIso",
    "MatryoshkaVSAEIsoTrainer",
    "VSAEPAnnealTrainer",
    "VSAEPAnneal", 
    "VSAEPAnnealConfig",
    "VSAEPAnnealTrainingConfig",
    "VSAETopK",
    "VSAETopKTrainer",
    "VSAEMultiGaussian",
    "VSAEMultiGaussianTrainer",
    'VSAEGatedAnneal', 
    'VSAEGatedAnnealTrainer',
    "VSAEMultiConfig",
    "VSAEMultiTrainingConfig",
]