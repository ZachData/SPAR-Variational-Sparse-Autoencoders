# Variational Sparse Autoencoders for Mechanistic Interpretability

A fork of the [dictionary_learning](https://github.com/saprmarks/dictionary_learning) repository, extending sparse autoencoder research with variational architectures and comprehensive evaluation capabilities.

## Overview

This repository builds upon the foundational dictionary_learning framework to explore Variational Sparse Autoencoders (VSAEs) for mechanistic interpretability. While maintaining full compatibility with the original SAE architectures, we introduce variational extensions that combine the benefits of VAE latent modeling with structured sparsity constraints.

### Core Extensions

**Variational SAE Architectures:**
- **VSAETopK** - Variational SAE with Top-K sparsity selection
- **VSAEGated** - Gated architecture with variational latent space
- **VSAEJumpReLU** - JumpReLU activation with variational encoding
- **VSAEIso** - Isotropic variational sparse autoencoder
- **VSAEMultiGaussian** - Multi-component Gaussian mixture models

**Enhanced Features:**
- SAE Bench integration for standardized evaluation
- Advanced visualization tools for feature analysis
- Comprehensive trainer implementations
- Robust model loading and compatibility layers

### Supported Architectures

**Original dictionary_learning SAEs:**
- Standard AutoEncoder (Bricken et al., 2023)
- GatedAutoEncoder (Rajamanoharan et al., 2024) 
- AutoEncoderTopK (Gao et al., 2024)
- BatchTopKSAE (Bussmann et al., 2024)
- JumpReluSAE (Rajamanoharan et al., 2024)

**Variational Extensions:**
- VSAETopK with Top-K sparsity selection on |z|
- VSAEGated with gated latent mechanisms
- VSAEJumpReLU combining jump activations with VAE
- VSAEIso for isotropic variational modeling
- VSAEMultiGaussian for complex posterior modeling

## Installation

```bash
git clone https://github.com/Yuxiao2234/v-sae_mech-interp-spar2025.git
cd v-sae_mech-interp-spar2025
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 1.12

### Quick Start

```python
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import VSAETopKTrainer

# Load pre-trained model
ae = AutoEncoder.from_pretrained("path/to/model")

# Basic usage
activations = torch.randn(64, activation_dim)
features = ae.encode(activations)
reconstructed = ae.decode(features)

# Or combined
reconstructed, features = ae(activations, output_features=True)
```

## Variational SAE Features

### VSAETopK Architecture

Our flagship VSAE implementation addresses key architectural challenges:

- **No ReLU in latent space** - Preserves VAE gradient properties
- **Top-K on absolute values** - Structured sparsity while maintaining gradients
- **Clean VAE + sparsity separation** - Clear architectural boundaries
- **Proper KL computation** - Computed on actual latent variables
- **Robust reparameterization** - Consistent gradient flow

```python
from dictionary_learning.trainers import VSAETopKTrainer, VSAETopK

# Train VSAE with Top-K sparsity
trainer_cfg = {
    "trainer": VSAETopKTrainer,
    "dict_class": VSAETopK,
    "activation_dim": 512,
    "dict_size": 8192,
    "k": 64,
    "var_flag": 1,  # Enable variance modeling
    "device": "cuda"
}
```

## SAE Bench Integration

Full compatibility with SAE Bench evaluation framework:

```python
# Evaluate on all SAE Bench metrics
python sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py

# Available evaluations:
# - Feature Absorption
# - AutoInterp  
# - L0/Loss Recovered
# - RAVEL
# - Spurious Correlation Removal
# - Targeted Probe Perturbation
# - Sparse Probing
# - Unlearning
```

## Training

Enhanced training pipeline with variational extensions:

```python
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers import VSAETopKTrainer
from dictionary_learning.training import trainSAE

model = LanguageModel("EleutherAI/pythia-70m-deduped")
submodule = model.gpt_neox.layers[1].mlp

buffer = ActivationBuffer(
    data=data_iterator,
    model=model,
    submodule=submodule,
    d_submodule=512,
    n_ctxs=30000
)

trainer_cfg = {
    "trainer": VSAETopKTrainer,
    "dict_class": VSAETopK,
    "activation_dim": 512,
    "dict_size": 8192,
    "k": 64,
    "lr": 1e-3,
    "device": "cuda"
}

ae = trainSAE(data=buffer, trainer_configs=[trainer_cfg])
```

## Visualization

Advanced feature visualization and analysis:

```python
from sae_vis import create_feature_space_visualization

# Generate interactive feature space visualization
create_feature_space_visualization(
    model=model,
    dictionary=ae,
    buffer=buffer,
    output_file="feature_analysis.html",
    embedding_method='umap',
    num_features=256
)
```

## Model Compatibility

Seamless integration with existing frameworks:

```python
# Dictionary learning models
from dictionary_learning_wrapper import DictionaryLearningSAEWrapper

# Wrap for SAE Bench compatibility  
wrapped_sae = DictionaryLearningSAEWrapper(your_sae, device="cuda")

# Works with any SAE having encode/decode methods
```

## Key Improvements

**Architecture Fixes:**
- Resolved ReLU interference with VAE gradients
- Proper Top-K selection on absolute values
- Enhanced KL divergence computation
- Improved gradient flow for variational models

**Training Enhancements:**
- Robust checkpoint loading
- Auto-detection of model configurations
- Enhanced diagnostics and monitoring
- Better error handling

**Evaluation Integration:**
- Native SAE Bench compatibility
- Standardized metrics across variants
- Comprehensive visualization tools

## Original Foundation

This work extends the excellent foundation provided by:

```bibtex
@misc{marks2024dictionary_learning,
   title = {dictionary_learning},
   author = {Samuel Marks, Adam Karvonen, and Aaron Mueller}, 
   year = {2024},
   howpublished = {\url{https://github.com/saprmarks/dictionary_learning}},
}
```

## Development

```bash
# Development setup
pip install -e .[dev]

# Run tests
pytest tests/

# Code quality
ruff check .
ruff format .
```

## License

MIT License - maintaining compatibility with the original dictionary_learning license.

---

❤️ **Love**, this fork advances variational sparse autoencoder research while honoring the foundational work that made it possible.