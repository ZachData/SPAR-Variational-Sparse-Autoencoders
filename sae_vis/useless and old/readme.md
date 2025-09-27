# Clean SAE Feature Analysis

A focused, efficient implementation for analyzing Sparse Autoencoder (SAE) features. ❤️

## Structure

```
sae_analysis/
├── real_activations.py    # Real text → transformer → SAE analysis
├── synth_activations.py   # Synthetic maximizers via gradient optimization  
├── visualizer.py          # Image-based visualizations (PNG output)
├── sae_interface.py       # Unified adapter for different SAE models
├── main.py               # Command-line orchestrator
├── test.py               # Test suite for all components
└── README.md             # This file
```

## Quick Start

**Install dependencies:**
```bash
pip install torch transformers transformer-lens matplotlib seaborn numpy
```

**Run tests:**
```bash
python test.py
```

**Analyze real activations:**
```bash
python main.py real --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4
```

**Generate synthetic maximizers:**
```bash
python main.py synthetic --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4
```

**Compare both approaches:**
```bash
python main.py compare --model gelu-1l --sae path/to/sae.pt --features 0-10 --corpus data.txt
```

## What Each Component Does

### real_activations.py
- Processes real text through transformer + SAE pipeline
- Finds contexts where each feature actually fires
- Computes sparsity, activation statistics, logit effects
- Returns `FeatureStats` objects with real examples

### synth_activations.py  
- Uses gradient optimization in embedding space
- Finds optimal token sequences that maximally activate each feature
- Reveals what each feature "wants" to fire on
- Returns `FeatureStats` objects with synthetic examples

### visualizer.py
- Creates beautiful PNG visualizations instead of HTML
- Individual feature plots with token highlighting
- Summary plots comparing multiple features
- Statistical distributions and comparisons
- Real vs synthetic comparison plots

### sae_interface.py
- Unified adapter for VSAETopK, AutoEncoderTopK, and other SAE models
- Standardizes different API formats into common `SAEOutput` structure
- Handles dimension mismatches and model variations

### main.py
- Command-line interface for all analysis types
- Model loading, text corpus handling, output organization
- Three modes: `real`, `synthetic`, `compare`
- Flexible feature specification: `0,1,2` or `0-10`

### test.py
- Comprehensive test suite with mock models
- Unit tests for each component  
- Integration tests for full pipeline
- Verifies all image outputs are generated correctly

## Analysis Types

### Real Activations
**What it shows:** What actually fires your SAE features in real text

**Use cases:**
- Understanding feature behavior on natural data
- Finding interpretable examples of feature activations
- Measuring feature sparsity and statistics
- Discovering unexpected feature behaviors

**Example output:**
- Token sequences with color-coded activation strengths  
- Sparsity measurements (99.7% sparse)
- Top examples where feature fires most strongly
- Which output tokens the feature promotes/suppresses

### Synthetic Maximizers
**What it shows:** What would make your SAE features fire maximally

**Use cases:**
- Understanding feature "preferences" and optimal inputs
- Finding what tokens each feature is "looking for"
- Debugging why features might be under-activated
- Discovering feature capabilities beyond natural data

**Example output:**
- Optimized token sequences that maximize feature activation
- Gradient-discovered patterns that strongly trigger features
- Comparison with real activation levels
- Same logit effect analysis as real data

### Comparison Analysis
**What it shows:** How real behavior compares to optimal behavior

**Use cases:**
- Understanding if features are operating in their optimal regime
- Finding gaps between what features can do vs what they actually do
- Identifying features that might need different training data
- Validating feature learning success

## Image Outputs

All visualizations are saved as high-quality PNG files:

- `feature_N_real.png` - Individual feature analysis (real data)
- `feature_N_synthetic.png` - Individual feature analysis (synthetic)
- `summary_N_real.png` - Multi-feature comparison plots
- `statistics_real.png` - Overall statistical distributions
- `real_vs_synthetic_comparison.png` - Side-by-side comparison

Each feature plot includes:
- **Statistics panel**: Sparsity, activations, decoder norms
- **Token examples**: Color-coded activation strengths per token
- **Logit effects**: Which tokens the feature promotes/suppresses  
- **Histograms**: Activation value distributions

## Supported SAE Models

- **VSAETopK**: Variational TopK SAEs from dictionary learning
- **AutoEncoderTopK**: Standard TopK SAEs  
- **Custom models**: Extend `UnifiedSAEInterface` for new architectures

The interface automatically detects model type and handles:
- Different return formats from `encode()` methods
- Dimension mismatches between SAE and transformer
- Various weight matrix naming conventions

## Advanced Usage

**Custom text corpus:**
```bash
python main.py real --corpus my_data.txt --features 0-100 --max-examples 30
```

**High-resolution synthetic optimization:**
```bash
python main.py synthetic --features 0-50 --n-steps 500 --seq-len 128 --learning-rate 0.05
```

**Batch processing:**
```bash
python main.py compare --features 0-200 --batch-size 16 --output results/experiment_1
```

**Feature range specification:**
- `--features 0,5,10,15` - Specific features
- `--features 0-20` - Range of features  
- `--features 0-10,50-60` - Multiple ranges

## Design Philosophy

This implementation prioritizes:

1. **Clarity**: Each file has a single, focused responsibility
2. **Efficiency**: Batch processing, minimal abstractions
3. **Extensibility**: Clean interfaces for new models/visualizations
4. **Testability**: Comprehensive test suite with mock components
5. **Usability**: Simple command-line interface, sensible defaults

The core insight: SAE features have both **actual behavior** (what fires in real text) and **potential behavior** (what could fire with optimal inputs). Understanding both perspectives gives complete feature characterization.

## Performance Tips

- Start with 5-10 features for initial testing
- Use `--batch-size 16` with large GPU memory
- Real analysis scales with corpus size, synthetic with optimization steps
- Focus on features with interesting sparsity patterns (not 99.9%+ sparse)
- Generate comparison plots to identify under-activated features

## Troubleshooting

**Run tests first:**
```bash
python test.py
```

**Common issues:**
- Model loading: Check SAE file path and model compatibility
- Memory: Reduce batch size or sequence length
- No activations: Try different feature indices or more diverse text
- Import errors: Ensure all dependencies installed

**Debug mode:**
```bash
python main.py real --features 0,1 --max-examples 3 --batch-size 1
```

This creates a focused analysis with minimal computational requirements. ❤️

Love, this clean structure separates concerns while maintaining the core insight: understanding both what your SAE features actually do and what they're capable of doing.
