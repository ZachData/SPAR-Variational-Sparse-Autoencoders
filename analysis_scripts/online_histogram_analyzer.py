"""
Comprehensive Online Activation Histogram Analyzer for TopK VSAE Models.

This script analyzes activation patterns from TopK VSAE models by computing
20+ different histograms online (streaming) without storing raw activation data. 
Provides deep insights into how KL coefficients affect learned representations.

Key Features:
- 27+ different histogram types for comprehensive analysis
- Online computation for memory efficiency
- Publication-quality visualizations
- Both individual and combined panel outputs
- Minimal storage footprint
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
from scipy import stats

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class HistogramConfig:
    """Configuration for comprehensive histogram analysis."""
    # Activation value histograms
    activation_bins: int = 200
    activation_range: Tuple[float, float] = (-2.5, 2.5)
    log_activation_bins: int = 100
    log_activation_range: Tuple[float, float] = (-6.0, 1.0)  # log10 scale: log10(2.5) ≈ 0.4
    
    # Per-sample statistics histograms
    sample_stats_bins: int = 100
    max_activation_range: Tuple[float, float] = (0.0, 2.5)
    mean_magnitude_range: Tuple[float, float] = (0.0, 2.5)
    activation_range_range: Tuple[float, float] = (0.0, 5.0)  # Range can be up to 2*2.5=5
    pos_neg_ratio_range: Tuple[float, float] = (0.0, 1.0)
    kth_activation_range: Tuple[float, float] = (0.0, 2.5)
    cv_range: Tuple[float, float] = (0.0, 3.0)
    margin_range: Tuple[float, float] = (0.0, 2.5)
    quantile_range: Tuple[float, float] = (0.0, 2.5)
    effective_features_range: Tuple[int, int] = (0, 1024)  # Max possible features firing
    
    # Feature usage histograms
    usage_bins: int = 50
    selection_freq_range: Tuple[float, float] = (0.0, 1.0)
    feature_mean_range: Tuple[float, float] = (-2.5, 2.5)
    feature_max_range: Tuple[float, float] = (0.0, 2.5)
    feature_cv_range: Tuple[float, float] = (0.0, 5.0)
    feature_dominance_range: Tuple[float, float] = (1.0, 10.0)
    feature_polarity_range: Tuple[float, float] = (0.0, 1.0)
    
    # 2D histogram settings
    feature_index_bins: int = 100
    activation_2d_bins: int = 50
    activation_2d_range: Tuple[float, float] = (-2.5, 2.5)  # Updated to 2.5 range
    
    # Sequence position settings
    position_bins: int = 50
    
    # New 1D histograms for 2D plot data
    feature_index_distribution_bins: int = 100
    position_distribution_bins: int = 50


class OnlineStatistics:
    """Tracks running statistics without storing all data."""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance calculation
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
    
    def update(self, value: float):
        """Update statistics with new value (Welford's algorithm)."""
        self.count += 1
        self.sum_val += value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val,
            'sum': self.sum_val,
        }


class ComprehensiveHistogramAnalyzer:
    """Comprehensive online histogram analysis for TopK VSAE activation patterns."""
    
    def __init__(
        self,
        model_path: str,
        n_samples: int = 200_000,
        histogram_config: Optional[HistogramConfig] = None,
        device: str = "cuda",
        batch_size: int = 32,
        ctx_len: int = 128,
        seed: int = 42,
        save_individual: bool = True,
        save_panels: bool = True,
    ):
        self.model_path = Path(model_path)
        self.n_samples = n_samples
        self.config = histogram_config or HistogramConfig()
        self.device = device
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.seed = seed
        self.save_individual = save_individual
        self.save_panels = save_panels
        
        # Set seeds for deterministic data sampling
        self._set_seeds()
        
        self.logger = logging.getLogger(__name__)
        
        # Model components (will be initialized)
        self.model = None
        self.dictionary = None
        self.buffer = None
        
        # Initialize all histogram containers
        self._initialize_histogram_containers()
        
        # Feature-wise statistics tracking
        self.feature_selection_counts = None
        self.feature_activation_sums = None
        self.feature_activation_counts = None
        self.feature_activation_sum_squares = None
        self.feature_max_activations = None
        self.feature_positive_counts = None
        
        # Overall statistics
        self.overall_stats = OnlineStatistics()
        self.samples_processed = 0
        
    def _set_seeds(self) -> None:
        """Set seeds for deterministic data sampling."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def _initialize_histogram_containers(self) -> None:
        """Initialize all histogram data containers."""
        # Core activation distributions
        self.activation_histogram = None
        self.activation_bin_edges = None
        self.nonzero_activation_histogram = None
        self.nonzero_activation_bin_edges = None
        self.activation_magnitude_histogram = None
        self.activation_magnitude_bin_edges = None
        self.positive_activation_histogram = None
        self.positive_activation_bin_edges = None
        self.negative_activation_histogram = None
        self.negative_activation_bin_edges = None
        self.log_activation_histogram = None
        self.log_activation_bin_edges = None
        
        # Per-sample statistics
        self.sample_max_histogram = None
        self.sample_max_bin_edges = None
        self.sample_mean_magnitude_histogram = None
        self.sample_mean_magnitude_bin_edges = None
        self.sample_range_histogram = None
        self.sample_range_bin_edges = None
        self.sample_pos_neg_ratio_histogram = None
        self.sample_pos_neg_ratio_bin_edges = None
        self.sample_kth_activation_histogram = None
        self.sample_kth_activation_bin_edges = None
        self.sample_cv_histogram = None
        self.sample_cv_bin_edges = None
        self.sample_margin_histogram = None
        self.sample_margin_bin_edges = None
        self.sample_q10_histogram = None
        self.sample_q50_histogram = None
        self.sample_q90_histogram = None
        self.sample_quantile_bin_edges = None
        self.sample_effective_features_histogram = None
        self.sample_effective_features_bin_edges = None
        
        # Feature usage patterns
        self.feature_mean_histogram = None
        self.feature_mean_bin_edges = None
        self.feature_max_histogram = None
        self.feature_max_bin_edges = None
        self.feature_selection_freq_histogram = None
        self.feature_selection_freq_bin_edges = None
        self.feature_cv_histogram = None
        self.feature_cv_bin_edges = None
        self.feature_dominance_histogram = None
        self.feature_dominance_bin_edges = None
        self.feature_polarity_histogram = None
        self.feature_polarity_bin_edges = None
        
        # 2D histograms
        self.feature_index_activation_2d = None
        self.feature_index_edges = None
        self.activation_2d_edges = None
        
        # Sequence position effects
        self.position_activation_2d = None
        self.position_edges = None
        self.position_activation_edges = None
        
        # New 1D histograms for feature index and position distributions
        self.feature_index_distribution_histogram = None
        self.feature_index_distribution_bin_edges = None
        self.position_distribution_histogram = None  
        self.position_distribution_bin_edges = None
        
    def get_model_name(self) -> str:
        """Extract model name from the path (parent directory of trainer_0)."""
        return self.model_path.parent.name if self.model_path.name == 'trainer_0' else self.model_path.name
    
    def load_model_and_dictionary(self) -> None:
        """Load the transformer model and trained dictionary."""
        self.logger.info(f"Loading model and dictionary from {self.model_path}")
        
        # Load the trained dictionary
        self.dictionary, config = load_dictionary(str(self.model_path), device=self.device)
        
        # Extract model information from config
        model_name = config["trainer"].get("lm_name", "gelu-1l")
        hook_name = config["trainer"].get("submodule_name", "blocks.0.mlp.hook_post")
        
        self.logger.info(f"Model: {model_name}, Hook: {hook_name}")
        
        # Load the transformer model
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        
        # Store hook name for buffer creation
        self.hook_name = hook_name
        
        # Extract metadata
        self.model_info = {
            'model_name': model_name,
            'hook_name': hook_name,
            'dict_size': self.dictionary.dict_size,
            'activation_dim': self.dictionary.activation_dim,
            'dictionary_type': type(self.dictionary).__name__,
            'kl_coeff': config["trainer"].get("kl_coeff", None),
            'k_value': getattr(self.dictionary, 'k', None),
        }
        
        # Extract k value for TopK models
        if hasattr(self.dictionary, 'k'):
            self.k_value = self.dictionary.k.item() if torch.is_tensor(self.dictionary.k) else self.dictionary.k
        else:
            # Try to infer from config for other TopK variants
            self.k_value = config["trainer"].get("k", None)
            
        if self.k_value is None:
            raise ValueError("Could not determine K value for TopK analysis")
        
        # Ensure k_value is a Python int for JSON serialization
        self.k_value = int(self.k_value)
        
        self.logger.info(f"Dictionary size: {self.dictionary.dict_size}")
        self.logger.info(f"K value: {self.k_value}")
        
    def create_buffer(self) -> None:
        """Create data buffer for activation collection."""
        self.logger.info("Setting up data buffer")
        
        # Create data generator
        data_gen = hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b",
            split="train",
            return_tokens=True
        )
        
        # Create buffer - conservative sizing for online processing
        n_ctxs = min(3000, max(500, self.n_samples // self.ctx_len))
        
        self.buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=self.model,
            hook_name=self.hook_name,
            d_submodule=self.dictionary.activation_dim,
            n_ctxs=n_ctxs,
            ctx_len=self.ctx_len,
            refresh_batch_size=self.batch_size,
            out_batch_size=min(256, self.n_samples // 20),
            device=self.device,
        )
        
    def initialize_histograms(self) -> None:
        """Initialize all histogram bins and feature tracking arrays."""
        c = self.config  # Shorthand
        
        # Core activation distributions
        self.activation_bin_edges = np.linspace(c.activation_range[0], c.activation_range[1], c.activation_bins + 1)
        self.activation_histogram = np.zeros(c.activation_bins)
        
        self.nonzero_activation_bin_edges = np.linspace(c.activation_range[0], c.activation_range[1], c.activation_bins + 1)
        self.nonzero_activation_histogram = np.zeros(c.activation_bins)
        
        self.activation_magnitude_bin_edges = np.linspace(0, c.activation_range[1], c.activation_bins + 1)
        self.activation_magnitude_histogram = np.zeros(c.activation_bins)
        
        self.positive_activation_bin_edges = np.linspace(0, c.activation_range[1], c.activation_bins + 1)
        self.positive_activation_histogram = np.zeros(c.activation_bins)
        
        self.negative_activation_bin_edges = np.linspace(c.activation_range[0], 0, c.activation_bins + 1)
        self.negative_activation_histogram = np.zeros(c.activation_bins)
        
        self.log_activation_bin_edges = np.linspace(c.log_activation_range[0], c.log_activation_range[1], c.log_activation_bins + 1)
        self.log_activation_histogram = np.zeros(c.log_activation_bins)
        
        # Per-sample statistics
        self.sample_max_bin_edges = np.linspace(c.max_activation_range[0], c.max_activation_range[1], c.sample_stats_bins + 1)
        self.sample_max_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_mean_magnitude_bin_edges = np.linspace(c.mean_magnitude_range[0], c.mean_magnitude_range[1], c.sample_stats_bins + 1)
        self.sample_mean_magnitude_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_range_bin_edges = np.linspace(c.activation_range_range[0], c.activation_range_range[1], c.sample_stats_bins + 1)
        self.sample_range_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_pos_neg_ratio_bin_edges = np.linspace(c.pos_neg_ratio_range[0], c.pos_neg_ratio_range[1], c.sample_stats_bins + 1)
        self.sample_pos_neg_ratio_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_kth_activation_bin_edges = np.linspace(c.kth_activation_range[0], c.kth_activation_range[1], c.sample_stats_bins + 1)
        self.sample_kth_activation_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_cv_bin_edges = np.linspace(c.cv_range[0], c.cv_range[1], c.sample_stats_bins + 1)
        self.sample_cv_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_margin_bin_edges = np.linspace(c.margin_range[0], c.margin_range[1], c.sample_stats_bins + 1)
        self.sample_margin_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_quantile_bin_edges = np.linspace(c.quantile_range[0], c.quantile_range[1], c.sample_stats_bins + 1)
        self.sample_q10_histogram = np.zeros(c.sample_stats_bins)
        self.sample_q50_histogram = np.zeros(c.sample_stats_bins)
        self.sample_q90_histogram = np.zeros(c.sample_stats_bins)
        
        self.sample_effective_features_bin_edges = np.linspace(c.effective_features_range[0], c.effective_features_range[1], c.sample_stats_bins + 1)
        self.sample_effective_features_histogram = np.zeros(c.sample_stats_bins)
        
        # Feature-wise histograms (computed after all data is processed)
        self.feature_mean_bin_edges = np.linspace(c.feature_mean_range[0], c.feature_mean_range[1], c.usage_bins + 1)
        self.feature_mean_histogram = np.zeros(c.usage_bins)
        
        self.feature_max_bin_edges = np.linspace(c.feature_max_range[0], c.feature_max_range[1], c.usage_bins + 1)
        self.feature_max_histogram = np.zeros(c.usage_bins)
        
        self.feature_selection_freq_bin_edges = np.linspace(c.selection_freq_range[0], c.selection_freq_range[1], c.usage_bins + 1)
        self.feature_selection_freq_histogram = np.zeros(c.usage_bins)
        
        self.feature_cv_bin_edges = np.linspace(c.feature_cv_range[0], c.feature_cv_range[1], c.usage_bins + 1)
        self.feature_cv_histogram = np.zeros(c.usage_bins)
        
        self.feature_dominance_bin_edges = np.linspace(c.feature_dominance_range[0], c.feature_dominance_range[1], c.usage_bins + 1)
        self.feature_dominance_histogram = np.zeros(c.usage_bins)
        
        self.feature_polarity_bin_edges = np.linspace(c.feature_polarity_range[0], c.feature_polarity_range[1], c.usage_bins + 1)
        self.feature_polarity_histogram = np.zeros(c.usage_bins)
        
        # 2D histograms
        self.feature_index_edges = np.linspace(0, self.dictionary.dict_size, c.feature_index_bins + 1)
        self.activation_2d_edges = np.linspace(c.activation_2d_range[0], c.activation_2d_range[1], c.activation_2d_bins + 1)
        self.feature_index_activation_2d = np.zeros((c.feature_index_bins, c.activation_2d_bins))
        
        self.position_edges = np.linspace(0, self.ctx_len, c.position_bins + 1)
        self.position_activation_edges = np.linspace(c.activation_2d_range[0], c.activation_2d_range[1], c.activation_2d_bins + 1)
        self.position_activation_2d = np.zeros((c.position_bins, c.activation_2d_bins))
        
        # New 1D histograms for feature index and position distributions
        self.feature_index_distribution_bin_edges = np.linspace(0, self.dictionary.dict_size, c.feature_index_distribution_bins + 1)
        self.feature_index_distribution_histogram = np.zeros(c.feature_index_distribution_bins)
        
        self.position_distribution_bin_edges = np.linspace(0, self.ctx_len, c.position_distribution_bins + 1) 
        self.position_distribution_histogram = np.zeros(c.position_distribution_bins)
        
        # Feature usage tracking
        self.feature_selection_counts = np.zeros(self.dictionary.dict_size)
        self.feature_activation_sums = np.zeros(self.dictionary.dict_size)
        self.feature_activation_counts = np.zeros(self.dictionary.dict_size)
        self.feature_activation_sum_squares = np.zeros(self.dictionary.dict_size)
        self.feature_max_activations = np.full(self.dictionary.dict_size, -np.inf)
        self.feature_positive_counts = np.zeros(self.dictionary.dict_size)
        
    def update_histograms(self, sparse_values: np.ndarray, sparse_indices: np.ndarray, positions: Optional[np.ndarray] = None) -> None:
        """Update all histograms with batch data."""
        batch_size = sparse_values.shape[0]
        
        # 1. Core activation distributions
        all_values = sparse_values.flatten()
        
        # All activations (including zeros from padding)
        hist, _ = np.histogram(all_values, bins=self.activation_bin_edges)
        self.activation_histogram += hist
        
        # Non-zero activations only
        nonzero_values = all_values[all_values != 0]
        if len(nonzero_values) > 0:
            hist, _ = np.histogram(nonzero_values, bins=self.nonzero_activation_bin_edges)
            self.nonzero_activation_histogram += hist
            
            # Activation magnitudes
            magnitude_values = np.abs(nonzero_values)
            hist, _ = np.histogram(magnitude_values, bins=self.activation_magnitude_bin_edges)
            self.activation_magnitude_histogram += hist
            
            # Positive and negative activations
            positive_values = nonzero_values[nonzero_values > 0]
            negative_values = nonzero_values[nonzero_values < 0]
            
            if len(positive_values) > 0:
                hist, _ = np.histogram(positive_values, bins=self.positive_activation_bin_edges)
                self.positive_activation_histogram += hist
            
            if len(negative_values) > 0:
                hist, _ = np.histogram(negative_values, bins=self.negative_activation_bin_edges)
                self.negative_activation_histogram += hist
            
            # Log-scale activations (for magnitude > 0)
            log_values = np.log10(np.maximum(magnitude_values, 1e-10))
            hist, _ = np.histogram(log_values, bins=self.log_activation_bin_edges)
            self.log_activation_histogram += hist
        
        # Update overall statistics
        for val in nonzero_values:
            self.overall_stats.update(float(val))
        
        # 2. Per-sample statistics
        sample_max_activations = np.max(np.abs(sparse_values), axis=1)
        sample_mean_magnitudes = np.mean(np.abs(sparse_values), axis=1)
        sample_ranges = np.max(sparse_values, axis=1) - np.min(sparse_values, axis=1)
        sample_pos_ratios = np.mean(sparse_values > 0, axis=1)
        
        # K-th largest activation (should be the smallest in TopK)
        sample_kth_activations = np.min(np.abs(sparse_values), axis=1)
        
        # Coefficient of variation per sample
        sample_means = np.mean(sparse_values, axis=1)
        sample_stds = np.std(sparse_values, axis=1)
        sample_cvs = np.divide(sample_stds, np.abs(sample_means), out=np.zeros_like(sample_stds), where=sample_means!=0)
        
        # Selection margins (difference between k-th and (k+1)-th - approximate as range for now)
        sample_margins = sample_ranges  # Approximation
        
        # Quantiles per sample
        sample_q10 = np.percentile(sparse_values, 10, axis=1)
        sample_q50 = np.percentile(sparse_values, 50, axis=1)
        sample_q90 = np.percentile(sparse_values, 90, axis=1)
        
        # Effective features firing per sample (non-zero count)
        sample_effective_features = np.sum(sparse_values != 0, axis=1)
        
        # Update sample histograms
        hist, _ = np.histogram(sample_max_activations, bins=self.sample_max_bin_edges)
        self.sample_max_histogram += hist
        
        hist, _ = np.histogram(sample_mean_magnitudes, bins=self.sample_mean_magnitude_bin_edges)
        self.sample_mean_magnitude_histogram += hist
        
        hist, _ = np.histogram(sample_ranges, bins=self.sample_range_bin_edges)
        self.sample_range_histogram += hist
        
        hist, _ = np.histogram(sample_pos_ratios, bins=self.sample_pos_neg_ratio_bin_edges)
        self.sample_pos_neg_ratio_histogram += hist
        
        hist, _ = np.histogram(sample_kth_activations, bins=self.sample_kth_activation_bin_edges)
        self.sample_kth_activation_histogram += hist
        
        hist, _ = np.histogram(sample_cvs, bins=self.sample_cv_bin_edges)
        self.sample_cv_histogram += hist
        
        hist, _ = np.histogram(sample_margins, bins=self.sample_margin_bin_edges)
        self.sample_margin_histogram += hist
        
        hist, _ = np.histogram(sample_q10, bins=self.sample_quantile_bin_edges)
        self.sample_q10_histogram += hist
        
        hist, _ = np.histogram(sample_q50, bins=self.sample_quantile_bin_edges)
        self.sample_q50_histogram += hist
        
        hist, _ = np.histogram(sample_q90, bins=self.sample_quantile_bin_edges)
        self.sample_q90_histogram += hist
        
        hist, _ = np.histogram(sample_effective_features, bins=self.sample_effective_features_bin_edges)
        self.sample_effective_features_histogram += hist
        
        # 3. Feature usage statistics (accumulate for later histogram computation)
        for i in range(batch_size):
            indices = sparse_indices[i]
            values = sparse_values[i]
            
            # Remove zero-padded values
            valid_mask = values != 0
            if not np.any(valid_mask):
                continue
                
            valid_indices = indices[valid_mask]
            valid_values = values[valid_mask]
            
            # Count selections
            self.feature_selection_counts[valid_indices] += 1
            
            # Sum activation values and squares for mean and variance calculation
            self.feature_activation_sums[valid_indices] += valid_values
            self.feature_activation_counts[valid_indices] += 1
            self.feature_activation_sum_squares[valid_indices] += valid_values ** 2
            
            # Track max activations
            for idx, val in zip(valid_indices, valid_values):
                self.feature_max_activations[idx] = max(self.feature_max_activations[idx], abs(val))
            
            # Count positive activations
            positive_mask = valid_values > 0
            self.feature_positive_counts[valid_indices[positive_mask]] += 1
        
        # 4. 2D histograms and new 1D distributions
        # Feature index vs activation value
        for i in range(batch_size):
            indices = sparse_indices[i]
            values = sparse_values[i]
            
            valid_mask = values != 0
            if np.any(valid_mask):
                valid_indices = indices[valid_mask]
                valid_values = values[valid_mask]
                
                # 2D histogram
                hist2d, _, _ = np.histogram2d(valid_indices, valid_values, 
                                            bins=[self.feature_index_edges, self.activation_2d_edges])
                self.feature_index_activation_2d += hist2d
                
                # New 1D histogram: Feature index distribution (which features are being selected)
                hist, _ = np.histogram(valid_indices, bins=self.feature_index_distribution_bin_edges)
                self.feature_index_distribution_histogram += hist
        
        # Position vs activation value (if positions provided)
        if positions is not None:
            for i in range(batch_size):
                sample_positions = positions[i]
                values = sparse_values[i]
                
                valid_mask = values != 0
                if np.any(valid_mask):
                    valid_positions = sample_positions[valid_mask]
                    valid_values = values[valid_mask]
                    
                    # 2D histogram  
                    hist2d, _, _ = np.histogram2d(valid_positions, valid_values,
                                                bins=[self.position_edges, self.position_activation_edges])
                    self.position_activation_2d += hist2d
                    
                    # New 1D histogram: Position distribution (which positions have activations)
                    hist, _ = np.histogram(valid_positions, bins=self.position_distribution_bin_edges)
                    self.position_distribution_histogram += hist
        
        self.samples_processed += batch_size
        
    def finalize_feature_histograms(self) -> None:
        """Compute feature-wise histograms after all data is processed."""
        # Compute feature statistics
        nonzero_mask = self.feature_activation_counts > 0
        
        # Feature means
        feature_means = np.zeros(self.dictionary.dict_size)
        feature_means[nonzero_mask] = (
            self.feature_activation_sums[nonzero_mask] / self.feature_activation_counts[nonzero_mask]
        )
        
        # Feature coefficient of variation
        feature_variances = np.zeros(self.dictionary.dict_size)
        feature_variances[nonzero_mask] = (
            (self.feature_activation_sum_squares[nonzero_mask] / self.feature_activation_counts[nonzero_mask]) -
            feature_means[nonzero_mask] ** 2
        )
        feature_stds = np.sqrt(np.maximum(feature_variances, 0))
        feature_cvs = np.divide(feature_stds, np.abs(feature_means), out=np.zeros_like(feature_stds), where=(feature_means!=0))
        
        # Feature dominance (max / mean)
        feature_dominance = np.divide(self.feature_max_activations, np.abs(feature_means), 
                                    out=np.ones_like(feature_means), where=(feature_means!=0))
        feature_dominance[~nonzero_mask] = 1.0  # Set to 1 for unused features
        
        # Feature polarity (fraction of positive activations)
        feature_polarity = np.divide(self.feature_positive_counts, self.feature_activation_counts,
                                   out=np.zeros_like(self.feature_positive_counts), where=nonzero_mask)
        
        # Selection frequencies
        selection_frequencies = self.feature_selection_counts / self.samples_processed
        
        # Create histograms for features that were actually used
        used_features_mask = nonzero_mask
        
        if np.any(used_features_mask):
            # Feature means
            hist, _ = np.histogram(feature_means[used_features_mask], bins=self.feature_mean_bin_edges)
            self.feature_mean_histogram = hist
            
            # Feature max activations
            valid_max_mask = self.feature_max_activations > -np.inf
            if np.any(valid_max_mask):
                hist, _ = np.histogram(self.feature_max_activations[valid_max_mask], bins=self.feature_max_bin_edges)
                self.feature_max_histogram = hist
            
            # Feature CVs
            finite_cv_mask = np.isfinite(feature_cvs) & used_features_mask
            if np.any(finite_cv_mask):
                hist, _ = np.histogram(feature_cvs[finite_cv_mask], bins=self.feature_cv_bin_edges)
                self.feature_cv_histogram = hist
            
            # Feature dominance
            finite_dom_mask = np.isfinite(feature_dominance) & (feature_dominance > 0) & used_features_mask
            if np.any(finite_dom_mask):
                hist, _ = np.histogram(feature_dominance[finite_dom_mask], bins=self.feature_dominance_bin_edges)
                self.feature_dominance_histogram = hist
            
            # Feature polarity
            hist, _ = np.histogram(feature_polarity[used_features_mask], bins=self.feature_polarity_bin_edges)
            self.feature_polarity_histogram = hist
        
        # Selection frequencies (all features)
        hist, _ = np.histogram(selection_frequencies, bins=self.feature_selection_freq_bin_edges)
        self.feature_selection_freq_histogram = hist
        
    def process_data_online(self) -> None:
            """Process activation data online, updating histograms incrementally."""
            self.logger.info(f"Starting comprehensive online processing of {self.n_samples} samples")
            
            self.initialize_histograms()
            samples_collected = 0
            
            # Progress bar
            pbar = tqdm(total=self.n_samples, desc="Processing activations online")
            
            try:
                while samples_collected < self.n_samples:
                    # Get batch of activations
                    try:
                        activations_batch = next(self.buffer).to(self.device)
                    except StopIteration:
                        self.logger.warning("Ran out of data before processing enough samples")
                        break
                    
                    # Get TopK sparse activations
                    with torch.no_grad():
                        if hasattr(self.dictionary, 'encode') and hasattr(self.dictionary, 'k'):
                            # TopK VSAE models with encode method that returns TopK info
                            try:
                                # Check if this is VSAETopK (has training parameter) or AutoEncoderTopK (doesn't)
                                if 'VSAETopK' in type(self.dictionary).__name__:
                                    sparse_features, _, _, _, top_indices, selected_vals = self.dictionary.encode(
                                        activations_batch, return_topk=True, training=False
                                    )
                                else:
                                    # AutoEncoderTopK doesn't have training parameter
                                    sparse_features, selected_vals, top_indices, _ = self.dictionary.encode(
                                        activations_batch, return_topk=True
                                    )
                                sparse_values_np = selected_vals.cpu().float().numpy()
                                sparse_indices_np = top_indices.cpu().numpy()
                            except Exception as e:
                                # Fallback: use sparse features and reconstruct indices
                                try:
                                    if 'VSAETopK' in type(self.dictionary).__name__:
                                        sparse_features, _, _, _ = self.dictionary.encode(activations_batch, training=False)
                                    else:
                                        # AutoEncoderTopK doesn't have training parameter
                                        sparse_features = self.dictionary.encode(activations_batch)
                                except Exception:
                                    # Final fallback: use forward pass
                                    _, sparse_features = self.dictionary(activations_batch, output_features=True)
                                
                                # Find non-zero indices and values
                                sparse_features_np = sparse_features.cpu().float().numpy()
                                batch_size = sparse_features_np.shape[0]
                                
                                sparse_values_list = []
                                sparse_indices_list = []
                                
                                for i in range(batch_size):
                                    nonzero_idx = np.nonzero(sparse_features_np[i])[0]
                                    nonzero_vals = sparse_features_np[i, nonzero_idx]
                                    
                                    # Pad or truncate to k values
                                    if len(nonzero_idx) >= self.k_value:
                                        # Take top k by absolute value
                                        top_k_mask = np.argpartition(np.abs(nonzero_vals), -self.k_value)[-self.k_value:]
                                        sparse_indices_list.append(nonzero_idx[top_k_mask])
                                        sparse_values_list.append(nonzero_vals[top_k_mask])
                                    else:
                                        # Pad with zeros
                                        padded_indices = np.zeros(self.k_value, dtype=np.int32)
                                        padded_values = np.zeros(self.k_value, dtype=np.float32)
                                        padded_indices[:len(nonzero_idx)] = nonzero_idx
                                        padded_values[:len(nonzero_vals)] = nonzero_vals
                                        sparse_indices_list.append(padded_indices)
                                        sparse_values_list.append(padded_values)
                                
                                sparse_values_np = np.array(sparse_values_list)
                                sparse_indices_np = np.array(sparse_indices_list)
                        else:
                            # Standard SAE models - get features and find top-k
                            _, features = self.dictionary(activations_batch, output_features=True)
                            features_np = features.cpu().float().numpy()
                            
                            batch_size = features_np.shape[0]
                            sparse_values_list = []
                            sparse_indices_list = []
                            
                            for i in range(batch_size):
                                # Get top-k by absolute value
                                abs_features = np.abs(features_np[i])
                                top_k_indices = np.argpartition(abs_features, -self.k_value)[-self.k_value:]
                                top_k_values = features_np[i, top_k_indices]
                                
                                sparse_indices_list.append(top_k_indices)
                                sparse_values_list.append(top_k_values)
                            
                            sparse_values_np = np.array(sparse_values_list)
                            sparse_indices_np = np.array(sparse_indices_list)
                    
                    # Generate position information for sequence analysis
                    batch_size = sparse_values_np.shape[0]
                    positions = np.tile(np.arange(self.k_value), (batch_size, 1))  # Simple position approximation
                    
                    # Update all histograms
                    self.update_histograms(sparse_values_np, sparse_indices_np, positions)
                    
                    samples_collected += batch_size
                    pbar.update(batch_size)
                    
                    # Stop if we have enough samples
                    if samples_collected >= self.n_samples:
                        break
                        
            except Exception as e:
                self.logger.error(f"Error during online processing: {e}")
                raise
            finally:
                pbar.close()
            
            # Finalize feature-wise statistics
            self.finalize_feature_histograms()
            
            self.logger.info(f"Processed {self.samples_processed} samples")
            self.logger.info("Computing final feature-wise histograms...")
        
    def create_individual_histograms(self, output_dir: Path) -> List[str]:
        """Create individual histogram plots and return list of filenames."""
        model_name = self.get_model_name()
        saved_files = []
        
        def save_histogram(data, bin_edges, title, xlabel, filename, log_scale=False):
            """Helper function to save individual histogram."""
            if np.sum(data) == 0:
                self.logger.warning(f"No data for {title}, skipping")
                return
                
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, data, width=np.diff(bin_edges), alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count (log scale)' if log_scale else 'Count')
            ax.set_title(f'{title}\n{model_name}\nTotal: {int(data.sum()):,}')
            ax.grid(True, alpha=0.3)
            
            if log_scale:
                ax.set_yscale('log')
            
            plt.tight_layout()
            file_path = output_dir / f"{filename}_{model_name.replace('/', '_')}.png"
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(file_path))
        
        # Core activation distributions
        save_histogram(self.activation_histogram, self.activation_bin_edges, 
                      'All Activation Values', 'Activation Value', 'hist_01_all_activations')
        
        save_histogram(self.nonzero_activation_histogram, self.nonzero_activation_bin_edges,
                      'Non-Zero Activation Values', 'Activation Value', 'hist_02_nonzero_activations')
        
        save_histogram(self.activation_magnitude_histogram, self.activation_magnitude_bin_edges,
                      'Activation Magnitudes', 'Activation Magnitude', 'hist_03_activation_magnitudes')
        
        save_histogram(self.positive_activation_histogram, self.positive_activation_bin_edges,
                      'Positive Activation Values', 'Positive Activation Value', 'hist_04_positive_activations')
        
        save_histogram(self.negative_activation_histogram, self.negative_activation_bin_edges,
                      'Negative Activation Values', 'Negative Activation Value', 'hist_05_negative_activations')
        
        save_histogram(self.log_activation_histogram, self.log_activation_bin_edges,
                      'Log-Scale Activation Magnitudes', 'Log₁₀(Activation Magnitude)', 'hist_06_log_activations')
        
        # Per-sample statistics
        save_histogram(self.sample_max_histogram, self.sample_max_bin_edges,
                      'Max Activation per Sample', 'Max Activation Magnitude', 'hist_07_sample_max')
        
        save_histogram(self.sample_mean_magnitude_histogram, self.sample_mean_magnitude_bin_edges,
                      'Mean Magnitude per Sample', 'Mean Activation Magnitude', 'hist_08_sample_mean_magnitude')
        
        save_histogram(self.sample_range_histogram, self.sample_range_bin_edges,
                      'Activation Range per Sample', 'Activation Range', 'hist_09_sample_range')
        
        save_histogram(self.sample_pos_neg_ratio_histogram, self.sample_pos_neg_ratio_bin_edges,
                      'Positive/Negative Ratio per Sample', 'Fraction Positive', 'hist_10_sample_pos_neg_ratio')
        
        save_histogram(self.sample_kth_activation_histogram, self.sample_kth_activation_bin_edges,
                      'K-th Largest Activation per Sample', 'K-th Activation Magnitude', 'hist_11_sample_kth_activation')
        
        save_histogram(self.sample_cv_histogram, self.sample_cv_bin_edges,
                      'Coefficient of Variation per Sample', 'CV (std/mean)', 'hist_12_sample_cv')
        
        save_histogram(self.sample_margin_histogram, self.sample_margin_bin_edges,
                      'Selection Margin per Sample', 'Selection Margin', 'hist_13_sample_margin')
        
        save_histogram(self.sample_q10_histogram, self.sample_quantile_bin_edges,
                      '10th Percentile per Sample', '10th Percentile Value', 'hist_14_sample_q10')
        
        save_histogram(self.sample_q50_histogram, self.sample_quantile_bin_edges,
                      'Median per Sample', 'Median Value', 'hist_15_sample_median')
        
        save_histogram(self.sample_q90_histogram, self.sample_quantile_bin_edges,
                      '90th Percentile per Sample', '90th Percentile Value', 'hist_16_sample_q90')
        
        save_histogram(self.sample_effective_features_histogram, self.sample_effective_features_bin_edges,
                      'Effective Features Firing per Sample', 'Number of Non-Zero Features', 'hist_17_sample_effective_features')
        
        # Feature usage patterns
        save_histogram(self.feature_selection_freq_histogram, self.feature_selection_freq_bin_edges,
                      'Feature Selection Frequency', 'Selection Frequency', 'hist_18_feature_selection_freq', log_scale=True)
        
        save_histogram(self.feature_mean_histogram, self.feature_mean_bin_edges,
                      'Feature Mean Activation', 'Mean Activation', 'hist_19_feature_means')
        
        save_histogram(self.feature_max_histogram, self.feature_max_bin_edges,
                      'Feature Max Activation', 'Max Activation', 'hist_20_feature_max')
        
        save_histogram(self.feature_cv_histogram, self.feature_cv_bin_edges,
                      'Feature Activation Consistency (CV)', 'Coefficient of Variation', 'hist_21_feature_cv')
        
        save_histogram(self.feature_dominance_histogram, self.feature_dominance_bin_edges,
                      'Feature Dominance (Max/Mean)', 'Dominance Ratio', 'hist_22_feature_dominance')
        
        save_histogram(self.feature_polarity_histogram, self.feature_polarity_bin_edges,
                      'Feature Polarity Bias', 'Fraction Positive', 'hist_23_feature_polarity')
        
        # New 1D distributions for 2D plot data
        save_histogram(self.feature_index_distribution_histogram, self.feature_index_distribution_bin_edges,
                      'Feature Index Usage Distribution', 'Feature Index', 'hist_24_feature_index_distribution')
        
        save_histogram(self.position_distribution_histogram, self.position_distribution_bin_edges,
                      'Sequence Position Activation Distribution', 'Sequence Position', 'hist_25_position_distribution')
        
        self.logger.info(f"Saved {len(saved_files)} individual histograms")
        return saved_files
    
    def create_2d_visualizations(self, output_dir: Path) -> List[str]:
        """Create 2D histogram visualizations."""
        model_name = self.get_model_name()
        saved_files = []
        
        # Feature index vs activation value
        if np.sum(self.feature_index_activation_2d) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            im = ax.imshow(self.feature_index_activation_2d.T, aspect='auto', origin='lower', 
                          extent=[self.feature_index_edges[0], self.feature_index_edges[-1],
                                 self.activation_2d_edges[0], self.activation_2d_edges[-1]],
                          cmap='hot')
            
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Activation Value')
            ax.set_title(f'Feature Index vs Activation Value\n{model_name}')
            plt.colorbar(im, ax=ax, label='Count')
            
            plt.tight_layout()
            file_path = output_dir / f"hist_2d_01_feature_index_activation_{model_name.replace('/', '_')}.png"
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(file_path))
        
        # Position vs activation value
        if np.sum(self.position_activation_2d) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            im = ax.imshow(self.position_activation_2d.T, aspect='auto', origin='lower',
                          extent=[self.position_edges[0], self.position_edges[-1],
                                 self.position_activation_edges[0], self.position_activation_edges[-1]],
                          cmap='viridis')
            
            ax.set_xlabel('Sequence Position')
            ax.set_ylabel('Activation Value')
            ax.set_title(f'Sequence Position vs Activation Value\n{model_name}')
            plt.colorbar(im, ax=ax, label='Count')
            
            plt.tight_layout()
            file_path = output_dir / f"hist_2d_02_position_activation_{model_name.replace('/', '_')}.png"
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(str(file_path))
        
        self.logger.info(f"Saved {len(saved_files)} 2D visualizations")
        return saved_files
    
    def create_panel_visualizations(self, output_dir: Path) -> List[str]:
        """Create multi-panel summary visualizations."""
        model_name = self.get_model_name()
        saved_files = []
        
        # Panel 1: Core activation distributions (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        def plot_panel_hist(ax, data, bin_edges, title, xlabel, log_scale=False):
            if np.sum(data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, data, width=np.diff(bin_edges), alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if log_scale:
                ax.set_yscale('log')
        
        plot_panel_hist(axes[0], self.activation_histogram, self.activation_bin_edges,
                       'All Activations', 'Activation Value')
        plot_panel_hist(axes[1], self.nonzero_activation_histogram, self.nonzero_activation_bin_edges,
                       'Non-Zero Activations', 'Activation Value')
        plot_panel_hist(axes[2], self.activation_magnitude_histogram, self.activation_magnitude_bin_edges,
                       'Activation Magnitudes', 'Magnitude')
        plot_panel_hist(axes[3], self.positive_activation_histogram, self.positive_activation_bin_edges,
                       'Positive Activations', 'Positive Value')
        plot_panel_hist(axes[4], self.negative_activation_histogram, self.negative_activation_bin_edges,
                       'Negative Activations', 'Negative Value')
        plot_panel_hist(axes[5], self.log_activation_histogram, self.log_activation_bin_edges,
                       'Log-Scale Magnitudes', 'Log₁₀(Magnitude)')
        
        plt.suptitle(f'Core Activation Distributions\n{model_name}', fontsize=16)
        plt.tight_layout()
        file_path = output_dir / f"panel_01_core_distributions_{model_name.replace('/', '_')}.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        # Panel 2: Per-sample statistics (3x4 instead of 3x3)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        plot_panel_hist(axes[0], self.sample_max_histogram, self.sample_max_bin_edges,
                       'Max per Sample', 'Max Magnitude')
        plot_panel_hist(axes[1], self.sample_mean_magnitude_histogram, self.sample_mean_magnitude_bin_edges,
                       'Mean Magnitude per Sample', 'Mean Magnitude')
        plot_panel_hist(axes[2], self.sample_range_histogram, self.sample_range_bin_edges,
                       'Range per Sample', 'Range')
        plot_panel_hist(axes[3], self.sample_pos_neg_ratio_histogram, self.sample_pos_neg_ratio_bin_edges,
                       'Pos/Neg Ratio per Sample', 'Fraction Positive')
        plot_panel_hist(axes[4], self.sample_kth_activation_histogram, self.sample_kth_activation_bin_edges,
                       'K-th Activation per Sample', 'K-th Magnitude')
        plot_panel_hist(axes[5], self.sample_cv_histogram, self.sample_cv_bin_edges,
                       'CV per Sample', 'Coefficient of Variation')
        plot_panel_hist(axes[6], self.sample_q10_histogram, self.sample_quantile_bin_edges,
                       '10th Percentile', '10th Percentile')
        plot_panel_hist(axes[7], self.sample_q50_histogram, self.sample_quantile_bin_edges,
                       'Median', 'Median')
        plot_panel_hist(axes[8], self.sample_q90_histogram, self.sample_quantile_bin_edges,
                       '90th Percentile', '90th Percentile')
        plot_panel_hist(axes[9], self.sample_effective_features_histogram, self.sample_effective_features_bin_edges,
                       'Effective Features Firing', 'Number of Non-Zero Features')
        
        # Hide unused subplots
        for i in range(10, 12):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Per-Sample Statistics\n{model_name}', fontsize=16)
        plt.tight_layout()
        file_path = output_dir / f"panel_02_sample_statistics_{model_name.replace('/', '_')}.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        # Panel 3: Feature patterns (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_panel_hist(axes[0], self.feature_selection_freq_histogram, self.feature_selection_freq_bin_edges,
                       'Selection Frequency', 'Frequency', log_scale=True)
        plot_panel_hist(axes[1], self.feature_mean_histogram, self.feature_mean_bin_edges,
                       'Feature Means', 'Mean Activation')
        plot_panel_hist(axes[2], self.feature_max_histogram, self.feature_max_bin_edges,
                       'Feature Max Values', 'Max Activation')
        plot_panel_hist(axes[3], self.feature_cv_histogram, self.feature_cv_bin_edges,
                       'Feature Consistency', 'CV')
        plot_panel_hist(axes[4], self.feature_dominance_histogram, self.feature_dominance_bin_edges,
                       'Feature Dominance', 'Max/Mean Ratio')
        plot_panel_hist(axes[5], self.feature_polarity_histogram, self.feature_polarity_bin_edges,
                       'Feature Polarity', 'Fraction Positive')
        
        plt.suptitle(f'Feature Usage Patterns\n{model_name}', fontsize=16)
        plt.tight_layout()
        file_path = output_dir / f"panel_03_feature_patterns_{model_name.replace('/', '_')}.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(str(file_path))
        
        self.logger.info(f"Saved {len(saved_files)} panel visualizations")
        return saved_files
    
    def save_comprehensive_results(self, output_dir: Path) -> None:
        """Save all histogram data and comprehensive summary statistics."""
        model_name = self.get_model_name()
        
        # Compute comprehensive summary statistics
        nonzero_mask = self.feature_activation_counts > 0
        feature_means = np.zeros(self.dictionary.dict_size)
        feature_means[nonzero_mask] = (
            self.feature_activation_sums[nonzero_mask] / self.feature_activation_counts[nonzero_mask]
        )
        
        selection_frequencies = self.feature_selection_counts / self.samples_processed
        
        comprehensive_summary = {
            'model_info': {
                'model_name': self.model_info['model_name'],
                'hook_name': self.model_info['hook_name'], 
                'dict_size': int(self.model_info['dict_size']),
                'activation_dim': int(self.model_info['activation_dim']),
                'dictionary_type': self.model_info['dictionary_type'],
                'kl_coeff': float(self.model_info['kl_coeff']) if self.model_info['kl_coeff'] is not None else None,
                'k_value': int(self.model_info['k_value']) if self.model_info['k_value'] is not None else None,
            },
            'processing_info': {
                'samples_processed': int(self.samples_processed),
                'target_samples': int(self.n_samples),
                'k_value': int(self.k_value),
                'seed': int(self.seed),
                'ctx_len': int(self.ctx_len),
            },
            'overall_activation_stats': self.overall_stats.to_dict(),
            'histogram_config': asdict(self.config),
            'feature_usage_summary': {
                'total_features': int(self.dictionary.dict_size),
                'features_used': int(np.sum(nonzero_mask)),
                'features_never_selected': int(np.sum(selection_frequencies == 0)),
                'mean_selection_frequency': float(np.mean(selection_frequencies)),
                'std_selection_frequency': float(np.std(selection_frequencies)),
                'most_used_features': np.argsort(selection_frequencies)[-20:].tolist(),
                'least_used_features': np.argsort(selection_frequencies)[:20].tolist(),
            },
            'activation_distribution_stats': {
                'total_activations_processed': int(self.overall_stats.count),
                'mean_activation': float(self.overall_stats.mean),
                'std_activation': float(self.overall_stats.std),
                'min_activation': float(self.overall_stats.min_val),
                'max_activation': float(self.overall_stats.max_val),
            }
        }
        
        # Save comprehensive summary
        summary_file = output_dir / f"comprehensive_summary_{model_name.replace('/', '_')}.json"
        with open(summary_file, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2)
        
        # Save all histogram data
        all_histogram_data = {
            # Core activation distributions
            'activation_histogram': self.activation_histogram,
            'activation_bin_edges': self.activation_bin_edges,
            'nonzero_activation_histogram': self.nonzero_activation_histogram,
            'nonzero_activation_bin_edges': self.nonzero_activation_bin_edges,
            'activation_magnitude_histogram': self.activation_magnitude_histogram,
            'activation_magnitude_bin_edges': self.activation_magnitude_bin_edges,
            'positive_activation_histogram': self.positive_activation_histogram,
            'positive_activation_bin_edges': self.positive_activation_bin_edges,
            'negative_activation_histogram': self.negative_activation_histogram,
            'negative_activation_bin_edges': self.negative_activation_bin_edges,
            'log_activation_histogram': self.log_activation_histogram,
            'log_activation_bin_edges': self.log_activation_bin_edges,
            
            # Per-sample statistics
            'sample_max_histogram': self.sample_max_histogram,
            'sample_max_bin_edges': self.sample_max_bin_edges,
            'sample_mean_magnitude_histogram': self.sample_mean_magnitude_histogram,
            'sample_mean_magnitude_bin_edges': self.sample_mean_magnitude_bin_edges,
            'sample_range_histogram': self.sample_range_histogram,
            'sample_range_bin_edges': self.sample_range_bin_edges,
            'sample_pos_neg_ratio_histogram': self.sample_pos_neg_ratio_histogram,
            'sample_pos_neg_ratio_bin_edges': self.sample_pos_neg_ratio_bin_edges,
            'sample_kth_activation_histogram': self.sample_kth_activation_histogram,
            'sample_kth_activation_bin_edges': self.sample_kth_activation_bin_edges,
            'sample_cv_histogram': self.sample_cv_histogram,
            'sample_cv_bin_edges': self.sample_cv_bin_edges,
            'sample_margin_histogram': self.sample_margin_histogram,
            'sample_margin_bin_edges': self.sample_margin_bin_edges,
            'sample_q10_histogram': self.sample_q10_histogram,
            'sample_q50_histogram': self.sample_q50_histogram,
            'sample_q90_histogram': self.sample_q90_histogram,
            'sample_quantile_bin_edges': self.sample_quantile_bin_edges,
            'sample_effective_features_histogram': self.sample_effective_features_histogram,
            'sample_effective_features_bin_edges': self.sample_effective_features_bin_edges,
            
            # Feature patterns
            'feature_selection_freq_histogram': self.feature_selection_freq_histogram,
            'feature_selection_freq_bin_edges': self.feature_selection_freq_bin_edges,
            'feature_mean_histogram': self.feature_mean_histogram,
            'feature_mean_bin_edges': self.feature_mean_bin_edges,
            'feature_max_histogram': self.feature_max_histogram,
            'feature_max_bin_edges': self.feature_max_bin_edges,
            'feature_cv_histogram': self.feature_cv_histogram,
            'feature_cv_bin_edges': self.feature_cv_bin_edges,
            'feature_dominance_histogram': self.feature_dominance_histogram,
            'feature_dominance_bin_edges': self.feature_dominance_bin_edges,
            'feature_polarity_histogram': self.feature_polarity_histogram,
            'feature_polarity_bin_edges': self.feature_polarity_bin_edges,
            
            # 2D histograms
            'feature_index_activation_2d': self.feature_index_activation_2d,
            'feature_index_edges': self.feature_index_edges,
            'activation_2d_edges': self.activation_2d_edges,
            'position_activation_2d': self.position_activation_2d,
            'position_edges': self.position_edges,
            'position_activation_edges': self.position_activation_edges,
            
            # New 1D distributions
            'feature_index_distribution_histogram': self.feature_index_distribution_histogram,
            'feature_index_distribution_bin_edges': self.feature_index_distribution_bin_edges,
            'position_distribution_histogram': self.position_distribution_histogram,
            'position_distribution_bin_edges': self.position_distribution_bin_edges,
            
            # Raw feature statistics for further analysis
            'feature_selection_counts': self.feature_selection_counts,
            'feature_activation_sums': self.feature_activation_sums,
            'feature_activation_counts': self.feature_activation_counts,
            'feature_activation_sum_squares': self.feature_activation_sum_squares,
            'feature_max_activations': self.feature_max_activations,
            'feature_positive_counts': self.feature_positive_counts,
        }
        
        histogram_file = output_dir / f"all_histograms_{model_name.replace('/', '_')}.npz"
        np.savez_compressed(histogram_file, **all_histogram_data)
        
        self.logger.info(f"Saved comprehensive summary to {summary_file}")
        self.logger.info(f"Saved all histogram data to {histogram_file}")
        
    def run(self, output_dir: Path) -> None:
        """Run the complete comprehensive analysis pipeline."""
        # Create model-specific output directory
        model_name = self.get_model_name()
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Output directory: {model_output_dir}")
        self.logger.info(f"Will generate ~27 different histograms")
        
        # Run pipeline
        self.load_model_and_dictionary()
        self.create_buffer()
        self.process_data_online()
        
        # Create visualizations
        saved_files = []
        
        if self.save_individual:
            self.logger.info("Creating individual histogram plots...")
            individual_files = self.create_individual_histograms(model_output_dir)
            saved_files.extend(individual_files)
        
        if self.save_panels:
            self.logger.info("Creating panel visualizations...")
            panel_files = self.create_panel_visualizations(model_output_dir)
            saved_files.extend(panel_files)
        
        self.logger.info("Creating 2D visualizations...")
        d2_files = self.create_2d_visualizations(model_output_dir)
        saved_files.extend(d2_files)
        
        # Save comprehensive results
        self.save_comprehensive_results(model_output_dir)
        
        self.logger.info(f"Analysis complete! Generated {len(saved_files)} visualization files")


def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'comprehensive_histogram_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TopK VSAE activation histogram analysis")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory (containing ae.pt and config.json)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1_000_000,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comprehensive_histogram_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--activation-range",
        nargs=2,
        type=float,
        default=[-2.5, 2.5],
        help="Range for activation value histograms"
    )
    parser.add_argument(
        "--activation-bins",
        type=int,
        default=200,
        help="Number of bins for activation histograms"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Skip individual histogram plots (save space)"
    )
    parser.add_argument(
        "--no-panels",
        action="store_true",
        help="Skip panel visualizations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic processing"
    )
    
    args = parser.parse_args()
    
    # Create output directory and setup logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive TopK VSAE histogram analysis")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Samples: {args.n_samples:,}")
    logger.info(f"Output: {output_dir}")
    logger.info("Will generate 25+ individual histograms + 3 panel plots + 2 2D plots")
    
    # Create histogram config
    histogram_config = HistogramConfig(
        activation_bins=args.activation_bins,
        activation_range=tuple(args.activation_range),
    )
    
    # Run comprehensive analysis
    analyzer = ComprehensiveHistogramAnalyzer(
        model_path=args.model_path,
        n_samples=args.n_samples,
        histogram_config=histogram_config,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        save_individual=not args.no_individual,
        save_panels=not args.no_panels,
    )
    
    try:
        analyzer.run(output_dir)
        logger.info("Comprehensive histogram analysis completed successfully!")
        logger.info("Generated histograms:")
        logger.info("  📊 25+ individual histogram plots")
        logger.info("  📋 3 multi-panel summary visualizations")
        logger.info("  🗺️  2D feature/position vs activation heatmaps")
        logger.info("  📈 Comprehensive summary statistics")
        logger.info("  💾 All raw histogram data for further analysis")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Usage examples:
# python online_histogram_analyzer.py --model-path ./experiments/VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl0.0001_aux0.03125_fixed_var/trainer_0

#python online_histogram_analyzer.py --model-path ./experiments/VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/trainer_0
#python online_histogram_analyzer.py --model-path ./experiments/TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto/trainer_0

# python online_histogram_analyzer.py --model-path ./experiments/VSAETopK_kl_1/trainer_0 --activation-range -3 3 --no-individual
# python online_histogram_analyzer.py --model-path ./experiments/VSAETopK_kl_500/trainer_0 --n-samples 500000 --activation-bins 300

# Output structure (~30+ files per model):
# comprehensive_histogram_analysis/
# ├── VSAETopK_model_name/
# │   ├── hist_01_all_activations_*.png
# │   ├── hist_02_nonzero_activations_*.png
# │   ├── ... (25+ individual histograms)
# │   ├── hist_25_position_distribution_*.png
# │   ├── panel_01_core_distributions_*.png
# │   ├── panel_02_sample_statistics_*.png
# │   ├── panel_03_feature_patterns_*.png
# │   ├── hist_2d_01_feature_index_activation_*.png
# │   ├── hist_2d_02_position_activation_*.png
# │   ├── comprehensive_summary_*.json
# │   └── all_histograms_*.npz
# └── logs/
#     └── comprehensive_histogram_analysis.log