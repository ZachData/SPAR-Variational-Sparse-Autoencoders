"""
Online Gaussianity Analysis for VSAE Feature Activations.

This script performs comprehensive Gaussianity analysis on VSAE feature activations
by processing data online (streaming) without storing raw activation data.
Combines data collection and analysis into a single efficient pipeline.

Key Features:
- Online processing for memory efficiency
- Q-Q plots with correlation scores
- Kolmogorov-Smirnov tests for normality
- Wasserstein distance to standard normal distribution
- Reservoir sampling for large datasets
- Minimal storage footprint
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
from scipy import stats
from scipy.stats import wasserstein_distance

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class OnlineGaussianityConfig:
    """Configuration for online Gaussianity analysis."""
    max_samples_per_feature: int = 50000  # Reservoir sampling limit
    target_total_samples: int = 1_000_000  # Total samples to process
    min_samples_for_analysis: int = 1000   # Minimum samples needed for analysis
    qq_quantiles: int = 1000              # Number of quantiles for Q-Q plots
    save_individual_plots: bool = False    # Save individual Q-Q plots
    create_summary_plots: bool = True      # Create summary visualizations


class ReservoirSampler:
    """Efficient reservoir sampling for online data collection."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.reservoir = []
        self.count = 0
    
    def add_samples(self, samples: np.ndarray) -> None:
        """Add new samples using reservoir sampling algorithm."""
        for sample in samples:
            self.count += 1
            if len(self.reservoir) < self.max_size:
                self.reservoir.append(float(sample))
            else:
                # Replace with probability max_size/count
                j = np.random.randint(0, self.count)
                if j < self.max_size:
                    self.reservoir[j] = float(sample)
    
    def get_samples(self) -> np.ndarray:
        """Get current reservoir samples."""
        return np.array(self.reservoir)
    
    def size(self) -> int:
        """Get current number of samples in reservoir."""
        return len(self.reservoir)


class OnlineStatistics:
    """Tracks running statistics without storing all data."""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance calculation
        self.m3 = 0.0  # For skewness calculation
        self.m4 = 0.0  # For kurtosis calculation
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
        self.zero_count = 0
    
    def update(self, value: float, is_zero: bool = False) -> None:
        """Update statistics with new value (Welford's algorithm)."""
        if is_zero:
            self.zero_count += 1
            return
            
        self.count += 1
        self.sum_val += value
        
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.m3 += delta * delta2 * (delta2)
        self.m4 += delta * delta2 * (delta2 * delta2)
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    @property
    def skewness(self) -> float:
        if self.count < 3 or self.variance == 0:
            return 0.0
        return (self.m3 / self.count) / (self.variance ** 1.5)
    
    @property
    def kurtosis(self) -> float:
        if self.count < 4 or self.variance == 0:
            return 0.0
        return (self.m4 / self.count) / (self.variance ** 2) - 3.0
    
    @property
    def zero_fraction(self) -> float:
        total = self.count + self.zero_count
        return self.zero_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'count': self.count,
            'zero_count': self.zero_count,
            'total_count': self.count + self.zero_count,
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'min': self.min_val if self.count > 0 else 0.0,
            'max': self.max_val if self.count > 0 else 0.0,
            'zero_fraction': self.zero_fraction,
        }


@dataclass
class GaussianityResults:
    """Container for Gaussianity analysis results."""
    feature_idx: int
    n_samples: int
    n_nonzero_samples: int
    
    # Basic statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float
    zero_fraction: float
    
    # Q-Q analysis
    qq_correlation: float
    qq_slope: float
    qq_intercept: float
    
    # Kolmogorov-Smirnov test
    ks_statistic: float
    ks_pvalue: float
    
    # Wasserstein distance
    wasserstein_dist: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'feature_idx': self.feature_idx,
            'n_samples': self.n_samples,
            'n_nonzero_samples': self.n_nonzero_samples,
            'mean': self.mean,
            'std': self.std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'zero_fraction': self.zero_fraction,
            'qq_correlation': self.qq_correlation,
            'qq_slope': self.qq_slope,
            'qq_intercept': self.qq_intercept,
            'ks_statistic': self.ks_statistic,
            'ks_pvalue': self.ks_pvalue,
            'wasserstein_dist': self.wasserstein_dist,
        }


class OnlineGaussianityAnalyzer:
    """Online Gaussianity analysis for VSAE feature activations."""
    
    def __init__(
        self,
        model_path: str,
        feature_indices: List[int],
        config: Optional[OnlineGaussianityConfig] = None,
        device: str = "cuda",
        batch_size: int = 32,
        ctx_len: int = 128,
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.feature_indices = feature_indices
        self.config = config or OnlineGaussianityConfig()
        self.device = device
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.seed = seed
        
        # Set seeds for deterministic data sampling
        self._set_seeds()
        
        self.logger = logging.getLogger(__name__)
        
        # Model components (will be initialized)
        self.model = None
        self.dictionary = None
        self.buffer = None
        
        # Feature data storage
        self.feature_samplers = {idx: ReservoirSampler(self.config.max_samples_per_feature) 
                               for idx in self.feature_indices}
        self.feature_stats = {idx: OnlineStatistics() for idx in self.feature_indices}
        
        # Global tracking
        self.samples_processed = 0
        
    def _set_seeds(self) -> None:
        """Set seeds for deterministic data sampling."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def get_model_name(self) -> str:
        """Extract model name from the path."""
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
        
        # Store hook name and metadata
        self.hook_name = hook_name
        self.model_info = {
            'model_name': model_name,
            'hook_name': hook_name,
            'dict_size': self.dictionary.dict_size,
            'activation_dim': self.dictionary.activation_dim,
            'dictionary_type': type(self.dictionary).__name__,
            'kl_coeff': config["trainer"].get("kl_coeff", None),
        }
        
        self.logger.info(f"Dictionary size: {self.dictionary.dict_size}")
        self.logger.info(f"Activation dim: {self.dictionary.activation_dim}")
        
    def create_buffer(self) -> None:
        """Create data buffer for activation collection."""
        self.logger.info("Setting up data buffer")
        
        # Create data generator
        data_gen = hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b",
            split="train",
            return_tokens=True
        )
        
        # Create buffer
        n_ctxs = min(3000, max(500, self.config.target_total_samples // self.ctx_len))
        
        self.buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=self.model,
            hook_name=self.hook_name,
            d_submodule=self.dictionary.activation_dim,
            n_ctxs=n_ctxs,
            ctx_len=self.ctx_len,
            refresh_batch_size=self.batch_size,
            out_batch_size=min(256, self.config.target_total_samples // 20),
            device=self.device,
        )
    
    def validate_features(self) -> None:
        """Validate that requested features exist in the dictionary."""
        max_feature = max(self.feature_indices)
        if max_feature >= self.dictionary.dict_size:
            raise ValueError(
                f"Feature index {max_feature} exceeds dictionary size {self.dictionary.dict_size}"
            )
        
        self.logger.info(f"Analyzing {len(self.feature_indices)} features: {self.feature_indices[:10]}{'...' if len(self.feature_indices) > 10 else ''}")
    
    def process_batch(self, features_batch: torch.Tensor) -> None:
        """Process a batch of features and update online statistics."""
        batch_size = features_batch.shape[0]
        
        # Convert to numpy
        if features_batch.dtype == torch.bfloat16:
            features_batch = features_batch.float()
        features_np = features_batch.cpu().numpy()
        
        # Update statistics for each feature of interest
        for feature_idx in self.feature_indices:
            feature_values = features_np[:, feature_idx]
            
            # Separate zeros from non-zeros
            nonzero_mask = feature_values != 0
            nonzero_values = feature_values[nonzero_mask]
            zero_count = np.sum(~nonzero_mask)
            
            # Update online statistics
            stats = self.feature_stats[feature_idx]
            
            # Add zeros
            for _ in range(zero_count):
                stats.update(0.0, is_zero=True)
            
            # Add non-zero values
            for val in nonzero_values:
                stats.update(val, is_zero=False)
            
            # Add non-zero samples to reservoir
            if len(nonzero_values) > 0:
                self.feature_samplers[feature_idx].add_samples(nonzero_values)
        
        self.samples_processed += batch_size
    
    def collect_activations_online(self) -> None:
        """Collect activation data online, updating statistics incrementally."""
        self.logger.info(f"Starting online collection of {self.config.target_total_samples} samples")
        
        samples_collected = 0
        pbar = tqdm(total=self.config.target_total_samples, desc="Processing activations")
        
        try:
            while samples_collected < self.config.target_total_samples:
                # Get batch of activations
                try:
                    activations_batch = next(self.buffer).to(self.device)
                except StopIteration:
                    self.logger.warning("Ran out of data before collecting enough samples")
                    break
                
                # Encode through dictionary to get feature activations
                with torch.no_grad():
                    if hasattr(self.dictionary, 'encode'):
                        # For VSAE models
                        if hasattr(self.dictionary, 'var_flag') and self.dictionary.var_flag == 1:
                            # VSAE with learned variance
                            mu, log_var = self.dictionary.encode(activations_batch)
                            features = self.dictionary.reparameterize(mu, log_var)
                        else:
                            # VSAE with fixed variance or other models  
                            result = self.dictionary.encode(activations_batch)
                            if isinstance(result, tuple):
                                features = result[0]  # Take sparse_features (first element)
                            else:
                                features = result
                    else:
                        # For standard SAE models
                        _, features = self.dictionary(activations_batch, output_features=True)
                
                # Process this batch
                self.process_batch(features)
                
                batch_size = features.shape[0]
                samples_collected += batch_size
                pbar.update(batch_size)
                
                # Stop if we have enough samples
                if samples_collected >= self.config.target_total_samples:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during collection: {e}")
            raise
        finally:
            pbar.close()
        
        self.logger.info(f"Collected data for {samples_collected} samples")
    
    def qq_analysis(self, feature_idx: int, save_plot: bool = False) -> Dict[str, float]:
        """
        Perform Q-Q plot analysis against standard normal distribution.
        """
        sampler = self.feature_samplers[feature_idx]
        
        if sampler.size() < self.config.min_samples_for_analysis:
            self.logger.warning(f"Feature {feature_idx}: Too few samples ({sampler.size()}) for Q-Q analysis")
            return {'correlation': 0.0, 'slope': 0.0, 'intercept': 0.0}
        
        # Get non-zero samples and standardize
        nonzero_activations = sampler.get_samples()
        standardized = (nonzero_activations - np.mean(nonzero_activations)) / np.std(nonzero_activations)
        
        # Generate Q-Q plot data
        n_quantiles = min(self.config.qq_quantiles, len(standardized))
        quantile_points = np.linspace(0.01, 0.99, n_quantiles)
        
        theoretical_quantiles = stats.norm.ppf(quantile_points)
        sample_quantiles = np.quantile(standardized, quantile_points)
        
        # Compute correlation (measure of linearity)
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        
        # Fit line to get slope and intercept
        slope, intercept, _, _, _ = stats.linregress(theoretical_quantiles, sample_quantiles)
        
        if save_plot:
            plt.figure(figsize=(8, 6))
            plt.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=20)
            plt.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, 
                    'r-', label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
            plt.plot([-3, 3], [-3, 3], 'k--', alpha=0.5, label='Perfect Normal')
            plt.xlabel('Theoretical Quantiles (Standard Normal)')
            plt.ylabel('Sample Quantiles (Standardized)')
            plt.title(f'Q-Q Plot: Feature {feature_idx}\nCorrelation = {correlation:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        
        return {
            'correlation': float(correlation),
            'slope': float(slope),
            'intercept': float(intercept),
        }
    
    def kolmogorov_smirnov_test(self, feature_idx: int) -> Dict[str, float]:
        """Perform Kolmogorov-Smirnov test against standard normal distribution."""
        sampler = self.feature_samplers[feature_idx]
        
        if sampler.size() < self.config.min_samples_for_analysis:
            self.logger.warning(f"Feature {feature_idx}: Too few samples for KS test")
            return {'statistic': 1.0, 'pvalue': 0.0}
        
        # Get samples and standardize
        nonzero_activations = sampler.get_samples()
        standardized = (nonzero_activations - np.mean(nonzero_activations)) / np.std(nonzero_activations)
        
        # Perform KS test against standard normal
        ks_statistic, ks_pvalue = stats.kstest(standardized, 'norm')
        
        return {
            'statistic': float(ks_statistic),
            'pvalue': float(ks_pvalue),
        }
    
    def wasserstein_analysis(self, feature_idx: int) -> float:
        """Compute Wasserstein distance between activation distribution and standard normal."""
        sampler = self.feature_samplers[feature_idx]
        
        if sampler.size() < self.config.min_samples_for_analysis:
            return float('inf')
        
        # Get samples and standardize
        nonzero_activations = sampler.get_samples()
        standardized = (nonzero_activations - np.mean(nonzero_activations)) / np.std(nonzero_activations)
        
        # Generate samples from standard normal with same size
        normal_samples = np.random.normal(0, 1, len(standardized))
        
        # Compute Wasserstein distance
        distance = wasserstein_distance(standardized, normal_samples)
        
        return float(distance)
    
    def analyze_single_feature(self, feature_idx: int) -> GaussianityResults:
        """Perform complete analysis on a single feature."""
        stats = self.feature_stats[feature_idx]
        sampler = self.feature_samplers[feature_idx]
        
        # Get basic statistics
        stats_dict = stats.to_dict()
        
        # Perform Gaussianity analyses
        qq_results = self.qq_analysis(feature_idx, save_plot=self.config.save_individual_plots)
        ks_results = self.kolmogorov_smirnov_test(feature_idx)
        wasserstein_dist = self.wasserstein_analysis(feature_idx)
        
        return GaussianityResults(
            feature_idx=feature_idx,
            n_samples=stats_dict['total_count'],
            n_nonzero_samples=stats_dict['count'],
            mean=stats_dict['mean'],
            std=stats_dict['std'],
            skewness=stats_dict['skewness'],
            kurtosis=stats_dict['kurtosis'],
            zero_fraction=stats_dict['zero_fraction'],
            qq_correlation=qq_results['correlation'],
            qq_slope=qq_results['slope'],
            qq_intercept=qq_results['intercept'],
            ks_statistic=ks_results['statistic'],
            ks_pvalue=ks_results['pvalue'],
            wasserstein_dist=wasserstein_dist,
        )
    
    def analyze_all_features(self) -> List[GaussianityResults]:
        """Analyze all features and return results."""
        self.logger.info(f"Analyzing {len(self.feature_indices)} features...")
        
        results = []
        for feature_idx in tqdm(self.feature_indices, desc="Analyzing features"):
            try:
                result = self.analyze_single_feature(feature_idx)
                results.append(result)
                
                # Log progress
                if result.n_nonzero_samples >= self.config.min_samples_for_analysis:
                    self.logger.debug(f"Feature {feature_idx}: {result.n_nonzero_samples} samples, "
                                    f"QQ-corr={result.qq_correlation:.3f}, KS-p={result.ks_pvalue:.3f}")
                else:
                    self.logger.warning(f"Feature {feature_idx}: Insufficient samples ({result.n_nonzero_samples})")
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature_idx}: {e}")
                continue
        
        return results
    
    def create_summary_plots(self, results: List[GaussianityResults], output_dir: Path) -> None:
        """Create comprehensive summary visualizations."""
        if not self.config.create_summary_plots:
            return
            
        self.logger.info("Creating summary plots...")
        
        # Filter results with sufficient data
        valid_results = [r for r in results if r.n_nonzero_samples >= self.config.min_samples_for_analysis]
        
        if len(valid_results) == 0:
            self.logger.warning("No features with sufficient data for summary plots")
            return
        
        # Convert to DataFrame for easy plotting
        df = pd.DataFrame([r.to_dict() for r in valid_results])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Online Gaussianity Analysis Summary\n{len(valid_results)} features, {self.get_model_name()}', 
                    fontsize=16, y=0.98)
        
        # Q-Q Correlation distribution
        axes[0, 0].hist(df['qq_correlation'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['qq_correlation'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["qq_correlation"].mean():.3f}')
        axes[0, 0].set_xlabel('Q-Q Correlation')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Q-Q Correlation Distribution\n(Higher = More Gaussian)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # KS p-value distribution
        axes[0, 1].hist(df['ks_pvalue'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
        axes[0, 1].set_xlabel('KS Test p-value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('KS Test p-values\n(Higher = More Gaussian)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Wasserstein distance distribution
        finite_wasserstein = df[np.isfinite(df['wasserstein_dist'])]
        if len(finite_wasserstein) > 0:
            axes[0, 2].hist(finite_wasserstein['wasserstein_dist'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(finite_wasserstein['wasserstein_dist'].mean(), color='red', linestyle='--',
                              label=f'Mean: {finite_wasserstein["wasserstein_dist"].mean():.3f}')
            axes[0, 2].set_xlabel('Wasserstein Distance')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].set_title('Wasserstein Distance Distribution\n(Lower = More Gaussian)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Skewness vs Kurtosis
        scatter = axes[1, 0].scatter(df['skewness'], df['kurtosis'], alpha=0.6, c=df['qq_correlation'], 
                                    cmap='viridis', s=30)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)  # Normal kurtosis
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)  # Normal skewness
        axes[1, 0].set_xlabel('Skewness')
        axes[1, 0].set_ylabel('Kurtosis')
        axes[1, 0].set_title('Skewness vs Kurtosis\n(Colored by Q-Q Correlation)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Q-Q Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample size vs Gaussianity
        axes[1, 1].scatter(df['n_nonzero_samples'], df['qq_correlation'], alpha=0.6, s=30, label='Q-Q Correlation')
        axes[1, 1].scatter(df['n_nonzero_samples'], df['ks_pvalue'], alpha=0.6, s=30, label='KS p-value')
        axes[1, 1].set_xlabel('Non-zero Sample Count')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('Sample Size vs Gaussianity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
        
        # Zero fraction vs Gaussianity metrics
        axes[1, 2].scatter(df['zero_fraction'], df['qq_correlation'], alpha=0.6, s=30, label='Q-Q Correlation')
        axes[1, 2].scatter(df['zero_fraction'], df['ks_pvalue'], alpha=0.6, s=30, label='KS p-value')
        axes[1, 2].set_xlabel('Zero Fraction')
        axes[1, 2].set_ylabel('Metric Value')
        axes[1, 2].set_title('Sparsity vs Gaussianity')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_file = output_dir / f"gaussianity_summary_{self.get_model_name().replace('/', '_')}.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plot saved to {summary_file}")
    
    def save_results(self, results: List[GaussianityResults], output_dir: Path) -> None:
        """Save results as CSV and JSON."""
        model_name = self.get_model_name()
        self.logger.info("Saving results...")
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame([r.to_dict() for r in results])
        csv_file = output_dir / f"gaussianity_results_{model_name.replace('/', '_')}.csv"
        df.to_csv(csv_file, index=False)
        
        # Filter valid results for summary statistics
        valid_results = df[df['n_nonzero_samples'] >= self.config.min_samples_for_analysis]
        
        # Save as JSON with metadata
        results_dict = {
            'model_info': self.model_info,
            'processing_info': {
                'target_samples': self.config.target_total_samples,
                'samples_processed': self.samples_processed,
                'max_samples_per_feature': self.config.max_samples_per_feature,
                'min_samples_for_analysis': self.config.min_samples_for_analysis,
                'seed': self.seed,
            },
            'analysis_summary': {
                'n_features_analyzed': len(results),
                'n_features_with_sufficient_data': len(valid_results),
                'mean_qq_correlation': float(valid_results['qq_correlation'].mean()) if len(valid_results) > 0 else 0.0,
                'std_qq_correlation': float(valid_results['qq_correlation'].std()) if len(valid_results) > 0 else 0.0,
                'mean_ks_pvalue': float(valid_results['ks_pvalue'].mean()) if len(valid_results) > 0 else 0.0,
                'fraction_ks_significant': float((valid_results['ks_pvalue'] < 0.05).mean()) if len(valid_results) > 0 else 0.0,
                'mean_wasserstein_dist': float(valid_results[np.isfinite(valid_results['wasserstein_dist'])]['wasserstein_dist'].mean()) if len(valid_results) > 0 else 0.0,
            },
            'feature_results': [r.to_dict() for r in results]
        }
        
        json_file = output_dir / f"gaussianity_results_{model_name.replace('/', '_')}.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {csv_file} and {json_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ONLINE GAUSSIANITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Features analyzed: {len(results)}")
        print(f"Features with sufficient data: {len(valid_results)}")
        print(f"Samples processed: {self.samples_processed:,}")
        print(f"Max samples per feature: {self.config.max_samples_per_feature:,}")
        
        if len(valid_results) > 0:
            print(f"\nQ-Q Correlation (higher = more Gaussian):")
            print(f"  Mean: {valid_results['qq_correlation'].mean():.4f} ± {valid_results['qq_correlation'].std():.4f}")
            print(f"  Range: {valid_results['qq_correlation'].min():.4f} to {valid_results['qq_correlation'].max():.4f}")
            print(f"\nKS Test p-values (higher = more Gaussian):")
            print(f"  Mean: {valid_results['ks_pvalue'].mean():.4f}")
            print(f"  Fraction significant (p < 0.05): {(valid_results['ks_pvalue'] < 0.05).mean():.2%}")
            
            finite_wasserstein = valid_results[np.isfinite(valid_results['wasserstein_dist'])]
            if len(finite_wasserstein) > 0:
                print(f"\nWasserstein Distance (lower = more Gaussian):")
                print(f"  Mean: {finite_wasserstein['wasserstein_dist'].mean():.4f} ± {finite_wasserstein['wasserstein_dist'].std():.4f}")
        print("="*60)
    
    def run(self, output_dir: Path) -> List[GaussianityResults]:
        """Run the complete online analysis pipeline."""
        # Create model-specific output directory
        model_name = self.get_model_name()
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Output directory: {model_output_dir}")
        
        # Run pipeline
        self.load_model_and_dictionary()
        self.validate_features()
        self.create_buffer()
        self.collect_activations_online()
        
        # Analyze features
        results = self.analyze_all_features()
        
        # Create visualizations and save results
        if self.config.create_summary_plots:
            self.create_summary_plots(results, model_output_dir)
        
        self.save_results(results, model_output_dir)
        
        return results


def parse_feature_spec(feature_spec: str) -> List[int]:
    """
    Parse feature specification string into list of feature indices.
    
    Examples:
        "0:100" -> [0, 1, 2, ..., 99]
        "[2,5,3]" -> [2, 5, 3]
        "42" -> [42]
        "0:8192:10" -> [0, 10, 20, ..., 8190]
    """
    feature_spec = feature_spec.strip()
    
    # Handle list format: [2,5,3]
    if feature_spec.startswith('[') and feature_spec.endswith(']'):
        return [int(x.strip()) for x in feature_spec[1:-1].split(',')]
    
    # Handle range format: 0:100 or 0:8192:10
    if ':' in feature_spec:
        parts = feature_spec.split(':')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end))
        elif len(parts) == 3:
            start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
            return list(range(start, end, step))
        else:
            raise ValueError(f"Invalid range format: {feature_spec}")
    
    # Handle single feature: 42
    try:
        return [int(feature_spec)]
    except ValueError:
        raise ValueError(f"Invalid feature specification: {feature_spec}")


def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'online_gaussianity_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Online Gaussianity analysis for VSAE features")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model directory (containing ae.pt and config.json)"
    )
    parser.add_argument(
        "--features", 
        type=str, 
        default="0:1000",
        help="Features to analyze. Examples: '0:100', '[2,5,3]', '42', '0:8192:50'"
    )
    parser.add_argument(
        "--n-samples", 
        type=int, 
        default=1_000_000,
        help="Total number of samples to process"
    )
    parser.add_argument(
        "--max-samples-per-feature",
        type=int,
        default=50000,
        help="Maximum samples to store per feature (reservoir sampling)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./online_gaussianity_analysis",
        help="Base output directory"
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
        help="Batch size for data processing"
    )
    parser.add_argument(
        "--no-summary-plots",
        action="store_true",
        help="Skip summary plot generation"
    )
    parser.add_argument(
        "--save-individual-plots",
        action="store_true",
        help="Save individual Q-Q plots for each feature"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for deterministic sampling"
    )
    
    args = parser.parse_args()
    
    # Parse feature specification
    try:
        feature_indices = parse_feature_spec(args.features)
    except ValueError as e:
        print(f"Error parsing features: {e}")
        return
    
    # Create output directory and setup logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting online VSAE Gaussianity analysis")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Features: {len(feature_indices)} features")
    logger.info(f"Total samples: {args.n_samples:,}")
    logger.info(f"Max samples per feature: {args.max_samples_per_feature:,}")
    logger.info(f"Output: {output_dir}")
    
    # Create configuration
    config = OnlineGaussianityConfig(
        max_samples_per_feature=args.max_samples_per_feature,
        target_total_samples=args.n_samples,
        save_individual_plots=args.save_individual_plots,
        create_summary_plots=not args.no_summary_plots,
    )
    
    # Run analysis
    analyzer = OnlineGaussianityAnalyzer(
        model_path=args.model_path,
        feature_indices=feature_indices,
        config=config,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    try:
        results = analyzer.run(output_dir)
        logger.info("Online Gaussianity analysis completed successfully!")
        logger.info(f"Analyzed {len(results)} features")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Usage examples:
# python online_gaussianity.py --model-path ./experiments/vsae_kl_100/trainer_0 --features "0:200" --n-samples 2000000
# python online_gaussianity.py --model-path ./experiments/vsae_kl_1/trainer_0 --features "[1,5,10,50,100]" --max-samples-per-feature 100000
# python online_gaussianity.py --model-path "C:\path\to\experiment\trainer_0" --features "42" --save-individual-plots

# Output structure:
# online_gaussianity_analysis/
#   ├── model_name/
#   │   ├── gaussianity_summary_model_name.png
#   │   ├── gaussianity_results_model_name.csv
#   │   └── gaussianity_results_model_name.json
#   └── logs/
#       └── online_gaussianity_analysis.log
