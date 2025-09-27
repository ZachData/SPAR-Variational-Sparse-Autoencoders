"""
Feature Degeneracy Analysis for VSAE Feature Activations.

This script identifies degenerate/redundant features in VSAE models using multiple methods:
- Correlation-based detection via Pearson correlations
- Weight similarity analysis via cosine similarity 
- Mutual information estimation for non-linear dependencies
- Hash-based detection via locality-sensitive hashing

Key Features:
- Online processing for memory efficiency with 4M samples by default
- Analyzes ALL available features by default  
- Sparse matrix operations for large feature sets
- Multiple detection methods with configurable thresholds
- Comprehensive reporting and visualization

Note: For VSAETopK models, decoder weights have shape [activation_dim, dict_size]
where dict_size is the number of features to analyze.
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
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, find
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
import hashlib
from collections import defaultdict

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class FeatureDegeneracyConfig: # Mutual information and Hash-based thresholds
    """Configuration for feature degeneracy analysis."""
    max_samples_per_feature: int = 70000      # Reservoir sampling limit
    target_total_samples: int = 1_400_000     # Total samples to process
    min_samples_for_analysis: int = 1000      # Minimum samples needed for analysis
    
    # Correlation thresholds
    correlation_threshold: float = 0.5         # Pearson correlation threshold
    use_sparse_correlation: bool = True        # Use sparse matrices for efficiency
    
    # Weight similarity thresholds  
    weight_similarity_threshold: float = 0.5   # Cosine similarity threshold for weights
    
    # Mutual information settings
    mi_k_neighbors: int = 3                    # k for k-NN MI estimation
    mi_sample_size: int = 10000               # Sample size for MI estimation
    mi_threshold: float = 0.5                 # MI threshold for degeneracy
    
    # Hash-based detection
    hash_bins: int = 100                      # Number of bins for LSH
    hash_threshold: float = 0.3              # Jaccard similarity threshold
    
    save_individual_plots: bool = False       # Save individual analysis plots
    create_summary_plots: bool = True         # Create summary visualizations


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
                j = np.random.randint(0, self.count)
                if j < self.max_size:
                    self.reservoir[j] = float(sample)
    
    def get_samples(self) -> np.ndarray:
        """Get current reservoir samples."""
        return np.array(self.reservoir)
    
    def size(self) -> int:
        """Get current number of samples in reservoir."""
        return len(self.reservoir)


class LocalitySensitiveHash:
    """LSH for detecting similar activation patterns."""
    
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins
        self.feature_hashes = {}
    
    def hash_activations(self, activations: np.ndarray) -> str:
        """Create hash of activation pattern."""
        # Normalize and bin activations
        if len(activations) == 0:
            return "empty"
        
        normalized = (activations - np.min(activations)) / (np.max(activations) - np.min(activations) + 1e-8)
        binned = np.digitize(normalized, np.linspace(0, 1, self.num_bins))
        
        # Create hash from binned pattern
        hash_input = ''.join(map(str, binned))
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_feature(self, feature_idx: int, activations: np.ndarray) -> None:
        """Add feature's activation hash."""
        self.feature_hashes[feature_idx] = self.hash_activations(activations)
    
    def find_similar_features(self, threshold: float = 0.95) -> List[Tuple[int, int]]:
        """Find features with similar hashes."""
        hash_groups = defaultdict(list)
        
        # Group features by hash
        for feature_idx, hash_val in self.feature_hashes.items():
            hash_groups[hash_val].append(feature_idx)
        
        # Find groups with multiple features
        similar_pairs = []
        for hash_val, features in hash_groups.items():
            if len(features) > 1:
                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        similar_pairs.append((features[i], features[j]))
        
        return similar_pairs


@dataclass
class DegeneracyResults:
    """Container for feature degeneracy analysis results."""
    feature_idx: int
    n_samples: int
    n_nonzero_samples: int
    
    # Correlation-based results
    high_correlation_features: List[Tuple[int, float]]
    max_correlation: float
    avg_correlation: float
    
    # Weight similarity results
    high_weight_similarity_features: List[Tuple[int, float]]
    max_weight_similarity: float
    
    # Mutual information results
    high_mi_features: List[Tuple[int, float]]
    max_mi: float
    avg_mi: float
    
    # Hash-based results
    hash_similar_features: List[int]
    activation_hash: str
    
    # Summary flags
    is_degenerate_correlation: bool
    is_degenerate_weights: bool
    is_degenerate_mi: bool
    is_degenerate_hash: bool
    is_degenerate_overall: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-safe types."""
        def convert_value(value):
            """Convert numpy types to JSON-safe Python types."""
            if isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(convert_value(v) for v in value)
            return value
        
        result = asdict(self)
        return {key: convert_value(value) for key, value in result.items()}


class OnlineFeatureDegeneracyAnalyzer:
    """Main analyzer for detecting degenerate features."""
    
    def __init__(self, config: FeatureDegeneracyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model and data
        self.model = None
        self.dictionary = None
        self.buffer = None
        self.hook_name = None
        self.feature_indices = []
        
        # Data collection (will be initialized after feature validation)
        self.feature_samplers = {}
        self.samples_processed = 0
        
        # Weight matrices for similarity analysis
        self.decoder_weights = None
        self.encoder_weights = None
        
        # LSH for hash-based detection
        self.lsh = LocalitySensitiveHash(self.config.hash_bins)
    
    def _initialize_samplers(self) -> None:
        """Initialize reservoir samplers after feature validation."""
        self.feature_samplers = {}
        for feature_idx in self.feature_indices:
            self.feature_samplers[feature_idx] = ReservoirSampler(self.config.max_samples_per_feature)
    
    def get_model_name(self) -> str:
        """Extract model name from path."""
        return self.model_path.name if hasattr(self, 'model_path') else "unknown_model"
    
    def load_model_and_dictionary(self) -> None:
        """Load model and dictionary from path."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dictionary and config
        self.dictionary, config = load_dictionary(str(self.model_path), device=device)
        
        # Extract model information from config  
        model_name = config["trainer"].get("lm_name", "gelu-1l")
        hook_name = config["trainer"].get("submodule_name", "blocks.0.mlp.hook_post")
        
        self.logger.info(f"Model: {model_name}, Hook: {hook_name}")
        
        # Load the transformer model
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        
        # Store hook name and metadata
        self.hook_name = hook_name
        
        # Extract weight matrices
        if hasattr(self.dictionary, 'decoder'):
            self.decoder_weights = self.dictionary.decoder.weight.detach().cpu().numpy()
        if hasattr(self.dictionary, 'encoder'):  
            self.encoder_weights = self.dictionary.encoder.weight.detach().cpu().numpy()
        
        self.logger.info(f"Dictionary loaded: {self.decoder_weights.shape if self.decoder_weights is not None else 'No decoder'} (activation_dim={self.decoder_weights.shape[0] if self.decoder_weights is not None else 'N/A'}, dict_size={self.decoder_weights.shape[1] if self.decoder_weights is not None else 'N/A'})")
        self.logger.info(f"Model loaded: {model_name} on {device}")
    
    def validate_features(self) -> None:
        """Validate feature indices against dictionary size."""
        # For VSAETopK, decoder weights shape is [activation_dim, dict_size]
        # So we need shape[1] to get the actual number of features (dict_size)
        if self.decoder_weights is not None:
            max_features = self.decoder_weights.shape[1]  # dict_size dimension
        else:
            max_features = 0
        
        # If features is "all" or empty, analyze all available features
        if not self.feature_indices or self.feature_indices == ["all"]:
            self.feature_indices = list(range(max_features))
            self.logger.info(f"Analyzing ALL {max_features} features")
        else:
            valid_features = [f for f in self.feature_indices if 0 <= f < max_features]
            invalid_features = [f for f in self.feature_indices if f not in valid_features]
            
            if invalid_features:
                self.logger.warning(f"Invalid features (will be skipped): {invalid_features}")
            
            self.feature_indices = valid_features
            self.logger.info(f"Analyzing {len(self.feature_indices)} valid features")
    
    def create_buffer(self) -> None:
        """Create activation buffer for data collection."""
        self.logger.info("Creating activation buffer...")
        
        # Create data generator
        data_gen = hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b",
            split="train",
            return_tokens=True
        )
        
        # Create buffer
        n_ctxs = min(3000, max(500, self.config.target_total_samples // 128))
        
        self.buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=self.model,
            hook_name=self.hook_name,
            d_submodule=self.dictionary.activation_dim,
            n_ctxs=n_ctxs,
            ctx_len=128,
            refresh_batch_size=32,
            out_batch_size=min(256, self.config.target_total_samples // 20),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Initialize reservoir samplers now that features are validated
        self._initialize_samplers()
    
    def collect_activations_online(self) -> None:
        """Collect activations using online processing."""
        self.logger.info(f"Collecting activations for {len(self.feature_indices)} features...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.buffer, desc="Processing batches")):
                if self.samples_processed >= self.config.target_total_samples:
                    break
                
                # Get activations for current batch
                activations = batch.cpu().numpy()
                
                # Process each feature
                for feature_idx in self.feature_indices:
                    if feature_idx < activations.shape[-1]:
                        feature_activations = activations[..., feature_idx].flatten()
                        nonzero_mask = feature_activations != 0
                        nonzero_activations = feature_activations[nonzero_mask]
                        
                        if len(nonzero_activations) > 0:
                            self.feature_samplers[feature_idx].add_samples(nonzero_activations)
                
                self.samples_processed += activations.shape[0] * activations.shape[1]  # batch_size * seq_len
                
                if batch_idx % 100 == 0:
                    self.logger.debug(f"Processed {self.samples_processed:,} samples")
        
        self.logger.info(f"Collection complete: {self.samples_processed:,} samples processed")
    
    def correlation_analysis(self, feature_indices: List[int]) -> Dict[int, List[Tuple[int, float]]]:
        """Compute pairwise correlations between features."""
        self.logger.info("Computing pairwise correlations...")
        
        # Collect activation matrices
        activation_matrix = []
        valid_features = []
        
        for feature_idx in feature_indices:
            sampler = self.feature_samplers[feature_idx]
            if sampler.size() >= self.config.min_samples_for_analysis:
                activations = sampler.get_samples()
                activation_matrix.append(activations[:min(len(activations), 10000)])  # Limit for efficiency
                valid_features.append(feature_idx)
        
        if len(valid_features) < 2:
            return {}
        
        # Pad sequences to same length
        max_len = max(len(acts) for acts in activation_matrix)
        padded_matrix = np.zeros((len(activation_matrix), max_len))
        
        for i, acts in enumerate(activation_matrix):
            padded_matrix[i, :len(acts)] = acts
        
        # Compute correlation matrix
        if self.config.use_sparse_correlation:
            # Use sparse operations for efficiency
            sparse_matrix = csr_matrix(padded_matrix)
            correlations = np.corrcoef(sparse_matrix.toarray())
        else:
            correlations = np.corrcoef(padded_matrix)
        
        # Extract high correlations
        high_correlations = {}
        for i, feature_i in enumerate(valid_features):
            high_corr_features = []
            for j, feature_j in enumerate(valid_features):
                if i != j and abs(correlations[i, j]) > self.config.correlation_threshold:
                    high_corr_features.append((feature_j, correlations[i, j]))
            high_correlations[feature_i] = high_corr_features
        
        return high_correlations
    
    def weight_similarity_analysis(self, feature_indices: List[int]) -> Dict[int, List[Tuple[int, float]]]:
        """Analyze decoder weight similarities."""
        self.logger.info("Computing weight similarities...")
        
        if self.decoder_weights is None:
            return {}
        
        high_similarities = {}
        
        # VSAETopK decoder weights shape: [activation_dim, dict_size]
        max_features = self.decoder_weights.shape[1]
        
        for feature_i in feature_indices:
            if feature_i >= max_features:
                continue
                
            high_sim_features = []
            weight_i = self.decoder_weights[:, feature_i]  # Get column for feature i
            
            for feature_j in feature_indices:
                if feature_i != feature_j and feature_j < max_features:
                    weight_j = self.decoder_weights[:, feature_j]  # Get column for feature j
                    
                    # Compute cosine similarity
                    similarity = 1 - cosine(weight_i, weight_j)
                    
                    if similarity > self.config.weight_similarity_threshold:
                        high_sim_features.append((feature_j, similarity))
            
            high_similarities[feature_i] = high_sim_features
        
        return high_similarities
    
    def mutual_information_analysis(self, feature_indices: List[int]) -> Dict[int, List[Tuple[int, float]]]:
        """Estimate mutual information between features."""
        self.logger.info("Computing mutual information...")
        
        # Sample subset for efficiency
        sample_features = feature_indices[:min(len(feature_indices), 50)]  # Limit for computational efficiency
        
        # Collect activation data
        activation_data = {}
        for feature_idx in sample_features:
            sampler = self.feature_samplers[feature_idx]
            if sampler.size() >= self.config.min_samples_for_analysis:
                activations = sampler.get_samples()
                # Sample subset for MI estimation
                sample_size = min(len(activations), self.config.mi_sample_size)
                sample_indices = np.random.choice(len(activations), sample_size, replace=False)
                activation_data[feature_idx] = activations[sample_indices]
        
        high_mi_features = {}
        
        for feature_i in activation_data:
            high_mi_list = []
            
            for feature_j in activation_data:
                if feature_i != feature_j:
                    # Align sample sizes
                    min_size = min(len(activation_data[feature_i]), len(activation_data[feature_j]))
                    acts_i = activation_data[feature_i][:min_size]
                    acts_j = activation_data[feature_j][:min_size]
                    
                    # Estimate mutual information using k-NN
                    try:
                        mi_score = mutual_info_regression(
                            acts_i.reshape(-1, 1), 
                            acts_j, 
                            n_neighbors=self.config.mi_k_neighbors
                        )[0]
                        
                        if mi_score > self.config.mi_threshold:
                            high_mi_list.append((feature_j, mi_score))
                    except Exception as e:
                        self.logger.debug(f"MI estimation failed for features {feature_i}-{feature_j}: {e}")
                        continue
            
            high_mi_features[feature_i] = high_mi_list
        
        return high_mi_features
    
    def hash_based_analysis(self, feature_indices: List[int]) -> Dict[int, List[int]]:
        """Detect similar features using locality-sensitive hashing."""
        self.logger.info("Computing LSH similarities...")
        
        # Add all features to LSH
        for feature_idx in feature_indices:
            sampler = self.feature_samplers[feature_idx]
            if sampler.size() >= self.config.min_samples_for_analysis:
                activations = sampler.get_samples()
                self.lsh.add_feature(feature_idx, activations)
        
        # Find similar feature pairs
        similar_pairs = self.lsh.find_similar_features(self.config.hash_threshold)
        
        # Group by feature
        hash_similarities = defaultdict(list)
        for feature_i, feature_j in similar_pairs:
            hash_similarities[feature_i].append(feature_j)
            hash_similarities[feature_j].append(feature_i)
        
        return dict(hash_similarities)
    
    def analyze_single_feature(self, feature_idx: int, correlation_results: Dict, 
                             weight_results: Dict, mi_results: Dict, hash_results: Dict) -> DegeneracyResults:
        """Perform complete degeneracy analysis on a single feature."""
        sampler = self.feature_samplers[feature_idx]
        
        # Get correlation results
        high_corr = correlation_results.get(feature_idx, [])
        max_corr = max([abs(corr) for _, corr in high_corr], default=0.0)
        avg_corr = np.mean([abs(corr) for _, corr in high_corr]) if high_corr else 0.0
        
        # Get weight similarity results
        high_weight_sim = weight_results.get(feature_idx, [])
        max_weight_sim = max([sim for _, sim in high_weight_sim], default=0.0)
        
        # Get MI results
        high_mi = mi_results.get(feature_idx, [])
        max_mi = max([mi for _, mi in high_mi], default=0.0)
        avg_mi = np.mean([mi for _, mi in high_mi]) if high_mi else 0.0
        
        # Get hash results
        hash_similar = hash_results.get(feature_idx, [])
        activation_hash = self.lsh.feature_hashes.get(feature_idx, "")
        
        # Determine degeneracy flags
        is_degenerate_corr = max_corr > self.config.correlation_threshold
        is_degenerate_weights = max_weight_sim > self.config.weight_similarity_threshold
        is_degenerate_mi = max_mi > self.config.mi_threshold
        is_degenerate_hash = len(hash_similar) > 0
        
        is_degenerate_overall = any([
            is_degenerate_corr, is_degenerate_weights, 
            is_degenerate_mi, is_degenerate_hash
        ])
        
        return DegeneracyResults(
            feature_idx=feature_idx,
            n_samples=sampler.count,
            n_nonzero_samples=sampler.size(),
            high_correlation_features=high_corr,
            max_correlation=max_corr,
            avg_correlation=avg_corr,
            high_weight_similarity_features=high_weight_sim,
            max_weight_similarity=max_weight_sim,
            high_mi_features=high_mi,
            max_mi=max_mi,
            avg_mi=avg_mi,
            hash_similar_features=hash_similar,
            activation_hash=activation_hash,
            is_degenerate_correlation=is_degenerate_corr,
            is_degenerate_weights=is_degenerate_weights,
            is_degenerate_mi=is_degenerate_mi,
            is_degenerate_hash=is_degenerate_hash,
            is_degenerate_overall=is_degenerate_overall,
        )
    
    def analyze_all_features(self) -> List[DegeneracyResults]:
        """Analyze all features for degeneracy."""
        self.logger.info(f"Running degeneracy analysis on {len(self.feature_indices)} features...")
        
        # Run all analysis methods
        correlation_results = self.correlation_analysis(self.feature_indices)
        weight_results = self.weight_similarity_analysis(self.feature_indices)
        mi_results = self.mutual_information_analysis(self.feature_indices)
        hash_results = self.hash_based_analysis(self.feature_indices)
        
        # Analyze each feature
        results = []
        for feature_idx in tqdm(self.feature_indices, desc="Analyzing features"):
            try:
                result = self.analyze_single_feature(
                    feature_idx, correlation_results, weight_results, mi_results, hash_results
                )
                results.append(result)
                
                if result.is_degenerate_overall:
                    self.logger.debug(f"Feature {feature_idx}: DEGENERATE - "
                                    f"Corr={result.is_degenerate_correlation}, "
                                    f"Weight={result.is_degenerate_weights}, "
                                    f"MI={result.is_degenerate_mi}, "
                                    f"Hash={result.is_degenerate_hash}")
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature_idx}: {e}")
                continue
        
        return results
    
    def create_summary_plots(self, results: List[DegeneracyResults], output_dir: Path) -> None:
        """Create comprehensive summary visualizations."""
        if not self.config.create_summary_plots:
            return
        
        self.logger.info("Creating summary plots...")
        
        valid_results = [r for r in results if r.n_nonzero_samples >= self.config.min_samples_for_analysis]
        
        if len(valid_results) == 0:
            self.logger.warning("No features with sufficient data for summary plots")
            return
        
        df = pd.DataFrame([r.to_dict() for r in valid_results])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Feature Degeneracy Analysis Summary\n{len(valid_results)} features, {self.get_model_name()}', 
                    fontsize=16, y=0.98)
        
        # Max correlation distribution
        axes[0, 0].hist(df['max_correlation'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.config.correlation_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {self.config.correlation_threshold}')
        axes[0, 0].set_xlabel('Max Correlation')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Maximum Correlation Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weight similarity distribution
        axes[0, 1].hist(df['max_weight_similarity'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.config.weight_similarity_threshold, color='red', linestyle='--',
                          label=f'Threshold: {self.config.weight_similarity_threshold}')
        axes[0, 1].set_xlabel('Max Weight Similarity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Maximum Weight Similarity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mutual information distribution
        axes[0, 2].hist(df['max_mi'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(self.config.mi_threshold, color='red', linestyle='--',
                          label=f'Threshold: {self.config.mi_threshold}')
        axes[0, 2].set_xlabel('Max Mutual Information')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Maximum Mutual Information Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Degeneracy summary
        degeneracy_counts = {
            'Correlation': df['is_degenerate_correlation'].sum(),
            'Weight Similarity': df['is_degenerate_weights'].sum(),
            'Mutual Information': df['is_degenerate_mi'].sum(),
            'Hash-based': df['is_degenerate_hash'].sum(),
            'Overall': df['is_degenerate_overall'].sum()
        }
        
        axes[1, 0].bar(degeneracy_counts.keys(), degeneracy_counts.values())
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Degenerate Features by Method')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Degeneracy overlap
        methods = ['is_degenerate_correlation', 'is_degenerate_weights', 'is_degenerate_mi', 'is_degenerate_hash']
        overlap_matrix = np.zeros((len(methods), len(methods)))
        
        for i, method_i in enumerate(methods):
            for j, method_j in enumerate(methods):
                overlap = (df[method_i] & df[method_j]).sum()
                overlap_matrix[i, j] = overlap
        
        im = axes[1, 1].imshow(overlap_matrix, cmap='Blues')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_yticks(range(len(methods)))
        axes[1, 1].set_xticklabels(['Correlation', 'Weight', 'MI', 'Hash'], rotation=45)
        axes[1, 1].set_yticklabels(['Correlation', 'Weight', 'MI', 'Hash'])
        axes[1, 1].set_title('Degeneracy Method Overlap')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(methods)):
                axes[1, 1].text(j, i, f'{int(overlap_matrix[i, j])}', 
                               ha="center", va="center", color="black")
        
        # Sample size distribution
        axes[1, 2].hist(df['n_nonzero_samples'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Non-zero Samples')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Sample Size Distribution')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        model_name = self.get_model_name()
        plot_file = output_dir / f"degeneracy_summary_{model_name.replace('/', '_')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plot saved to {plot_file}")
    
    def save_results(self, results: List[DegeneracyResults], output_dir: Path) -> None:
        """Save analysis results to files."""
        model_name = self.get_model_name()
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to JSON-safe Python types."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            return obj
        
        # Save CSV
        df = pd.DataFrame([r.to_dict() for r in results])
        csv_file = output_dir / f"degeneracy_results_{model_name.replace('/', '_')}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save JSON with summary statistics
        valid_results = df[df['n_nonzero_samples'] >= self.config.min_samples_for_analysis]
        
        results_dict = {
            'config': asdict(self.config),
            'summary': {
                'total_features': len(results),
                'valid_features': len(valid_results),
                'samples_processed': self.samples_processed,
                'degenerate_correlation': int(valid_results['is_degenerate_correlation'].sum()) if len(valid_results) > 0 else 0,
                'degenerate_weights': int(valid_results['is_degenerate_weights'].sum()) if len(valid_results) > 0 else 0,
                'degenerate_mi': int(valid_results['is_degenerate_mi'].sum()) if len(valid_results) > 0 else 0,
                'degenerate_hash': int(valid_results['is_degenerate_hash'].sum()) if len(valid_results) > 0 else 0,
                'degenerate_overall': int(valid_results['is_degenerate_overall'].sum()) if len(valid_results) > 0 else 0,
                'mean_max_correlation': float(valid_results['max_correlation'].mean()) if len(valid_results) > 0 else 0.0,
                'mean_max_weight_similarity': float(valid_results['max_weight_similarity'].mean()) if len(valid_results) > 0 else 0.0,
                'mean_max_mi': float(valid_results['max_mi'].mean()) if len(valid_results) > 0 else 0.0,
            },
            'feature_results': [r.to_dict() for r in results]
        }
        
        # Convert all numpy types to JSON-safe types
        results_dict = convert_numpy_types(results_dict)
        
        json_file = output_dir / f"degeneracy_results_{model_name.replace('/', '_')}.json"
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {csv_file} and {json_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("FEATURE DEGENERACY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Features analyzed: {len(results)}")
        print(f"Features with sufficient data: {len(valid_results)}")
        print(f"Samples processed: {self.samples_processed:,}")
        
        if len(valid_results) > 0:
            print(f"\nDegeneracy Detection Results:")
            print(f"  Correlation-based: {valid_results['is_degenerate_correlation'].sum()} features")
            print(f"  Weight similarity: {valid_results['is_degenerate_weights'].sum()} features")
            print(f"  Mutual information: {valid_results['is_degenerate_mi'].sum()} features")
            print(f"  Hash-based: {valid_results['is_degenerate_hash'].sum()} features")
            print(f"  Overall degenerate: {valid_results['is_degenerate_overall'].sum()} features ({valid_results['is_degenerate_overall'].mean():.1%})")
            
            print(f"\nAverage Similarity Scores:")
            print(f"  Max correlation: {valid_results['max_correlation'].mean():.3f} ± {valid_results['max_correlation'].std():.3f}")
            print(f"  Max weight similarity: {valid_results['max_weight_similarity'].mean():.3f} ± {valid_results['max_weight_similarity'].std():.3f}")
            print(f"  Max mutual information: {valid_results['max_mi'].mean():.3f} ± {valid_results['max_mi'].std():.3f}")
        print("="*60)
    
    def run(self, model_path: Path, feature_indices: List[int], output_dir: Path) -> List[DegeneracyResults]:
        """Run the complete degeneracy analysis pipeline."""
        self.model_path = model_path
        self.feature_indices = feature_indices
        
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
    """Parse feature specification string into list of feature indices."""
    feature_spec = feature_spec.strip()
    
    # Handle "all" case - return empty list to be filled later
    if feature_spec.lower() == "all":
        return []
    
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
            logging.FileHandler(log_dir / 'feature_degeneracy_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Feature degeneracy analysis for VSAE features")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model directory (containing ae.pt and config.json)"
    )
    parser.add_argument(
        "--features", 
        type=str, 
        default="all",
        help="Features to analyze. Examples: 'all', '0:100', '[1,5,10]', '42'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./feature_degeneracy_analysis",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4_000_000,
        help="Total number of samples to process (doubled again from 2M)"
    )
    parser.add_argument(
        "--max-samples-per-feature",
        type=int,
        default=400_000,
        help="Maximum samples per feature (reservoir sampling limit, doubled again)"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.1,
        help="Pearson correlation threshold for degeneracy detection"
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.1,
        help="Weight similarity threshold for degeneracy detection"
    )
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=0.1,
        help="Mutual information threshold for degeneracy detection"
    )
    parser.add_argument(
        "--save-individual-plots",
        action="store_true",
        help="Save individual feature analysis plots"
    )
    
    args = parser.parse_args()
    
    # Parse inputs
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    feature_indices = parse_feature_spec(args.features)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Create config
    config = FeatureDegeneracyConfig(
        target_total_samples=args.n_samples,
        max_samples_per_feature=args.max_samples_per_feature,
        correlation_threshold=args.correlation_threshold,
        weight_similarity_threshold=args.weight_threshold,
        mi_threshold=args.mi_threshold,
        save_individual_plots=args.save_individual_plots,
    )
    
    # Run analysis
    try:
        analyzer = OnlineFeatureDegeneracyAnalyzer(config)
        results = analyzer.run(model_path, feature_indices, output_dir)
        
        logger.info(f"Analysis completed successfully")
        logger.info(f"Analyzed {len(results)} features")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()


# Usage examples:
# python analyze_feature_degeneracy.py --model-path ./experiments/vsae_kl_100/trainer_0  # Analyzes ALL features with 4M samples
# python analyze_feature_degeneracy.py --model-path ./experiments/vsae_kl_1/trainer_0 --features "0:200" --n-samples 8000000
# python analyze_feature_degeneracy.py --model-path ./experiments/vsae_kl_1/trainer_0 --features "[1,5,10,50,100]" --correlation-threshold 0.85
# python analyze_feature_degeneracy.py --model-path "C:\path\to\experiment\trainer_0" --features "42" --save-individual-plots

# Output structure:
# feature_degeneracy_analysis/
#   ├── model_name/
#   │   ├── degeneracy_summary_model_name.png
#   │   ├── degeneracy_results_model_name.csv
#   │   └── degeneracy_results_model_name.json
#   └── logs/
#       └── feature_degeneracy_analysis.log