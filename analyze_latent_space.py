"""
Comprehensive analysis of VSAE latent spaces and feature relationships.

This script provides a complete suite of analysis tools for understanding:
1. Latent space structure via dimensionality reduction
2. Numerical properties (clustering, intrinsic dimensionality, sparsity, density)
3. Feature relationships and interactions
4. Network analysis of both samples and features

Perfect for deep understanding of how different KL coefficients and architectures
affect latent space organization and feature learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors, KernelDensity, kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
import torch
from tqdm import tqdm
import warnings

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class LatentSpaceData:
    """Container for latent space activation data."""
    activations: np.ndarray  # Shape: [n_samples, latent_dim]
    model_name: str
    kl_coeff: Optional[float]
    model_path: str
    n_samples: int
    latent_dim: int
    
    def __post_init__(self):
        assert self.activations.shape == (self.n_samples, self.latent_dim), \
            f"Shape mismatch: got {self.activations.shape}, expected ({self.n_samples}, {self.latent_dim})"


class LatentSpaceCollector:
    """Collects full activation vectors from VSAE models."""
    
    def __init__(
        self,
        model_path: str,
        n_samples: int = 1000,  # Default to 2k samples
        device: str = "cuda",
        batch_size: int = 8,
        ctx_len: int = 128,
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.n_samples = n_samples
        self.device = device
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.seed = seed
        
        # Set seeds for deterministic sampling
        self._set_seeds()
        
        self.logger = logging.getLogger(__name__)
        
        # Will be set during initialization
        self.model = None
        self.dictionary = None
        self.buffer = None
        self.config = None
        
    def _set_seeds(self) -> None:
        """Set seeds for deterministic data sampling."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def load_model_and_dictionary(self) -> None:
        """Load the transformer model and trained dictionary."""
        self.logger.info(f"Loading model and dictionary from {self.model_path}")
        
        # Load the trained dictionary
        self.dictionary, self.config = load_dictionary(str(self.model_path), device=self.device)
        
        # Extract model information from config
        model_name = self.config["trainer"].get("lm_name", "gelu-1l")
        hook_name = self.config["trainer"].get("submodule_name", "blocks.0.mlp.hook_post")
        
        self.logger.info(f"Model: {model_name}, Hook: {hook_name}")
        
        # Load the transformer model
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        
        # Store hook name for buffer creation
        self.hook_name = hook_name
        
        # Extract SAE experiment name from path
        self.sae_experiment_name = self.model_path.parent.name if self.model_path.name == 'trainer_0' else self.model_path.name
        
        self.logger.info(f"SAE experiment: {self.sae_experiment_name}")
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
        
        # Create buffer - conservative sizing to avoid OOM errors
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
        
        self.logger.info(f"Buffer config: n_ctxs={n_ctxs}, refresh_batch={self.batch_size}, out_batch={min(256, self.n_samples // 20)}")
        
    def collect_latent_vectors(self) -> LatentSpaceData:
        """Collect full activation vectors from the latent space."""
        self.logger.info(f"Collecting {self.n_samples} latent activation vectors")
        
        all_activations = []
        samples_collected = 0
        
        # Progress bar
        pbar = tqdm(total=self.n_samples, desc="Collecting latent vectors")
        
        try:
            while samples_collected < self.n_samples:
                # Get batch of activations
                try:
                    activations_batch = next(self.buffer).to(self.device)
                except StopIteration:
                    self.logger.warning("Ran out of data before collecting enough samples")
                    break
                
                # Encode through dictionary to get latent activations
                with torch.no_grad():
                    if hasattr(self.dictionary, 'encode'):
                        # For VSAE models
                        if hasattr(self.dictionary, 'var_flag') and self.dictionary.var_flag == 1:
                            # VSAE with learned variance
                            mu, log_var = self.dictionary.encode(activations_batch)
                            latent_vectors = self.dictionary.reparameterize(mu, log_var)
                        else:
                            # VSAE with fixed variance or other models
                            result = self.dictionary.encode(activations_batch)
                            if isinstance(result, tuple):
                                latent_vectors = result[0]  # Take mean if tuple
                            else:
                                latent_vectors = result
                    else:
                        # For standard SAE models
                        _, latent_vectors = self.dictionary(activations_batch, output_features=True)
                
                # Add to collection
                batch_vectors = latent_vectors.cpu().numpy()
                all_activations.append(batch_vectors)
                
                samples_collected += batch_vectors.shape[0]
                pbar.update(batch_vectors.shape[0])
                
                # Stop if we have enough samples
                if samples_collected >= self.n_samples:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during collection: {e}")
            raise
        finally:
            pbar.close()
        
        # Concatenate all batches and trim to exact sample size
        all_activations = np.vstack(all_activations)[:self.n_samples]
        
        # Extract metadata
        base_model_name = self.config["trainer"].get("lm_name", "unknown")
        kl_coeff = self.config["trainer"].get("kl_coeff", None)
        
        self.logger.info(f"Collected {all_activations.shape[0]} vectors of dimension {all_activations.shape[1]}")
        self.logger.info(f"Sparsity: {np.mean(all_activations == 0):.2%} zeros")
        self.logger.info(f"Mean activation: {np.mean(all_activations):.4f}")
        self.logger.info(f"Std activation: {np.std(all_activations):.4f}")
        
        return LatentSpaceData(
            activations=all_activations,
            model_name=self.sae_experiment_name,
            kl_coeff=kl_coeff,
            model_path=str(self.model_path),
            n_samples=all_activations.shape[0],
            latent_dim=all_activations.shape[1],
        )
    
    def run(self) -> LatentSpaceData:
        """Run the complete collection pipeline."""
        self.load_model_and_dictionary()
        self.create_buffer()
        return self.collect_latent_vectors()


class ComprehensiveAnalyzer:
    """Handles all types of latent space analysis."""
    
    def __init__(self, output_dir: str, model_name: str = None):
        self.base_output_dir = Path(output_dir)
        
        # Create model-specific subdirectory if model name provided
        if model_name:
            clean_model_name = model_name.replace('/', '_').replace('\\', '_')
            self.output_dir = self.base_output_dir / clean_model_name
        else:
            self.output_dir = self.base_output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Store fitted reducers for reuse
        self._fitted_reducers = {}
    
    def _preprocess_for_analysis(self, X: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Standard preprocessing for numerical analysis."""
        # Remove dead features
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        # Dimensionality reduction if needed
        if X_filtered.shape[1] > max_features:
            pca = PCA(n_components=max_features)
            X_filtered = pca.fit_transform(X_filtered)
        
        return X_filtered
    
    # =============================================================================
    # DIMENSIONALITY REDUCTION METHODS
    # =============================================================================
    
    def fit_pca(self, data: LatentSpaceData, n_components: int = 50, save_results: bool = True) -> Dict[str, Any]:
        """Perform PCA analysis on latent space."""
        self.logger.info(f"Running PCA with {n_components} components on {data.model_name}")
        
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for PCA")
        
        # Standardize
        X_std = (X_filtered - np.mean(X_filtered, axis=0)) / (np.std(X_filtered, axis=0) + 1e-8)
        
        # Fit PCA
        pca = PCA(n_components=min(n_components, X_std.shape[1]))
        X_pca = pca.fit_transform(X_std)
        
        self._fitted_reducers[f"pca_{data.model_name}"] = (pca, active_features)
        
        results = {
            'embeddings': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components_90': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1,
            'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1,
            'n_active_features': np.sum(active_features),
            'total_features': len(active_features),
        }
        
        if save_results:
            self._save_pca_plots(data, results)
        
        return results
    
    def fit_umap(self, data: LatentSpaceData, n_neighbors: int = 15, min_dist: float = 0.1, 
                 n_components: int = 2, metric: str = "cosine", save_results: bool = True) -> Optional[Dict[str, Any]]:
        """Perform UMAP analysis on latent space."""
        if not UMAP_AVAILABLE:
            self.logger.warning("UMAP not available, skipping UMAP analysis")
            return None
            
        self.logger.info(f"Running UMAP on {data.model_name} (neighbors={n_neighbors}, metric={metric})")
        
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for UMAP")
        
        try:
            reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, X_filtered.shape[0] - 1),
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=42,
                n_jobs=1,
                low_memory=True
            )
            
            X_umap = reducer.fit_transform(X_filtered)
            
        except Exception as e:
            self.logger.warning(f"UMAP failed: {str(e)[:100]}...")
            return None
        
        self._fitted_reducers[f"umap_{data.model_name}"] = (reducer, active_features)
        
        results = {
            'embeddings': X_umap,
            'n_neighbors': min(n_neighbors, X_filtered.shape[0] - 1),
            'min_dist': min_dist,
            'metric': metric,
            'n_components': n_components,
            'n_active_features': np.sum(active_features),
        }
        
        if save_results:
            self._save_umap_plots(data, results)
        
        return results
    
    def fit_tsne(self, data: LatentSpaceData, perplexity: int = 30, n_components: int = 2,
                 learning_rate: str = "auto", n_iter: int = 1000, save_results: bool = True) -> Dict[str, Any]:
        """Perform t-SNE analysis on latent space."""
        self.logger.info(f"Running t-SNE on {data.model_name} (perplexity={perplexity})")
        
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        if X_filtered.shape[1] > 100:
            self.logger.info("Pre-reducing dimensionality with PCA for t-SNE")
            pca = PCA(n_components=50)
            X_filtered = pca.fit_transform(X_filtered)
        
        self.logger.info(f"Using {X_filtered.shape[1]} features for t-SNE")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=42,
            verbose=0
        )
        
        X_tsne = tsne.fit_transform(X_filtered)
        
        results = {
            'embeddings': X_tsne,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'n_components': n_components,
            'n_active_features': np.sum(active_features),
            'kl_divergence': tsne.kl_divergence_,
        }
        
        if save_results:
            self._save_tsne_plots(data, results)
        
        return results
    
    def fit_isomap(self, data: LatentSpaceData, n_neighbors: int = 10, n_components: int = 2, 
                   save_results: bool = True) -> Dict[str, Any]:
        """Perform Isomap analysis on latent space."""
        self.logger.info(f"Running Isomap on {data.model_name} (neighbors={n_neighbors})")
        
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        if X_filtered.shape[1] > 100:
            self.logger.info("Pre-reducing dimensionality with PCA for Isomap")
            pca = PCA(n_components=50)
            X_filtered = pca.fit_transform(X_filtered)
        
        self.logger.info(f"Using {X_filtered.shape[1]} features for Isomap")
        
        isomap = Isomap(
            n_neighbors=min(n_neighbors, X_filtered.shape[0] - 1),
            n_components=n_components,
            n_jobs=1
        )
        
        X_isomap = isomap.fit_transform(X_filtered)
        
        results = {
            'embeddings': X_isomap,
            'n_neighbors': min(n_neighbors, X_filtered.shape[0] - 1),
            'n_components': n_components,
            'n_active_features': np.sum(active_features),
            'reconstruction_error': isomap.reconstruction_error(),
        }
        
        if save_results:
            self._save_isomap_plots(data, results)
        
        return results
    
    # =============================================================================
    # CLUSTERING ANALYSIS
    # =============================================================================
    
    def analyze_clustering(self, data: LatentSpaceData, save_results: bool = True) -> Dict[str, Any]:
        """Perform comprehensive clustering analysis."""
        self.logger.info(f"Running clustering analysis on {data.model_name}")
        
        X = self._preprocess_for_analysis(data.activations)
        
        results = {}
        
        # K-means with multiple k values
        silhouette_scores = []
        k_range = range(2, min(21, X.shape[0]//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans_final.fit_predict(X)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        # Gaussian Mixture
        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm_labels = gmm.fit_predict(X)
        
        results = {
            'optimal_k': optimal_k,
            'kmeans_labels': kmeans_labels,
            'kmeans_silhouette': silhouette_scores[optimal_k-2],
            'dbscan_labels': dbscan_labels,
            'dbscan_n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'gmm_labels': gmm_labels,
            'gmm_aic': gmm.aic(X),
            'gmm_bic': gmm.bic(X),
            'silhouette_scores': silhouette_scores,
            'k_range': list(k_range),
        }
        
        if save_results:
            self._save_clustering_plots(data, results)
        
        return results
    
    # =============================================================================
    # INTRINSIC DIMENSIONALITY
    # =============================================================================
    
    def estimate_intrinsic_dimensionality(self, data: LatentSpaceData) -> Dict[str, Any]:
        """Estimate intrinsic dimensionality using multiple methods."""
        self.logger.info(f"Estimating intrinsic dimensionality for {data.model_name}")
        
        X = self._preprocess_for_analysis(data.activations)
        
        # MLE estimator
        def mle_dimensionality(X, k=10):
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, _ = nbrs.kneighbors(X)
            distances = distances[:, 1:]  # Remove self-distance
            
            r_k = distances[:, -1]
            sum_log_ratios = np.sum(np.log(r_k[:, np.newaxis] / distances[:, :-1]), axis=1)
            mle_dim = (k-1) / np.mean(sum_log_ratios)
            return mle_dim
        
        # Correlation dimension
        def correlation_dimension(X, r_values=None):
            distances = pdist(X)
            if r_values is None:
                r_values = np.logspace(np.log10(np.min(distances[distances>0])), 
                                     np.log10(np.max(distances)), 20)
            
            correlations = []
            for r in r_values:
                count = np.sum(distances < r)
                total_pairs = len(X) * (len(X) - 1) / 2
                correlations.append(count / total_pairs)
            
            log_r = np.log(r_values[1:])
            log_c = np.log(np.array(correlations[1:]) + 1e-10)
            
            valid_idx = np.isfinite(log_c)
            if np.sum(valid_idx) > 1:
                slope, _ = np.polyfit(log_r[valid_idx], log_c[valid_idx], 1)
                return slope
            return np.nan
        
        mle_dim = mle_dimensionality(X)
        corr_dim = correlation_dimension(X)
        
        # Participation ratio
        activation_norms = np.linalg.norm(X, axis=1)
        participation_ratio = np.sum(activation_norms)**2 / (len(activation_norms) * np.sum(activation_norms**2))
        
        # Effective rank
        svd_values = np.linalg.svd(X, compute_uv=False)
        effective_rank = np.sum(svd_values)**2 / np.sum(svd_values**2)
        
        return {
            'mle_dimension': mle_dim,
            'correlation_dimension': corr_dim,
            'participation_ratio': participation_ratio,
            'effective_rank': effective_rank,
        }
    
    # =============================================================================
    # SPARSITY ANALYSIS
    # =============================================================================
    
    def analyze_sparsity_patterns(self, data: LatentSpaceData) -> Dict[str, Any]:
        """Comprehensive sparsity analysis."""
        self.logger.info(f"Analyzing sparsity patterns for {data.model_name}")
        
        X = data.activations
        
        # Basic sparsity metrics
        sparsity = np.mean(X == 0)
        feature_sparsity = np.mean(X == 0, axis=0)
        sample_sparsity = np.mean(X == 0, axis=1)
        
        # Dead features
        dead_features = np.sum(feature_sparsity == 1)
        
        # Feature utilization
        activation_counts = np.sum(X != 0, axis=0)
        utilization = activation_counts / X.shape[0]
        
        # Gini coefficient
        def gini_coefficient(x):
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return 1 - 2 * np.sum(cumsum) / (n * cumsum[-1])
        
        gini_features = gini_coefficient(utilization)
        
        # Norms
        l0_norms = np.sum(X != 0, axis=1)
        l1_norms = np.sum(np.abs(X), axis=1)
        l2_norms = np.linalg.norm(X, axis=1)
        
        # Co-activation matrix
        active_mask = X != 0
        coactivation_matrix = np.dot(active_mask.T, active_mask) / X.shape[0]
        
        return {
            'overall_sparsity': sparsity,
            'dead_features': dead_features,
            'dead_feature_ratio': dead_features / X.shape[1],
            'feature_utilization_gini': gini_features,
            'mean_l0_norm': np.mean(l0_norms),
            'mean_l1_norm': np.mean(l1_norms),
            'mean_l2_norm': np.mean(l2_norms),
            'coactivation_matrix': coactivation_matrix,
            'feature_activation_counts': activation_counts,
            'utilization_distribution': utilization,
        }
    
    # =============================================================================
    # FEATURE RELATIONSHIP ANALYSIS
    # =============================================================================
    
    def compute_feature_similarities(self, data: LatentSpaceData) -> Dict[str, np.ndarray]:
        """Compute multiple types of feature similarity matrices."""
        self.logger.info(f"Computing feature similarities for {data.model_name}")
        
        X = data.activations  # Shape: [n_samples, n_features]
        n_features = X.shape[1]
        
        similarities = {}
        
        # 1. Correlation similarity
        self.logger.info("Computing correlation similarity matrix")
        corr_matrix = np.corrcoef(X.T)
        similarities['correlation'] = corr_matrix
        
        # 2. Cosine similarity
        self.logger.info("Computing cosine similarity matrix")
        cosine_matrix = cosine_similarity(X.T)
        similarities['cosine'] = cosine_matrix
        
        # 3. Co-activation frequency
        self.logger.info("Computing co-activation similarity")
        binary_activations = (X != 0).astype(float)
        coactivation_counts = np.dot(binary_activations.T, binary_activations)
        activation_counts = np.sum(binary_activations, axis=0)
        normalization = np.sqrt(np.outer(activation_counts, activation_counts))
        coactivation_matrix = coactivation_counts / (normalization + 1e-10)
        similarities['coactivation'] = coactivation_matrix
        
        # 4. Mutual Information (sampled for efficiency)
        self.logger.info("Computing mutual information similarity")
        mi_matrix = np.zeros((n_features, n_features))
        
        n_samples_mi = min(5000, X.shape[0])
        X_sample = X[:n_samples_mi]
        
        # Sample features for MI computation
        feature_subset = min(100, n_features)
        feature_indices = np.random.choice(n_features, feature_subset, replace=False)
        
        for i, feat_i in enumerate(feature_indices):
            if i % 20 == 0:
                self.logger.info(f"MI progress: {i}/{len(feature_indices)}")
            
            feature_i = X_sample[:, feat_i].reshape(-1, 1)
            mi_scores = mutual_info_regression(
                X_sample[:, feature_indices], 
                feature_i.ravel(),
                discrete_features=False,
                random_state=42
            )
            mi_matrix[feat_i, feature_indices] = mi_scores
        
        # Make symmetric
        mi_matrix = (mi_matrix + mi_matrix.T) / 2
        similarities['mutual_info'] = mi_matrix
        
        return similarities
    
    def cluster_features(self, similarities: Dict[str, np.ndarray], data: LatentSpaceData) -> Dict[str, Any]:
        """Cluster features based on similarity matrices."""
        self.logger.info(f"Clustering features for {data.model_name}")
        
        results = {}
        
        for sim_type, sim_matrix in similarities.items():
            self.logger.info(f"Clustering features using {sim_type} similarity")
            
            # Convert similarity to distance
            distance_matrix = 1 - np.abs(sim_matrix)
            np.fill_diagonal(distance_matrix, 0)
            
            # Find optimal number of clusters
            silhouette_scores = []
            k_range = range(2, min(21, sim_matrix.shape[0]//10))
            
            for k in k_range:
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=k, 
                        metric='precomputed', 
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    silhouette_scores.append(sil_score)
                except:
                    silhouette_scores.append(-1)
            
            if silhouette_scores and max(silhouette_scores) > -1:
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                # Final clustering
                final_clustering = AgglomerativeClustering(
                    n_clusters=optimal_k,
                    metric='precomputed',
                    linkage='average'
                )
                feature_labels = final_clustering.fit_predict(distance_matrix)
                
                results[sim_type] = {
                    'optimal_k': optimal_k,
                    'feature_labels': feature_labels,
                    'silhouette_score': max(silhouette_scores),
                    'similarity_matrix': sim_matrix,
                }
        
        return results
    
    def visualize_feature_space(self, similarities: Dict[str, np.ndarray], data: LatentSpaceData) -> Dict[str, Any]:
        """Create visualizations of feature relationships."""
        self.logger.info(f"Creating feature space visualizations for {data.model_name}")
        
        visualization_results = {}
        
        for sim_type, sim_matrix in similarities.items():
            self.logger.info(f"Creating feature space visualization for {sim_type}")
            
            # Convert similarity to distance for embedding
            distance_matrix = 1 - np.abs(sim_matrix)
            np.fill_diagonal(distance_matrix, 0)
            
            # t-SNE on feature similarity
            try:
                tsne = TSNE(
                    n_components=2,
                    metric='precomputed',
                    perplexity=min(30, sim_matrix.shape[0]//4),
                    random_state=42,
                    n_iter=1000
                )
                
                feature_tsne = tsne.fit_transform(distance_matrix)
            except:
                feature_tsne = None
            
            # UMAP if available
            feature_umap = None
            if UMAP_AVAILABLE:
                try:
                    umap_reducer = umap.UMAP(
                        n_components=2,
                        metric='precomputed',
                        random_state=42
                    )
                    feature_umap = umap_reducer.fit_transform(distance_matrix)
                except Exception as e:
                    self.logger.warning(f"UMAP failed for {sim_type}: {e}")
            
            visualization_results[sim_type] = {
                'tsne_embedding': feature_tsne,
                'umap_embedding': feature_umap,
                'similarity_matrix': sim_matrix,
            }
            
            # Save visualizations
            self._save_feature_visualizations(data, sim_type, visualization_results[sim_type])
        
        return visualization_results
    
    # =============================================================================
    # NETWORK ANALYSIS
    # =============================================================================
    
    def analyze_latent_network(self, data: LatentSpaceData, k: int = 10) -> Dict[str, Any]:
        """Analyze the k-nearest neighbor graph structure."""
        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available, skipping network analysis")
            return {}
            
        self.logger.info(f"Analyzing latent network for {data.model_name}")
        
        X = self._preprocess_for_analysis(data.activations)
        
        # Build k-NN graph
        knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity')
        G = nx.from_scipy_sparse_array(knn_graph)
        
        # Graph metrics
        clustering_coeff = nx.average_clustering(G)
        
        # Connected components
        components = list(nx.connected_components(G))
        largest_component_size = len(max(components, key=len)) if components else 0
        
        # Small world properties
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except:
            avg_path_length = np.inf
        
        # Degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        
        return {
            'clustering_coefficient': clustering_coeff,
            'n_connected_components': len(components),
            'largest_component_ratio': largest_component_size / len(G.nodes()),
            'average_path_length': avg_path_length,
            'degree_stats': {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'max': np.max(degrees),
                'min': np.min(degrees),
            },
            'small_world_index': clustering_coeff / (k / len(G.nodes())) if avg_path_length != np.inf else np.inf,
        }
    
    # =============================================================================
    # COMPREHENSIVE ANALYSIS
    # =============================================================================
    
    def comprehensive_analysis(self, data: LatentSpaceData, methods: List[str]) -> Dict[str, Any]:
        """Run all requested analyses."""
        self.logger.info(f"Starting comprehensive analysis for {data.model_name}")
        
        results = {}
        
        # Dimensionality reduction
        if 'pca' in methods:
            results['pca'] = self.fit_pca(data)
        if 'umap' in methods:
            results['umap'] = self.fit_umap(data)
        if 'tsne' in methods:
            results['tsne'] = self.fit_tsne(data)
        if 'isomap' in methods:
            results['isomap'] = self.fit_isomap(data)
        
        # Numerical analyses
        if 'clustering' in methods:
            results['clustering'] = self.analyze_clustering(data)
        if 'intrinsic_dim' in methods:
            results['intrinsic_dim'] = self.estimate_intrinsic_dimensionality(data)
        if 'sparsity' in methods:
            results['sparsity'] = self.analyze_sparsity_patterns(data)
        if 'network' in methods:
            results['network'] = self.analyze_latent_network(data)
        
        # Feature analysis
        if 'features' in methods:
            similarities = self.compute_feature_similarities(data)
            results['feature_similarities'] = similarities
            results['feature_clustering'] = self.cluster_features(similarities, data)
            results['feature_visualizations'] = self.visualize_feature_space(similarities, data)
        
        # Save comprehensive summary
        self._save_comprehensive_summary(data, results)
        
        return results
    
    # =============================================================================
    # PLOTTING AND SAVING FUNCTIONS
    # =============================================================================
    
    def _save_pca_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save PCA analysis plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Explained variance
        n_components = len(results['explained_variance_ratio'])
        axes[0].plot(range(1, n_components + 1), results['explained_variance_ratio'], 'o-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title(f'PCA Explained Variance\n{data.model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(range(1, n_components + 1), results['cumulative_variance'], 'o-')
        axes[1].axhline(0.9, color='red', linestyle='--', label='90%')
        axes[1].axhline(0.95, color='orange', linestyle='--', label='95%')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 2D projection
        embeddings = results['embeddings']
        scatter = axes[2].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[2].set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]:.2%} var)')
        axes[2].set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]:.2%} var)')
        axes[2].set_title('PCA Projection (First 2 Components)')
        
        plt.tight_layout()
        pca_file = self.output_dir / f"pca_{data.model_name.replace('/', '_')}.png"
        plt.savefig(pca_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_umap_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save UMAP analysis plots."""
        embeddings = results['embeddings']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Basic UMAP plot
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].set_title(f'UMAP Projection\n{data.model_name}\n'
                         f'neighbors={results["n_neighbors"]}, metric={results["metric"]}')
        
        # Density plot
        axes[1].hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Blues')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].set_title('UMAP Density Plot')
        
        plt.tight_layout()
        umap_file = self.output_dir / f"umap_{data.model_name.replace('/', '_')}.png"
        plt.savefig(umap_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_tsne_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save t-SNE analysis plots."""
        embeddings = results['embeddings']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Basic t-SNE plot
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].set_title(f't-SNE Projection\n{data.model_name}\n'
                         f'perplexity={results["perplexity"]}, KL={results["kl_divergence"]:.2f}')
        
        # Density plot
        axes[1].hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Reds')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE Density Plot')
        
        plt.tight_layout()
        tsne_file = self.output_dir / f"tsne_{data.model_name.replace('/', '_')}.png"
        plt.savefig(tsne_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_isomap_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save Isomap analysis plots."""
        embeddings = results['embeddings']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Basic Isomap plot
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[0].set_xlabel('Isomap 1')
        axes[0].set_ylabel('Isomap 2')
        axes[0].set_title(f'Isomap Projection\n{data.model_name}\n'
                         f'neighbors={results["n_neighbors"]}, error={results["reconstruction_error"]:.3f}')
        
        # Density plot
        axes[1].hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Greens')
        axes[1].set_xlabel('Isomap 1')
        axes[1].set_ylabel('Isomap 2')
        axes[1].set_title('Isomap Density Plot')
        
        plt.tight_layout()
        isomap_file = self.output_dir / f"isomap_{data.model_name.replace('/', '_')}.png"
        plt.savefig(isomap_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_clustering_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save clustering analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette scores
        axes[0, 0].plot(results['k_range'], results['silhouette_scores'], 'o-')
        axes[0, 0].axvline(results['optimal_k'], color='red', linestyle='--', label=f'Optimal k={results["optimal_k"]}')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Optimal Cluster Number')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cluster distribution
        unique_labels, counts = np.unique(results['kmeans_labels'], return_counts=True)
        axes[0, 1].bar(unique_labels, counts)
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('K-means Cluster Sizes')
        
        # DBSCAN results
        if results['dbscan_n_clusters'] > 0:
            unique_labels, counts = np.unique(results['dbscan_labels'], return_counts=True)
            axes[1, 0].bar(unique_labels, counts)
            axes[1, 0].set_xlabel('Cluster (-1 = noise)')
            axes[1, 0].set_ylabel('Number of Samples')
            axes[1, 0].set_title(f'DBSCAN Clusters ({results["dbscan_n_clusters"]} clusters)')
        
        # GMM comparison
        axes[1, 1].bar(['AIC', 'BIC'], [results['gmm_aic'], results['gmm_bic']])
        axes[1, 1].set_ylabel('Information Criterion')
        axes[1, 1].set_title('GMM Model Selection')
        
        plt.tight_layout()
        clustering_file = self.output_dir / f"clustering_{data.model_name.replace('/', '_')}.png"
        plt.savefig(clustering_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_feature_visualizations(self, data: LatentSpaceData, sim_type: str, results: Dict[str, Any]) -> None:
        """Save feature relationship visualizations."""
        n_plots = 2 + (1 if results['umap_embedding'] is not None else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Similarity matrix heatmap
        im = axes[plot_idx].imshow(results['similarity_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[plot_idx].set_title(f'{sim_type.title()} Similarity Matrix\n{data.model_name}')
        axes[plot_idx].set_xlabel('Feature Index')
        axes[plot_idx].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1
        
        # t-SNE feature embedding
        if results['tsne_embedding'] is not None:
            tsne_emb = results['tsne_embedding']
            axes[plot_idx].scatter(tsne_emb[:, 0], tsne_emb[:, 1], alpha=0.7, s=20)
            axes[plot_idx].set_xlabel('t-SNE 1')
            axes[plot_idx].set_ylabel('t-SNE 2')
            axes[plot_idx].set_title(f'Feature t-SNE ({sim_type})')
            plot_idx += 1
        
        # UMAP feature embedding
        if results['umap_embedding'] is not None:
            umap_emb = results['umap_embedding']
            axes[plot_idx].scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=0.7, s=20)
            axes[plot_idx].set_xlabel('UMAP 1')
            axes[plot_idx].set_ylabel('UMAP 2')
            axes[plot_idx].set_title(f'Feature UMAP ({sim_type})')
        
        plt.tight_layout()
        feature_file = self.output_dir / f"features_{sim_type}_{data.model_name.replace('/', '_')}.png"
        plt.savefig(feature_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_summary(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save comprehensive analysis summary."""
        summary = {
            'model_info': {
                'name': data.model_name,
                'kl_coeff': data.kl_coeff,
                'n_samples': data.n_samples,
                'latent_dim': data.latent_dim,
                'sparsity': float(np.mean(data.activations == 0)),
            }
        }
        
        # Add numerical summaries from each analysis
        for analysis_type, analysis_results in results.items():
            if isinstance(analysis_results, dict):
                # Extract key metrics
                if analysis_type == 'pca':
                    summary[analysis_type] = {
                        'n_components_90': analysis_results.get('n_components_90'),
                        'n_components_95': analysis_results.get('n_components_95'),
                        'n_active_features': analysis_results.get('n_active_features'),
                    }
                elif analysis_type == 'clustering':
                    summary[analysis_type] = {
                        'optimal_k': analysis_results.get('optimal_k'),
                        'kmeans_silhouette': analysis_results.get('kmeans_silhouette'),
                        'dbscan_n_clusters': analysis_results.get('dbscan_n_clusters'),
                    }
                elif analysis_type == 'intrinsic_dim':
                    summary[analysis_type] = analysis_results
                elif analysis_type == 'sparsity':
                    summary[analysis_type] = {
                        'overall_sparsity': analysis_results.get('overall_sparsity'),
                        'dead_features': analysis_results.get('dead_features'),
                        'dead_feature_ratio': analysis_results.get('dead_feature_ratio'),
                        'mean_l0_norm': analysis_results.get('mean_l0_norm'),
                    }
        
        # Save summary as JSON
        summary_file = self.output_dir / f"analysis_summary_{data.model_name.replace('/', '_')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Analysis summary saved to {summary_file}")
    
    def compare_models(self, datasets: List[LatentSpaceData], method: str = "umap") -> None:
        """Create comparative visualization of multiple models."""
        self.logger.info(f"Creating comparative {method.upper()} visualization for {len(datasets)} models")
        
        all_results = []
        for data in datasets:
            try:
                if method == "umap":
                    results = self.fit_umap(data, save_results=False)
                elif method == "tsne":
                    results = self.fit_tsne(data, save_results=False)
                elif method == "pca":
                    results = self.fit_pca(data, save_results=False)
                elif method == "isomap":
                    results = self.fit_isomap(data, save_results=False)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if results is not None:
                    all_results.append((data, results))
                else:
                    self.logger.warning(f"Skipping {data.model_name} due to {method} failure")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {data.model_name} with {method}: {e}")
                continue
        
        if not all_results:
            self.logger.error(f"No successful {method} results for comparison")
            return
        
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (data, results) in enumerate(all_results):
            embeddings = results['embeddings']
            
            axes[i].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=8)
            
            title = data.model_name
            if data.kl_coeff is not None:
                title += f"\nKL coeff: {data.kl_coeff}"
            axes[i].set_title(title)
            axes[i].set_xlabel(f'{method.upper()} 1')
            axes[i].set_ylabel(f'{method.upper()} 2')
        
        plt.tight_layout()
        comp_file = self.output_dir / f"comparison_{method}.png"
        plt.savefig(comp_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparative plot saved to {comp_file}")


def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'comprehensive_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VSAE latent space and feature analysis")
    
    parser.add_argument(
        "--model-paths",
        nargs='+',
        required=True,
        help="Paths to trained model directories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./comprehensive_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        choices=['pca', 'umap', 'tsne', 'isomap', 'clustering', 'intrinsic_dim', 'sparsity', 'network', 'features'],
        default=['pca', 'tsne', 'clustering', 'intrinsic_dim', 'sparsity', 'network', 'features'],
        help="Which analysis methods to use"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50_000,
        help="Number of samples to collect per model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling"
    )
    parser.add_argument(
        "--compare",
        action='store_true',
        help="Create comparative visualizations across models"
    )
    
    args = parser.parse_args()
    
    # Set up output directory and logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive latent space analysis ❤️")
    logger.info(f"Model paths: {args.model_paths}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Samples per model: {args.n_samples:,}")
    
    # Check dependencies
    if 'umap' in args.methods and not UMAP_AVAILABLE:
        logger.warning("UMAP requested but not available. Using t-SNE instead.")
        methods = [m if m != 'umap' else 'tsne' for m in args.methods]
        args.methods = list(set(methods))
    
    if ('network' in args.methods or 'features' in args.methods) and not NETWORKX_AVAILABLE:
        logger.warning("NetworkX not available. Skipping network analysis.")
        args.methods = [m for m in args.methods if m != 'network']
    
    # Collect data from all models
    datasets = []
    for model_path in args.model_paths:
        logger.info(f"Processing model: {model_path}")
        
        collector = LatentSpaceCollector(
            model_path=model_path,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
        )
        
        try:
            data = collector.run()
            datasets.append(data)
            
            # Save the collected data
            clean_sae_name = data.model_name.replace('/', '_').replace('\\', '_')
            model_output_dir = output_dir / clean_sae_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            data_file = model_output_dir / f"latent_data_{args.n_samples//1000}k_samples.npz"
            np.savez_compressed(
                data_file,
                activations=data.activations,
                metadata={
                    'sae_experiment_name': data.model_name,
                    'kl_coeff': data.kl_coeff,
                    'model_path': data.model_path,
                    'n_samples': data.n_samples,
                    'latent_dim': data.latent_dim,
                }
            )
            logger.info(f"Saved latent data to {data_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            continue
    
    if not datasets:
        logger.error("No datasets collected successfully")
        return
    
    # Run comprehensive analyses for each model
    for data in datasets:
        logger.info(f"🔬 Comprehensive analysis for: {data.model_name}")
        
        analyzer = ComprehensiveAnalyzer(str(output_dir), data.model_name)
        results = analyzer.comprehensive_analysis(data, args.methods)
        
        # Log key findings
        if 'pca' in results:
            logger.info(f"PCA: {results['pca']['n_components_90']} components for 90% variance")
        if 'clustering' in results:
            logger.info(f"Clustering: optimal k = {results['clustering']['optimal_k']}")
        if 'sparsity' in results:
            logger.info(f"Sparsity: {results['sparsity']['dead_features']} dead features ({results['sparsity']['dead_feature_ratio']:.1%})")
        if 'intrinsic_dim' in results:
            logger.info(f"Intrinsic dim: MLE = {results['intrinsic_dim']['mle_dimension']:.1f}")
    
    # Create comparative visualizations
    if args.compare and len(datasets) > 1:
        comparison_analyzer = ComprehensiveAnalyzer(str(output_dir))
        dim_red_methods = [m for m in args.methods if m in ['pca', 'umap', 'tsne', 'isomap']]
        for method in dim_red_methods:
            comparison_analyzer.compare_models(datasets, method=method)
    
    logger.info(f"Comprehensive analysis completed! ❤️ Results saved to {output_dir}")


if __name__ == "__main__":
    main()


# Usage examples:
# python analyze_latent_space.py --model-paths "C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var\trainer_0"
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_0.01/trainer_0 ./experiments/vsae_kl_100/trainer_0 --compare
# python analyze_latent_space.py --model-paths ./experiments/VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var/trainer_0 --methods pca tsne clustering sparsity features
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_1/trainer_0 --methods pca tsne isomap clustering intrinsic_dim sparsity network features --n-samples 100000
# python analyze_latent_space.py --model-paths ./models/topk_vsae --n-samples 25000 --methods pca clustering sparsity features
# python analyze_latent_space.py --model-paths ./models/vsae --methods pca tsne clustering intrinsic_dim sparsity features --compareBatch