"""
Comprehensive VSAE latent space analysis using multiple numerical techniques.

This script performs deep analysis of latent space structure including:
- Dimensionality reduction (PCA, t-SNE, UMAP, Isomap, LLE, MDS)
- Clustering analysis (K-means, DBSCAN, Gaussian Mixture)
- Intrinsic dimensionality estimation
- Sparsity and feature utilization patterns
- Density structure analysis
- Network/graph analysis of latent neighborhoods
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
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KernelDensity
from scipy.stats import entropy
import networkx as nx
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
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
        n_samples: int = 2_000_00,  # 2 million samples for comprehensive analysis
        device: str = "cuda",
        batch_size: int = 32,
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
        print(f"DEBUG: Loading model and dictionary from {self.model_path}")
        
        # Load the trained dictionary
        self.dictionary, self.config = load_dictionary(str(self.model_path), device=self.device)
        print(f"DEBUG: Dictionary loaded successfully")
        
        # Extract model information from config
        model_name = self.config["trainer"].get("lm_name", "gelu-1l")
        hook_name = self.config["trainer"].get("submodule_name", "blocks.0.mlp.hook_post")
        
        self.logger.info(f"Model: {model_name}, Hook: {hook_name}")
        print(f"DEBUG: Model: {model_name}, Hook: {hook_name}")
        
        # Load the transformer model
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        print(f"DEBUG: Transformer model loaded successfully")
        
        # Store hook name for buffer creation
        self.hook_name = hook_name
        
        # Extract SAE experiment name from path (the parent directory name)
        # This gives us names like 'VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var'
        self.sae_experiment_name = self.model_path.parent.name if self.model_path.name == 'trainer_0' else self.model_path.name
        
        self.logger.info(f"SAE experiment: {self.sae_experiment_name}")
        self.logger.info(f"Dictionary size: {self.dictionary.dict_size}")
        self.logger.info(f"Activation dim: {self.dictionary.activation_dim}")
        
        print(f"DEBUG: SAE experiment: {self.sae_experiment_name}")
        print(f"DEBUG: Dictionary size: {self.dictionary.dict_size}")
        print(f"DEBUG: Activation dim: {self.dictionary.activation_dim}")
        
    def create_buffer(self) -> None:
        """Create data buffer for activation collection."""
        self.logger.info("Setting up data buffer")
        print(f"DEBUG: Setting up data buffer")
        
        # Create data generator
        data_gen = hf_dataset_to_generator(
            "roneneldan/TinyStories",
            split="train",
            return_tokens=False
        )
        print(f"DEBUG: Data generator created")
        
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
        print(f"DEBUG: Buffer created successfully. n_ctxs={n_ctxs}, refresh_batch={self.batch_size}, out_batch={min(256, self.n_samples // 20)}")
        
    def collect_latent_vectors(self) -> LatentSpaceData:
        """
        Collect full activation vectors from the latent space.
        
        Returns:
            LatentSpaceData containing the collected activations and metadata
        """
        self.logger.info(f"Collecting {self.n_samples} latent activation vectors")
        print(f"DEBUG: Starting collection of {self.n_samples} latent activation vectors")
        
        all_activations = []
        samples_collected = 0
        
        # Progress bar
        pbar = tqdm(total=self.n_samples, desc="Collecting latent vectors")
        
        try:
            while samples_collected < self.n_samples:
                # Get batch of activations
                try:
                    activations_batch = next(self.buffer).to(self.device)
                    print(f"DEBUG: Got batch with shape {activations_batch.shape}")
                except StopIteration:
                    self.logger.warning("Ran out of data before collecting enough samples")
                    print("DEBUG: StopIteration - ran out of data")
                    break
                
                # Encode through dictionary to get latent activations
                with torch.no_grad():
                    if hasattr(self.dictionary, 'encode'):
                        print("DEBUG: Using dictionary.encode()")
                        # For VSAE models
                        if hasattr(self.dictionary, 'var_flag') and self.dictionary.var_flag == 1:
                            # VSAE with learned variance
                            print("DEBUG: VSAE with learned variance")
                            mu, log_var = self.dictionary.encode(activations_batch)
                            latent_vectors = self.dictionary.reparameterize(mu, log_var)
                        else:
                            # VSAE with fixed variance or other models
                            print("DEBUG: VSAE with fixed variance or other model")
                            result = self.dictionary.encode(activations_batch)
                            if isinstance(result, tuple):
                                latent_vectors = result[0]  # Take mean if tuple
                            else:
                                latent_vectors = result
                    else:
                        # For standard SAE models
                        print("DEBUG: Using standard SAE model")
                        _, latent_vectors = self.dictionary(activations_batch, output_features=True)
                
                print(f"DEBUG: Latent vectors shape: {latent_vectors.shape}")
                
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
            print(f"DEBUG: Error during collection: {e}")
            raise
        finally:
            pbar.close()
        
        # Concatenate all batches and trim to exact sample size
        all_activations = np.vstack(all_activations)[:self.n_samples]
        print(f"DEBUG: Final activations shape: {all_activations.shape}")
        
        # Extract metadata - use SAE experiment name instead of base model name
        base_model_name = self.config["trainer"].get("lm_name", "unknown")
        kl_coeff = self.config["trainer"].get("kl_coeff", None)
        
        self.logger.info(f"Collected {all_activations.shape[0]} vectors of dimension {all_activations.shape[1]}")
        self.logger.info(f"Sparsity: {np.mean(all_activations == 0):.2%} zeros")
        self.logger.info(f"Mean activation: {np.mean(all_activations):.4f}")
        self.logger.info(f"Std activation: {np.std(all_activations):.4f}")
        
        print(f"DEBUG: Collection complete - {all_activations.shape[0]} vectors of dimension {all_activations.shape[1]}")
        print(f"DEBUG: Sparsity: {np.mean(all_activations == 0):.2%} zeros")
        
        return LatentSpaceData(
            activations=all_activations,
            model_name=self.sae_experiment_name,  # Use full SAE experiment name
            kl_coeff=kl_coeff,
            model_path=str(self.model_path),
            n_samples=all_activations.shape[0],
            latent_dim=all_activations.shape[1],
        )
    
    def run(self) -> LatentSpaceData:
        """Run the complete collection pipeline."""
        print("DEBUG: Starting data collection pipeline")
        self.load_model_and_dictionary()
        self.create_buffer()
        return self.collect_latent_vectors()


class ComprehensiveLatentAnalyzer:
    """Comprehensive numerical analysis of latent spaces."""
    
    def __init__(self, output_dir: str, model_name: str = None):
        self.base_output_dir = Path(output_dir)
        
        # Create model-specific subdirectory if model name provided
        if model_name:
            # Clean model name for use as directory name
            clean_model_name = model_name.replace('/', '_').replace('\\', '_')
            self.output_dir = self.base_output_dir / clean_model_name
        else:
            self.output_dir = self.base_output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Store fitted reducers for reuse
        self._fitted_reducers = {}
        
        print(f"DEBUG: ComprehensiveLatentAnalyzer initialized. Output dir: {self.output_dir}")
    
    def _preprocess_for_analysis(self, X: np.ndarray, max_features: int = 100) -> np.ndarray:
        """Standard preprocessing for numerical analysis."""
        print(f"DEBUG: Preprocessing data with shape {X.shape}")
        
        # Remove dead features
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features")
        print(f"DEBUG: Using {np.sum(active_features)}/{len(active_features)} active features")
        
        # Dimensionality reduction if needed
        if X_filtered.shape[1] > max_features:
            self.logger.info(f"Pre-reducing dimensionality with PCA to {max_features} components")
            print(f"DEBUG: Pre-reducing dimensionality with PCA to {max_features} components")
            pca = PCA(n_components=max_features)
            X_filtered = pca.fit_transform(X_filtered)
            print(f"DEBUG: After PCA, shape is {X_filtered.shape}")
        
        return X_filtered
    
    # ===== DIMENSIONALITY REDUCTION METHODS =====
    
    def fit_pca(
        self, 
        data: LatentSpaceData, 
        n_components: int = 50,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform PCA analysis on latent space."""
        print(f"DEBUG: Starting PCA analysis on {data.model_name}")
        self.logger.info(f"Running PCA with {n_components} components on {data.model_name}")
        
        try:
            # Handle sparsity - PCA works better with standardized non-sparse data
            X = data.activations
            print(f"DEBUG: Input data shape: {X.shape}")
            
            # Remove completely dead features (all zeros)
            feature_activity = np.sum(X != 0, axis=0)
            active_features = feature_activity > (0.01 * X.shape[0])  # Active in >1% of samples
            X_filtered = X[:, active_features]
            
            self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for PCA")
            print(f"DEBUG: After filtering dead features: {X_filtered.shape}")
            
            # Standardize
            X_std = (X_filtered - np.mean(X_filtered, axis=0)) / (np.std(X_filtered, axis=0) + 1e-8)
            print(f"DEBUG: After standardization: {X_std.shape}")
            
            # Fit PCA
            pca = PCA(n_components=min(n_components, X_std.shape[1]))
            X_pca = pca.fit_transform(X_std)
            print(f"DEBUG: PCA completed. Output shape: {X_pca.shape}")
            
            # Store reducer
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
            
            print(f"DEBUG: PCA analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in PCA analysis: {e}")
            self.logger.error(f"Error in PCA analysis: {e}")
            raise
    
    def fit_umap(
        self,
        data: LatentSpaceData,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "cosine",
        save_results: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Perform UMAP analysis on latent space."""
        print(f"DEBUG: Starting UMAP analysis on {data.model_name}")
        
        try:
            import umap
        except ImportError:
            self.logger.warning("UMAP not available, skipping UMAP analysis")
            print("DEBUG: UMAP not available, skipping")
            return None
            
        self.logger.info(f"Running UMAP on {data.model_name} (neighbors={n_neighbors}, metric={metric})")
        
        try:
            # Handle sparsity - use only active features
            X = data.activations
            print(f"DEBUG: UMAP input data shape: {X.shape}")
            feature_activity = np.sum(X != 0, axis=0)
            active_features = feature_activity > (0.01 * X.shape[0])
            X_filtered = X[:, active_features]
            
            self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for UMAP")
            print(f"DEBUG: UMAP after filtering: {X_filtered.shape}")
            
            # First try with default settings
            reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, X_filtered.shape[0] - 1),
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=42,
                n_jobs=1,  # Force single-threaded to avoid numba issues
                low_memory=True  # Use low memory mode
            )
            
            X_umap = reducer.fit_transform(X_filtered)
            print(f"DEBUG: UMAP completed. Output shape: {X_umap.shape}")
            
        except Exception as e:
            self.logger.warning(f"UMAP failed due to dependency issues: {str(e)[:100]}...")
            self.logger.info("This is likely due to numba/numpy version conflicts")
            print(f"DEBUG: UMAP failed: {e}")
            return None
        
        # Store reducer
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
        
        print(f"DEBUG: UMAP analysis completed successfully")
        return results
    
    def fit_tsne(
        self,
        data: LatentSpaceData,
        perplexity: int = 30,
        n_components: int = 2,
        learning_rate: str = "auto",
        n_iter: int = 1000,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform t-SNE analysis on latent space."""
        print(f"DEBUG: Starting t-SNE analysis on {data.model_name}")
        self.logger.info(f"Running t-SNE on {data.model_name} (perplexity={perplexity})")
        
        try:
            X_filtered = self._preprocess_for_analysis(data.activations, max_features=50)
            print(f"DEBUG: t-SNE preprocessed data shape: {X_filtered.shape}")
            
            # Fit t-SNE
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42,
                verbose=0
            )
            
            X_tsne = tsne.fit_transform(X_filtered)
            print(f"DEBUG: t-SNE completed. Output shape: {X_tsne.shape}")
            
            results = {
                'embeddings': X_tsne,
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'n_iter': n_iter,
                'n_components': n_components,
                'kl_divergence': tsne.kl_divergence_,
            }
            
            if save_results:
                self._save_tsne_plots(data, results)
            
            print(f"DEBUG: t-SNE analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in t-SNE analysis: {e}")
            self.logger.error(f"Error in t-SNE analysis: {e}")
            raise
    
    def fit_isomap(
        self,
        data: LatentSpaceData,
        n_neighbors: int = 10,
        n_components: int = 2,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform Isomap analysis on latent space."""
        print(f"DEBUG: Starting Isomap analysis on {data.model_name}")
        self.logger.info(f"Running Isomap on {data.model_name} (neighbors={n_neighbors})")
        
        try:
            X_filtered = self._preprocess_for_analysis(data.activations, max_features=50)
            print(f"DEBUG: Isomap preprocessed data shape: {X_filtered.shape}")
            
            # Fit Isomap
            isomap = Isomap(
                n_neighbors=min(n_neighbors, X_filtered.shape[0] - 1),
                n_components=n_components,
                n_jobs=1
            )
            
            X_isomap = isomap.fit_transform(X_filtered)
            print(f"DEBUG: Isomap completed. Output shape: {X_isomap.shape}")
            
            results = {
                'embeddings': X_isomap,
                'n_neighbors': min(n_neighbors, X_filtered.shape[0] - 1),
                'n_components': n_components,
                'reconstruction_error': isomap.reconstruction_error(),
            }
            
            if save_results:
                self._save_isomap_plots(data, results)
            
            print(f"DEBUG: Isomap analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in Isomap analysis: {e}")
            self.logger.error(f"Error in Isomap analysis: {e}")
            raise
    
    def fit_feature_tsne(
        self,
        data: LatentSpaceData,
        perplexity: int = 30,
        n_components: int = 2,
        learning_rate: str = "auto",
        n_iter: int = 1000,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Perform t-SNE analysis on features (transposed) to cluster similar features."""
        print(f"DEBUG: Starting feature t-SNE analysis on {data.model_name}")
        self.logger.info(f"Running feature t-SNE on {data.model_name} (perplexity={perplexity})")
        
        try:
            X = data.activations
            print(f"DEBUG: Original data shape: {X.shape}")
            
            # Remove completely dead features first
            feature_activity = np.sum(X != 0, axis=0)
            active_features = feature_activity > (0.01 * X.shape[0])  # Active in >1% of samples
            X_active = X[:, active_features]
            active_feature_indices = np.where(active_features)[0]
            
            print(f"DEBUG: After removing dead features: {X_active.shape}")
            print(f"DEBUG: Using {np.sum(active_features)}/{len(active_features)} active features")
            
            # Transpose so features become the "samples" to cluster
            X_features = X_active.T  # Shape: (n_features, n_samples)
            print(f"DEBUG: Transposed feature matrix shape: {X_features.shape}")
            
            # Normalize features (each feature vector represents its activation pattern across samples)
            # Use standardization to make features comparable
            X_features_norm = (X_features - np.mean(X_features, axis=1, keepdims=True)) / (np.std(X_features, axis=1, keepdims=True) + 1e-8)
            print(f"DEBUG: Normalized feature matrix shape: {X_features_norm.shape}")
            
            # Adjust perplexity if we have fewer features than expected
            actual_perplexity = min(perplexity, (X_features_norm.shape[0] - 1) // 3)
            print(f"DEBUG: Using perplexity: {actual_perplexity}")
            
            # Fit t-SNE on features
            tsne = TSNE(
                n_components=n_components,
                perplexity=actual_perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42,
                verbose=0
            )
            
            X_tsne = tsne.fit_transform(X_features_norm)
            print(f"DEBUG: Feature t-SNE completed. Output shape: {X_tsne.shape}")
            
            # Cluster the features in the t-SNE space
            n_clusters = min(10, max(2, X_tsne.shape[0] // 10))
            print(f"DEBUG: Clustering features into {n_clusters} clusters")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            feature_clusters = kmeans.fit_predict(X_tsne)
            
            # Calculate feature statistics for visualization
            feature_utilization = np.sum(X_active != 0, axis=0) / X_active.shape[0]  # Utilization rate
            feature_mean_activation = np.mean(X_active, axis=0)  # Mean activation when active
            feature_sparsity = np.mean(X_active == 0, axis=0)  # Sparsity per feature
            
            results = {
                # 'embeddings': X_tsne,
                # 'feature_clusters': feature_clusters,
                'active_feature_indices': active_feature_indices,
                # 'feature_utilization': feature_utilization,
                # 'feature_mean_activation': feature_mean_activation,
                # 'feature_sparsity': feature_sparsity,
                # 'n_clusters': n_clusters,
                # 'perplexity': actual_perplexity,
                # 'learning_rate': learning_rate,
                # 'n_iter': n_iter,
                # 'kl_divergence': tsne.kl_divergence_,
            }
            print(active_feature_indices)
            if save_results:
                self._save_feature_tsne_plots(data, results)
            
            print(f"DEBUG: Feature t-SNE analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in feature t-SNE analysis: {e}")
            self.logger.error(f"Error in feature t-SNE analysis: {e}")
            raise
    
    # ===== CLUSTERING ANALYSIS =====
    
    def analyze_clustering(self, data: LatentSpaceData, save_results: bool = True) -> Dict[str, Any]:
        """Perform comprehensive clustering analysis."""
        print(f"DEBUG: Starting clustering analysis on {data.model_name}")
        self.logger.info(f"Running clustering analysis on {data.model_name}")
        
        try:
            X = self._preprocess_for_analysis(data.activations)
            print(f"DEBUG: Clustering preprocessed data shape: {X.shape}")
            
            results = {}
            
            # K-means with multiple k values
            silhouette_scores = []
            k_range = range(5, min(21, X.shape[0]//10))  
            print(f"DEBUG: Testing k values from {min(k_range)} to {max(k_range)}")
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                sil_score = silhouette_score(X, labels)
                silhouette_scores.append(sil_score)
                print(f"DEBUG: k={k}, silhouette={sil_score:.3f}")
            
            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"DEBUG: Optimal k: {optimal_k}")
            
            # Final clustering with optimal k
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
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
                'kmeans_silhouette': silhouette_scores[optimal_k-5],  # Updated index offset
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
            
            print(f"DEBUG: Clustering analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in clustering analysis: {e}")
            self.logger.error(f"Error in clustering analysis: {e}")
            raise
    
    # ===== INTRINSIC DIMENSIONALITY =====
    
    def estimate_intrinsic_dimensionality(self, data: LatentSpaceData) -> Dict[str, Any]:
        """Estimate intrinsic dimensionality using multiple methods."""
        print(f"DEBUG: Starting intrinsic dimensionality estimation for {data.model_name}")
        self.logger.info(f"Estimating intrinsic dimensionality for {data.model_name}")
        
        try:
            X = self._preprocess_for_analysis(data.activations)
            print(f"DEBUG: Intrinsic dim preprocessed data shape: {X.shape}")
            
            # MLE estimator
            def mle_dimensionality(X, k=10):
                """Maximum Likelihood Estimation of intrinsic dimensionality."""
                print(f"DEBUG: MLE dimensionality estimation with k={k}")
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
                distances, _ = nbrs.kneighbors(X)
                print(f"DEBUG: Distances shape: {distances.shape}")
                
                # Remove self-distance (first column)
                distances = distances[:, 1:]
                print(f"DEBUG: After removing self-distance: {distances.shape}")
                
                # MLE formula
                r_k = distances[:, -1]  # k-th nearest neighbor distance
                print(f"DEBUG: r_k shape: {r_k.shape}")
                print(f"DEBUG: distances[:, :-1] shape: {distances[:, :-1].shape}")
                
                # Fix the broadcasting issue by reshaping r_k
                r_k_expanded = r_k[:, np.newaxis]  # Shape: (n_samples, 1)
                print(f"DEBUG: r_k_expanded shape: {r_k_expanded.shape}")
                
                # Now the division should work properly
                log_ratios = np.log(r_k_expanded / distances[:, :-1])
                sum_log_ratios = np.sum(log_ratios, axis=1)
                print(f"DEBUG: sum_log_ratios shape: {sum_log_ratios.shape}")
                
                mle_dim = (k-1) / np.mean(sum_log_ratios)
                print(f"DEBUG: MLE dimension: {mle_dim}")
                return mle_dim
            
            # Correlation dimension
            def correlation_dimension(X, r_values=None):
                """Estimate correlation dimension using box-counting."""
                print(f"DEBUG: Correlation dimension estimation")
                from scipy.spatial.distance import pdist
                
                if r_values is None:
                    distances = pdist(X)
                    r_values = np.logspace(np.log10(np.min(distances[distances>0])), 
                                         np.log10(np.max(distances)), 20)
                
                correlations = []
                for r in r_values:
                    # Count pairs within distance r
                    count = np.sum(pdist(X) < r)
                    total_pairs = len(X) * (len(X) - 1) / 2
                    correlations.append(count / total_pairs)
                
                # Fit line to log-log plot
                log_r = np.log(r_values[1:])  # Skip first point (might be 0)
                log_c = np.log(np.array(correlations[1:]) + 1e-10)
                
                valid_idx = np.isfinite(log_c)
                if np.sum(valid_idx) > 1:
                    slope, _ = np.polyfit(log_r[valid_idx], log_c[valid_idx], 1)
                    print(f"DEBUG: Correlation dimension: {slope}")
                    return slope
                return np.nan
            
            mle_dim = mle_dimensionality(X)
            corr_dim = correlation_dimension(X)
            
            # Participation ratio (for sparse data)
            activation_norms = np.linalg.norm(X, axis=1)
            participation_ratio = np.sum(activation_norms)**2 / (len(activation_norms) * np.sum(activation_norms**2))
            
            # Effective rank
            svd_vals = np.linalg.svd(X, compute_uv=False)
            effective_rank = np.sum(svd_vals)**2 / np.sum(svd_vals**2)
            
            results = {
                'mle_dimension': mle_dim,
                'correlation_dimension': corr_dim,
                'participation_ratio': participation_ratio,
                'effective_rank': effective_rank,
            }
            
            self.logger.info(f"Intrinsic dimensionality estimates: MLE={mle_dim:.2f}, Corr={corr_dim:.2f}, EffRank={effective_rank:.2f}")
            print(f"DEBUG: Intrinsic dimensionality completed successfully")
            
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in intrinsic dimensionality estimation: {e}")
            self.logger.error(f"Error in intrinsic dimensionality estimation: {e}")
            raise
    
    # ===== SPARSITY ANALYSIS =====
    
    def analyze_sparsity_patterns(self, data: LatentSpaceData) -> Dict[str, Any]:
            """Comprehensive sparsity analysis."""
            print(f"DEBUG: Starting sparsity analysis for {data.model_name}")
            self.logger.info(f"Analyzing sparsity patterns for {data.model_name}")
            
            try:
                X = data.activations
                print(f"DEBUG: Sparsity analysis data shape: {X.shape}")
                
                # Basic sparsity metrics
                sparsity = np.mean(X == 0)
                feature_sparsity = np.mean(X == 0, axis=0)
                sample_sparsity = np.mean(X == 0, axis=1)
                
                # Dead features (never activate)
                dead_features = np.sum(feature_sparsity == 1)
                
                # Feature utilization distribution and firing counts
                activation_counts = np.sum(X != 0, axis=0)  # Raw count of times each feature fires
                utilization = activation_counts / X.shape[0]  # Proportion of samples where feature fires
                
                # Find most and least active features
                most_active_indices = np.argsort(activation_counts)[-10:][::-1]  # Top 10 most active
                least_active_indices = np.argsort(activation_counts)[:10]  # Bottom 10 least active (excluding dead)
                
                # Feature firing statistics
                feature_firing_stats = {
                    'total_firing_events': np.sum(activation_counts),
                    'max_firing_count': np.max(activation_counts),
                    'min_firing_count': np.min(activation_counts[activation_counts > 0]) if np.any(activation_counts > 0) else 0,
                    'median_firing_count': np.median(activation_counts[activation_counts > 0]) if np.any(activation_counts > 0) else 0,
                    'mean_firing_count': np.mean(activation_counts),
                    'std_firing_count': np.std(activation_counts),
                    'most_active_features': {
                        'indices': most_active_indices.tolist(),
                        'firing_counts': activation_counts[most_active_indices].tolist(),
                        'utilization_rates': utilization[most_active_indices].tolist(),
                    },
                    'least_active_features': {
                        'indices': least_active_indices.tolist(),
                        'firing_counts': activation_counts[least_active_indices].tolist(),
                        'utilization_rates': utilization[least_active_indices].tolist(),
                    },
                }
                
                # Gini coefficient for feature usage inequality
                def gini_coefficient(x):
                    """Calculate Gini coefficient."""
                    sorted_x = np.sort(x)
                    n = len(x)
                    cumsum = np.cumsum(sorted_x)
                    return 1 - 2 * np.sum(cumsum) / (n * cumsum[-1])
                
                gini_features = gini_coefficient(utilization)
                
                # L0, L1, L2 norms
                l0_norms = np.sum(X != 0, axis=1)
                l1_norms = np.sum(np.abs(X), axis=1)
                l2_norms = np.linalg.norm(X, axis=1)
                
                # Feature co-activation patterns
                active_mask = X != 0
                coactivation_matrix = np.dot(active_mask.T, active_mask) / X.shape[0]
                
                results = {
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
                    'feature_firing_stats': feature_firing_stats,
                    'sparsity_stats': {
                        'feature_sparsity_mean': np.mean(feature_sparsity),
                        'feature_sparsity_std': np.std(feature_sparsity),
                        'sample_sparsity_mean': np.mean(sample_sparsity),
                        'sample_sparsity_std': np.std(sample_sparsity),
                    }
                }
                
                # Log key findings about feature firing
                self.logger.info(f"Sparsity analysis: {sparsity:.2%} overall, {dead_features} dead features, Gini={gini_features:.3f}")
                self.logger.info(f"Feature firing: max={feature_firing_stats['max_firing_count']}, "
                            f"median={feature_firing_stats['median_firing_count']:.0f}, "
                            f"total events={feature_firing_stats['total_firing_events']:,}")
                print(f"DEBUG: Sparsity analysis completed successfully")
                print(f"DEBUG: Most active feature (index {most_active_indices[0]}) fired {activation_counts[most_active_indices[0]]} times")
                
                return results
                
            except Exception as e:
                print(f"DEBUG: Error in sparsity analysis: {e}")
                self.logger.error(f"Error in sparsity analysis: {e}")
                raise
            
    # ===== DENSITY ANALYSIS =====
    
    def analyze_density_structure(self, data: LatentSpaceData) -> Dict[str, Any]:
        """Analyze density structure of latent space."""
        print(f"DEBUG: Starting density analysis for {data.model_name}")
        self.logger.info(f"Analyzing density structure for {data.model_name}")
        
        try:
            X = self._preprocess_for_analysis(data.activations)
            print(f"DEBUG: Density analysis preprocessed data shape: {X.shape}")
            
            # Local density estimation
            nbrs = NearestNeighbors(n_neighbors=10).fit(X)
            distances, indices = nbrs.kneighbors(X)
            local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
            
            # Kernel density estimation (on 2D projection for visualization)
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
            kde.fit(X_2d)
            log_densities = kde.score_samples(X_2d)
            
            # Distribution analysis per dimension
            feature_entropies = []
            for i in range(min(50, X.shape[1])):  # Sample of features
                hist, _ = np.histogram(X[:, i], bins=50, density=True)
                hist = hist + 1e-10  # Avoid log(0)
                feature_entropies.append(entropy(hist))
            
            results = {
                'local_density_stats': {
                    'mean': np.mean(local_densities),
                    'std': np.std(local_densities),
                    'min': np.min(local_densities),
                    'max': np.max(local_densities),
                },
                'kde_density_stats': {
                    'mean_log_density': np.mean(log_densities),
                    'std_log_density': np.std(log_densities),
                },
                'mean_feature_entropy': np.mean(feature_entropies),
                'density_concentration': np.std(local_densities) / np.mean(local_densities),
            }
            
            print(f"DEBUG: Density analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in density analysis: {e}")
            self.logger.error(f"Error in density analysis: {e}")
            raise
    
    # ===== NETWORK ANALYSIS =====
    
    def analyze_latent_network(self, data: LatentSpaceData, k: int = 10) -> Dict[str, Any]:
        """Analyze the k-nearest neighbor graph structure."""
        print(f"DEBUG: Starting network analysis for {data.model_name}")
        self.logger.info(f"Analyzing network structure for {data.model_name}")
        
        try:
            X = self._preprocess_for_analysis(data.activations)
            print(f"DEBUG: Network analysis preprocessed data shape: {X.shape}")
            
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
            
            results = {
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
            
            print(f"DEBUG: Network analysis completed successfully")
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in network analysis: {e}")
            self.logger.error(f"Error in network analysis: {e}")
            raise
    
    # ===== COMPREHENSIVE ANALYSIS =====
    
    def comprehensive_analysis(self, data: LatentSpaceData, methods: List[str]) -> Dict[str, Any]:
        """Run comprehensive analysis with specified methods."""
        print(f"DEBUG: Starting comprehensive analysis for {data.model_name}")
        print(f"DEBUG: Methods to run: {methods}")
        self.logger.info(f"Running comprehensive analysis for {data.model_name}")
        
        results = {}
        
        # Dimensionality reduction
        if 'pca' in methods:
            print("DEBUG: Running PCA analysis...")
            try:
                results['pca'] = self.fit_pca(data)
                print("DEBUG: PCA completed successfully")
            except Exception as e:
                print(f"DEBUG: PCA failed: {e}")
                self.logger.error(f"PCA analysis failed: {e}")
                
        if 'umap' in methods:
            print("DEBUG: Running UMAP analysis...")
            try:
                umap_result = self.fit_umap(data)
                if umap_result:
                    results['umap'] = umap_result
                    print("DEBUG: UMAP completed successfully")
                else:
                    print("DEBUG: UMAP returned None (likely dependency issue)")
            except Exception as e:
                print(f"DEBUG: UMAP failed: {e}")
                self.logger.error(f"UMAP analysis failed: {e}")
                
        if 'tsne' in methods:
            print("DEBUG: Running t-SNE analysis...")
            try:
                results['tsne'] = self.fit_tsne(data)
                print("DEBUG: t-SNE completed successfully")
            except Exception as e:
                print(f"DEBUG: t-SNE failed: {e}")
                self.logger.error(f"t-SNE analysis failed: {e}")
                
        if 'isomap' in methods:
            print("DEBUG: Running Isomap analysis...")
            try:
                results['isomap'] = self.fit_isomap(data)
                print("DEBUG: Isomap completed successfully")
            except Exception as e:
                print(f"DEBUG: Isomap failed: {e}")
                self.logger.error(f"Isomap analysis failed: {e}")
                
        if 'feature_tsne' in methods:
            print("DEBUG: Running feature t-SNE analysis...")
            try:
                results['feature_tsne'] = self.fit_feature_tsne(data)
                print("DEBUG: Feature t-SNE completed successfully")
            except Exception as e:
                print(f"DEBUG: Feature t-SNE failed: {e}")
                self.logger.error(f"Feature t-SNE analysis failed: {e}")
        
        # Advanced analyses
        if 'clustering' in methods:
            print("DEBUG: Running clustering analysis...")
            try:
                results['clustering'] = self.analyze_clustering(data)
                print("DEBUG: Clustering completed successfully")
            except Exception as e:
                print(f"DEBUG: Clustering failed: {e}")
                self.logger.error(f"Clustering analysis failed: {e}")
                
        if 'intrinsic_dim' in methods:
            print("DEBUG: Running intrinsic dimensionality analysis...")
            try:
                results['intrinsic_dim'] = self.estimate_intrinsic_dimensionality(data)
                print("DEBUG: Intrinsic dimensionality completed successfully")
            except Exception as e:
                print(f"DEBUG: Intrinsic dimensionality failed: {e}")
                self.logger.error(f"Intrinsic dimensionality analysis failed: {e}")
                
        if 'sparsity' in methods:
            print("DEBUG: Running sparsity analysis...")
            try:
                results['sparsity'] = self.analyze_sparsity_patterns(data)
                print("DEBUG: Sparsity analysis completed successfully")
            except Exception as e:
                print(f"DEBUG: Sparsity analysis failed: {e}")
                self.logger.error(f"Sparsity analysis failed: {e}")
                
        if 'density' in methods:
            print("DEBUG: Running density analysis...")
            try:
                results['density'] = self.analyze_density_structure(data)
                print("DEBUG: Density analysis completed successfully")
            except Exception as e:
                print(f"DEBUG: Density analysis failed: {e}")
                self.logger.error(f"Density analysis failed: {e}")
                
        if 'network' in methods:
            print("DEBUG: Running network analysis...")
            try:
                results['network'] = self.analyze_latent_network(data)
                print("DEBUG: Network analysis completed successfully")
            except Exception as e:
                print(f"DEBUG: Network analysis failed: {e}")
                self.logger.error(f"Network analysis failed: {e}")
        
        # Save comprehensive summary
        print("DEBUG: Saving comprehensive summary...")
        try:
            self._save_comprehensive_summary(data, results)
            print("DEBUG: Summary saved successfully")
        except Exception as e:
            print(f"DEBUG: Failed to save summary: {e}")
            self.logger.error(f"Failed to save summary: {e}")
        
        print(f"DEBUG: Comprehensive analysis completed for {data.model_name}")
        return results
    
    # ===== PLOTTING METHODS =====
    
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
    
    def _save_feature_tsne_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save feature t-SNE analysis plots."""
        embeddings = results['embeddings']
        feature_clusters = results['feature_clusters']
        feature_utilization = results['feature_utilization']
        feature_mean_activation = results['feature_mean_activation']
        
        # Create 4x1 subplot layout
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # Plot 1: Features colored by cluster
        scatter = axes[0].scatter(
            embeddings[:, 0], embeddings[:, 1], 
            c=feature_clusters, 
            alpha=0.7, s=30, cmap='tab10'
        )
        axes[0].set_xlabel('Feature t-SNE 1')
        axes[0].set_ylabel('Feature t-SNE 2')
        axes[0].set_title(f'Feature Clusters\n{data.model_name}\n'
                        f'{results["n_clusters"]} clusters, perplexity={results["perplexity"]}')
        plt.colorbar(scatter, ax=axes[0], label='Cluster ID')
        
        # Plot 2: Features colored by utilization rate
        scatter = axes[1].scatter(
            embeddings[:, 0], embeddings[:, 1], 
            c=feature_utilization, 
            alpha=0.7, s=30, cmap='viridis'
        )
        axes[1].set_xlabel('Feature t-SNE 1')
        axes[1].set_ylabel('Feature t-SNE 2')
        axes[1].set_title('Feature Utilization Rate')
        plt.colorbar(scatter, ax=axes[1], label='Utilization Rate')
        
        # Plot 3: Features sized by utilization, colored by mean activation
        scatter = axes[2].scatter(
            embeddings[:, 0], embeddings[:, 1], 
            c=feature_mean_activation, 
            s=feature_utilization * 100 + 10,  # Size by utilization
            alpha=0.7, cmap='plasma'
        )
        axes[2].set_xlabel('Feature t-SNE 1')
        axes[2].set_ylabel('Feature t-SNE 2')
        axes[2].set_title('Features: Color=Mean Activation, Size=Utilization')
        plt.colorbar(scatter, ax=axes[2], label='Mean Activation')
        
        # Plot 4: Cluster utilization statistics
        cluster_stats = []
        for cluster_id in range(results['n_clusters']):
            cluster_mask = feature_clusters == cluster_id
            cluster_util = feature_utilization[cluster_mask]
            cluster_stats.append({
                'cluster': cluster_id,
                'size': np.sum(cluster_mask),
                'mean_util': np.mean(cluster_util),
                'std_util': np.std(cluster_util)
            })
        
        cluster_ids = [s['cluster'] for s in cluster_stats]
        cluster_utils = [s['mean_util'] for s in cluster_stats]
        cluster_sizes = [s['size'] for s in cluster_stats]
        
        bars = axes[3].bar(cluster_ids, cluster_utils, 
                        width=0.8, alpha=0.7, 
                        color=plt.cm.tab10(np.array(cluster_ids) / max(cluster_ids)))
        
        # Add cluster sizes as text on bars
        for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={size}', ha='center', va='bottom', fontsize=9)
        
        axes[3].set_xlabel('Cluster ID')
        axes[3].set_ylabel('Mean Utilization Rate')
        axes[3].set_title('Feature Cluster Utilization')
        axes[3].set_ylim(0, max(cluster_utils) * 1.2)
        
        plt.tight_layout()
        feature_tsne_file = self.output_dir / f"feature_tsne_{data.model_name.replace('/', '_')}.pdf"
        plt.savefig(feature_tsne_file, bbox_inches='tight')
        plt.close()
        
        print(f"DEBUG: Feature t-SNE plots saved to {feature_tsne_file}")
        self.logger.info(f"Feature t-SNE plots saved to {feature_tsne_file}")
    
    def _save_clustering_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save clustering analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Silhouette scores
        axes[0, 0].plot(results['k_range'], results['silhouette_scores'], 'o-')
        axes[0, 0].axvline(results['optimal_k'], color='red', linestyle='--', label=f'Optimal k={results["optimal_k"]}')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('K-means Cluster Selection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Get 2D embedding for visualization
        X = self._preprocess_for_analysis(data.activations)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # K-means clustering
        scatter = axes[0, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=results['kmeans_labels'], 
                                   alpha=0.6, s=10, cmap='tab10')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        axes[0, 1].set_title(f'K-means Clustering (k={results["optimal_k"]})')
        
        # DBSCAN clustering
        scatter = axes[0, 2].scatter(X_2d[:, 0], X_2d[:, 1], c=results['dbscan_labels'], 
                                   alpha=0.6, s=10, cmap='tab10')
        axes[0, 2].set_xlabel('PC1')
        axes[0, 2].set_ylabel('PC2')
        axes[0, 2].set_title(f'DBSCAN Clustering ({results["dbscan_n_clusters"]} clusters)')
        
        # Gaussian Mixture
        scatter = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=results['gmm_labels'], 
                                   alpha=0.6, s=10, cmap='tab10')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].set_title(f'Gaussian Mixture (AIC={results["gmm_aic"]:.1f})')
        
        # Cluster size distributions
        unique_kmeans, counts_kmeans = np.unique(results['kmeans_labels'], return_counts=True)
        axes[1, 1].bar(unique_kmeans, counts_kmeans)
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Size')
        axes[1, 1].set_title('K-means Cluster Sizes')
        
        # DBSCAN cluster sizes (excluding noise)
        unique_dbscan, counts_dbscan = np.unique(results['dbscan_labels'][results['dbscan_labels'] != -1], return_counts=True)
        if len(unique_dbscan) > 0:
            axes[1, 2].bar(unique_dbscan, counts_dbscan)
            axes[1, 2].set_xlabel('Cluster ID')
            axes[1, 2].set_ylabel('Size')
            axes[1, 2].set_title('DBSCAN Cluster Sizes (no noise)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No clusters found', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('DBSCAN: No clusters')
        
        plt.tight_layout()
        clustering_file = self.output_dir / f"clustering_{data.model_name.replace('/', '_')}.png"
        plt.savefig(clustering_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_comprehensive_summary(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save a comprehensive summary of all analyses."""
        # Create summary text
        summary = f"Comprehensive Latent Space Analysis: {data.model_name}\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Model: {data.model_name}\n"
        summary += f"KL Coefficient: {data.kl_coeff}\n"
        summary += f"Samples: {data.n_samples:,}\n"
        summary += f"Latent Dimension: {data.latent_dim}\n\n"
        
        # Dimensionality results
        if 'pca' in results:
            pca = results['pca']
            summary += f"PCA Analysis:\n"
            summary += f"  - Components for 90% variance: {pca['n_components_90']}\n"
            summary += f"  - Components for 95% variance: {pca['n_components_95']}\n"
            summary += f"  - Active features: {pca['n_active_features']}/{pca['total_features']}\n\n"
        
        # Feature clustering results
        if 'feature_tsne' in results:
            feat_tsne = results['feature_tsne']
            summary += f"Feature t-SNE Analysis:\n"
            summary += f"  - Features clustered: {len(feat_tsne['active_feature_indices'])}\n"
            summary += f"  - Feature clusters: {feat_tsne['n_clusters']}\n"
            summary += f"  - Mean feature utilization: {np.mean(feat_tsne['feature_utilization']):.3f}\n"
            summary += f"  - KL divergence: {feat_tsne['kl_divergence']:.2f}\n\n"
        
        # Clustering results
        if 'clustering' in results:
            clust = results['clustering']
            summary += f"Clustering Analysis:\n"
            summary += f"  - Optimal K-means clusters: {clust['optimal_k']}\n"
            summary += f"  - K-means silhouette score: {clust['kmeans_silhouette']:.3f}\n"
            summary += f"  - DBSCAN clusters: {clust['dbscan_n_clusters']}\n"
            summary += f"  - GMM AIC: {clust['gmm_aic']:.1f}\n\n"
        
        # Intrinsic dimensionality
        if 'intrinsic_dim' in results:
            intrin = results['intrinsic_dim']
            summary += f"Intrinsic Dimensionality:\n"
            summary += f"  - MLE estimate: {intrin['mle_dimension']:.2f}\n"
            summary += f"  - Correlation dimension: {intrin['correlation_dimension']:.2f}\n"
            summary += f"  - Effective rank: {intrin['effective_rank']:.2f}\n"
            summary += f"  - Participation ratio: {intrin['participation_ratio']:.3f}\n\n"
        
        # Sparsity analysis
        if 'sparsity' in results:
            sparse = results['sparsity']
            summary += f"Sparsity Analysis:\n"
            summary += f"  - Overall sparsity: {sparse['overall_sparsity']:.2%}\n"
            summary += f"  - Dead features: {sparse['dead_features']} ({sparse['dead_feature_ratio']:.2%})\n"
            summary += f"  - Feature utilization Gini: {sparse['feature_utilization_gini']:.3f}\n"
            summary += f"  - Mean L0 norm: {sparse['mean_l0_norm']:.1f}\n\n"
        
        # Density analysis
        if 'density' in results:
            density = results['density']
            summary += f"Density Analysis:\n"
            summary += f"  - Mean feature entropy: {density['mean_feature_entropy']:.3f}\n"
            summary += f"  - Density concentration: {density['density_concentration']:.3f}\n\n"
        
        # Network analysis
        if 'network' in results:
            network = results['network']
            summary += f"Network Analysis:\n"
            summary += f"  - Clustering coefficient: {network['clustering_coefficient']:.3f}\n"
            summary += f"  - Connected components: {network['n_connected_components']}\n"
            summary += f"  - Largest component ratio: {network['largest_component_ratio']:.3f}\n"
            summary += f"  - Average path length: {network['average_path_length']:.2f}\n"
        
        # Save summary
        summary_file = self.output_dir / f"summary_{data.model_name.replace('/', '_')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        # Save results as JSON with proper type conversion
        def convert_numpy_types(obj):
            """Recursively convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                if obj.size < 1000:  # Only save small arrays
                    return obj.tolist()
                else:
                    return f"<large_array_shape_{obj.shape}>"
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy_types(results)
        
        json_file = self.output_dir / f"results_{data.model_name.replace('/', '_')}.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Summary saved to {summary_file}")
        self.logger.info(f"Results saved to {json_file}")
    
    def compare_models(self, datasets: List[LatentSpaceData], method: str = "pca") -> None:
        """Create comparative visualization of multiple models."""
        self.logger.info(f"Creating comparative {method.upper()} visualization for {len(datasets)} models")
        print(f"DEBUG: Creating comparative {method.upper()} visualization for {len(datasets)} models")
        
        # First, collect results for all models
        all_results = []
        for data in datasets:
            try:
                if method == "pca":
                    results = self.fit_pca(data, save_results=False)
                elif method == "umap":
                    results = self.fit_umap(data, save_results=False)
                elif method == "tsne":
                    results = self.fit_tsne(data, save_results=False)
                elif method == "isomap":
                    results = self.fit_isomap(data, save_results=False)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if results is not None:
                    all_results.append((data, results))
                else:
                    self.logger.warning(f"Skipping {data.model_name} due to {method} failure")
                    print(f"DEBUG: Skipping {data.model_name} due to {method} failure")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {data.model_name} with {method}: {e}")
                print(f"DEBUG: Failed to process {data.model_name} with {method}: {e}")
                continue
        
        if not all_results:
            self.logger.error(f"No successful {method} results for comparison")
            print(f"DEBUG: No successful {method} results for comparison")
            return
        
        n_models = len(all_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (data, results) in enumerate(all_results):
            embeddings = results['embeddings']
            
            # Create scatter plot
            axes[i].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=8)
            
            # Title with KL coefficient if available
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
        print(f"DEBUG: Comparative plot saved to {comp_file}")


def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'latent_analysis.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VSAE latent space analysis")
    
    parser.add_argument(
        "--model-paths",
        nargs='+',
        required=True,
        help="Paths to trained model directories (containing ae.pt and config.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./latent_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        choices=['pca', 'tsne', 'isomap', 'feature_tsne', 'clustering', 'intrinsic_dim', 'sparsity', 'density', 'network'], #umap always fails
        default=['feature_tsne', 'clustering', 'intrinsic_dim', 'sparsity'],
        help="Which analysis methods to use"
    )
    parser.add_argument(
        "--comprehensive",
        action='store_true',
        help="Run all available analyses (overrides --methods)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100_000,
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
    
    print("DEBUG: Script started with arguments:")
    print(f"DEBUG: Model paths: {args.model_paths}")
    print(f"DEBUG: Output dir: {args.output_dir}")
    print(f"DEBUG: Methods: {args.methods}")
    print(f"DEBUG: Comprehensive: {args.comprehensive}")
    print(f"DEBUG: N samples: {args.n_samples}")
    print(f"DEBUG: Device: {args.device}")
    
    # Override methods if comprehensive analysis requested
    if args.comprehensive:
        args.methods = ['pca', 'tsne', 'isomap', 'feature_tsne', 'clustering', 'intrinsic_dim', 'sparsity', 'density', 'network']
        print(f"DEBUG: Comprehensive mode - using all methods: {args.methods}")
    
    # Set up output directory and logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive latent space analysis")
    logger.info(f"Model paths: {args.model_paths}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Samples per model: {args.n_samples:,}")
    
    print("DEBUG: Logger initialized")
    
    # Check UMAP availability
    if 'umap' in args.methods and not UMAP_AVAILABLE:
        logger.warning("UMAP requested but not available (dependency issues). Using alternatives.")
        print("DEBUG: UMAP not available, removing from methods")
        methods = [m for m in args.methods if m != 'umap']
        if 'isomap' not in methods:
            methods.append('isomap')
        args.methods = methods
        logger.info(f"Updated methods: {args.methods}")
        print(f"DEBUG: Updated methods: {args.methods}")
    
    # Collect data from all models
    datasets = []
    for i, model_path in enumerate(args.model_paths):
        print(f"DEBUG: Processing model {i+1}/{len(args.model_paths)}: {model_path}")
        logger.info(f"Processing model: {model_path}")
        
        collector = LatentSpaceCollector(
            model_path=model_path,
            n_samples=args.n_samples,
            device=args.device,
            seed=args.seed,
        )
        
        try:
            print(f"DEBUG: Running collector for {model_path}")
            data = collector.run()
            datasets.append(data)
            
            # Create model-specific output directory using SAE experiment name
            clean_sae_name = data.model_name.replace('/', '_').replace('\\', '_')
            model_output_dir = output_dir / clean_sae_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the collected data in SAE-specific directory
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
            print(f"DEBUG: Successfully collected data from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            print(f"DEBUG: Failed to process {model_path}: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            continue
    
    if not datasets:
        logger.error("No datasets collected successfully")
        print("DEBUG: No datasets collected successfully - exiting")
        return
    
    print(f"DEBUG: Successfully collected {len(datasets)} datasets")
    
    # Run comprehensive analyses for each model
    for i, data in enumerate(datasets):
        print(f"DEBUG: Analyzing dataset {i+1}/{len(datasets)}: {data.model_name}")
        logger.info(f"Analyzing SAE: {data.model_name}")
        
        # Create model-specific analyzer
        analyzer = ComprehensiveLatentAnalyzer(str(output_dir), data.model_name)
        
        # Run comprehensive analysis
        try:
            print(f"DEBUG: Starting comprehensive analysis for {data.model_name}")
            results = analyzer.comprehensive_analysis(data, args.methods)
            print(f"DEBUG: Completed comprehensive analysis for {data.model_name}")
            
            # Log key findings
            if 'pca' in results:
                logger.info(f"  PCA: {results['pca']['n_components_90']} components for 90% variance")
                print(f"DEBUG: PCA: {results['pca']['n_components_90']} components for 90% variance")
            if 'clustering' in results:
                logger.info(f"  Clustering: optimal k={results['clustering']['optimal_k']}, silhouette={results['clustering']['kmeans_silhouette']:.3f}")
                print(f"DEBUG: Clustering: optimal k={results['clustering']['optimal_k']}")
            if 'intrinsic_dim' in results:
                logger.info(f"  Intrinsic dim: MLE={results['intrinsic_dim']['mle_dimension']:.2f}")
                print(f"DEBUG: Intrinsic dim: MLE={results['intrinsic_dim']['mle_dimension']:.2f}")
            if 'sparsity' in results:
                logger.info(f"  Sparsity: {results['sparsity']['overall_sparsity']:.2%}, dead={results['sparsity']['dead_features']}")
                print(f"DEBUG: Sparsity: {results['sparsity']['overall_sparsity']:.2%}")
                
        except Exception as e:
            logger.error(f"Failed comprehensive analysis for {data.model_name}: {e}")
            print(f"DEBUG: Failed comprehensive analysis for {data.model_name}: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            continue
    
    # Create comparative visualizations
    if args.compare and len(datasets) > 1:
        print("DEBUG: Creating comparative visualizations")
        comparison_analyzer = ComprehensiveLatentAnalyzer(str(output_dir))
        for method in ['feature_tsne']:
            if method in args.methods:
                print(f"DEBUG: Creating comparative plot for {method}")
                try:
                    comparison_analyzer.compare_models(datasets, method=method)
                    print(f"DEBUG: Successfully created comparative plot for {method}")
                except Exception as e:
                    print(f"DEBUG: Failed to create comparative plot for {method}: {e}")
    
    logger.info(f"Comprehensive analysis completed! Results saved to {output_dir}")
    logger.info("Each model has its own subdirectory with detailed analyses and visualizations.")
    print("DEBUG: Script completed successfully")


if __name__ == "__main__":
    main()


# Usage examples:
# 
# Basic analysis with key methods:
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_1/trainer_0 --methods pca clustering sparsity
# python analyze_latent_space.py --model-paths ./experiments/VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/trainer_0 --methods tsne feature_tsne
# 
# Include feature clustering analysis:
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_1/trainer_0 --methods pca tsne feature_tsne clustering sparsity
# 
# Comprehensive analysis (all methods):
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_1/trainer_0 --comprehensive
# 
# Compare multiple models:
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_0.01/trainer_0 ./experiments/vsae_kl_100/trainer_0 --comprehensive --compare
# 
# Large-scale analysis:
# python analyze_latent_space.py --model-paths ./experiments/VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var/trainer_0 --n-samples 1000000 --comprehensive
# 
# Feature analysis focus:
# python analyze_latent_space.py --model-paths ./models/vsae --methods feature_tsne sparsity clustering
# 
# Output structure (comprehensive):
# latent_analysis/
# ├── logs/
# ├── VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var/
# │   ├── latent_data_20k_samples.npz
# │   ├── pca_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   ├── tsne_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   ├── feature_tsne_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   ├── clustering_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   ├── summary_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.txt
# │   └── results_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.json
# └── comparison_pca.png  # If --compare is used