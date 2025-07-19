"""
Analyze VSAE latent space structure using dimensionality reduction techniques.

This script collects full activation vectors from trained VSAE models and analyzes
the overall latent space structure using UMAP, t-SNE, and PCA. Perfect for comparing
how different KL coefficients affect latent space organization.
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
        n_samples: int = 2_000_000,  # 2 million samples for comprehensive analysis
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
        
        # Extract SAE experiment name from path (the parent directory name)
        # This gives us names like 'VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var'
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
        """
        Collect full activation vectors from the latent space.
        
        Returns:
            LatentSpaceData containing the collected activations and metadata
        """
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
        
        # Extract metadata - use SAE experiment name instead of base model name
        base_model_name = self.config["trainer"].get("lm_name", "unknown")
        kl_coeff = self.config["trainer"].get("kl_coeff", None)
        
        self.logger.info(f"Collected {all_activations.shape[0]} vectors of dimension {all_activations.shape[1]}")
        self.logger.info(f"Sparsity: {np.mean(all_activations == 0):.2%} zeros")
        self.logger.info(f"Mean activation: {np.mean(all_activations):.4f}")
        self.logger.info(f"Std activation: {np.std(all_activations):.4f}")
        
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
        self.load_model_and_dictionary()
        self.create_buffer()
        return self.collect_latent_vectors()


class DimensionalityReducer:
    """Handles dimensionality reduction analysis of latent spaces."""
    
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
    
    def fit_pca(
        self, 
        data: LatentSpaceData, 
        n_components: int = 50,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform PCA analysis on latent space.
        
        Returns:
            Dictionary with PCA results and explained variance info
        """
        self.logger.info(f"Running PCA with {n_components} components on {data.model_name}")
        
        # Handle sparsity - PCA works better with standardized non-sparse data
        X = data.activations
        
        # Remove completely dead features (all zeros)
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])  # Active in >1% of samples
        X_filtered = X[:, active_features]
        
        self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for PCA")
        
        # Standardize
        X_std = (X_filtered - np.mean(X_filtered, axis=0)) / (np.std(X_filtered, axis=0) + 1e-8)
        
        # Fit PCA
        pca = PCA(n_components=min(n_components, X_std.shape[1]))
        X_pca = pca.fit_transform(X_std)
        
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
        
        return results
    
    def fit_isomap(
        self,
        data: LatentSpaceData,
        n_neighbors: int = 10,
        n_components: int = 2,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Isomap analysis on latent space (great UMAP alternative).
        
        Returns:
            Dictionary with Isomap embeddings and parameters
        """
        self.logger.info(f"Running Isomap on {data.model_name} (neighbors={n_neighbors})")
        
        # Handle sparsity - use only active features
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        # If too many dimensions, use PCA first
        if X_filtered.shape[1] > 100:
            self.logger.info("Pre-reducing dimensionality with PCA for Isomap")
            pca = PCA(n_components=50)
            X_filtered = pca.fit_transform(X_filtered)
        
        self.logger.info(f"Using {X_filtered.shape[1]} features for Isomap")
        
        # Fit Isomap
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
    
    def fit_lle(
        self,
        data: LatentSpaceData,
        n_neighbors: int = 10,
        n_components: int = 2,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Locally Linear Embedding analysis on latent space.
        
        Returns:
            Dictionary with LLE embeddings and parameters
        """
        self.logger.info(f"Running LLE on {data.model_name} (neighbors={n_neighbors})")
        
        # Handle sparsity - use only active features
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        # If too many dimensions, use PCA first
        if X_filtered.shape[1] > 100:
            self.logger.info("Pre-reducing dimensionality with PCA for LLE")
            pca = PCA(n_components=50)
            X_filtered = pca.fit_transform(X_filtered)
        
        self.logger.info(f"Using {X_filtered.shape[1]} features for LLE")
        
        # Fit LLE
        lle = LocallyLinearEmbedding(
            n_neighbors=min(n_neighbors, X_filtered.shape[0] - 1),
            n_components=n_components,
            random_state=42,
            n_jobs=1
        )
        
        X_lle = lle.fit_transform(X_filtered)
        
        results = {
            'embeddings': X_lle,
            'n_neighbors': min(n_neighbors, X_filtered.shape[0] - 1),
            'n_components': n_components,
            'n_active_features': np.sum(active_features),
            'reconstruction_error': lle.reconstruction_error_,
        }
        
        if save_results:
            self._save_lle_plots(data, results)
        
        return results
    
    def fit_mds(
        self,
        data: LatentSpaceData,
        n_components: int = 2,
        metric: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Multidimensional Scaling analysis on latent space.
        
        Returns:
            Dictionary with MDS embeddings and parameters
        """
        self.logger.info(f"Running MDS on {data.model_name} (metric={metric})")
        
        # Handle sparsity - use only active features
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        # For MDS, we need to be more aggressive about dimensionality reduction
        if X_filtered.shape[1] > 50:
            self.logger.info("Pre-reducing dimensionality with PCA for MDS")
            pca = PCA(n_components=30)
            X_filtered = pca.fit_transform(X_filtered)
        
        # Also limit sample size for MDS (it's O(n^3))
        if X_filtered.shape[0] > 1000:
            self.logger.info("Subsampling for MDS to avoid memory issues")
            indices = np.random.choice(X_filtered.shape[0], 1000, replace=False)
            X_filtered = X_filtered[indices]
        
        self.logger.info(f"Using {X_filtered.shape} data for MDS")
        
        # Fit MDS
        mds = MDS(
            n_components=n_components,
            metric=metric,
            random_state=42,
            n_jobs=1
        )
        
        X_mds = mds.fit_transform(X_filtered)
        
        results = {
            'embeddings': X_mds,
            'n_components': n_components,
            'metric': metric,
            'n_active_features': np.sum(active_features),
            'stress': mds.stress_,
        }
        
        if save_results:
            self._save_mds_plots(data, results)
        
        return results
    
    def fit_umap(
        self,
        data: LatentSpaceData,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "cosine",
        save_results: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Perform UMAP analysis on latent space.
        
        Returns:
            Dictionary with UMAP embeddings and parameters, or None if UMAP fails
        """
        try:
            import umap
        except ImportError:
            self.logger.warning("UMAP not available, skipping UMAP analysis")
            return None
            
        self.logger.info(f"Running UMAP on {data.model_name} (neighbors={n_neighbors}, metric={metric})")
        
        # Handle sparsity - use only active features
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        self.logger.info(f"Using {np.sum(active_features)}/{len(active_features)} active features for UMAP")
        
        try:
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
            
        except Exception as e:
            self.logger.warning(f"UMAP failed due to dependency issues: {str(e)[:100]}...")
            self.logger.info("This is likely due to numba/numpy version conflicts")
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
        """
        Perform t-SNE analysis on latent space.
        
        Returns:
            Dictionary with t-SNE embeddings and parameters
        """
        self.logger.info(f"Running t-SNE on {data.model_name} (perplexity={perplexity})")
        
        # Handle sparsity and potentially reduce dimensionality first
        X = data.activations
        feature_activity = np.sum(X != 0, axis=0)
        active_features = feature_activity > (0.01 * X.shape[0])
        X_filtered = X[:, active_features]
        
        # If too many dimensions, use PCA first
        if X_filtered.shape[1] > 100:
            self.logger.info("Pre-reducing dimensionality with PCA for t-SNE")
            pca = PCA(n_components=50)
            X_filtered = pca.fit_transform(X_filtered)
        
        self.logger.info(f"Using {X_filtered.shape[1]} features for t-SNE")
        
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
    
    def _save_lle_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save LLE analysis plots."""
        embeddings = results['embeddings']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Basic LLE plot
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[0].set_xlabel('LLE 1')
        axes[0].set_ylabel('LLE 2')
        axes[0].set_title(f'LLE Projection\n{data.model_name}\n'
                         f'neighbors={results["n_neighbors"]}, error={results["reconstruction_error"]:.3f}')
        
        # Density plot
        axes[1].hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Purples')
        axes[1].set_xlabel('LLE 1')
        axes[1].set_ylabel('LLE 2')
        axes[1].set_title('LLE Density Plot')
        
        plt.tight_layout()
        lle_file = self.output_dir / f"lle_{data.model_name.replace('/', '_')}.png"
        plt.savefig(lle_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_mds_plots(self, data: LatentSpaceData, results: Dict[str, Any]) -> None:
        """Save MDS analysis plots."""
        embeddings = results['embeddings']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Basic MDS plot
        scatter = axes[0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        axes[0].set_xlabel('MDS 1')
        axes[0].set_ylabel('MDS 2')
        axes[0].set_title(f'MDS Projection\n{data.model_name}\n'
                         f'metric={results["metric"]}, stress={results["stress"]:.3f}')
        
        # Density plot
        axes[1].hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Oranges')
        axes[1].set_xlabel('MDS 1')
        axes[1].set_ylabel('MDS 2')
        axes[1].set_title('MDS Density Plot')
        
        plt.tight_layout()
        mds_file = self.output_dir / f"mds_{data.model_name.replace('/', '_')}.png"
        plt.savefig(mds_file, dpi=150, bbox_inches='tight')
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
    
    def compare_models(self, datasets: List[LatentSpaceData], method: str = "umap") -> None:
        """Create comparative visualization of multiple models."""
        self.logger.info(f"Creating comparative {method.upper()} visualization for {len(datasets)} models")
        
        # First, collect results for all models
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
                elif method == "lle":
                    results = self.fit_lle(data, save_results=False)
                elif method == "mds":
                    results = self.fit_mds(data, save_results=False)
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
    parser = argparse.ArgumentParser(description="Analyze VSAE latent space structure")
    
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
        choices=['pca', 'umap', 'tsne', 'isomap', 'lle', 'mds'],
        default=['pca', 'tsne', 'isomap'],  # Skip UMAP by default due to dependency issues
        help="Which dimensionality reduction methods to use"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20_000,  # 20k samples for comprehensive latent space analysis
        help="Number 2of samples to collect per model"  
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
    logger.info("Starting latent space analysis")
    logger.info(f"Model paths: {args.model_paths}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Samples per model: {args.n_samples:,}")
    
    if args.n_samples >= 1_000_000:
        logger.info(f"⚡ Large-scale analysis with {args.n_samples:,} samples - this may take a while but will give comprehensive results!")
    
    # Check UMAP availability
    if 'umap' in args.methods and not UMAP_AVAILABLE:
        logger.warning("UMAP requested but not available (dependency issues). Using alternatives.")
        # Replace umap with isomap if not already present
        methods = [m for m in args.methods if m != 'umap']
        if 'isomap' not in methods:
            methods.append('isomap')
        args.methods = methods
        logger.info(f"Updated methods: {args.methods}")
    
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
                    'sae_experiment_name': data.model_name,  # This is now the full SAE experiment name
                    'kl_coeff': data.kl_coeff,
                    'model_path': data.model_path,
                    'n_samples': data.n_samples,
                    'latent_dim': data.latent_dim,
                }
            )
            logger.info(f"Saved latent data to {data_file}")
            logger.info(f"📁 Results organized in: {model_output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to process {model_path}: {e}")
            continue
    
    if not datasets:
        logger.error("No datasets collected successfully")
        return
    
    # Run dimensionality reduction analyses for each model
    for data in datasets:
        logger.info(f"🔬 Analyzing SAE: {data.model_name}")
        
        # Create model-specific reducer
        reducer = DimensionalityReducer(str(output_dir), data.model_name)
        
        if 'pca' in args.methods:
            pca_results = reducer.fit_pca(data)
            logger.info(f"PCA: {pca_results['n_components_90']} components for 90% variance")
        
        if 'umap' in args.methods:
            umap_results = reducer.fit_umap(data)
            if umap_results:
                logger.info(f"UMAP: completed with {umap_results['n_active_features']} active features")
            else:
                logger.warning(f"UMAP: failed for {data.model_name}")
        
        if 'tsne' in args.methods:
            tsne_results = reducer.fit_tsne(data)
            logger.info(f"t-SNE: KL divergence = {tsne_results['kl_divergence']:.2f}")
        
        if 'isomap' in args.methods:
            isomap_results = reducer.fit_isomap(data)
            logger.info(f"Isomap: reconstruction error = {isomap_results['reconstruction_error']:.3f}")
        
        if 'lle' in args.methods:
            lle_results = reducer.fit_lle(data)
            logger.info(f"LLE: reconstruction error = {lle_results['reconstruction_error']:.3f}")
        
        if 'mds' in args.methods:
            mds_results = reducer.fit_mds(data)
            logger.info(f"MDS: stress = {mds_results['stress']:.3f}")
    
    # Create comparative visualizations (in base output directory)
    if args.compare and len(datasets) > 1:
        comparison_reducer = DimensionalityReducer(str(output_dir))  # Use base directory for comparisons
        for method in args.methods:
            comparison_reducer.compare_models(datasets, method=method)
    
    logger.info(f"Analysis completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()


# Usage examples:
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_0.01/trainer_0 ./experiments/vsae_kl_100/trainer_0 --compare
# python analyze_latent_space.py --model-paths ./experiments/VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var/trainer_0
# python analyze_latent_space.py --model-paths ./experiments/vsae_kl_1/trainer_0 --methods tsne isomap --n-samples 1000000
# python analyze_latent_space.py --model-paths ./models/topk_vsae --n-samples 500000 --methods pca lle
# python analyze_latent_space.py --model-paths ./models/vsae --methods pca tsne isomap lle mds --compare
# 
# Output structure (organized by SAE experiment name):
# latent_analysis/
# ├── logs/
# ├── VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var/
# │   ├── latent_data_2000k_samples.npz
# │   ├── pca_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   ├── tsne_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# │   └── isomap_VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var.png
# ├── VSAEIso_gelu-1l_d4096_kl500_lr5e-4/
# │   └── ...
# └── comparison_pca.png  # If --compare is used