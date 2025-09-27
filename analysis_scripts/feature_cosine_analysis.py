"""
Feature Cosine Similarity Analysis for VSAETopK vs SAETopK

This script analyzes the cosine similarities between features in trained SAE models,
computing statistical properties and creating visualizations for comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple, List, Optional
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, load_dictionary


class FeatureSimilarityAnalyzer:
    """Analyzes cosine similarities between features in SAE models."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        
    def load_model_and_buffer(self, model_name: str = "gelu-1l", layer: int = 0) -> Tuple:
        """Load transformer model and create activation buffer."""
        print(f"Loading model: {model_name}")
        model = HookedTransformer.from_pretrained(model_name, device=self.device)
        
        # Create buffer with same config as training script
        buffer = TransformerLensActivationBuffer(
            data=hf_dataset_to_generator("NeelNanda/pile-10k"),
            model=model,
            hook_name=f"blocks.{layer}.hook_resid_post",
            d_submodule=model.cfg.d_model,
            n_ctxs=3000,
            ctx_len=128,
            refresh_batch_size=32,
            out_batch_size=1024,
            device=self.device
        )
        
        return model, buffer
    
    def load_sae(self, sae_path: str, model_name: str = "VSAETopK") -> torch.nn.Module:
        """Load trained SAE model."""
        print(f"Loading {model_name} from {sae_path}")
        
        try:
            sae, config = load_dictionary(sae_path, device=self.device)
            
            # Get decoder shape for different SAE types
            if hasattr(sae, 'W_dec'):
                decoder_shape = sae.W_dec.shape
            elif hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
                decoder_shape = sae.decoder.weight.shape
            else:
                decoder_shape = "unknown"
            
            print(f"Loaded {model_name} with {decoder_shape} decoder weights")
            return sae
        except Exception as e:
            print(f"Error loading SAE: {e}")
            raise
    
    def compute_feature_cosine_similarities(self, sae: torch.nn.Module, 
                                          activations: torch.Tensor,
                                          n_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarities for features when they fire.
        
        Args:
            sae: The SAE model
            activations: Input activations [batch, seq, d_model]
            n_samples: Number of activation samples to analyze
            
        Returns:
            Dictionary with similarity matrices and firing patterns
        """
        print("Computing feature cosine similarities...")
        
        sae.eval()
        with torch.no_grad():
            # Get feature representations (decoder weights) - handle different architectures
            if hasattr(sae, 'W_dec'):
                features = sae.W_dec  # [dict_size, d_model] - standard SAE format
            elif hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
                features = sae.decoder.weight.T  # [d_model, dict_size] -> [dict_size, d_model]
            else:
                raise ValueError("Cannot find decoder weights in SAE")
            
            print(f"Feature matrix shape: {features.shape}")
            
            # Normalize features for cosine similarity
            features_normalized = torch.nn.functional.normalize(features, dim=1)
            
            # Compute full cosine similarity matrix between all features
            cosine_matrix = torch.mm(features_normalized, features_normalized.t())
            
            # Get feature activations for a subset of data
            activations_flat = activations.view(-1, activations.shape[-1])[:n_samples]
            
            # Get SAE encodings to see which features fire
            if hasattr(sae, 'encode'):
                encoding_result = sae.encode(activations_flat)
                # Handle different return formats
                if isinstance(encoding_result, tuple):
                    # VSAETopK returns (sparse_features, dense_features, kl_loss, aux_loss)
                    feature_acts = encoding_result[0]  # Use sparse features
                else:
                    feature_acts = encoding_result
            else:
                # Manual encoding for standard SAE
                if hasattr(sae, 'W_enc'):
                    pre_acts = torch.mm(activations_flat - sae.b_dec, sae.W_enc.t()) + sae.b_enc
                elif hasattr(sae, 'encoder'):
                    pre_acts = sae.encoder(activations_flat)
                else:
                    raise ValueError("Cannot find encoder in SAE")
                
                if hasattr(sae, 'threshold'):
                    feature_acts = torch.nn.functional.relu(pre_acts - sae.threshold)
                else:
                    feature_acts = torch.nn.functional.relu(pre_acts)
            
            # Find which features fire (non-zero activations)
            firing_mask = feature_acts > 0  # [n_samples, dict_size]
            firing_counts = firing_mask.sum(dim=0)  # How often each feature fires
            
            # Compute statistics for firing features only
            active_features = firing_counts > 0
            n_active = active_features.sum().item()
            
            print(f"Found {n_active} active features out of {features.shape[0]} total")
            
        return {
            'cosine_matrix': cosine_matrix,
            'features_normalized': features_normalized,
            'firing_counts': firing_counts,
            'active_features': active_features,
            'feature_acts': feature_acts,
            'n_active_features': n_active
        }
    
    def analyze_similarity_statistics(self, similarity_data: Dict[str, torch.Tensor], 
                                    model_name: str) -> Dict[str, float]:
        """Compute mean, variance, and other statistics of cosine similarities."""
        print(f"Analyzing similarity statistics for {model_name}...")
        
        cosine_matrix = similarity_data['cosine_matrix']
        active_features = similarity_data['active_features']
        
        # Focus on active features only
        active_indices = torch.where(active_features)[0]
        active_cosine_matrix = cosine_matrix[active_indices][:, active_indices]
        
        # Remove diagonal (self-similarities = 1.0)
        mask = ~torch.eye(active_cosine_matrix.shape[0], dtype=torch.bool)
        off_diagonal_sims = active_cosine_matrix[mask]
        
        # Compute statistics
        stats = {
            'mean_cosine_similarity': off_diagonal_sims.mean().item(),
            'std_cosine_similarity': off_diagonal_sims.std().item(),
            'var_cosine_similarity': off_diagonal_sims.var().item(),
            'min_cosine_similarity': off_diagonal_sims.min().item(),
            'max_cosine_similarity': off_diagonal_sims.max().item(),
            'median_cosine_similarity': off_diagonal_sims.median().item(),
            'q25_cosine_similarity': off_diagonal_sims.quantile(0.25).item(),
            'q75_cosine_similarity': off_diagonal_sims.quantile(0.75).item(),
            'n_active_features': similarity_data['n_active_features'],
            'total_features': cosine_matrix.shape[0]
        }
        
        # Store for later comparison
        self.results[model_name] = {
            'stats': stats,
            'similarity_data': similarity_data
        }
        
        print(f"Mean cosine similarity: {stats['mean_cosine_similarity']:.4f}")
        print(f"Std cosine similarity: {stats['std_cosine_similarity']:.4f}")
        
        return stats
    
    def create_visualizations(self, output_dir: Path, k_value: int = None):
        """Create various visualizations of the similarity data."""
        print("Creating visualizations...")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Comparison bar chart of statistics
        self._plot_statistics_comparison(output_dir, k_value)
        
        # 2. Distribution histograms
        self._plot_similarity_distributions(output_dir)
        
        # 3. PCA visualization of feature space
        self._plot_pca_features(output_dir)
        
        # 4. Heatmaps of similarity matrices (subset)
        self._plot_similarity_heatmaps(output_dir)
        
        # 5. Line chart showing similarity trends
        self._plot_similarity_trends(output_dir)
    
    def _plot_statistics_comparison(self, output_dir: Path, k_value: int = None):
        """Plot comparison of key statistics between models."""
        if len(self.results) < 2:
            print("Need at least 2 models for comparison")
            return
            
        stats_to_compare = ['mean_cosine_similarity', 'std_cosine_similarity', 
                           'var_cosine_similarity', 'median_cosine_similarity']
        
        model_names = list(self.results.keys())
        
        # Create enhanced labels with k value
        enhanced_labels = []
        for name in model_names:
            if k_value is not None and 'TopK' in name:
                enhanced_labels.append(f"{name}, k={k_value}")
            else:
                enhanced_labels.append(name)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # Slightly taller figure
        axes = axes.flatten()
        
        for i, stat in enumerate(stats_to_compare):
            values = [self.results[model]['stats'][stat] for model in model_names]
            max_val = max(values)
            min_val = min(values)
            
            axes[i].bar(enhanced_labels, values, alpha=0.7)
            axes[i].set_title(f'{stat.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
            
            # Calculate appropriate y-axis limits to accommodate text labels
            y_range = max_val - min_val
            if y_range == 0:  # All values are the same
                y_margin_top = abs(max_val) * 0.15 if max_val != 0 else 0.1
                y_margin_bottom = abs(min_val) * 0.05 if min_val != 0 else 0.05
            else:
                y_margin_top = y_range * 0.2  # 20% margin at top for text
                y_margin_bottom = y_range * 0.05  # 5% margin at bottom
            
            # Set y-axis limits with proper margins
            axes[i].set_ylim(min_val - y_margin_bottom, max_val + y_margin_top)
            
            # Add value labels on bars with proper positioning
            for j, v in enumerate(values):
                # Position text at 10% of the top margin above the bar
                text_y = v + y_margin_top * 0.1
                axes[i].text(j, text_y, f'{v:.4f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # Use tight_layout with extra padding to prevent cutoff
        plt.tight_layout(pad=3.0)
        plt.savefig(output_dir / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_distributions(self, output_dir: Path):
        """Plot distributions of cosine similarities with mean and std in legend."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(6*len(self.results), 5))
        if len(self.results) == 1:
            axes = [axes]
        
        # First pass: collect all similarities to determine global ranges
        all_similarities = []
        model_similarities = {}
        
        for model_name, data in self.results.items():
            cosine_matrix = data['similarity_data']['cosine_matrix']
            active_features = data['similarity_data']['active_features']
            
            # Get off-diagonal similarities for active features
            active_indices = torch.where(active_features)[0]
            active_cosine_matrix = cosine_matrix[active_indices][:, active_indices]
            mask = ~torch.eye(active_cosine_matrix.shape[0], dtype=torch.bool)
            similarities = active_cosine_matrix[mask].cpu().numpy()
            
            all_similarities.extend(similarities)
            model_similarities[model_name] = similarities
        
        # Calculate global ranges
        global_x_min = -0.3
        global_x_max = 0.3
        x_range = global_x_max - global_x_min
        x_margin = x_range * 0.05  # 5% margin
        
        # Second pass: create plots with consistent ranges
        max_density = 0
        
        for i, (model_name, similarities) in enumerate(model_similarities.items()):
            counts, bins, patches = axes[i].hist(similarities, bins=50, alpha=0.7, density=True)
            max_density = max(max_density, counts.max())
            
            axes[i].set_title(f'{model_name} Cosine Similarity Distribution')
            axes[i].set_xlabel('Cosine Similarity')
            axes[i].set_ylabel('Density')
            
            mean_val = similarities.mean()
            std_val = similarities.std()
            
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}, Std: {std_val:.3f}')
            axes[i].legend()
            
            # Set consistent x-axis range
            axes[i].set_xlim(global_x_min - x_margin, global_x_max + x_margin)
        
        # Set consistent y-axis range for all subplots
        y_margin = max_density * 0.05
        for ax in axes:
            ax.set_ylim(0, max_density + y_margin)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_features(self, output_dir: Path):
        """Create PCA visualization of feature representations."""
        for model_name, data in self.results.items():
            features = data['similarity_data']['features_normalized']
            active_features = data['similarity_data']['active_features']
            
            # PCA on active features only
            active_feature_vecs = features[active_features].cpu().numpy()
            
            if active_feature_vecs.shape[0] < 2:
                continue
                
            # Fit PCA
            pca = PCA(n_components=min(2, active_feature_vecs.shape[0]))
            features_2d = pca.fit_transform(active_feature_vecs)
            
            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20)
            plt.title(f'{model_name} Feature Space (PCA)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            if features_2d.shape[1] > 1:
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            
            plt.colorbar(scatter, label='Feature Index')
            plt.savefig(output_dir / f'{model_name}_pca.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_similarity_heatmaps(self, output_dir: Path):
        """Plot heatmaps of similarity matrices (subset for visibility) with logarithmic scaling."""
        for model_name, data in self.results.items():
            cosine_matrix = data['similarity_data']['cosine_matrix']
            active_features = data['similarity_data']['active_features']
            
            # Take subset of most active features
            active_indices = torch.where(active_features)[0]
            firing_counts = data['similarity_data']['firing_counts']
            
            # Get top 100 most active features for visualization
            top_active = firing_counts[active_indices].topk(min(100, len(active_indices)))
            top_indices = active_indices[top_active.indices]
            
            subset_matrix = cosine_matrix[top_indices][:, top_indices].cpu().numpy()
            
            plt.figure(figsize=(10, 8))
            
            # Transform data for logarithmic scaling
            # Shift values to be positive (cosine range -1,1 -> 0,2) and add epsilon
            epsilon = 1e-8
            transformed_matrix = subset_matrix + 1 + epsilon
            
            # Use logarithmic normalization
            log_norm = colors.LogNorm(vmin=transformed_matrix.min(), vmax=transformed_matrix.max())
            
            sns.heatmap(transformed_matrix, 
                       cmap='coolwarm', 
                       norm=log_norm,
                       square=True, 
                       cbar_kws={'label': 'Log(Cosine Similarity + 1)'})
            
            plt.title(f'{model_name} Feature Similarity Matrix (Top 100 Active Features) - Log Scale')
            plt.xlabel('Feature Index')
            plt.ylabel('Feature Index')
            plt.savefig(output_dir / f'{model_name}_similarity_heatmap_log.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_similarity_trends(self, output_dir: Path):
        """Create line chart showing similarity trends."""
        if len(self.results) < 2:
            return
            
        # Prepare data for comparison line chart
        model_names = list(self.results.keys())
        metrics = ['mean_cosine_similarity', 'std_cosine_similarity', 
                  'median_cosine_similarity', 'var_cosine_similarity']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.results[model]['stats'][metric] for model in model_names]
            
            axes[i].plot(model_names, values, 'o-', linewidth=2, markersize=8)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value annotations
            for j, (name, val) in enumerate(zip(model_names, values)):
                axes[i].annotate(f'{val:.4f}', (j, val), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: Path):
        """Save analysis results to files."""
        print("Saving results...")
        output_dir.mkdir(exist_ok=True)
        
        # Save statistics as JSON
        stats_only = {model: data['stats'] for model, data in self.results.items()}
        with open(output_dir / 'similarity_statistics.json', 'w') as f:
            json.dump(stats_only, f, indent=2)
        
        # Save raw similarity matrices
        for model_name, data in self.results.items():
            similarity_data = data['similarity_data']
            torch.save({
                'cosine_matrix': similarity_data['cosine_matrix'],
                'firing_counts': similarity_data['firing_counts'],
                'active_features': similarity_data['active_features']
            }, output_dir / f'{model_name}_similarity_data.pt')
        
        print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature cosine similarities")
    parser.add_argument("--vsae_path", type=str, required=True,
                       help="Path to trained VSAETopK model")
    parser.add_argument("--sae_path", type=str, required=True,
                       help="Path to trained SAETopK model")
    parser.add_argument("--output_dir", type=str, default="./similarity_analysis",
                       help="Output directory for results")
    parser.add_argument("--model_name", type=str, default="gelu-1l",
                       help="Transformer model name")
    parser.add_argument("--layer", type=int, default=0,
                       help="Layer to analyze")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of activation samples to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--k", type=int, default=None,
                       help="TopK value for labeling graphs (e.g., 512)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FeatureSimilarityAnalyzer(device=args.device)
    
    # Load model and buffer
    model, buffer = analyzer.load_model_and_buffer(args.model_name, args.layer)
    
    # Get some activations for analysis
    print("Getting activation samples...")
    activations = []
    for i, batch in enumerate(buffer):
        activations.append(batch)
        if i * buffer.out_batch_size >= args.n_samples:
            break
    activations = torch.cat(activations, dim=0)[:args.n_samples]
    
    # Analyze VSAETopK
    print("\n=== Analyzing VSAETopK ===")
    vsae = analyzer.load_sae(args.vsae_path, "VSAETopK")
    vsae_similarity = analyzer.compute_feature_cosine_similarities(vsae, activations, args.n_samples)
    vsae_stats = analyzer.analyze_similarity_statistics(vsae_similarity, "VSAETopK")
    
    # Analyze SAETopK
    print("\n=== Analyzing SAETopK ===")
    sae = analyzer.load_sae(args.sae_path, "SAETopK")
    sae_similarity = analyzer.compute_feature_cosine_similarities(sae, activations, args.n_samples)
    sae_stats = analyzer.analyze_similarity_statistics(sae_similarity, "SAETopK")
    
    # Create output directory and save results
    output_dir = Path(args.output_dir)
    analyzer.save_results(output_dir)
    analyzer.create_visualizations(output_dir, args.k)
    
    # Print comparison summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Metric':<25} | {'VSAETopK':<12} | {'SAETopK':<12} | {'Difference':<12}")
    print("-" * 70)
    
    for metric in ['mean_cosine_similarity', 'std_cosine_similarity', 'var_cosine_similarity']:
        vsae_val = vsae_stats[metric]
        sae_val = sae_stats[metric]
        diff = vsae_val - sae_val
        print(f"{metric:<25} | {vsae_val:<12.4f} | {sae_val:<12.4f} | {diff:<12.4f}")
    
    print(f"\nResults and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

# Example usage:
# python feature_cosine_analysis.py --sae_path ./experiments/TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto/trainer_0 --vsae_path ./experiments/VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var/trainer_0 --output_dir ./similarity_analysis/512 --n_samples 500000 --k 512