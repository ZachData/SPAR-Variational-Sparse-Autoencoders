# sae_feature_explorer.py
import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm

from feature_maximizer import FeatureMaximizer, load_components

class SAEFeatureExplorer:
    """
    Utilities for exploring SAE features and preparing data for visualization.
    This class extends FeatureMaximizer functionality to create visualizations and 
    prepare data for downstream visualization tools.
    """
    def __init__(self, maximizer: FeatureMaximizer):
        self.maximizer = maximizer
        self.model = maximizer.model
        self.sae = maximizer.sae
        self.device = maximizer.device
        
    def create_feature_embedding(
        self,
        feature_indices: list,
        embedding_method: str = "tsne",
        perplexity: int = 30,
        n_iter: int = 1000,
        output_path: str = None,
    ):
        """
        Create 2D embedding of features for visualization
        
        Args:
            feature_indices: List of feature indices to embed
            embedding_method: Method to use for embedding ('tsne', 'umap', etc.)
            perplexity: Perplexity parameter for t-SNE
            n_iter: Number of iterations for t-SNE
            output_path: Path to save the embedding visualization
        
        Returns:
            Dictionary with embedding results
        """
        # Get feature weights
        if hasattr(self.sae, 'decoder') and hasattr(self.sae.decoder, 'weight'):
            decoder_weights = self.sae.decoder.weight
            selected_weights = decoder_weights[:, feature_indices].T.cpu().numpy()
        elif hasattr(self.sae, 'W_dec'):
            decoder_weights = self.sae.W_dec
            selected_weights = decoder_weights[feature_indices].cpu().numpy()
        else:
            raise ValueError("Could not access feature weights")
        
        # Perform dimensionality reduction
        if embedding_method == "tsne":
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
            embedding = tsne.fit_transform(selected_weights)
        elif embedding_method == "umap":
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(selected_weights)
            except ImportError:
                print("UMAP not available. Falling back to t-SNE.")
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
                embedding = tsne.fit_transform(selected_weights)
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=50)
        
        # Add feature indices as labels
        for i, feat_idx in enumerate(feature_indices):
            plt.annotate(str(feat_idx), (embedding[i, 0], embedding[i, 1]))
        
        plt.title(f"Feature Embedding ({embedding_method.upper()})")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved embedding visualization to {output_path}")
        else:
            plt.show()
        
        # Return embedding results
        return {
            "method": embedding_method,
            "embedding": embedding.tolist(),
            "feature_indices": feature_indices
        }
    
    def create_feature_hierarchy(
        self,
        feature_indices: list,
        distance_method: str = "activation_correlation",
        output_path: str = None,
        num_samples: int = 1000,
    ):
        """
        Create a hierarchical clustering of features
        
        Args:
            feature_indices: List of feature indices to cluster
            distance_method: Method to use for distance calculation
            output_path: Path to save the clustering visualization
            num_samples: Number of samples for activation correlation
        
        Returns:
            Dictionary with clustering results
        """
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
            
            # Get distance matrix
            distance_matrix = self.maximizer.compute_feature_distance_matrix(
                feature_indices=feature_indices,
                method=distance_method,
                num_samples=num_samples
            )
            
            # Compute linkage
            Z = linkage(distance_matrix, method='ward')
            
            # Plot dendrogram
            plt.figure(figsize=(14, 10))
            dendrogram(
                Z,
                labels=feature_indices,
                leaf_rotation=90.,
                leaf_font_size=10.,
                color_threshold=0.7 * max(Z[:, 2])  # Color threshold for visual groups
            )
            plt.title(f"Hierarchical Clustering of SAE Features ({distance_method})")
            plt.xlabel("Feature Index")
            plt.ylabel("Distance")
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                print(f"Saved clustering visualization to {output_path}")
            else:
                plt.show()
            
            # Form clusters at a specific threshold (0.7 of max distance)
            clusters = fcluster(Z, 0.7 * max(Z[:, 2]), criterion='distance')
            
            # Create cluster dictionary
            cluster_dict = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_dict:
                    cluster_dict[int(cluster_id)] = []
                cluster_dict[int(cluster_id)].append(int(feature_indices[i]))
            
            return {
                "method": distance_method,
                "linkage": Z.tolist(),
                "clusters": cluster_dict,
                "feature_indices": feature_indices
            }
            
        except ImportError:
            print("SciPy not available, skipping hierarchical clustering")
            return None
    
    def prepare_saevis_data(
        self,
        feature_indices: list,
        maximization_results: dict,
        comparison_results: dict = None,
        embedding_results: dict = None,
        clustering_results: dict = None,
        output_dir: str = None,
    ):
        """
        Prepare data for SAE visualization tools
        
        Args:
            feature_indices: List of feature indices
            maximization_results: Results from maximize_feature_* methods
            comparison_results: Optional results from compare_features method
            embedding_results: Optional results from create_feature_embedding method
            clustering_results: Optional results from create_feature_hierarchy method
            output_dir: Directory to save prepared data
        
        Returns:
            Dictionary with prepared data
        """
        # Create data structure
        saevis_data = {
            "features": {},
            "feature_relationships": {}
        }
        
        # Add individual feature data
        for feat_idx in feature_indices:
            if feat_idx in maximization_results:
                result = maximization_results[feat_idx]
                
                # Extract top activating sequences
                saevis_data["features"][str(feat_idx)] = {
                    "top_activations": {
                        "texts": result["texts"][:10],  # Top 10 sequences
                        "activations": result["activations"][:10],
                        "positions": result["max_positions"][:10]
                    }
                }
        
        # Add feature relationship data
        if comparison_results:
            saevis_data["feature_relationships"]["comparison"] = {
                "method": comparison_results["method"],
                "feature_indices": comparison_results["feature_indices"]
            }
            
            if comparison_results["method"] == "activation_correlation":
                saevis_data["feature_relationships"]["comparison"]["correlation_matrix"] = comparison_results["correlation_matrix"].tolist()
            elif comparison_results["method"] == "weight_similarity":
                saevis_data["feature_relationships"]["comparison"]["similarity_matrix"] = comparison_results["similarity_matrix"].tolist()
        
        # Add embedding data
        if embedding_results:
            saevis_data["feature_relationships"]["embedding"] = embedding_results
        
        # Add clustering data
        if clustering_results:
            saevis_data["feature_relationships"]["clustering"] = clustering_results
        
        # Save data if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "saevis_data.json")
            with open(output_path, "w") as f:
                json.dump(saevis_data, f, indent=2)
            print(f"Saved SAE visualization data to {output_path}")
        
        return saevis_data
    
    def analyze_feature_group(
        self,
        feature_indices: list,
        shared_patterns: bool = True,
        output_dir: str = None,
        num_sequences: int = 10,
        seq_len: int = 20,
        iterations: int = 100,
        optimization_method: str = "discrete",
    ):
        """
        Comprehensive analysis of a group of features
        
        Args:
            feature_indices: List of feature indices to analyze
            shared_patterns: Whether to find shared patterns across features
            output_dir: Directory to save analysis results
            num_sequences: Number of sequences for maximization
            seq_len: Sequence length for maximization
            iterations: Number of iterations for maximization
            optimization_method: Method for maximization
        
        Returns:
            Dictionary with analysis results
        """
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Maximize activation for each feature
        print("Maximizing feature activations...")
        maximization_results = {}
        for feat_idx in tqdm(feature_indices):
            if optimization_method == "discrete":
                result = self.maximizer.maximize_feature_discrete(
                    feature_idx=feat_idx,
                    seq_len=seq_len,
                    num_sequences=num_sequences,
                    num_iterations=iterations
                )
            else:
                result = self.maximizer.maximize_feature_continuous(
                    feature_idx=feat_idx,
                    seq_len=seq_len,
                    num_sequences=num_sequences,
                    num_iterations=iterations
                )
            
            maximization_results[feat_idx] = result
            
            # Create and save report
            if output_dir:
                self.maximizer.create_activation_maximization_report(
                    feature_idx=feat_idx,
                    optimization_results=result,
                    output_dir=output_dir
                )
        
        # 2. Compare features
        print("Comparing features...")
        comparison_results = self.maximizer.compare_features(
            feature_indices=feature_indices,
            comparison_method="activation_correlation",
            num_samples=1000
        )
        
        # Save visualization
        if output_dir:
            self.maximizer.visualize_feature_comparison(
                comparison_results=comparison_results,
                output_path=os.path.join(output_dir, "feature_correlation.png"),
                title="Feature Activation Correlation"
            )
        
        # 3. Create feature embedding
        print("Creating feature embedding...")
        embedding_results = self.create_feature_embedding(
            feature_indices=feature_indices,
            embedding_method="tsne",
            output_path=os.path.join(output_dir, "feature_embedding.png") if output_dir else None
        )
        
        # 4. Create feature hierarchy
        print("Creating feature hierarchy...")
        clustering_results = self.create_feature_hierarchy(
            feature_indices=feature_indices,
            distance_method="activation_correlation",
            output_path=os.path.join(output_dir, "feature_hierarchy.png") if output_dir else None
        )
        
        # 5. Analyze shared patterns if requested
        shared_pattern_results = None
        if shared_patterns:
            print("Analyzing shared patterns...")
            shared_pattern_results = self._find_shared_patterns(
                feature_indices=feature_indices,
                maximization_results=maximization_results,
                comparison_results=comparison_results,
                clustering_results=clustering_results,
                output_dir=output_dir
            )
        
        # 6. Prepare data for visualization
        print("Preparing visualization data...")
        saevis_data = self.prepare_saevis_data(
            feature_indices=feature_indices,
            maximization_results=maximization_results,
            comparison_results=comparison_results,
            embedding_results=embedding_results,
            clustering_results=clustering_results,
            output_dir=output_dir
        )
        
        # 7. Compile all results
        analysis_results = {
            "feature_indices": feature_indices,
            "maximization_results": maximization_results,
            "comparison_results": comparison_results,
            "embedding_results": embedding_results,
            "clustering_results": clustering_results,
            "shared_pattern_results": shared_pattern_results,
            "saevis_data": saevis_data
        }
        
        # Save complete results
        if output_dir:
            output_path = os.path.join(output_dir, "analysis_results.json")
            try:
                # Some results might not be JSON serializable, so only save what we can
                serializable_results = {
                    "feature_indices": feature_indices,
                    "saevis_data": saevis_data,
                }
                if shared_pattern_results:
                    serializable_results["shared_pattern_results"] = shared_pattern_results
                
                with open(output_path, "w") as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"Saved analysis results to {output_path}")
            except:
                print("Could not save complete analysis results due to serialization issues")
        
        return analysis_results
    
    def _find_shared_patterns(
        self,
        feature_indices: list,
        maximization_results: dict,
        comparison_results: dict,
        clustering_results: dict,
        output_dir: str = None
    ):
        """
        Find shared patterns across features
        
        Args:
            feature_indices: List of feature indices
            maximization_results: Results from maximize_feature_* methods
            comparison_results: Results from compare_features method
            clustering_results: Results from create_feature_hierarchy method
            output_dir: Directory to save results
            
        Returns:
            Dictionary with shared pattern analysis
        """
        # This is a placeholder for more sophisticated pattern analysis
        shared_patterns = {
            "token_overlap": {},
            "clusters": {}
        }
        
        # 1. Find token overlap in top activating sequences
        for i, feat_i in enumerate(feature_indices):
            for j, feat_j in enumerate(feature_indices[i+1:], i+1):
                # Get top tokens from each feature
                tokens_i = set()
                for text in maximization_results[feat_i]["texts"][:5]:
                    tokens_i.update(text.split())
                
                tokens_j = set()
                for text in maximization_results[feat_j]["texts"][:5]:
                    tokens_j.update(text.split())
                
                # Compute overlap
                overlap = tokens_i.intersection(tokens_j)
                if len(overlap) > 0:
                    shared_patterns["token_overlap"][f"{feat_i}_{feat_j}"] = {
                        "features": [feat_i, feat_j],
                        "overlap_tokens": list(overlap),
                        "overlap_percentage": len(overlap) / min(len(tokens_i), len(tokens_j))
                    }
        
        # 2. Use clustering results if available
        if clustering_results and "clusters" in clustering_results:
            shared_patterns["clusters"] = clustering_results["clusters"]
        
        # Create a report
        if output_dir:
            report = ["# Shared Patterns Analysis", ""]
            
            # Token overlap section
            report.append("## Token Overlap Analysis")
            report.append("")
            for overlap_key, overlap_data in shared_patterns["token_overlap"].items():
                feat_i, feat_j = overlap_data["features"]
                tokens = overlap_data["overlap_tokens"]
                percentage = overlap_data["overlap_percentage"]
                
                report.append(f"### Features {feat_i} and {feat_j}")
                report.append(f"Overlap percentage: {percentage:.2f}")
                report.append("Shared tokens:")
                report.append(", ".join(tokens))
                report.append("")
            
            # Clusters section
            if "clusters" in shared_patterns:
                report.append("## Feature Clusters")
                report.append("")
                for cluster_id, features in shared_patterns["clusters"].items():
                    report.append(f"### Cluster {cluster_id}")
                    report.append(f"Features: {', '.join(map(str, features))}")
                    report.append("")
            
            # Save report
            report_path = os.path.join(output_dir, "shared_patterns_report.md")
            with open(report_path, "w") as f:
                f.write("\n".join(report))
            print(f"Saved shared patterns report to {report_path}")
        
        return shared_patterns

def main():
    parser = argparse.ArgumentParser(description="SAE Feature Explorer")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the transformer model")
    parser.add_argument("--sae", type=str, required=True, help="Path to the SAE model")
    parser.add_argument("--features", type=str, required=True, help="Feature indices to analyze (comma-separated or range)")
    parser.add_argument("--output-dir", type=str, default="feature_analysis", help="Directory to save results")
    parser.add_argument("--num-sequences", type=int, default=10, help="Number of sequences for maximization")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length for maximization")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for maximization")
    parser.add_argument("--method", type=str, choices=["discrete", "continuous"], default="discrete", 
                        help="Optimization method")
    parser.add_argument("--shared-patterns", action="store_true", help="Find shared patterns across features")
    parser.add_argument("--prepare-vis", action="store_true", help="Prepare data for visualization")
    parser.add_argument("--hook-name", type=str, default="blocks.0.mlp.hook_post", 
                        help="Hook name for extracting activations")
    parser.add_argument("--hook-layer", type=int, default=0, help="Layer index for hook")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    
    args = parser.parse_args()
    
    # Parse feature indices
    if "," in args.features:
        # Comma-separated list
        feature_indices = [int(f.strip()) for f in args.features.split(",")]
    elif "-" in args.features:
        # Range
        start, end = map(int, args.features.split("-"))
        feature_indices = list(range(start, end + 1))
    else:
        # Single feature
        feature_indices = [int(args.features)]
    
    # Load model and SAE
    model, sae, config = load_components(args.model, args.sae, args.device)
    
    # Create feature maximizer
    maximizer = FeatureMaximizer(
        model=model,
        sae=sae,
        device=args.device,
        hook_name=args.hook_name,
        hook_layer=args.hook_layer
    )
    
    # Create feature explorer
    explorer = SAEFeatureExplorer(maximizer)
    
    # Analyze feature group
    results = explorer.analyze_feature_group(
        feature_indices=feature_indices,
        shared_patterns=args.shared_patterns,
        output_dir=args.output_dir,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        iterations=args.iterations,
        optimization_method=args.method
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Prepare for visualization if requested
    if args.prepare_vis:
        try:
            from sae_vis.visualizer import SAEVisAdapter
            
            print("Preparing data for SAE visualization...")
            # Adapt SAE for visualization
            visualization_path = os.path.join(args.output_dir, "sae_visualization.html")
            
            # Generate dummy tokens for visualization
            tokens = torch.randint(0, maximizer.vocab_size, (100, args.seq_len), device=args.device)
            
            # Create visualization
            SAEVisAdapter.create_visualization(
                sae_path=args.sae,
                tokens=tokens,
                output_file=visualization_path,
                model_name=args.model,
                hook_name=args.hook_name,
                hook_layer=args.hook_layer,
                feature_indices=feature_indices,
                device=args.device
            )
            
            print(f"Visualization prepared and saved to {visualization_path}")
        except ImportError:
            print("Could not prepare visualization. Make sure sae_vis is installed.")

if __name__ == "__main__":
    main()