import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from pathlib import Path

# Function to load the necessary components 
def load_components(
    model_path: str,
    sae_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple:
    """
    Load the transformer model and SAE for feature activation
    
    Args:
        model_path: Path to the transformer model or name of model to load
        sae_path: Path to the SAE model
        device: Device to load models on
    
    Returns:
        Tuple of (transformer_model, sae_model)
    """
    # Import necessary libraries based on model type
    try:
        from transformer_lens import HookedTransformer
        from dictionary_learning.utils import load_dictionary
        
        # Load model - try transformer lens first
        model = HookedTransformer.from_pretrained(model_path, device=device)
        
        # Load SAE
        sae, config = load_dictionary(sae_path, device)
        
        return model, sae, config
    except Exception as e:
        print(f"Error loading with TransformerLens: {e}")
        
        # Try with HuggingFace transformers
        try:
            from transformers import AutoModel, AutoTokenizer
            from dictionary_learning.dictionary import AutoEncoder
            
            model = AutoModel.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # For this case, load SAE directly
            sae_state = torch.load(sae_path)
            
            # Get SAE dimensions from state dict
            if "encoder.weight" in sae_state:
                dict_size, activation_dim = sae_state["encoder.weight"].shape
                sae = AutoEncoder(activation_dim, dict_size).to(device)
                sae.load_state_dict(sae_state)
                return model, sae, tokenizer
            else:
                raise ValueError(f"Unexpected SAE state dict format")
        except Exception as e2:
            raise ValueError(f"Could not load models: {e}, then {e2}")

class FeatureMaximizer:
    """
    Class to find inputs that maximize activation of specific SAE features.
    """
    def __init__(
        self,
        model,
        sae,
        tokenizer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hook_name="blocks.0.mlp.hook_post",
        hook_layer=0
    ):
        self.model = model
        self.sae = sae
        self.device = device
        self.hook_name = hook_name
        self.hook_layer = hook_layer
        
        # Determine if model is TransformerLens or HuggingFace
        self.model_type = 'transformer_lens' if hasattr(model, 'cfg') else 'huggingface'
        
        # Set tokenizer based on model type
        if self.model_type == 'transformer_lens':
            self.tokenizer = model.tokenizer
        else:
            self.tokenizer = tokenizer
            
        # Get model dimensions
        if self.model_type == 'transformer_lens':
            self.d_model = model.cfg.d_model
            self.vocab_size = model.cfg.vocab_size
        else:
            self.d_model = model.config.hidden_size
            self.vocab_size = model.config.vocab_size
            
    def get_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get activations at specific hook point for the given input tokens
        
        Args:
            tokens: Batch of token ids [batch, seq_len]
            
        Returns:
            Activations at hook point [batch, seq_len, d_model]
        """
        if self.model_type == 'transformer_lens':
            # Use TransformerLens hook system
            def hook_fn(act, hook):
                return act
                
            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=self.hook_name
                )
                return cache[self.hook_name]
        else:
            # For HuggingFace, extract activations manually
            with torch.no_grad():
                outputs = self.model(
                    tokens,
                    output_hidden_states=True
                )
                # Get hidden states at specific layer
                hidden_states = outputs.hidden_states[self.hook_layer]
                # We need to extract the specific activation based on hook_name
                # This will depend on the actual model architecture
                if 'mlp' in self.hook_name:
                    # For now, assume we want the MLP output
                    return hidden_states
                else:
                    return hidden_states
    
    def get_feature_activations(self, tokens: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """
        Get specific feature activation values for the input tokens
        
        Args:
            tokens: Batch of token ids [batch, seq_len]
            feature_idx: Index of the feature to get activations for
            
        Returns:
            Feature activations [batch, seq_len]
        """
        activations = self.get_activations(tokens)
        
        # Encode with SAE
        with torch.no_grad():
            batch_size, seq_len, _ = activations.shape
            # Reshape to 2D for SAE
            reshaped_acts = activations.reshape(-1, activations.shape[-1])
            # Get feature activations
            feature_acts = self.sae.encode(reshaped_acts)
            # Extract specific feature
            specific_feature = feature_acts[:, feature_idx]
            # Reshape back
            return specific_feature.reshape(batch_size, seq_len)
    
    def maximize_feature_discrete(
        self,
        feature_idx: int,
        seq_len: int = 20,
        num_sequences: int = 10,
        num_iterations: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        seed_texts: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Find input sequences that maximize a specific feature using discrete optimization
        
        Args:
            feature_idx: Index of the feature to maximize
            seq_len: Length of sequences to generate
            num_sequences: Number of sequences to generate
            num_iterations: Number of iterations of optimization
            temperature: Temperature for token sampling
            top_k: Number of top tokens to consider in sampling
            seed_texts: Optional list of seed texts to start from
            
        Returns:
            Dictionary with maximizing sequences and their activations
        """
        # Initialize sequences - either randomly or from seed texts
        if seed_texts:
            assert len(seed_texts) == num_sequences, "Number of seed texts must match num_sequences"
            token_seqs = [self.tokenizer.encode(text, return_tensors="pt").to(self.device) for text in seed_texts]
            # Pad or truncate to seq_len
            token_seqs = [self._pad_or_truncate(seq, seq_len) for seq in token_seqs]
            tokens = torch.stack(token_seqs)
        else:
            # Initialize with random tokens
            tokens = torch.randint(0, self.vocab_size, (num_sequences, seq_len), device=self.device)
        
        best_tokens = tokens.clone()
        best_activations = torch.zeros(num_sequences, device=self.device)
        
        progress_bar = tqdm(range(num_iterations))
        
        for _ in progress_bar:
            # For each position in the sequence
            for pos in range(seq_len):
                # Get current feature activations
                feature_acts = self.get_feature_activations(tokens, feature_idx)
                
                # For each sequence, try different tokens at the current position
                for seq_idx in range(num_sequences):
                    original_token = tokens[seq_idx, pos].item()
                    best_token = original_token
                    best_act = feature_acts[seq_idx, pos].item()
                    
                    # Try top_k random tokens
                    candidate_tokens = torch.randint(0, self.vocab_size, (top_k,), device=self.device)
                    
                    for cand_token in candidate_tokens:
                        tokens[seq_idx, pos] = cand_token
                        # Get activation with this token
                        new_act = self.get_feature_activations(tokens[seq_idx:seq_idx+1], feature_idx)[0, pos].item()
                        
                        if new_act > best_act:
                            best_token = cand_token.item()
                            best_act = new_act
                    
                    # Restore the best token at this position
                    tokens[seq_idx, pos] = best_token
            
            # After optimizing all positions, check if we improved the overall activation
            current_acts = self.get_feature_activations(tokens, feature_idx).max(dim=1)[0]
            
            # Update best tokens and activations
            improved = current_acts > best_activations
            best_tokens[improved] = tokens[improved]
            best_activations[improved] = current_acts[improved]
            
            # Update progress bar
            progress_bar.set_description(f"Best activation: {best_activations.max().item():.4f}")
        
        # Convert tokens back to text
        texts = [self.tokenizer.decode(seq) for seq in best_tokens]
        
        # Get position of max activation for each sequence
        with torch.no_grad():
            final_acts = self.get_feature_activations(best_tokens, feature_idx)
            max_pos = final_acts.argmax(dim=1)
        
        return {
            "texts": texts,
            "tokens": best_tokens,
            "activations": best_activations.tolist(),
            "max_positions": max_pos.tolist(),
            "feature_idx": feature_idx
        }
    
    def maximize_feature_continuous(
        self,
        feature_idx: int,
        seq_len: int = 20,
        num_sequences: int = 10,
        num_iterations: int = 200,
        learning_rate: float = 0.1,
        l2_reg: float = 0.01,
        seed_texts: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Find input embeddings that maximize a specific feature using continuous optimization
        
        Args:
            feature_idx: Index of the feature to maximize
            seq_len: Length of sequences to generate
            num_sequences: Number of sequences to generate
            num_iterations: Number of iterations of optimization
            learning_rate: Learning rate for optimizer
            l2_reg: L2 regularization strength
            seed_texts: Optional list of seed texts to start from
            
        Returns:
            Dictionary with maximizing sequences and their activations
        """
        # Get embedding matrix
        if self.model_type == 'transformer_lens':
            embedding_matrix = self.model.W_E
        else:
            embedding_matrix = self.model.get_input_embeddings().weight
        
        # Initialize embeddings - either randomly or from seed texts
        if seed_texts:
            assert len(seed_texts) == num_sequences, "Number of seed texts must match num_sequences"
            token_seqs = [self.tokenizer.encode(text, return_tensors="pt").to(self.device) for text in seed_texts]
            # Pad or truncate to seq_len
            token_seqs = [self._pad_or_truncate(seq, seq_len) for seq in token_seqs]
            tokens = torch.stack(token_seqs)
            # Convert tokens to embeddings
            embeddings = self._tokens_to_embeddings(tokens, embedding_matrix)
        else:
            # Initialize with random embeddings in the embedding space
            embeddings = torch.randn(num_sequences, seq_len, self.d_model, device=self.device)
            # Normalize to match typical embedding magnitudes
            embeddings = embeddings * (embedding_matrix.std() / embeddings.std())
        
        # Make embeddings a parameter that requires grad
        embeddings = nn.Parameter(embeddings)
        
        # Setup optimizer
        optimizer = optim.Adam([embeddings], lr=learning_rate)
        
        best_embeddings = embeddings.clone().detach()
        best_activations = torch.zeros(num_sequences, device=self.device)
        
        progress_bar = tqdm(range(num_iterations))
        
        for _ in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass through model (custom implementation)
            if self.model_type == 'transformer_lens':
                # Use model's forward pass but replace embedding lookup
                activations = self._forward_with_embeddings(embeddings)
            else:
                # For HuggingFace, implement a custom forward pass
                activations = self._forward_with_embeddings_hf(embeddings)
            
            # Encode with SAE
            batch_size, seq_len, _ = activations.shape
            reshaped_acts = activations.reshape(-1, activations.shape[-1])
            feature_acts = self.sae.encode(reshaped_acts)
            specific_feature = feature_acts[:, feature_idx]
            specific_feature = specific_feature.reshape(batch_size, seq_len)
            
            # Loss is negative feature activation (we want to maximize)
            loss = -specific_feature.mean()
            
            # Add L2 regularization to keep embeddings in a reasonable range
            l2_loss = l2_reg * embeddings.pow(2).mean()
            loss = loss + l2_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Get current max activations
            current_acts = specific_feature.detach().max(dim=1)[0]
            
            # Update best embeddings and activations
            improved = current_acts > best_activations
            if improved.any():
                best_embeddings[improved] = embeddings.detach()[improved]
                best_activations[improved] = current_acts[improved]
            
            # Update progress bar
            progress_bar.set_description(f"Best activation: {best_activations.max().item():.4f}")
        
        # Convert best embeddings to nearest tokens
        best_tokens = self._embeddings_to_tokens(best_embeddings, embedding_matrix)
        texts = [self.tokenizer.decode(seq) for seq in best_tokens]
        
        # Get position of max activation for each sequence
        with torch.no_grad():
            # Forward pass with best embeddings
            if self.model_type == 'transformer_lens':
                activations = self._forward_with_embeddings(best_embeddings)
            else:
                activations = self._forward_with_embeddings_hf(best_embeddings)
            
            # Get feature activations
            batch_size, seq_len, _ = activations.shape
            reshaped_acts = activations.reshape(-1, activations.shape[-1])
            feature_acts = self.sae.encode(reshaped_acts)
            specific_feature = feature_acts[:, feature_idx]
            final_acts = specific_feature.reshape(batch_size, seq_len)
            max_pos = final_acts.argmax(dim=1)
        
        return {
            "texts": texts,
            "tokens": best_tokens,
            "embeddings": best_embeddings.detach(),
            "activations": best_activations.tolist(),
            "max_positions": max_pos.tolist(),
            "feature_idx": feature_idx
        }
    
    def _tokens_to_embeddings(self, tokens, embedding_matrix):
        """Convert token IDs to embeddings"""
        return embedding_matrix[tokens]
    
    def _embeddings_to_tokens(self, embeddings, embedding_matrix):
        """Convert embeddings to nearest token IDs"""
        # Normalize embeddings and embedding matrix for cosine similarity
        embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-7)
        embedding_matrix_norm = embedding_matrix / (embedding_matrix.norm(dim=-1, keepdim=True) + 1e-7)
        
        # Compute cosine similarity
        similarity = torch.matmul(embeddings_norm.reshape(-1, self.d_model), embedding_matrix_norm.T)
        
        # Get indices of most similar tokens
        tokens = similarity.argmax(dim=-1).reshape(embeddings.shape[0], embeddings.shape[1])
        return tokens
    
    def _forward_with_embeddings(self, embeddings):
        """
        Custom forward pass for TransformerLens models using provided embeddings
        """
        # This is a simplified implementation - actual implementation would depend on model details
        # In practice, you'd need to handle position embeddings, etc.
        
        # Get activations at the hook point
        hidden_states = embeddings
        for layer_idx in range(self.hook_layer + 1):
            # Apply layer
            layer = self.model.blocks[layer_idx]
            hidden_states = layer(hidden_states)
            
            # If we've reached the target layer and hook is mid-MLP, return here
            if layer_idx == self.hook_layer and 'mlp.hook_post' in self.hook_name:
                return layer.mlp.hook_post
        
        # If hook is at end of layer, return current hidden states
        return hidden_states
    
    def _forward_with_embeddings_hf(self, embeddings):
        """
        Custom forward pass for HuggingFace models using provided embeddings
        """
        # This is a placeholder - implementation depends on model architecture
        # In practice, you'd need a custom forward pass for the specific model
        raise NotImplementedError("Custom forward pass for HuggingFace models not implemented")
    
    def _pad_or_truncate(self, token_seq, target_len):
        """Pad or truncate token sequence to target length"""
        current_len = token_seq.shape[1]
        if current_len < target_len:
            # Pad
            padding = torch.zeros(1, target_len - current_len, dtype=torch.long, device=self.device)
            return torch.cat([token_seq, padding], dim=1)
        elif current_len > target_len:
            # Truncate
            return token_seq[:, :target_len]
        else:
            return token_seq
    
    def compare_features(
        self,
        feature_indices: List[int],
        comparison_method: str = "activation_correlation",
        num_samples: int = 1000,
        tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple features to understand their relationships and distances
        
        Args:
            feature_indices: List of feature indices to compare
            comparison_method: Method to use for comparison
                - "activation_correlation": Correlation of feature activations
                - "weight_similarity": Similarity of feature weights
                - "output_similarity": Similarity of feature effects on output
            num_samples: Number of samples to use for activation correlation
            tokens: Optional tensor of tokens to use for activation correlation
            
        Returns:
            Dictionary with comparison results
        """
        if comparison_method == "activation_correlation":
            # Get random tokens if none provided
            if tokens is None:
                # Generate random token sequences
                tokens = torch.randint(0, self.vocab_size, (num_samples, 20), device=self.device)
            
            # Get activations for all features
            activations = self.get_activations(tokens)
            batch_size, seq_len, _ = activations.shape
            reshaped_acts = activations.reshape(-1, activations.shape[-1])
            
            # Get feature activations for selected features
            with torch.no_grad():
                all_feature_acts = self.sae.encode(reshaped_acts)
                selected_feature_acts = all_feature_acts[:, feature_indices]
            
            # Compute correlation matrix
            correlation_matrix = np.corrcoef(selected_feature_acts.cpu().numpy().T)
            
            return {
                "method": "activation_correlation",
                "feature_indices": feature_indices,
                "correlation_matrix": correlation_matrix,
            }
            
        elif comparison_method == "weight_similarity":
            # Get feature weights
            if hasattr(self.sae, 'decoder') and hasattr(self.sae.decoder, 'weight'):
                decoder_weights = self.sae.decoder.weight
                # Extract weights for selected features
                selected_weights = decoder_weights[:, feature_indices].T.cpu().numpy()
            elif hasattr(self.sae, 'W_dec'):
                decoder_weights = self.sae.W_dec
                # Extract weights for selected features (may need transposing depending on format)
                selected_weights = decoder_weights[feature_indices].cpu().numpy()
            else:
                raise ValueError("Could not access feature weights")
            
            # Compute cosine similarity
            norm_weights = selected_weights / np.linalg.norm(selected_weights, axis=1, keepdims=True)
            similarity_matrix = np.dot(norm_weights, norm_weights.T)
            
            return {
                "method": "weight_similarity",
                "feature_indices": feature_indices,
                "similarity_matrix": similarity_matrix,
            }
            
        elif comparison_method == "output_similarity":
            # This would compare how features affect model outputs
            # Implementation depends on model architecture and goals
            raise NotImplementedError("Output similarity comparison not implemented")
            
        else:
            raise ValueError(f"Unknown comparison method: {comparison_method}")
    
    def visualize_feature_comparison(
        self,
        comparison_results: Dict[str, Any],
        output_path: str = None,
        title: str = None
    ):
        """
        Visualize feature comparison results
        
        Args:
            comparison_results: Results from compare_features method
            output_path: Path to save visualization
            title: Optional title for the visualization
        """
        method = comparison_results["method"]
        feature_indices = comparison_results["feature_indices"]
        
        if method == "activation_correlation":
            matrix = comparison_results["correlation_matrix"]
            matrix_title = "Activation Correlation"
        elif method == "weight_similarity":
            matrix = comparison_results["similarity_matrix"]
            matrix_title = "Weight Cosine Similarity"
        else:
            raise ValueError(f"Cannot visualize results for method: {method}")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar()
        
        # Set feature indices as tick labels
        plt.xticks(np.arange(len(feature_indices)), feature_indices)
        plt.yticks(np.arange(len(feature_indices)), feature_indices)
        
        # Rotate x tick labels for better readability
        plt.xticks(rotation=90)
        
        # Add title
        if title:
            plt.title(title)
        else:
            plt.title(f"Feature Comparison: {matrix_title}")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
            
    def compute_feature_distance_matrix(
        self,
        feature_indices: List[int],
        method: str = "activation_correlation",
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Compute distance matrix between features
        
        Args:
            feature_indices: List of feature indices to compare
            method: Method to use for distance computation
            num_samples: Number of samples for activation correlation
            
        Returns:
            Distance matrix (numpy array)
        """
        comparison = self.compare_features(
            feature_indices=feature_indices,
            comparison_method=method,
            num_samples=num_samples
        )
        
        if method == "activation_correlation":
            # Convert correlation to distance: 1 - |correlation|
            # This means highly correlated (positive or negative) features are close
            similarity = comparison["correlation_matrix"]
            distance = 1 - np.abs(similarity)
            
        elif method == "weight_similarity":
            # Convert cosine similarity to distance: 1 - similarity
            similarity = comparison["similarity_matrix"]
            distance = 1 - similarity
            
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        return distance
            
    def create_activation_maximization_report(
        self,
        feature_idx: int,
        optimization_results: Dict[str, Any],
        output_dir: str = None,
        include_tokens: bool = False
    ):
        """
        Create a report for activation maximization results
        
        Args:
            feature_idx: Feature index
            optimization_results: Results from maximize_feature_* methods
            output_dir: Directory to save report
            include_tokens: Whether to include tokens in report
            
        Returns:
            Path to report file
        """
        texts = optimization_results["texts"]
        activations = optimization_results["activations"]
        max_positions = optimization_results["max_positions"]
        
        # Create report string
        report = [
            f"# Activation Maximization for Feature {feature_idx}",
            "",
            f"Maximum activation achieved: {max(activations):.4f}",
            "",
            "## Top activating sequences:",
            ""
        ]
        
        # Sort by activation value
        sorted_indices = np.argsort(activations)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            act = activations[idx]
            text = texts[idx]
            pos = max_positions[idx]
            
            # Highlight the token with max activation
            tokens = text.split()
            if pos < len(tokens):
                tokens[pos] = f"**{tokens[pos]}**"
                highlighted_text = " ".join(tokens)
            else:
                highlighted_text = text
                
            report.append(f"{i+1}. Activation: {act:.4f}")
            report.append(f"   Text: {highlighted_text}")
            
            if include_tokens:
                report.append(f"   Tokens: {optimization_results['tokens'][idx].tolist()}")
                
            report.append("")
        
        # Join report lines
        report_text = "\n".join(report)
        
        # Save report if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"feature_{feature_idx}_report.md")
            with open(report_path, "w") as f:
                f.write(report_text)
            return report_path
        
        return report_text

def main():
    parser = argparse.ArgumentParser(description="Maximize SAE Feature Activations")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the transformer model")
    parser.add_argument("--sae", type=str, required=True, help="Path to the SAE model")
    parser.add_argument("--features", type=str, required=True, help="Feature indices to maximize (comma-separated or range)")
    parser.add_argument("--method", type=str, choices=["discrete", "continuous"], default="discrete", 
                        help="Optimization method")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    parser.add_argument("--num-sequences", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--compare", action="store_true", help="Compare features in addition to maximization")
    parser.add_argument("--output-dir", type=str, default="feature_activations", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    parser.add_argument("--hook-name", type=str, default="blocks.0.mlp.hook_post", 
                        help="Hook name for extracting activations")
    parser.add_argument("--hook-layer", type=int, default=0, help="Layer index for hook")
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Optimize for each feature
    results = {}
    for feature_idx in feature_indices:
        print(f"\nMaximizing feature {feature_idx}...")
        
        if args.method == "discrete":
            result = maximizer.maximize_feature_discrete(
                feature_idx=feature_idx,
                seq_len=args.seq_len,
                num_sequences=args.num_sequences,
                num_iterations=args.iterations
            )
        else:
            result = maximizer.maximize_feature_continuous(
                feature_idx=feature_idx,
                seq_len=args.seq_len,
                num_sequences=args.num_sequences,
                num_iterations=args.iterations
            )
        
        results[feature_idx] = result
        
        # Create and save report
        report_path = maximizer.create_activation_maximization_report(
            feature_idx=feature_idx,
            optimization_results=result,
            output_dir=args.output_dir
        )
        print(f"Report saved to {report_path}")
    
    # Compare features if requested
    if args.compare and len(feature_indices) > 1:
        print("\nComparing features...")
        
        # Compare with activation correlation
        act_comparison = maximizer.compare_features(
            feature_indices=feature_indices,
            comparison_method="activation_correlation",
            num_samples=1000
        )
        
        # Save visualization
        maximizer.visualize_feature_comparison(
            comparison_results=act_comparison,
            output_path=os.path.join(args.output_dir, "feature_correlation.png"),
            title=f"Feature Activation Correlation for {args.features}"
        )
        
        # Compare with weight similarity
        weight_comparison = maximizer.compare_features(
            feature_indices=feature_indices,
            comparison_method="weight_similarity"
        )
        
# Save visualization
        maximizer.visualize_feature_comparison(
            comparison_results=weight_comparison,
            output_path=os.path.join(args.output_dir, "feature_weight_similarity.png"),
            title=f"Feature Weight Similarity for {args.features}"
        )
        
        # Compute distance matrix for hierarchical clustering
        distance_matrix = maximizer.compute_feature_distance_matrix(
            feature_indices=feature_indices,
            method="activation_correlation"
        )
        
        # Save distance matrix
        np.save(os.path.join(args.output_dir, "feature_distance_matrix.npy"), distance_matrix)
        
        # Perform hierarchical clustering
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            
            # Compute linkage
            Z = linkage(distance_matrix, method='ward')
            
            # Plot dendrogram
            plt.figure(figsize=(12, 8))
            dendrogram(
                Z,
                labels=feature_indices,
                leaf_rotation=90.,
                leaf_font_size=10.,
            )
            plt.title(f"Hierarchical Clustering of Features {args.features}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "feature_clustering.png"))
            print(f"Clustering visualization saved to {os.path.join(args.output_dir, 'feature_clustering.png')}")
        except ImportError:
            print("SciPy not available, skipping hierarchical clustering")
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()