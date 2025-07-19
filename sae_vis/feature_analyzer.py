"""
Fast, efficient feature analysis for SAE visualization.
Designed to work with the UnifiedSAEInterface.
Now includes histogram analysis for activation and logit distributions.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .sae_interface import UnifiedSAEInterface


@dataclass
class HistogramData:
    """Container for histogram data"""
    bin_edges: List[float]
    bar_heights: List[int]
    title: str
    x_label: str
    y_label: str = "Count"


@dataclass
class ActivationExample:
    """Single example of feature activation"""
    text: str                    # The text context
    tokens: List[str]           # Tokenized text
    token_ids: List[int]        # Token IDs
    activations: List[float]    # Feature activation for each token
    positions: List[int]        # Position indices where feature is active
    max_activation: float       # Peak activation value
    seq_pos: int               # Position of peak activation


@dataclass  
class FeatureAnalysis:
    """Analysis results for a single feature"""
    feature_idx: int
    
    # Activation data
    top_examples: List[ActivationExample]  # Top activating contexts
    sparsity: float                        # Fraction of zero activations
    mean_activation: float                 # Mean of non-zero activations
    max_activation: float                  # Maximum activation seen
    
    # Logit effects
    top_boosted_tokens: List[Tuple[str, float]]    # Tokens this feature promotes
    top_suppressed_tokens: List[Tuple[str, float]] # Tokens this feature suppresses
    
    # Feature properties
    decoder_norm: float                    # L2 norm of decoder direction
    
    # Histogram data
    activation_histogram: HistogramData    # Distribution of activation values
    logit_histogram: HistogramData         # Distribution of logit effects
    

class FeatureAnalyzer:
    """
    Core engine for analyzing SAE features.
    Fast, efficient, and works with both VSAETopK and AutoEncoderTopK.
    Now includes histogram analysis.
    """
    
    def __init__(self, sae_interface: UnifiedSAEInterface, model, tokenizer):
        self.sae = sae_interface
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Move SAE to same device as model
        self.sae.sae.to(self.device)
        
    def analyze_features(
        self,
        texts: List[str],
        feature_indices: List[int],
        max_examples: int = 20,
        max_seq_len: int = 512,
        batch_size: int = 8
    ) -> Dict[int, FeatureAnalysis]:
        """
        Analyze multiple features efficiently.
        
        Args:
            texts: List of text strings to analyze
            feature_indices: Which features to analyze
            max_examples: Maximum examples per feature
            max_seq_len: Maximum sequence length
            batch_size: Processing batch size
            
        Returns:
            Dictionary mapping feature_idx -> FeatureAnalysis
        """
        print(f"Analyzing {len(feature_indices)} features on {len(texts)} texts...")
        
        # Step 1: Process all texts through model + SAE
        all_activations = []
        all_tokens = []
        all_token_ids = []
        all_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_data = self._process_text_batch(batch_texts, max_seq_len)
            
            all_activations.extend(batch_data['activations'])
            all_tokens.extend(batch_data['tokens'])
            all_token_ids.extend(batch_data['token_ids'])
            all_texts.extend(batch_data['texts'])
        
        print(f"Processed {len(all_activations)} sequences through SAE")
        
        # Step 2: Analyze each feature
        results = {}
        for feature_idx in feature_indices:
            print(f"Analyzing feature {feature_idx}...")
            results[feature_idx] = self._analyze_single_feature(
                feature_idx,
                all_activations,
                all_tokens, 
                all_token_ids,
                all_texts,
                max_examples
            )
        
        return results
    
    def _process_text_batch(self, texts: List[str], max_seq_len: int) -> Dict[str, List]:
        """Process a batch of texts through model + SAE"""
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        )
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Get model activations (assume we want the residual stream from a specific layer)
            # You'll need to adjust this based on where your SAE was trained
            if hasattr(self.model, 'run_with_cache'):
                # TransformerLens style
                _, cache = self.model.run_with_cache(input_ids, stop_at_layer=1)
                
                # Choose the right hook based on SAE's expected input dimension
                # 512D -> residual stream hooks, 2048D -> MLP hooks
                if self.sae.activation_dim == 512:
                    # Residual stream SAE (like your new model)
                    possible_hooks = [
                        'blocks.0.hook_resid_post',  # After layer 0 (512D)
                        'blocks.0.hook_resid_pre',   # Before layer 0 (512D)
                        'blocks.0.hook_resid_mid',   # Middle of layer 0 (512D)
                        'hook_embed',                # Embedding layer (512D)
                        'blocks.0.attn.hook_result', # Attention output (512D)
                    ]
                elif self.sae.activation_dim == 2048:
                    # MLP SAE (like your old model) 
                    possible_hooks = [
                        'blocks.0.mlp.hook_post',    # After MLP (2048D)
                        'blocks.0.mlp.hook_pre',     # Before MLP (2048D)
                    ]
                else:
                    # General fallback - try both
                    possible_hooks = [
                        'blocks.0.hook_resid_post',  # Residual stream
                        'blocks.0.mlp.hook_post',    # MLP 
                        'blocks.0.hook_resid_pre',   
                        'blocks.0.mlp.hook_pre',
                        'hook_embed',
                    ]
                
                activations = None
                for hook_name in possible_hooks:
                    if hook_name in cache:
                        activations = cache[hook_name]
                        print(f"Using activations from {hook_name}")
                        break
                
                if activations is None:
                    # Fallback: just use whatever we can find
                    available_hooks = list(cache.keys())
                    print(f"Available hooks: {available_hooks}")
                    if available_hooks:
                        activations = cache[available_hooks[0]]
                        print(f"Using fallback activations from {available_hooks[0]}")
                    else:
                        raise ValueError("No activations found in cache")
                        
            else:
                # Standard transformers library
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # Use the first layer's hidden states as example
                activations = outputs.hidden_states[1]  # Layer 0 output
        
        # Run through SAE
        sae_output = self.sae.encode(activations)
        
        # Debug: Print SAE output info
        print(f"   SAE encoding debug:")
        print(f"     Input activations shape: {activations.shape}")
        print(f"     Sparse features shape: {sae_output.sparse_features.shape}")
        print(f"     Top values shape: {sae_output.top_values.shape}")
        print(f"     Top indices shape: {sae_output.top_indices.shape}")
        print(f"     Non-zero sparse features: {(sae_output.sparse_features != 0).sum().item()}")
        print(f"     Max sparse activation: {sae_output.sparse_features.max().item():.6f}")
        print(f"     Mean top values: {sae_output.top_values.mean().item():.6f}")
        print(f"     Max top values: {sae_output.top_values.max().item():.6f}")
        
        # Convert back to lists for easier processing
        batch_results = {
            'activations': [],
            'tokens': [],
            'token_ids': [],
            'texts': []
        }
        
        for i, text in enumerate(texts):
            # Get valid sequence length (excluding padding)
            seq_len = attention_mask[i].sum().item()
            
            # Decode tokens for this sequence
            tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[i, :seq_len]]
            
            batch_results['activations'].append(sae_output.sparse_features[i, :seq_len].cpu())
            batch_results['tokens'].append(tokens)
            batch_results['token_ids'].append(input_ids[i, :seq_len].cpu().tolist())
            batch_results['texts'].append(text)
        
        return batch_results
    
    def _analyze_single_feature(
        self,
        feature_idx: int,
        all_activations: List[torch.Tensor],
        all_tokens: List[List[str]],
        all_token_ids: List[List[int]],
        all_texts: List[str],
        max_examples: int
    ) -> FeatureAnalysis:
        """Analyze a single feature across all sequences"""
        
        # Collect all activations for this feature
        feature_activations = []
        activation_contexts = []
        
        print(f"   Checking {len(all_activations)} sequences for feature {feature_idx}")
        
        for seq_idx, activations in enumerate(all_activations):
            feature_acts = activations[:, feature_idx]  # [seq_len]
            
            # Debug: Check if any activations are non-zero
            non_zero_count = (feature_acts > 0).sum().item()
            max_act = feature_acts.max().item()
            
            if seq_idx < 3:  # Debug first few sequences
                print(f"     Seq {seq_idx}: {non_zero_count} non-zero activations, max = {max_act:.6f}")
            
            # Find positions where this feature is active
            active_positions = (feature_acts > 0).nonzero(as_tuple=True)[0]
            
            for pos in active_positions:
                pos_idx = pos.item()
                activation_val = feature_acts[pos_idx].item()
                
                feature_activations.append(activation_val)
                activation_contexts.append({
                    'seq_idx': seq_idx,
                    'pos': pos_idx,
                    'activation': activation_val,
                    'text': all_texts[seq_idx],
                    'tokens': all_tokens[seq_idx],
                    'token_ids': all_token_ids[seq_idx]
                })
        
        print(f"   Found {len(feature_activations)} total activations for feature {feature_idx}")
        
        # Sort by activation strength
        sorted_contexts = sorted(activation_contexts, key=lambda x: x['activation'], reverse=True)
        
        # Take top examples
        top_contexts = sorted_contexts[:max_examples]
        
        # Create ActivationExample objects
        top_examples = []
        for ctx in top_contexts:
            # Create context window around the activation
            pos = ctx['pos']
            tokens = ctx['tokens']
            token_ids = ctx['token_ids']
            
            # Get activations for the entire sequence
            seq_activations = all_activations[ctx['seq_idx']][:, feature_idx]
            
            example = ActivationExample(
                text=ctx['text'],
                tokens=tokens,
                token_ids=token_ids,
                activations=seq_activations.tolist(),
                positions=[pos],  # Positions where feature is active
                max_activation=ctx['activation'],
                seq_pos=pos
            )
            top_examples.append(example)
        
        # Compute statistics
        if feature_activations:
            sparsity = 1.0 - (len(feature_activations) / sum(len(acts) for acts in all_activations))
            mean_activation = np.mean(feature_activations)
            max_activation = max(feature_activations)
        else:
            sparsity = 1.0
            mean_activation = 0.0
            max_activation = 0.0
        
        # Compute logit effects
        top_boosted, top_suppressed = self._compute_logit_effects(feature_idx)
        
        # Feature properties
        decoder_weights = self.sae.decoder_weights
        decoder_norm = torch.norm(decoder_weights[feature_idx]).item()
        
        # Generate histograms
        activation_hist = self._create_activation_histogram(feature_activations, feature_idx)
        logit_hist = self._create_logit_histogram(top_boosted, top_suppressed, feature_idx)
        
        return FeatureAnalysis(
            feature_idx=feature_idx,
            top_examples=top_examples,
            sparsity=sparsity,
            mean_activation=mean_activation,
            max_activation=max_activation,
            top_boosted_tokens=top_boosted,
            top_suppressed_tokens=top_suppressed,
            decoder_norm=decoder_norm,
            activation_histogram=activation_hist,
            logit_histogram=logit_hist
        )
    
    def _create_activation_histogram(self, activations: List[float], feature_idx: int) -> HistogramData:
        """Create histogram data for feature activations (matching Anthropic's design)"""
        if not activations:
            return HistogramData(
                bin_edges=[0, 1],
                bar_heights=[0],
                title=f"Histogram of randomly sampled non-zero activations",
                x_label="Activation Value"
            )
        
        # Filter to non-zero activations only (like Anthropic's design)
        nonzero_activations = [a for a in activations if a > 0]
        
        if not nonzero_activations:
            return HistogramData(
                bin_edges=[0, 1],
                bar_heights=[0],
                title="Histogram of randomly sampled non-zero activations",
                x_label="Activation Value"
            )
        
        # Sample randomly if we have too many points (for performance)
        if len(nonzero_activations) > 2000:
            import random
            nonzero_activations = random.sample(nonzero_activations, 2000)
        
        # Create histogram with many more bins for finer resolution
        n_unique = len(set(nonzero_activations))
        n_bins = min(50, max(15, n_unique // 3))  # 15-50 bins instead of 5-20
            
        counts, edges = np.histogram(nonzero_activations, bins=n_bins)
        
        return HistogramData(
            bin_edges=edges.tolist(),
            bar_heights=counts.tolist(),
            title="Histogram of randomly sampled non-zero activations",
            x_label="Activation Value"
        )
    
    def _create_logit_histogram(self, boosted: List[Tuple[str, float]], 
                               suppressed: List[Tuple[str, float]], feature_idx: int) -> HistogramData:
        """Create a high-resolution histogram for logit effects"""
        # Combine positive and negative effects for a continuous distribution
        all_effects = []
        
        # Add all boosted (positive) effects
        for _, effect in boosted:
            all_effects.append(effect)
        
        # Add all suppressed (negative) effects  
        for _, effect in suppressed:
            all_effects.append(effect)
        
        if not all_effects:
            return HistogramData(
                bin_edges=[-1, 0, 1],
                bar_heights=[0, 0],
                title="Top 10 negative and positive output logits",
                x_label="Logit Effect"
            )
        
        # Create high-resolution histogram
        n_unique = len(set(all_effects))
        n_bins = min(40, max(10, n_unique // 2))  # 10-40 bins instead of 2-10
        
        counts, edges = np.histogram(all_effects, bins=n_bins)
        
        return HistogramData(
            bin_edges=edges.tolist(),
            bar_heights=counts.tolist(),
            title="Top 10 negative and positive output logits",
            x_label="Logit Effect",
            y_label="Count"
        )
    
    def _compute_logit_effects(self, feature_idx: int, top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Compute which tokens this feature promotes/suppresses.
        Handles the projection from SAE decoder space to logit space correctly.
        """
        try:
            # Get the decoder direction for this feature
            decoder_direction = self.sae.decoder_weights[feature_idx]  # Shape depends on SAE type
            
            # For VSAETopK trained on MLP activations, we need to project to residual stream first
            if hasattr(self.model, 'W_out') and decoder_direction.shape[0] != self.model.cfg.d_model:
                # This is likely a VSAETopK trained on MLP activations (2048D -> 512D projection needed)
                print(f"   Projecting from {decoder_direction.shape[0]}D MLP space to {self.model.cfg.d_model}D residual space")
                
                # Project through MLP output matrix: decoder_direction @ W_out
                # W_out shape: [d_mlp, d_model] = [2048, 512] for gelu-1l
                W_out = self.model.W_out[0]  # Layer 0's MLP output matrix
                residual_direction = decoder_direction @ W_out  # [2048] @ [2048, 512] -> [512]
            else:
                # Direct residual stream SAE
                residual_direction = decoder_direction
            
            # Now project to logit space
            if hasattr(self.model, 'W_U'):
                # TransformerLens style
                unembedding = self.model.W_U  # [d_model, vocab_size] = [512, vocab_size]
            elif hasattr(self.model, 'lm_head'):
                # Standard transformers library
                unembedding = self.model.lm_head.weight.T  # [hidden_size, vocab_size]
            else:
                # Can't compute logit effects without unembedding matrix
                return [], []
            
            # Compute logit effects: how much this feature changes each token's logit
            logit_effects = residual_direction @ unembedding  # [512] @ [512, vocab_size] -> [vocab_size]
            
            # Get top boosted and suppressed tokens
            top_indices = torch.topk(logit_effects, top_k).indices
            bottom_indices = torch.topk(logit_effects, top_k, largest=False).indices
            
            top_boosted = []
            for idx in top_indices:
                token = self.tokenizer.decode([idx.item()])
                effect = logit_effects[idx].item()
                top_boosted.append((token, effect))
            
            top_suppressed = []
            for idx in bottom_indices:
                token = self.tokenizer.decode([idx.item()])
                effect = logit_effects[idx].item()
                top_suppressed.append((token, effect))
            
            return top_boosted, top_suppressed
            
        except Exception as e:
            print(f"Warning: Could not compute logit effects for feature {feature_idx}: {e}")
            return [], []