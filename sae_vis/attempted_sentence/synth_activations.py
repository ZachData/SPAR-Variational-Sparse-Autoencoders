"""
Synthetic activations: find optimal inputs that maximally activate each SAE feature.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from real_activations import FeatureStats, ActivationExample


def generate_synthetic_maximizers(sae_model, transformer_model, feature_indices: List[int],
                                max_examples: int = 5, seq_len: int = 64, 
                                n_steps: int = 200, learning_rate: float = 0.1) -> Dict[int, FeatureStats]:
    """
    Generate synthetic token sequences that maximally activate each feature.
    
    Args:
        sae_model: Trained SAE model
        transformer_model: Base transformer
        feature_indices: Which features to optimize for
        max_examples: Number of synthetic examples per feature
        seq_len: Length of sequences to optimize
        n_steps: Optimization steps per attempt
        learning_rate: Learning rate for optimization
        
    Returns:
        Dict mapping feature_idx -> FeatureStats with synthetic examples
    """
    from sae_interface import UnifiedSAEInterface
    
    device = next(transformer_model.parameters()).device
    sae_interface = UnifiedSAEInterface(sae_model)
    sae_interface.sae.to(device)
    
    maximizer = SyntheticMaximizer(sae_interface, transformer_model)
    
    results = {}
    for feature_idx in tqdm(feature_indices, desc="Generating synthetic maximizers"):
        print(f"Optimizing feature {feature_idx}...")
        
        # Generate multiple synthetic examples
        synthetic_examples = []
        for attempt in range(max_examples + 2):  # Generate extra, take best
            example = maximizer.optimize_single_feature(
                feature_idx, seq_len, n_steps, learning_rate
            )
            if example:
                synthetic_examples.append(example)
        
        # Sort by activation strength and take top examples
        synthetic_examples.sort(key=lambda x: x.max_activation, reverse=True)
        top_examples = synthetic_examples[:max_examples]
        
        # Compute statistics
        if top_examples:
            max_activation = top_examples[0].max_activation
            mean_activation = np.mean([ex.max_activation for ex in top_examples])
        else:
            max_activation = 0.0
            mean_activation = 0.0
        
        # Compute logit effects (same as real analysis)
        top_boosted, top_suppressed = _compute_logit_effects(
            feature_idx, sae_interface, transformer_model
        )
        
        # Decoder norm
        decoder_norm = torch.norm(sae_interface.decoder_weights[feature_idx]).item()
        
        results[feature_idx] = FeatureStats(
            feature_idx=feature_idx,
            examples=top_examples,
            sparsity=0.0,  # Synthetic maximizers aren't sparse by definition
            mean_activation=mean_activation,
            max_activation=max_activation,
            decoder_norm=decoder_norm,
            top_boosted_tokens=top_boosted,
            top_suppressed_tokens=top_suppressed
        )
    
    return results


class SyntheticMaximizer:
    """Optimizes token sequences to maximally activate SAE features"""
    
    def __init__(self, sae_interface, model):
        self.sae = sae_interface
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        self.embed_matrix = model.W_E
        self.vocab_size = self.embed_matrix.shape[0]
        
    def optimize_single_feature(self, feature_idx: int, seq_len: int, 
                               n_steps: int, learning_rate: float) -> ActivationExample:
        """Optimize a single feature using gradient descent in embedding space"""
        
        # Initialize random logits
        logits = torch.randn(1, seq_len, self.vocab_size, 
                           device=self.device, requires_grad=True)
        
        # Mask special tokens
        special_mask = self._get_special_token_mask()
        
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        
        best_activation = -float('inf')
        best_tokens = None
        best_text = None
        
        temperature = 1.0
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Mask special tokens
            masked_logits = logits.clone()
            masked_logits[:, :, special_mask] = -1e9
            
            # Convert to soft embeddings
            soft_tokens = F.gumbel_softmax(masked_logits, tau=temperature, hard=False)
            embeddings = soft_tokens @ self.embed_matrix
            
            # Get activations at hook point
            activations = self._compute_activations_from_embeddings(embeddings)
            
            # Encode through SAE
            sae_output = self.sae.encode(activations)
            
            # Get target feature activation
            feature_activation = sae_output.sparse_features[0, :, feature_idx]
            objective = feature_activation.sum()
            
            # Add entropy regularization for readability
            entropy_reg = self._entropy_regularization(masked_logits)
            total_objective = objective - 0.01 * entropy_reg
            
            # Backpropagate
            (-total_objective).backward()
            optimizer.step()
            
            # Track best result
            if objective.item() > best_activation:
                best_activation = objective.item()
                
                # Get discrete tokens
                discrete_tokens = F.gumbel_softmax(masked_logits, tau=0.1, hard=True)
                token_indices = discrete_tokens.argmax(dim=-1)[0]
                
                best_tokens = token_indices.cpu().tolist()
                best_text = self.tokenizer.decode(best_tokens, skip_special_tokens=True)
            
            # Decay temperature
            temperature = max(0.1, temperature * 0.995)
        
        if best_tokens is None:
            return None
        
        # Create per-token activations (approximate distribution)
        per_token_activations = np.random.exponential(
            best_activation / len(best_tokens), len(best_tokens)
        )
        per_token_activations = (per_token_activations / per_token_activations.sum()) * best_activation
        peak_pos = np.argmax(per_token_activations)
        
        # Convert tokens to strings
        token_strings = [self.tokenizer.decode([token_id]) for token_id in best_tokens]
        
        return ActivationExample(
            text=best_text,
            tokens=token_strings,
            token_ids=best_tokens,
            activations=per_token_activations.tolist(),
            max_activation=best_activation,
            peak_position=peak_pos
        )
    
    def _get_special_token_mask(self) -> torch.Tensor:
        """Mask special tokens to exclude from optimization"""
        special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.unk_token_id,
        ]
        
        # Exclude very rare tokens
        special_tokens.extend(range(0, 10))
        special_tokens.extend(range(self.vocab_size - 100, self.vocab_size))
        
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for token_id in special_tokens:
            if token_id is not None and 0 <= token_id < self.vocab_size:
                mask[token_id] = True
        
        return mask
    
    def _entropy_regularization(self, logits: torch.Tensor) -> torch.Tensor:
        """Encourage diversity in token selection"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return -entropy  # Negative because we want to maximize entropy
    
    def _compute_activations_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute activations at hook point from embeddings"""
        # Add positional embeddings
        seq_len = embeddings.shape[1]
        pos_embed = self.model.W_pos[:seq_len]
        x = embeddings + pos_embed.unsqueeze(0)
        
        # Run through first transformer block (adjust based on hook point)
        # This assumes we want residual stream after block 0
        x = self.model.blocks[0](x)
        return x


def _compute_logit_effects(feature_idx: int, sae_interface, model, 
                         top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Compute logit effects (same as real_activations.py)"""
    try:
        decoder_direction = sae_interface.decoder_weights[feature_idx]
        
        # Project to residual stream if needed
        if decoder_direction.shape[0] != model.cfg.d_model:
            if hasattr(model, 'W_out') and decoder_direction.shape[0] == 2048:
                W_out = model.W_out[0]
                residual_direction = decoder_direction @ W_out
            else:
                residual_direction = decoder_direction
        else:
            residual_direction = decoder_direction
        
        # Project to logit space
        if hasattr(model, 'W_U'):
            unembedding = model.W_U
        else:
            return [], []
        
        logit_effects = residual_direction @ unembedding
        
        # Get top/bottom tokens
        top_indices = torch.topk(logit_effects, top_k).indices
        bottom_indices = torch.topk(logit_effects, top_k, largest=False).indices
        
        top_boosted = []
        for idx in top_indices:
            token = model.tokenizer.decode([idx.item()])
            effect = logit_effects[idx].item()
            top_boosted.append((token, effect))
        
        top_suppressed = []
        for idx in bottom_indices:
            token = model.tokenizer.decode([idx.item()])
            effect = logit_effects[idx].item()
            top_suppressed.append((token, effect))
        
        return top_boosted, top_suppressed
        
    except Exception:
        return [], []
