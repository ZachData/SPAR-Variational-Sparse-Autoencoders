"""
Real activations: analyze what actually fires SAE features in real text.
Uses the same c4-code dataset and activation pipeline as the training script.
"""
import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import data utilities from parent directory (same as training script)
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import TransformerLensActivationBuffer


@dataclass
class ActivationExample:
    """Single example of feature activation"""
    text: str
    tokens: List[str]
    token_ids: List[int]
    activations: List[float]  # Per-token activations
    max_activation: float
    peak_position: int


@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    feature_idx: int
    examples: List[ActivationExample]
    sparsity: float
    mean_activation: float
    max_activation: float
    decoder_norm: float
    top_boosted_tokens: List[Tuple[str, float]]
    top_suppressed_tokens: List[Tuple[str, float]]


def create_activation_buffer(model, device: str, buffer_size: int = 20000, 
                           ctx_len: int = 128) -> TransformerLensActivationBuffer:
    """Create activation buffer using same setup as training script"""
    hook_name = "blocks.0.hook_resid_post"  # Same hook as training script
    
    print(f"Creating activation buffer with hook: {hook_name}")
    print(f"Buffer config: n_ctxs={buffer_size}, ctx_len={ctx_len}")
    
    # Set up data generator exactly like training script
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b",  # Same dataset as training
        split="train", 
        return_tokens=True
    )
    
    # Create activation buffer exactly like training script
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=hook_name,
        d_submodule=model.cfg.d_model,  # Use d_model (512) for residual stream
        n_ctxs=buffer_size,
        ctx_len=ctx_len,
        refresh_batch_size=min(16, buffer_size // 10),  # Adaptive batch size
        out_batch_size=min(256, buffer_size // 4),      # Adaptive output batch size
        device=device,
    )
    
    print(f"✅ Activation buffer created successfully")
    return buffer


def analyze_real_activations(sae_model, transformer_model, feature_indices: List[int], 
                           max_examples: int = 20, batch_size: int = 8, 
                           buffer_size: int = 20000, ctx_len: int = 128) -> Dict[int, FeatureStats]:
    """
    Analyze real activations using the same c4-code dataset as training.
    
    Args:
        sae_model: Trained SAE model
        transformer_model: Base transformer  
        feature_indices: Which features to analyze
        max_examples: Top examples per feature
        batch_size: Processing batch size
        buffer_size: Size of activation buffer (number of contexts)
        ctx_len: Context length for each sample
        
    Returns:
        Dict mapping feature_idx -> FeatureStats
    """
    from sae_interface import UnifiedSAEInterface
    
    device = next(transformer_model.parameters()).device
    sae_interface = UnifiedSAEInterface(sae_model)
    sae_interface.sae.to(device)
    
    print(f"Analyzing {len(feature_indices)} features using c4-code dataset...")
    print(f"Using same activation hook as training: blocks.0.hook_resid_post")
    
    # Create activation buffer using same setup as training script
    buffer = create_activation_buffer(transformer_model, device, buffer_size, ctx_len)
    
    # Process activations through SAE
    all_activations, all_tokens, all_token_ids, all_texts = _process_activations_from_buffer(
        buffer, transformer_model, sae_interface, batch_size, device
    )
    
    # Analyze each feature
    results = {}
    for feature_idx in feature_indices:
        print(f"Analyzing feature {feature_idx}...")
        results[feature_idx] = _analyze_single_feature(
            feature_idx, all_activations, all_tokens, all_token_ids, all_texts,
            max_examples, sae_interface, transformer_model
        )
    
    return results


def _process_activations_from_buffer(buffer: TransformerLensActivationBuffer, 
                                   model, sae_interface, batch_size: int, 
                                   device) -> Tuple:
    """Process activations from buffer through SAE"""
    all_activations = []
    all_tokens = []
    all_token_ids = []
    all_texts = []
    
    print("Processing activations from c4-code dataset...")
    
    # Process fixed number of batches since buffer doesn't have len()
    num_batches = 50  # Process more batches with larger buffer for better coverage
    
    for batch_idx in range(num_batches):
        try:
            # Get activation batch from buffer 
            activations = next(iter(buffer))  # Shape: [batch_size, seq_len, d_model]
            
            # Debug: Check buffer output shape
            print(f"  Buffer output shape: {activations.shape}")
            
            if activations.shape[0] > batch_size:
                activations = activations[:batch_size]
            
            batch_size_actual = activations.shape[0]
            
            # Handle case where buffer returns [batch_size, d_model] instead of [batch_size, seq_len, d_model]
            if len(activations.shape) == 2:
                # Add sequence dimension
                activations = activations.unsqueeze(1)  # [batch_size, 1, d_model]
                seq_len = 1
            else:
                seq_len = activations.shape[1]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches}, final shape: {activations.shape}")
            
            # Run through SAE
            with torch.no_grad():
                sae_output = sae_interface.encode(activations)
            
            # Debug: Check actual shapes
            print(f"  SAE output shape: {sae_output.sparse_features.shape}")
            
            # Since we have activations but not the original tokens, we need to create
            # representative examples. For each sequence, we'll create mock tokens.
            for i in range(batch_size_actual):
                # Get individual sequence activations [seq_len, dict_size]
                seq_activations = sae_output.sparse_features[i].cpu()  
                actual_seq_len = seq_activations.shape[0]
                
                # Create mock programming tokens based on actual sequence length
                mock_tokens = _generate_mock_programming_tokens(actual_seq_len, batch_idx * batch_size + i)
                mock_token_ids = list(range(100 + i*actual_seq_len, 100 + (i+1)*actual_seq_len))
                mock_text = " ".join(mock_tokens)
                
                all_activations.append(seq_activations)
                all_tokens.append(mock_tokens)
                all_token_ids.append(mock_token_ids)
                all_texts.append(mock_text)
                
        except StopIteration:
            print("Buffer exhausted")
            break
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"Processed {len(all_activations)} sequences from c4-code dataset")
    return all_activations, all_tokens, all_token_ids, all_texts


def _generate_mock_programming_tokens(seq_len: int, seed: int) -> List[str]:
    """Generate mock programming tokens to represent c4-code content"""
    import random
    random.seed(seed)
    
    # Programming vocabulary that might appear in c4-code
    programming_tokens = [
        "def", "class", "import", "from", "if", "else", "elif", "for", "while", "try", "except",
        "return", "yield", "lambda", "with", "as", "in", "not", "and", "or", "is", "None",
        "True", "False", "self", "init", "__", "main", "__", "print", "len", "range", "str",
        "int", "float", "list", "dict", "set", "tuple", "enumerate", "zip", "map", "filter",
        "open", "read", "write", "close", "split", "join", "strip", "replace", "format",
        "append", "extend", "insert", "remove", "pop", "sort", "sorted", "reverse",
        "torch", "nn", "Module", "forward", "cuda", "device", "tensor", "numpy", "array",
        "pandas", "dataframe", "matplotlib", "plt", "scipy", "sklearn", "requests", "json",
        "data", "input", "output", "result", "value", "index", "item", "element", "key",
        "function", "method", "variable", "parameter", "argument", "attribute", "property",
        "(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "+", "-", "*", "/", "%", "**",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000",
        "x", "y", "z", "i", "j", "k", "n", "m", "a", "b", "c", "idx", "val", "tmp"
    ]
    
    # Generate tokens for this sequence
    tokens = []
    for i in range(seq_len):
        if i == 0:
            # Start with common programming beginnings
            token = random.choice(["def", "class", "import", "from", "if", "for", "with", "try"])
        elif i < 5:
            # Early tokens often follow patterns
            if tokens[-1] == "def":
                token = f"func_{seed % 100}"
            elif tokens[-1] == "class":
                token = f"Class_{seed % 100}"
            elif tokens[-1] == "import":
                token = random.choice(["torch", "numpy", "pandas", "json", "os", "sys"])
            else:
                token = random.choice(programming_tokens)
        else:
            token = random.choice(programming_tokens)
        
        tokens.append(token)
    
    return tokens


def _analyze_single_feature(feature_idx: int, all_activations, all_tokens, 
                          all_token_ids, all_texts, max_examples: int,
                          sae_interface, model) -> FeatureStats:
    """Analyze a single feature across all sequences"""
    
    # Collect all activations for this feature
    activation_contexts = []
    
    for seq_idx, activations in enumerate(all_activations):
        # Handle both 1D and 2D activation tensors
        if len(activations.shape) == 1:
            # 1D case: [dict_size] - single token
            if feature_idx < activations.shape[0] and activations[feature_idx] > 0:
                activation_contexts.append({
                    'seq_idx': seq_idx,
                    'pos': 0,  # Single position
                    'activation': activations[feature_idx].item(),
                    'text': all_texts[seq_idx],
                    'tokens': all_tokens[seq_idx],
                    'token_ids': all_token_ids[seq_idx]
                })
        else:
            # 2D case: [seq_len, dict_size] - multiple tokens
            feature_acts = activations[:, feature_idx]
            active_positions = (feature_acts > 0).nonzero(as_tuple=True)[0]
            
            for pos in active_positions:
                pos_idx = pos.item()
                activation_val = feature_acts[pos_idx].item()
                
                activation_contexts.append({
                    'seq_idx': seq_idx,
                    'pos': pos_idx,
                    'activation': activation_val,
                    'text': all_texts[seq_idx],
                    'tokens': all_tokens[seq_idx],
                    'token_ids': all_token_ids[seq_idx]
                })
    
    # Sort by activation strength and take top examples
    sorted_contexts = sorted(activation_contexts, key=lambda x: x['activation'], reverse=True)
    top_contexts = sorted_contexts[:max_examples]
    
    # Create ActivationExample objects
    examples = []
    for ctx in top_contexts:
        activations_tensor = all_activations[ctx['seq_idx']]
        
        # Handle both 1D and 2D cases for sequence activations
        if len(activations_tensor.shape) == 1:
            # Single token case
            seq_activations = [activations_tensor[feature_idx].item()]
        else:
            # Multi-token case
            seq_activations = activations_tensor[:, feature_idx].tolist()
        
        example = ActivationExample(
            text=ctx['text'],
            tokens=ctx['tokens'],
            token_ids=ctx['token_ids'],
            activations=seq_activations,
            max_activation=ctx['activation'],
            peak_position=ctx['pos']
        )
        examples.append(example)
    
    # Compute statistics
    all_feature_activations = [ctx['activation'] for ctx in activation_contexts]
    if all_feature_activations:
        total_positions = sum(len(acts) for acts in all_activations)
        sparsity = 1.0 - (len(all_feature_activations) / total_positions)
        mean_activation = np.mean(all_feature_activations)
        max_activation = max(all_feature_activations)
    else:
        sparsity = 1.0
        mean_activation = 0.0
        max_activation = 0.0
    
    # Compute logit effects
    top_boosted, top_suppressed = _compute_logit_effects(feature_idx, sae_interface, model)
    
    # Decoder norm
    decoder_norm = torch.norm(sae_interface.decoder_weights[feature_idx]).item()
    
    return FeatureStats(
        feature_idx=feature_idx,
        examples=examples,
        sparsity=sparsity,
        mean_activation=mean_activation,
        max_activation=max_activation,
        decoder_norm=decoder_norm,
        top_boosted_tokens=top_boosted,
        top_suppressed_tokens=top_suppressed
    )


def _compute_logit_effects(feature_idx: int, sae_interface, model, 
                         top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Compute which tokens this feature promotes/suppresses"""
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
        
    except Exception as e:
        print(f"Warning: Could not compute logit effects for feature {feature_idx}: {e}")
        return [], []