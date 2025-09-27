"""
Real activations: analyze what actually fires SAE features in real text.
Uses the same c4-code dataset and activation pipeline as the training script.
Enhanced with 25-token context windows by default.
"""
import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import data utilities from parent directory (same as training script)
try:
    from dictionary_learning.utils import hf_dataset_to_generator
except ImportError:
    # Fallback if structure is different
    from utils import hf_dataset_to_generator


@dataclass
class ActivationExample:
    """Single example of feature activation with enhanced context"""
    text: str
    tokens: List[str]
    token_ids: List[int]
    activations: List[float]  # Per-token activations in context window
    max_activation: float
    peak_position: int        # Position of peak within context window
    full_context: str        # Human-readable context text


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


def analyze_real_activations(sae_model, transformer_model, feature_indices: List[int], 
                           max_examples: int = 20, batch_size: int = 8, 
                           buffer_size: int = 20000, ctx_len: int = 128,
                           context_window: int = 25) -> Dict[int, FeatureStats]:
    """
    Analyze real activations using the same c4-code dataset as training.
    
    Args:
        sae_model: Trained SAE model
        transformer_model: Base transformer  
        feature_indices: Which features to analyze
        max_examples: Top examples per feature
        batch_size: Processing batch size
        buffer_size: Number of sequences to process
        ctx_len: Context length for each sample
        context_window: Size of context around firing tokens (default 25)
        
    Returns:
        Dict mapping feature_idx -> FeatureStats
    """
    from sae_interface import UnifiedSAEInterface
    
    device = next(transformer_model.parameters()).device
    sae_interface = UnifiedSAEInterface(sae_model)
    sae_interface.sae.to(device)
    
    print(f"Analyzing {len(feature_indices)} features using c4-code dataset...")
    print(f"Using activation hook: blocks.0.hook_resid_post")
    print(f"Context window: {context_window} tokens around firing positions")
    
    # Process sequences and collect real activations
    all_sequences = _process_real_sequences(
        transformer_model, sae_interface, batch_size, device, buffer_size, ctx_len
    )
    
    # Analyze each feature
    results = {}
    for feature_idx in feature_indices:
        print(f"Analyzing feature {feature_idx} with {context_window}-token context...")
        results[feature_idx] = _analyze_single_feature(
            feature_idx, all_sequences, max_examples, sae_interface, 
            transformer_model, context_window
        )
    
    return results


def _process_real_sequences(model, sae_interface, batch_size: int, device, 
                           n_sequences: int, ctx_len: int) -> List[Dict]:
    """Process real sequences from c4-code dataset through transformer and SAE"""
    
    # Set up data generator for c4-code dataset
    print("Loading c4-code dataset...")
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b",
        split="train", 
        return_tokens=True
    )
    
    all_sequences = []
    sequences_processed = 0
    
    # Process sequences in batches
    batch_tokens = []
    batch_texts = []
    
    for data_item in data_gen:
        if sequences_processed >= n_sequences:
            break
            
        # Extract tokens from dataset
        if isinstance(data_item, dict):
            tokens = data_item.get('tokens', data_item.get('input_ids', []))
            text = data_item.get('text', '')
        else:
            continue
            
        # Truncate to context length
        if len(tokens) > ctx_len:
            tokens = tokens[:ctx_len]
        elif len(tokens) < ctx_len:
            # Pad if needed
            tokens = tokens + [model.tokenizer.pad_token_id] * (ctx_len - len(tokens))
        
        batch_tokens.append(tokens)
        batch_texts.append(text if text else model.tokenizer.decode(tokens))
        
        # Process batch when full
        if len(batch_tokens) >= batch_size:
            _process_batch(
                batch_tokens, batch_texts, model, sae_interface, 
                device, all_sequences
            )
            sequences_processed += len(batch_tokens)
            print(f"Processed {sequences_processed}/{n_sequences} sequences...")
            
            batch_tokens = []
            batch_texts = []
    
    # Process remaining batch
    if batch_tokens:
        _process_batch(
            batch_tokens, batch_texts, model, sae_interface, 
            device, all_sequences
        )
        sequences_processed += len(batch_tokens)
    
    print(f"Processed {len(all_sequences)} sequences from c4-code dataset")
    return all_sequences


def _process_batch(batch_token_ids: List[List[int]], batch_texts: List[str], 
                  model, sae_interface, device, all_sequences: List[Dict]):
    """Process a batch of sequences through transformer and SAE"""
    
    # Convert to tensor
    input_ids = torch.tensor(batch_token_ids, device=device)
    
    # Get transformer activations at hook point
    with torch.no_grad():
        # Run model and get cache
        _, cache = model.run_with_cache(input_ids, stop_at_layer=1)
        
        # Get residual stream activations after block 0
        hook_name = "blocks.0.hook_resid_post"
        if hook_name in cache:
            activations = cache[hook_name]
        else:
            # Fallback to other possible names
            if "blocks.0.hook_resid_pre" in cache:
                activations = cache["blocks.0.hook_resid_pre"]
            elif "resid_post.0" in cache:
                activations = cache["resid_post.0"]
            else:
                print(f"Warning: Could not find activation hook. Available keys: {list(cache.keys())[:5]}")
                return
        
        # Pass through SAE
        sae_output = sae_interface.encode(activations)
    
    # Process each sequence in the batch
    for seq_idx in range(len(batch_token_ids)):
        token_ids = batch_token_ids[seq_idx]
        text = batch_texts[seq_idx]
        
        # Get token strings
        token_strings = []
        for token_id in token_ids:
            if token_id == model.tokenizer.pad_token_id:
                token_strings.append("<pad>")
            else:
                token_strings.append(model.tokenizer.decode([token_id]))
        
        # Get SAE activations for this sequence
        sequence_activations = sae_output.sparse_features[seq_idx].cpu().numpy()
        
        # Store sequence data
        all_sequences.append({
            'text': text,
            'token_ids': token_ids,
            'token_strings': token_strings,
            'sae_activations': sequence_activations,  # Shape: [seq_len, dict_size]
        })


def _analyze_single_feature(feature_idx: int, all_sequences: List[Dict], 
                          max_examples: int, sae_interface, model,
                          context_window: int = 25) -> FeatureStats:
    """Analyze a single feature across all sequences with enhanced context"""
    
    # Collect all activations for this feature
    activation_contexts = []
    total_positions = 0
    
    for seq_data in all_sequences:
        sae_activations = seq_data['sae_activations']
        
        # Handle both 2D and 3D activation arrays
        if len(sae_activations.shape) == 2:
            # Shape: [seq_len, dict_size]
            seq_len, dict_size = sae_activations.shape
            
            # Check if feature index is valid
            if feature_idx >= dict_size:
                continue
                
            # Get activations for this feature across the sequence
            feature_acts = sae_activations[:, feature_idx]
            
            # Find positions where this feature fires (any positive activation)
            active_positions = np.where(feature_acts > 0)[0]
            
            for pos in active_positions:
                activation_val = feature_acts[pos]
                
                # ENHANCED: Get coherent context from original text
                context_text, peak_in_context, context_tokens = _get_coherent_context(
                    original_text=seq_data['text'],
                    token_ids=seq_data['token_ids'],
                    peak_token_pos=pos,
                    context_window=context_window,
                    tokenizer=model.tokenizer
                )
                
                # Get corresponding token IDs for the context (approximate)
                half_window = context_window // 2
                context_start = max(0, pos - half_window)
                context_end = min(len(seq_data['token_ids']), pos + half_window + 1)
                context_token_ids = seq_data['token_ids'][context_start:context_end]
                
                # Get activations for the context window
                context_activations = feature_acts[context_start:context_end]
                
                activation_contexts.append({
                    'activation': activation_val,
                    'position': pos,
                    'firing_pos_in_context': peak_in_context,
                    'context_tokens': context_tokens,  # Now coherent!
                    'context_token_ids': context_token_ids,
                    'context_activations': context_activations,
                    'full_context': context_text,  # Clean, readable text
                    'full_text': seq_data['text'][:1000],
                })
            
            total_positions += seq_len
        else:
            # Handle unexpected shapes
            print(f"Warning: Unexpected activation shape {sae_activations.shape}")
            continue
    
    # Sort by activation strength and take top examples
    activation_contexts.sort(key=lambda x: x['activation'], reverse=True)
    top_contexts = activation_contexts[:max_examples]
    
    # Create ActivationExample objects with enhanced context
    examples = []
    for ctx in top_contexts:
        example = ActivationExample(
            text=ctx['full_context'],  # Use coherent context instead of fragmented text
            tokens=ctx['context_tokens'],  # Now contains coherent tokens
            token_ids=ctx['context_token_ids'],
            activations=ctx['context_activations'].tolist(),
            max_activation=ctx['activation'],
            peak_position=ctx['firing_pos_in_context'],
            full_context=ctx['full_context']  # This is now clean, readable text
        )
        examples.append(example)
    
    # Compute statistics
    if activation_contexts:
        all_activations = [ctx['activation'] for ctx in activation_contexts]
        sparsity = 1.0 - (len(all_activations) / max(total_positions, 1))
        mean_activation = np.mean(all_activations)
        max_activation = max(all_activations)
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
    """Compute which tokens this feature promotes/suppresses in the output"""
    try:
        # Get the decoder direction for this feature
        decoder_direction = sae_interface.decoder_weights[feature_idx]
        
        # Project to residual stream dimension if needed
        if decoder_direction.shape[0] != model.cfg.d_model:
            # If decoder outputs to MLP dimension, project back to residual
            if hasattr(model, 'W_out') and model.W_out:
                # W_out projects from MLP to residual stream
                if len(model.W_out) > 0:
                    W_out = model.W_out[0]  # Layer 0 output projection
                    # Check dimensions match
                    if W_out.shape[0] == decoder_direction.shape[0]:
                        residual_direction = decoder_direction @ W_out
                    else:
                        residual_direction = decoder_direction[:model.cfg.d_model]
                else:
                    residual_direction = decoder_direction[:model.cfg.d_model]
            else:
                # Truncate or pad to match d_model
                if decoder_direction.shape[0] > model.cfg.d_model:
                    residual_direction = decoder_direction[:model.cfg.d_model]
                else:
                    residual_direction = decoder_direction
        else:
            residual_direction = decoder_direction
        
        # Project to vocabulary logits using unembedding matrix
        if hasattr(model, 'W_U'):
            unembedding = model.W_U
            
            # Ensure dimensions match for matrix multiplication
            if residual_direction.shape[0] != unembedding.shape[0]:
                print(f"Warning: Dimension mismatch - residual: {residual_direction.shape}, unembed: {unembedding.shape}")
                return [], []
            
            logit_effects = residual_direction @ unembedding
        else:
            print("Warning: Model has no unembedding matrix W_U")
            return [], []
        
        # Get top boosted tokens (highest logit effects)
        top_values, top_indices = torch.topk(logit_effects, min(top_k, logit_effects.shape[0]))
        top_boosted = []
        for val, idx in zip(top_values, top_indices):
            token = model.tokenizer.decode([idx.item()])
            effect = val.item()
            top_boosted.append((token, effect))
        
        # Get top suppressed tokens (lowest logit effects)
        bottom_values, bottom_indices = torch.topk(logit_effects, min(top_k, logit_effects.shape[0]), largest=False)
        top_suppressed = []
        for val, idx in zip(bottom_values, bottom_indices):
            token = model.tokenizer.decode([idx.item()])
            effect = val.item()
            top_suppressed.append((token, effect))
        
        return top_boosted, top_suppressed
        
    except Exception as e:
        print(f"Warning: Could not compute logit effects for feature {feature_idx}: {e}")
        import traceback
        traceback.print_exc()
        return [], []
    

def _get_coherent_context(original_text: str, token_ids: List[int], 
                         peak_token_pos: int, context_window: int,
                         tokenizer) -> Tuple[str, int, List[str]]:
    """Extract coherent context from original text around peak position"""
    
    try:
        # Tokenize the original text to get token-to-char mapping
        # Use add_special_tokens=False to match the stored token_ids
        encoding = tokenizer(original_text, return_offsets_mapping=True, 
                           add_special_tokens=False, truncation=False)
        
        if not hasattr(encoding, 'offset_mapping') or not encoding.offset_mapping:
            # Fallback: use simple word splitting
            return _fallback_coherent_context(original_text, peak_token_pos, context_window)
        
        # Ensure we don't go beyond available tokens
        if peak_token_pos >= len(encoding.offset_mapping):
            return _fallback_coherent_context(original_text, peak_token_pos, context_window)
        
        # Define context window in tokens
        half_window = context_window // 2
        context_start_token = max(0, peak_token_pos - half_window)
        context_end_token = min(len(encoding.offset_mapping), peak_token_pos + half_window + 1)
        
        # Get character bounds for context
        context_char_start = encoding.offset_mapping[context_start_token][0]
        context_char_end = encoding.offset_mapping[context_end_token - 1][1]
        
        # Extract coherent text
        context_text = original_text[context_char_start:context_char_end]
        
        # Get individual token texts within context for highlighting
        context_tokens = []
        for i in range(context_start_token, context_end_token):
            token_start, token_end = encoding.offset_mapping[i]
            token_text = original_text[token_start:token_end]
            context_tokens.append(token_text)
        
        # Calculate peak position within context
        peak_in_context = peak_token_pos - context_start_token
        
        return context_text, peak_in_context, context_tokens
        
    except Exception as e:
        print(f"Warning: Could not extract coherent context: {e}")
        return _fallback_coherent_context(original_text, peak_token_pos, context_window)
    
def _fallback_coherent_context(text: str, peak_pos: int, context_window: int) -> Tuple[str, int, List[str]]:
    """Fallback: split by whitespace and extract context"""
    words = text.split()
    half_window = context_window // 2
    
    # Map token position to approximate word position
    word_pos = min(peak_pos, len(words) - 1)
    
    context_start = max(0, word_pos - half_window)
    context_end = min(len(words), word_pos + half_window + 1)
    
    context_words = words[context_start:context_end]
    context_text = ' '.join(context_words)
    peak_in_context = word_pos - context_start
    
    return context_text, peak_in_context, context_words