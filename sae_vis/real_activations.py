"""
Real activations: analyze what actually fires SAE features in real text.
Uses TinyStories dataset with proper tokenization to avoid vocabulary mismatches.
Enhanced with coherent context extraction.
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
                           max_examples: int = 11, batch_size: int = 8, 
                           buffer_size: int = 2000, ctx_len: int = 128,
                           context_window: int = 25) -> Dict[int, FeatureStats]:
    """
    Analyze real activations using TinyStories dataset with proper tokenization.
    
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
    
    print(f"Analyzing {len(feature_indices)} features using TinyStories dataset...")
    print(f"Using activation hook: blocks.3.hook_resid_post")
    print(f"Context window: {context_window} tokens around firing positions")
    
    # Process sequences and collect real activations
    all_sequences = _process_real_sequences(
        transformer_model, sae_interface, batch_size, device, buffer_size, ctx_len
    )
    
    if not all_sequences:
        print("Error: No sequences were processed successfully!")
        return {}
    
    print(f"Successfully processed {len(all_sequences)} sequences")
    
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
    """Process real sequences from TinyStories dataset through transformer and SAE"""
    
    print("Loading TinyStories dataset...")
    try:
        data_gen = hf_dataset_to_generator(
            "roneneldan/TinyStories",
            split="train",
            return_tokens=False  # We want raw text, not tokens
        )
    except Exception as e:
        print(f"Failed to load TinyStories dataset: {e}")
        return []

    all_sequences = []
    sequences_processed = 0
    batch_texts = []
    
    try:
        for data_item in data_gen:
            if sequences_processed >= n_sequences:
                break
                
            # Extract TEXT from dataset - handle both dict and string formats
            if isinstance(data_item, dict):
                text = data_item.get('text', '')
            elif isinstance(data_item, str):
                text = data_item  # TinyStories returns raw strings
            else:
                continue
            
            # Skip empty or very short texts
            if not text or len(text.strip()) < 50:
                continue
                
            batch_texts.append(text.strip())
            
            # Process batch when full
            if len(batch_texts) >= batch_size:
                _process_text_batch(
                    batch_texts, model, sae_interface, 
                    device, all_sequences, ctx_len
                )
                sequences_processed += len(batch_texts)
                if sequences_processed % 100 == 0:
                    print(f"Processed {sequences_processed}/{n_sequences} sequences...")
                
                batch_texts = []
        
        # Process remaining batch
        if batch_texts:
            _process_text_batch(
                batch_texts, model, sae_interface, 
                device, all_sequences, ctx_len
            )
            sequences_processed += len(batch_texts)
    
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Processed {len(all_sequences)} sequences from TinyStories dataset")
    return all_sequences


def _process_text_batch(batch_texts: List[str], model, sae_interface, 
                       device, all_sequences: List[Dict], ctx_len: int):
    """Process a batch of raw texts by tokenizing with the model's tokenizer"""
    
    # Tokenize texts with the model's own tokenizer
    tokenized_batch = []
    valid_texts = []
    
    for text in batch_texts:
        try:
            # Tokenize with model's tokenizer (this creates the token IDs)
            tokens = model.tokenizer.encode(text, add_special_tokens=False)
            
            # Skip very short sequences
            if len(tokens) < 10:
                continue
                
            # Truncate to context length
            if len(tokens) > ctx_len:
                tokens = tokens[:ctx_len]
                # Also truncate text to roughly match
                words = text.split()
                text = ' '.join(words[:min(len(words), ctx_len)])
                
            tokenized_batch.append(tokens)
            valid_texts.append(text)
            
        except Exception as e:
            continue  # Skip problematic texts
    
    if not tokenized_batch:
        return
        
    # Pad sequences to same length for batch processing
    max_len = max(len(tokens) for tokens in tokenized_batch)
    max_len = min(max_len, ctx_len)  # Cap at context length
    
    padded_batch = []
    for tokens in tokenized_batch:
        if len(tokens) < max_len:
            pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
            tokens = tokens + [pad_id] * (max_len - len(tokens))
        elif len(tokens) > max_len:
            tokens = tokens[:max_len]
        padded_batch.append(tokens)
    
    try:
        # Convert to tensor
        input_ids = torch.tensor(padded_batch, device=device)
        
        # Get transformer activations at hook point
        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, stop_at_layer=4)  # Stop after block 3
            
            # Get residual stream activations after block 3
            hook_name = "blocks.3.hook_resid_post"
            if hook_name in cache:
                activations = cache[hook_name]
            else:
                # Fallback to other possible names
                possible_hooks = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post", "blocks.2.hook_resid_post"]
                activations = None
                for hook in possible_hooks:
                    if hook in cache:
                        activations = cache[hook]
                        print(f"Using fallback hook: {hook}")
                        break
                
                if activations is None:
                    return
            
            # Pass through SAE
            sae_output = sae_interface.encode(activations)
            
    except Exception as e:
        return
    
    # Process each sequence in the batch
    for seq_idx in range(len(valid_texts)):
        if seq_idx >= len(tokenized_batch):
            break
            
        text = valid_texts[seq_idx]
        original_tokens = tokenized_batch[seq_idx]  # Original length before padding
        
        # Get token strings by decoding with the SAME tokenizer that created the IDs
        token_strings = []
        for token_id in original_tokens:
            try:
                token_str = model.tokenizer.decode([token_id])
                token_strings.append(token_str if token_str.strip() else "<unk>")
            except:
                token_strings.append("<unk>")
        
        # Get SAE activations for this sequence (remove padding dimension)
        seq_len = len(original_tokens)
        if seq_idx < sae_output.sparse_features.shape[0] and seq_len <= sae_output.sparse_features.shape[1]:
            sequence_activations = sae_output.sparse_features[seq_idx, :seq_len].cpu().numpy()
            
            # Store sequence data
            all_sequences.append({
                'text': text,
                'token_ids': original_tokens,
                'token_strings': token_strings,  # Now decoded with correct tokenizer!
                'sae_activations': sequence_activations,
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
                    token_strings=seq_data['token_strings'],
                    peak_token_pos=pos,
                    context_window=context_window
                )
                
                # Get corresponding token IDs for the context
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
                    'context_tokens': context_tokens,
                    'context_token_ids': context_token_ids,
                    'context_activations': context_activations,
                    'full_context': context_text,
                    'full_text': seq_data['text'][:1000],
                })
            
            total_positions += seq_len
        else:
            # Handle unexpected shapes
            continue
    
    # Sort by activation strength and take top examples
    activation_contexts.sort(key=lambda x: x['activation'], reverse=True)
    top_contexts = activation_contexts[:max_examples]
    
    # Create ActivationExample objects with enhanced context
    examples = []
    for ctx in top_contexts:
        example = ActivationExample(
            text=ctx['full_context'],  # Use coherent context
            tokens=ctx['context_tokens'],  # Coherent tokens
            token_ids=ctx['context_token_ids'],
            activations=ctx['context_activations'].tolist(),
            max_activation=ctx['activation'],
            peak_position=ctx['firing_pos_in_context'],
            full_context=ctx['full_context']  # Clean, readable text
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


def _get_coherent_context(original_text: str, token_strings: List[str], 
                         peak_token_pos: int, context_window: int) -> Tuple[str, int, List[str]]:
    """Extract coherent context around peak position"""
    
    # Define context window in tokens
    half_window = context_window // 2
    context_start = max(0, peak_token_pos - half_window)
    context_end = min(len(token_strings), peak_token_pos + half_window + 1)
    
    # Get context tokens
    context_tokens = token_strings[context_start:context_end]
    
    # Create coherent text by joining tokens
    context_text = ''.join(context_tokens)
    
    # Clean up common tokenizer artifacts
    context_text = context_text.replace('Ġ', ' ')  # GPT-2 style
    context_text = context_text.replace('▁', ' ')  # SentencePiece style
    context_text = context_text.strip()
    
    # Calculate peak position within context
    peak_in_context = peak_token_pos - context_start
    
    return context_text, peak_in_context, context_tokens


def _compute_logit_effects(feature_idx: int, sae_interface, model, 
                         top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Compute which tokens this feature promotes/suppresses in the output"""
    try:
        # Get the decoder direction for this feature
        decoder_direction = sae_interface.decoder_weights[feature_idx]
        
        # Ensure decoder direction matches d_model
        if decoder_direction.shape[0] != model.cfg.d_model:
            if decoder_direction.shape[0] > model.cfg.d_model:
                residual_direction = decoder_direction[:model.cfg.d_model]
            else:
                # Pad with zeros if needed
                pad_size = model.cfg.d_model - decoder_direction.shape[0]
                residual_direction = torch.cat([decoder_direction, torch.zeros(pad_size, device=decoder_direction.device)])
        else:
            residual_direction = decoder_direction
        
        # Project to vocabulary logits using unembedding matrix
        if hasattr(model, 'W_U'):
            unembedding = model.W_U
            logit_effects = residual_direction @ unembedding
        else:
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
        return [], []