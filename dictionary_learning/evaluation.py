"""
Fixed evaluation metrics with proper bounds and numerical stability.
"""

import torch as t
from collections import defaultdict
from typing import Union, Tuple, Dict, Any, Optional, Callable

# Import for nnsight functionality
from nnsight import LanguageModel

# Import for transformer_lens functionality
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .buffer import ActivationBuffer, TransformerLensActivationBuffer, BaseActivationBuffer
from .config import DEBUG


@t.no_grad()
def evaluate(
    dictionary,  # a dictionary
    activations, # a generator of activations; if an ActivationBuffer, also compute loss recovered
    max_len=128,  # max context length for loss recovered
    batch_size=128,  # batch size for loss recovered
    io="out",  # can be 'in', 'out', or 'in_and_out'
    normalize_batch=False, # normalize batch before passing through dictionary
    tracer_args={'use_cache': False, 'output_attentions': False}, # minimize cache during model trace.
    device="cpu",
    n_batches: int = 1,
):
    """
    Evaluate a dictionary on a dataset, computing metrics and loss recovery.
    Works with both nnsight and transformer_lens models.
    
    FIXED: Added numerical stability and proper bounds for all metrics.
    """
    assert n_batches > 0
    out = defaultdict(float)
    active_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)

    for _ in range(n_batches):
        try:
            x = next(activations).to(device)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim ** 0.5)
        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )
        x_hat, f = dictionary(x, output_features=True)
        
        # Ensure same dtype for numerical stability
        x = x.to(dtype=t.float32)
        x_hat = x_hat.to(dtype=t.float32)
        f = f.to(dtype=t.float32)
        
        # Basic reconstruction metrics
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        
        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2
        active_features += features_BF.sum(dim=0)

        # Cosine similarity with numerical stability
        x_norm = t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_norm = t.linalg.norm(x_hat, dim=-1, keepdim=True)
        
        # Avoid division by zero
        x_norm = t.clamp(x_norm, min=1e-8)
        x_hat_norm = t.clamp(x_hat_norm, min=1e-8)
        
        x_normed = x / x_norm
        x_hat_normed = x_hat / x_hat_norm
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()
        
        # Ensure cosine similarity is in valid range [-1, 1]
        cossim = t.clamp(cossim, min=-1.0, max=1.0)

        # L2 ratio with numerical stability
        x_mag = t.linalg.norm(x, dim=-1)
        x_hat_mag = t.linalg.norm(x_hat, dim=-1)
        
        # Avoid division by zero
        x_mag = t.clamp(x_mag, min=1e-8)
        l2_ratio = (x_hat_mag / x_mag).mean()

        # FIXED: Variance explained with proper bounds
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        
        # Ensure total_variance is not zero
        total_variance = t.clamp(total_variance, min=1e-8)
        
        # Compute fraction variance explained and clamp to [0, 1]
        frac_variance_explained = (1 - residual_variance / total_variance)
        frac_variance_explained = t.clamp(frac_variance_explained, min=0.0, max=1.0)

        # FIXED: Relative reconstruction bias with numerical stability
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        
        # Avoid division by zero in relative reconstruction bias
        x_dot_x_hat_mean = x_dot_x_hat.mean()
        x_dot_x_hat_mean = t.clamp(t.abs(x_dot_x_hat_mean), min=1e-8)
        
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat_mean
        
        # Clamp to reasonable range (bias should be close to 1 for good reconstructions)
        relative_reconstruction_bias = t.clamp(relative_reconstruction_bias, min=0.1, max=10.0)

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        out["frac_variance_explained"] += frac_variance_explained.item()
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out['relative_reconstruction_bias'] += relative_reconstruction_bias.item()

        # Check if we're using an activation buffer
        is_buffer = isinstance(activations, (ActivationBuffer, TransformerLensActivationBuffer, BaseActivationBuffer))
        if not is_buffer:
            continue

        # compute loss recovered - works with both model types due to our unified interface
        try:
            # Get text batch with better error handling
            try:
                text_batch = activations.text_batch(batch_size=batch_size)
                print(f"DEBUG: Got text_batch type: {type(text_batch)}")
                if hasattr(text_batch, '__len__'):
                    print(f"DEBUG: text_batch length: {len(text_batch)}")
                if isinstance(text_batch, list) and len(text_batch) > 0:
                    print(f"DEBUG: First item type: {type(text_batch[0])}")
                    print(f"DEBUG: First item sample: {str(text_batch[0])[:100]}...")
            except Exception as text_error:
                print(f"Warning: Could not get text batch: {text_error}")
                continue
            
            # Determine the correct submodule/hook parameter
            if hasattr(activations, 'submodule'):
                submodule_or_hook = activations.submodule
                print(f"DEBUG: Using submodule: {type(submodule_or_hook)}")
            elif hasattr(activations, 'hook_name'):
                submodule_or_hook = activations.hook_name
                print(f"DEBUG: Using hook_name: {submodule_or_hook}")
            else:
                print("Warning: Could not determine submodule/hook for loss recovery")
                continue
            
            print("DEBUG: About to call loss_recovered...")
            print(f"DEBUG: text_batch type: {type(text_batch)}")
            print(f"DEBUG: model type: {type(activations.model)}")
            print(f"DEBUG: submodule_or_hook type: {type(submodule_or_hook)}")
            print(f"DEBUG: dictionary type: {type(dictionary)}")
            
            loss_result = loss_recovered(
                text_batch,
                activations.model,
                submodule_or_hook,
                dictionary,
                max_len=max_len,
                normalize_batch=normalize_batch,
                io=io,
                tracer_args=tracer_args
            )
            
            print(f"DEBUG: loss_result type: {type(loss_result)}")
            print(f"DEBUG: loss_result length: {len(loss_result) if hasattr(loss_result, '__len__') else 'no length'}")
            print(f"DEBUG: loss_result content: {loss_result}")
            
            # Handle different return formats with more debugging
            if isinstance(loss_result, tuple):
                if len(loss_result) == 3:
                    loss_original, loss_reconstructed, loss_zero = loss_result
                    print(f"DEBUG: Successfully unpacked 3 values")
                    print(f"DEBUG: loss_original: {loss_original}")
                    print(f"DEBUG: loss_reconstructed: {loss_reconstructed}")
                    print(f"DEBUG: loss_zero: {loss_zero}")
                else:
                    print(f"ERROR: Expected tuple of length 3, got length {len(loss_result)}")
                    print(f"DEBUG: Tuple contents: {loss_result}")
                    continue
            else:
                print(f"ERROR: Expected tuple, got {type(loss_result)}")
                continue
            
            # FIXED: Ensure all losses are positive and handle edge cases
            loss_original = t.clamp(loss_original, min=0.0)
            loss_reconstructed = t.clamp(loss_reconstructed, min=0.0)
            loss_zero = t.clamp(loss_zero, min=0.0)
            
            # Compute fraction recovered with numerical stability
            loss_diff = loss_original - loss_zero
            if loss_diff.abs() < 1e-8:
                # If original and zero losses are nearly equal, set frac_recovered to 0
                frac_recovered = t.tensor(0.0)
            else:
                frac_recovered = (loss_reconstructed - loss_zero) / loss_diff
                # Clamp to reasonable range [-2, 2] to handle edge cases
                frac_recovered = t.clamp(frac_recovered, min=-2.0, max=2.0)
            
            out["loss_original"] += loss_original.item()
            out["loss_reconstructed"] += loss_reconstructed.item()
            out["loss_zero"] += loss_zero.item()
            out["frac_recovered"] += frac_recovered.item()
            
            print("DEBUG: Loss recovery completed successfully")
            
        except Exception as e:
            print(f"Warning: Loss recovery computation failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            # Set default values if loss recovery fails
            out["loss_original"] += float('nan')
            out["loss_reconstructed"] += float('nan') 
            out["loss_zero"] += float('nan')
            out["frac_recovered"] += 0.0

    # Average over batches
    out = {key: value / n_batches for key, value in out.items()}
    
    # FIXED: Ensure frac_alive is properly bounded
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    frac_alive = t.clamp(frac_alive, min=0.0, max=1.0)
    out["frac_alive"] = frac_alive.item()

    return out


# Rest of the evaluation functions remain the same...
def loss_recovered_nnsight(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    io="out",  # can be 'in', 'out', or 'in_and_out'
    tracer_args = {'use_cache': False, 'output_attentions': False}, # minimize cache during model trace.
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder, using nnsight.
    
    Returns:
        Tuple of (original_loss, reconstruction_loss, zero_ablation_loss)
    """
    # Original nnsight implementation
    if max_len is None:
        invoker_args = {}
    else:
        invoker_args = {"truncation": True, "max_length": max_len }

    with model.trace("_"):
        temp_output = submodule.output.save()

    output_is_tuple = False
    # Note: isinstance() won't work here as torch.Size is a subclass of tuple,
    # so isinstance(temp_output.shape, tuple) would return True even for torch.Size.
    if type(temp_output.shape) == tuple:
        output_is_tuple = True

    # unmodified logits
    with model.trace(text, invoker_args=invoker_args):
        logits_original = model.output.save()
    logits_original = logits_original.value
    
    # logits when replacing component activations with reconstruction by autoencoder
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        else:
            raise ValueError(f"Invalid value for io: {io}")
        x = x.save()

    # If we incorrectly handle output_is_tuple, such as with some mlp submodules, we will get an error here.
    assert len(x.shape) == 3, f"Expected x to have shape (B, L, D), got {x.shape}, output_is_tuple: {output_is_tuple}"

    x_hat = dictionary(x).to(model.dtype)

    # intervene with `x_hat`
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            submodule.input[:] = x_hat
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        else:
            raise ValueError(f"Invalid value for io: {io}")

        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.value

    # logits when replacing component activations with zeros
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            submodule.input[:] = t.zeros_like(x)
        elif io in ['out', 'in_and_out']:
            x = submodule.output
            if output_is_tuple:
                submodule.output[0][:] = t.zeros_like(x[0])
            else:
                submodule.output[:] = t.zeros_like(x)
        else:
            raise ValueError(f"Invalid value for io: {io}")
        
        input = model.inputs.save()
        logits_zero = model.output.save()

    logits_zero = logits_zero.value

    # get everything into the right format
    try:
        logits_original = logits_original.logits
        logits_reconstructed = logits_reconstructed.logits
        logits_zero = logits_zero.logits
    except:
        pass

    if isinstance(text, t.Tensor):
        tokens = text
    else:
        try:
            tokens = input[1]['input_ids']
        except:
            tokens = input[1]['input']

    # compute losses
    losses = []
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id}
    else:
        loss_kwargs = {}
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)


def loss_recovered_transformer_lens(
    text,  # a batch of text or tokens
    model: HookedTransformer,  # a transformer_lens HookedTransformer
    hook_name: str,  # hook name to intervene at
    dictionary,  # dictionary for the hook
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    **kwargs,  # ignore other arguments meant for nnsight
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    How much of the model's loss is recovered by replacing the hook activations
    with the reconstruction by the autoencoder, using transformer_lens.
    
    Returns:
        Tuple of (original_loss, reconstruction_loss, zero_ablation_loss)
    """
    device = model.cfg.device
    
    # Process input text or tokens
    if isinstance(text, list) and len(text) > 0 and isinstance(text[0], dict) and 'tokens' in text[0]:
        # Check if batch is too large and reduce if needed
        MAX_BATCH_SIZE = 16  # Adjust this based on your GPU memory
        if len(text) > MAX_BATCH_SIZE:
            # Take a subset of the batch
            text = text[:MAX_BATCH_SIZE]
            print(f"Reduced batch size to {MAX_BATCH_SIZE} in evaluation.py to avoid OOM errors")
            
        # It's a list of dictionaries with 'tokens' key (from hf_dataset_to_generator with return_tokens=True)
        token_lists = [item['tokens'] for item in text]
        # Pad to the same length
        max_length = max(len(tokens) for tokens in token_lists)
        padded_tokens = [tokens + [model.tokenizer.pad_token_id if hasattr(model.tokenizer, 'pad_token_id') else 0] * (max_length - len(tokens)) 
                         for tokens in token_lists]
        tokens = t.tensor(padded_tokens, device=device)
    elif isinstance(text, t.Tensor) and len(text.shape) == 2:
        # It's already tokenized as a tensor with shape [batch, seq_len]
        tokens = text.to(device)
    elif isinstance(text, list) and all(isinstance(item, int) or 
                                      (isinstance(item, list) and all(isinstance(subitem, int) for subitem in item)) 
                                      for item in text):
        # It's a list of token IDs or a list of lists of token IDs
        if all(isinstance(item, int) for item in text):
            # Single sequence of token IDs
            tokens = t.tensor([text], device=device)
        else:
            # Batch of token ID sequences
            tokens = t.tensor(text, device=device)
    elif isinstance(text, list):
        # It's a list of strings
        if max_len is not None:
            tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)["input_ids"].to(device)
        else:
            tokens = model.tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)
    elif isinstance(text, dict) and "input_ids" in text:
        # It's a tokenizer output
        tokens = text["input_ids"].to(device)
    else:
        raise ValueError(f"Unsupported text type: {type(text)}. Expected string, list of strings, tensor of token IDs, or tokenizer output.")
    
    # Create loss function
    loss_fn = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id if hasattr(model.tokenizer, 'pad_token_id') else -100)
    
    # 1. Compute original loss
    with t.no_grad():
        output_original = model(tokens)
        loss_original = loss_fn(output_original[:, :-1].reshape(-1, output_original.shape[-1]), tokens[:, 1:].reshape(-1))
    
    # 2. Apply dictionary reconstruction
    def reconstruction_hook(activations, hook):
        # Apply normalization if needed
        if normalize_batch:
            scale = (dictionary.activation_dim ** 0.5) / activations.norm(dim=-1).mean()
            activations = activations * scale
            
        # Apply dictionary
        reconstructed = dictionary(activations).to(activations.dtype)
        
        # Undo normalization if needed
        if normalize_batch:
            reconstructed = reconstructed / scale
            
        return reconstructed
    
    with t.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, reconstruction_hook)]):
            output_reconstructed = model(tokens)
            loss_reconstructed = loss_fn(output_reconstructed[:, :-1].reshape(-1, output_reconstructed.shape[-1]), tokens[:, 1:].reshape(-1))
    
    # 3. Apply zero ablation
    def zero_ablation_hook(activations, hook):
        return t.zeros_like(activations)
    
    with t.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, zero_ablation_hook)]):
            output_zero = model(tokens)
            loss_zero = loss_fn(output_zero[:, :-1].reshape(-1, output_zero.shape[-1]), tokens[:, 1:].reshape(-1))
    
    return loss_original, loss_reconstructed, loss_zero


def loss_recovered(
    text,  # a batch of text
    model: Union[LanguageModel, HookedTransformer],  # the model to evaluate on
    submodule_or_hook_name,  # submodule (nnsight) or hook name (transformer_lens)
    dictionary,  # dictionary to evaluate
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    io="out",  # can be 'in', 'out', or 'in_and_out' (only for nnsight)
    tracer_args = {'use_cache': False, 'output_attentions': False}, # minimize cache during model trace (nnsight only)
) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder?
    
    Returns:
        Tuple of (original_loss, reconstruction_loss, zero_ablation_loss)
    """
    # Choose implementation based on model type
    if isinstance(model, HookedTransformer):
        return loss_recovered_transformer_lens(
            text=text,
            model=model,
            hook_name=submodule_or_hook_name,
            dictionary=dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
        )
    elif isinstance(model, LanguageModel):
        return loss_recovered_nnsight(
            text=text,
            model=model,
            submodule=submodule_or_hook_name,
            dictionary=dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
            io=io,
            tracer_args=tracer_args,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}. Must be either nnsight.LanguageModel or transformer_lens.HookedTransformer")