"""
Utilities for evaluating dictionaries on a model and dataset.
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
    text,  # a batch of text
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
    
    # Process input text
    if isinstance(text, list):
        # It's a list of strings
        if max_len is not None:
            tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)["input_ids"].to(device)
        else:
            tokens = model.tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)
    elif isinstance(text, dict):
        # It's a tokenizer output
        tokens = text["input_ids"].to(device)
    elif isinstance(text, t.Tensor):
        # It's already tokenized
        tokens = text.to(device)
    else:
        raise ValueError(f"Unsupported text type: {type(text)}")
    
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
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        
        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32) # If f is shape (B, L, D), flatten to (B*L, D)
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2

        active_features += features_BF.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        #compute variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance)

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

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
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            activations.model,
            activations.submodule if hasattr(activations, 'submodule') else activations.hook_name,
            dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
            io=io,
            tracer_args=tracer_args
        )
        
        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
        
        out["loss_original"] += loss_original.item()
        out["loss_reconstructed"] += loss_reconstructed.item()
        out["loss_zero"] += loss_zero.item()
        out["frac_recovered"] += frac_recovered.item()

    out = {key: value / n_batches for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    return out