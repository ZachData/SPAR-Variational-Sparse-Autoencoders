"""
Utility classes for storing and retrieving activations from various models.
"""

import torch as t
from typing import Optional, Generator, Any, Union, Type
import gc
from tqdm import tqdm

# Import for nnsight functionality
from nnsight import LanguageModel

# Import for transformer_lens functionality
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}


class BaseActivationBuffer:
    """
    Abstract base class that defines the interface for activation buffers.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 d_submodule: Optional[int] = None, # submodule dimension
                 n_ctxs: int = 3e4, # approximate number of contexts to store in the buffer
                 ctx_len: int = 128, # length of each context
                 refresh_batch_size: int = 512, # size of batches in which to process the data when adding to buffer
                 out_batch_size: int = 8192, # size of batches in which to yield activations
                 device: str = 'cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        self.data = data
        self.d_submodule = d_submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos
        
        # These will be initialized in subclasses
        self.activations = None
        self.read = None
        
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def refresh(self):
        """
        Refresh the activation buffer by gathering new activations.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement refresh()")
    
    @property
    def config(self):
        """Return configuration details for the buffer"""
        return {
            'd_submodule': self.d_submodule,
            'n_ctxs': self.n_ctxs,
            'ctx_len': self.ctx_len,
            'refresh_batch_size': self.refresh_batch_size,
            'out_batch_size': self.out_batch_size,
            'device': self.device
        }
    
    def close(self):
        """Close the text stream"""
        if hasattr(self, 'text_stream'):
            self.text_stream.close()


class ActivationBuffer(BaseActivationBuffer):
    """
    Implements a buffer of activations from an nnsight model.
    The buffer stores activations from a model, yields them in batches,
    and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model: LanguageModel, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 io='out', # can be 'in' or 'out'; whether to extract input or output activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        
        super().__init__(data, d_submodule, n_ctxs, ctx_len, refresh_batch_size, 
                         out_batch_size, device, remove_bos)
        
        self.model = model
        self.submodule = submodule
        self.io = io
        
        # Initialize activations buffer
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
        self.read = t.zeros(0).bool()
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar when filling buffer
        pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                # Get batch of text or tokens
                batch_data = self.text_batch()
                
                # Handle different types of inputs
                if isinstance(batch_data, t.Tensor):
                    # Direct tensor of tokens
                    tokens = batch_data.to(self.model.device)
                    with self.model.trace(tokens, **tracer_kwargs):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input_vals = self.model.inputs.save()
                        self.submodule.output.stop()
                    
                    # Create attention mask (all ones)
                    attn_mask = t.ones_like(tokens)
                    
                elif isinstance(batch_data, dict) and "tokens" in batch_data:
                    # Dictionary with tokens
                    if isinstance(batch_data["tokens"], t.Tensor):
                        tokens = batch_data["tokens"].to(self.model.device)
                    elif isinstance(batch_data["tokens"], list):
                        tokens = t.tensor(batch_data["tokens"]).to(self.model.device)
                    
                    with self.model.trace(tokens, **tracer_kwargs):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input_vals = self.model.inputs.save()
                        self.submodule.output.stop()
                    
                    # Create attention mask (all ones)
                    attn_mask = t.ones_like(tokens)
                    
                elif isinstance(batch_data, list) and all(isinstance(item, dict) and "tokens" in item for item in batch_data):
                    # List of dictionaries with tokens
                    token_lists = [item["tokens"] for item in batch_data]
                    if all(isinstance(tokens, t.Tensor) for tokens in token_lists):
                        # Convert list of tensors to batched tensor
                        tokens = t.stack(token_lists).to(self.model.device)
                    else:
                        # Convert list of lists to tensor
                        tokens = t.tensor(token_lists).to(self.model.device)
                    
                    with self.model.trace(tokens, **tracer_kwargs):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input_vals = self.model.inputs.save()
                        self.submodule.output.stop()
                    
                    # Create attention mask (all ones)
                    attn_mask = t.ones_like(tokens)
                    
                elif isinstance(batch_data, list) and all(isinstance(item, int) for item in batch_data[0]):
                    # List of token lists
                    tokens = t.tensor(batch_data).to(self.model.device)
                    with self.model.trace(tokens, **tracer_kwargs):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input_vals = self.model.inputs.save()
                        self.submodule.output.stop()
                    
                    # Create attention mask (all ones)
                    attn_mask = t.ones_like(tokens)
                    
                else:
                    # Raw text that needs tokenization
                    with self.model.trace(
                        batch_data,
                        **tracer_kwargs,
                        invoker_args={"truncation": True, "max_length": self.ctx_len},
                    ):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input_vals = self.model.inputs.save()
                        self.submodule.output.stop()
                    
                    # Get attention mask from tokenization
                    if hasattr(input_vals.value, "__getitem__") and isinstance(input_vals.value[1], dict) and "attention_mask" in input_vals.value[1]:
                        attn_mask = input_vals.value[1]["attention_mask"]
                    else:
                        # If attention mask is not available, create one (all ones)
                        attn_mask = t.ones_like(input_vals.value[1]["input_ids"] if hasattr(input_vals.value, "__getitem__") else input_vals.value)
                
                # Process the hidden states
                hidden_states = hidden_states.value
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                
                # Handle BOS token removal if needed
                if self.remove_bos:
                    hidden_states = hidden_states[:, 1:, :]
                    attn_mask = attn_mask[:, 1:]
                
                # Use attention mask to filter valid tokens
                hidden_states = hidden_states[attn_mask != 0]

                # Add to buffer
                remaining_space = self.activation_buffer_size - current_idx
                assert remaining_space > 0
                hidden_states = hidden_states[:remaining_space]

                self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                    self.device
                )
                current_idx += len(hidden_states)

                pbar.update(len(hidden_states))

        pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
class TransformerLensActivationBuffer(BaseActivationBuffer):
    """
    Implements a buffer of activations from a transformer_lens model.
    The buffer stores activations from a model, yields them in batches,
    and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model: HookedTransformer, # HookedTransformer from which to extract activations
                 hook_name: str, # name of the hook point to extract activations from
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        # Validate hook name exists in the model
        if hook_name not in model.hook_dict:
            available_hooks = list(model.hook_dict.keys())
            raise ValueError(f"Hook name '{hook_name}' not found in model. Available hooks: {available_hooks}")
            
        # Auto-detect dimension if not provided
        if d_submodule is None:
            # First, run a small forward pass to see the shape of activations
            dummy_input = model.tokenizer("Test input", return_tensors="pt")
            dummy_tokens = dummy_input["input_ids"].to(model.cfg.device)
            
            # Create a temporary hook to get the activation shape
            act_shape = [None]
            def get_shape_hook(tensor, hook):
                act_shape[0] = tensor.shape[-1]
                return tensor
                
            with model.hooks(fwd_hooks=[(hook_name, get_shape_hook)]):
                model(dummy_tokens)
                
            if act_shape[0] is None:
                raise ValueError(f"Could not determine dimension for hook '{hook_name}'")
                
            d_submodule = act_shape[0]
            
        super().__init__(data, d_submodule, n_ctxs, ctx_len, refresh_batch_size, 
                         out_batch_size, device, remove_bos)
        
        self.model = model
        self.hook_name = hook_name
        
        # Initialize activations buffer
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.cfg.dtype)
        self.read = t.zeros(0).bool()
        
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )
        
    def refresh(self):
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()
                
        # Keep unread activations
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.cfg.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                # Get batch of text or tokens
                batch_data = self.text_batch()
                
                # Handle different types of inputs
                if isinstance(batch_data, t.Tensor):
                    # Direct tensor of tokens
                    tokens = batch_data.to(self.model.cfg.device)
                    attention_mask = t.ones_like(tokens).to(self.model.cfg.device)
                    
                elif isinstance(batch_data, dict) and "tokens" in batch_data:
                    # Dictionary with tokens
                    if isinstance(batch_data["tokens"], t.Tensor):
                        tokens = batch_data["tokens"].to(self.model.cfg.device)
                    elif isinstance(batch_data["tokens"], list):
                        tokens = t.tensor(batch_data["tokens"]).to(self.model.cfg.device)
                    attention_mask = t.ones_like(tokens).to(self.model.cfg.device)
                    
                elif isinstance(batch_data, list) and all(isinstance(item, dict) and "tokens" in item for item in batch_data):
                    # List of dictionaries with tokens
                    token_lists = [item["tokens"] for item in batch_data]
                    if all(isinstance(tokens, t.Tensor) for tokens in token_lists):
                        # Convert list of tensors to batched tensor
                        tokens = t.stack(token_lists).to(self.model.cfg.device)
                    else:
                        # Convert list of lists to tensor
                        tokens = t.tensor(token_lists).to(self.model.cfg.device)
                    attention_mask = t.ones_like(tokens).to(self.model.cfg.device)
                    
                elif isinstance(batch_data, list) and batch_data and all(isinstance(item, int) for item in batch_data[0] if batch_data[0]):
                    # List of token lists
                    tokens = t.tensor(batch_data).to(self.model.cfg.device)
                    attention_mask = t.ones_like(tokens).to(self.model.cfg.device)
                    
                else:
                    # Raw text that needs tokenization
                    tokenized = self.model.tokenizer(
                        batch_data,
                        return_tensors='pt',
                        max_length=self.ctx_len,
                        padding=True,
                        truncation=True
                    )
                    tokens = tokenized["input_ids"].to(self.model.cfg.device)
                    attention_mask = tokenized["attention_mask"].to(self.model.cfg.device)
                
                # Storage for activations
                batch_activations = []
                
                # Define hook function to collect activations
                def activation_hook(tensor, hook):
                    # Store a copy of the tensor
                    batch_activations.append(tensor.detach().clone())
                    return tensor
                
                # Run forward pass with hook
                with self.model.hooks(fwd_hooks=[(self.hook_name, activation_hook)]):
                    self.model(tokens)
                
                # Get the collected activations (should be just one tensor)
                if len(batch_activations) != 1:
                    raise ValueError(f"Expected 1 activation tensor, got {len(batch_activations)}")
                
                hidden_states = batch_activations[0]
                
                # Apply attention mask and flatten activations
                if self.remove_bos:
                    hidden_states = hidden_states[:, 1:, :]
                    attention_mask = attention_mask[:, 1:]
                
                # Filter by attention mask and flatten
                batch_size, seq_len = attention_mask.shape
                hidden_states = hidden_states.reshape(batch_size * seq_len, -1)
                attention_mask = attention_mask.reshape(-1)
                hidden_states = hidden_states[attention_mask != 0]
                
                # Add to buffer
                remaining_space = self.activation_buffer_size - current_idx
                assert remaining_space > 0
                hidden_states = hidden_states[:remaining_space]

                self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                    self.device
                )
                current_idx += len(hidden_states)

                pbar.update(len(hidden_states))

        pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
        
# Factory function to create the appropriate buffer based on model type
def create_activation_buffer(
    data,
    model: Union[LanguageModel, HookedTransformer],
    submodule_or_hook_name,
    d_submodule=None,
    io='out',
    n_ctxs=3e4,
    ctx_len=128,
    refresh_batch_size=512,
    out_batch_size=8192,
    device='cpu',
    remove_bos=False,
) -> BaseActivationBuffer:
    """
    Create an activation buffer appropriate for the model type.
    
    Args:
        model: Either an nnsight LanguageModel or a transformer_lens HookedTransformer
        submodule_or_hook_name: For nnsight, a submodule; for transformer_lens, a hook name
        
    Returns:
        An appropriate activation buffer instance
    """
    if isinstance(model, HookedTransformer):
        # It's a transformer_lens model
        return TransformerLensActivationBuffer(
            data=data,
            model=model,
            hook_name=submodule_or_hook_name,
            d_submodule=d_submodule,
            n_ctxs=n_ctxs,
            ctx_len=ctx_len,
            refresh_batch_size=refresh_batch_size,
            out_batch_size=out_batch_size,
            device=device,
            remove_bos=remove_bos,
        )
    elif isinstance(model, LanguageModel):
        # It's an nnsight model
        return ActivationBuffer(
            data=data,
            model=model,
            submodule=submodule_or_hook_name,
            d_submodule=d_submodule,
            io=io,
            n_ctxs=n_ctxs,
            ctx_len=ctx_len,
            refresh_batch_size=refresh_batch_size,
            out_batch_size=out_batch_size,
            device=device,
            remove_bos=remove_bos,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}. Must be either nnsight.LanguageModel or transformer_lens.HookedTransformer")