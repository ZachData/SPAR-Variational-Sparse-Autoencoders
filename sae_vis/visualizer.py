"""
Connector script to link dictionary learning models with the visualization framework.
"""
import sys
import torch as t
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

# Add support for dictionary learning models
sys.path.append(str(Path(__file__).parent.parent))

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name

from sae_vis.data_config_classes import (
    ActsHistogramConfig, Column, FeatureTablesConfig, 
    SaeVisConfig, SaeVisLayoutConfig, SeqMultiGroupConfig,
    LogitsTableConfig, LogitsHistogramConfig
)
from sae_vis.data_storing_fns import SaeVisData
import sae_vis.model_fns

class AutoEncoderAdapter:
    """
    Adapter class to map your AutoEncoder to what sae_vis expects.
    Handles differences in dimensions and structure between various SAE implementations.
    """
    
    def __init__(self, sae, hook_name="blocks.0.mlp.hook_post", hook_layer=0):
        self.sae = sae
        
        # Create a config object with necessary attributes
        class Config:
            def __init__(self):
                self.hook_name = hook_name
                self.hook_layer = hook_layer
                self.d_in = sae.activation_dim
                self.d_sae = sae.dict_size
        
        # Add the config to the adapter
        self.cfg = Config()
        
        # Store the original activation dimensions
        self.activation_dim = sae.activation_dim
        self.dict_size = sae.dict_size
        
        # Map weight matrices
        # The visualization expects decoder weights in shape [dict_size, d_model]
        # and encoder weights in shape [d_model, dict_size]
        self.W_dec = sae.decoder.weight  # [dict_size, activation_dim]
        self.W_enc = sae.encoder.weight.T  # [activation_dim, dict_size]
        
        # Bias terms
        self.b_dec = sae.bias if hasattr(sae, 'bias') else (
            sae.decoder.bias if hasattr(sae.decoder, 'bias') else None
        )
        self.b_enc = sae.encoder.bias if hasattr(sae.encoder, 'bias') else None
        
        # Other potentially needed attributes
        self.use_error_term = False
    
    def encode(self, x, **kwargs):
        # Forward any additional kwargs that SAE-vis might pass
        return self.sae.encode(x)
    
    def decode(self, f, **kwargs):
        # Forward any additional kwargs that SAE-vis might pass
        return self.sae.decode(f)
    
    def forward(self, x, output_features=False, **kwargs):
        # Make sure this returns both decoded x and features if requested
        if output_features:
            x_hat, f = self.sae(x, output_features=True)
            return x_hat, f
        else:
            return self.sae(x)
            
    def state_dict(self):
        """Return the state dictionary of the wrapped SAE."""
        return self.sae.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load a state dictionary into the wrapped SAE."""
        return self.sae.load_state_dict(state_dict)
    
    def to(self, *args, **kwargs):
        """Move the SAE to the specified device."""
        return self.sae.to(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return getattr(self.sae, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class ModelAdapter:
    """
    Adapter class to make HookedTransformer compatible with sae_vis.
    Provides the expected interface and handles hook name mapping.
    """
    
    def __init__(self, model, hook_name=None):
        self.model = model
        self.W_U = model.W_U
        self.b_U = model.b_U
        self.cfg = model.cfg
        self.tokenizer = model.tokenizer
        self.hook_name = hook_name
        
        # Define mappings between common hook names
        self.hook_mappings = {
            "blocks.0.mlp.hook_post": ["blocks.0.hook_mlp_out", "blocks.0.mlp.hook_post"],
            "blocks.0.mlp.hook_pre": ["blocks.0.mlp.hook_pre"],
            "blocks.0.hook_mlp_out": ["blocks.0.hook_mlp_out", "blocks.0.mlp.hook_post"]
        }
    
    def run_with_cache_with_saes(self, tokens, saes=None, stop_at_layer=None, names_filter=None, remove_batch_dim=False):
        """
        Run the model with cache and add SAE activations to the cache.
        
        This method emulates the behavior expected by sae_vis when running with SAEs.
        """
        # Ensure we're requesting all needed hooks
        required_hooks = ['blocks.0.hook_resid_post']
        
        if saes:
            for sae in saes:
                hook_name = sae.cfg.hook_name
                # Add original hook name and any alternative names
                required_hooks.append(hook_name)
                if hook_name in self.hook_mappings:
                    required_hooks.extend(self.hook_mappings[hook_name])
        
        # Combine with any names_filter provided
        if names_filter:
            required_hooks.extend(names_filter)
            
        # Remove duplicates
        required_hooks = list(set(required_hooks))
            
        # Run the model with cache
        output, cache = self.model.run_with_cache(
            tokens, 
            stop_at_layer=stop_at_layer,
            names_filter=required_hooks
        )
        
        # Get the cache as a dictionary we can modify
        cache_dict = dict(cache.cache_dict)
        
        # Print available hooks for debugging
        print(f"Available hooks in cache: {sorted(list(cache_dict.keys()))}")
        
        # Process SAEs
        if saes:
            for sae in saes:
                original_hook_name = sae.cfg.hook_name
                
                # Find the best available hook for this SAE
                hook_name = self._find_best_hook(original_hook_name, cache_dict)
                
                if hook_name:
                    # Get activations and compute SAE outputs
                    activations = cache_dict[hook_name].to(sae.W_enc.device)
                    acts_post = sae.encode(activations)
                    
                    # Add to cache
                    cache_dict[f"{original_hook_name}.hook_sae_acts_post"] = acts_post
                    cache_dict[f"{original_hook_name}.hook_sae_input"] = activations
                    
                    print(f"Added SAE activations using hook '{hook_name}' to cache")
                else:
                    # If no suitable hook is found, use residual post as fallback
                    if 'blocks.0.hook_resid_post' in cache_dict:
                        activations = cache_dict['blocks.0.hook_resid_post'].to(sae.W_enc.device)
                        acts_post = sae.encode(activations)
                        
                        cache_dict[f"{original_hook_name}.hook_sae_acts_post"] = acts_post
                        cache_dict[f"{original_hook_name}.hook_sae_input"] = activations
                        
                        print(f"Using residual post as fallback for '{original_hook_name}'")
                    else:
                        raise KeyError(f"No suitable hook found for '{original_hook_name}'")
        
        # Handle remove_batch_dim if needed
        if remove_batch_dim:
            assert tokens.shape[0] == 1, "Can only remove batch dim if batch size is 1"
            output = output[0]
            cache_dict = {k: v[0] if isinstance(v, t.Tensor) and v.dim() > 0 and v.shape[0] == 1 else v 
                         for k, v in cache_dict.items()}
        
        # Create a new ActivationCache object
        new_cache = ActivationCache(cache_dict, model=self.model)
        
        return output, new_cache
    
    def _find_best_hook(self, hook_name, cache_dict):
        """Find the best available hook that matches the given hook name."""
        # First check if the exact hook is available
        if hook_name in cache_dict:
            return hook_name
        
        # Then check the mappings
        if hook_name in self.hook_mappings:
            for alt_hook in self.hook_mappings[hook_name]:
                if alt_hook in cache_dict:
                    return alt_hook
        
        # Try to guess based on components of the name
        if "mlp" in hook_name and "post" in hook_name:
            possible_hooks = [k for k in cache_dict if "mlp" in k.lower() and 
                             ("post" in k.lower() or "out" in k.lower())]
            if possible_hooks:
                return possible_hooks[0]
        
        return None
    
    def __getattr__(self, name):
        """Forward attributes to the underlying model."""
        return getattr(self.model, name)


class SAEVisAdapter:
    """Helper class to adapt dictionary learning models to the visualization framework."""
    
    @staticmethod
    def create_default_layout(othello_mode=False, height=1000, simplified=False):
        """
        Create a visualization layout configuration.
        
        Args:
            othello_mode: Whether to use the Othello-specific layout
            height: Height of the visualization in pixels
            simplified: Whether to use a simplified layout with fewer components
        """
        if simplified:
            return SaeVisLayoutConfig(
                columns=[
                    Column(ActsHistogramConfig(), width=400),
                ],
                height=500,
            )
        elif othello_mode:
            return SaeVisLayoutConfig.default_othello_layout()
        else:
            return SaeVisLayoutConfig(
                columns=[
                    Column(FeatureTablesConfig(n_rows=5), width=400),
                    Column(
                        ActsHistogramConfig(), 
                        LogitsTableConfig(n_rows=10), 
                        LogitsHistogramConfig(),
                        width=500
                    ),
                    Column(
                        SeqMultiGroupConfig(
                            buffer=(5, 5), 
                            n_quantiles=5, 
                            top_acts_group_size=20
                        ),
                        width=1000
                    ),
                ],
                height=height,
            )
    
    @staticmethod
    def patch_to_resid_dir():
        """
        Patch the to_resid_dir function to handle dimension mismatches.
        Returns the original function for restoration.
        """
        original_to_resid_dir = sae_vis.model_fns.to_resid_dir
        
        def flexible_to_resid_dir(dir, sae, model, input=False):
            """
            A more flexible version of to_resid_dir that handles dimension mismatches.
            """
            # Handle case where dir and model dimensions don't match by returning identity
            try:
                # First, determine the hook type
                if hasattr(sae, 'cfg') and hasattr(sae.cfg, 'hook_name'):
                    hook_type = sae.cfg.hook_name.split(".hook_")[-1]
                else:
                    hook_type = "custom"
                    
                # For most common hook types, just return identity
                if hook_type in ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "custom"]:
                    return dir
                
                # For MLP pre/post hooks, check dimensions before multiplying
                elif hook_type in ["pre", "post"]:
                    # Check if dimensions are compatible
                    if input:
                        W_in = model.W_in[sae.cfg.hook_layer].T
                        if dir.shape[-1] != W_in.shape[0]:
                            print(f"Dimension input mismatch: {dir.shape[-1]} != {W_in.shape[0]}, using identity")
                            return dir
                        return dir @ W_in
                    else:
                        W_out = model.W_out[sae.cfg.hook_layer]
                        if dir.shape[-1] != W_out.shape[0]:
                            print(f"Dimension output mismatch: {dir.shape[-1]} != {W_out.shape[0]}, using identity")
                            return dir
                        return dir @ W_out
                
                # For attention hooks
                elif hook_type == "z":
                    if input:
                        W_V = model.W_V[sae.cfg.hook_layer]
                        V_flat = einops.rearrange(W_V, "n_heads d_model d_head -> (n_heads d_head) d_model")
                        if dir.shape[-1] != V_flat.shape[0]:
                            print(f"Dimension mismatch: {dir.shape[-1]} != {V_flat.shape[0]}, using identity")
                            return dir
                        return dir @ V_flat
                    else:
                        W_O = model.W_O[sae.cfg.hook_layer]
                        O_flat = einops.rearrange(W_O, "n_heads d_head d_model -> (n_heads d_head) d_model")
                        if dir.shape[-1] != O_flat.shape[0]:
                            print(f"Dimension mismatch: {dir.shape[-1]} != {O_flat.shape[0]}, using identity")
                            return dir
                        return dir @ O_flat
                else:
                    print(f"Unknown hook type '{hook_type}', using identity mapping")
                    return dir
            except Exception as e:
                print(f"Error in to_resid_dir: {e}, using identity mapping")
                return dir
        
        # Apply the patch
        sae_vis.model_fns.to_resid_dir = flexible_to_resid_dir
        
        return original_to_resid_dir
    
    @staticmethod
    def create_visualization(
        sae_path: str,
        tokens: t.Tensor,
        output_file: str,
        model_name: str,
        hook_name: str = "blocks.0.mlp.hook_post",
        hook_layer: int = 0,
        feature_indices: Optional[Union[int, List[int]]] = None,
        layout: Optional[SaeVisLayoutConfig] = None,
        device: str = "cuda",
        verbose: bool = True,
        simplified: bool = False,
        feature_limit: Optional[int] = None,
    ):
        """
        Create a visualization for a trained SAE model.
        
        Args:
            sae_path: Path to the trained SAE
            tokens: Tokenized data for visualization
            output_file: Path to save the HTML visualization
            model_name: Name of the transformer model to load
            hook_name: Name of the hook point in the model
            hook_layer: Layer index of the hook point
            feature_indices: Specific feature indices to visualize
            layout: Visualization layout configuration
            device: Device to run computations on
            verbose: Whether to print progress information
            simplified: Whether to use a simplified layout initially
            feature_limit: Maximum number of features to visualize (to avoid OOM)
        
        Returns:
            SaeVisData or None: The visualization data if successful, None otherwise
        """
        # Create output directory if needed
        Path(output_file).parent.mkdir(exist_ok=True, parents=True)
        
        # 1. Load SAE
        if verbose:
            print(f"Loading SAE from {sae_path}...")
            
        from dictionary_learning.utils import load_dictionary
        sae, config = load_dictionary(sae_path, device)
        
        # 2. Adapt the SAE
        adapted_sae = AutoEncoderAdapter(sae, hook_name=hook_name, hook_layer=hook_layer)
        
        # 3. Load the model
        if verbose:
            print(f"Loading model {model_name}...")
            
        model = HookedTransformer.from_pretrained(model_name, device=device)
        
        # 4. Configure which features to visualize
        if feature_indices is None:
            # Limit to a reasonable number of features to avoid OOM
            max_features = feature_limit or min(256, sae.dict_size)
            feature_indices = list(range(max_features))
        elif isinstance(feature_indices, int):
            feature_indices = [feature_indices]
            
        # Limit to feature_limit if specified
        if feature_limit is not None and len(feature_indices) > feature_limit:
            feature_indices = feature_indices[:feature_limit]
        
        # 5. Use default layout if none specified
        if layout is None:
            layout = SAEVisAdapter.create_default_layout(simplified=simplified)
        
        # 6. Create model adapter
        adapted_model = ModelAdapter(model, hook_name=hook_name)
        
        # 7. Patch necessary functions
        original_to_resid_dir = SAEVisAdapter.patch_to_resid_dir()
        
        try:
            # 8. Create visualization data
            if verbose:
                print(f"Creating visualization data for {len(feature_indices)} features...")
            
            sae_vis_data = SaeVisData.create(
                sae=adapted_sae,
                model=adapted_model,
                tokens=tokens,
                cfg=SaeVisConfig(
                    features=feature_indices,
                    feature_centric_layout=layout
                ),
                verbose=verbose,
            )
            
            # 9. Save the visualization
            if verbose:
                print(f"Saving visualization to {output_file}...")
                
            sae_vis_data.save_feature_centric_vis(
                filename=output_file,
                feature=feature_indices[0],
                verbose=verbose
            )
            
            if verbose:
                print(f"✅ Visualization saved successfully to {output_file}")
            
            return sae_vis_data
            
        except Exception as e:
            print(f"❌ Error during visualization creation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a fallback visualization if everything else fails
            try:
                if verbose:
                    print("Attempting to create a basic fallback visualization...")
                
                # Create an extremely simplified visualization that just shows activations
                fallback_visualization(
                    sae=adapted_sae,
                    model=model,
                    tokens=tokens,
                    output_file=output_file,
                    feature_indices=feature_indices[:5],  # Just show first 5 features
                    device=device
                )
                
                if verbose:
                    print(f"✅ Basic fallback visualization saved to {output_file}")
            except Exception as fallback_error:
                print(f"❌ Fallback visualization also failed: {fallback_error}")
            
            return None
        finally:
            # Restore original functions
            sae_vis.model_fns.to_resid_dir = original_to_resid_dir


def fallback_visualization(
    sae,
    model,
    tokens,
    output_file,
    feature_indices=[0],
    device="cuda"
):
    """
    Create a very basic HTML visualization of feature activations as fallback.
    
    Args:
        sae: The SAE model (adapted)
        model: The transformer model
        tokens: Input tokens
        output_file: Path to save HTML file
        feature_indices: Feature indices to visualize
        device: Device to run on
    """
    # Limit to max 10 features for the fallback
    feature_indices = feature_indices[:10]
    
    # Run the model and get activations
    with t.inference_mode():
        # Run on a subset of tokens to avoid OOM
        sample_tokens = tokens[:min(32, tokens.shape[0])].to(device)
        _, cache = model.run_with_cache(sample_tokens, names_filter=['blocks.0.hook_resid_post'])
        
        # Get residual post activations
        resid_post = cache['blocks.0.hook_resid_post']
        
        # Get SAE activations for features of interest
        sae_acts = sae.encode(resid_post)
        
        # Extract activations for the specified features
        feature_acts = []
        for idx in feature_indices:
            if idx < sae_acts.shape[-1]:
                acts = sae_acts[..., idx].detach().cpu().numpy().flatten()
                # Count non-zero activations
                nonzero = (acts > 0).sum()
                sparsity = nonzero / acts.size
                feature_acts.append((idx, acts, sparsity))
    
    # Create a simple HTML file with histograms
    with open(output_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SAE Feature Visualization (Fallback)</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .histogram { width: 800px; height: 300px; margin-bottom: 30px; }
                h1 { color: #333; }
                .info { margin-bottom: 20px; color: #666; }
            </style>
        </head>
        <body>
            <h1>SAE Feature Visualization (Fallback)</h1>
            <div class="info">
                <p>This is a simplified fallback visualization showing feature activation histograms.</p>
            </div>
        """)
        
        # Generate histogram divs
        for i, (feat_idx, acts, sparsity) in enumerate(feature_acts):
            f.write(f"""
            <div class="feature-section">
                <h2>Feature {feat_idx} (Sparsity: {sparsity:.2%})</h2>
                <div id="histogram-{i}" class="histogram"></div>
            </div>
            """)
        
        # Generate JavaScript for the histograms
        f.write("<script>")
        
        for i, (feat_idx, acts, sparsity) in enumerate(feature_acts):
            # Convert to list for JSON serialization
            acts_list = acts.tolist()
            
            f.write(f"""
            // Data for histogram {i}
            const data{i} = {{
                x: {acts_list},
                type: 'histogram',
                marker: {{
                    color: 'rgba(255, 100, 0, 0.7)',
                }}
            }};
            
            // Layout
            const layout{i} = {{
                title: 'Feature {feat_idx} Activations',
                xaxis: {{ title: 'Activation Value' }},
                yaxis: {{ title: 'Frequency' }}
            }};
            
            // Create the plot
            Plotly.newPlot('histogram-{i}', [data{i}], layout{i});
            """)
        
        f.write("</script></body></html>")