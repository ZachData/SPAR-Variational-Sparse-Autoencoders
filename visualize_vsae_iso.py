"""
Visualization script for VSAE ISO models trained on GELU-1L.
"""
import torch as t
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from sae_vis.visualizer import SAEVisAdapter
from sae_vis.data_config_classes import (
    Column, 
    FeatureTablesConfig, 
    ActsHistogramConfig,
    LogitsTableConfig,
    LogitsHistogramConfig,
    SaeVisLayoutConfig,
    SeqMultiGroupConfig
)

# Patch the SAE visualization functions to handle dimension mismatches
def patch_visualization_functions():
    """Patch sae_vis functions to handle dimension mismatches."""
    import torch
    import einops
    from sae_vis import model_fns, data_fetching_fns
    
    # Store original function for later restoration if needed
    original_to_resid_dir = model_fns.to_resid_dir
    
    def improved_to_resid_dir(dir, sae, model, input=False):
        """Enhanced version that handles dimension mismatches."""
        # First, determine the hook type
        if hasattr(sae, 'cfg') and hasattr(sae.cfg, 'hook_name'):
            hook_type = sae.cfg.hook_name.split(".hook_")[-1]
        else:
            # For adapters and custom SAEs
            hook_type = "custom"
            
        if hook_type in ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "custom"]:
            return dir
        elif hook_type in ["pre", "post"]:
            # Check if dimensions are compatible
            if input:
                W_in = model.W_in[sae.cfg.hook_layer].T
                if dir.shape[-1] != W_in.shape[0]:
                    print(f"Dimension mismatch: {dir.shape[-1]} != {W_in.shape[0]}, projecting dimensions")
                    # Project to the correct dimension
                    d_model = model.cfg.d_model
                    projected_dir = torch.zeros(*dir.shape[:-1], d_model, device=dir.device)
                    min_dim = min(dir.shape[-1], d_model)
                    projected_dir[..., :min_dim] = dir[..., :min_dim]
                    return projected_dir
                return dir @ W_in
            else:
                W_out = model.W_out[sae.cfg.hook_layer]
                if dir.shape[-1] != W_out.shape[0]:
                    print(f"Dimension mismatch: {dir.shape[-1]} != {W_out.shape[0]}, projecting dimensions")
                    # Project to the correct dimension
                    d_model = model.cfg.d_model
                    projected_dir = torch.zeros(*dir.shape[:-1], d_model, device=dir.device)
                    min_dim = min(dir.shape[-1], d_model)
                    projected_dir[..., :min_dim] = dir[..., :min_dim]
                    return projected_dir
                return dir @ W_out
        elif hook_type == "z":
            # Handle attention hooks
            if input:
                W_V = model.W_V[sae.cfg.hook_layer]
                V_flat = einops.rearrange(W_V, "n_heads d_model d_head -> (n_heads d_head) d_model")
                if dir.shape[-1] != V_flat.shape[0]:
                    print(f"Dimension mismatch: {dir.shape[-1]} != {V_flat.shape[0]}, projecting dimensions")
                    # Project to the correct dimension
                    d_model = model.cfg.d_model
                    projected_dir = torch.zeros(*dir.shape[:-1], d_model, device=dir.device)
                    min_dim = min(dir.shape[-1], d_model)
                    projected_dir[..., :min_dim] = dir[..., :min_dim]
                    return projected_dir
                return dir @ V_flat
            else:
                W_O = model.W_O[sae.cfg.hook_layer]
                O_flat = einops.rearrange(W_O, "n_heads d_head d_model -> (n_heads d_head) d_model")
                if dir.shape[-1] != O_flat.shape[0]:
                    print(f"Dimension mismatch: {dir.shape[-1]} != {O_flat.shape[0]}, projecting dimensions")
                    # Project to the correct dimension
                    d_model = model.cfg.d_model
                    projected_dir = torch.zeros(*dir.shape[:-1], d_model, device=dir.device)
                    min_dim = min(dir.shape[-1], d_model)
                    projected_dir[..., :min_dim] = dir[..., :min_dim]
                    return projected_dir
                return dir @ O_flat
        else:
            print(f"Unknown hook type '{hook_type}', using identity mapping")
            return dir
    
    # Replace the original function with our improved version
    model_fns.to_resid_dir = improved_to_resid_dir
    
    print("SAE visualization functions patched successfully.")

def main():
    # Set up argument parser for flexible configuration
    parser = argparse.ArgumentParser(description='Visualize VSAE ISO models')
    parser.add_argument('--sae_path', type=str, default='./trained_vsae_iso/trainer_0', 
                        help='Path to trained SAE')
    parser.add_argument('--output_file', type=str, default='./trained_vsae_iso/visualization.html',
                        help='Where to save visualization')
    parser.add_argument('--dataset', type=str, default='NeelNanda/c4-code-20k',
                        help='Dataset for visualization')
    parser.add_argument('--model_name', type=str, default='gelu-1l',
                        help='Model name')
    parser.add_argument('--hook_name', type=str, default='blocks.0.mlp.hook_post',
                        help='Hook name')
    parser.add_argument('--hook_layer', type=int, default=0,
                        help='Hook layer')
    parser.add_argument('--num_features', type=int, default=100,
                        help='Number of features to visualize')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for token sampling')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--simplified', action='store_true',
                        help='Use simplified visualization layout')
    parser.add_argument('--no_patch', action='store_true',
                        help='Disable patching of visualization functions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(exist_ok=True, parents=True)
    
    # Patch visualization functions to handle dimension mismatches
    if not args.no_patch:
        patch_visualization_functions()
    
    # 1. Load the model using transformer_lens
    print("Loading model...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    
    # 2. Set up data generator
    print("Setting up data pipeline...")
    data_gen = hf_dataset_to_generator(args.dataset, split="train")
    
    # 3. Create activation buffer to get tokenized data for visualization
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=args.hook_name,
        d_submodule=model.cfg.d_mlp,
        n_ctxs=100,  # Fewer contexts needed for visualization than training
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        device=args.device,
    )
    
    # 4. Get some tokens for visualization
    print("Getting sample tokens...")
    tokens = buffer.tokenized_batch(batch_size=args.batch_size)["input_ids"].to(args.device)
    
    # 5. Create layout based on simplified flag
    if args.simplified:
        layout = SaeVisLayoutConfig(
            columns=[
                Column(ActsHistogramConfig(), width=400),
            ],
            height=500,
        )
    else:
        layout = SaeVisLayoutConfig(
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
                    width=800
                ),
            ],
            height=750,
        )
    
    # 6. Create visualization
    print("Creating visualization...")
    try:
        SAEVisAdapter.create_visualization(
            sae_path=args.sae_path,
            tokens=tokens,
            output_file=args.output_file,
            model_name=args.model_name,
            hook_name=args.hook_name,
            hook_layer=args.hook_layer,
            feature_indices=range(args.num_features),
            layout=layout,
            device=args.device,
            verbose=True,
            simplified=args.simplified,
        )
        
        print(f"✅ Visualization saved successfully to {args.output_file}")
    except Exception as e:
        print(f"❌ Error during visualization creation: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a fallback with simplified layout
        if not args.simplified:
            try:
                print("\nAttempting simplified visualization instead...")
                fallback_output = args.output_file.replace(".html", "_simplified.html")
                layout = SaeVisLayoutConfig(
                    columns=[Column(ActsHistogramConfig(), width=400)],
                    height=500,
                )
                
                SAEVisAdapter.create_visualization(
                    sae_path=args.sae_path,
                    tokens=tokens,
                    output_file=fallback_output,
                    model_name=args.model_name,
                    hook_name=args.hook_name,
                    hook_layer=args.hook_layer,
                    feature_indices=range(min(25, args.num_features)),
                    layout=layout,
                    device=args.device,
                    verbose=True,
                    simplified=True,
                )
                
                print(f"✅ Simplified visualization saved to {fallback_output}")
            except Exception as e2:
                print(f"❌ Simplified visualization also failed: {e2}")

if __name__ == "__main__":
    main()