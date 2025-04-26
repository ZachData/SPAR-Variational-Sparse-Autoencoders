"""
Visualization script for Standard SAE models trained on GELU-1L.
"""
import torch as t
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from sae_vis.visualizer import SAEVisAdapter

def main():
    # Configuration parameters
    sae_path = "./trained_sae/trainer_0"  # Path to trained SAE
    output_file = "./trained_sae/visualization.html"  # Where to save visualization
    dataset = "NeelNanda/c4-code-20k"  # Dataset for visualization
    model_name = "gelu-1l"  # Model name
    hook_name = "blocks.0.mlp.hook_post"  # Hook name
    hook_layer = 0  # Hook layer
    batch_size = 64  # Batch size for token sampling
    device = "cuda"  # Device to run on
    
    # Advanced options
    simplified = True  # Start with simplified layout
    feature_limit = 100  # Limit number of features for initial testing
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    try:
        # 1. Load the model using transformer_lens
        print("Loading model...")
        model = HookedTransformer.from_pretrained(model_name, device=device)
        
        # 2. Set up data generator
        print("Setting up data pipeline...")
        data_gen = hf_dataset_to_generator(dataset, split="train")
        
        # 3. Create activation buffer to get tokenized data for visualization
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=hook_name,
            d_submodule=model.cfg.d_mlp,
            n_ctxs=100,  # Fewer contexts needed for visualization than training
            ctx_len=128,
            refresh_batch_size=32,
            out_batch_size=1024,
            device=device,
        )
        
        # 4. Get some tokens for visualization
        print("Getting sample tokens...")
        tokens = buffer.tokenized_batch(batch_size=batch_size)["input_ids"].to(device)
        
        # 5. Create visualization
        print("Creating visualization...")
        SAEVisAdapter.create_visualization(
            sae_path=sae_path,
            tokens=tokens,
            output_file=output_file,
            model_name=model_name,
            hook_name=hook_name,
            hook_layer=hook_layer,
            feature_indices=range(feature_limit),  # Start with limited features
            device=device,
            verbose=True,
            simplified=simplified,
        )
        
        print(f"Done! Visualization saved to {output_file}")
        
        # Optionally, try a full visualization after the simplified one works
        if simplified:
            try:
                full_output_file = output_file.replace(".html", "_full.html")
                print(f"\nAttempting full visualization to {full_output_file}...")
                SAEVisAdapter.create_visualization(
                    sae_path=sae_path,
                    tokens=tokens,
                    output_file=full_output_file,
                    model_name=model_name,
                    hook_name=hook_name,
                    hook_layer=hook_layer,
                    feature_indices=range(feature_limit),
                    device=device,
                    verbose=True,
                    simplified=False,  # Full visualization
                )
                print(f"Full visualization saved to {full_output_file}")
            except Exception as e:
                print(f"Full visualization failed: {e}")
                print("The simplified visualization is still available.")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()