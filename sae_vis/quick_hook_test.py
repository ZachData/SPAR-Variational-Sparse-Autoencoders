#!/usr/bin/env python3
"""
Quick test to verify the hook name fix works.
"""

import sys
import os
from pathlib import Path
import torch

# Add the parent directory to the path
current_dir = Path(__file__).parent
if 'sae_vis' in str(current_dir):
    spar_dir = current_dir.parent
else:
    spar_dir = current_dir

sys.path.insert(0, str(spar_dir))

from transformer_lens import HookedTransformer

def verify_hook_fix():
    """Verify that blocks.0.mlp.hook_post gives us the right dimensions"""
    print("🔧 Verifying Hook Fix")
    print("=" * 40)
    
    try:
        from dictionary_learning.trainers.vsae_topk import VSAETopK
        
        # Load models
        print("Loading gelu-1l...")
        model = HookedTransformer.from_pretrained("gelu-1l", device="cpu")
        print(f"✅ gelu-1l loaded")
        print(f"   d_model: {model.cfg.d_model}")
        print(f"   d_mlp: {model.cfg.d_mlp}")
        
        model_path = r"C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var\trainer_0\ae.pt"
        print(f"\nLoading VSAETopK...")
        sae_model = VSAETopK.from_pretrained(model_path, device="cpu")
        print(f"✅ VSAETopK loaded")
        print(f"   Expected activation_dim: {sae_model.activation_dim}")
        
        # Test with different hooks
        test_text = "The quick brown fox jumps over the lazy dog."
        tokens = model.tokenizer.encode(test_text, return_tensors="pt")
        print(f"\n🧪 Testing hooks with text: '{test_text}'")
        
        _, cache = model.run_with_cache(tokens)
        
        hooks_to_test = [
            'blocks.0.mlp.hook_post',
            'blocks.0.hook_resid_post', 
            'blocks.0.mlp.hook_pre',
            'blocks.0.hook_resid_pre'
        ]
        
        for hook_name in hooks_to_test:
            if hook_name in cache:
                activation = cache[hook_name]
                print(f"   {hook_name}: {activation.shape}")
                
                if activation.shape[-1] == sae_model.activation_dim:
                    print(f"     ✅ MATCH! This gives us {sae_model.activation_dim}D")
                    
                    # Test if this works with the SAE
                    print(f"     🧪 Testing with VSAETopK...")
                    with torch.no_grad():
                        # Flatten for SAE
                        flat_activation = activation.view(-1, activation.shape[-1])
                        
                        try:
                            result = sae_model.encode(flat_activation, return_topk=True, training=False)
                            print(f"     ✅ VSAETopK encoding successful!")
                            print(f"     Sparse features shape: {result[0].shape}")
                            print(f"     Top values shape: {result[5].shape}")
                            
                            print(f"\n🎉 SOLUTION CONFIRMED!")
                            print(f"   Use hook: {hook_name}")
                            print(f"   Dimension: {activation.shape[-1]}")
                            
                            return hook_name
                            
                        except Exception as e:
                            print(f"     ❌ VSAETopK failed: {e}")
                else:
                    print(f"     ❌ Wrong dimension: {activation.shape[-1]} != {sae_model.activation_dim}")
            else:
                print(f"   {hook_name}: ❌ Not found")
        
        print(f"\n❌ No working hook found")
        return None
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    hook_name = verify_hook_fix()
    
    if hook_name:
        print(f"\n✅ Ready to test the full visualization!")
        print(f"The feature analyzer should now use {hook_name} and work correctly.")
        print(f"\nRun: python test_real_sae.py")
    else:
        print(f"\n⚠️  Still have issues. Need to investigate further.")

if __name__ == "__main__":
    main()
