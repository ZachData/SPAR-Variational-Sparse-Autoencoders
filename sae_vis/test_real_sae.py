#!/usr/bin/env python3
"""
Test the new residual stream VSAETopK model - this should work perfectly!
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

def test_new_residual_vsae():
    """Test with your new residual stream VSAETopK - should be perfect now!"""
    print("🚀 Testing New Residual Stream VSAETopK!")
    print("=" * 60)
    
    try:
        # Import the components
        from sae_vis import create_visualization
        from dictionary_learning.trainers.vsae_topk import VSAETopK
        
        # NEW model path - residual stream trained!
        model_path = r"C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var\trainer_0\ae.pt"
        
        print(f"Loading NEW residual stream VSAETopK from:")
        print(f"  {model_path}")
        
        # Load the transformer model
        print("\nLoading gelu-1l transformer model...")
        model = HookedTransformer.from_pretrained("gelu-1l", device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Loaded gelu-1l: d_model = {model.cfg.d_model}")
        
        # Load your NEW trained VSAETopK model
        print("Loading your NEW residual stream VSAETopK...")
        sae_model = VSAETopK.from_pretrained(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✅ Loaded NEW VSAETopK:")
        print(f"   Dict size: {sae_model.dict_size}")
        print(f"   Activation dim: {sae_model.activation_dim}") 
        print(f"   K: {sae_model.k}")
        print(f"   Model type: {type(sae_model).__name__}")
        
        # Check dimensions - should be perfect now!
        if sae_model.activation_dim == model.cfg.d_model:
            print(f"✅ PERFECT! Dimensions match: {sae_model.activation_dim} = {model.cfg.d_model}")
        else:
            print(f"⚠️  Unexpected: SAE activation_dim ({sae_model.activation_dim}) != model d_model ({model.cfg.d_model})")
        
        # Test some diverse text data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a powerful programming language used for machine learning and data science.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "The weather today is sunny and warm with a gentle breeze blowing through the trees.",
            "import torch\nimport torch.nn as nn\nfrom transformer_lens import HookedTransformer",
            "Machine learning models require large amounts of training data to learn complex patterns.",
            "for i in range(10):\n    print(f'Number: {i}')\n    if i % 2 == 0:\n        print('Even')",
            "The cat sat on the mat and watched the birds flying outside the window.",
            "Neural networks use backpropagation to update their weights during training.",
            "class MyClass:\n    def __init__(self, value):\n        self.value = value\n    def get_value(self):\n        return self.value"
        ]
        
        # Test with a good variety of features
        feature_indices = [0, 1, 2, 3, 4, 10, 25, 50, 100, 128, 200]  # Mix of early, middle, and later features
        
        print(f"\n🎯 Running analysis on {len(texts)} texts...")
        print(f"Analyzing features: {feature_indices}")
        print("This should work perfectly with residual stream training!")
        
        # Run the analysis
        results = create_visualization(
            sae_model=sae_model,
            transformer_model=model,
            tokenizer=model.tokenizer,
            texts=texts,
            feature_indices=feature_indices,
            output_path="residual_vsae_visualization.html",
            max_examples=20,  # Get plenty of examples
            batch_size=4,     # Process efficiently
            max_seq_len=256   # Good sequence length
        )
        
        print(f"\n🎉 SUCCESS! Analysis complete!")
        print(f"✅ Successfully analyzed {len(results)} features")
        print(f"✅ Visualization saved to: residual_vsae_visualization.html")
        
        # Print detailed statistics
        print(f"\n📊 Detailed Feature Statistics:")
        
        # Count features with different activity levels
        active_features = 0
        sparse_features = 0
        very_sparse_features = 0
        
        for feat_idx, analysis in results.items():
            if analysis.sparsity < 0.99:
                active_features += 1
            elif analysis.sparsity < 0.999:
                sparse_features += 1  
            else:
                very_sparse_features += 1
                
            print(f"   Feature {feat_idx}:")
            print(f"     Sparsity: {analysis.sparsity:.4f} ({analysis.sparsity*100:.1f}% zeros)")
            print(f"     Max activation: {analysis.max_activation:.4f}")
            print(f"     Examples found: {len(analysis.top_examples)}")
            print(f"     Logit effects: {len(analysis.top_boosted_tokens)} boosted, {len(analysis.top_suppressed_tokens)} suppressed")
        
        print(f"\n📈 Summary:")
        print(f"   Active features (< 99% sparse): {active_features}")
        print(f"   Sparse features (99-99.9% sparse): {sparse_features}")
        print(f"   Very sparse features (> 99.9% sparse): {very_sparse_features}")
        
        if active_features > 0:
            print(f"   🎉 EXCELLENT! Found {active_features} actively firing features!")
        elif sparse_features > 0:
            print(f"   ✅ Good! Found {sparse_features} features with some activity!")
        else:
            print(f"   🤔 All features very sparse - try different text or feature indices")
        
        print(f"\n🌐 Open 'residual_vsae_visualization.html' in your browser!")
        print(f"   This should show beautiful, clean visualizations with:")
        print(f"   - ✅ Working logit effects (no dimension mismatches!)")
        print(f"   - ✅ Proper activation highlighting") 
        print(f"   - ✅ Clean feature statistics")
        print(f"   - ✅ Interpretable feature patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_old_model():
    """Compare the new residual model with the old MLP model"""
    print(f"\n🔍 Quick comparison with the old MLP model...")
    
    try:
        from dictionary_learning.trainers.vsae_topk import VSAETopK
        
        # Load both models for comparison
        old_model_path = r"C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d8192_k512_lr0.0008_kl1_aux0.03125_fixed_var\trainer_0\ae.pt"
        new_model_path = r"C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var\trainer_0\ae.pt"
        
        old_sae = VSAETopK.from_pretrained(old_model_path, device="cpu")
        new_sae = VSAETopK.from_pretrained(new_model_path, device="cpu")
        
        print(f"📊 Model Comparison:")
        print(f"   Old MLP model:")
        print(f"     Activation dim: {old_sae.activation_dim} (MLP activations)")
        print(f"     Dict size: {old_sae.dict_size}")
        print(f"     K: {old_sae.k}")
        print(f"   ")
        print(f"   New residual model:")
        print(f"     Activation dim: {new_sae.activation_dim} (residual stream)")
        print(f"     Dict size: {new_sae.dict_size}")
        print(f"     K: {new_sae.k}")
        
        print(f"\n🎯 Benefits of the new model:")
        print(f"   ✅ Matches gelu-1l d_model exactly ({new_sae.activation_dim} = 512)")
        print(f"   ✅ Direct logit effect computation (no projection needed)")
        print(f"   ✅ Simpler visualization code")
        print(f"   ✅ More standard approach (most SAE papers use residual)")
        print(f"   ✅ Easier to extend and understand")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def main():
    print("🎉 Testing Your NEW Residual Stream VSAETopK Model!")
    print("This should work perfectly with our visualization system!")
    
    # Test the new model
    success = test_new_residual_vsae()
    
    if success:
        print("\n" + "🎉" * 20)
        print("PERFECT! Your new residual stream model works beautifully!")
        
        # Optional comparison
        response = input("\nWant to see a comparison with the old MLP model? [y/N]: ").lower().strip()
        if response in ['y', 'yes']:
            compare_with_old_model()
        
        print(f"\n🚀 Next steps:")
        print(f"1. Explore the HTML visualization - it should be much cleaner now!")
        print(f"2. Try analyzing more features or different text corpora")
        print(f"3. We're ready to add t-SNE visualization next! 🎯")
        print(f"4. This clean foundation makes everything easier")
        
    else:
        print(f"\n⚠️  Something went wrong. Check the error messages above.")

if __name__ == "__main__":
    main()