#!/usr/bin/env python3
"""
Test the new histogram features added to the visualization system.
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

def test_histogram_features():
    """Test the new histogram visualization features"""
    print("📊 Testing New Histogram Features!")
    print("=" * 50)
    
    try:
        # Import the components
        from sae_vis import create_visualization
        from dictionary_learning.trainers.vsae_topk import VSAETopK
        
        # Load your residual stream model
        model_path = r"C:\Users\WeeSnaw\Desktop\spar2\experiments\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var\trainer_0\ae.pt"
        
        print(f"Loading models...")
        model = HookedTransformer.from_pretrained("gelu-1l", device="cuda" if torch.cuda.is_available() else "cpu")
        sae_model = VSAETopK.from_pretrained(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"✅ Models loaded")
        print(f"   VSAETopK: {sae_model.dict_size} features, activation_dim = {sae_model.activation_dim}")
        
        # Test with some diverse text that should activate features
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "HELLO WORLD this has CAPITAL LETTERS and numbers 123456",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "import torch\nimport numpy as np\nfrom transformer_lens import HookedTransformer",
            "ABC123 XYZ789 uppercase and numbers everywhere",
            "Machine learning models require large amounts of training data.",
            "base64 encoding: eWVnZnJnZjE3dmdEf",  # This might activate the base64 feature!
            "HTTP headers: Content-Type: application/json; Authorization: Bearer abc123",
            "The weather today is sunny and warm with temperatures of 25°C.",
            "for i in range(10):\n    print(f'Hello {i}')\n    time.sleep(1)"
        ]
        
        # Test with Feature 0 (which we know activates) plus some others
        feature_indices = [0, 1, 5, 10, 25]  # Mix of features
        
        print(f"\n🎯 Testing histogram generation...")
        print(f"Features to analyze: {feature_indices}")
        print(f"Text samples: {len(texts)}")
        
        # Run the analysis with histogram generation
        results = create_visualization(
            sae_model=sae_model,
            transformer_model=model,
            tokenizer=model.tokenizer,
            texts=texts,
            feature_indices=feature_indices,
            output_path="histogram_test_visualization.html",
            model_path=model_path,
            max_examples=15,
            batch_size=4,
            max_seq_len=256
        )
        
        print(f"\n🎉 SUCCESS! Histogram visualization created!")
        
        # Test the histogram data
        print(f"\n📊 Histogram Analysis:")
        for feat_idx, analysis in results.items():
            print(f"   Feature {feat_idx}:")
            print(f"     Sparsity: {analysis.sparsity:.3f}")
            print(f"     Examples: {len(analysis.top_examples)}")
            
            # Check histogram data
            act_hist = analysis.activation_histogram
            logit_hist = analysis.logit_histogram
            
            print(f"     Activation histogram: {len(act_hist.bar_heights)} bins, title: '{act_hist.title}'")
            print(f"     Logit histogram: {len(logit_hist.bar_heights)} bins, title: '{logit_hist.title}'")
            
            if len(act_hist.bar_heights) > 0:
                max_bin = max(act_hist.bar_heights)
                print(f"     Max activation bin count: {max_bin}")
            
            if len(logit_hist.bar_heights) > 0:
                max_logit_bin = max(logit_hist.bar_heights)
                print(f"     Max logit bin count: {max_logit_bin}")
        
        print(f"\n🌐 Open the visualization to see the new features:")
        print(f"   📊 Activation histograms showing distribution of feature activations")
        print(f"   📈 Logit effect histograms showing positive/negative token effects")
        print(f"   🎨 Interactive Plotly charts with hover tooltips")
        print(f"   📱 Responsive design that works on mobile")
        
        # Show the file location
        organized_path = f"visualizations/VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var/histogram_test_visualization.html"
        print(f"\n📁 File saved to: {organized_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Testing Enhanced SAE Visualization with Histograms!")
    print("This adds beautiful interactive charts to your feature analysis!")
    
    success = test_histogram_features()
    
    if success:
        print(f"\n" + "🎉" * 20)
        print("AMAZING! Your visualization now has beautiful histograms!")
        print("\nWhat's new:")
        print("✅ Interactive activation histograms (showing feature activation distribution)")
        print("✅ Logit effect histograms (showing positive/negative token effects)")
        print("✅ Clean, professional layout inspired by the Anthropic SAE visualization")
        print("✅ Responsive design that works on all devices")
        print("✅ Organized file structure")
        
        print(f"\n🎯 Next phase ideas:")
        print("1. Add neuron correlation tables")
        print("2. Add feature-to-feature correlation analysis")
        print("3. Add auto-interpretation (what does this feature detect?)")
        print("4. Add t-SNE visualization for feature space exploration")
        
    else:
        print(f"\n⚠️  Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()
