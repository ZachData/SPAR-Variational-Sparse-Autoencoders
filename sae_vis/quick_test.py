#!/usr/bin/env python3
"""
Quick test script for the SAE visualization system.
This can be run from anywhere in your spar directory.

Usage from spar/:
    python sae_vis/quick_test.py

Usage from spar/sae_vis/:
    python quick_test.py
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the trainers
current_dir = Path(__file__).parent
if 'sae_vis' in str(current_dir):
    # We're in sae_vis directory, need to go up to spar
    spar_dir = current_dir.parent
else:
    # We're in spar directory
    spar_dir = current_dir

sys.path.insert(0, str(spar_dir))

import torch
from transformer_lens import HookedTransformer

def test_sae_interface():
    """Test the SAE interface with a dummy model"""
    print("🧪 Testing SAE Interface...")
    
    try:
        # Import the sae_vis components
        from sae_vis.sae_interface import UnifiedSAEInterface
        from sae_vis.feature_analyzer import FeatureAnalyzer  
        from sae_vis.html_generator import HTMLGenerator
        from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKConfig
        
        print("✅ All imports successful!")
        
        # Create a dummy SAE for testing
        config = TopKConfig(
            activation_dim=512,
            dict_size=2048,
            k=64,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        dummy_sae = AutoEncoderTopK(config)
        print(f"✅ Created dummy SAE: {dummy_sae.dict_size} features, k={dummy_sae.k}")
        
        # Test the interface
        interface = UnifiedSAEInterface(dummy_sae)
        info = interface.get_info()
        print(f"✅ Interface info: {info}")
        
        # Test encoding
        test_input = torch.randn(2, 10, config.activation_dim)
        output = interface.encode(test_input)
        print(f"✅ Encoding test passed!")
        print(f"   Input: {test_input.shape}")
        print(f"   Sparse features: {output.sparse_features.shape}")
        print(f"   Top values: {output.top_values.shape}")
        print(f"   Top indices: {output.top_indices.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_model():
    """Test with a real transformer model (requires internet)"""
    print("\n🤖 Testing with real transformer model...")
    
    try:
        from sae_vis import create_visualization
        from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKConfig
        
        # Load a small transformer model
        print("Loading gelu-1l model...")
        model = HookedTransformer.from_pretrained("gelu-1l", device="cpu")
        
        # Create a dummy SAE that matches the model's dimensions
        # gelu-1l has d_model = 512
        config = TopKConfig(
            activation_dim=512,  # matches gelu-1l's d_model
            dict_size=2048,
            k=64,
            dtype=torch.float32,
            device=torch.device('cpu')
        )
        
        dummy_sae = AutoEncoderTopK(config)
        
        # Test data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a powerful programming language.",
            "Hello world, this is a test sentence.",
        ]
        
        feature_indices = [0, 1, 2]  # Just test 3 features
        
        print("Running analysis...")
        results = create_visualization(
            sae_model=dummy_sae,
            transformer_model=model,
            tokenizer=model.tokenizer,
            texts=texts,
            feature_indices=feature_indices,
            output_path="test_visualization.html",
            max_examples=5,
            batch_size=2
        )
        
        print(f"✅ Analysis complete! Generated visualization for {len(results)} features")
        print("   Check 'test_visualization.html' in your browser")
        
        return True
        
    except Exception as e:
        print(f"❌ Real model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Fast SAE Visualization - Quick Test")
    print("=" * 50)
    
    # Test 1: Basic interface
    success1 = test_sae_interface()
    
    if success1:
        # Test 2: Real model (optional)
        print("\nWould you like to test with a real model? (requires downloading gelu-1l)")
        response = input("Test with real model? [y/N]: ").lower().strip()
        
        if response in ['y', 'yes']:
            success2 = test_with_real_model()
        else:
            success2 = True
            print("Skipping real model test.")
    else:
        success2 = False
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! The SAE visualization system is working.")
        print("\nNext steps:")
        print("1. Replace the SAE model with your trained model")
        print("2. Use your own text corpus")
        print("3. Analyze interesting features")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()