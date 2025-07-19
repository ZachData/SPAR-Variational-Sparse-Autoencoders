#!/usr/bin/env python3
"""
Simple test for the histogram functionality.
This just tests that the FeatureAnalysis objects have histogram attributes.
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

def test_histogram_attributes():
    """Test that we can create FeatureAnalysis objects with histogram attributes"""
    print("🧪 Testing Histogram Attributes...")
    
    try:
        # Import the updated components
        from sae_vis.feature_analyzer import FeatureAnalysis, ActivationExample, HistogramData
        
        print("✅ Successfully imported FeatureAnalysis with histogram support")
        
        # Create a dummy HistogramData object
        dummy_activation_hist = HistogramData(
            bin_edges=[0, 1, 2, 3],
            bar_heights=[5, 10, 3],
            title="Test Activation Histogram",
            x_label="Activation Value"
        )
        
        dummy_logit_hist = HistogramData(
            bin_edges=[-1, 0, 1],
            bar_heights=[2, 8],
            title="Test Logit Histogram", 
            x_label="Logit Effect"
        )
        
        print("✅ Created dummy histogram data objects")
        
        # Create a dummy FeatureAnalysis object
        dummy_analysis = FeatureAnalysis(
            feature_idx=42,
            top_examples=[],
            sparsity=0.95,
            mean_activation=1.23,
            max_activation=4.56,
            top_boosted_tokens=[("hello", 0.5), ("world", 0.3)],
            top_suppressed_tokens=[("bad", -0.4), ("awful", -0.2)],
            decoder_norm=2.34,
            activation_histogram=dummy_activation_hist,
            logit_histogram=dummy_logit_hist
        )
        
        print("✅ Created FeatureAnalysis object with histogram attributes")
        
        # Test that we can access the histogram attributes
        print(f"   Activation histogram title: {dummy_analysis.activation_histogram.title}")
        print(f"   Logit histogram title: {dummy_analysis.logit_histogram.title}")
        print(f"   Activation histogram bins: {len(dummy_analysis.activation_histogram.bar_heights)}")
        print(f"   Logit histogram bins: {len(dummy_analysis.logit_histogram.bar_heights)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_visualization():
    """Test creating a minimal visualization with dummy data"""
    print("\n🎨 Testing Minimal Visualization...")
    
    try:
        from sae_vis.html_generator import HTMLGenerator
        from sae_vis.feature_analyzer import FeatureAnalysis, HistogramData
        
        # Create minimal test data
        activation_hist = HistogramData(
            bin_edges=[0, 0.5, 1.0, 1.5, 2.0],
            bar_heights=[10, 25, 15, 5],
            title="Histogram of randomly sampled non-zero activations",
            x_label="Activation Value"
        )
        
        logit_hist = HistogramData(
            bin_edges=[-0.5, 0, 0.5],
            bar_heights=[3, 7],
            title="Top 10 negative and positive output logits",
            x_label="Logit Effect"
        )
        
        test_analysis = FeatureAnalysis(
            feature_idx=0,
            top_examples=[],
            sparsity=0.90,
            mean_activation=0.8,
            max_activation=2.1,
            top_boosted_tokens=[("the", 0.5), ("and", 0.3)],
            top_suppressed_tokens=[("not", -0.4), ("no", -0.2)],
            decoder_norm=1.5,
            activation_histogram=activation_hist,
            logit_histogram=logit_hist
        )
        
        # Create HTML
        html_gen = HTMLGenerator()
        feature_data = {0: test_analysis}
        
        html_gen.create_visualization(
            feature_analyses=feature_data,
            output_path="minimal_histogram_test.html",
            title="Minimal Histogram Test"
        )
        
        print("✅ Created minimal HTML visualization")
        print("   Check 'minimal_histogram_test.html' to see if histograms render")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Simple Histogram Test")
    print("Testing the basic histogram functionality step by step...")
    print("=" * 60)
    
    # Test 1: Basic attribute access
    success1 = test_histogram_attributes()
    
    if success1:
        # Test 2: HTML generation
        success2 = test_minimal_visualization()
    else:
        success2 = False
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 SUCCESS! Basic histogram functionality is working!")
        print("\nNext steps:")
        print("1. Run your original test_histograms.py script")
        print("2. Check that the generated HTML shows histograms")
        print("3. If that works, we can add more histogram features!")
    else:
        print("⚠️  Some tests failed. Let's fix the basic functionality first.")

if __name__ == "__main__":
    main()
