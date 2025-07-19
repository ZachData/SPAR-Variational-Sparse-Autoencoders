#!/usr/bin/env python3
"""
Example usage for the fast SAE visualization system.
Run this from within the spar/sae_vis/ directory.

Usage:
    cd spar/sae_vis/
    python example_usage.py
"""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# Import our new components (relative imports for use within sae_vis package)
from .sae_interface import UnifiedSAEInterface
from .feature_analyzer import FeatureAnalyzer
from .html_generator import HTMLGenerator

# Example for your TopK model (adjust imports as needed)
from ..trainers.top_k import AutoEncoderTopK, TopKConfig


def create_visualization_example():
    """
    Complete example showing how to create a visualization.
    """
    
    # 1. Load your models
    print("Loading models...")
    
    # Load the transformer model (adjust model name as needed)
    model = HookedTransformer.from_pretrained("gelu-1l", device="cuda")
    tokenizer = model.tokenizer
    
    # Load your trained SAE model (adjust path and loading method)
    # Option A: Load AutoEncoderTopK
    sae_model = AutoEncoderTopK.from_pretrained(
        "path/to/your/trained/sae.pt",
        device="cuda"
    )
    
    # Option B: If you have a VSAETopK model, load it like this:
    # from dictionary_learning.trainers.vsae_topk import VSAETopK
    # sae_model = VSAETopK.from_pretrained("path/to/your/vsae.pt", device="cuda")
    
    print(f"Loaded SAE with {sae_model.dict_size} features")
    
    # 2. Create unified interface
    sae_interface = UnifiedSAEInterface(sae_model)
    print(f"SAE Interface Info: {sae_interface.get_info()}")
    
    # 3. Create feature analyzer
    analyzer = FeatureAnalyzer(sae_interface, model, tokenizer)
    
    # 4. Prepare some text data for analysis
    # You can use any text corpus - this is just an example
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language.",
        "Machine learning models can learn complex patterns.",
        "The weather today is sunny and warm.",
        "Deep neural networks have many layers.",
        "Natural language processing is fascinating.",
        "The cat sat on the mat.",
        "Artificial intelligence is transforming technology.",
        "def hello_world(): print('Hello, World!')",
        "The function returns the sum of two numbers.",
    ]
    
    # 5. Choose which features to analyze (start small for testing)
    feature_indices = [0, 1, 2, 3, 4]  # Analyze first 5 features
    
    # 6. Run the analysis
    print("Running feature analysis...")
    results = analyzer.analyze_features(
        texts=texts,
        feature_indices=feature_indices,
        max_examples=10,  # Top 10 examples per feature
        batch_size=4     # Process 4 texts at a time
    )
    
    # 7. Generate HTML visualization
    print("Generating HTML visualization...")
    html_gen = HTMLGenerator()
    html_gen.create_visualization(
        feature_analyses=results,
        output_path="sae_visualization.html",
        title="My SAE Feature Analysis"
    )
    
    print("✅ Visualization saved to 'sae_visualization.html'")
    print("Open the file in your browser to explore the features!")


def analyze_more_features_example():
    """
    Example for analyzing a larger number of features with a bigger dataset.
    """
    print("Loading models for large-scale analysis...")
    
    # Load models (same as above)
    model = HookedTransformer.from_pretrained("gelu-1l", device="cuda")
    sae_model = AutoEncoderTopK.from_pretrained("path/to/your/sae.pt", device="cuda")
    
    sae_interface = UnifiedSAEInterface(sae_model)
    analyzer = FeatureAnalyzer(sae_interface, model, model.tokenizer)
    
    # Use a larger text corpus
    # You could load from file, use datasets library, etc.
    texts = []
    with open("your_text_corpus.txt", "r") as f:
        texts = [line.strip() for line in f if line.strip()][:1000]  # First 1000 lines
    
    # Analyze more features
    feature_indices = list(range(50))  # First 50 features
    
    # Run analysis with larger batches for efficiency
    results = analyzer.analyze_features(
        texts=texts,
        feature_indices=feature_indices,
        max_examples=20,
        batch_size=16
    )
    
    # Generate visualization
    html_gen = HTMLGenerator()
    html_gen.create_visualization(
        feature_analyses=results,
        output_path="large_sae_analysis.html",
        title="Large-Scale SAE Analysis"
    )
    
    print("✅ Large-scale analysis complete!")


def debug_sae_interface():
    """
    Utility function to debug your SAE model interface.
    Use this if you're having trouble with model loading.
    """
    # Load your SAE model
    sae_model = AutoEncoderTopK.from_pretrained("path/to/your/sae.pt")
    
    # Create interface and print debug info
    interface = UnifiedSAEInterface(sae_model)
    info = interface.get_info()
    
    print("SAE Model Debug Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test encoding
    test_input = torch.randn(2, 10, interface.activation_dim)
    try:
        output = interface.encode(test_input)
        print(f"\n✅ Encoding test passed!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output sparse features shape: {output.sparse_features.shape}")
        print(f"  Output top values shape: {output.top_values.shape}")
        print(f"  Output top indices shape: {output.top_indices.shape}")
        print(f"  K value: {output.k}")
        
    except Exception as e:
        print(f"\n❌ Encoding test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the basic example
    create_visualization_example()
    
    # Uncomment to test interface debugging
    # debug_sae_interface()
    
    # Uncomment to run large-scale analysis
    # analyze_more_features_example()
