"""
Fast SAE Visualization Package

A clean, efficient implementation for visualizing Sparse Autoencoder features.
Designed specifically for TopK SAE models (VSAETopK, AutoEncoderTopK).
"""

from .sae_interface import UnifiedSAEInterface, SAEOutput
from .feature_analyzer import FeatureAnalyzer, FeatureAnalysis, ActivationExample
from .html_generator import HTMLGenerator
from .model_utils import get_organized_output_path, extract_model_name_from_path

__version__ = "0.1.0"
__all__ = [
    "UnifiedSAEInterface",
    "SAEOutput", 
    "FeatureAnalyzer",
    "FeatureAnalysis",
    "ActivationExample",
    "HTMLGenerator",
    "create_visualization",
    "get_organized_output_path",
    "extract_model_name_from_path",
]

def create_visualization(sae_model, transformer_model, tokenizer, texts, feature_indices, output_path, model_path=None, **kwargs):
    """
    Convenience function to create a complete visualization in one call.
    
    Args:
        sae_model: Your trained SAE (VSAETopK, AutoEncoderTopK, etc.)
        transformer_model: The transformer model (HookedTransformer or similar)
        tokenizer: Tokenizer for the transformer model
        texts: List of text strings to analyze
        feature_indices: List of feature indices to visualize
        output_path: Path to save the HTML visualization (can be just filename)
        model_path: Optional path where the model was loaded from (for better naming)
        **kwargs: Additional arguments passed to analyze_features()
        
    Returns:
        Dictionary of feature analyses
    """
    # Create the pipeline
    sae_interface = UnifiedSAEInterface(sae_model)
    analyzer = FeatureAnalyzer(sae_interface, transformer_model, tokenizer)
    
    # Set default kwargs
    analysis_kwargs = {
        'max_examples': 20,
        'max_seq_len': 512,
        'batch_size': 8
    }
    analysis_kwargs.update(kwargs)
    
    # Run analysis
    results = analyzer.analyze_features(texts, feature_indices, **analysis_kwargs)
    
    # Create organized output path
    from pathlib import Path
    output_filename = Path(output_path).name
    organized_output_path = get_organized_output_path(sae_model, output_filename, model_path)
    
    print(f"📁 Saving visualization to: {organized_output_path}")
    
    # Generate visualization
    html_gen = HTMLGenerator()
    html_gen.create_visualization(results, organized_output_path)
    
    return results