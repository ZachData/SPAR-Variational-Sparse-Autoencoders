"""
Utilities for working with SAE models, especially name extraction for organization.
"""

import re
from pathlib import Path
from typing import Optional


def extract_model_name_from_path(model_path: str) -> Optional[str]:
    """
    Extract the model/experiment name from a file path.
    
    Handles paths like:
    C:\...\experiments\VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var\trainer_0\ae.pt
    
    Returns: VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var
    """
    try:
        path = Path(model_path)
        
        # Look for experiment name in the path
        # Usually it's the parent or grandparent of the model file
        parts = path.parts
        
        # Common patterns for experiment directories
        experiment_patterns = [
            r'^VSAETopK_.*',
            r'^AutoEncoderTopK_.*', 
            r'^TopK_.*',
            r'^SAE_.*',
            r'.*_d\d+_k\d+.*',  # Contains dict_size and k parameters
        ]
        
        # Look through path parts for experiment-like names
        for part in reversed(parts):
            for pattern in experiment_patterns:
                if re.match(pattern, part):
                    return part
        
        # Fallback: if we find "trainer_X" directory, use its parent
        for i, part in enumerate(parts):
            if re.match(r'^trainer_\d+$', part) and i > 0:
                return parts[i-1]
        
        # Another fallback: look for the longest descriptive part
        descriptive_parts = []
        for part in parts:
            if len(part) > 10 and ('_' in part or '-' in part):
                descriptive_parts.append(part)
        
        if descriptive_parts:
            return descriptive_parts[-1]  # Take the last (most specific) one
        
        return None
        
    except Exception:
        return None


def extract_model_name_from_model(sae_model) -> str:
    """
    Extract a descriptive name from the SAE model object itself.
    
    Args:
        sae_model: The SAE model object
        
    Returns:
        Descriptive model name
    """
    # Try to get experiment name from model attributes
    if hasattr(sae_model, 'config'):
        config = sae_model.config
        if hasattr(config, 'experiment_name'):
            return config.experiment_name
        if hasattr(config, 'wandb_name'):
            return config.wandb_name
    
    # Try to get from trainer config if it exists
    if hasattr(sae_model, 'trainer_config'):
        trainer_config = sae_model.trainer_config
        if hasattr(trainer_config, 'wandb_name'):
            return trainer_config.wandb_name
    
    # Generate name from model properties
    model_type = type(sae_model).__name__
    
    try:
        dict_size = getattr(sae_model, 'dict_size', 'unknown')
        activation_dim = getattr(sae_model, 'activation_dim', 'unknown')
        
        # Get k value (handle tensor vs int)
        k = getattr(sae_model, 'k', 'unknown')
        if hasattr(k, 'item'):
            k = k.item()
        
        return f"{model_type}_d{dict_size}_k{k}_dim{activation_dim}"
        
    except Exception:
        return model_type


def get_organized_output_path(sae_model, output_filename: str, model_path: Optional[str] = None) -> str:
    """
    Create an organized output path for visualizations.
    
    Args:
        sae_model: The SAE model
        output_filename: The desired output filename (e.g., "visualization.html")
        model_path: Optional path where the model was loaded from
        
    Returns:
        Organized path like: visualizations/ModelName/output_filename
    """
    # Try to extract name from model path first
    model_name = None
    if model_path:
        model_name = extract_model_name_from_path(model_path)
    
    # Fallback to extracting from model object
    if not model_name:
        model_name = extract_model_name_from_model(sae_model)
    
    # Create organized directory structure
    base_dir = Path("visualizations")
    model_dir = base_dir / model_name
    
    # Create the directory
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Return the organized path
    organized_path = model_dir / output_filename
    
    return str(organized_path)


def clean_model_name(name: str) -> str:
    """
    Clean up a model name to be filesystem-friendly.
    
    Args:
        name: Raw model name
        
    Returns:
        Cleaned name safe for use as directory name
    """
    # Replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    # Limit length
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    
    return cleaned
