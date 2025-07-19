#!/usr/bin/env python3
"""
SAEBench Evaluation Script for TopK and VSAETopK Models

Uses SAEBench's custom SAE interface to evaluate dictionary_learning models.
This approach creates a wrapper that makes your models compatible with SAEBench.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
import torch

# Add SAEBench to path
SAEBENCH_PATH = Path(__file__).parent / "SAEBench-main"
sys.path.insert(0, str(SAEBENCH_PATH))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('saebench_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = [
    # TopK models
    "TopK_SAE_gelu-1l_d2048_k512_auxk0.03125_lr_auto",
    "TopK_SAE_gelu-1l_d2048_k256_auxk0.03125_lr_auto", 
    "TopK_SAE_gelu-1l_d2048_k128_auxk0.03125_lr_auto",
    "TopK_SAE_gelu-1l_d2048_k64_auxk0.03125_lr_auto",
    # VSAETopK models
    "VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var",
    "VSAETopK_gelu-1l_d2048_k256_lr0.0008_kl1.0_aux0.03125_fixed_var",
    "VSAETopK_gelu-1l_d2048_k128_lr0.0008_kl1.0_aux0.03125_fixed_var", 
    "VSAETopK_gelu-1l_d2048_k64_lr0.0008_kl1.0_aux0.03125_fixed_var"
]

# Configuration
BASE_MODEL = "gelu-1l"
HOOK_NAME = "blocks.0.hook_resid_post"  # or blocks.0.mlp.hook_post - adjust as needed
EXPERIMENTS_DIR = Path("./experiments")
RESULTS_DIR = Path("./saebench_results")


def setup_environment():
    """Setup the environment for SAEBench evaluation."""
    logger.info("Setting up SAEBench environment...")
    
    if not SAEBENCH_PATH.exists():
        logger.error(f"SAEBench not found at {SAEBENCH_PATH}")
        sys.exit(1)
    
    os.chdir(SAEBENCH_PATH)
    logger.info(f"Changed to SAEBench directory: {SAEBENCH_PATH}")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    logger.info(f"Results will be saved to: {RESULTS_DIR}")


def find_model_path(model_name: str) -> Path:
    """Find the path to a trained model."""
    model_path = EXPERIMENTS_DIR / model_name / "trainer_0"
    
    if model_path.exists() and (model_path / "ae.pt").exists():
        return model_path
    
    # Try relative path from SAEBench directory
    relative_path = Path("..") / EXPERIMENTS_DIR / model_name / "trainer_0"
    if relative_path.exists() and (relative_path / "ae.pt").exists():
        return relative_path
        
    logger.error(f"Model not found: {model_name}")
    return None


def create_sae_wrapper_class():
    """Create a wrapper class that makes dictionary_learning SAEs compatible with SAEBench."""
    
    wrapper_code = '''
import sys
from pathlib import Path
import torch
import json

# Add the parent directory to path to import dictionary_learning
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.trainers.vsae_topk import VSAETopK
    from dictionary_learning.trainers.top_k import AutoEncoderTopK
except ImportError as e:
    print(f"Error importing dictionary_learning: {e}")
    sys.exit(1)

class DictionaryLearningSAEWrapper:
    """
    Wrapper to make dictionary_learning SAEs compatible with SAEBench.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        
        # Load the model
        self.sae, self.config = load_dictionary(str(model_path), device=device)
        
        # Create config object for SAEBench
        self.cfg = self._create_config()
        
    def _create_config(self):
        """Create a config object that SAEBench expects."""
        
        # Get model info from config
        trainer_config = self.config["trainer"]
        
        class SAEConfig:
            def __init__(self):
                self.d_in = trainer_config["activation_dim"]
                self.d_sae = trainer_config["dict_size"] 
                self.hook_name = "blocks.0.hook_resid_post"  # Adjust as needed
                self.hook_layer = 0
                self.device = "cuda"
                
        return SAEConfig()
    
    def encode(self, x):
        """Encode activations to features."""
        if hasattr(self.sae, 'encode'):
            # For VSAETopK, get just the sparse features
            if hasattr(self.sae, 'encode') and 'VSAETopK' in str(type(self.sae)):
                sparse_features, _, _, _ = self.sae.encode(x)
                return sparse_features
            else:
                return self.sae.encode(x)
        else:
            raise NotImplementedError("SAE does not have encode method")
    
    def decode(self, f):
        """Decode features back to activations."""
        if hasattr(self.sae, 'decode'):
            return self.sae.decode(f)
        else:
            raise NotImplementedError("SAE does not have decode method")
    
    def forward(self, x, output_features=False):
        """Forward pass through the SAE."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        if output_features:
            return reconstruction, features
        else:
            return reconstruction

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda"):
        """Load a pretrained model."""
        return cls(model_path, device)

# Export the wrapper for SAEBench to use
SAE = DictionaryLearningSAEWrapper
'''
    
    # Write the wrapper to SAEBench directory
    wrapper_path = SAEBENCH_PATH / "dictionary_learning_wrapper.py"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    logger.info(f"Created SAE wrapper at: {wrapper_path}")
    return wrapper_path


def create_custom_evaluation_script():
    """Create a custom evaluation script using SAEBench's API."""
    
    script_content = f'''
import sys
import json
from pathlib import Path
import torch

# Import SAEBench modules
sys.path.append(str(Path.cwd()))

from dictionary_learning_wrapper import DictionaryLearningSAEWrapper
from sae_bench.evals.sparse_probing.eval_config import EvalConfig as SparseProbingConfig
from sae_bench.evals.sparse_probing.main import run_eval as run_sparse_probing

from sae_bench.evals.core.eval_config import EvalConfig as CoreConfig  
from sae_bench.evals.core.main import run_eval as run_core

from sae_bench.evals.absorption.eval_config import EvalConfig as AbsorptionConfig
from sae_bench.evals.absorption.main import run_eval as run_absorption

# Add other evaluations as needed

def evaluate_model(model_path: str, model_name: str, output_dir: str):
    """Evaluate a single model with SAEBench."""
    
    print(f"Evaluating model: {{model_name}}")
    print(f"Model path: {{model_path}}")
    
    try:
        # Load the SAE using our wrapper
        sae = DictionaryLearningSAEWrapper.from_pretrained(model_path)
        print(f"Loaded SAE with dimensions: {{sae.cfg.d_in}} -> {{sae.cfg.d_sae}}")
        
        results = {{}}
        
        # Run sparse probing evaluation
        try:
            print("Running sparse probing...")
            config = SparseProbingConfig()
            result = run_sparse_probing(sae, "{BASE_MODEL}", config)
            results["sparse_probing"] = result
            print("Sparse probing completed")
        except Exception as e:
            print(f"Sparse probing failed: {{e}}")
            results["sparse_probing"] = None
        
        # Run core evaluation  
        try:
            print("Running core evaluation...")
            config = CoreConfig()
            result = run_core(sae, "{BASE_MODEL}", config)
            results["core"] = result
            print("Core evaluation completed")
        except Exception as e:
            print(f"Core evaluation failed: {{e}}")
            results["core"] = None
        
        # Save results
        output_path = Path(output_dir) / f"{{model_name}}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {{output_path}}")
        return results
        
    except Exception as e:
        print(f"Error evaluating {{model_name}}: {{e}}")
        return None

def main():
    """Run evaluations on all models."""
    
    models = {MODELS}
    experiments_dir = Path("../experiments")
    results_dir = Path("./saebench_results")
    
    all_results = {{}}
    
    for model_name in models:
        model_path = experiments_dir / model_name / "trainer_0"
        
        if model_path.exists():
            results = evaluate_model(str(model_path), model_name, str(results_dir))
            all_results[model_name] = results
        else:
            print(f"Model not found: {{model_path}}")
            all_results[model_name] = None
    
    # Save summary
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Evaluation summary saved to: {{summary_path}}")

if __name__ == "__main__":
    main()
'''
    
    script_path = SAEBENCH_PATH / "run_custom_evaluations.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"Created custom evaluation script: {script_path}")
    return script_path


def run_evaluations():
    """Run the custom evaluation script."""
    
    try:
        # Create wrapper and evaluation script
        create_sae_wrapper_class()
        script_path = create_custom_evaluation_script()
        
        # Run the evaluation script
        logger.info("Running custom evaluation script...")
        
        import subprocess
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            cwd=SAEBENCH_PATH
        )
        
        if result.returncode == 0:
            logger.info("Evaluations completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
        else:
            logger.error("Evaluations failed")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running evaluations: {e}")
        return False


def main():
    """Main evaluation function."""
    logger.info("Starting SAEBench evaluation using custom SAE wrapper")
    logger.info(f"Models to evaluate: {len(MODELS)}")
    
    start_time = time.time()
    
    try:
        # Setup environment
        setup_environment()
        
        # Verify models exist
        valid_models = []
        for model_name in MODELS:
            model_path = find_model_path(model_name)
            if model_path:
                valid_models.append(model_name)
                logger.info(f"Found model: {model_name}")
            else:
                logger.warning(f"Model not found: {model_name}")
        
        if not valid_models:
            logger.error("No valid models found!")
            return
        
        logger.info(f"Will evaluate {len(valid_models)} valid models")
        
        # Run evaluations
        success = run_evaluations()
        
        elapsed_time = time.time() - start_time
        
        if success:
            logger.info(f"All evaluations completed successfully!")
        else:
            logger.error(f"Some evaluations failed - check logs")
            
        logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    finally:
        # Return to original directory
        os.chdir(Path(__file__).parent)


if __name__ == "__main__":
    main()