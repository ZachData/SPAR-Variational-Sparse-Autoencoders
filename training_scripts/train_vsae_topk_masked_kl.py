"""
Enhanced training script for VSAETopK with sweep functionality and optimized configurations.

Key improvements:
# - Hyperparameter sweep support using BaseSweepRunner
- Multiple configuration presets including 10GB GPU optimized settings
- Enhanced evaluation and logging
- Better memory management and error handling
- Comprehensive argument parsing
- FIXED: All configurations now use residual stream (blocks.0.hook_resid_post)
"""

import torch
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import multiprocessing

from transformer_lens import HookedTransformer
# from dictionary_learning.base_sweep import BaseSweepRunner
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

from dictionary_learning.trainers.vsae_topk_masked_kl import (
    VSAETopKTrainer,
    VSAETopKConfig,
    VSAETopKTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment with enhanced TopK support."""
    # Model configuration
    model_name: str = "EleutherAI/pythia-70m-deduped"  # Changed from "gelu-1l"
    layer: int = 3  # Changed from 0 - typical layers for pythia-70m are 3,4
    hook_name: str = "blocks.3.hook_resid_post"  # Updated to match layer
    dict_size_multiple: float = 4.0
    k_fraction: float = 0.08  # Fraction of dictionary size for Top-K
    
    # Model-specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    threshold_beta: float = 0.999
    threshold_start_step: Optional[int] = None
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32  # TopK auxiliary loss coefficient
    
    # Schedule configuration
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start_step: Optional[int] = None
    
    # Buffer configuration
    n_ctxs: int = 3000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024
    
    # Logging and saving
    checkpoint_steps: tuple = (5000, 10000)
    log_steps: int = 100
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "vsae_topk_masked"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42
    
    # Evaluation configuration
    eval_batch_size: int = 64
    eval_n_batches: int = 10
    
    def __post_init__(self):
        """Set derived configuration values."""
        # Set default step values based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.total_steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.decay_start_step is None:
            decay_start = int(0.8 * self.total_steps)
            min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
            self.decay_start_step = max(decay_start, min_decay_start)
        if self.threshold_start_step is None:
            self.threshold_start_step = int(0.1 * self.total_steps)
    
    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.dtype]
    
    def get_autocast_dtype(self) -> torch.dtype:
        """Convert string autocast dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.autocast_dtype]
    
    def get_device(self) -> torch.device:
        """Convert string device to torch device."""
        return torch.device(self.device)


class ExperimentRunner:
    """Manages the entire training experiment with enhanced VSAETopK support."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_logging()
        self.setup_reproducibility()
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'vsae_topk_masked_kl_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_reproducibility(self) -> None:
        """Set up reproducible training."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info(f"Set random seed to {self.config.seed}")
            
    def load_model(self) -> HookedTransformer:
        """Load and configure the transformer model."""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        model = HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.config.device
        )
        
        self.logger.info(f"Model loaded. d_model: {model.cfg.d_model}")
        return model
        
    def create_buffer(self, model: HookedTransformer) -> TransformerLensActivationBuffer:
        """Create activation buffer for training data."""
        self.logger.info("Setting up data generator and activation buffer")
        
        # Set up data generator
        data_gen = hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b",
            split="train",
            return_tokens=True
        )
        
        # Create activation buffer - FIXED: Uses d_model for residual stream
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=self.config.hook_name,
            d_submodule=model.cfg.d_model,  # FIXED: d_model (512) not d_mlp (2048) for residual stream
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> VSAETopKConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_model)
        k = int(self.config.k_fraction * dict_size)
        
        # Create base config - FIXED: Uses d_model for residual stream
        model_config = VSAETopKConfig(
            activation_dim=model.cfg.d_model,  # FIXED: d_model (512) for residual stream
            dict_size=dict_size,
            k=k,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
        # Add threshold parameters if they exist in the config class
        if hasattr(model_config, 'threshold_beta'):
            model_config.threshold_beta = self.config.threshold_beta
        
        return model_config
        
    def create_training_config(self) -> VSAETopKTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAETopKTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            auxk_alpha=self.config.auxk_alpha,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
            # Note: threshold_beta and threshold_start_step may not be in VSAETopKTrainingConfig
            # These might be model-level parameters instead
        )
        
    def create_trainer_config(self, model_config: VSAETopKConfig, training_config: VSAETopKTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        trainer_config = {
            "trainer": VSAETopKTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "wandb_name": self.get_experiment_name(),
            "submodule_name": self.config.hook_name,
            # "seed": self.config.seed,
        }
        
        # # Add threshold parameters if the trainer accepts them
        # if hasattr(VSAETopKTrainer, '__init__'):
        #     # Add threshold parameters as separate config items
        #     trainer_config.update({
        #         "threshold_beta": self.config.threshold_beta,
        #         "threshold_start_step": self.config.threshold_start_step,
        #     })
        
        return trainer_config

    def clean_model_name_for_path(self, model_name: str) -> str:
        """Clean model name for use in file paths."""
        # Remove organization prefix (everything before and including the last '/')
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        
        # Remove or replace problematic characters
        model_name = model_name.replace('-deduped', '')
        model_name = model_name.replace('-', '')  # Remove hyphens if desired
        
        return model_name

    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        # Use cleaned model name for path safety
        clean_model_name = self.clean_model_name_for_path(self.config.model_name)
        
        k_value = int(self.config.k_fraction * self.config.dict_size_multiple * 512)  # Assuming d_mlp=2048 for gelu-1l
        # var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"
        
        return (
            f"MaskedVSAETopK_{clean_model_name}_"
            f"d{int(self.config.dict_size_multiple * 512)}_"
            f"k{k_value}_lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"aux{self.config.auxk_alpha}"
        )
        
    def get_save_directory(self) -> Path:
        """Get the save directory for this experiment."""
        save_dir = Path(self.config.save_dir) / self.get_experiment_name()
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
        
    def save_config(self, save_dir: Path) -> None:
        """Save experiment configuration."""
        config_path = save_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            import json
            json.dump(asdict(self.config), f, indent=2, default=str)
        self.logger.info(f"Saved experiment config to {config_path}")
        
    def run_training(self) -> Optional[Dict[str, float]]:
        """Run the complete training experiment."""
        start_time = time.time()
        
        try:
            # Setup
            model = self.load_model()
            buffer = self.create_buffer(model)
            model_config = self.create_model_config(model)
            training_config = self.create_training_config()
            trainer_config = self.create_trainer_config(model_config, training_config)
            
            # Save directory and config
            save_dir = self.get_save_directory()
            self.save_config(save_dir)
            
            # Log experiment details
            self.logger.info(f"Experiment: {self.get_experiment_name()}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}")
            self.logger.info(f"K value: {model_config.k}")
            self.logger.info(f"K fraction: {self.config.k_fraction}, Auxiliary alpha: {self.config.auxk_alpha}")
            self.logger.info(f"Save directory: {save_dir}")
            
            # Run training
            self.logger.info("Starting VSAETopK training...")
            
            trainSAE(
                data=buffer,
                trainer_configs=[trainer_config],  # FIXED: trainSAE expects a list of configs
                steps=self.config.total_steps,
                save_dir=str(save_dir),
                save_steps=list(self.config.checkpoint_steps) if self.config.checkpoint_steps else None,
                log_steps=self.config.log_steps,
                verbose=True,
                normalize_activations=True,
                autocast_dtype=self.config.get_autocast_dtype(),
                use_wandb=self.config.use_wandb,
                wandb_entity=self.config.wandb_entity,
                wandb_project=self.config.wandb_project,
                run_cfg={
                    "model_type": self.config.model_name,
                    "experiment_type": "vsae_topk_masked_kl",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k_fraction": self.config.k_fraction,
                    "var_flag": self.config.var_flag,
                    "auxk_alpha": self.config.auxk_alpha,
                    **asdict(self.config)
                }
            )
            
            # Evaluate the trained model
            eval_results = self.evaluate_model(save_dir, buffer)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def evaluate_model(self, save_dir: Path, buffer: TransformerLensActivationBuffer) -> Dict[str, float]:
        """Run evaluation on the trained model."""
        self.logger.info("Evaluating trained model...")
        
        try:
            from dictionary_learning.utils import load_dictionary
            import json
            
            model_path = save_dir / "trainer_0"
            vsae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # CRITICAL: Scale biases back down since buffer normalizes activations
            # The saved model has biases scaled UP by norm_factor to work with unnormalized data
            # But our buffer divides by norm_factor, so we need to undo the bias scaling
            norm_factor = config.get('norm_factor', None)
            if norm_factor is not None and hasattr(vsae, 'scale_biases'):
                self.logger.info(f"Scaling biases by 1/{norm_factor:.4f} for evaluation with normalized activations")
                vsae.scale_biases(1.0 / norm_factor)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=vsae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add VSAETopK-specific diagnostics
            try:
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                # Normalize if needed
                if norm_factor is not None:
                    sample_batch = sample_batch / norm_factor
                    
                sample_batch = sample_batch.to(self.config.device)
                
                if hasattr(vsae, 'get_topk_analysis'):
                    topk_diagnostics = vsae.get_topk_analysis(sample_batch)
                    for key, value in topk_diagnostics.items():
                        eval_results[f"final_{key}"] = value if not torch.is_tensor(value) else value.item()
                
                if hasattr(vsae, 'get_kl_diagnostics'):
                    kl_diagnostics = vsae.get_kl_diagnostics(sample_batch)
                    for key, value in kl_diagnostics.items():
                        eval_results[f"final_{key}"] = value if not torch.is_tensor(value) else value.item()
                        
            except Exception as e:
                self.logger.warning(f"Could not compute TopK/KL diagnostics: {e}")
            
            # Log results
            self.logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    if not (math.isnan(value) or math.isinf(value)):
                        self.logger.info(f"  {metric}: {value:.4f}")
                    else:
                        self.logger.warning(f"  {metric}: {value} (invalid)")
                else:
                    self.logger.info(f"  {metric}: {value}")
            
            # Save evaluation results
            eval_path = save_dir / "evaluation_results.json"
            json_results = {}
            for k, v in eval_results.items():
                if torch.is_tensor(v):
                    json_results[k] = v.item()
                elif isinstance(v, (int, float)):
                    json_results[k] = float(v)
                else:
                    json_results[k] = str(v)
            
            with open(eval_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Saved evaluation results to {eval_path}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

def create_quick_test_config() -> ExperimentConfig:
    """Create a configuration for quick testing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",  # FIXED: Residual stream
        dict_size_multiple=4.0,
        k_fraction=0.08,
        
        # Test parameters
        total_steps=1000,
        checkpoint_steps=list(),
        log_steps=50,
        
        # Small buffer for testing
        n_ctxs=500,
        refresh_batch_size=16,
        out_batch_size=128,
        
        # Evaluation
        eval_batch_size=32,
        eval_n_batches=3,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )


def create_full_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory with consistent speed."""
    return ExperimentConfig(
        model_name = "EleutherAI/pythia-70m-deduped",  # Changed from "gelu-1l"
        layer = 3,  # Changed from 0 - typical layers for pythia-70m are 3,4
        hook_name = "blocks.3.hook_resid_post",  # Updated to match layer
        dict_size_multiple=16.0,
        k_fraction=0.0625, #0.0078125, 0.015625, 0.03125, 0.0625
        
        # Training parameters optimized for 10GB GPU
        total_steps=10000,
        lr=8e-4,
        kl_coeff=1.0,
        auxk_alpha=1/32,
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
        use_april_update_mode=True,
        threshold_beta=0.999,
        
        # OPTIMIZED buffer settings for consistent speed
        n_ctxs=2500,     # Reduced to prevent memory pressure
        ctx_len=128,
        refresh_batch_size=12,  # Smaller batches for consistent memory usage
        out_batch_size=192,     # Smaller output batches
        
        # Checkpointing
        checkpoint_steps=(10000,),
        log_steps=1000,  # More frequent logging to monitor performance
        
        # Evaluation - smaller to reduce memory spikes
        eval_batch_size=24,
        eval_n_batches=4,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )

def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAETopK Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
        ], 
        default="full",
        help="Configuration preset for single training runs"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="zachdata",
        help="WandB entity/username for sweep logging"
    )
    args = parser.parse_args()
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
    }
    config = config_functions[args.config]()
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<25} | Value")
        print("-" * 35)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        # Show TopK-specific diagnostics if available
        topk_metrics = [k for k in results.keys() if "topk" in k.lower() or "aux" in k.lower()]
        if topk_metrics:
            print("\nTopK Diagnostics:")
            for metric in topk_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
    else:
        print("No evaluation results available")


# Usage examples:
# python train_vsae_topk_masked_kl.py --config quick_test
# python train_vsae_topk_masked_kl.py   


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()