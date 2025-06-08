"""
Training script for Matryoshka Batch Top-K SAE.

This script provides a clean interface for training hierarchical SAEs
with batch-level top-k selection and dead feature revival mechanisms.
"""

import torch
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import multiprocessing

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer, 
    MatryoshkaBatchTopKSAE, 
    MatryoshkaConfig, 
    MatryoshkaTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for Matryoshka Batch Top-K SAE experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # Matryoshka-specific configuration
    k: int = 64  # Number of features active per batch
    group_fractions: List[float] = None  # Will be set in __post_init__
    group_weights: Optional[List[float]] = None  # Equal weights by default
    auxk_alpha: float = 1/32  # Auxiliary loss coefficient
    threshold_beta: float = 0.999  # EMA coefficient for threshold
    threshold_start_step: int = 1000  # When to start threshold updates
    dead_feature_threshold: int = 10_000_000  # Steps before feature considered dead
    top_k_aux_fraction: float = 0.5  # Fraction of activation_dim for auxiliary k
    
    # Training configuration
    total_steps: int = 15000
    lr: Optional[float] = None  # Will be auto-calculated
    
    # Buffer configuration
    n_ctxs: int = 3000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024
    
    # Logging and saving
    checkpoint_steps: tuple = (5000, 10000, 15000)
    log_steps: int = 100
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "matryoshka-batch-topk-experiments"
    
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
        # Set default group fractions (3 groups with decreasing importance)
        if self.group_fractions is None:
            self.group_fractions = [0.5, 0.3, 0.2]  # 50%, 30%, 20%
    
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
    """Manages Matryoshka Batch Top-K SAE training experiments."""
    
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
                logging.FileHandler(log_dir / 'matryoshka_training.log'),
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
        
        self.logger.info(f"Model loaded. d_mlp: {model.cfg.d_mlp}")
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
        
        # Create activation buffer
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=self.config.hook_name,
            d_submodule=model.cfg.d_mlp,
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> MatryoshkaConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return MatryoshkaConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            k=self.config.k,
            group_fractions=self.config.group_fractions,
            group_weights=self.config.group_weights,
            auxk_alpha=self.config.auxk_alpha,
            threshold_beta=self.config.threshold_beta,
            threshold_start_step=self.config.threshold_start_step,
            dead_feature_threshold=self.config.dead_feature_threshold,
            top_k_aux_fraction=self.config.top_k_aux_fraction,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> MatryoshkaTrainingConfig:
        """Create training configuration from experiment config."""
        return MatryoshkaTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
        )
        
    def create_trainer_config(self, model_config: MatryoshkaConfig, training_config: MatryoshkaTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": MatryoshkaBatchTopKTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "wandb_name": self.get_experiment_name(),
            "submodule_name": self.config.hook_name,
            "seed": self.config.seed,
        }
        
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        groups_str = f"g{len(self.config.group_fractions)}"
        fractions_str = "_".join([f"{f:.1f}" for f in self.config.group_fractions])
        
        return (
            f"MatryoshkaBatchTopK_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"k{self.config.k}_{groups_str}_{fractions_str}_"
            f"aux{self.config.auxk_alpha:.3f}"
        )
        
    def get_save_directory(self) -> Path:
        """Get the save directory for this experiment."""
        save_dir = Path(self.config.save_dir) / self.get_experiment_name()
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
        
    def save_config(self, save_dir: Path) -> None:
        """Save experiment configuration."""
        config_path = save_dir / "experiment_config.json"
        
        import json
        with open(config_path, 'w') as f:
            # Convert config to dict, handling special types
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2, default=str)
            
        self.logger.info(f"Saved experiment config to {config_path}")
        
    def run_training(self) -> Dict[str, float]:
        """Run the complete training experiment."""
        self.logger.info("Starting Matryoshka Batch Top-K SAE training experiment")
        self.logger.info(f"Configuration: {self.config}")
        
        start_time = time.time()
        
        try:
            # Load model and create buffer
            model = self.load_model()
            buffer = self.create_buffer(model)
            
            # Create configurations
            model_config = self.create_model_config(model)
            training_config = self.create_training_config()
            trainer_config = self.create_trainer_config(model_config, training_config)
            
            # Set up save directory
            save_dir = self.get_save_directory()
            self.save_config(save_dir)
            
            self.logger.info(f"Model config: {model_config}")
            self.logger.info(f"Training config: {training_config}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}")
            self.logger.info(f"Group sizes: {model_config.group_sizes}")
            self.logger.info(f"Group fractions: {model_config.group_fractions}")
            self.logger.info(f"K (batch top-k): {model_config.k}")
            self.logger.info(f"Auxiliary loss coefficient: {model_config.auxk_alpha}")
            
            # Run training
            self.logger.info("Starting training...")
            trainSAE(
                data=buffer,
                trainer_configs=[trainer_config],
                steps=self.config.total_steps,
                save_dir=str(save_dir),
                save_steps=list(self.config.checkpoint_steps),
                log_steps=self.config.log_steps,
                verbose=True,
                normalize_activations=True,
                autocast_dtype=self.config.get_autocast_dtype(),
                use_wandb=self.config.use_wandb,
                wandb_entity=self.config.wandb_entity,
                wandb_project=self.config.wandb_project,
                run_cfg={
                    "model_type": self.config.model_name,
                    "experiment_type": "matryoshka_batch_topk",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k": self.config.k,
                    "group_fractions": self.config.group_fractions,
                    "auxk_alpha": self.config.auxk_alpha,
                    "hierarchical_structure": True,
                    "batch_level_topk": True,
                    "dead_feature_revival": True,
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
        """Evaluate the trained model with hierarchical analysis."""
        self.logger.info("Evaluating trained model...")
        
        try:
            # Load the trained model
            from dictionary_learning.utils import load_dictionary
            
            model_path = save_dir / "trainer_0"
            sae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run standard evaluation
            eval_results = evaluate(
                dictionary=sae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add Matryoshka-specific diagnostics
            try:
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                sample_batch = sample_batch.to(self.config.device)
                
                # Analyze group-wise activation patterns
                with torch.no_grad():
                    f, active_indices, post_relu_acts = sae.encode(
                        sample_batch, return_active=True, use_threshold=False
                    )
                    
                    # Group-wise statistics
                    f_chunks = torch.split(f, sae.config.group_sizes, dim=1)
                    
                    for i, f_chunk in enumerate(f_chunks):
                        group_l0 = (f_chunk != 0).float().sum(dim=-1).mean()
                        group_active_frac = (f_chunk.sum(0) > 0).float().mean()
                        
                        eval_results[f"group_{i}_l0"] = group_l0.item()
                        eval_results[f"group_{i}_active_fraction"] = group_active_frac.item()
                    
                    # Dead feature analysis
                    total_active = (f.sum(0) > 0).float().sum()
                    eval_results["total_active_features"] = total_active.item()
                    eval_results["dead_feature_fraction"] = 1.0 - (total_active / sae.dict_size).item()
                    
                    # Threshold information
                    if hasattr(sae, 'threshold') and sae.threshold >= 0:
                        eval_results["current_threshold"] = sae.threshold.item()
                    
            except Exception as e:
                self.logger.warning(f"Could not compute Matryoshka-specific diagnostics: {e}")
            
            # Log results
            self.logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                if not torch.isnan(torch.tensor(value)) and not torch.isinf(torch.tensor(value)):
                    self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.warning(f"  {metric}: {value} (invalid)")
                
            # Save evaluation results
            eval_path = save_dir / "evaluation_results.json"
            import json
            with open(eval_path, 'w') as f:
                # Convert any tensors to floats for JSON serialization
                json_results = {}
                for k, v in eval_results.items():
                    if torch.is_tensor(v):
                        json_results[k] = v.item()
                    else:
                        json_results[k] = float(v) if isinstance(v, (int, float)) else str(v)
                json.dump(json_results, f, indent=2)
                
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {}


def create_quick_test_config() -> ExperimentConfig:
    """Create a configuration for quick testing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Matryoshka parameters
        k=32,  # Smaller k for testing
        group_fractions=[0.6, 0.4],  # 2 groups for simplicity
        auxk_alpha=1/32,
        threshold_start_step=500,  # Earlier start for testing
        
        # Test parameters
        total_steps=1000,
        checkpoint_steps=(),
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
    """Create a configuration for full training."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Matryoshka parameters
        k=64,  # Standard k
        group_fractions=[0.5, 0.3, 0.2],  # 3 groups
        auxk_alpha=1/32,
        threshold_beta=0.999,
        threshold_start_step=1000,
        dead_feature_threshold=10_000_000,
        
        # Full training parameters
        total_steps=25000,
        lr=None,  # Auto-calculated
        
        # Buffer settings (memory efficient)
        n_ctxs=8000,
        ctx_len=128,
        refresh_batch_size=24,
        out_batch_size=768,
        
        # Checkpointing
        checkpoint_steps=(8000, 16000, 25000),
        log_steps=100,
        
        # Evaluation
        eval_batch_size=48,
        eval_n_batches=8,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_high_sparsity_config() -> ExperimentConfig:
    """Create a configuration for high sparsity experiments."""
    config = create_full_config()
    config.k = 32  # Lower k for higher sparsity
    config.auxk_alpha = 1/16  # Higher auxiliary loss
    config.threshold_start_step = 500  # Earlier threshold updates
    return config


def create_many_groups_config() -> ExperimentConfig:
    """Create a configuration with many hierarchical groups."""
    config = create_full_config()
    config.group_fractions = [0.4, 0.25, 0.2, 0.1, 0.05]  # 5 groups
    config.k = 80  # Slightly higher k to accommodate more groups
    config.group_weights = [0.4, 0.25, 0.2, 0.1, 0.05]  # Weighted by importance
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Matryoshka parameters
        k=48,  # Moderate k for memory efficiency
        group_fractions=[0.6, 0.4],  # 2 groups only
        auxk_alpha=1/32,
        threshold_start_step=800,
        
        # Training parameters optimized for 10GB GPU
        total_steps=20000,
        lr=None,  # Auto-calculated
        
        # GPU memory optimized buffer settings
        n_ctxs=3000,     # Small enough to fit in 10GB
        ctx_len=128,
        refresh_batch_size=16,  # Conservative batch size
        out_batch_size=256,     # Conservative output size
        
        # Checkpointing
        checkpoint_steps=(20000,),
        log_steps=100,
        
        # Evaluation - small and efficient
        eval_batch_size=32,
        eval_n_batches=5,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_balanced_groups_config() -> ExperimentConfig:
    """Create a configuration with balanced group sizes."""
    config = create_full_config()
    config.group_fractions = [0.33, 0.33, 0.34]  # Nearly equal groups
    config.group_weights = [1.0, 1.0, 1.0]  # Equal importance
    config.k = 72  # Adjust k for balanced structure
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Matryoshka Batch Top-K SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "high_sparsity",
            "many_groups",
            "gpu_10gb",
            "balanced_groups"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "high_sparsity": create_high_sparsity_config,
        "many_groups": create_many_groups_config,
        "gpu_10gb": create_gpu_10gb_config,
        "balanced_groups": create_balanced_groups_config,
    }
    
    config = config_functions[args.config]()
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<30} | Value")
        print("-" * 40)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<30} | {results[metric]:.4f}")
        
        # Show group-specific metrics
        group_metrics = [k for k in results.keys() if k.startswith("group_")]
        if group_metrics:
            print("\nGroup-Specific Metrics:")
            for metric in sorted(group_metrics):
                print(f"{metric:<30} | {results[metric]:.4f}")
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered", "dead_feature_fraction"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<30} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
    else:
        print("No evaluation results available")


# Usage examples:
# python train_matryoshka_batch_topk.py --config quick_test
# python train_matryoshka_batch_topk.py --config full
# python train_matryoshka_batch_topk.py --config high_sparsity
# python train_matryoshka_batch_topk.py --config many_groups
# python train_matryoshka_batch_topk.py --config gpu_10gb
# python train_matryoshka_batch_topk.py --config balanced_groups

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()