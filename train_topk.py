"""
Training script for robust Top-K SAE implementation.

Key features:
- Robust configuration management
- Enhanced error handling
- Geometric median initialization
- Dead feature resurrection
- Adaptive threshold mechanism
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
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK, TopKConfig, TopKTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire Top-K SAE experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    k: int = 32  # Top-K sparsity level
    
    # Training configuration
    total_steps: int = 10000
    lr: Optional[float] = None  # Auto-computed if None
    warmup_steps: Optional[int] = None  # Will be auto-computed if None
    auxk_alpha: float = 1/32  # Dead feature resurrection coefficient
    threshold_beta: float = 0.999  # Threshold EMA coefficient
    threshold_start_step: int = 1000  # When to start threshold updates
    
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
    wandb_project: str = "topk-sae-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42
    
    # Evaluation configuration
    eval_batch_size: int = 64
    eval_n_batches: int = 10
    
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
    """Manages the entire Top-K SAE training experiment."""
    
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
                logging.FileHandler(log_dir / 'topk_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> TopKConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return TopKConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            k=self.config.k,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> TopKTrainingConfig:
        """Create training configuration from experiment config."""
        # Use explicit warmup_steps from config, or calculate reasonable default
        if self.config.warmup_steps is not None:
            warmup_steps = self.config.warmup_steps
        else:
            # For short training runs, use smaller warmup
            if self.config.total_steps <= 2000:
                warmup_steps = max(50, int(0.05 * self.config.total_steps))
            else:
                warmup_steps = max(200, int(0.02 * self.config.total_steps))
        
        return TopKTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,  # Will be auto-computed if None
            auxk_alpha=self.config.auxk_alpha,
            threshold_beta=self.config.threshold_beta,
            threshold_start_step=self.config.threshold_start_step,
            warmup_steps=warmup_steps,  # Explicitly override the problematic default
        )
        
    def create_trainer_config(self, model_config: TopKConfig, training_config: TopKTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": TopKTrainer,
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
        lr_str = f"_lr{self.config.lr}" if self.config.lr else "_lr_auto"
        
        return (
            f"TopK_SAE_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"k{self.config.k}_auxk{self.config.auxk_alpha}"
            f"{lr_str}"
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
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2, default=str)
            
        self.logger.info(f"Saved experiment config to {config_path}")
        
    def check_existing_checkpoint(self, save_dir: Path) -> bool:
        """Check if there's an existing checkpoint to continue from."""
        checkpoint_path = save_dir / "trainer_0" / "checkpoints"
        exists = checkpoint_path.exists() and any(checkpoint_path.glob("*.pt"))
        
        if exists:
            self.logger.info(f"Found existing checkpoint at {checkpoint_path}")
        else:
            self.logger.info("No existing checkpoint found, starting from scratch")
            
        return exists
        
    def run_training(self) -> Dict[str, float]:
        """Run the complete Top-K SAE training experiment."""
        self.logger.info("Starting Top-K SAE training experiment")
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
            
            # Check for existing checkpoints
            self.check_existing_checkpoint(save_dir)
            
            self.logger.info(f"Model config: {model_config}")
            self.logger.info(f"Training config: {training_config}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}")
            self.logger.info(f"Top-K sparsity: {model_config.k}")
            self.logger.info(f"Auxiliary alpha: {training_config.auxk_alpha}")
            
            # Auto-compute LR if not provided
            if training_config.lr is None:
                scale = model_config.dict_size / (2**14)
                auto_lr = 2e-4 / (scale**0.5)
                self.logger.info(f"Auto-computed learning rate: {auto_lr:.2e}")
            
            # Run training
            self.logger.info("Starting Top-K SAE training...")
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
                    "experiment_type": "top_k_sae",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k": self.config.k,
                    "auxk_alpha": self.config.auxk_alpha,
                    "threshold_beta": self.config.threshold_beta,
                    "features": [
                        "geometric_median_initialization",
                        "dead_feature_resurrection",
                        "adaptive_threshold",
                        "unit_norm_decoder_constraint",
                        "auto_lr_scaling"
                    ],
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
        """Evaluate the trained Top-K SAE model."""
        self.logger.info("Evaluating trained Top-K SAE model...")
        
        try:
            # Load the trained model
            from dictionary_learning.utils import load_dictionary
            
            model_path = save_dir / "trainer_0"
            topk_sae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=topk_sae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add Top-K specific diagnostics
            try:
                # Get a sample batch for sparsity diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                sparsity_diagnostics = topk_sae.get_sparsity_diagnostics(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in sparsity_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                    
                # Add threshold information
                eval_results["final_threshold"] = topk_sae.threshold.item() if topk_sae.threshold >= 0 else -1
                    
            except Exception as e:
                self.logger.warning(f"Could not compute sparsity diagnostics: {e}")
            
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
        k=32,
        
        # Test parameters
        total_steps=1000,
        warmup_steps=50,  # Explicitly set for quick test
        threshold_start_step=100,  # Also reduce this for quick test
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
    """Create a configuration for full training."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        k=32,
        
        # Full training parameters
        total_steps=25000,
        lr=None,  # Auto-computed
        auxk_alpha=1/32,
        threshold_beta=0.999,
        threshold_start_step=1000,
        
        # Buffer settings
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


def create_high_k_config() -> ExperimentConfig:
    """Create a configuration with higher k for denser representation."""
    config = create_full_config()
    config.k = 64  # Higher sparsity level
    config.auxk_alpha = 1/16  # Stronger auxiliary loss for higher k
    return config


def create_low_k_config() -> ExperimentConfig:
    """Create a configuration with lower k for sparser representation."""
    config = create_full_config()
    config.k = 16  # Lower sparsity level
    config.auxk_alpha = 1/64  # Weaker auxiliary loss for lower k
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        k=32,
        
        # Training parameters optimized for 10GB GPU
        total_steps=20000,
        lr=None,  # Auto-computed
        auxk_alpha=1/32,
        threshold_beta=0.999,
        threshold_start_step=1000,
        
        # GPU memory optimized buffer settings
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=16,
        out_batch_size=256,
        
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


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Top-K SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "high_k",
            "low_k",
            "gpu_10gb"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "high_k": create_high_k_config,
        "low_k": create_low_k_config,
        "gpu_10gb": create_gpu_10gb_config,
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
                
        # Show Top-K specific metrics
        topk_metrics = [k for k in results.keys() if "threshold" in k.lower() or "effective_l0" in k.lower()]
        if topk_metrics:
            print("\nTop-K Specific Metrics:")
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
# python train_topk.py --config quick_test
# python train_topk.py --config full
# python train_topk.py --config high_k    # k=64 for denser representation
# python train_topk.py --config low_k     # k=16 for sparser representation
# python train_topk.py --config gpu_10gb  # Optimized for 10GB GPU


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()
