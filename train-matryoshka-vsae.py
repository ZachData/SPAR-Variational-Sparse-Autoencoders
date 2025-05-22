"""
Improved training script for MatryoshkaVSAE with better practices.
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

# Import our improved implementations
from dictionary_learning.trainers.vsae_matryoshka import (
    MatryoshkaVSAEIso, 
    MatryoshkaVSAEIsoTrainer,
    MatryoshkaConfig,
    TrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32
    
    # Schedule configuration
    warmup_steps: Optional[int] = None  # Will be set to 5% of total_steps
    sparsity_warmup_steps: Optional[int] = None  # Will be set to 5% of total_steps
    decay_start_step: Optional[int] = None  # Will be set to 80% of total_steps
    
    # Matryoshka configuration
    group_fractions: tuple = (0.25, 0.25, 0.25, 0.25)
    group_weights: tuple = (0.4, 0.3, 0.2, 0.1)
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    
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
    wandb_project: str = "vsae-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"  # or "float32"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42
    
    def __post_init__(self):
        """Set derived configuration values."""
        # Set default step values based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.05 * self.total_steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.decay_start_step is None:
            self.decay_start_step = int(0.8 * self.total_steps)
    
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
    """Manages the entire training experiment with proper setup and cleanup."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_logging()
        self.setup_reproducibility()
        
    def setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
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
            group_fractions=list(self.config.group_fractions),
            group_weights=list(self.config.group_weights),
            var_flag=self.config.var_flag,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
    def create_training_config(self) -> TrainingConfig:
        """Create training configuration from experiment config."""
        return TrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            auxk_alpha=self.config.auxk_alpha,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
        )
        
    def create_trainer_config(self, model_config: MatryoshkaConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        # Convert non-serializable objects to serializable formats
        serializable_model_config = {
            "activation_dim": model_config.activation_dim,
            "dict_size": model_config.dict_size,
            "group_fractions": model_config.group_fractions,
            "group_weights": model_config.group_weights,
            "var_flag": model_config.var_flag,
            "dtype": str(model_config.dtype),  # Convert to string
            "device": str(model_config.device),  # Convert to string
        }
        
        serializable_training_config = {
            "steps": training_config.steps,
            "lr": training_config.lr,
            "kl_coeff": training_config.kl_coeff,
            "auxk_alpha": training_config.auxk_alpha,
            "warmup_steps": training_config.warmup_steps,
            "sparsity_warmup_steps": training_config.sparsity_warmup_steps,
            "decay_start": training_config.decay_start,
            "dead_feature_threshold": training_config.dead_feature_threshold,
            "gradient_clip_norm": training_config.gradient_clip_norm,
        }
        
        return {
            "trainer": MatryoshkaVSAEIsoTrainer,
            "model_config": model_config,  # Pass the actual objects to trainer
            "training_config": training_config,  # Pass the actual objects to trainer
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "submodule_name": self.config.hook_name,
            "wandb_name": self.get_experiment_name(),
            "seed": self.config.seed,
            # Add serializable versions for JSON saving
            "_serializable_model_config": serializable_model_config,
            "_serializable_training_config": serializable_training_config,
        }
        
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        return (
            f"MatryoshkaVSAE_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"
            f"lr{self.config.lr}_kl{self.config.kl_coeff}_"
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
        
        import json
        with open(config_path, 'w') as f:
            # Convert config to dict, handling special types
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2, default=str)
            
        self.logger.info(f"Saved experiment config to {config_path}")
        
    def run_training(self) -> Dict[str, float]:
        """Run the complete training experiment."""
        self.logger.info("Starting MatryoshkaVSAE training experiment")
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
                    "experiment_type": "matryoshka_vsae",
                    "dict_size_multiple": self.config.dict_size_multiple,
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
        """Evaluate the trained model."""
        self.logger.info("Evaluating trained model...")
        
        try:
            # Load the trained model
            from dictionary_learning.utils import load_dictionary
            
            model_path = save_dir / "trainer_0"
            vsae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=vsae,
                activations=buffer,
                batch_size=64,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=10
            )
            
            # Log results
            self.logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                self.logger.info(f"  {metric}: {value:.4f}")
                
            # Save evaluation results
            eval_path = save_dir / "evaluation_results.json"
            import json
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
                
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {}


def create_quick_test_config() -> ExperimentConfig:
    """Create a configuration for quick testing (small steps)."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Quick test parameters
        total_steps=10,
        warmup_steps=2,
        sparsity_warmup_steps=5,
        decay_start_step=8,
        checkpoint_steps=(5, 10),
        log_steps=2,
        
        # Small buffer for testing
        n_ctxs=1000,
        refresh_batch_size=16,
        out_batch_size=256,
        
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
        
        # Full training parameters
        total_steps=50000,
        lr=5e-4,
        kl_coeff=500.0,
        auxk_alpha=1/32,
        
        # Matryoshka settings
        group_fractions=(0.25, 0.25, 0.25, 0.25),
        group_weights=(0.4, 0.3, 0.2, 0.1),
        
        # Full buffer settings
        n_ctxs=30000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        
        # Checkpointing
        checkpoint_steps=(10000, 25000, 50000),
        log_steps=100,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def main():
    """Main training function."""
    # Choose configuration
    # For quick testing:
    config = create_quick_test_config()
    
    # For full training, uncomment:
    # config = create_full_config()
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    
    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive"]
        print(f"{'Metric':<25} | Value")
        print("-" * 35)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results: {results}")
    else:
        print("No evaluation results available")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()