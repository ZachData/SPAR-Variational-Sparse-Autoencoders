"""
Clean training script for Standard SAE with robust configuration management.

Usage examples:
python train_standard_sae.py --config quick_test
python train_standard_sae.py --config full
python train_standard_sae.py --config april_update
python train_standard_sae.py --config with_resampling
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
from dictionary_learning.trainers.standard import (
    StandardTrainer, 
    StandardTrainerAprilUpdate,
    StandardSAEConfig, 
    StandardSAETrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire Standard SAE experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 8.0
    
    # Model-specific config
    use_april_update_mode: bool = False  # True for April update training
    
    # Training configuration
    total_steps: int = 25000
    lr: float = 1e-3
    l1_penalty: float = 1e-1
    resample_steps: Optional[int] = None  # Set to enable neuron resampling
    
    # Buffer configuration
    n_ctxs: int = 8000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024
    
    # Logging and saving
    checkpoint_steps: tuple = (10000, 20000, 25000)
    log_steps: int = 100
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "standard-sae-experiments"
    
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


class StandardSAEExperimentRunner:
    """Manages the entire Standard SAE training experiment."""
    
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
                logging.FileHandler(log_dir / 'standard_sae_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> StandardSAEConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return StandardSAEConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> StandardSAETrainingConfig:
        """Create training configuration from experiment config."""
        return StandardSAETrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            l1_penalty=self.config.l1_penalty,
            resample_steps=self.config.resample_steps,
        )
        
    def create_trainer_config(self, model_config: StandardSAEConfig, training_config: StandardSAETrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": StandardTrainer,
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
        update_suffix = "_april" if self.config.use_april_update_mode else "_original"
        resample_suffix = f"_resample{self.config.resample_steps}" if self.config.resample_steps else "_no_resample"
        
        return (
            f"StandardSAE_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_l1{self.config.l1_penalty}"
            f"{update_suffix}{resample_suffix}"
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
        
    def run_training(self) -> Dict[str, float]:
        """Run the complete training experiment."""
        self.logger.info("Starting Standard SAE training experiment")
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
            self.logger.info(f"Using April update: {model_config.use_april_update_mode}")
            self.logger.info(f"Resampling enabled: {training_config.resample_steps is not None}")
            
            # Run training
            self.logger.info("Starting Standard SAE training...")
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
                    "experiment_type": "standard_sae",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "use_april_update_mode": self.config.use_april_update_mode,
                    "resample_steps": self.config.resample_steps,
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
            sae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=sae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
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
        
        # Quick test parameters
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
        dict_size_multiple=8.0,
        
        # Full training parameters
        total_steps=50000,
        lr=1e-3,
        l1_penalty=1e-1,
        
        # Buffer settings
        n_ctxs=30000,
        ctx_len=128,
        refresh_batch_size=512,
        out_batch_size=8192,
        
        # Checkpointing
        checkpoint_steps=(15000, 30000, 45000, 50000),
        log_steps=100,
        
        # Evaluation
        eval_batch_size=64,
        eval_n_batches=10,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_april_update_config() -> ExperimentConfig:
    """Create a configuration for April update training."""
    config = create_full_config()
    config.use_april_update_mode = True
    config.wandb_project = "standard-sae-april-update"
    return config


def create_with_resampling_config() -> ExperimentConfig:
    """Create a configuration with neuron resampling enabled."""
    config = create_full_config()
    config.resample_steps = 25000  # Resample every 25k steps
    config.wandb_project = "standard-sae-with-resampling"
    return config


def create_memory_efficient_config() -> ExperimentConfig:
    """Create a memory-efficient configuration for smaller GPUs."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Training parameters
        total_steps=25000,
        lr=1e-3,
        l1_penalty=1e-1,
        
        # Memory-efficient buffer settings
        n_ctxs=5000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=512,
        
        # Checkpointing
        checkpoint_steps=(25000,),
        log_steps=100,
        
        # Evaluation
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
    
    parser = argparse.ArgumentParser(description="Standard SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "april_update",
            "with_resampling",
            "memory_efficient"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    # Select configuration
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "april_update": create_april_update_config,
        "with_resampling": create_with_resampling_config,
        "memory_efficient": create_memory_efficient_config,
    }
    
    config = config_functions[args.config]()
    runner = StandardSAEExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<25} | Value")
        print("-" * 35)
        
        for metric in key_metrics:
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


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()