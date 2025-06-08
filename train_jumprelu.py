"""
Training script for robust JumpReLU SAE implementation.

Key features:
- Clean configuration management
- Multiple preset configurations
- Comprehensive evaluation
- Robust error handling
- Memory-efficient settings
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
from dictionary_learning.trainers.jumprelu import JumpReluTrainer, JumpReluSAE, JumpReluConfig, JumpReluTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire JumpReLU experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # JumpReLU-specific config
    bandwidth: float = 0.001
    threshold_init: float = 0.001
    sparsity_penalty: float = 1.0
    target_l0: float = 20.0
    apply_b_dec_to_input: bool = False  # SAE-lens compatibility
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 7e-5
    
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
    wandb_project: str = "jumprelu-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "float32"
    autocast_dtype: str = "float32"
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
    """Manages the entire JumpReLU training experiment."""
    
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
                logging.FileHandler(log_dir / 'jumprelu_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> JumpReluConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return JumpReluConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            bandwidth=self.config.bandwidth,
            threshold_init=self.config.threshold_init,
            apply_b_dec_to_input=self.config.apply_b_dec_to_input,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> JumpReluTrainingConfig:
        """Create training configuration from experiment config."""
        return JumpReluTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            bandwidth=self.config.bandwidth,
            sparsity_penalty=self.config.sparsity_penalty,
            target_l0=self.config.target_l0,
        )
        
    def create_trainer_config(self, model_config: JumpReluConfig, training_config: JumpReluTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": JumpReluTrainer,
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
        return (
            f"JumpReLU_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_l0{self.config.target_l0}_"
            f"bw{self.config.bandwidth}_sp{self.config.sparsity_penalty}"
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
        """Run the complete training experiment."""
        self.logger.info("Starting JumpReLU training experiment")
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
            self.logger.info(f"Target L0: {training_config.target_l0}")
            self.logger.info(f"Bandwidth: {training_config.bandwidth}")
            
            # Run training
            self.logger.info("Starting JumpReLU training...")
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
                    "experiment_type": "jumprelu",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "target_l0": self.config.target_l0,
                    "bandwidth": self.config.bandwidth,
                    "sparsity_penalty": self.config.sparsity_penalty,
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
        """Evaluate the trained model with JumpReLU-specific diagnostics."""
        self.logger.info("Evaluating trained model...")
        
        try:
            # Load the trained model
            from dictionary_learning.utils import load_dictionary
            
            model_path = save_dir / "trainer_0"
            jumprelu_sae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=jumprelu_sae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add JumpReLU-specific diagnostics if possible
            try:
                # Get a sample batch for feature diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                feature_diagnostics = jumprelu_sae.get_feature_diagnostics(sample_batch.to(self.config.device))
                l0_diagnostic = jumprelu_sae.get_l0(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in feature_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                    
                eval_results["final_l0"] = l0_diagnostic.item()
                eval_results["final_threshold_mean"] = jumprelu_sae.threshold.mean().item()
                eval_results["final_threshold_std"] = jumprelu_sae.threshold.std().item()
                    
            except Exception as e:
                self.logger.warning(f"Could not compute JumpReLU diagnostics: {e}")
            
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
        
        # JumpReLU parameters
        bandwidth=0.001,
        threshold_init=0.001,
        sparsity_penalty=1.0,
        target_l0=20.0,
        
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


def create_standard_config() -> ExperimentConfig:
    """Create a configuration for standard training."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # JumpReLU parameters
        bandwidth=0.001,
        threshold_init=0.001,
        sparsity_penalty=1.0,
        target_l0=20.0,
        
        # Standard training parameters
        total_steps=25000,
        lr=7e-5,
        
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
        dtype="float32",
        autocast_dtype="float32",
        seed=42,
    )


def create_high_sparsity_config() -> ExperimentConfig:
    """Create a configuration for high sparsity training."""
    config = create_standard_config()
    
    # High sparsity settings
    config.target_l0 = 10.0  # Lower target L0 for higher sparsity
    config.sparsity_penalty = 2.0  # Higher penalty
    config.bandwidth = 0.0005  # Smaller bandwidth for sharper thresholds
    
    return config


def create_low_sparsity_config() -> ExperimentConfig:
    """Create a configuration for low sparsity training."""
    config = create_standard_config()
    
    # Low sparsity settings
    config.target_l0 = 40.0  # Higher target L0 for lower sparsity
    config.sparsity_penalty = 0.5  # Lower penalty
    config.bandwidth = 0.002  # Larger bandwidth for smoother thresholds
    
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # JumpReLU parameters
        bandwidth=0.001,
        threshold_init=0.001,
        sparsity_penalty=1.0,
        target_l0=20.0,
        
        # GPU memory optimized parameters
        total_steps=20000,
        lr=7e-5,
        
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
        dtype="float32",
        autocast_dtype="float32",
        seed=42,
    )


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JumpReLU SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "standard",
            "high_sparsity",
            "low_sparsity", 
            "gpu_10gb"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    # Select configuration
    config_functions = {
        "quick_test": create_quick_test_config,
        "standard": create_standard_config,
        "high_sparsity": create_high_sparsity_config,
        "low_sparsity": create_low_sparsity_config,
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
                
        # Show JumpReLU-specific diagnostics if available
        jumprelu_metrics = [k for k in results.keys() if any(x in k.lower() for x in ["l0", "threshold", "feature"])]
        if jumprelu_metrics:
            print("\nJumpReLU Diagnostics:")
            for metric in jumprelu_metrics:
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
# python train_jumprelu.py --config quick_test
# python train_jumprelu.py --config standard
# python train_jumprelu.py --config high_sparsity
# python train_jumprelu.py --config low_sparsity
# python train_jumprelu.py --config gpu_10gb


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()
