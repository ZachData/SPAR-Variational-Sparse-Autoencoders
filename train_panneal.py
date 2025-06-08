"""
Training script for P-Annealing SAE with enhanced configuration management.

This script implements the P-annealing approach that gradually transitions
from L1 to near-L0 sparsity penalties for better sparsity-reconstruction tradeoffs.
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
from dictionary_learning.trainers.p_anneal import PAnnealTrainer, PAnnealConfig, PAnnealTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the P-Annealing experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # P-Annealing specific parameters
    sparsity_function: str = 'Lp'  # 'Lp' or 'Lp^p'
    initial_sparsity_penalty: float = 1e-1
    p_start: float = 1.0  # Start with L1 norm
    p_end: float = 0.1  # End near L0 norm
    anneal_start_fraction: float = 0.3  # Start annealing at 30% of training
    n_sparsity_updates: int = 10  # Number of p value updates
    sparsity_queue_length: int = 10
    
    # Training configuration
    total_steps: int = 50000
    lr: float = 1e-3
    warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    resample_steps: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # Buffer configuration
    n_ctxs: int = 3000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024
    
    # Logging and saving
    checkpoint_steps: tuple = (25000, 50000)
    log_steps: int = 200
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "p-anneal-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "float32"
    autocast_dtype: str = "float32"
    seed: Optional[int] = 42
    
    # Evaluation configuration
    eval_batch_size: int = 64
    eval_n_batches: int = 10
    
    def __post_init__(self):
        """Set derived configuration values."""
        # Set default annealing start
        if self.anneal_start_fraction is not None:
            self.anneal_start = int(self.anneal_start_fraction * self.total_steps)
        else:
            self.anneal_start = int(0.3 * self.total_steps)  # Default to 30%
    
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
    """Manages the entire P-Annealing training experiment."""
    
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
                logging.FileHandler(log_dir / 'p_anneal_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> PAnnealConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return PAnnealConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            sparsity_function=self.config.sparsity_function,
            initial_sparsity_penalty=self.config.initial_sparsity_penalty,
            p_start=self.config.p_start,
            p_end=self.config.p_end,
            anneal_start=self.config.anneal_start,
            anneal_end=None,  # Will default to total_steps - 1
            n_sparsity_updates=self.config.n_sparsity_updates,
            sparsity_queue_length=self.config.sparsity_queue_length,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> PAnnealTrainingConfig:
        """Create training configuration from experiment config."""
        return PAnnealTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            decay_start=self.config.decay_start,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            resample_steps=self.config.resample_steps,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )
        
    def create_trainer_config(self, model_config: PAnnealConfig, training_config: PAnnealTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": PAnnealTrainer,
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
        p_range = f"p{self.config.p_start}-{self.config.p_end}"
        sparsity_fn = self.config.sparsity_function
        
        return (
            f"PAnneal_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_{p_range}_{sparsity_fn}_"
            f"anneal{self.config.anneal_start}_updates{self.config.n_sparsity_updates}"
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
        """Run the complete P-Annealing training experiment."""
        self.logger.info("Starting P-Annealing training experiment")
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
            self.logger.info(f"P-annealing: {model_config.p_start} → {model_config.p_end} "
                           f"starting at step {model_config.anneal_start}")
            
            # Run training
            self.logger.info("Starting P-Annealing training...")
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
                    "experiment_type": "p_anneal",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "sparsity_function": self.config.sparsity_function,
                    "p_start": self.config.p_start,
                    "p_end": self.config.p_end,
                    "anneal_start": self.config.anneal_start,
                    "n_sparsity_updates": self.config.n_sparsity_updates,
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
            
            # Add P-annealing specific diagnostics if possible
            try:
                if hasattr(sae, 'get_p_diagnostics'):
                    p_diagnostics = sae.get_p_diagnostics()
                    for key, value in p_diagnostics.items():
                        eval_results[f"final_{key}"] = value
                        
            except Exception as e:
                self.logger.warning(f"Could not compute P-annealing diagnostics: {e}")
            
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
        
        # P-annealing parameters
        sparsity_function='Lp',
        initial_sparsity_penalty=1e-1,
        p_start=1.0,
        p_end=0.3,
        anneal_start_fraction=0.4,  # Start annealing at 40% of training
        n_sparsity_updates=5,
        
        # Quick test parameters
        total_steps=2500,
        checkpoint_steps=(),
        log_steps=100,
        
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
    """Create a standard configuration for full training."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # P-annealing parameters
        sparsity_function='Lp',
        initial_sparsity_penalty=1e-1,
        p_start=1.0,
        p_end=0.1,
        anneal_start_fraction=0.3,  # Start annealing at 30%
        n_sparsity_updates=10,
        
        # Full training parameters
        total_steps=50000,
        lr=1e-3,
        
        # Buffer settings
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        
        # Checkpointing
        checkpoint_steps=(25000, 50000),
        log_steps=200,
        
        # Evaluation
        eval_batch_size=64,
        eval_n_batches=10,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
        autocast_dtype="float32",
        seed=42,
    )


def create_aggressive_annealing_config() -> ExperimentConfig:
    """Create configuration with more aggressive p-annealing."""
    config = create_standard_config()
    
    # More aggressive annealing
    config.p_end = 0.05  # Go closer to L0
    config.n_sparsity_updates = 20  # More gradual transition
    config.anneal_start_fraction = 0.2  # Start earlier
    config.initial_sparsity_penalty = 2e-1  # Higher initial penalty
    
    return config


def create_lp_power_config() -> ExperimentConfig:
    """Create configuration using Lp^p sparsity function."""
    config = create_standard_config()
    
    # Use Lp^p instead of Lp
    config.sparsity_function = 'Lp^p'
    config.initial_sparsity_penalty = 5e-2  # Lower penalty for Lp^p
    
    return config


def create_with_resampling_config() -> ExperimentConfig:
    """Create configuration with dead neuron resampling."""
    config = create_standard_config()
    
    # Enable resampling
    config.resample_steps = 5000  # Resample every 5000 steps
    config.total_steps = 75000  # Longer training with resampling
    config.checkpoint_steps = (25000, 50000, 75000)
    
    return config


def create_memory_efficient_config() -> ExperimentConfig:
    """Create configuration optimized for limited GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # P-annealing parameters
        sparsity_function='Lp',
        initial_sparsity_penalty=1e-1,
        p_start=1.0,
        p_end=0.2,
        anneal_start_fraction=0.3,
        n_sparsity_updates=8,
        
        # Memory-efficient parameters
        total_steps=30000,
        lr=1e-3,
        
        # Smaller buffer
        n_ctxs=1500,
        ctx_len=128,
        refresh_batch_size=16,
        out_batch_size=512,
        
        # Checkpointing
        checkpoint_steps=(30000,),
        log_steps=200,
        
        # Smaller evaluation
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
    
    parser = argparse.ArgumentParser(description="P-Annealing SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "standard", 
            "aggressive_annealing",
            "lp_power",
            "with_resampling",
            "memory_efficient"
        ], 
        default="quick_test",
        help="Configuration preset to use"
    )
    args = parser.parse_args()
    
    # Configuration selection
    config_functions = {
        "quick_test": create_quick_test_config,
        "standard": create_standard_config,
        "aggressive_annealing": create_aggressive_annealing_config,
        "lp_power": create_lp_power_config,
        "with_resampling": create_with_resampling_config,
        "memory_efficient": create_memory_efficient_config,
    }
    
    config = config_functions[args.config]()
    
    print(f"Starting P-Annealing experiment with config: {args.config}")
    print(f"P-annealing: {config.p_start} → {config.p_end}")
    print(f"Sparsity function: {config.sparsity_function}")
    print(f"Total steps: {config.total_steps}")
    print(f"Annealing starts at step: {config.anneal_start}")
    
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        # Display key metrics
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"\n{'Metric':<25} | Value")
        print("-" * 35)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        # Show P-annealing specific metrics
        p_metrics = [k for k in results.keys() if k.startswith("final_") and "p" in k.lower()]
        if p_metrics:
            print("\nP-Annealing Diagnostics:")
            for metric in p_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]}")
                
        # Additional metrics
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
    else:
        print("No evaluation results available")


# Usage examples:
# python train_p_anneal.py --config quick_test
# python train_p_anneal.py --config standard
# python train_p_anneal.py --config aggressive_annealing
# python train_p_anneal.py --config lp_power
# python train_p_anneal.py --config with_resampling
# python train_p_anneal.py --config memory_efficient

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()
