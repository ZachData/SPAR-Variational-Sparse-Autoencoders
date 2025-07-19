"""
Training script for Enhanced Gated SAE with P-Annealing.

This script demonstrates the sophisticated P-annealing approach which gradually
transitions from L1 to L0 sparsity penalty during training.
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
from dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer, GatedAnnealConfig, GatedAnnealTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire Gated Annealing experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.hook_resid_post" # was blocks.0.mlp.hook_post
    dict_size_multiple: float = 4.0
    
    # Gated Annealing specific config
    sparsity_function: str = 'Lp^p'  # 'Lp' or 'Lp^p'
    initial_sparsity_penalty: float = 1e-1
    anneal_start: int = 15000  # Step to start annealing p
    anneal_end: Optional[int] = None  # Will be set to total_steps - 1
    p_start: float = 1.0  # L1 penalty
    p_end: float = 0.0    # Approaching L0
    n_sparsity_updates: int = 10
    sparsity_queue_length: int = 10
    
    # Training configuration
    total_steps: int = 50000
    lr: float = 3e-4
    resample_steps: Optional[int] = 5000  # Resample dead neurons every 5k steps
    
    # Buffer configuration
    n_ctxs: int = 8000
    ctx_len: int = 128
    refresh_batch_size: int = 24
    out_batch_size: int = 768
    
    # Logging and saving
    checkpoint_steps: tuple = (15000, 30000, 50000)
    log_steps: int = 100
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "gated-anneal-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42
    
    # Evaluation configuration
    eval_batch_size: int = 48
    eval_n_batches: int = 8
    
    def __post_init__(self):
        """Set derived configuration values."""
        # Set anneal_end if not provided
        if self.anneal_end is None:
            self.anneal_end = self.total_steps - 1
    
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
    """Manages the entire training experiment with Gated Annealing SAE."""
    
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
                logging.FileHandler(log_dir / 'gated_anneal_training.log'),
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
        
        self.logger.info(f"Model loaded. d_model: {model.cfg.d_model}, d_mlp: {model.cfg.d_model}")
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
        
        # FIXED: Use d_model for residual stream hooks, d_mlp for MLP hooks
        if "resid" in self.config.hook_name or "attn" in self.config.hook_name:
            d_submodule = model.cfg.d_model
        else:
            d_submodule = model.cfg.d_model
        
        # Create activation buffer
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=self.config.hook_name,
            d_submodule=d_submodule,
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> GatedAnnealConfig:
        """Create model configuration from experiment config."""
        # FIXED: Use d_model for residual stream hooks, d_mlp for MLP hooks
        if "resid" in self.config.hook_name or "attn" in self.config.hook_name:
            activation_dim = model.cfg.d_model
        else:
            activation_dim = model.cfg.d_model
            
        dict_size = int(self.config.dict_size_multiple * activation_dim)
        
        return GatedAnnealConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            sparsity_function=self.config.sparsity_function,
            initial_sparsity_penalty=self.config.initial_sparsity_penalty,
            anneal_start=self.config.anneal_start,
            anneal_end=self.config.anneal_end,
            p_start=self.config.p_start,
            p_end=self.config.p_end,
            n_sparsity_updates=self.config.n_sparsity_updates,
            sparsity_queue_length=self.config.sparsity_queue_length,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )
        
    def create_training_config(self) -> GatedAnnealTrainingConfig:
        """Create training configuration from experiment config."""
        return GatedAnnealTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            resample_steps=self.config.resample_steps,
        )
        
    def create_trainer_config(self, model_config: GatedAnnealConfig, training_config: GatedAnnealTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": GatedAnnealTrainer,
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
        p_range = f"p{self.config.p_start:.1f}to{self.config.p_end:.1f}"
        anneal_range = f"anneal{self.config.anneal_start}to{self.config.anneal_end}"
        
        # FIXED: Use actual activation dimension instead of hardcoded 512
        activation_dim = self.config.dict_size_multiple * 512 if "resid" in self.config.hook_name else self.config.dict_size_multiple * 2048
        
        return (
            f"GatedAnneal_{self.config.model_name}_"
            f"d{int(activation_dim)}_"
            f"lr{self.config.lr}_{p_range}_{anneal_range}_"
            f"{self.config.sparsity_function}"
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
        """Run the complete training experiment with Gated Annealing."""
        self.logger.info("Starting Gated Annealing SAE training experiment")
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
            self.logger.info(f"Activation dimension: {model_config.activation_dim}")
            self.logger.info(f"P-annealing: {model_config.p_start} → {model_config.p_end} over {model_config.n_sparsity_updates} steps")
            self.logger.info(f"Annealing schedule: steps {model_config.anneal_start} to {model_config.anneal_end}")
            
            # Run training
            self.logger.info("Starting Gated Annealing training...")
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
                    "experiment_type": "gated_anneal",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "p_annealing": f"{self.config.p_start}→{self.config.p_end}",
                    "sparsity_function": self.config.sparsity_function,
                    "anneal_start": self.config.anneal_start,
                    "anneal_end": self.config.anneal_end,
                    "n_sparsity_updates": self.config.n_sparsity_updates,
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
            gated_sae, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=gated_sae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add Gated SAE-specific diagnostics if possible
            try:
                # Get a sample batch for additional diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                if hasattr(gated_sae, 'get_diagnostics'):
                    diagnostics = gated_sae.get_diagnostics(sample_batch.to(self.config.device))
                    
                    # Add to eval results
                    for key, value in diagnostics.items():
                        eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                        
            except Exception as e:
                self.logger.warning(f"Could not compute additional diagnostics: {e}")
            
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
    """Create a configuration for quick testing of Gated Annealing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",
        dict_size_multiple=4.0,
        
        # Test parameters
        total_steps=2000,
        anneal_start=500,  # Start annealing early for testing
        anneal_end=1800,
        resample_steps=None,  # Disable resampling for quick test
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
    """Create a configuration for full Gated Annealing training."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",
        dict_size_multiple=4.0,
        
        # Full training parameters
        total_steps=20000,
        lr=4e-5,
        
        # P-annealing schedule
        anneal_start=9000,  # Start annealing after initial training
        p_start=1.0,         # L1 penalty
        p_end=0.9,           # Approaching L0
        n_sparsity_updates=30,  # More granular annealing
        
        # Resampling
        resample_steps=5000,

        # GPU memory optimized buffer settings
        n_ctxs=2500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=192,
        
        # Checkpointing
        checkpoint_steps=(20000,),
        log_steps=1000,
        
        # Evaluation - faster and more memory efficient
        eval_batch_size=24,  # Smaller eval batches
        eval_n_batches=6,    # Fewer eval batches
        
        # System settings for performance
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",    # Memory efficient
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_aggressive_annealing_config() -> ExperimentConfig:
    """Create a configuration with more aggressive P-annealing."""
    config = create_full_config()
    
    # More aggressive annealing
    config.anneal_start = 5000   # Start earlier
    config.p_end = -0.5          # Go below 0 for stronger L0 approximation
    config.n_sparsity_updates = 50  # More frequent updates
    config.initial_sparsity_penalty = 5e-2  # Stronger initial penalty
    
    return config


def create_conservative_annealing_config() -> ExperimentConfig:
    """Create a configuration with conservative P-annealing."""
    config = create_full_config()
    
    # Conservative annealing
    config.anneal_start = 25000  # Start later
    config.p_end = 0.5           # Don't go all the way to 0
    config.n_sparsity_updates = 10  # Fewer updates
    config.initial_sparsity_penalty = 2e-1  # Weaker initial penalty
    
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gated Annealing SAE Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "aggressive_annealing",
            "conservative_annealing"
        ], 
        default="quick_test",
        help="Configuration preset for training runs"
    )
    args = parser.parse_args()
    
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "aggressive_annealing": create_aggressive_annealing_config,
        "conservative_annealing": create_conservative_annealing_config,
    }
    
    config = config_functions[args.config]()
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<25} | Value")
        print("-" * 35)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        # Show gated SAE specific metrics if available
        gated_metrics = [k for k in results.keys() if any(x in k.lower() for x in ["gate", "aux", "sparsity"])]
        if gated_metrics:
            print("\nGated SAE Specific Metrics:")
            for metric in gated_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
        print("="*60)
    else:
        print("No evaluation results available")


# Usage examples:
# python train_gated_anneal.py --config quick_test
# python train_gated_anneal.py --config full
# python train_gated_anneal.py --config aggressive_annealing
# python train_gated_anneal.py --config conservative_annealing


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()