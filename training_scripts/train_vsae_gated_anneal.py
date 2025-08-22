"""
Training script for VSAEGatedAnneal with all the robustness improvements.

This script provides clean, memory-efficient training for VSAE with gated architecture
and p-annealing, following the robust patterns from train_isovsae_1lgelu.py.

Key features:
- Multiple configuration presets for different use cases
- Memory-efficient buffer settings
- Comprehensive evaluation and logging
- Robust error handling
- GPU memory optimization
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
# FIXED: Import from vsae_gated_anneal instead of vsae_gated
from dictionary_learning.trainers.vsae_gated_anneal import VSAEGatedAnnealTrainer, VSAEGatedAnneal, VSAEGatedAnnealConfig, VSAEGatedAnnealTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire VSAEGatedAnneal experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # Model-specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 300.0  # Lower for gated models
    sparsity_coeff: float = 0.1
    kl_warmup_steps: Optional[int] = None
    
    # P-annealing configuration
    p_start: float = 1.0
    p_end: float = 0.5
    anneal_start_frac: float = 0.2
    anneal_end_frac: float = 0.8
    
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
    wandb_project: str = "vsae-gated-anneal-experiments"  # Updated project name
    
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
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.total_steps)
    
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
    """Manages the entire VSAEGatedAnneal training experiment."""
    
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
                logging.FileHandler(log_dir / 'vsae_gated_anneal_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEGatedAnnealConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEGatedAnnealConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init
        )
        
    def create_training_config(self) -> VSAEGatedAnnealTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEGatedAnnealTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            sparsity_coeff=self.config.sparsity_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
            p_start=self.config.p_start,
            p_end=self.config.p_end,
            anneal_start_frac=self.config.anneal_start_frac,
            anneal_end_frac=self.config.anneal_end_frac,
        )
        
    def create_trainer_config(self, model_config: VSAEGatedAnnealConfig, training_config: VSAEGatedAnnealTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEGatedAnnealTrainer,
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
        var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"
        p_suffix = f"_p{self.config.p_start}to{self.config.p_end}"
        
        return (
            f"VSAEGatedAnneal_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}_sp{self.config.sparsity_coeff}"
            f"{var_suffix}{p_suffix}"
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
        self.logger.info("Starting VSAEGatedAnneal training experiment")
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
            self.logger.info(f"P-annealing: {training_config.p_start} -> {training_config.p_end}")
            
            # Run training
            self.logger.info("Starting VSAEGatedAnneal training...")
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
                    "experiment_type": "vsae_gated_anneal",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "var_flag": self.config.var_flag,
                    "p_annealing": f"{self.config.p_start}->{self.config.p_end}",
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
        """Evaluate the trained model with enhanced diagnostics."""
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
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add VSAEGatedAnneal-specific diagnostics
            try:
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                kl_diagnostics = vsae.get_kl_diagnostics(sample_batch.to(self.config.device))
                gating_diagnostics = vsae.get_gating_diagnostics(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in kl_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                for key, value in gating_diagnostics.items():
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
    """Create a configuration for quick testing with safe LR schedule."""
    config = ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Test parameters - use enough steps for safe LR schedule
        total_steps=3000,  # Enough steps to avoid warmup/decay conflicts
        checkpoint_steps=(),
        log_steps=100,
        kl_warmup_steps=300,  # 10% of total_steps
        
        # P-annealing parameters for testing
        p_start=1.0,
        p_end=0.7,  # Less aggressive for quick test
        anneal_start_frac=0.3,
        anneal_end_frac=0.9,
        
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
    return config


def create_full_config() -> ExperimentConfig:
    """Create a configuration for full training - memory efficient."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Full training parameters
        total_steps=25000,
        lr=5e-4,
        kl_coeff=300.0,
        sparsity_coeff=0.1,
        kl_warmup_steps=2500,
        
        # P-annealing settings
        p_start=1.0,
        p_end=0.5,
        anneal_start_frac=0.2,
        anneal_end_frac=0.8,
        
        # Model settings
        var_flag=0,  # Start with fixed variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        
        # Memory efficient buffer settings
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


def create_learned_variance_config() -> ExperimentConfig:
    """Create a configuration for training with learned variance."""
    config = create_full_config()
    
    # Enable learned variance with adjusted settings
    config.var_flag = 1
    config.kl_coeff = 200.0  # Lower KL coeff for learned variance
    config.total_steps = 30000
    config.kl_warmup_steps = 6000  # Longer KL warmup
    config.log_var_init = -2.5  # Start with smaller variance
    config.checkpoint_steps = (10000, 20000, 30000)
    
    # More conservative memory settings
    config.n_ctxs = 5000
    config.refresh_batch_size = 16
    config.out_batch_size = 512
    config.eval_batch_size = 32
    config.eval_n_batches = 5
    
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Training parameters optimized for 10GB GPU
        total_steps=20000,
        lr=8e-4,
        kl_coeff=0.01,
        sparsity_coeff=0.01,
        kl_warmup_steps=2000,
        
        # P-annealing
        p_start=1.0,
        p_end=0.4,
        anneal_start_frac=0.2,
        anneal_end_frac=0.8,
        
        # Model settings
        var_flag=0,
        use_april_update_mode=True,
        log_var_init=-2.0,
        
        # GPU memory optimized buffer settings
        n_ctxs=2500,           
        ctx_len=128,
        refresh_batch_size=12, 
        out_batch_size=192,    
        
        # Checkpointing
        checkpoint_steps=(60000,),
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
    """Create a configuration with more aggressive p-annealing."""
    config = create_full_config()
    
    # More aggressive p-annealing
    config.p_start = 1.5
    config.p_end = 0.2
    config.anneal_start_frac = 0.1  # Start earlier
    config.anneal_end_frac = 0.9   # End later
    config.sparsity_coeff = 0.2    # Higher sparsity coefficient
    
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEGatedAnneal Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "learned_variance",
            "gpu_10gb",
            "aggressive_annealing"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "learned_variance": create_learned_variance_config,
        "gpu_10gb": create_gpu_10gb_config,
        "aggressive_annealing": create_aggressive_annealing_config,
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
                
        # Show gating diagnostics if available
        gating_metrics = [k for k in results.keys() if "gate" in k.lower()]
        if gating_metrics:
            print("\nGating Diagnostics:")
            for metric in gating_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
                
        # Show KL diagnostics if available
        kl_metrics = [k for k in results.keys() if "kl" in k.lower()]
        if kl_metrics:
            print("\nKL Diagnostics:")
            for metric in kl_metrics:
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
# python train_vsae_gated_anneal.py --config quick_test
# python train_vsae_gated_anneal.py --config full
# python train_vsae_gated_anneal.py --config learned_variance
# python train_vsae_gated_anneal.py --config gpu_10gb
# python train_vsae_gated_anneal.py --config aggressive_annealing

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()