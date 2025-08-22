"""
Training script for VSAEGated with comprehensive configuration management.

This script provides a clean interface for training VSAEGated models
with different configurations and robust error handling.
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
from dictionary_learning.trainers.vsae_gated import VSAEGatedTrainer, VSAEGated, VSAEGatedConfig, VSAEGatedTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for VSAEGated training experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # VSAEGated-specific config
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0
    init_strategy: str = "tied"  # "tied", "independent", "xavier"
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    l1_penalty: float = 0.1
    aux_weight: float = 0.1
    kl_warmup_steps: Optional[int] = None
    use_constrained_optimizer: bool = True
    temperature_schedule: str = "constant"  # "constant", "linear_decay", "cosine_decay"
    
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
    wandb_project: str = "vsae-gated-experiments"
    
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
        # Set default KL warmup steps if not provided
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.total_steps)  # 10% of training
    
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
    """Manages VSAEGated training experiments."""
    
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
                logging.FileHandler(log_dir / 'vsae_gated_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEGatedConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEGatedConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init,
            init_strategy=self.config.init_strategy
        )
        
    def create_training_config(self) -> VSAEGatedTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEGatedTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            l1_penalty=self.config.l1_penalty,
            aux_weight=self.config.aux_weight,
            kl_warmup_steps=self.config.kl_warmup_steps,
            use_constrained_optimizer=self.config.use_constrained_optimizer,
            temperature_schedule=self.config.temperature_schedule,
        )
        
    def create_trainer_config(self, model_config: VSAEGatedConfig, training_config: VSAEGatedTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEGatedTrainer,
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
        temp_suffix = f"_{self.config.temperature_schedule}_temp" if self.config.temperature_schedule != "constant" else ""
        kl_suffix = f"_kl_warmup{self.config.kl_warmup_steps}" if self.config.kl_warmup_steps else ""
        
        return (
            f"VSAEGated_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"l1{self.config.l1_penalty}_aux{self.config.aux_weight}"
            f"{var_suffix}{temp_suffix}{kl_suffix}"
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
        self.logger.info("Starting VSAEGated training experiment")
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
            self.logger.info(f"KL warmup steps: {training_config.kl_warmup_steps}")
            self.logger.info(f"Using constrained optimizer: {training_config.use_constrained_optimizer}")
            self.logger.info(f"Temperature schedule: {training_config.temperature_schedule}")
            
            # Run training
            self.logger.info("Starting VSAEGated training...")
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
                    "experiment_type": "vsae_gated",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "var_flag": self.config.var_flag,
                    "kl_warmup_steps": self.config.kl_warmup_steps,
                    "l1_penalty": self.config.l1_penalty,
                    "aux_weight": self.config.aux_weight,
                    "temperature_schedule": self.config.temperature_schedule,
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
            
            # Add VSAEGated-specific diagnostics if possible
            try:
                # Get a sample batch for diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                sample_batch = sample_batch.to(self.config.device)
                
                # Get KL diagnostics
                kl_diagnostics = vsae.get_kl_diagnostics(sample_batch)
                
                # Get reconstruction diagnostics
                recon_diagnostics = vsae.get_reconstruction_diagnostics(sample_batch)
                
                # Add to eval results
                for key, value in {**kl_diagnostics, **recon_diagnostics}.items():
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
    """Create a configuration for quick testing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Test parameters
        total_steps=1000,
        checkpoint_steps=(),
        log_steps=50,
        kl_warmup_steps=100,  # 10% of training for KL annealing

        # Training parameters optimized for 10GB GPU
        lr=5e-5,
        kl_coeff=50.0,
        l1_penalty=0.01,
        aux_weight=0.01,

        
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
        
        # Full training parameters
        total_steps=25000,
        lr=5e-4,
        kl_coeff=500.0,
        l1_penalty=0.1,
        aux_weight=0.1,
        kl_warmup_steps=2500,  # 10% of training for KL annealing
        
        # Model settings
        var_flag=1,  # Use learned variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        init_strategy="tied",
        use_constrained_optimizer=True,
        temperature_schedule="constant",
        
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


def create_fixed_variance_config() -> ExperimentConfig:
    """Create a configuration for training with fixed variance."""
    config = create_full_config()
    config.var_flag = 0  # Fixed variance
    config.kl_coeff = 100.0  # Lower KL coefficient for fixed variance
    config.kl_warmup_steps = 1000  # Shorter warmup
    return config


def create_temperature_annealing_config() -> ExperimentConfig:
    """Create a configuration with temperature annealing."""
    config = create_full_config()
    config.temperature_schedule = "linear_decay"
    config.total_steps = 30000  # Longer training for temperature annealing
    config.kl_warmup_steps = 3000
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
        lr=5e-4,
        kl_coeff=500.0,
        l1_penalty=0.1,
        aux_weight=0.1,
        kl_warmup_steps=2000,  # 10% for KL annealing
        
        # Model settings
        var_flag=1,  # Learned variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        init_strategy="tied",
        use_constrained_optimizer=True,
        
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


def create_high_sparsity_config() -> ExperimentConfig:
    """Create a configuration for high sparsity training."""
    config = create_full_config()
    config.l1_penalty = 0.2  # Higher L1 penalty for more sparsity
    config.aux_weight = 0.2  # Higher auxiliary weight to help gate network
    config.kl_coeff = 300.0  # Lower KL coefficient to balance sparsity
    return config


def create_no_auxiliary_config() -> ExperimentConfig:
    """Create a configuration without auxiliary loss."""
    config = create_full_config()
    config.aux_weight = 0.0  # No auxiliary loss
    config.l1_penalty = 0.05  # Lower L1 penalty to compensate
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEGated Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "fixed_variance",
            "temperature_annealing",
            "gpu_10gb",
            "high_sparsity",
            "no_auxiliary"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    # Configuration functions
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "fixed_variance": create_fixed_variance_config,
        "temperature_annealing": create_temperature_annealing_config,
        "gpu_10gb": create_gpu_10gb_config,
        "high_sparsity": create_high_sparsity_config,
        "no_auxiliary": create_no_auxiliary_config,
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
                
        # Show KL diagnostics if available
        kl_metrics = [k for k in results.keys() if "kl" in k.lower()]
        if kl_metrics:
            print("\nKL Diagnostics:")
            for metric in kl_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
        
        # Show gating diagnostics if available
        gate_metrics = [k for k in results.keys() if "gate" in k.lower() or "sparsity" in k.lower()]
        if gate_metrics:
            print("\nGating/Sparsity Diagnostics:")
            for metric in gate_metrics:
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
# python train_vsae_gated.py --config quick_test
# python train_vsae_gated.py --config full
# python train_vsae_gated.py --config fixed_variance
# python train_vsae_gated.py --config temperature_annealing
# python train_vsae_gated.py --config gpu_10gb
# python train_vsae_gated.py --config high_sparsity
# python train_vsae_gated.py --config no_auxiliary


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()
