"""
Fixed training script for VSAEIso with all improvements.

Key improvements:
- Uses the fixed VSAEIso implementation
- Enhanced configuration management
- Better KL annealing parameters
- Improved evaluation and logging
- More robust error handling
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
from dictionary_learning.base_sweep import BaseSweepRunner
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
from dictionary_learning.trainers.vsae_iso import VSAEIsoTrainer, VSAEIsoGaussian, VSAEIsoConfig, VSAEIsoTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment with enhanced KL annealing support."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # Model-specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # NEW: KL annealing steps
    
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
    wandb_project: str = "vsae-experiments"  # Updated project name
    
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
    """Manages the entire training experiment with the fixed VSAEIso implementation."""
    
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
                logging.FileHandler(log_dir / 'vsae_iso_fixed_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEIsoConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEIsoConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init
        )
        
    def create_training_config(self) -> VSAEIsoTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEIsoTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,  # NEW: KL annealing
        )
        
    def create_trainer_config(self, model_config: VSAEIsoConfig, training_config: VSAEIsoTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEIsoTrainer,
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
        kl_suffix = f"_kl_warmup{self.config.kl_warmup_steps}" if self.config.kl_warmup_steps else ""
        
        return (
            f"VSAEIso_FIXED_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}"
            f"{var_suffix}{kl_suffix}"
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
        """Run the complete training experiment with fixed VSAEIso."""
        self.logger.info("Starting FIXED VSAEIso training experiment")
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
            
            # Run training
            self.logger.info("Starting training with FIXED implementation...")
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
                    "experiment_type": "vsae_iso_fixed",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "var_flag": self.config.var_flag,
                    "kl_warmup_steps": self.config.kl_warmup_steps,
                    "fixes_applied": [
                        "removed_relu_from_log_var",
                        "removed_decoder_norm_weighting", 
                        "added_kl_annealing",
                        "improved_numerical_stability"
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
            
            # Add VSAEIso-specific diagnostics if possible
            try:
                # Get a sample batch for KL diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                kl_diagnostics = vsae.get_kl_diagnostics(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in kl_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                    
            except Exception as e:
                self.logger.warning(f"Could not compute KL diagnostics: {e}")
            
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


class VSAEIsoFixedSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for the FIXED VSAEIso trainer.
    
    Enhanced to include KL annealing parameters in the search space.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize fixed VSAEIso sweep runner."""
        super().__init__(trainer_name="vsae-iso-fixed", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for fixed VSAEIso.
        
        Enhanced to include KL annealing parameters.
        """
        return {
            'method': 'bayes',
            'metric': {
                'goal': 'minimize', 
                'name': 'mse_loss'
            },
            'parameters': {
                # Learning rate: log-uniform distribution
                'lr': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-2
                },
                # KL coefficient: uniform distribution
                'kl_coeff': {
                    'distribution': 'uniform',
                    'min': 1.0,
                    'max': 1000.0
                },
                # KL warmup fraction: what fraction of training to use for KL annealing
                'kl_warmup_fraction': {
                    'distribution': 'uniform',
                    'min': 0.05,  # 5% of training
                    'max': 0.3    # 30% of training
                },
                # Dictionary size multiplier: discrete choices
                'dict_size_multiple': {
                    'values': [4.0]
                },
                # Variance flag: discrete choice
                'var_flag': {
                    'values': [0, 1]
                },
                # Log var initialization (only relevant when var_flag=1)
                'log_var_init': {
                    'distribution': 'uniform',
                    'min': -3.0,
                    'max': -1.0
                }
            }
        }
    
    def get_run_name(self, sweep_params: dict) -> str:
        """
        Generate descriptive run name from sweep parameters.
        """
        lr_str = f"{sweep_params['lr']:.1e}".replace('-0', '-')
        kl_str = f"{int(sweep_params['kl_coeff'])}"
        dict_str = f"{sweep_params['dict_size_multiple']:.0f}x"
        var_str = f"var{sweep_params['var_flag']}"
        kl_warmup_str = f"klw{sweep_params['kl_warmup_fraction']:.2f}"
        
        return f"FIXED_lr{lr_str}_kl{kl_str}_d{dict_str}_{var_str}_{kl_warmup_str}"
    
    def create_experiment_config(self, sweep_params: dict) -> ExperimentConfig:
        """
        Create an ExperimentConfig from wandb sweep parameters.
        """
        # Start with the quick test config
        config = create_quick_test_config()
        
        # Override with sweep parameters
        config.lr = sweep_params['lr']
        config.kl_coeff = sweep_params['kl_coeff']
        config.dict_size_multiple = sweep_params['dict_size_multiple']
        config.var_flag = sweep_params['var_flag']
        config.log_var_init = sweep_params['log_var_init']
        
        # Set KL warmup steps based on fraction
        config.kl_warmup_steps = int(sweep_params['kl_warmup_fraction'] * config.total_steps)
        
        # Adjust settings for sweep runs
        config.total_steps = 15000  # Longer to see KL annealing effects
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-iso-fixed-sweeps"
        config.save_dir = "./temp_sweep_run"
        config.use_wandb = False  # Sweep handles wandb
        
        return config


def create_quick_test_config() -> ExperimentConfig:
    """Create a configuration for quick testing with KL annealing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Test parameters with KL annealing
        total_steps=1000,
        checkpoint_steps=list(),
        log_steps=50,
        kl_warmup_steps=100,  # 10% of training for KL annealing
        
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
    """Create a configuration for full training with proper KL annealing - memory efficient."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Full training parameters with KL annealing
        total_steps=25000,  # Reduced from 50000
        lr=5e-4,
        kl_coeff=500.0,
        kl_warmup_steps=2500,  # 10% of training for KL annealing
        
        # Model settings
        var_flag=0,  # Start with fixed variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        
        # MEMORY EFFICIENT buffer settings
        n_ctxs=8000,   # Reduced from 30000
        ctx_len=128,
        refresh_batch_size=24,  # Smaller batches
        out_batch_size=768,     # Smaller output batches
        
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
    """Create a configuration for training with learned variance - memory efficient version."""
    config = ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Training parameters optimized for learned variance
        total_steps=30000,  # Reduced from 75000 for faster iteration
        lr=3e-4,  # Slightly lower LR for learned variance stability
        kl_coeff=300.0,  # Lower KL coeff since learned variance provides more flexibility
        kl_warmup_steps=6000,  # 20% of training for KL annealing
        
        # Model settings for learned variance
        var_flag=1,  # Enable learned variance
        use_april_update_mode=True,
        log_var_init=-2.5,  # Start with smaller variance for stability
        
        # MEMORY EFFICIENT buffer settings (key fix!)
        n_ctxs=5000,  # Much smaller than 30000
        ctx_len=128,
        refresh_batch_size=16,  # Smaller batches
        out_batch_size=512,  # Smaller output batches
        
        # Checkpointing
        checkpoint_steps=(10000, 20000, 30000),
        log_steps=100,
        
        # Evaluation - smaller to save memory
        eval_batch_size=32,  # Reduced from 64
        eval_n_batches=5,   # Reduced from 10
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )
    return config


def create_minimal_kl_annealing_config() -> ExperimentConfig:
    """Create a configuration with minimal KL annealing for comparison."""
    config = create_full_config()
    config.kl_warmup_steps = 1000  # Very short KL annealing
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Training parameters optimized for 10GB GPU
        total_steps=30000,
        lr=4e-5,
        kl_coeff=100.0,
        kl_warmup_steps=2000,  # 10% for KL annealing
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
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


def create_gpu_10gb_learned_var_config() -> ExperimentConfig:
    """Create a learned variance configuration optimized for 10GB GPU."""
    config = create_gpu_10gb_config()
    
    # Enable learned variance with conservative settings
    config.var_flag = 1
    config.total_steps = 60000  # Even shorter for learned variance
    config.kl_warmup_steps = 3000  # 20% for learned variance
    config.lr = 4e-4  # Lower LR for stability
    config.kl_coeff = 50.0  # Lower KL coefficient
    config.log_var_init = -3.0  # Start with very small variance
    config.checkpoint_steps = (15000,)
    
    # Even more conservative memory settings
    config.n_ctxs = 2000
    config.refresh_batch_size = 12
    config.out_batch_size = 192
    config.eval_batch_size = 24
    config.eval_n_batches = 3
    
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed VSAEIso Training and Hyperparameter Sweeps")
    parser.add_argument(
        "--sweep", 
        action="store_true", 
        help="Run hyperparameter sweep instead of single training"
    )
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "full", 
            "learned_variance",
            "gpu_10gb",
            "gpu_10gb_learned_var",
            "minimal_kl_annealing",
            "no_kl_annealing"
        ], 
        default="quick_test",
        help="Configuration preset for single training runs"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="zachdata",
        help="WandB entity/username for sweep logging"
    )
    args = parser.parse_args()
    
    if args.sweep:
        # Run hyperparameter sweep
        print("Starting hyperparameter sweep for FIXED VSAEIso...")
        print(f"Project: vsae-iso-fixed-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAEIsoFixedSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "learned_variance": create_learned_variance_config,
        "gpu_10gb": create_gpu_10gb_config,
        "gpu_10gb_learned_var": create_gpu_10gb_learned_var_config,
        "minimal_kl_annealing": create_minimal_kl_annealing_config,
        # "no_kl_annealing": create_no_kl_annealing_config,
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
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
    else:
        print("No evaluation results available")


# Usage examples:
# python train_isovsae_1lgelu.py --config quick_test
# python train_isovsae_1lgelu.py --config learned_variance
# python train_isovsae_1lgelu.py --config no_kl_annealing    # For comparison
# python train_isovsae_1lgelu.py --sweep --wandb-entity your-username


'''# Quick test (should work fine)
python train_isovsae_1lgelu.py --config quick_test

# Optimized for 10GB GPU - fixed variance
python train_isovsae_1lgelu.py --config gpu_10gb

# Optimized for 10GB GPU - learned variance  
python train_isovsae_1lgelu.py --config gpu_10gb_learned_var

# Original full config (now smaller but still substantial)
python train_isovsae_1lgelu.py --config full

# Original learned variance (much more memory efficient now)
python train_isovsae_1lgelu.py --config learned_variance
'''

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()