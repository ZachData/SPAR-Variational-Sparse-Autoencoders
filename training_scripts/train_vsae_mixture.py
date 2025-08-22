"""
Improved training script for VSAEMixture following the same pattern as train_isovsae_1lgelu.py.

Key improvements:
- Clean dataclass-based configuration management
- ExperimentRunner class for better organization
- Multiple configuration presets
- Proper logging and error handling
- Hyperparameter sweep support
- Better memory management
- Command-line interface
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
from dictionary_learning.trainers.vsae_mixture import (
    VSAEMixtureTrainer, 
    VSAEMixtureGaussian, 
    VSAEMixtureConfig,
    VSAEMixtureTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire VSAEMixture experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # VSAEMixture-specific config
    var_flag: int = 0
    use_april_update_mode: bool = True
    n_correlated_pairs: int = 10
    n_anticorrelated_pairs: int = 10
    log_var_init: float = -2.0
    prior_std: float = 1.0
    
    # Training configuration
    total_steps: int = 30000
    lr: float = 5e-5
    kl_coeff: float = 50.0
    kl_warmup_steps: Optional[int] = None  # KL annealing
    resample_steps: Optional[int] = None  # Dead neuron resampling
    gradient_clip_norm: float = 1.0
    
    # Buffer configuration
    n_ctxs: int = 3000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024
    
    # Logging and saving
    checkpoint_steps: tuple = (30000,)
    log_steps: int = 100
    save_dir: str = "./experiments"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "vsae-mixture-experiments"
    
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
    """Manages the entire VSAEMixture training experiment."""
    
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
                logging.FileHandler(log_dir / 'vsae_mixture_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEMixtureConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEMixtureConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            n_correlated_pairs=self.config.n_correlated_pairs,
            n_anticorrelated_pairs=self.config.n_anticorrelated_pairs,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init,
            prior_std=self.config.prior_std,
        )
        
    def create_training_config(self) -> VSAEMixtureTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEMixtureTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
            resample_steps=self.config.resample_steps,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )
        
    def create_trainer_config(self, model_config: VSAEMixtureConfig, training_config: VSAEMixtureTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEMixtureTrainer,
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
            f"VSAEMix_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"cor{self.config.n_correlated_pairs}_"
            f"anticor{self.config.n_anticorrelated_pairs}"
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
        self.logger.info("Starting VSAEMixture training experiment")
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
            self.logger.info(f"Correlated pairs: {model_config.n_correlated_pairs}")
            self.logger.info(f"Anticorrelated pairs: {model_config.n_anticorrelated_pairs}")
            
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
                    "experiment_type": "vsae_mixture",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "n_correlated_pairs": self.config.n_correlated_pairs,
                    "n_anticorrelated_pairs": self.config.n_anticorrelated_pairs,
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


class VSAEMixtureSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for VSAEMixture trainer.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize VSAEMixture sweep runner."""
        super().__init__(trainer_name="vsae-mixture", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for VSAEMixture.
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
                    'min': 1e-6,
                    'max': 1e-3
                },
                # KL coefficient: uniform distribution
                'kl_coeff': {
                    'distribution': 'uniform',
                    'min': 10.0,
                    'max': 200.0
                },
                # Dictionary size multiplier: discrete choices
                'dict_size_multiple': {
                    'values': [4.0, 8.0]
                },
                # Correlation structure parameters
                'n_correlated_pairs': {
                    'values': [0, 5, 10, 20]
                },
                'n_anticorrelated_pairs': {
                    'values': [0, 5, 10, 20]
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
                },
                # Prior standard deviation
                'prior_std': {
                    'distribution': 'uniform',
                    'min': 0.5,
                    'max': 2.0
                },
                # KL warmup fraction
                'kl_warmup_fraction': {
                    'distribution': 'uniform',
                    'min': 0.05,  # 5% of training
                    'max': 0.25   # 25% of training
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
        cor_str = f"c{sweep_params['n_correlated_pairs']}"
        anticor_str = f"a{sweep_params['n_anticorrelated_pairs']}"
        var_str = f"v{sweep_params['var_flag']}"
        prior_str = f"p{sweep_params['prior_std']:.1f}"
        
        return f"Mix_lr{lr_str}_kl{kl_str}_d{dict_str}_{cor_str}_{anticor_str}_{var_str}_{prior_str}"
    
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
        config.n_correlated_pairs = sweep_params['n_correlated_pairs']
        config.n_anticorrelated_pairs = sweep_params['n_anticorrelated_pairs']
        config.var_flag = sweep_params['var_flag']
        config.log_var_init = sweep_params['log_var_init']
        config.prior_std = sweep_params['prior_std']
        
        # Set KL warmup steps based on fraction
        config.kl_warmup_steps = int(sweep_params['kl_warmup_fraction'] * config.total_steps)
        
        # Adjust settings for sweep runs
        config.total_steps = 15000
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-mixture-sweeps"
        config.save_dir = "./temp_sweep_run"
        config.use_wandb = False  # Sweep handles wandb
        
        return config


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
        
        # Small buffer for testing
        n_ctxs=500,
        refresh_batch_size=16,
        out_batch_size=128,
        
        # Correlation structure
        n_correlated_pairs=5,
        n_anticorrelated_pairs=5,
        
        # VSAEMixture specific
        var_flag=0,
        log_var_init=-2.0,
        prior_std=1.0,
        
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
        total_steps=30000,
        lr=5e-5,
        kl_coeff=50.0,
        kl_warmup_steps=3000,  # 10% of training for KL annealing
        
        # Model settings
        var_flag=0,  # Fixed variance
        use_april_update_mode=True,
        n_correlated_pairs=10,
        n_anticorrelated_pairs=10,
        log_var_init=-2.0,
        prior_std=1.0,
        
        # Buffer settings
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        
        # Checkpointing
        checkpoint_steps=(30000,),
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


def create_high_correlation_config() -> ExperimentConfig:
    """Create a configuration with high correlation structure."""
    config = create_full_config()
    config.n_correlated_pairs = 50
    config.n_anticorrelated_pairs = 50
    config.kl_coeff = 25.0  # Lower KL coeff for complex structure
    config.prior_std = 1.5  # Slightly higher prior std for more flexibility
    return config


def create_learned_variance_config() -> ExperimentConfig:
    """Create a configuration for training with learned variance."""
    config = create_full_config()
    config.var_flag = 1
    config.lr = 3e-5  # Lower LR for stability
    config.kl_coeff = 30.0  # Lower KL coeff
    config.log_var_init = -2.5  # Start with smaller variance
    config.total_steps = 40000  # Longer training
    config.kl_warmup_steps = 8000  # 20% of training
    config.checkpoint_steps = (20000, 40000)
    return config


def create_with_resampling_config() -> ExperimentConfig:
    """Create a configuration with dead neuron resampling enabled."""
    config = create_full_config()
    config.resample_steps = 5000  # Resample every 5k steps
    config.total_steps = 35000  # Slightly longer to see resampling effects
    config.kl_warmup_steps = 3500  # 10% of training
    config.checkpoint_steps = (15000, 35000)
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
        lr=5e-5,
        kl_coeff=50.0,
        kl_warmup_steps=2000,  # 10% for KL annealing
        
        # Model settings
        var_flag=0,
        use_april_update_mode=True,
        n_correlated_pairs=10,
        n_anticorrelated_pairs=10,
        log_var_init=-2.0,
        prior_std=1.0,
        
        # GPU memory optimized buffer settings
        n_ctxs=2000,     # Small enough to fit in 10GB
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
    
    parser = argparse.ArgumentParser(description="VSAEMixture Training and Hyperparameter Sweeps")
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
            "high_correlation",
            "learned_variance",
            "with_resampling",
            "gpu_10gb"
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
        print("Starting hyperparameter sweep for VSAEMixture...")
        print(f"Project: vsae-mixture-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAEMixtureSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "high_correlation": create_high_correlation_config,
        "learned_variance": create_learned_variance_config,
        "with_resampling": create_with_resampling_config,
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

# Usage examples:
# python train_vsae_mixture.py --config quick_test
# python train_vsae_mixture.py --config full
# python train_vsae_mixture.py --config high_correlation
# python train_vsae_mixture.py --config learned_variance
# python train_vsae_mixture.py --config with_resampling
# python train_vsae_mixture.py --config gpu_10gb
# python train_vsae_mixture.py --sweep --wandb-entity your-username