"""
Training script for VSAEMultiGaussian with multivariate Gaussian priors.

Key features:
- Multivariate Gaussian prior with configurable correlation structure
- Enhanced configuration management with correlation parameters
- KL annealing to prevent posterior collapse
- Improved evaluation and logging for multivariate case
- Multiple memory-efficient configuration presets
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
from dictionary_learning.trainers.vsae_multi import VSAEMultiGaussianTrainer, VSAEMultiGaussian, VSAEMultiConfig, VSAEMultiTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire VSAEMultiGaussian experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # VSAEMultiGaussian specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    corr_rate: float = 0.5  # Correlation rate for multivariate Gaussian prior
    corr_matrix: Optional[torch.Tensor] = None  # Custom correlation matrix
    use_april_update_mode: bool = True
    log_var_init: float = -2.0  # Initialize log_var around exp(-2) ≈ 0.135 variance
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    kl_warmup_steps: Optional[int] = None  # KL annealing steps
    
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
    wandb_project: str = "vsae-multi-experiments"
    
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
    """Manages the entire training experiment with VSAEMultiGaussian."""
    
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
                logging.FileHandler(log_dir / 'vsae_multi_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEMultiConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEMultiConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            corr_rate=self.config.corr_rate,
            corr_matrix=self.config.corr_matrix,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init
        )
        
    def create_training_config(self) -> VSAEMultiTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEMultiTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
        )
        
    def create_trainer_config(self, model_config: VSAEMultiConfig, training_config: VSAEMultiTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEMultiGaussianTrainer,
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
        corr_suffix = f"_corr{self.config.corr_rate:.2f}"
        kl_suffix = f"_kl_warmup{self.config.kl_warmup_steps}" if self.config.kl_warmup_steps else ""
        
        return (
            f"VSAEMulti_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}"
            f"{var_suffix}{corr_suffix}{kl_suffix}"
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
            # Handle torch tensors in corr_matrix
            if self.config.corr_matrix is not None:
                config_dict['corr_matrix'] = "custom_tensor"
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
        """Run the complete training experiment with VSAEMultiGaussian."""
        self.logger.info("Starting VSAEMultiGaussian training experiment")
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
            self.logger.info(f"Correlation rate: {model_config.corr_rate}")
            self.logger.info(f"KL warmup steps: {training_config.kl_warmup_steps}")
            
            # Log memory usage estimate
            memory_mb = model_config.estimate_memory_mb()
            self.logger.info(f"Estimated model memory usage: {memory_mb:.1f} MB")
            
            # Run training
            self.logger.info("Starting VSAEMultiGaussian training...")
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
                    "experiment_type": "vsae_multi_gaussian",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "var_flag": self.config.var_flag,
                    "corr_rate": self.config.corr_rate,
                    "kl_warmup_steps": self.config.kl_warmup_steps,
                    "features": [
                        "multivariate_gaussian_prior",
                        "configurable_correlation", 
                        "kl_annealing",
                        "unconstrained_mean_encoding",
                        "numerical_stability_fixes"
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
        """Evaluate the trained model with enhanced multivariate diagnostics."""
        self.logger.info("Evaluating trained VSAEMultiGaussian model...")
        
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
            
            # Add VSAEMultiGaussian-specific diagnostics
            try:
                # Get a sample batch for multivariate diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                kl_diagnostics = vsae.get_kl_diagnostics(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in kl_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                
                # Add memory usage info
                memory_usage = vsae.get_memory_usage()
                for key, value in memory_usage.items():
                    eval_results[f"memory_{key}_mb"] = value
                    
            except Exception as e:
                self.logger.warning(f"Could not compute multivariate diagnostics: {e}")
            
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
    """Create a configuration for quick testing with multivariate Gaussian prior."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Multivariate Gaussian parameters
        corr_rate=0.3,  # Moderate correlation for testing
        var_flag=0,  # Fixed variance for quick test
        
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


def create_independent_config() -> ExperimentConfig:
    """Create a configuration with independent Gaussian prior (for comparison)."""
    config = create_quick_test_config()
    config.corr_rate = 0.0  # Independent features
    config.total_steps = 5000
    config.kl_warmup_steps = 500
    config.n_ctxs = 2000
    config.checkpoint_steps = (5000,)
    return config


def create_correlated_config() -> ExperimentConfig:
    """Create a configuration with moderate correlation."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Multivariate Gaussian with correlation
        corr_rate=0.5,  # Moderate positive correlation
        var_flag=0,  # Fixed variance
        
        # Training parameters
        total_steps=15000,
        lr=5e-4,
        kl_coeff=300.0,  # Lower KL coeff with correlation
        kl_warmup_steps=1500,  # 10% of training
        
        # Memory efficient buffer settings
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=24,
        out_batch_size=512,
        
        # Checkpointing
        checkpoint_steps=(5000, 10000, 15000),
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


def create_anticorrelated_config() -> ExperimentConfig:
    """Create a configuration with negative correlation."""
    config = create_correlated_config()
    config.corr_rate = -0.3  # Negative correlation
    config.kl_coeff = 400.0  # Slightly higher KL coeff for anticorrelation
    return config


def create_learned_variance_correlated_config() -> ExperimentConfig:
    """Create a configuration with learned variance and correlation."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Multivariate Gaussian with learned variance
        corr_rate=0.4,  # Moderate correlation
        var_flag=1,  # Learned variance
        log_var_init=-2.5,  # Start with smaller variance
        
        # Training parameters for learned variance
        total_steps=20000,
        lr=3e-4,  # Lower LR for stability
        kl_coeff=200.0,  # Lower KL coeff with learned variance
        kl_warmup_steps=4000,  # 20% of training
        
        # Conservative memory settings
        n_ctxs=2000,
        ctx_len=128,
        refresh_batch_size=16,
        out_batch_size=384,
        
        # Checkpointing
        checkpoint_steps=(8000, 16000, 20000),
        log_steps=100,
        
        # Evaluation
        eval_batch_size=32,
        eval_n_batches=6,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=2.0,  # REDUCED: Correlation matrix scales as O(dict_size^2)
        
        # Multivariate Gaussian optimized for memory
        corr_rate=0.2,  # Lower correlation to reduce numerical issues
        var_flag=0,  # Fixed variance for memory efficiency
        
        # Training parameters optimized for 10GB GPU
        total_steps=10000,  # Shorter due to smaller model
        lr=6e-4,  # Slightly higher LR for smaller model
        kl_coeff=300.0,
        kl_warmup_steps=1000,  # 10% for KL annealing
        
        # GPU memory optimized buffer settings
        n_ctxs=3000,     # Can use more contexts with smaller dict
        ctx_len=128,
        refresh_batch_size=20,  # Slightly larger batches
        out_batch_size=400,     # Larger output batches
        
        # Checkpointing
        checkpoint_steps=(10000,),
        log_steps=100,
        
        # Evaluation - small and efficient
        eval_batch_size=48,
        eval_n_batches=6,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_gpu_10gb_independent_config() -> ExperimentConfig:
    """Create an independent (no correlation) config for 10GB GPU - most memory efficient."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,  # Can use full size with independence
        
        # Independent Gaussian (no correlation matrix overhead)
        corr_rate=0.0,  # Independent features - no correlation matrix!
        var_flag=0,  # Fixed variance
        
        # Training parameters
        total_steps=15000,
        lr=5e-4,
        kl_coeff=500.0,
        kl_warmup_steps=1500,
        
        # Can use larger buffers without correlation matrix
        n_ctxs=4000,
        ctx_len=128,
        refresh_batch_size=24,
        out_batch_size=512,
        
        # Checkpointing
        checkpoint_steps=(7500, 15000),
        log_steps=100,
        
        # Evaluation
        eval_batch_size=64,
        eval_n_batches=8,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_high_correlation_config() -> ExperimentConfig:
    """Create a configuration with high correlation (testing numerical stability)."""
    config = create_gpu_10gb_config()
    config.corr_rate = 0.8  # High correlation
    config.kl_coeff = 100.0  # Lower KL coeff due to high correlation
    config.total_steps = 10000
    config.kl_warmup_steps = 2000  # Longer warmup for stability
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEMultiGaussian Training")
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "independent",
            "correlated", 
            "anticorrelated",
            "learned_variance_correlated",
            "gpu_10gb",
            "gpu_10gb_independent",
            "high_correlation"
        ], 
        default="quick_test",
        help="Configuration preset for training"
    )
    args = parser.parse_args()
    
    config_functions = {
        "quick_test": create_quick_test_config,
        "independent": create_independent_config,
        "correlated": create_correlated_config,
        "anticorrelated": create_anticorrelated_config,
        "learned_variance_correlated": create_learned_variance_correlated_config,
        "gpu_10gb": create_gpu_10gb_config,
        "gpu_10gb_independent": create_gpu_10gb_independent_config,
        "high_correlation": create_high_correlation_config,
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
                
        # Show multivariate KL diagnostics if available
        kl_metrics = [k for k in results.keys() if "kl" in k.lower()]
        if kl_metrics:
            print("\nMultivariate KL Diagnostics:")
            for metric in kl_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
        
        # Show correlation effects
        correlation_metrics = [k for k in results.keys() if "correlation" in k.lower()]
        if correlation_metrics:
            print("\nCorrelation Effects:")
            for metric in correlation_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.4f}")
                
        # Show memory usage
        memory_metrics = [k for k in results.keys() if "memory" in k.lower()]
        if memory_metrics:
            print("\nMemory Usage:")
            for metric in memory_metrics:
                if metric in results:
                    print(f"{metric:<25} | {results[metric]:.2f}")
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
                
        print(f"\nFull results saved to experiment directory")
    else:
        print("No evaluation results available")


# Usage examples:
# python train_vsae_multi.py --config quick_test
# python train_vsae_multi.py --config independent          # No correlation (like VSAEIso)
# python train_vsae_multi.py --config correlated          # Positive correlation
# python train_vsae_multi.py --config anticorrelated      # Negative correlation  
# python train_vsae_multi.py --config learned_variance_correlated  # Learned variance + correlation
# python train_vsae_multi.py --config gpu_10gb           # Memory optimized with correlation
# python train_vsae_multi.py --config gpu_10gb_independent  # Most memory efficient (no correlation)
# python train_vsae_multi.py --config high_correlation   # Test numerical stability

# Note: Use gpu_10gb_independent for best memory efficiency on 10GB GPUs


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()