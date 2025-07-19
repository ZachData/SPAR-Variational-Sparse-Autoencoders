"""
Enhanced training script for VSAETopK with sweep functionality and optimized configurations.

Key improvements:
- Hyperparameter sweep support using BaseSweepRunner
- Multiple configuration presets including 10GB GPU optimized settings
- Enhanced evaluation and logging
- Better memory management and error handling
- Comprehensive argument parsing
- FIXED: All configurations now use residual stream (blocks.0.hook_resid_post)
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

# Import our VSAETopK implementations
from dictionary_learning.trainers.vsae_topk import (
    VSAETopK, 
    VSAETopKTrainer,
    VSAETopKConfig,
    VSAETopKTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment with enhanced TopK support."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.hook_resid_post"  # FIXED: Residual stream for saebench compatibility
    dict_size_multiple: float = 4.0
    k_fraction: float = 0.08  # Fraction of dictionary size for Top-K
    
    # Model-specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    threshold_beta: float = 0.999
    threshold_start_step: Optional[int] = None
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32  # TopK auxiliary loss coefficient
    
    # Schedule configuration
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start_step: Optional[int] = None
    
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
    wandb_project: str = "vsae-topk-experiments"
    
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
        # Set default step values based on total steps
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.total_steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.decay_start_step is None:
            decay_start = int(0.8 * self.total_steps)
            min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
            self.decay_start_step = max(decay_start, min_decay_start)
        if self.threshold_start_step is None:
            self.threshold_start_step = int(0.1 * self.total_steps)
    
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
    """Manages the entire training experiment with enhanced VSAETopK support."""
    
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
                logging.FileHandler(log_dir / 'vsae_topk_training.log'),
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
        
        self.logger.info(f"Model loaded. d_model: {model.cfg.d_model}")
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
        
        # Create activation buffer - FIXED: Uses d_model for residual stream
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=self.config.hook_name,
            d_submodule=model.cfg.d_model,  # FIXED: d_model (512) not d_mlp (2048) for residual stream
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> VSAETopKConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_model)
        k = int(self.config.k_fraction * dict_size)
        
        # Create base config - FIXED: Uses d_model for residual stream
        model_config = VSAETopKConfig(
            activation_dim=model.cfg.d_model,  # FIXED: d_model (512) for residual stream
            dict_size=dict_size,
            k=k,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
        # Add threshold parameters if they exist in the config class
        if hasattr(model_config, 'threshold_beta'):
            model_config.threshold_beta = self.config.threshold_beta
        
        return model_config
        
    def create_training_config(self) -> VSAETopKTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAETopKTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            auxk_alpha=self.config.auxk_alpha,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
            # Note: threshold_beta and threshold_start_step may not be in VSAETopKTrainingConfig
            # These might be model-level parameters instead
        )
        
    def create_trainer_config(self, model_config: VSAETopKConfig, training_config: VSAETopKTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        trainer_config = {
            "trainer": VSAETopKTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "wandb_name": self.get_experiment_name(),
            "submodule_name": self.config.hook_name,
            "seed": self.config.seed,
        }
        
        # Add threshold parameters if the trainer accepts them
        if hasattr(VSAETopKTrainer, '__init__'):
            # Add threshold parameters as separate config items
            trainer_config.update({
                "threshold_beta": self.config.threshold_beta,
                "threshold_start_step": self.config.threshold_start_step,
            })
        
        return trainer_config
        
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        k_value = int(self.config.k_fraction * self.config.dict_size_multiple * 512)  # FIXED: Using d_model=512
        var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"
        
        return (
            f"VSAETopK_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 512)}_"
            f"k{k_value}_lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"aux{self.config.auxk_alpha}{var_suffix}"
        )
        
    def get_save_directory(self) -> Path:
        """Get the save directory for this experiment."""
        save_dir = Path(self.config.save_dir) / self.get_experiment_name()
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
        
    def save_config(self, save_dir: Path) -> None:
        """Save experiment configuration."""
        config_path = save_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            import json
            json.dump(asdict(self.config), f, indent=2, default=str)
        self.logger.info(f"Saved experiment config to {config_path}")
        
    def run_training(self) -> Optional[Dict[str, float]]:
        """Run the complete training experiment."""
        start_time = time.time()
        
        try:
            # Setup
            model = self.load_model()
            buffer = self.create_buffer(model)
            model_config = self.create_model_config(model)
            training_config = self.create_training_config()
            trainer_config = self.create_trainer_config(model_config, training_config)
            
            # Save directory and config
            save_dir = self.get_save_directory()
            self.save_config(save_dir)
            
            # Log experiment details
            self.logger.info(f"Experiment: {self.get_experiment_name()}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}")
            self.logger.info(f"K value: {model_config.k}")
            self.logger.info(f"K fraction: {self.config.k_fraction}, Auxiliary alpha: {self.config.auxk_alpha}")
            self.logger.info(f"Save directory: {save_dir}")
            
            # Run training
            self.logger.info("Starting VSAETopK training...")
            
            trainSAE(
                data=buffer,
                trainer_configs=[trainer_config],  # FIXED: trainSAE expects a list of configs
                steps=self.config.total_steps,
                save_dir=str(save_dir),
                save_steps=list(self.config.checkpoint_steps) if self.config.checkpoint_steps else None,
                log_steps=self.config.log_steps,
                verbose=True,
                normalize_activations=True,
                autocast_dtype=self.config.get_autocast_dtype(),
                use_wandb=self.config.use_wandb,
                wandb_entity=self.config.wandb_entity,
                wandb_project=self.config.wandb_project,
                run_cfg={
                    "model_type": self.config.model_name,
                    "experiment_type": "vsae_topk",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k_fraction": self.config.k_fraction,
                    "var_flag": self.config.var_flag,
                    "auxk_alpha": self.config.auxk_alpha,
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
        """Run evaluation on the trained model."""
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
            
            # Add VSAETopK-specific diagnostics if possible
            try:
                # Get a sample batch for TopK diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                if hasattr(vsae, 'get_topk_diagnostics'):
                    topk_diagnostics = vsae.get_topk_diagnostics(sample_batch.to(self.config.device))
                    
                    # Add to eval results
                    for key, value in topk_diagnostics.items():
                        eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                        
            except Exception as e:
                self.logger.warning(f"Could not compute TopK diagnostics: {e}")
            
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


class VSAETopKSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for VSAETopK trainer.
    
    Optimizes TopK-specific parameters like k_fraction and auxk_alpha.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize VSAETopK sweep runner."""
        super().__init__(trainer_name="vsae-topk", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for VSAETopK.
        
        Focuses on TopK-specific parameters.
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
                    'min': 100.0,
                    'max': 1000.0
                },
                # K fraction: what fraction of dictionary to keep active
                'k_fraction': {
                    'distribution': 'uniform',
                    'min': 0.05,  # 5% of dictionary
                    'max': 0.15   # 15% of dictionary
                },
                # Auxiliary TopK loss coefficient
                'auxk_alpha': {
                    'distribution': 'log_uniform_values',
                    'min': 1/64,  # 0.015625
                    'max': 1/8    # 0.125
                },
                # Dictionary size multiplier: discrete choices
                'dict_size_multiple': {
                    'values': [4.0, 8.0]
                },
                # Variance flag: discrete choice
                'var_flag': {
                    'values': [0, 1]
                },
                # Note: Removed threshold_beta for now - may not be supported in current implementation
            }
        }
    
    def get_run_name(self, sweep_params: dict) -> str:
        """
        Generate descriptive run name from sweep parameters.
        """
        lr_str = f"{sweep_params['lr']:.1e}".replace('-0', '-')
        kl_str = f"{int(sweep_params['kl_coeff'])}"
        k_str = f"k{sweep_params['k_fraction']:.3f}"
        aux_str = f"aux{sweep_params['auxk_alpha']:.3f}"
        dict_str = f"{sweep_params['dict_size_multiple']:.0f}x"
        var_str = f"var{sweep_params['var_flag']}"
        
        return f"TOPK_lr{lr_str}_kl{kl_str}_{k_str}_{aux_str}_d{dict_str}_{var_str}"
    
    def create_experiment_config(self, sweep_params: dict) -> ExperimentConfig:
        """
        Create an ExperimentConfig from wandb sweep parameters.
        """
        # Start with the quick test config
        config = create_quick_test_config()
        
        # Override with sweep parameters
        config.lr = sweep_params['lr']
        config.kl_coeff = sweep_params['kl_coeff']
        config.k_fraction = sweep_params['k_fraction']
        config.auxk_alpha = sweep_params['auxk_alpha']
        config.dict_size_multiple = sweep_params['dict_size_multiple']
        config.var_flag = sweep_params['var_flag']
        # Note: threshold_beta removed from sweep for now
        
        # Adjust settings for sweep runs
        config.total_steps = 15000  # Moderate length for sweep
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-topk-sweeps"
        config.save_dir = "./temp_sweep_run"
        config.use_wandb = False  # Sweep handles wandb
        
        return config


def create_quick_test_config() -> ExperimentConfig:
    """Create a configuration for quick testing."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",  # FIXED: Residual stream
        dict_size_multiple=4.0,
        k_fraction=0.08,
        
        # Test parameters
        total_steps=1000,
        checkpoint_steps=list(),
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
    """Create a configuration for full training - memory efficient."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",  # FIXED: Residual stream
        dict_size_multiple=4.0,
        k_fraction=0.08,
        
        # Full training parameters
        total_steps=25000,  # Reduced from 50000
        lr=5e-4,
        kl_coeff=500.0,
        auxk_alpha=1/32,
        
        # Model settings
        var_flag=0,  # Start with fixed variance
        use_april_update_mode=True,
        threshold_beta=0.999,
        
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
    config = create_full_config()
    
    # Enable learned variance with adjusted settings
    config.var_flag = 1
    config.total_steps = 30000  # Slightly longer for learned variance
    config.lr = 3e-4  # Slightly lower LR for learned variance stability
    config.kl_coeff = 300.0  # Lower KL coeff since learned variance provides more flexibility
    config.auxk_alpha = 1/48  # Slightly lower auxiliary loss for stability
    
    # Smaller buffer for learned variance (uses more memory)
    config.n_ctxs = 5000
    config.refresh_batch_size = 16
    config.out_batch_size = 512
    config.checkpoint_steps = (10000, 20000, 30000)
    
    # Smaller evaluation for memory
    config.eval_batch_size = 32
    config.eval_n_batches = 5
    
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory with consistent speed."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",  # FIXED: Residual stream
        dict_size_multiple=4.0,
        k_fraction=0.125,
        
        # Training parameters optimized for 10GB GPU
        total_steps=20000,
        lr=8e-4,
        kl_coeff=10.0,
        auxk_alpha=1/32,
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
        use_april_update_mode=True,
        threshold_beta=0.999,
        
        # OPTIMIZED buffer settings for consistent speed
        n_ctxs=2500,     # Reduced to prevent memory pressure
        ctx_len=128,
        refresh_batch_size=12,  # Smaller batches for consistent memory usage
        out_batch_size=192,     # Smaller output batches
        
        # Checkpointing
        checkpoint_steps=(20000,),
        log_steps=1000,  # More frequent logging to monitor performance
        
        # Evaluation - smaller to reduce memory spikes
        eval_batch_size=24,
        eval_n_batches=4,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_gpu_10gb_learned_var_config() -> ExperimentConfig:
    """Create a learned variance configuration optimized for 10GB GPU."""
    config = create_gpu_10gb_config()
    
    # Enable learned variance with very conservative settings for speed
    config.var_flag = 1
    config.total_steps = 15000  # Shorter for learned variance
    config.lr = 3e-4  # Lower LR for stability
    config.kl_coeff = 300.0  # Lower KL coefficient
    config.auxk_alpha = 1/48  # Lower auxiliary coefficient
    config.checkpoint_steps = (15000,)
    
    # Extra conservative memory settings for learned variance
    config.n_ctxs = 1800      # Even smaller for learned variance
    config.refresh_batch_size = 10  # Very small batches
    config.out_batch_size = 150     # Very small output
    config.eval_batch_size = 16     # Tiny evaluation batches
    config.eval_n_batches = 3
    
    return config


def create_gpu_10gb_speed_optimized_config() -> ExperimentConfig:
    """Create a configuration optimized for maximum speed on 10GB GPU."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",  # FIXED: Residual stream
        dict_size_multiple=3.0,  # Smaller dictionary for speed
        k_fraction=0.06,         # Lower K for speed
        
        # Training parameters optimized for speed
        total_steps=25000,       # Can afford more steps with faster training
        lr=6e-4,                 # Slightly higher LR to compensate for smaller model
        kl_coeff=400.0,
        auxk_alpha=1/40,
        
        # Model settings
        var_flag=0,  # Fixed variance for speed
        use_april_update_mode=True,
        threshold_beta=0.999,
        
        # SPEED-OPTIMIZED buffer settings
        n_ctxs=2000,     # Smaller buffer for consistent memory usage
        ctx_len=128,
        refresh_batch_size=10,   # Very small batches to prevent memory spikes
        out_batch_size=160,      # Optimized for speed
        
        # Checkpointing
        checkpoint_steps=(25000,),
        log_steps=40,  # Frequent logging to monitor
        
        # Evaluation - minimal to save time
        eval_batch_size=20,
        eval_n_batches=3,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_high_k_config() -> ExperimentConfig:
    """Create a configuration with higher K fraction for comparison."""
    config = create_full_config()
    config.k_fraction = 0.15  # 15% instead of 8%
    config.auxk_alpha = 1/24   # Higher auxiliary loss for higher K
    return config


def create_low_k_config() -> ExperimentConfig:
    """Create a configuration with lower K fraction for sparsity."""
    config = create_full_config()
    config.k_fraction = 0.04  # 4% instead of 8%
    config.auxk_alpha = 1/48   # Lower auxiliary loss for lower K
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAETopK Training and Hyperparameter Sweeps")
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
            "gpu_10gb_speed",
            "high_k",
            "low_k"
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
        print("Starting hyperparameter sweep for VSAETopK...")
        print(f"Project: vsae-topk-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAETopKSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "learned_variance": create_learned_variance_config,
        "gpu_10gb": create_gpu_10gb_config,
        "gpu_10gb_learned_var": create_gpu_10gb_learned_var_config,
        "gpu_10gb_speed": create_gpu_10gb_speed_optimized_config,
        "high_k": create_high_k_config,
        "low_k": create_low_k_config,
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
                
        # Show TopK-specific diagnostics if available
        topk_metrics = [k for k in results.keys() if "topk" in k.lower() or "aux" in k.lower()]
        if topk_metrics:
            print("\nTopK Diagnostics:")
            for metric in topk_metrics:
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
# python train_vsae_topk.py --config quick_test
# python train_vsae_topk.py --config learned_variance  
# python train_vsae_topk.py --config gpu_10gb          # Balanced 10GB config
# python train_vsae_topk.py --config gpu_10gb_speed    # Speed-optimized 10GB config
# python train_vsae_topk.py --config high_k           # For comparison with higher K
# python train_vsae_topk.py --sweep --wandb-entity your-username


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()