"""
Enhanced training script for VSAEBatchTopK with sweeps, memory optimization, and improved practices.

Key improvements:
- Hyperparameter sweep support using BaseSweepRunner
- Multiple configuration presets including 10GB GPU optimized settings
- Enhanced evaluation and logging
- Better memory management and error handling
- Consistent structure with train_isovsae_1lgelu.py
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

# Import our VSAEBatchTopK implementations
from dictionary_learning.trainers.vsae_batch_topk import (
    VSAEBatchTopK, 
    VSAEBatchTopKTrainer,
    VSAEBatchTopKConfig,
    TrainingConfig
)


@dataclass
class ExperimentConfig:
    """Enhanced configuration for the entire experiment with memory optimization support."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # VSAEBatchTopK specific configuration
    k_ratio: float = 0.05  # Fraction of dictionary size to keep active
    auxk_alpha: float = 1.0  # Auxiliary loss coefficient
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    constrain_decoder: bool = False
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    
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
    wandb_project: str = "vsae-batchtopk-experiments"
    
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
            self.warmup_steps = max(200, int(0.05 * self.total_steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.decay_start_step is None:
            self.decay_start_step = int(0.8 * self.total_steps)
    
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
    """Manages the entire training experiment with the VSAEBatchTopK implementation."""
    
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
                logging.FileHandler(log_dir / 'vsae_batchtopk_training.log'),
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
        
        # Determine model dimension based on hook name
        if "mlp" in self.config.hook_name:
            d_submodule = model.cfg.d_mlp
        else:
            d_submodule = model.cfg.d_model
        
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
            d_submodule=d_submodule,
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> VSAEBatchTopKConfig:
        """Create model configuration from experiment config."""
        # Determine model dimension based on hook name
        if "mlp" in self.config.hook_name:
            activation_dim = model.cfg.d_mlp
        else:
            activation_dim = model.cfg.d_model
            
        dict_size = int(self.config.dict_size_multiple * activation_dim)
        k = max(1, int(self.config.k_ratio * dict_size))
        
        return VSAEBatchTopKConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            var_flag=self.config.var_flag,
            constrain_decoder=self.config.constrain_decoder,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
    def create_training_config(self) -> TrainingConfig:
        """Create training configuration from experiment config."""
        return TrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            auxk_alpha=self.config.auxk_alpha,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
        )
        
    def create_trainer_config(self, model_config: VSAEBatchTopKConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEBatchTopKTrainer,
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
        decoder_suffix = "_constrained" if self.config.constrain_decoder else "_unconstrained"
        
        return (
            f"VSAEBatchTopK_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"k{self.config.k_ratio}_lr{self.config.lr}_"
            f"kl{self.config.kl_coeff}_aux{self.config.auxk_alpha}"
            f"{var_suffix}{decoder_suffix}"
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
        """Run the complete training experiment with VSAEBatchTopK."""
        self.logger.info("Starting VSAEBatchTopK training experiment")
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
            self.logger.info(f"K (active features): {model_config.k}")
            self.logger.info(f"K ratio: {self.config.k_ratio}")
            
            # Run training
            self.logger.info("Starting training with VSAEBatchTopK...")
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
                    "experiment_type": "vsae_batch_topk",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k_ratio": self.config.k_ratio,
                    "var_flag": self.config.var_flag,
                    "constrain_decoder": self.config.constrain_decoder,
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
            
            # Add VSAEBatchTopK-specific diagnostics if possible
            try:
                # Get a sample batch for additional diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                # Try to get batch top-k specific metrics
                if hasattr(vsae, 'get_topk_diagnostics'):
                    topk_diagnostics = vsae.get_topk_diagnostics(sample_batch.to(self.config.device))
                    
                    # Add to eval results
                    for key, value in topk_diagnostics.items():
                        eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                        
            except Exception as e:
                self.logger.warning(f"Could not compute batch top-k diagnostics: {e}")
            
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


class VSAEBatchTopKSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for VSAEBatchTopK trainer.
    
    Optimizes key hyperparameters including k_ratio, auxk_alpha, and threshold parameters.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize VSAEBatchTopK sweep runner."""
        super().__init__(trainer_name="vsae-batchtopk", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for VSAEBatchTopK.
        
        Focuses on key hyperparameters specific to batch top-k sparse autoencoders.
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
                # K ratio: key parameter for batch top-k
                'k_ratio': {
                    'distribution': 'uniform',
                    'min': 0.02,  # 2% of dictionary
                    'max': 0.1    # 10% of dictionary
                },
                # Auxiliary loss coefficient: specific to batch top-k
                'auxk_alpha': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 2.0
                },
                # Dictionary size multiplier
                'dict_size_multiple': {
                    'values': [4.0, 8.0]
                },
                # Variance flag
                'var_flag': {
                    'values': [0, 1]
                },
                # Decoder constraint
                'constrain_decoder': {
                    'values': [True, False]
                }
            }
        }
    
    def get_run_name(self, sweep_params: dict) -> str:
        """Generate descriptive run name from sweep parameters."""
        lr_str = f"{sweep_params['lr']:.1e}".replace('-0', '-')
        kl_str = f"{int(sweep_params['kl_coeff'])}"
        k_str = f"{sweep_params['k_ratio']:.3f}"
        aux_str = f"{sweep_params['auxk_alpha']:.2f}"
        dict_str = f"{sweep_params['dict_size_multiple']:.0f}x"
        var_str = f"var{sweep_params['var_flag']}"
        dec_str = "constrained" if sweep_params['constrain_decoder'] else "free"
        
        return f"BatchTopK_lr{lr_str}_kl{kl_str}_k{k_str}_aux{aux_str}_d{dict_str}_{var_str}_{dec_str}"
    
    def create_experiment_config(self, sweep_params: dict) -> ExperimentConfig:
        """Create an ExperimentConfig from wandb sweep parameters."""
        # Start with the quick test config
        config = create_quick_test_config()
        
        # Override with sweep parameters
        config.lr = sweep_params['lr']
        config.kl_coeff = sweep_params['kl_coeff']
        config.k_ratio = sweep_params['k_ratio']
        config.auxk_alpha = sweep_params['auxk_alpha']
        config.dict_size_multiple = sweep_params['dict_size_multiple']
        config.var_flag = sweep_params['var_flag']
        config.constrain_decoder = sweep_params['constrain_decoder']
        
        # Adjust settings for sweep runs
        config.total_steps = 15000  # Sufficient to see convergence
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency during sweeps
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-batchtopk-sweeps"
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
        k_ratio=0.05,
        
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
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        k_ratio=0.05,
        
        # Full training parameters
        total_steps=25000,  # Reduced from 50000 for faster iteration
        lr=5e-4,
        kl_coeff=500.0,
        auxk_alpha=1.0,
        
        # Batch Top-K specific settings
        var_flag=0,
        constrain_decoder=False,
        
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
    """Create a configuration for training with learned variance - memory efficient."""
    config = create_full_config()
    
    # Enable learned variance with optimized settings
    config.var_flag = 1
    config.total_steps = 30000  # Slightly longer for learned variance
    config.lr = 3e-4  # Lower LR for stability with learned variance
    config.kl_coeff = 300.0  # Lower KL coeff since learned variance provides flexibility
    config.auxk_alpha = 0.8  # Slightly lower auxiliary loss
    
    # More conservative memory settings for learned variance
    config.n_ctxs = 5000
    config.refresh_batch_size = 16
    config.out_batch_size = 512
    config.eval_batch_size = 32
    config.eval_n_batches = 5
    
    config.checkpoint_steps = (10000, 20000, 30000)
    
    return config


def create_gpu_10gb_config() -> ExperimentConfig:
    """Create a configuration optimized for 10GB GPU memory with better performance."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        k_ratio=0.008,
        
        # Training parameters optimized for 10GB GPU - FASTER
        total_steps=60000,  # Reduced from 20000 for faster completion
        lr=8e-4,  # Slightly higher LR for faster convergence
        kl_coeff=0.01,  # Reduced KL for faster convergence
        auxk_alpha=0.8,  # Slightly lower auxiliary loss
        
        # Model settings optimized for memory and speed
        var_flag=0,  # Fixed variance for memory efficiency
        constrain_decoder=False,  # Skip expensive decoder normalization
        
        # MEMORY & SPEED OPTIMIZED buffer settings
        n_ctxs=2500,     # Reduced from 3000 - less memory pressure
        ctx_len=128,
        refresh_batch_size=12,  # Smaller refresh batches for consistent speed
        out_batch_size=192,     # Smaller output batches to reduce memory spikes
        
        # More frequent checkpointing for safety but less total
        checkpoint_steps=(60000,),  # Only final checkpoint
        log_steps=100,  # Less frequent logging to reduce overhead
        
        # Evaluation - faster and more memory efficient
        eval_batch_size=24,  # Smaller eval batches
        eval_n_batches=3,    # Fewer eval batches
        
        # System settings for performance
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",    # Memory efficient
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_gpu_10gb_learned_var_config() -> ExperimentConfig:
    """Create a learned variance configuration optimized for 10GB GPU - FASTER."""
    config = create_gpu_10gb_config()
    
    # Enable learned variance with very conservative and FAST settings
    config.var_flag = 1
    config.total_steps = 12000  # Even shorter for learned variance
    config.lr = 4e-4  # Slightly lower LR for stability
    config.kl_coeff = 150.0  # Much lower KL coefficient for faster convergence
    config.auxk_alpha = 0.4  # Lower auxiliary loss
    config.checkpoint_steps = (12000,)
    
    # AGGRESSIVE memory settings for speed
    config.n_ctxs = 1800     # Much smaller context buffer
    config.refresh_batch_size = 8   # Very small refresh batches
    config.out_batch_size = 128     # Very small output batches
    config.eval_batch_size = 16     # Tiny eval batches
    config.eval_n_batches = 2       # Minimal evaluation
    config.log_steps = 200          # Less frequent logging
    
    return config


def create_high_sparsity_config() -> ExperimentConfig:
    """Create a configuration for high sparsity (lower k_ratio)."""
    config = create_full_config()
    
    # High sparsity settings
    config.k_ratio = 0.02  # Only 2% of features active
    config.auxk_alpha = 1.5  # Higher auxiliary loss to encourage diversity
    
    return config


def create_low_sparsity_config() -> ExperimentConfig:
    """Create a configuration for lower sparsity (higher k_ratio)."""
    config = create_full_config()
    
    # Lower sparsity settings
    config.k_ratio = 0.08  # 8% of features active
    config.auxk_alpha = 0.5  # Lower auxiliary loss
    
    return config


def create_gpu_10gb_speed_config() -> ExperimentConfig:
    """Create a SPEED-OPTIMIZED configuration for 10GB GPU - prioritizes consistent performance."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=3.0,  # Smaller dictionary for speed
        k_ratio=0.04,  # Slightly lower sparsity for speed
        
        # SPEED-FOCUSED training parameters
        total_steps=10000,  # Shorter training for faster completion
        lr=8e-4,  # Higher LR for faster convergence
        kl_coeff=1.0,  # Lower KL for faster convergence
        auxk_alpha=0.5,  # Lower auxiliary loss
        
        # Model settings for maximum speed
        var_flag=0,  # Fixed variance only
        constrain_decoder=False,  # No expensive normalization
        
        # AGGRESSIVE memory settings for consistent speed
        n_ctxs=1500,     # Very small buffer to avoid memory pressure
        ctx_len=128,
        refresh_batch_size=8,   # Small batches for consistent memory usage
        out_batch_size=96,      # Small output batches
        
        # Minimal checkpointing for speed
        checkpoint_steps=(10000,),  # Only final checkpoint
        log_steps=250,  # Less frequent logging
        
        # Minimal evaluation for speed
        eval_batch_size=16,
        eval_n_batches=2,
        
        # System settings optimized for speed
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )
    """Create a configuration for lower sparsity (higher k_ratio)."""
    config = create_full_config()
    
    # Lower sparsity settings
    config.k_ratio = 0.08  # 8% of features active
    config.auxk_alpha = 0.5  # Lower auxiliary loss
    
    return config


def main():
    """Main training function with multiple configuration options and sweep support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEBatchTopK Training and Hyperparameter Sweeps")
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
            "gpu_10gb_speed",  # NEW: Speed-optimized config
            "high_sparsity",
            "low_sparsity"
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
        print("Starting hyperparameter sweep for VSAEBatchTopK...")
        print(f"Project: vsae-batchtopk-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAEBatchTopKSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "learned_variance": create_learned_variance_config,
        "gpu_10gb": create_gpu_10gb_config,
        "gpu_10gb_learned_var": create_gpu_10gb_learned_var_config,
        "gpu_10gb_speed": create_gpu_10gb_speed_config,  # NEW: Speed-optimized
        "high_sparsity": create_high_sparsity_config,
        "low_sparsity": create_low_sparsity_config,
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
                
        # Show batch top-k specific diagnostics if available
        topk_metrics = [k for k in results.keys() if "topk" in k.lower() or "aux" in k.lower()]
        if topk_metrics:
            print("\nBatch Top-K Diagnostics:")
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
# python train_vsae_batchtopk.py --config quick_test
# python train_vsae_batchtopk.py --config gpu_10gb           # Balanced 10GB config
# python train_vsae_batchtopk.py --config gpu_10gb_speed     # FASTEST - prioritizes consistent speed over quality
# python train_vsae_batchtopk.py --config gpu_10gb_learned_var  # Learned variance for 10GB
# python train_vsae_batchtopk.py --config learned_variance
# python train_vsae_batchtopk.py --config high_sparsity
# python train_vsae_batchtopk.py --sweep --wandb-entity your-username


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()