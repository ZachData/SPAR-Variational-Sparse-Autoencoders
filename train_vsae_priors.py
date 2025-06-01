"""
Improved training script for VSAEPriors (renamed from VSAEMixture) following the pattern from train_isovsae_1lgelu.py.

Key improvements:
- Clean dataclass-based configuration management
- Multiple prior types support (Gaussian, Laplace, Exponential + placeholders for advanced priors)
- ExperimentRunner class for better organization
- Multiple configuration presets for empirical testing
- Proper logging and error handling
- Hyperparameter sweep support
- Better memory management
- Command-line interface

Prior Types Supported:
- Easy (Implemented): Gaussian, Laplace, Exponential
- Planned (Placeholders): Spike-and-slab, Horseshoe, Beta, Student's t, Gamma
- Future (Advanced): Learnable priors, Hierarchical priors, Normalizing flows
"""

import torch
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import multiprocessing

from transformer_lens import HookedTransformer
from dictionary_learning.base_sweep import BaseSweepRunner
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
from dictionary_learning.trainers.vsae_priors import VSAEPriorsTrainer, VSAEPriorsGaussian, VSAEPriorsConfig, VSAEPriorsTrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for the entire VSAEPriors experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # Prior configuration - This is the key new feature!
    prior_types: List[str] = field(default_factory=lambda: ["gaussian"])  # Available: gaussian, laplace, exponential
    prior_assignment_strategy: str = "uniform"  # How to assign priors to features: uniform, random, layered, single
    prior_proportions: Optional[Dict[str, float]] = None  # For random assignment, what proportion of each prior
    prior_params: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Parameters for each prior type
    
    # VSAEPriors-specific config  
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0
    
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
    wandb_project: str = "vsae-priors-experiments"
    
    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42
    
    # Evaluation configuration
    eval_batch_size: int = 64
    eval_n_batches: int = 10
    
    def __post_init__(self):
        """Set default prior proportions if not specified."""
        if self.prior_proportions is None and len(self.prior_types) > 1:
            # Default to uniform distribution across prior types
            uniform_prop = 1.0 / len(self.prior_types)
            self.prior_proportions = {prior_type: uniform_prop for prior_type in self.prior_types}
    
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
    """Manages the entire VSAEPriors training experiment."""
    
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
                logging.FileHandler(log_dir / 'vsae_priors_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> VSAEPriorsConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEPriorsConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            prior_types=self.config.prior_types,
            prior_assignment_strategy=self.config.prior_assignment_strategy,
            prior_proportions=self.config.prior_proportions,
            prior_params=self.config.prior_params,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
            log_var_init=self.config.log_var_init,
        )
        
    def create_training_config(self) -> VSAEPriorsTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEPriorsTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
            resample_steps=self.config.resample_steps,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )
        
    def create_trainer_config(self, model_config: VSAEPriorsConfig, training_config: VSAEPriorsTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEPriorsTrainer,
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
        prior_str = "_".join(self.config.prior_types)
        assignment_str = self.config.prior_assignment_strategy
        
        return (
            f"VSAEPriors_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"  # Assuming d_mlp=2048 for gelu-1l
            f"lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"priors_{prior_str}_assign_{assignment_str}"
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
        self.logger.info("Starting VSAEPriors training experiment")
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
            self.logger.info(f"Prior types: {model_config.prior_types}")
            self.logger.info(f"Prior assignment: {model_config.prior_assignment_strategy}")
            
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
                    "experiment_type": "vsae_priors",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "prior_types": self.config.prior_types,
                    "prior_assignment_strategy": self.config.prior_assignment_strategy,
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
            
            # Run evaluation with explicit parameters to avoid loss recovery issues
            eval_results = evaluate(
                dictionary=vsae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches,
                # Skip loss recovery for now to avoid the unpacking error
                # The core metrics (reconstruction, sparsity, etc.) are more important
            )
            
            # Add VSAEPriors-specific diagnostics if possible
            try:
                # Get a sample batch for diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                # Get prior diagnostics
                prior_diagnostics = vsae.get_prior_diagnostics(sample_batch.to(self.config.device))
                
                # Add to eval results
                for key, value in prior_diagnostics.items():
                    eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                        
            except Exception as e:
                self.logger.warning(f"Could not compute prior diagnostics: {e}")
            
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


class VSAEPriorsSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for VSAEPriors trainer.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize VSAEPriors sweep runner."""
        super().__init__(trainer_name="vsae-priors", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for VSAEPriors.
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
                # Prior configuration - this is the key new sweep dimension!
                'prior_setup': {
                    'values': [
                        'all_gaussian',
                        'all_laplace', 
                        'mixed_gauss_laplace',
                        'all_exponential',
                        'mixed_sparse_trio'
                    ]
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
        prior_str = sweep_params['prior_setup']
        var_str = f"v{sweep_params['var_flag']}"
        
        return f"Priors_lr{lr_str}_kl{kl_str}_d{dict_str}_{prior_str}_{var_str}"
    
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
        
        # Configure priors based on sweep parameter
        prior_setup = sweep_params['prior_setup']
        if prior_setup == 'all_gaussian':
            config.prior_types = ['gaussian']
        elif prior_setup == 'all_laplace':
            config.prior_types = ['laplace']
        elif prior_setup == 'mixed_gauss_laplace':
            config.prior_types = ['gaussian', 'laplace']
            config.prior_proportions = {'gaussian': 0.5, 'laplace': 0.5}
        elif prior_setup == 'all_exponential':
            config.prior_types = ['exponential']
        elif prior_setup == 'mixed_sparse_trio':
            config.prior_types = ['gaussian', 'laplace', 'exponential']
            config.prior_proportions = {'gaussian': 0.33, 'laplace': 0.34, 'exponential': 0.33}
        
        # Adjust settings for sweep runs
        config.total_steps = 15000
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-priors-sweeps"
        config.save_dir = "./temp_sweep_run"
        config.use_wandb = False  # Sweep handles wandb
        
        return config


# ============================================================================
# CONFIGURATION PRESETS - Easy empirical testing of different prior setups!
# ============================================================================

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
        
        # Simple prior setup for testing
        prior_types=["gaussian"],
        prior_assignment_strategy="single",
        
        # Small buffer for testing
        n_ctxs=500,
        refresh_batch_size=16,
        out_batch_size=128,
        
        # VSAEPriors specific
        var_flag=0,
        log_var_init=-2.0,
        
        # Evaluation
        eval_batch_size=32,
        eval_n_batches=3,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
    )


def create_baseline_gaussian_config() -> ExperimentConfig:
    """Baseline: All Gaussian priors (like standard VAE)."""
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
        
        # All Gaussian priors (baseline)
        prior_types=["gaussian"],
        prior_assignment_strategy="single",
        
        # Model settings
        var_flag=0,  # Fixed variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        
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


def create_sparse_laplace_config() -> ExperimentConfig:
    """Test sparsity-inducing Laplace priors."""
    config = create_baseline_gaussian_config()
    config.prior_types = ["laplace"]
    config.prior_assignment_strategy = "single"
    config.prior_params = {
        "laplace": {"scale": 1.0}  # TODO: Could make this tunable
    }
    config.kl_coeff = 30.0  # Might need lower KL coeff for Laplace
    return config


def create_mixed_gaussian_laplace_config() -> ExperimentConfig:
    """Test mixed Gaussian + Laplace priors."""
    config = create_baseline_gaussian_config()
    config.prior_types = ["gaussian", "laplace"]
    config.prior_assignment_strategy = "uniform"  # 50-50 split
    config.prior_params = {
        "gaussian": {"scale": 1.0},
        "laplace": {"scale": 1.0}
    }
    return config


def create_exponential_only_config() -> ExperimentConfig:
    """Test non-negative exponential priors."""
    config = create_baseline_gaussian_config()
    config.prior_types = ["exponential"]
    config.prior_assignment_strategy = "single"
    config.prior_params = {
        "exponential": {"scale": 1.0}
    }
    config.kl_coeff = 40.0  # Might need different KL coeff
    return config


def create_mixed_sparse_trio_config() -> ExperimentConfig:
    """Test all three easy priors mixed together."""
    config = create_baseline_gaussian_config()
    config.prior_types = ["gaussian", "laplace", "exponential"]
    config.prior_assignment_strategy = "uniform"  # Equal split
    config.prior_params = {
        "gaussian": {"scale": 1.0},
        "laplace": {"scale": 1.0},
        "exponential": {"scale": 1.0}
    }
    return config


def create_learned_variance_mixed_config() -> ExperimentConfig:
    """Test learned variance with mixed priors."""
    config = create_mixed_gaussian_laplace_config()
    config.var_flag = 1
    config.lr = 3e-5  # Lower LR for stability with learned variance
    config.kl_coeff = 25.0  # Lower KL coeff
    config.log_var_init = -2.5  # Start with smaller variance
    config.total_steps = 40000  # Longer training
    config.kl_warmup_steps = 8000  # 20% of training
    config.checkpoint_steps = (20000, 40000)
    return config


def create_with_resampling_config() -> ExperimentConfig:
    """Test with dead neuron resampling enabled."""
    config = create_sparse_laplace_config()
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
        
        # Simple prior setup for memory efficiency
        # prior_types=["gaussian"],
        # prior_types=["laplace"],
        # prior_types=["exponential"],
        prior_types=["spike_slab"],
        prior_assignment_strategy="single",
        
        # Model settings
        var_flag=0,
        use_april_update_mode=True,
        log_var_init=-2.0,
        
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


"""
Spike-and-Slab Configuration Functions for VSAEPriors Trainer

Add these to your train_vsae_priors.py file alongside the other config functions!
Multiple variants for different spike-and-slab experiments.
"""

def create_spike_slab_config() -> ExperimentConfig:
    """Pure spike-and-slab configuration - the main event! 🎯"""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Pure spike-and-slab setup
        prior_types=["spike_slab"],
        prior_assignment_strategy="single",  # All features use spike-and-slab
        
        # Spike-and-slab specific parameters
        spike_slab_spike_scale=0.01,      # Very tight spike for true sparsity
        spike_slab_slab_scale=1.0,        # Normal slab width
        spike_slab_spike_prob=0.85,       # High sparsity prior (85% spike)
        spike_slab_temperature=0.5,       # Start with moderate temperature
        spike_slab_anneal_temp=True,      # Anneal to more discrete selection
        spike_slab_entropy_reg=0.1,       # Balance spike/slab usage
        
        # Training parameters optimized for spike-and-slab
        total_steps=30000,
        lr=3e-5,                          # Lower LR for stability with discrete sampling
        kl_coeff=80.0,                    # Lower KL coeff since spike-slab is naturally sparse
        kl_warmup_steps=6000,             # 20% of training for KL annealing
        gradient_clip_norm=1.0,           # Important for stability
        
        # Buffer settings
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        
        # Model settings
        var_flag=0,                       # Start with fixed variance
        use_april_update_mode=True,
        log_var_init=-2.0,
        
        # Checkpointing
        checkpoint_steps=(15000, 30000),
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


def create_mixed_gaussian_spike_slab_config() -> ExperimentConfig:
    """50-50 mix of Gaussian and spike-and-slab priors."""
    config = create_spike_slab_config()
    
    # Mix of priors
    config.prior_types = ["gaussian", "spike_slab"]
    config.prior_assignment_strategy = "uniform"  # 50-50 split
    
    # Adjust parameters for mixed setup
    config.kl_coeff = 120.0  # Slightly higher since only half features are sparse
    config.spike_slab_spike_prob = 0.8  # Still high sparsity for spike-slab features
    
    return config


def create_sparse_trio_with_spike_slab_config() -> ExperimentConfig:
    """All three sparsity-inducing priors: Laplace, Exponential, Spike-and-Slab."""
    config = create_spike_slab_config()
    
    # The sparsity dream team!
    config.prior_types = ["laplace", "exponential", "spike_slab"]
    config.prior_assignment_strategy = "uniform"  # Equal split
    
    # Parameters for each prior type
    config.prior_params = {
        "laplace": {"scale": 1.0},
        "exponential": {"rate": 1.0},
        # spike_slab params come from the main config
    }
    
    # Adjust for triple-prior setup
    config.kl_coeff = 100.0
    config.lr = 2e-5  # Even more conservative
    config.total_steps = 35000  # Longer training for complex setup
    config.kl_warmup_steps = 7000
    config.checkpoint_steps = (15000, 25000, 35000)
    
    return config


def create_learned_variance_spike_slab_config() -> ExperimentConfig:
    """Spike-and-slab with learned variance - advanced mode!"""
    config = create_spike_slab_config()
    
    # Enable learned variance
    config.var_flag = 1
    config.log_var_init = -2.5  # Start with smaller variance for stability
    
    # Adjust training for learned variance
    config.lr = 2e-5  # Lower LR for stability
    config.kl_coeff = 60.0  # Lower since learned variance provides flexibility
    config.total_steps = 40000  # Longer training
    config.kl_warmup_steps = 8000  # 20% of training
    config.checkpoint_steps = (20000, 40000)
    
    # More conservative spike-and-slab settings with learned variance
    config.spike_slab_spike_prob = 0.75  # Slightly less aggressive sparsity
    config.spike_slab_temperature = 0.3   # Start cooler for stability
    
    return config


def create_aggressive_spike_slab_config() -> ExperimentConfig:
    """Very aggressive sparsity with tight spike and high spike probability."""
    config = create_spike_slab_config()
    
    # Aggressive sparsity settings
    config.spike_slab_spike_scale = 0.005   # Even tighter spike!
    config.spike_slab_spike_prob = 0.95     # 95% sparsity prior
    config.spike_slab_temperature = 0.3     # Start cooler for more discrete behavior
    config.spike_slab_entropy_reg = 0.05    # Less entropy reg to allow extreme sparsity
    
    # Adjust training for aggressive setup
    config.kl_coeff = 50.0   # Lower KL coeff since we want extreme sparsity
    config.lr = 1e-5         # Very conservative LR
    config.gradient_clip_norm = 0.5  # Tighter gradient clipping
    
    return config


def create_gentle_spike_slab_config() -> ExperimentConfig:
    """Gentler spike-and-slab for easier training and comparison."""
    config = create_spike_slab_config()
    
    # Gentler settings
    config.spike_slab_spike_scale = 0.05    # Less tight spike
    config.spike_slab_spike_prob = 0.7      # More moderate sparsity
    config.spike_slab_temperature = 0.7     # Warmer temperature
    config.spike_slab_entropy_reg = 0.2     # More entropy reg for balance
    
    # More standard training settings
    config.lr = 5e-5         # Higher LR since it's gentler
    config.kl_coeff = 150.0  # Higher KL coeff since less natural sparsity
    
    return config


def create_spike_slab_comparison_config() -> ExperimentConfig:
    """Four-way comparison: Gaussian, Laplace, Exponential, Spike-and-Slab."""
    config = create_spike_slab_config()
    
    # All four priors for direct comparison
    config.prior_types = ["gaussian", "laplace", "exponential", "spike_slab"]
    config.prior_assignment_strategy = "uniform"  # Equal 25% each
    
    # Parameters for each prior
    config.prior_params = {
        "gaussian": {"scale": 1.0},
        "laplace": {"scale": 1.0},
        "exponential": {"rate": 1.0},
        # spike_slab from main config
    }
    
    # Balanced settings for fair comparison
    config.kl_coeff = 120.0
    config.lr = 3e-5
    config.total_steps = 35000
    config.kl_warmup_steps = 7000
    config.checkpoint_steps = (15000, 25000, 35000)
    
    return config


def create_gpu_10gb_spike_slab_config() -> ExperimentConfig:
    """Memory-efficient spike-and-slab for 10GB GPUs."""
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        
        # Pure spike-and-slab
        prior_types=["spike_slab"],
        prior_assignment_strategy="single",
        
        # Spike-and-slab parameters
        spike_slab_spike_scale=0.01,
        spike_slab_slab_scale=1.0,
        spike_slab_spike_prob=0.8,
        spike_slab_temperature=0.5,
        spike_slab_anneal_temp=True,
        spike_slab_entropy_reg=0.1,
        
        # Training optimized for 10GB GPU
        total_steps=20000,
        lr=3e-5,
        kl_coeff=80.0,
        kl_warmup_steps=4000,
        gradient_clip_norm=1.0,
        
        # Memory-efficient buffer settings
        n_ctxs=2000,        # Smaller buffer
        ctx_len=128,
        refresh_batch_size=16,   # Smaller batches
        out_batch_size=256,      # Smaller output batches
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
        use_april_update_mode=True,
        log_var_init=-2.0,
        
        # Checkpointing
        checkpoint_steps=(20000,),
        log_steps=100,
        
        # Efficient evaluation
        eval_batch_size=32,
        eval_n_batches=5,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_spike_slab_ablation_study_config() -> ExperimentConfig:
    """Configuration for ablation studies - testing different spike-and-slab parameters."""
    config = create_spike_slab_config()
    
    # Good baseline for ablation studies
    config.total_steps = 25000  # Shorter for multiple runs
    config.checkpoint_steps = (25000,)
    
    # These can be varied in ablation:
    # - spike_slab_spike_prob: [0.6, 0.7, 0.8, 0.9, 0.95]
    # - spike_slab_temperature: [0.1, 0.3, 0.5, 0.7, 1.0]
    # - spike_slab_entropy_reg: [0.0, 0.05, 0.1, 0.2, 0.5]
    # - kl_coeff: [20, 50, 80, 120, 200]
    
    return config


# ============================================================================
# UPDATE THE MAIN FUNCTION to include these new configs!
# ============================================================================

def main():
    """Updated main function with all the new spike-and-slab configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEPriors Training and Hyperparameter Sweeps")
    parser.add_argument(
        "--sweep", 
        action="store_true", 
        help="Run hyperparameter sweep instead of single training"
    )
    parser.add_argument(
        "--config", 
        choices=[
            "quick_test", 
            "baseline_gaussian",
            "sparse_laplace",
            "mixed_gaussian_laplace", 
            "exponential_only",
            "mixed_sparse_trio",
            "learned_variance_mixed",
            "with_resampling",
            "gpu_10gb",
            # NEW SPIKE-AND-SLAB CONFIGS! 🎯
            "spike_slab",                    # Pure spike-and-slab
            "mixed_gaussian_spike_slab",     # 50-50 Gaussian + spike-slab
            "sparse_trio_with_spike_slab",   # Laplace + Exponential + Spike-slab
            "learned_variance_spike_slab",   # Spike-slab with learned variance
            "aggressive_spike_slab",         # Very high sparsity
            "gentle_spike_slab",            # Milder spike-slab
            "spike_slab_comparison",        # All 4 priors compared
            "gpu_10gb_spike_slab",          # Memory-efficient spike-slab
            "spike_slab_ablation",          # For parameter studies
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
        print("Starting hyperparameter sweep for VSAEPriors...")
        print(f"Project: vsae-priors-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAEPriorsSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "baseline_gaussian": create_baseline_gaussian_config,
        "sparse_laplace": create_sparse_laplace_config,
        "mixed_gaussian_laplace": create_mixed_gaussian_laplace_config,
        "exponential_only": create_exponential_only_config,
        "mixed_sparse_trio": create_mixed_sparse_trio_config,
        "learned_variance_mixed": create_learned_variance_mixed_config,
        "with_resampling": create_with_resampling_config,
        "gpu_10gb": create_gpu_10gb_config,
        # NEW SPIKE-AND-SLAB FUNCTIONS! ✨
        "spike_slab": create_spike_slab_config,
        "mixed_gaussian_spike_slab": create_mixed_gaussian_spike_slab_config,
        "sparse_trio_with_spike_slab": create_sparse_trio_with_spike_slab_config,
        "learned_variance_spike_slab": create_learned_variance_spike_slab_config,
        "aggressive_spike_slab": create_aggressive_spike_slab_config,
        "gentle_spike_slab": create_gentle_spike_slab_config,
        "spike_slab_comparison": create_spike_slab_comparison_config,
        "gpu_10gb_spike_slab": create_gpu_10gb_spike_slab_config,
        "spike_slab_ablation": create_spike_slab_ablation_study_config,
    }
    
    config = config_functions[args.config]()
    runner = ExperimentRunner(config)
    results = runner.run_training()
    
    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<30} | Value")
        print("-" * 40)
        
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<30} | {results[metric]:.4f}")
                
        # Show KL and prior-type diagnostics if available
        kl_metrics = [k for k in results.keys() if "kl" in k.lower()]
        prior_metrics = [k for k in results.keys() if any(prior in k.lower() for prior in ["gaussian", "laplace", "exponential", "spike_slab", "prior"])]
        
        if kl_metrics:
            print("\nKL Diagnostics:")
            for metric in kl_metrics:
                if metric in results:
                    print(f"{metric:<30} | {results[metric]:.4f}")
        
        if prior_metrics:
            print("\nPrior-Type Diagnostics:")
            for metric in prior_metrics:
                if metric in results:
                    print(f"{metric:<30} | {results[metric]:.4f}")
        
        # SPIKE-AND-SLAB SPECIFIC METRICS! 🎯
        spike_slab_metrics = [k for k in results.keys() if "spike_slab" in k.lower()]
        if spike_slab_metrics:
            print("\n🎯 Spike-and-Slab Diagnostics:")
            for metric in spike_slab_metrics:
                if metric in results:
                    print(f"{metric:<30} | {results[metric]:.4f}")
                
        additional_metrics = ["loss_original", "loss_reconstructed", "frac_recovered"]
        print("\nAdditional Metrics:")
        for metric in additional_metrics:
            if metric in results:
                print(f"{metric:<30} | {results[metric]:.4f}")
                
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
# python train_vsae_priors.py --config quick_test
# python train_vsae_priors.py --config baseline_gaussian
# python train_vsae_priors.py --config sparse_laplace  
# python train_vsae_priors.py --config mixed_gaussian_laplace
# python train_vsae_priors.py --config exponential_only
# python train_vsae_priors.py --config mixed_sparse_trio
# python train_vsae_priors.py --config learned_variance_mixed
# python train_vsae_priors.py --config with_resampling
# python train_vsae_priors.py --config gpu_10gb
# python train_vsae_priors.py --sweep --wandb-entity your-username

# Usage examples with all the new spike-and-slab configs:
"""
# Basic spike-and-slab
python train_vsae_priors.py --config spike_slab

# Memory-efficient for smaller GPUs  
python train_vsae_priors.py --config gpu_10gb_spike_slab

# Compare spike-and-slab with Gaussian
python train_vsae_priors.py --config mixed_gaussian_spike_slab

# Compare all sparsity-inducing priors
python train_vsae_priors.py --config sparse_trio_with_spike_slab

# Advanced: learned variance + spike-and-slab
python train_vsae_priors.py --config learned_variance_spike_slab

# Extreme sparsity experiment
python train_vsae_priors.py --config aggressive_spike_slab

# Gentler version for easier training
python train_vsae_priors.py --config gentle_spike_slab

# Full 4-way comparison
python train_vsae_priors.py --config spike_slab_comparison

# For ablation studies
python train_vsae_priors.py --config spike_slab_ablation
"""