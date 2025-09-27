"""
Enhanced training script for VSAEJumpReLU with sweep functionality and optimized configurations.

Key features:
- Hyperparameter sweep support using wandb
- Multiple configuration presets including 10GB GPU optimized settings
- Command line interface for easy experiment management
- Enhanced evaluation with detailed diagnostics
- Memory efficient configurations
- Robust error handling and logging
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

# Import our improved implementations
from dictionary_learning.trainers.vsae_jump_relu import (
    VSAEJumpReLU,
    VSAEJumpReLUTrainer,
    VSAEJumpReLUConfig,
    VSAEJumpReLUTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire experiment with enhanced support for different scenarios."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    
    # VSAEJumpReLU specific configuration
    var_flag: int = 1  # 0: fixed variance, 1: learned variance
    threshold: float = 0.001  # Threshold parameter for JumpReLU
    use_april_update_mode: bool = True
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    l0_coeff: float = 1.0
    target_l0: float = 20.0
    
    # Schedule configuration
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    decay_start: Optional[int] = None
    
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
    wandb_project: str = "vsae-jumprelu-experiments"
    
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
        
        # Set decay start with proper constraints
        min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
        default_decay_start = int(0.8 * self.total_steps)
        
        if default_decay_start <= max(self.warmup_steps, self.sparsity_warmup_steps):
            self.decay_start = None  # Disable decay
        elif self.decay_start is None or self.decay_start < min_decay_start:
            self.decay_start = default_decay_start
    
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
    """Manages the entire training experiment with enhanced evaluation and diagnostics."""
    
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
                logging.FileHandler(log_dir / 'vsae_jumprelu_training.log'),
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
        
        self.logger.info(f"Model loaded. model.cfg.d_model: {model.cfg.d_model}")
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
            d_submodule=model.cfg.d_model,
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        
        return buffer
        
    def create_model_config(self, model: HookedTransformer) -> VSAEJumpReLUConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        
        return VSAEJumpReLUConfig(
            activation_dim=model.cfg.d_model,
            dict_size=dict_size,
            var_flag=self.config.var_flag,
            threshold=self.config.threshold,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
    def create_training_config(self) -> VSAEJumpReLUTrainingConfig:
        """Create training configuration from experiment config."""
        return VSAEJumpReLUTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start,
        )                                           
        
    def create_trainer_config(self, model_config: VSAEJumpReLUConfig, training_config: VSAEJumpReLUTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        return {
            "trainer": VSAEJumpReLUTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "submodule_name": self.config.hook_name,
            "wandb_name": self.get_experiment_name(),
            "seed": self.config.seed,
        }
        
    def get_experiment_name(self) -> str:
        """
        Generate a descriptive experiment name safe for filesystems.
        
        Format: VSAEJumpReLU_{model}_d{dict}x_lr{lr}_kl{kl}_l0c{l0c}_tl0{tl0}_th{th}_{var}
        No periods, dashes, or special characters that could cause filesystem issues.
        """
        var_suffix = "_learnedvar" if self.config.var_flag == 1 else "_fixedvar"
        
        # Make numbers filesystem-safe (no periods)
        lr_str = f"{self.config.lr:.1e}".replace('-', '').replace('.', '')  # 5e-04 -> 5e04
        kl_str = f"{int(self.config.kl_coeff)}"
        l0c_str = f"{int(self.config.l0_coeff * 10)}"  # 1.0 -> 10
        tl0_str = f"{int(self.config.target_l0)}"
        th_str = f"{self.config.threshold:.1e}".replace('-', '').replace('.', '')  # 1e-03 -> 1e03
        dict_str = f"{int(self.config.dict_size_multiple)}"  # 4.0 -> 4
        
        return (
            f"VSAEJumpReLU_{self.config.model_name}_"
            f"d{dict_str}x_lr{lr_str}_kl{kl_str}_"
            f"l0c{l0c_str}_tl0{tl0_str}_th{th_str}{var_suffix}"
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
        self.logger.info("Starting VSAEJumpReLU training experiment")
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
            self.logger.info(f"Target L0: {self.config.target_l0}")
            self.logger.info(f"Threshold: {self.config.threshold}")
            
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
                    "experiment_type": "vsae_jumprelu",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "var_flag": self.config.var_flag,
                    "threshold": self.config.threshold,
                    "target_l0": self.config.target_l0,
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
            
            # Add VSAEJumpReLU-specific diagnostics if possible
            try:
                # Get a sample batch for additional diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                # Forward pass to get features
                with torch.no_grad():
                    sample_batch = sample_batch.to(self.config.device)
                    _, features = vsae(sample_batch, output_features=True)
                    
                    # Compute L0 statistics
                    l0_per_sample = (features != 0).float().sum(dim=-1)
                    eval_results.update({
                        "actual_l0_mean": l0_per_sample.mean().item(),
                        "actual_l0_std": l0_per_sample.std().item(),
                        "target_l0": self.config.target_l0,
                        "l0_target_ratio": l0_per_sample.mean().item() / self.config.target_l0,
                    })
                    
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


class VSAEJumpReLUSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for VSAEJumpReLU trainer.
    
    Optimizes key parameters including KL coefficient, L0 coefficient, target L0,
    learning rate, threshold, and variance learning.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize VSAEJumpReLU sweep runner."""
        super().__init__(trainer_name="vsae-jumprelu", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for VSAEJumpReLU.
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
                # L0 coefficient: log-uniform distribution
                'l0_coeff': {
                    'distribution': 'log_uniform_values',
                    'min': 0.1,
                    'max': 10.0
                },
                # Target L0: discrete choices
                'target_l0': {
                    'values': [10.0, 20.0, 30.0, 50.0]
                },
                # Threshold: log-uniform distribution
                'threshold': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-2
                },
                # Dictionary size multiplier: discrete choices
                'dict_size_multiple': {
                    'values': [4.0, 8.0]
                },
                # Variance flag: discrete choice
                'var_flag': {
                    'values': [0, 1]
                }
            }
        }
    
    def get_run_name(self, sweep_params: dict) -> str:
        """
        Generate descriptive run name from sweep parameters.
        Safe for filesystems - no periods, dashes, or special characters.
        
        Format: JumpReLU_lr{lr}_kl{kl}_l0c{l0c}_tl0{tl0}_th{th}_d{dict}x_v{var}
        Examples:
        - lr=5e-4 -> lr5e04
        - l0_coeff=1.5 -> l0c15 (multiply by 10)
        - threshold=1e-3 -> th1e03
        """
        lr_str = f"{sweep_params['lr']:.1e}".replace('-', '').replace('.', '')  # 5e-04 -> 5e04
        kl_str = f"{int(sweep_params['kl_coeff'])}"
        l0_str = f"{int(sweep_params['l0_coeff'] * 10)}"  # 1.0 -> 10, 0.5 -> 5
        target_l0_str = f"{int(sweep_params['target_l0'])}"
        th_str = f"{sweep_params['threshold']:.1e}".replace('-', '').replace('.', '')  # 1e-03 -> 1e03
        dict_str = f"{int(sweep_params['dict_size_multiple'])}"  # 4.0 -> 4
        var_str = f"v{sweep_params['var_flag']}"
        
        return f"JumpReLU_lr{lr_str}_kl{kl_str}_l0c{l0_str}_tl0{target_l0_str}_th{th_str}_d{dict_str}x_{var_str}"
    
    def create_experiment_config(self, sweep_params: dict) -> ExperimentConfig:
        """
        Create an ExperimentConfig from wandb sweep parameters.
        """
        # Start with the quick test config
        config = create_quick_test_config()
        
        # Override with sweep parameters
        config.lr = sweep_params['lr']
        config.kl_coeff = sweep_params['kl_coeff']
        config.l0_coeff = sweep_params['l0_coeff']
        config.target_l0 = sweep_params['target_l0']
        config.threshold = sweep_params['threshold']
        config.dict_size_multiple = sweep_params['dict_size_multiple']
        config.var_flag = sweep_params['var_flag']
        
        # Adjust settings for sweep runs
        config.total_steps = 15000  # Longer to see convergence
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "vsae-jumprelu-sweeps"
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
        checkpoint_steps=list(),
        log_steps=50,
        
        # VSAEJumpReLU specific
        var_flag=1,
        threshold=0.001,
        l0_coeff=1.0,
        target_l0=20.0,
        
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
        model_name = "EleutherAI/pythia-70m-deduped",  # Changed from "gelu-1l"
        layer = 3,  # Changed from 0 - typical layers for pythia-70m are 3,4
        hook_name = "blocks.3.hook_resid_post",  # Updated to match layer
        dict_size_multiple=16.0,
        
        # Full training parameters
        total_steps=20001,  # Reduced from 50000
        lr=5e-4,
        kl_coeff=1,
        l0_coeff=0.1,
        target_l0=512.0,
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
        threshold=0.0045,
        use_april_update_mode=True,
        
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


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSAEJumpReLU Training and Hyperparameter Sweeps")
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
        ], 
        default="full",
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
        print("Starting hyperparameter sweep for VSAEJumpReLU...")
        print(f"Project: vsae-jumprelu-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = VSAEJumpReLUSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
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
                
        # Show L0-specific diagnostics if available
        l0_metrics = [k for k in results.keys() if "l0" in k.lower()]
        if l0_metrics:
            print("\nL0 Diagnostics:")
            for metric in l0_metrics:
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
# python train-vsae-jump-relu.py
# python train-vsae-jump-relu.py --config quick_test
# python train-vsae-jump-relu.py --sweep --wandb-entity your-username


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()