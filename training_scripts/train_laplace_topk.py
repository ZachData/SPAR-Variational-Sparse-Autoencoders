"""
Enhanced training script for Block Diagonal Laplacian TopK with sweep functionality and optimized configurations.

This script implements training for the BDL-VSAE-TopK (Block Diagonal Laplacian Variational Sparse Autoencoder with TopK),
which combines:
1. Variational autoencoder framework with reparameterization
2. Top-K structured sparsity mechanism  
3. Block diagonal Laplacian regularization for hierarchical feature organization

Key improvements over VSAE-TopK:
✅ Block diagonal Laplacian regularization for structured smoothness
✅ Hierarchical feature organization within and across blocks
✅ Configurable block sizes and Laplacian structures (chain, complete, ring)
✅ Separate annealing schedules for KL and Laplacian terms
✅ Enhanced interpretability through block-wise feature grouping
✅ Hyperparameter sweep support for Laplacian-specific parameters
✅ Comprehensive block structure analysis and diagnostics

Mathematical Framework:
- VAE loss: KL[q(z|x) || p(z)] + ||x - x̂||²
- Top-K sparsity: Select k largest |z| values while preserving signs
- Laplacian regularization: R(z) = Σᵢ zᵢᵀLᵢzᵢ where L = block_diag(L₁, L₂, ..., Lₖ)
- Total loss: MSE + λ₁·KL + λ₂·R(z) + λ₃·AuxLoss
"""

import torch
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import multiprocessing

from transformer_lens import HookedTransformer
from dictionary_learning.base_sweep import BaseSweepRunner
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

# Import our LaplacianTopK implementations
from dictionary_learning.trainers.laplace_topk import (
    LaplacianTopK, 
    LaplacianTopKTrainer,
    LaplacianTopKConfig,
    LaplacianTopKTrainingConfig
)


@dataclass
class ExperimentConfig:
    """Configuration for the entire Block Diagonal Laplacian TopK experiment."""
    # Model configuration
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.mlp.hook_post"
    dict_size_multiple: float = 4.0
    k_fraction: float = 0.08  # Fraction of dictionary size for Top-K
    
    # Block Diagonal Laplacian configuration
    block_sizes: Optional[List[int]] = None  # Auto-computed if None
    laplacian_type: str = "chain"  # 'chain', 'complete', 'ring'
    laplacian_coeff: float = 1.0  # Block diagonal Laplacian regularization coefficient
    laplacian_warmup_steps: Optional[int] = None  # Warmup for Laplacian regularization
    
    # Model-specific config
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    
    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    auxk_alpha: float = 1/32  # TopK auxiliary loss coefficient
    
    # Schedule configuration
    warmup_steps: Optional[int] = None
    kl_warmup_steps: Optional[int] = None  # KL annealing to prevent posterior collapse
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
    wandb_project: str = "laplace-topk-experiments"
    
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
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.total_steps)  # KL annealing
        if self.laplacian_warmup_steps is None:
            self.laplacian_warmup_steps = int(0.15 * self.total_steps)  # Laplacian annealing
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.decay_start_step is None:
            decay_start = int(0.8 * self.total_steps)
            min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
            self.decay_start_step = max(decay_start, min_decay_start)
    
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
    """Manages the entire training experiment with Block Diagonal Laplacian TopK support."""
    
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
                logging.FileHandler(log_dir / 'laplace_topk_training.log'),
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
        
    def create_model_config(self, model: HookedTransformer) -> LaplacianTopKConfig:
        """Create model configuration from experiment config."""
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_mlp)
        k = int(self.config.k_fraction * dict_size)
        
        # Create base config
        model_config = LaplacianTopKConfig(
            activation_dim=model.cfg.d_mlp,
            dict_size=dict_size,
            k=k,
            block_sizes=self.config.block_sizes,
            laplacian_type=self.config.laplacian_type,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device()
        )
        
        return model_config
        
    def create_training_config(self) -> LaplacianTopKTrainingConfig:
        """Create training configuration from experiment config."""
        return LaplacianTopKTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
            laplacian_coeff=self.config.laplacian_coeff,
            laplacian_warmup_steps=self.config.laplacian_warmup_steps,
            auxk_alpha=self.config.auxk_alpha,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
        )
        
    def create_trainer_config(self, model_config: LaplacianTopKConfig, training_config: LaplacianTopKTrainingConfig) -> Dict[str, Any]:
        """Create trainer configuration for the training loop."""
        trainer_config = {
            "trainer": LaplacianTopKTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "wandb_name": self.get_experiment_name(),
            "submodule_name": self.config.hook_name,
            "seed": self.config.seed,
        }
        
        return trainer_config
        
    def get_experiment_name(self) -> str:
        """Generate a descriptive experiment name."""
        k_value = int(self.config.k_fraction * self.config.dict_size_multiple * 2048)  # Assuming d_mlp=2048 for gelu-1l
        var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"
        laplacian_suffix = f"_{self.config.laplacian_type}_lap{self.config.laplacian_coeff}"
        
        return (
            f"LaplacianTopK_{self.config.model_name}_"
            f"d{int(self.config.dict_size_multiple * 2048)}_"
            f"k{k_value}_lr{self.config.lr}_kl{self.config.kl_coeff}_"
            f"aux{self.config.auxk_alpha}{laplacian_suffix}{var_suffix}"
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
        """Run the complete training experiment with Block Diagonal Laplacian TopK."""
        self.logger.info("Starting Block Diagonal Laplacian TopK training experiment")
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
            
            # Log configuration details
            self.logger.info(f"Model config: {model_config}")
            self.logger.info(f"Training config: {training_config}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}, K value: {model_config.k}")
            self.logger.info(f"K fraction: {self.config.k_fraction}, Auxiliary alpha: {self.config.auxk_alpha}")
            self.logger.info(f"Laplacian type: {self.config.laplacian_type}, coefficient: {self.config.laplacian_coeff}")
            
            # Log block structure information
            dummy_model = LaplacianTopK(model_config)
            block_info = dummy_model.laplacian.get_block_info()
            self.logger.info(f"Block structure: {block_info}")
            
            # Run training
            self.logger.info("Starting Block Diagonal Laplacian TopK training...")
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
                    "experiment_type": "laplace_topk",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "k_fraction": self.config.k_fraction,
                    "var_flag": self.config.var_flag,
                    "auxk_alpha": self.config.auxk_alpha,
                    "laplacian_type": self.config.laplacian_type,
                    "laplacian_coeff": self.config.laplacian_coeff,
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
        """Evaluate the trained model with enhanced Laplacian diagnostics."""
        self.logger.info("Evaluating trained model...")
        
        try:
            # Load the trained model
            from dictionary_learning.utils import load_dictionary
            
            model_path = save_dir / "trainer_0"
            laplace_model, config = load_dictionary(str(model_path), device=self.config.device)
            
            # Run evaluation
            eval_results = evaluate(
                dictionary=laplace_model,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches
            )
            
            # Add Laplacian-specific diagnostics if possible
            try:
                # Get a sample batch for diagnostics
                sample_batch = next(iter(buffer))
                if len(sample_batch) > self.config.eval_batch_size:
                    sample_batch = sample_batch[:self.config.eval_batch_size]
                
                # Laplacian diagnostics
                if hasattr(laplace_model, 'get_laplacian_diagnostics'):
                    laplacian_diagnostics = laplace_model.get_laplacian_diagnostics(sample_batch.to(self.config.device))
                    
                    # Add to eval results
                    for key, value in laplacian_diagnostics.items():
                        if key != 'block_info':  # Skip non-numeric block info
                            eval_results[f"final_{key}"] = value.item() if torch.is_tensor(value) else value
                
                # Block structure analysis
                if hasattr(laplace_model, 'analyze_block_structure'):
                    block_analysis = laplace_model.analyze_block_structure(sample_batch.to(self.config.device))
                    
                    # Add summary statistics
                    if 'block_selection_stats' in block_analysis:
                        total_selection_rate = sum(
                            stats['selection_rate'] 
                            for stats in block_analysis['block_selection_stats'].values()
                        ) / len(block_analysis['block_selection_stats'])
                        eval_results['final_avg_block_selection_rate'] = total_selection_rate
                        
                    eval_results['final_laplacian_reg'] = block_analysis.get('laplacian_reg', 0.0)
                        
            except Exception as e:
                self.logger.warning(f"Could not compute Laplacian diagnostics: {e}")
            
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


class LaplacianTopKSweepRunner(BaseSweepRunner):
    """
    Hyperparameter sweep runner for Block Diagonal Laplacian TopK trainer.
    
    Optimizes both TopK-specific parameters and Laplacian regularization parameters.
    """
    
    def __init__(self, wandb_entity: str):
        """Initialize Laplacian TopK sweep runner."""
        super().__init__(trainer_name="laplace-topk", wandb_entity=wandb_entity)
    
    def get_sweep_config(self) -> dict:
        """
        Define the wandb sweep configuration for Laplacian TopK.
        
        Focuses on both TopK and Laplacian-specific parameters.
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
                # NEW: Laplacian regularization coefficient
                'laplacian_coeff': {
                    'distribution': 'log_uniform_values',
                    'min': 0.1,
                    'max': 10.0
                },
                # NEW: Laplacian type - discrete choice
                'laplacian_type': {
                    'values': ['chain', 'complete', 'ring']
                },
                # Dictionary size multiplier: discrete choices
                'dict_size_multiple': {
                    'values': [4.0, 8.0]
                },
                # Variance flag: discrete choice
                'var_flag': {
                    'values': [0, 1]
                },
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
        lap_str = f"lap{sweep_params['laplacian_coeff']:.2f}_{sweep_params['laplacian_type']}"
        dict_str = f"{sweep_params['dict_size_multiple']:.0f}x"
        var_str = f"var{sweep_params['var_flag']}"
        
        return f"LAP_lr{lr_str}_kl{kl_str}_{k_str}_{aux_str}_{lap_str}_d{dict_str}_{var_str}"
    
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
        config.laplacian_coeff = sweep_params['laplacian_coeff']  # NEW
        config.laplacian_type = sweep_params['laplacian_type']  # NEW
        config.dict_size_multiple = sweep_params['dict_size_multiple']
        config.var_flag = sweep_params['var_flag']
        
        # Adjust settings for sweep runs
        config.total_steps = 15000  # Moderate length for sweep
        config.checkpoint_steps = ()
        config.log_steps = 250
        
        # Smaller buffer for memory efficiency
        config.n_ctxs = 1000
        config.refresh_batch_size = 16
        config.out_batch_size = 256
        
        # Use fixed project name for sweeps
        config.wandb_project = "laplace-topk-sweeps"
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
        k_fraction=0.0079,
        
        # Laplacian configuration
        laplacian_type="chain",
        laplacian_coeff=1.0,
        
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
        k_fraction=0.08,
        
        # Laplacian configuration
        laplacian_type="chain",
        laplacian_coeff=1.0,
        
        # Full training parameters
        total_steps=25000,
        lr=5e-4,
        kl_coeff=500.0,
        auxk_alpha=1/32,
        
        # Model settings
        var_flag=0,  # Start with fixed variance
        use_april_update_mode=True,
        
        # MEMORY EFFICIENT buffer settings
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


def create_complete_graph_config() -> ExperimentConfig:
    """Create a configuration using complete graph Laplacian."""
    config = create_full_config()
    config.laplacian_type = "complete"
    config.laplacian_coeff = 0.5  # Lower coefficient for complete graph
    return config


def create_ring_graph_config() -> ExperimentConfig:
    """Create a configuration using ring graph Laplacian."""
    config = create_full_config()
    config.laplacian_type = "ring"
    config.laplacian_coeff = 1.5  # Higher coefficient for ring graph
    return config


def create_learned_variance_config() -> ExperimentConfig:
    """Create a configuration for training with learned variance - memory efficient version."""
    config = create_full_config()
    
    # Enable learned variance with adjusted settings
    config.var_flag = 1
    config.total_steps = 30000  # Slightly longer for learned variance
    config.lr = 3e-4  # Slightly lower LR for learned variance stability
    config.kl_coeff = 300.0  # Lower KL coeff since learned variance provides more flexibility
    config.auxk_alpha = 1/48  # Slightly lower auxiliary loss for stability
    config.laplacian_coeff = 0.8  # Slightly lower Laplacian coeff for learned variance
    
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
        hook_name="blocks.0.mlp.hook_post",
        dict_size_multiple=4.0,
        k_fraction=0.0625,
        
        # Laplacian configuration - conservative for memory
        laplacian_type="chain",  # Most memory efficient
        laplacian_coeff=1.0,     # Slightly lower for stability
        
        # Training parameters optimized for 10GB GPU
        total_steps=20000,
        lr=8e-4,
        kl_coeff=0.0079,
        auxk_alpha=1/32,
        
        # Model settings
        var_flag=0,  # Fixed variance for memory efficiency
        use_april_update_mode=True,
        
        # OPTIMIZED buffer settings for consistent speed
        n_ctxs=2500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=192,
        
        # Checkpointing
        checkpoint_steps=(20000,),
        log_steps=100,
        
        # Evaluation - smaller to reduce memory spikes
        eval_batch_size=24,
        eval_n_batches=4,
        
        # System settings
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        seed=42,
    )


def create_custom_blocks_config() -> ExperimentConfig:
    """Create a configuration with custom block sizes."""
    config = create_full_config()
    
    # Custom block sizes - example for d_mlp=2048, dict_size=8192
    # These will be automatically adjusted based on actual dict_size
    config.block_sizes = None  # Will use automatic block sizing
    config.laplacian_type = "chain"
    config.laplacian_coeff = 1.2  # Slightly higher for custom structure
    
    return config


def create_high_regularization_config() -> ExperimentConfig:
    """Create a configuration with high Laplacian regularization."""
    config = create_full_config()
    config.laplacian_coeff = 5.0  # High regularization
    config.laplacian_type = "complete"  # Strong within-block coupling
    config.lr = 3e-4  # Lower LR for stability with high regularization
    return config


def main():
    """Main training function with multiple configuration options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Block Diagonal Laplacian TopK Training and Hyperparameter Sweeps")
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
            "complete_graph",
            "ring_graph",
            "learned_variance",
            "gpu_10gb",
            "custom_blocks",
            "high_regularization"
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
        print("Starting hyperparameter sweep for Block Diagonal Laplacian TopK...")
        print(f"Project: laplace-topk-sweeps")
        print(f"Entity: {args.wandb_entity}")
        
        sweep_runner = LaplacianTopKSweepRunner(wandb_entity=args.wandb_entity)
        sweep_runner.run_sweep()
        return
    
    # Regular single training run
    config_functions = {
        "quick_test": create_quick_test_config,
        "full": create_full_config,
        "complete_graph": create_complete_graph_config,
        "ring_graph": create_ring_graph_config,
        "learned_variance": create_learned_variance_config,
        "gpu_10gb": create_gpu_10gb_config,
        "custom_blocks": create_custom_blocks_config,
        "high_regularization": create_high_regularization_config,
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
                
        # Show Laplacian-specific diagnostics if available
        laplacian_metrics = [k for k in results.keys() if "laplacian" in k.lower() or "block" in k.lower()]
        if laplacian_metrics:
            print("\nLaplacian Diagnostics:")
            for metric in laplacian_metrics:
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
# python train_laplace_topk.py --config quick_test
# python train_laplace_topk.py --config complete_graph     # Use complete graph Laplacian
# python train_laplace_topk.py --config ring_graph        # Use ring graph Laplacian  
# python train_laplace_topk.py --config learned_variance  # With learned variance
# python train_laplace_topk.py --config gpu_10gb          # Optimized for 10GB GPU
# python train_laplace_topk.py --config high_regularization  # High Laplacian regularization
# python train_laplace_topk.py --sweep --wandb-entity your-username  # Hyperparameter sweep


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
        
    main()
