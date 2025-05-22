import os
import wandb
import argparse
import time
import uuid
import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.trainers.vsae_gated import VSAEGatedTrainer, VSAEGated
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

# Unique run identifier to keep WandB runs organized
RUN_ID = str(uuid.uuid4())[:8]

def train_single_model(config=None, dict_size_multiple=4.0):
    """
    Train a model with the parameters specified in config
    This function is called by wandb.agent for each hyperparameter combination
    """
    # Initialize wandb
    with wandb.init(
        project=f"gvsae_{MODEL_NAME}", 
        group=f"dict_size_{dict_size_multiple}x",
        job_type="sweep",
        name=f"sweep_{dict_size_multiple}x_{RUN_ID}",
        config=config
    ) as run:
        # Get the WandB config
        config = wandb.config
        
        # Create model
        model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
        
        # Create hook name based on layer
        hook_name = f"blocks.{LAYER}.mlp.hook_post"
        
        # Calculate dictionary size
        d_mlp = model.cfg.d_mlp
        dict_size = int(dict_size_multiple * d_mlp)

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
            hook_name=hook_name,
            d_submodule=d_mlp,
            n_ctxs=3000,
            ctx_len=128,
            refresh_batch_size=32,
            out_batch_size=1024,
            device="cpu",  # Store on CPU to save GPU memory
        )
        
        # Calculate step-based hyperparameters
        total_steps = MAX_STEPS
        warmup_steps = int(config.warmup_frac * total_steps)
        sparsity_warmup_steps = int(config.sparsity_warmup_frac * total_steps)
        decay_start = int(config.decay_start_frac * total_steps)
        
        # Configure trainer
        trainer_config = {
            "trainer": VSAEGatedTrainer,
            "steps": total_steps,
            "activation_dim": d_mlp,
            "dict_size": dict_size,
            "layer": LAYER,
            "lm_name": MODEL_NAME,
            "lr": config.lr,
            "kl_coeff": config.kl_coeff,
            "l1_penalty": config.l1_penalty,
            "aux_weight": config.aux_weight,
            "warmup_steps": warmup_steps,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "decay_start": decay_start,
            "var_flag": VAR_FLAG,
            "use_april_update_mode": True,
            "use_constrained_optimizer": config.use_constrained_optimizer,
            "device": "cuda",
            "wandb_name": f"GVSAE_{MODEL_NAME}_{dict_size}_{config.lr}_{config.kl_coeff}_{config.l1_penalty}_{config.aux_weight}",
            "dict_class": VSAEGated
        }
        
        # Configure save directories
        save_dir = f"./sweep_results/gvsae_{MODEL_NAME}_d{dict_size}_run_{RUN_ID}"
        save_steps = [int(MAX_STEPS * 0.5), MAX_STEPS]  # Save at 50% and 100%
        
        # Log the configuration
        wandb.log({
            "dict_size": dict_size,
            "dict_size_multiple": dict_size_multiple,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "decay_start": decay_start,
            "model_name": MODEL_NAME,
            "layer": LAYER
        })
        
        # Train the model
        print(f"Starting training with dictionary size: {dict_size}")
        start_time = time.time()
        
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=total_steps,
            save_dir=save_dir,
            save_steps=save_steps,
            log_steps=LOG_STEPS,
            verbose=True,
            normalize_activations=True,
            autocast_dtype=t.bfloat16,
            use_wandb=True,  # Always use wandb for sweeps
            wandb_entity=WANDB_ENTITY,
            wandb_project=f"gvsae_{MODEL_NAME}",
            run_cfg={
                "model_type": MODEL_NAME,
                "experiment_type": "gvsae_sweep",
                "dict_size_multiple": dict_size_multiple,
                "run_id": RUN_ID
            }
        )
        
        elapsed_time = time.time() - start_time
        print(f"Completed training in {elapsed_time:.2f} seconds")
        
        # Look for the final model
        model_path = os.path.join(save_dir, "trainer_0", "ae.pt")
        if not os.path.exists(model_path):
            # Try to find latest checkpoint
            checkpoint_dir = os.path.join(save_dir, "trainer_0", "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("ae_")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                    model_path = os.path.join(checkpoint_dir, latest)
                    print(f"Using checkpoint: {model_path}")
                else:
                    print("No checkpoints found!")
                    return None
            else:
                print("No model found!")
                return None
        
        # Load the trained dictionary and evaluate
        vsae = VSAEGated.from_pretrained(model_path, device="cuda", var_flag=VAR_FLAG)
        
        # Evaluate on a small batch
        eval_results = evaluate(
            dictionary=vsae,
            activations=buffer,
            batch_size=64,
            max_len=128,
            device="cuda",
            n_batches=10
        )
        
        # Log evaluation metrics
        for metric, value in eval_results.items():
            wandb.log({"eval/" + metric: value})
        
        # Clear CUDA cache to prevent memory issues between runs
        t.cuda.empty_cache()
        
        # Return validation metric for WandB to optimize
        # Choose the most relevant metric for your use case
        return eval_results["frac_variance_explained"]

def sweep_dictionary_size(dict_size_multiple):
    """Run a sweep for a specific dictionary size multiple"""
    print(f"\n\n{'='*50}")
    print(f"STARTING SWEEP FOR DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"{'='*50}\n")
    
    # Early termination configuration
    # Hyperband requires:
    # - max_iter: maximum number of epochs/iterations
    # - eta: reduction factor between brackets (typically 2-3)
    # - s: number of brackets (higher = more rounds of elimination)
    sweep_config = {
        "method": "bayes",  # Use Bayesian optimization
        "metric": {
            "name": "eval/frac_variance_explained",  # Optimize for variance explained
            "goal": "maximize"  # We want to maximize variance explained
        },
        "parameters": {
            # Learning rate - typically small for these models
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-3
            },
            # KL coefficient - controls variational regularization strength
            "kl_coeff": {
                "distribution": "log_uniform_values",
                "min": 0.1,
                "max": 5000.0
            },
            # L1 penalty - controls sparsity of gating network
            "l1_penalty": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 1.0
            },
            # Auxiliary loss weight - controls influence of gate network loss
            "aux_weight": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 1.0
            },
            # Learning rate warm-up fraction
            "warmup_frac": {
                "values": [0.02, 0.05, 0.1]
            },
            # Sparsity warm-up fraction
            "sparsity_warmup_frac": {
                "values": [0.02, 0.05, 0.1]
            },
            # Learning rate decay start fraction
            "decay_start_frac": {
                "values": [0.7, 0.8, 0.9]
            },
            # Whether to use constrained optimizer
            "use_constrained_optimizer": {
                "values": [True, False]
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "eta": 3,            # Each bracket reduces candidates by a factor of eta
            "s": 2,              # Number of brackets
            "max_iter": MAX_STEPS # Maximum number of iterations/steps
        },
        # Run at most this many combinations
        "run_cap": MAX_SWEEP_RUNS_PER_SIZE
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=f"gvsae_{MODEL_NAME}"
    )
    
    # Start the sweep agent
    wandb.agent(
        sweep_id,
        function=lambda config: train_single_model(config, dict_size_multiple),
        count=MAX_SWEEP_RUNS_PER_SIZE
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a hyperparameter sweep for GVSAE model')
    parser.add_argument('--model', type=str, default="gelu-1l", help='Model name')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--var_flag', type=int, default=1, choices=[0, 1], help='Variational mode (0: fixed, 1: learned)')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of training steps')
    parser.add_argument('--max_runs', type=int, default=20, help='Maximum number of runs per dictionary size')
    parser.add_argument('--log_steps', type=int, default=100, help='Log frequency')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    # Global parameters
    MODEL_NAME = args.model
    LAYER = args.layer
    VAR_FLAG = args.var_flag
    MAX_STEPS = args.max_steps
    MAX_SWEEP_RUNS_PER_SIZE = args.max_runs
    LOG_STEPS = args.log_steps
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "zachdata")
    USE_WANDB = not args.no_wandb
    
    # Dictionary sizes to sweep (as multiples of model dimension)
    dict_size_multiples = [4.0, 8.0, 16.0]
    
    # Run sweep for each dictionary size
    for dict_size_multiple in dict_size_multiples:
        sweep_dictionary_size(dict_size_multiple)
