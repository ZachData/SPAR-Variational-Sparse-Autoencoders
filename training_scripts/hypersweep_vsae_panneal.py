import torch as t
import wandb
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_panneal import VSAEPAnnealTrainer, VSAEPAnneal
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
import multiprocessing
import os
import time
import argparse
import random
import uuid

def train_with_config(config):
    """
    Train a VSAEPAnneal model with the given hyperparameter configuration.
    This function is called by wandb.agent() for each run in the sweep.
    """
    # Initialize wandb run
    run_id = str(uuid.uuid4())[:8]
    wandb.init(
        project=config["project_name"],
        group=f"size_{config['dict_size_multiple']}x",
        job_type="sweep",
        name=f"sweep_{run_id}",
        config=config
    )
    
    # Get the updated config from wandb
    config = wandb.config
    
    # Extract parameters
    MODEL_NAME = config["model_name"]
    LAYER = config["layer"]
    HOOK_NAME = config["hook_name"]
    TOTAL_STEPS = config["total_steps"]
    DICT_SIZE_MULTIPLE = config["dict_size_multiple"]
    LR = config["lr"]
    SPARSITY_PENALTY = config["sparsity_penalty"]
    SPARSITY_FUNCTION = config["sparsity_function"]
    P_START = config["p_start"]
    P_END = config["p_end"]
    ANNEAL_START_FRAC = config["anneal_start_frac"]
    ANNEAL_END_FRAC = config["anneal_end_frac"]
    N_SPARSITY_UPDATES = config["n_sparsity_updates"]
    WARMUP_FRAC = config["warmup_frac"]
    SPARSITY_WARMUP_FRAC = config["sparsity_warmup_frac"]
    DECAY_START_FRAC = config["decay_start_frac"]
    VAR_FLAG = config["var_flag"]
    
    # Buffer parameters - fixed for all runs
    N_CTXS = 3000
    CTX_LEN = 128
    REFRESH_BATCH_SIZE = 32
    OUT_BATCH_SIZE = 1024
    
    # Print configuration
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH THE FOLLOWING CONFIG:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*50}\n")
    
    # Load the model
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, 
        device="cuda"
    )
    
    # Calculate dictionary size based on model and multiplier
    DICT_SIZE = int(model.cfg.d_mlp * DICT_SIZE_MULTIPLE)
    
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
        hook_name=HOOK_NAME,
        d_submodule=model.cfg.d_mlp,
        n_ctxs=N_CTXS,
        ctx_len=CTX_LEN,
        refresh_batch_size=REFRESH_BATCH_SIZE,
        out_batch_size=OUT_BATCH_SIZE,
        device="cuda",
    )
    
    # Configure trainer
    ANNEAL_START = int(ANNEAL_START_FRAC * TOTAL_STEPS)
    ANNEAL_END = int(ANNEAL_END_FRAC * TOTAL_STEPS)
    SAVE_STEPS = [int(TOTAL_STEPS)]  # Only save at the end
    
    trainer_config = {
        "trainer": VSAEPAnnealTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": model.cfg.d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "sparsity_penalty": SPARSITY_PENALTY,
        "warmup_steps": int(WARMUP_FRAC * TOTAL_STEPS),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * TOTAL_STEPS),
        "decay_start": int(DECAY_START_FRAC * TOTAL_STEPS),
        "var_flag": VAR_FLAG,
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEPAnneal_{DICT_SIZE}_p{P_START}-{P_END}",
        "dict_class": VSAEPAnneal,
        "sparsity_function": SPARSITY_FUNCTION,
        "anneal_start": ANNEAL_START,
        "anneal_end": ANNEAL_END,
        "p_start": P_START,
        "p_end": P_END,
        "n_sparsity_updates": N_SPARSITY_UPDATES
    }
    
    # Set unique save directory for this run
    save_dir = f"./sweep_results/VSAEPAnneal_d{DICT_SIZE}_run{run_id}"
    
    # Run training
    start_time = time.time()
    
    trainSAE(
        data=buffer,
        trainer_configs=[trainer_config],
        steps=TOTAL_STEPS,
        save_dir=save_dir,
        save_steps=SAVE_STEPS,
        log_steps=100,  # Log every 100 steps
        verbose=True,
        normalize_activations=True,
        autocast_dtype=t.bfloat16,
        use_wandb=True,  # Always use wandb for sweep
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=config["project_name"],
        run_cfg={
            "model_type": MODEL_NAME,
            "experiment_type": "vsae_panneal_sweep",
            "dict_size_multiple": DICT_SIZE_MULTIPLE
        }
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    
    # Load the trained dictionary
    vsae, _ = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
    
    # Evaluate on a small batch
    eval_results = evaluate(
        dictionary=vsae,
        activations=buffer,
        batch_size=64,
        max_len=CTX_LEN,
        device="cuda",
        n_batches=10
    )
    
    # Log final evaluation metrics to wandb
    wandb.log(eval_results)
    
    # Calculate validation loss (composite metric for optimization)
    # We want high variance explained and low L0 (sparsity)
    validation_loss = -eval_results["frac_variance_explained"] + 0.01 * eval_results["l0"]
    wandb.log({"validation_loss": validation_loss})
    
    # Print metrics
    print("\n===== EVALUATION RESULTS =====")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Clear CUDA cache to prevent memory issues between runs
    t.cuda.empty_cache()
    
    # Finish the wandb run
    wandb.finish()
    
    return validation_loss, save_dir

def run_sweep_for_dict_size(dict_size_multiple, args):
    """Run a sweep for a specific dictionary size multiple"""
    
    # Set project name for this sweep
    project_name = f"vsae_panneal_{args.model}_d{dict_size_multiple}x"
    
    # Create a sweep configuration
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {
            "name": "validation_loss",
            "goal": "minimize"
        },
        "parameters": {
            # Fixed parameters
            "model_name": {"value": args.model},
            "layer": {"value": args.layer},
            "hook_name": {"value": f"blocks.{args.layer}.mlp.hook_post"},
            "total_steps": {"value": args.steps},
            "dict_size_multiple": {"value": dict_size_multiple},
            "project_name": {"value": project_name},
            
            # Hyperparameters to sweep over
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3
            },
            "sparsity_penalty": {
                "distribution": "log_uniform_values",
                "min": 0.1,
                "max": 50.0
            },
            "sparsity_function": {
                "values": ["Lp", "Lp^p"]
            },
            "p_start": {
                "values": [1.0, 2.0]  # Try L1 and L2 regularization as starting points
            },
            "p_end": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.9
            },
            "anneal_start_frac": {
                "distribution": "uniform",
                "min": 0.05,
                "max": 0.3
            },
            "anneal_end_frac": {
                "distribution": "uniform",
                "min": 0.7,
                "max": 0.95
            },
            "n_sparsity_updates": {
                "values": [5, 10, 20, "continuous"]
            },
            "warmup_frac": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.1
            },
            "sparsity_warmup_frac": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.1
            },
            "decay_start_frac": {
                "distribution": "uniform",
                "min": 0.7,
                "max": 0.9
            },
            "var_flag": {
                "values": [0, 1]  # Try both fixed and learned variance
            },
        },
        "early_terminate": {
            "type": "hyperband",
            "max_iter": args.steps,
            "s": 2,
            "eta": 3
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    # Run the sweep
    print(f"\n\n{'='*50}")
    print(f"STARTING SWEEP FOR DICTIONARY SIZE {dict_size_multiple}x")
    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project_name}")
    print(f"{'='*50}\n")
    
    # Run the agent
    wandb.agent(sweep_id, function=train_with_config, count=args.runs_per_size)
    
    return sweep_id

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for VSAEPAnneal model")
    
    # Basic parameters
    parser.add_argument("--model", default="gelu-1l", help="Model name to use")
    parser.add_argument("--layer", type=int, default=0, help="Layer to train on")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps per run")
    parser.add_argument("--runs-per-size", type=int, default=20, help="Number of runs per dictionary size")
    parser.add_argument("--sizes", nargs="+", type=float, default=[4, 8, 16], help="Dictionary size multiples to sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    t.manual_seed(args.seed)
    
    # Create directory for saving results
    os.makedirs("./sweep_results", exist_ok=True)
    
    # Run sweeps for each dictionary size multiple
    sweep_ids = []
    for dict_size_multiple in args.sizes:
        sweep_id = run_sweep_for_dict_size(dict_size_multiple, args)
        sweep_ids.append((dict_size_multiple, sweep_id))
    
    # Print summary of all sweeps
    print("\n\n" + "="*50)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Steps per run: {args.steps}")
    print(f"Runs per size: {args.runs_per_size}")
    print("\nSweep IDs:")
    for size, sweep_id in sweep_ids:
        print(f"  Dictionary size {size}x: {sweep_id}")
    print("="*50)

if __name__ == "__main__":
    # Set the start method to spawn for multiprocessing compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()