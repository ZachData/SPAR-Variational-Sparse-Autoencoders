"""
Hyperparameter sweep for VSAEJumpReLU.
This script performs Bayesian hyperparameter optimization for the VSAEJumpReLU model
across different dictionary sizes (4x, 8x, and 16x).

Usage:
    python sweep_vsae_jump_relu.py [--options]

Example:
    python sweep_vsae_jump_relu.py --model gelu-1l --wandb_project vsae_jump_relu_sweeps
"""

import os
import argparse
import time
import wandb
import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_jump_relu import VSAEJumpReLUTrainer, VSAEJumpReLU
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for VSAEJumpReLU model')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="gelu-1l", help='Model name')
    parser.add_argument('--layer', type=int, default=0, help='Layer to extract activations from')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=10000, 
                        help='Total training steps (reduced for sweeps)')
    
    # Buffer parameters
    parser.add_argument('--n_ctxs', type=int, default=2000, help='Number of contexts for buffer')
    parser.add_argument('--ctx_len', type=int, default=128, help='Context length')
    parser.add_argument('--refresh_batch', type=int, default=32, help='Refresh batch size')
    parser.add_argument('--out_batch', type=int, default=1024, help='Output batch size')
    
    # Sweep parameters
    parser.add_argument('--sweep_runs', type=int, default=20, 
                        help='Number of runs per sweep')
    parser.add_argument('--sweep_count', type=int, default=5, 
                        help='Number of runs per agent')
    
    # WandB parameters
    parser.add_argument('--wandb_project', type=str, default='vsae_jump_relu_sweeps',
                       help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='',
                       help='WandB entity (username or team name)')
    
    return parser.parse_args()


def train_model_from_config():
    """
    Training function to be called by wandb.agent.
    Uses the hyperparameters from wandb.config to train the model.
    """
    # Initialize a new wandb run with the config set by the sweep controller
    with wandb.init() as run:
        config = wandb.config
        
        # Retrieve run context
        model = run.config.model
        layer = run.config.layer
        d_mlp = run.config.d_mlp
        dict_size = int(run.config.dict_size_multiple * d_mlp)
        total_steps = run.config.steps
        data_gen = run.config.data_gen
        buffer = run.config.buffer
        
        # Configure trainer based on sweep hyperparameters
        trainer_config = {
            "trainer": VSAEJumpReLUTrainer,
            "steps": total_steps,
            "activation_dim": d_mlp,
            "dict_size": dict_size,
            "layer": layer,
            "lm_name": model,
            "lr": config.lr,
            "kl_coeff": config.kl_coeff,
            "l0_coeff": config.l0_coeff,
            "target_l0": config.target_l0,
            "warmup_steps": int(config.warmup_frac * total_steps),
            "sparsity_warmup_steps": int(config.sparsity_warmup_frac * total_steps),
            "decay_start": int(config.decay_start_frac * total_steps),
            "var_flag": config.var_flag,
            "bandwidth": config.bandwidth,
            "use_april_update_mode": True,
            "device": "cuda",
            "wandb_name": f"VSAEJumpReLU_{model}_{dict_size}_sweep",
            "dict_class": VSAEJumpReLU
        }
        
        # Create a unique run name with timestamp for the save directory
        timestamp = int(time.time())
        run_id = f"{timestamp}_{wandb.run.id}"
        save_dir = f"./sweep_results/VSAEJumpReLU_{model}_d{dict_size}_{run_id}"
        
        # Train the model
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=total_steps,
            save_dir=save_dir,
            save_steps=[total_steps],  # Only save at the end
            log_steps=100,
            verbose=False,  # Reduce verbosity for sweeps
            normalize_activations=True,
            autocast_dtype=t.bfloat16,
            use_wandb=True,  # Already inside a wandb run
            wandb_entity=run.config.wandb_entity,
            wandb_project=run.config.wandb_project,
        )
        
        # Evaluate the trained model
        from dictionary_learning.utils import load_dictionary
        from dictionary_learning.evaluation import evaluate
        
        # Load the trained dictionary
        vsae, _ = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
        
        # Evaluate on a small batch
        eval_results = evaluate(
            dictionary=vsae,
            activations=buffer,
            batch_size=64,
            max_len=run.config.ctx_len,
            device="cuda",
            n_batches=10
        )
        
        # Log final evaluation metrics to wandb
        for metric_name, metric_value in eval_results.items():
            wandb.log({f"final_{metric_name}": metric_value})
        
        # Set the important metrics for sweep optimization
        wandb.log({
            "validation_loss": -eval_results["frac_variance_explained"],
            "final_l0": eval_results["l0"],
            "final_frac_alive": eval_results["frac_alive"],
            "final_frac_recovered": eval_results.get("frac_recovered", 0.0)
        })
        
        # Return metrics for sweep optimization
        return eval_results


def create_sweep_config(dict_size_multiple):
    """Create a sweep configuration for the given dictionary size multiple"""
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "validation_loss",
            "goal": "minimize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-3
            },
            "kl_coeff": {
                "distribution": "log_uniform_values",
                "min": 0.1,
                "max": 1000.0
            },
            "l0_coeff": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 100.0
            },
            "target_l0": {
                "distribution": "q_uniform",
                "min": 5,
                "max": 50,
                "q": 1
            },
            "bandwidth": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-2
            },
            "var_flag": {
                "values": [0, 1]
            },
            "warmup_frac": {
                "distribution": "uniform",
                "min": 0.02,
                "max": 0.2
            },
            "sparsity_warmup_frac": {
                "distribution": "uniform",
                "min": 0.02,
                "max": 0.2
            },
            "decay_start_frac": {
                "distribution": "uniform",
                "min": 0.6,
                "max": 0.9
            },
            "dict_size_multiple": {
                "value": dict_size_multiple
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 100,
            "eta": 3,
            "max_iter": 1000
        }
    }
    
    return sweep_config


def run_sweep(args, model, buffer, dict_size_multiple):
    """Run a sweep for a specific dictionary size multiple"""
    # Create a sweep configuration
    sweep_config = create_sweep_config(dict_size_multiple)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.wandb_project,
        entity=args.wandb_entity
    )
    
    print(f"\n\n{'='*50}")
    print(f"Starting sweep for dictionary size multiple: {dict_size_multiple}x")
    print(f"Sweep ID: {sweep_id}")
    print(f"{'='*50}\n")
    
    # Create a function to initialize runs with common config
    def sweep_run():
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=f"dict_size_{dict_size_multiple}x",
            job_type="sweep",
            name=f"sweep_{dict_size_multiple}x_{wandb.run.id}"
        )
        
        # Add fixed parameters that won't be swept
        run.config.update({
            "model": args.model,
            "layer": args.layer,
            "d_mlp": model.cfg.d_mlp,
            "steps": args.steps,
            "ctx_len": args.ctx_len,
            "data_gen": None,  # Can't serialize the generator
            "buffer": buffer,  # Pass the buffer directly
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project
        })
        
        return train_model_from_config()
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)
    
    return sweep_id


def main():
    # Set up multiprocessing method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    args = parse_args()
    
    # Dictionary size multiples to sweep over
    dict_size_multiples = [4, 8, 16]
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    print(f"Loading model: {args.model}")
    model = HookedTransformer.from_pretrained(
        args.model, 
        device="cuda"
    )
    
    # Set up data generator
    print("Setting up data generator...")
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b", 
        split="train", 
        return_tokens=True
    )
    
    # Create activation buffer
    print("Creating activation buffer...")
    hook_name = f"blocks.{args.layer}.mlp.hook_post"
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=hook_name,
        d_submodule=model.cfg.d_mlp,
        n_ctxs=args.n_ctxs,
        ctx_len=args.ctx_len,
        refresh_batch_size=args.refresh_batch,
        out_batch_size=args.out_batch,
        device="cuda",
    )
    
    # Create directory for sweep results
    os.makedirs("./sweep_results", exist_ok=True)
    
    # Run sweeps for each dictionary size multiple
    sweep_ids = []
    for dict_size_multiple in dict_size_multiples:
        sweep_id = run_sweep(args, model, buffer, dict_size_multiple)
        sweep_ids.append((dict_size_multiple, sweep_id))
        
        # Clear CUDA cache between sweeps
        t.cuda.empty_cache()
    
    # Print summary of all sweeps
    print("\n\n" + "="*50)
    print("SWEEP SUMMARY")
    print("="*50)
    for dict_size_multiple, sweep_id in sweep_ids:
        print(f"Dictionary Size Multiple: {dict_size_multiple}x")
        print(f"Sweep ID: {sweep_id}")
        print(f"WandB URL: https://wandb.ai/{args.wandb_entity or 'user'}/{args.wandb_project}/sweeps/{sweep_id.split('/')[-1]}")
        print("-" * 50)
    
    print("\nTo find the best models, check the WandB dashboard and look for runs with the lowest validation_loss values.")
    print("The best hyperparameters will be displayed in the sweep results.")


if __name__ == "__main__":
    main()
