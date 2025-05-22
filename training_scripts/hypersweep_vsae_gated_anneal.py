#!/usr/bin/env python
"""
Hyperparameter sweep script for VSAEGatedAnneal model.
This script runs a Bayesian hyperparameter optimization using Weights & Biases
to find the optimal parameters for the VSAEGatedAnneal model.
"""

import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_gated_anneal import VSAEGatedAnnealTrainer, VSAEGatedAutoEncoder
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
import multiprocessing
import os
import time
import wandb
import argparse
import random
import string
from datetime import datetime
import json

def run_training(config=None, buffer=None, model=None):
    """
    Run a single training with parameters from the wandb config
    
    Args:
        config: Configuration from wandb
        buffer: Activation buffer
        model: Transformer model
        
    Returns:
        None (logs results to wandb)
    """
    with wandb.init(config=config) as run:
        # Get the config from wandb
        config = wandb.config
        
        # Calculate derived parameters
        d_mlp = model.cfg.d_mlp
        DICT_SIZE = int(config.dict_size_multiple * d_mlp)
        TOTAL_STEPS = config.steps
        SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in [0.5, 1.0]]
        ANNEAL_START = int(config.anneal_start_frac * TOTAL_STEPS)
        ANNEAL_END = int(config.anneal_end_frac * TOTAL_STEPS)
        LAYER = config.layer
        
        # Create unique run ID
        run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./VSAEGatedAnneal_{model.cfg.model_name}_{timestamp}_{run_id}"
        
        print(f"\n\n{'='*50}")
        print(f"STARTING TRAINING WITH CONFIG:")
        print(f"Dictionary size multiple: {config.dict_size_multiple} (size: {DICT_SIZE})")
        print(f"Learning rate: {config.lr}")
        print(f"KL coefficient: {config.kl_coeff}")
        print(f"P-annealing: {config.p_start} to {config.p_end}")
        print(f"VAR flag: {config.var_flag}")
        print(f"Sparsity function: {config.sparsity_function}")
        print(f"{'='*50}\n")
        
        # Configure trainer for this run
        trainer_config = {
            "trainer": VSAEGatedAnnealTrainer,
            "steps": TOTAL_STEPS,
            "activation_dim": d_mlp,
            "dict_size": DICT_SIZE,
            "layer": LAYER,
            "lm_name": model.cfg.model_name,
            "lr": config.lr,
            "kl_coeff": config.kl_coeff,
            "warmup_steps": int(config.warmup_frac * TOTAL_STEPS),
            "sparsity_warmup_steps": int(config.sparsity_warmup_frac * TOTAL_STEPS),
            "decay_start": int(config.decay_start_frac * TOTAL_STEPS),
            "var_flag": config.var_flag,
            "p_start": config.p_start,
            "p_end": config.p_end,
            "anneal_start": ANNEAL_START,
            "anneal_end": ANNEAL_END,
            "sparsity_function": config.sparsity_function,
            "n_sparsity_updates": config.n_sparsity_updates,
            "sparsity_queue_length": config.sparsity_queue_length,
            "resample_steps": int(config.resample_steps_frac * TOTAL_STEPS) if config.use_resampling else None,
            "device": "cuda" if t.cuda.is_available() else "cpu",
            "wandb_name": f"VSAEGatedAnneal_{run_id}",
            "dict_class": VSAEGatedAutoEncoder
        }
        
        # Run training with this configuration
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=TOTAL_STEPS,
            save_dir=save_dir,
            save_steps=SAVE_STEPS,
            log_steps=100,
            verbose=True,
            normalize_activations=True,
            autocast_dtype=t.bfloat16,
            use_wandb=True,  # Already in a wandb run
            run_cfg={
                "model_type": model.cfg.model_name,
                "experiment_type": "vsae_gated_anneal",
                "kl_coefficient": config.kl_coeff,
                "dict_size_multiple": config.dict_size_multiple,
                "p_start": config.p_start,
                "p_end": config.p_end,
                "var_flag": config.var_flag,
                "sparsity_function": config.sparsity_function
            }
        )
        
        # Evaluate the trained model
        try:
            from dictionary_learning.utils import load_dictionary
            
            # Load the trained dictionary
            vsae, _ = load_dictionary(f"{save_dir}/trainer_0", device="cuda" if t.cuda.is_available() else "cpu")
            
            # Evaluate on a small batch
            eval_results = evaluate(
                dictionary=vsae,
                activations=buffer,
                batch_size=64,
                max_len=config.ctx_len,
                device="cuda" if t.cuda.is_available() else "cpu",
                n_batches=10
            )
            
            # Log evaluation metrics to wandb
            wandb.log(eval_results)
            
            # Return the validation metric we want to optimize
            return eval_results["frac_variance_explained"]
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0  # Return a default value to continue the sweep
        finally:
            # Clear CUDA cache to prevent memory issues between runs
            if t.cuda.is_available():
                t.cuda.empty_cache()

def run_sweep(model_name="gelu-1l", dict_size_multiples=None, variation_name="experiment_1"):
    """
    Run a hyperparameter sweep for the VSAEGatedAnneal model.
    
    Args:
        model_name: Model name to use
        dict_size_multiples: List of dictionary size multiples to try
        variation_name: Name for the experiment variation
    """
    if dict_size_multiples is None:
        dict_size_multiples = [4, 8, 16]
    
    # Load model
    try:
        model = HookedTransformer.from_pretrained(
            model_name, 
            device="cuda" if t.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using gelu-1l as fallback")
        model_name = "gelu-1l"
        model = HookedTransformer.from_pretrained(
            "gelu-1l", 
            device="cuda" if t.cuda.is_available() else "cpu"
        )
    
    # Set up data generator
    try:
        data_gen = hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b", 
            split="train", 
            return_tokens=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using 'EleutherAI/pile-uncopyrighted' as fallback")
        data_gen = hf_dataset_to_generator(
            "EleutherAI/pile-uncopyrighted", 
            split="train", 
            return_tokens=False
        )
    
    # Create activation buffer
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=f"blocks.0.mlp.hook_post",  # Default to first layer
        d_submodule=model.cfg.d_mlp,
        n_ctxs=3000,
        ctx_len=128,
        refresh_batch_size=32,
        out_batch_size=1024,
        device="cuda" if t.cuda.is_available() else "cpu",
    )
    
    # We'll create a sweep for each dictionary size multiple
    for dict_size_multiple in dict_size_multiples:
        # Generate a unique run ID
        run_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        
        # Define sweep config for this dictionary size
        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "frac_variance_explained",
                "goal": "maximize"
            },
            "parameters": {
                # Fixed parameters
                "steps": {"value": 30000},
                "layer": {"value": 0},
                "dict_size_multiple": {"value": dict_size_multiple},
                "ctx_len": {"value": 128},
                
                # Search space
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-3
                },
                "kl_coeff": {
                    "distribution": "log_uniform_values",
                    "min": 1e3,
                    "max": 1e6
                },
                "p_start": {
                    "values": [1.0, 2.0]
                },
                "p_end": {
                    "values": [0.01, 0.05, 0.1, 0.2]
                },
                "var_flag": {
                    "values": [0, 1]
                },
                "sparsity_function": {
                    "values": ["Lp", "Lp^p"]
                },
                "n_sparsity_updates": {
                    "values": [10, 20, "continuous"]
                },
                "sparsity_queue_length": {
                    "values": [5, 10, 20]
                },
                "warmup_frac": {
                    "values": [0.03, 0.05, 0.1]
                },
                "sparsity_warmup_frac": {
                    "values": [0.03, 0.05, 0.1]
                },
                "decay_start_frac": {
                    "values": [0.7, 0.8, 0.9]
                },
                "anneal_start_frac": {
                    "values": [0.2, 0.3, 0.4]
                },
                "anneal_end_frac": {
                    "values": [0.8, 0.9]
                },
                "use_resampling": {
                    "values": [True, False]
                },
                "resample_steps_frac": {
                    "values": [0.05, 0.1, 0.2]
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "max_iter": 30000,
                "s": 2,
                "eta": 3
            }
        }
        
        # Initialize the sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config, 
            project=f"vsae_gated_anneal_{model_name}",
            entity=os.environ.get("WANDB_ENTITY", None)
        )
        
        print(f"\nStarting sweep for dictionary size multiple: {dict_size_multiple}")
        print(f"Sweep ID: {sweep_id}")
        
        # Run the sweep agent
        wandb.agent(
            sweep_id, 
            function=lambda config: run_training(config, buffer, model),
            count=10  # Run 10 experiments for each dictionary size
        )
        
        print(f"Completed sweep for dictionary size multiple: {dict_size_multiple}")
        
        # Clear CUDA cache between sweeps
        if t.cuda.is_available():
            t.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for VSAEGatedAnneal")
    parser.add_argument("--model_name", type=str, default="gelu-1l", 
                        help="Model name to use")
    parser.add_argument("--variation_name", type=str, default="experiment_1", 
                        help="Name for the experiment variation")
    parser.add_argument("--dict_sizes", type=str, default="4,8,16", 
                        help="Comma-separated list of dictionary size multiples to try")
    
    args = parser.parse_args()
    
    # Parse dictionary sizes
    dict_size_multiples = [float(x) for x in args.dict_sizes.split(",")]
    
    # Make sure wandb is logged in
    wandb.login()
    
    # Run the sweep
    run_sweep(
        model_name=args.model_name,
        dict_size_multiples=dict_size_multiples,
        variation_name=args.variation_name
    )

if __name__ == "__main__":
    # Set the start method to spawn for better compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()
