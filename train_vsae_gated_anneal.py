#!/usr/bin/env python
"""
Training script for VSAEGatedAnneal.
Combines variational sparse autoencoder (VSAE) with p-annealing for improved feature learning.
"""

import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_gated_anneal import VSAEGatedAnnealTrainer, VSAEGatedAutoEncoder
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time
import argparse

def run_training(dict_size_multiple, model, buffer, base_params):
    """
    Run a single training with the specified dictionary size multiple
    
    Args:
        dict_size_multiple: Factor to multiply the model's hidden dimension by to get dictionary size
        model: Transformer model
        buffer: Activation buffer
        base_params: Dictionary of base parameters
        
    Returns:
        Evaluation results
    """
    # Extract base parameters
    MODEL_NAME = base_params["MODEL_NAME"]
    LAYER = base_params["LAYER"]
    TOTAL_STEPS = base_params["TOTAL_STEPS"]
    LR = base_params["LR"]
    KL_COEFF = base_params["KL_COEFF"]
    WARMUP_FRAC = base_params["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = base_params["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = base_params["DECAY_START_FRAC"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_FRACS = base_params["CHECKPOINT_FRACS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    
    # p-annealing specific parameters
    P_START = base_params["P_START"]
    P_END = base_params["P_END"]
    ANNEAL_START_FRAC = base_params["ANNEAL_START_FRAC"]
    ANNEAL_END_FRAC = base_params["ANNEAL_END_FRAC"]
    N_SPARSITY_UPDATES = base_params["N_SPARSITY_UPDATES"]
    SPARSITY_FUNCTION = base_params["SPARSITY_FUNCTION"]
    SPARSITY_QUEUE_LENGTH = base_params["SPARSITY_QUEUE_LENGTH"]
    VAR_FLAG = base_params["VAR_FLAG"]
    RESAMPLE_STEPS = base_params.get("RESAMPLE_STEPS", None)
    
    # Calculate derived parameters
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(dict_size_multiple * d_mlp)
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    ANNEAL_START = int(ANNEAL_START_FRAC * TOTAL_STEPS)
    ANNEAL_END = int(ANNEAL_END_FRAC * TOTAL_STEPS)
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": VSAEGatedAnnealTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "kl_coeff": KL_COEFF,
        "warmup_steps": int(WARMUP_FRAC * TOTAL_STEPS),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * TOTAL_STEPS),
        "decay_start": int(DECAY_START_FRAC * TOTAL_STEPS),
        "var_flag": VAR_FLAG,
        "p_start": P_START,
        "p_end": P_END,
        "anneal_start": ANNEAL_START,
        "anneal_end": ANNEAL_END,
        "sparsity_function": SPARSITY_FUNCTION,
        "n_sparsity_updates": N_SPARSITY_UPDATES,
        "sparsity_queue_length": SPARSITY_QUEUE_LENGTH,
        "resample_steps": RESAMPLE_STEPS,
        "device": "cuda",
        "wandb_name": f"VSAEGatedAnneal_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_kl{KL_COEFF}",
        "dict_class": VSAEGatedAutoEncoder
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items() if k != "trainer" and k != "dict_class"))
    
    # Set unique save directory for this run
    save_dir = f"./VSAEGatedAnneal_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_p{P_START}to{P_END}_var{VAR_FLAG}"
    
    # Run training
    trainSAE(
        data=buffer,
        trainer_configs=[trainer_config],
        steps=TOTAL_STEPS,
        save_dir=save_dir,
        save_steps=SAVE_STEPS,
        log_steps=LOG_STEPS,
        verbose=True,
        normalize_activations=True,
        autocast_dtype=t.bfloat16,
        use_wandb=USE_WANDB,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        run_cfg={
            "model_type": MODEL_NAME,
            "experiment_type": "vsae_gated_anneal",
            "kl_coefficient": KL_COEFF,
            "dict_size_multiple": dict_size_multiple,
            "p_start": P_START,
            "p_end": P_END,
            "var_flag": VAR_FLAG,
            "sparsity_function": SPARSITY_FUNCTION
        }
    )
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.evaluation import evaluate
    
    # Load the trained dictionary
    vsae, config = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
    
    # Evaluate on a small batch
    eval_results = evaluate(
        dictionary=vsae,
        activations=buffer,
        batch_size=64,
        max_len=base_params["CTX_LEN"],
        device="cuda",
        n_batches=10
    )
    
    # Print metrics
    print("\n===== EVALUATION RESULTS =====")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Clear CUDA cache to prevent memory issues between runs
    t.cuda.empty_cache()
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Train a VSAEGatedAnneal model")
    parser.add_argument("--model_name", type=str, default="gelu-1l", help="Model name")
    parser.add_argument("--layer", type=int, default=0, help="Layer to train on")
    parser.add_argument("--dict_size_multiple", type=float, default=4.0, help="Dictionary size multiple")
    parser.add_argument("--steps", type=int, default=30000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--kl_coeff", type=float, default=5e4, help="KL coefficient")
    parser.add_argument("--p_start", type=float, default=1.0, help="Starting p value for annealing")
    parser.add_argument("--p_end", type=float, default=0.1, help="Ending p value for annealing")
    parser.add_argument("--var_flag", type=int, default=1, help="Use learned variance (1) or fixed (0)")
    parser.add_argument("--sparsity_function", type=str, default="Lp^p", help="Sparsity function: 'Lp' or 'Lp^p'")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity")
    parser.add_argument("--wandb_project", type=str, default="vsae-experiments", help="W&B project name")
    
    args = parser.parse_args()
    
    # ========== HYPERPARAMETERS ==========
    # Define base parameters that will be used for all runs
    base_params = {
        # Model parameters
        "MODEL_NAME": args.model_name,
        "LAYER": args.layer,
        "HOOK_NAME": f"blocks.{args.layer}.mlp.hook_post",
        
        # Training parameters        
        "TOTAL_STEPS": args.steps,
        "LR": args.lr,
        "KL_COEFF": args.kl_coeff,
        
        # Step fractions
        "WARMUP_FRAC": 0.05,
        "SPARSITY_WARMUP_FRAC": 0.05,
        "DECAY_START_FRAC": 0.8,
        
        # p-annealing parameters
        "P_START": args.p_start,
        "P_END": args.p_end,
        "ANNEAL_START_FRAC": 0.3,  # Start annealing at 30% of training
        "ANNEAL_END_FRAC": 0.9,    # End annealing at 90% of training
        "N_SPARSITY_UPDATES": 20,   # Number of p updates during annealing
        "SPARSITY_FUNCTION": args.sparsity_function,
        "SPARSITY_QUEUE_LENGTH": 10,
        "VAR_FLAG": args.var_flag,
        "RESAMPLE_STEPS": int(args.steps * 0.1),  # Resample dead neurons every 10% of training
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],
        "LOG_STEPS": 100,
        
        # WandB parameters
        "WANDB_ENTITY": args.wandb_entity if args.wandb_entity else os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": args.wandb_project if args.wandb_project else os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": args.use_wandb,
    }
    
    dict_size_multiple = args.dict_size_multiple
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    try:
        model = HookedTransformer.from_pretrained(
            base_params["MODEL_NAME"], 
            device="cuda"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using gelu-1l as fallback")
        base_params["MODEL_NAME"] = "gelu-1l"
        model = HookedTransformer.from_pretrained(
            "gelu-1l", 
            device="cuda"
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
        hook_name=base_params["HOOK_NAME"],
        d_submodule=model.cfg.d_mlp,
        n_ctxs=base_params["N_CTXS"],
        ctx_len=base_params["CTX_LEN"],
        refresh_batch_size=base_params["REFRESH_BATCH_SIZE"],
        out_batch_size=base_params["OUT_BATCH_SIZE"],
        device="cuda",
    )
    
    # ========== RUN TRAINING ==========
    print(f"\nStarting training with dictionary size multiple: {dict_size_multiple}")
    start_time = time.time()
    
    # Run training with this dictionary size
    results = run_training(dict_size_multiple, model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    print(f"Completed training in {elapsed_time:.2f} seconds")
    
    # ========== PRINT RESULTS =====
    print("\n\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Print key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive"]
    dict_size = int(dict_size_multiple * model.cfg.d_mlp)
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    print(f"{dict_size:<15} | " + " | ".join(f"{results[metric]:<22.4f}" for metric in metrics))

if __name__ == "__main__":
    # Set the start method to spawn for better compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()
