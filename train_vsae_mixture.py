import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_mixture import VSAEMixtureTrainer, VSAEMixtureGaussian
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time

def run_training(dict_size_multiple, model, buffer, base_params):
    """Run a single training with the specified dictionary size multiple"""
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
    N_CORRELATED_PAIRS = base_params["N_CORRELATED_PAIRS"]
    N_ANTICORRELATED_PAIRS = base_params["N_ANTICORRELATED_PAIRS"]
    VAR_FLAG = base_params["VAR_FLAG"]
    
    # Calculate derived parameters
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(dict_size_multiple * d_mlp)
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"Correlated pairs: {N_CORRELATED_PAIRS}, Anticorrelated pairs: {N_ANTICORRELATED_PAIRS}")
    print(f"Variance flag: {VAR_FLAG}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": VSAEMixtureTrainer,
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
        "n_correlated_pairs": N_CORRELATED_PAIRS,
        "n_anticorrelated_pairs": N_ANTICORRELATED_PAIRS,
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEMixture_{MODEL_NAME}_{DICT_SIZE}_{N_CORRELATED_PAIRS}cp_{N_ANTICORRELATED_PAIRS}ap",
        "dict_class": VSAEMixtureGaussian
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set unique save directory for this run
    save_dir = f"./trained_vsae_mix_{DICT_SIZE}_{N_CORRELATED_PAIRS}cp_{N_ANTICORRELATED_PAIRS}ap"
    
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
            "experiment_type": "vsae_mixture",
            "kl_coefficient": KL_COEFF,
            "dict_size_multiple": dict_size_multiple,
            "n_correlated_pairs": N_CORRELATED_PAIRS,
            "n_anticorrelated_pairs": N_ANTICORRELATED_PAIRS,
            "var_flag": VAR_FLAG
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
    # ========== HYPERPARAMETERS ==========
    # Define base parameters that will be used for all runs
    base_params = {
        # Model parameters
        "MODEL_NAME": "gelu-1l",
        "LAYER": 0,
        "HOOK_NAME": "blocks.0.mlp.hook_post",
        
        # Training parameters        
        "TOTAL_STEPS": 10000,
        "LR": 1e-3,
        "KL_COEFF": 1e1,
        
        # Step fractions
        "WARMUP_FRAC": 0.05,
        "SPARSITY_WARMUP_FRAC": 0.05,
        "DECAY_START_FRAC": 0.8,
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],
        "LOG_STEPS": 100,
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "zachdata"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": False,
        
        # VSAE Mixture specific parameters
        "N_CORRELATED_PAIRS": 1,
        "N_ANTICORRELATED_PAIRS": 1,
        "VAR_FLAG": 0,  # 0 for fixed variance, 1 for learned variance
    }
    
    # List of dictionary size multipliers to run
    dict_size_multiples = [4]
    
    # ========== COMMAND LINE ARGUMENT PARSING ==========
    import argparse
    parser = argparse.ArgumentParser(description='Train VSAE with Gaussian mixture prior')
    parser.add_argument('--model', type=str, default=base_params["MODEL_NAME"], 
                        help='Model name (default: gelu-1l)')
    parser.add_argument('--dict_size_multiples', type=str, default=','.join(map(str, dict_size_multiples)),
                        help='Dictionary size multiples separated by commas (default: 4,8,16)')
    parser.add_argument('--corr_pairs', type=int, default=base_params["N_CORRELATED_PAIRS"],
                        help='Number of correlated pairs (default: 1)')
    parser.add_argument('--anticorr_pairs', type=int, default=base_params["N_ANTICORRELATED_PAIRS"],
                        help='Number of anticorrelated pairs (default: 1)')
    parser.add_argument('--var_flag', type=int, default=base_params["VAR_FLAG"], choices=[0, 1],
                        help='Variance flag: 0 for fixed, 1 for learned (default: 0)')
    parser.add_argument('--steps', type=int, default=base_params["TOTAL_STEPS"],
                        help='Total training steps (default: 20000)')
    parser.add_argument('--kl_coeff', type=float, default=base_params["KL_COEFF"],
                        help='KL divergence coefficient (default: 100)')
    
    args = parser.parse_args()
    
    # Update base parameters with command line arguments
    base_params["MODEL_NAME"] = args.model
    base_params["TOTAL_STEPS"] = args.steps
    base_params["KL_COEFF"] = args.kl_coeff
    base_params["N_CORRELATED_PAIRS"] = args.corr_pairs
    base_params["N_ANTICORRELATED_PAIRS"] = args.anticorr_pairs
    base_params["VAR_FLAG"] = args.var_flag
    
    # Parse dictionary size multiples
    dict_size_multiples = [int(x) for x in args.dict_size_multiples.split(',')]
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    # Only load the model once for all runs
    model = HookedTransformer.from_pretrained(
        base_params["MODEL_NAME"], 
        device="cuda"
    )
    
    # Set up data generator
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b", 
        split="train", 
        return_tokens=True
    )
    
    # Create activation buffer (reused for all runs)
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
    
    # ========== RUN SEQUENTIAL TRAININGS ==========
    all_results = {}
    
    for multiple in dict_size_multiples:
        print(f"\nStarting training with dictionary size multiple: {multiple}")
        start_time = time.time()
        
        # Run training with this dictionary size
        results = run_training(multiple, model, buffer, base_params)
        
        # Store results
        all_results[multiple] = results
        
        elapsed_time = time.time() - start_time
        print(f"Completed training with multiple {multiple} in {elapsed_time:.2f} seconds")
        
        # Short pause between runs
        print("Cooling down before next run...")
        time.sleep(10)
    
    # ========== PRINT COMPARATIVE RESULTS ==========
    print("\n\n" + "="*50)
    print("COMPARATIVE RESULTS ACROSS ALL RUNS")
    print("="*50)
    
    # Print comparison table of key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive"]
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    
    for multiple in dict_size_multiples:
        dict_size = int(multiple * model.cfg.d_mlp)
        print(f"{dict_size:<15} | " + " | ".join(f"{all_results[multiple][metric]:<22.4f}" for metric in metrics))

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()