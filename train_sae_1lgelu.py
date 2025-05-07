import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.standard import StandardTrainerAprilUpdate
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import load_dictionary
from dictionary_learning.evaluation import evaluate
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
    L1_PENALTY = base_params["L1_PENALTY"]
    WARMUP_STEPS = base_params["WARMUP_STEPS"]
    SPARSITY_WARMUP_STEPS = base_params["SPARSITY_WARMUP_STEPS"]
    # RESAMPLE_STEPS = base_params["RESAMPLE_STEPS"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_FRACS = base_params["CHECKPOINT_FRACS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    
    # Calculate derived parameters
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(dict_size_multiple * d_mlp)
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": StandardTrainerAprilUpdate,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "l1_penalty": L1_PENALTY,
        "warmup_steps": WARMUP_STEPS,
        "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS,
        # "resample_steps": RESAMPLE_STEPS,
        "device": "cuda",
        "wandb_name": f"StandardSAE_{MODEL_NAME}_{DICT_SIZE}"
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set unique save directory for this run
    save_dir = f"./trained_sae_{DICT_SIZE}"
    
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
            "experiment_type": "standard_sae",
            "l1_penalty": L1_PENALTY,
            "dict_size_multiple": dict_size_multiple
        }
    )
    
    # Evaluate the trained model
    # Load the trained dictionary
    sae, config = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
    
    # Evaluate on a small batch
    eval_results = evaluate(
        dictionary=sae,
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
        "L1_PENALTY": 1e-1,
        
        # Step parameters
        "WARMUP_STEPS": 1000,
        "SPARSITY_WARMUP_STEPS": 1000,
        # "RESAMPLE_STEPS": 2000,
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],
        "LOG_STEPS": 1000,
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "zachdata"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "sae-experiments"),
        "USE_WANDB": True,
    }
    
    # List of dictionary size multipliers to run
    dict_size_multiples = [4, 8, 16]
    
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