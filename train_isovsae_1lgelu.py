import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_iso import VSAEIsoTrainer, VSAEIsoGaussian
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
        "trainer": VSAEIsoTrainer,
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
        "var_flag": 1,  # Use fixed variance
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEIso_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_warm{WARMUP_FRAC}_spwarm{SPARSITY_WARMUP_FRAC}",
        "dict_class": VSAEIsoGaussian
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set unique save directory for this run
    save_dir = f"./VSAEIso_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_warm{WARMUP_FRAC}_spwarm{SPARSITY_WARMUP_FRAC}"
    
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
            "experiment_type": "isovsae",
            "kl_coefficient": KL_COEFF,
            "dict_size_multiple": dict_size_multiple
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
        "TOTAL_STEPS": 30000,
        "LR": 5e-5,  # Updated from 1e-2 to 0.0005
        "KL_COEFF": 500000,  # Updated from 1e2 to 50
        
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
        "USE_WANDB": True,
    }
    
    # Only train with 4x dictionary size multiple
    dict_size_multiples = [4]
    
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
    
    # ========== RUN TRAINING ==========
    print(f"\nStarting training with dictionary size multiple: {dict_size_multiples[0]}")
    start_time = time.time()
    
    # Run training with this dictionary size
    results = run_training(dict_size_multiples[0], model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    print(f"Completed training in {elapsed_time:.2f} seconds")
    
    # ========== PRINT RESULTS ==========
    print("\n\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Print key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive"]
    dict_size = int(dict_size_multiples[0] * model.cfg.d_mlp)
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    print(f"{dict_size:<15} | " + " | ".join(f"{results[metric]:<22.4f}" for metric in metrics))

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()