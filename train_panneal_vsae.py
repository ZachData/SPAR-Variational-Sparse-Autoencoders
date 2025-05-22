import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_panneal import VSAEPAnnealTrainer, VSAEPAnneal
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
    SPARSITY_PENALTY = base_params["SPARSITY_PENALTY"]
    WARMUP_STEPS = base_params["WARMUP_STEPS"]
    SPARSITY_WARMUP_STEPS = base_params["SPARSITY_WARMUP_STEPS"]
    DECAY_START_STEP = base_params["DECAY_START_STEP"]
    P_START = base_params["P_START"]
    P_END = base_params["P_END"]
    ANNEAL_START = base_params["ANNEAL_START"]
    ANNEAL_END = base_params["ANNEAL_END"]
    N_SPARSITY_UPDATES = base_params["N_SPARSITY_UPDATES"]
    SPARSITY_FUNCTION = base_params["SPARSITY_FUNCTION"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_STEPS = base_params["CHECKPOINT_STEPS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    
    # Calculate dictionary size from multiple
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(dict_size_multiple * d_mlp)
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"P-Annealing: {P_START} -> {P_END}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": VSAEPAnnealTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "sparsity_penalty": SPARSITY_PENALTY,
        "warmup_steps": WARMUP_STEPS,
        "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS,
        "decay_start": DECAY_START_STEP,
        "var_flag": 0,  # Use fixed variance
        "use_april_update_mode": True,
        "p_start": P_START,
        "p_end": P_END,
        "anneal_start": ANNEAL_START,
        "anneal_end": ANNEAL_END,
        "n_sparsity_updates": N_SPARSITY_UPDATES,
        "sparsity_function": SPARSITY_FUNCTION,
        "device": "cuda",
        "wandb_name": f"VSAEPAnneal_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_sp{SPARSITY_PENALTY}_p{P_START}-{P_END}",
        "dict_class": VSAEPAnneal
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    for k, v in trainer_config.items():
        if k != "trainer" and k != "dict_class":
            print(f"{k}: {v}")
    
    # Set unique save directory for this run
    save_dir = f"./VSAEPAnneal_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_sp{SPARSITY_PENALTY}_p{P_START}-{P_END}"
    
    # Check if we're loading from a checkpoint or training from scratch
    checkpoint_path = f"{save_dir}/trainer_0/checkpoints"
    is_continuing = os.path.exists(checkpoint_path)
    
    if is_continuing:
        print(f"Loading existing SAE from {checkpoint_path}")
        print(f"Continuing training from checkpoint")
    else:
        print(f"No existing checkpoint found at {checkpoint_path}")
        print(f"Starting training from scratch")
    
    # Run training
    trainSAE(
        data=buffer,
        trainer_configs=[trainer_config],
        steps=TOTAL_STEPS,
        save_dir=save_dir,
        save_steps=CHECKPOINT_STEPS,
        log_steps=LOG_STEPS,
        verbose=True,
        normalize_activations=True,
        autocast_dtype=t.bfloat16,
        use_wandb=USE_WANDB,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        run_cfg={
            "model_type": MODEL_NAME,
            "experiment_type": "panneal_vsae",
            "sparsity_penalty": SPARSITY_PENALTY,
            "p_start": P_START,
            "p_end": P_END,
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
        "TOTAL_STEPS": 10000,
        "LR": 5e-4,  # 0.0005
        "SPARSITY_PENALTY": 5e2,  # 500
        
        # Fixed step values instead of fractions
        "WARMUP_STEPS": 200,          # Warmup period for learning rate
        "SPARSITY_WARMUP_STEPS": 500, # Warmup period for sparsity
        "DECAY_START_STEP": 8000,     # When to start LR decay
        
        # P-annealing parameters
        "P_START": 1.0,              # Starting p-norm value
        "P_END": 0.5,                # Ending p-norm value
        "ANNEAL_START": 1000,        # When to start annealing p
        "ANNEAL_END": 8000,          # When to end annealing p
        "N_SPARSITY_UPDATES": 10,    # How many times to update p
        "SPARSITY_FUNCTION": "Lp",   # Can be "Lp" or "Lp^p"
        
        # Checkpointing
        "CHECKPOINT_STEPS": [5000, 10000],  # Save at these steps
        "LOG_STEPS": 100,
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "zachdata"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": True,
    }

    # Only train with 4x dictionary size multiple
    dict_size_multiple = 4
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        base_params["MODEL_NAME"], 
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
    
    # Run training
    results = run_training(dict_size_multiple, model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    print(f"Completed training in {elapsed_time:.2f} seconds")
    
    # ========== PRINT RESULTS ==========
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
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()