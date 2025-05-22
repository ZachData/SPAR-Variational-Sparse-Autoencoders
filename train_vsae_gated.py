import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_gated import VSAEGatedTrainer, VSAEGated
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
    L1_PENALTY = base_params["L1_PENALTY"]
    AUX_WEIGHT = base_params["AUX_WEIGHT"]
    WARMUP_STEPS = base_params["WARMUP_STEPS"]
    SPARSITY_WARMUP_STEPS = base_params["SPARSITY_WARMUP_STEPS"]
    DECAY_START_STEP = base_params["DECAY_START_STEP"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_STEPS = base_params["CHECKPOINT_STEPS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    VAR_FLAG = base_params["VAR_FLAG"]
    USE_CONSTRAINED = base_params["USE_CONSTRAINED"]
    
    # Calculate dictionary size from multiple
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(dict_size_multiple * d_mlp)
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE_MULTIPLE = {dict_size_multiple}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": VSAEGatedTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "kl_coeff": KL_COEFF,
        "l1_penalty": L1_PENALTY,
        "aux_weight": AUX_WEIGHT,
        "warmup_steps": WARMUP_STEPS,
        "sparsity_warmup_steps": SPARSITY_WARMUP_STEPS,
        "decay_start": DECAY_START_STEP,
        "var_flag": VAR_FLAG,
        "use_constrained_optimizer": USE_CONSTRAINED,
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEGated_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_l1{L1_PENALTY}_aux{AUX_WEIGHT}_var{VAR_FLAG}",
        "dict_class": VSAEGated
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set unique save directory for this run
    save_dir = f"./VSAEGated_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_l1{L1_PENALTY}_aux{AUX_WEIGHT}_var{VAR_FLAG}"
    
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
            "experiment_type": "gatedvsae",
            "kl_coefficient": KL_COEFF,
            "l1_penalty": L1_PENALTY,
            "aux_weight": AUX_WEIGHT,
            "var_flag": VAR_FLAG,
            "use_constrained": USE_CONSTRAINED,
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
        "TOTAL_STEPS": 10,
        "LR": 5e-4,             # 0.0005
        "KL_COEFF": 5e2,        # 500
        "L1_PENALTY": 0.1,      # Penalty for L1 on gating network
        "AUX_WEIGHT": 0.1,      # Weight for auxiliary reconstruction loss
        "VAR_FLAG": 1,          # Use learned variance (1) or fixed (0)
        "USE_CONSTRAINED": True, # Use constrained optimizer for decoder weights
        
        # Fixed step values instead of fractions
        "WARMUP_STEPS": 2,          # Warmup period for learning rate
        "SPARSITY_WARMUP_STEPS": 5, # Warmup period for sparsity
        "DECAY_START_STEP": 8,     # When to start LR decay
        
        # Checkpointing
        "CHECKPOINT_STEPS": [10],  # Save at these steps
        "LOG_STEPS": 10,
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "zachdata"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": False,
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
    metrics = ["frac_variance_explained", "l0", "frac_alive", "frac_recovered"]
    dict_size = int(dict_size_multiples[0] * model.cfg.d_mlp)
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    print(f"{dict_size:<15} | " + " | ".join(f"{results.get(metric, 0):<22.4f}" for metric in metrics))

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()