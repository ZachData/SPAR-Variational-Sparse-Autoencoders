import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_mixture import VSAEMixtureTrainer, VSAEMixtureGaussian
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time

def run_training(model, buffer, params):
    """Run training with the specified parameters"""
    # Extract parameters
    MODEL_NAME = params["MODEL_NAME"]
    LAYER = params["LAYER"]
    TOTAL_STEPS = params["TOTAL_STEPS"]
    LR = params["LR"]
    KL_COEFF = params["KL_COEFF"]
    WARMUP_FRAC = params["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = params["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = params["DECAY_START_FRAC"]
    LOG_STEPS = params["LOG_STEPS"]
    CHECKPOINT_FRACS = params["CHECKPOINT_FRACS"]
    WANDB_ENTITY = params["WANDB_ENTITY"]
    WANDB_PROJECT = params["WANDB_PROJECT"]
    USE_WANDB = params["USE_WANDB"]
    DICT_SIZE = params["DICT_SIZE"]
    N_CORRELATED_PAIRS = params["N_CORRELATED_PAIRS"]
    N_ANTICORRELATED_PAIRS = params["N_ANTICORRELATED_PAIRS"]
    VAR_FLAG = params["VAR_FLAG"]
    
    d_mlp = model.cfg.d_mlp
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH DICT_SIZE = {DICT_SIZE} (Multiple: {DICT_SIZE/d_mlp}x)")
    print(f"Learning rate: {LR}, KL coefficient: {KL_COEFF}")
    print(f"Correlated pairs: {N_CORRELATED_PAIRS}, Anticorrelated pairs: {N_ANTICORRELATED_PAIRS}")
    print(f"Variance flag: {VAR_FLAG}")
    print(f"{'='*50}\n")
    
    # Configure trainer
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
        "wandb_name": f"VSAEMix_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_corr{N_CORRELATED_PAIRS}_anticorr{N_ANTICORRELATED_PAIRS}",
        "dict_class": VSAEMixtureGaussian
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set save directory
    save_dir = f"./trained_vsae_mix_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_corr{N_CORRELATED_PAIRS}_anticorr{N_ANTICORRELATED_PAIRS}"
    
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
            "dict_size": DICT_SIZE,
            "learning_rate": LR,
            "kl_coefficient": KL_COEFF,
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
        max_len=params["CTX_LEN"],
        device="cuda",
        n_batches=10
    )
    
    # Print metrics
    print("\n===== EVALUATION RESULTS =====")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Clear CUDA cache to prevent memory issues
    t.cuda.empty_cache()
    
    return eval_results

def main():
    # Define parameters for this specific run
    params = {
        # Model parameters
        "MODEL_NAME": "gelu-1l",
        "LAYER": 0,
        "HOOK_NAME": "blocks.0.mlp.hook_post",
        
        # Dictionary size parameters
        "DICT_SIZE": 8192,  # 4x default d_mlp (2048)
        
        # Training parameters        
        "TOTAL_STEPS": 30000,  
        "LR": 5e-4,  # 0.0005
        "KL_COEFF": 1e2,  # 100
        
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
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": False,
        
        # VSAE Mixture specific parameters
        "N_CORRELATED_PAIRS": 0,  # No correlated pairs
        "N_ANTICORRELATED_PAIRS": 0,  # No anticorrelated pairs
        "VAR_FLAG": 0,  # Fixed variance
    }
    
    # Load model
    print(f"Loading model: {params['MODEL_NAME']}")
    model = HookedTransformer.from_pretrained(
        params["MODEL_NAME"], 
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
        hook_name=params["HOOK_NAME"],
        d_submodule=model.cfg.d_mlp,
        n_ctxs=params["N_CTXS"],
        ctx_len=params["CTX_LEN"],
        refresh_batch_size=params["REFRESH_BATCH_SIZE"],
        out_batch_size=params["OUT_BATCH_SIZE"],
        device="cuda",
    )
    
    # Run training
    print("Starting training...")
    start_time = time.time()
    results = run_training(model, buffer, params)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Print final results summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {params['MODEL_NAME']}")
    print(f"Dictionary size: {params['DICT_SIZE']} (Multiple: {params['DICT_SIZE']/model.cfg.d_mlp}x)")
    print(f"Learning rate: {params['LR']}, KL coefficient: {params['KL_COEFF']}")
    
    # Print key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive"]
    for metric in metrics:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")

if __name__ == "__main__":
    # Set the start method to spawn for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()