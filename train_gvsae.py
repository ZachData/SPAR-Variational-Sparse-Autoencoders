import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_gated import VSAEGatedTrainer, VSAEGated
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time
import argparse

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
    WARMUP_FRAC = base_params["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = base_params["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = base_params["DECAY_START_FRAC"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_FRACS = base_params["CHECKPOINT_FRACS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    USE_CONSTRAINED_OPTIMIZER = base_params["USE_CONSTRAINED_OPTIMIZER"]
    VAR_FLAG = base_params["VAR_FLAG"]
    
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
        "warmup_steps": int(WARMUP_FRAC * TOTAL_STEPS),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * TOTAL_STEPS),
        "decay_start": int(DECAY_START_FRAC * TOTAL_STEPS),
        "var_flag": VAR_FLAG,
        "use_april_update_mode": True,
        "use_constrained_optimizer": USE_CONSTRAINED_OPTIMIZER,
        "device": "cuda",
        "wandb_name": f"VSAEGated_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_l1{L1_PENALTY}_aux{AUX_WEIGHT}",
        "dict_class": VSAEGated
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items() if k != "trainer" and k != "dict_class"))
    
    # Set unique save directory for this run
    save_dir = f"./VSAEGated_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_l1{L1_PENALTY}_aux{AUX_WEIGHT}"
    
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
            "experiment_type": "vsae_gated",
            "kl_coefficient": KL_COEFF,
            "l1_penalty": L1_PENALTY,
            "aux_weight": AUX_WEIGHT,
            "dict_size_multiple": dict_size_multiple,
            "var_flag": VAR_FLAG,
            "use_constrained_optimizer": USE_CONSTRAINED_OPTIMIZER
        }
    )
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.evaluation import evaluate
    
    # Look for the final model or the last checkpoint
    model_path = os.path.join(save_dir, "trainer_0", "ae.pt")
    if not os.path.exists(model_path):
        # Try to find latest checkpoint
        checkpoint_dir = os.path.join(save_dir, "trainer_0", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("ae_")]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                model_path = os.path.join(checkpoint_dir, latest)
                print(f"Using checkpoint: {model_path}")
            else:
                print("No checkpoints found!")
                return None
        else:
            print("No model found!")
            return None
    
    # Load the trained dictionary
    vsae = VSAEGated.from_pretrained(model_path, device="cuda", var_flag=VAR_FLAG)
    
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a VSAEGated model')
    parser.add_argument('--model', type=str, default="gelu-1l", help='Model name')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--dict_size_multiple', type=float, default=4.0, help='Dictionary size multiplier')
    parser.add_argument('--steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--kl_coeff', type=float, default=5.0, help='KL divergence coefficient')
    parser.add_argument('--l1_penalty', type=float, default=0.01, help='L1 penalty for gating')
    parser.add_argument('--aux_weight', type=float, default=0.1, help='Auxiliary loss weight')
    parser.add_argument('--var_flag', type=int, default=1, choices=[0, 1], help='Variational mode (0: fixed, 1: learned)')
    parser.add_argument('--constrained', action='store_true', help='Use constrained optimizer')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    # ========== HYPERPARAMETERS ==========
    # Define base parameters that will be used for all runs
    base_params = {
        # Model parameters
        "MODEL_NAME": args.model,
        "LAYER": args.layer,
        "HOOK_NAME": f"blocks.{args.layer}.mlp.hook_post",
        
        # Training parameters        
        "TOTAL_STEPS": args.steps,
        "LR": args.lr, 
        "KL_COEFF": args.kl_coeff,
        "L1_PENALTY": args.l1_penalty,
        "AUX_WEIGHT": args.aux_weight,
        
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
        
        # Model configuration
        "VAR_FLAG": args.var_flag,
        "USE_CONSTRAINED_OPTIMIZER": args.constrained,
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", "zachdata"),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": not args.no_wandb,
    }
    
    # Dictionary size multiple provided by command line
    dict_size_multiple = args.dict_size_multiple
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    # Load the model
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
        device="cpu",  # Store on CPU to save GPU memory
    )
    
    # ========== RUN TRAINING ==========
    print(f"\nStarting training with dictionary size multiple: {dict_size_multiple}")
    start_time = time.time()
    
    # Run training with specified dictionary size
    results = run_training(dict_size_multiple, model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    print(f"Completed training in {elapsed_time:.2f} seconds")
    
    # ========== PRINT RESULTS ==========
    print("\n\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    if results:
        # Print key metrics
        metrics = ["frac_variance_explained", "l0", "frac_alive", "frac_recovered"]
        dict_size = int(dict_size_multiple * model.cfg.d_mlp)
        
        print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
        print("-" * (15 + 25 * len(metrics)))
        print(f"{dict_size:<15} | " + " | ".join(f"{results.get(metric, 0.0):<22.4f}" for metric in metrics))
    else:
        print("No results available")

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()
