import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.matryoshka_vsae_iso import MatryoshkaVSAEIsoTrainer, MatryoshkaVSAEIso
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
    AUXK_ALPHA = base_params["AUXK_ALPHA"]
    WARMUP_FRAC = base_params["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = base_params["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = base_params["DECAY_START_FRAC"]
    LOG_STEPS = base_params["LOG_STEPS"]
    CHECKPOINT_FRACS = base_params["CHECKPOINT_FRACS"]
    WANDB_ENTITY = base_params["WANDB_ENTITY"]
    WANDB_PROJECT = base_params["WANDB_PROJECT"]
    USE_WANDB = base_params["USE_WANDB"]
    VAR_FLAG = base_params["VAR_FLAG"]
    GROUP_FRACTIONS = base_params["GROUP_FRACTIONS"]
    GROUP_WEIGHTS = base_params["GROUP_WEIGHTS"]
    
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
        "trainer": MatryoshkaVSAEIsoTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "kl_coeff": KL_COEFF,
        "auxk_alpha": AUXK_ALPHA,
        "warmup_steps": int(WARMUP_FRAC * TOTAL_STEPS),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * TOTAL_STEPS),
        "decay_start": int(DECAY_START_FRAC * TOTAL_STEPS),
        "var_flag": VAR_FLAG,
        "group_fractions": GROUP_FRACTIONS,
        "group_weights": GROUP_WEIGHTS,
        "device": "cuda",
        "wandb_name": f"MatryoshkaVSAE_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_warm{WARMUP_FRAC}",
        "dict_class": MatryoshkaVSAEIso
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items() if k != "trainer" and k != "dict_class"))
    
    # Set unique save directory for this run
    save_dir = f"./MatryoshkaVSAEIso_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_auxk{AUXK_ALPHA}"
    
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
            "experiment_type": "matryoshka_vsae_iso",
            "kl_coefficient": KL_COEFF,
            "auxk_alpha": AUXK_ALPHA,
            "dict_size_multiple": dict_size_multiple,
            "var_flag": VAR_FLAG,
            "group_fractions": GROUP_FRACTIONS,
        }
    )
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.evaluation import evaluate
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Matryoshka VSAE model")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gelu-1l", help="Model to use")
    parser.add_argument("--layer", type=int, default=0, help="Layer to extract activations from")
    parser.add_argument("--hook_name", type=str, default="blocks.0.mlp.hook_post", help="Hook name for activation extraction")
    
    # Training parameters
    parser.add_argument("--total_steps", type=int, default=30000, help="Total number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--kl_coeff", type=float, default=5.0, help="KL divergence coefficient")
    parser.add_argument("--auxk_alpha", type=float, default=1/32, help="Auxiliary loss coefficient")
    parser.add_argument("--dict_size_multiple", type=float, default=4.0, help="Dictionary size as multiple of d_mlp")
    
    # Step fractions
    parser.add_argument("--warmup_frac", type=float, default=0.05, help="Fraction of steps for warmup")
    parser.add_argument("--sparsity_warmup_frac", type=float, default=0.05, help="Fraction of steps for sparsity warmup")
    parser.add_argument("--decay_start_frac", type=float, default=0.8, help="Fraction of steps for decay start")
    
    # Buffer parameters
    parser.add_argument("--n_ctxs", type=int, default=3000, help="Number of contexts for buffer")
    parser.add_argument("--ctx_len", type=int, default=128, help="Context length")
    parser.add_argument("--refresh_batch_size", type=int, default=32, help="Refresh batch size")
    parser.add_argument("--out_batch_size", type=int, default=1024, help="Output batch size")
    
    # Saving parameters
    parser.add_argument("--log_steps", type=int, default=100, help="Steps between logging")
    
    # Variational and Matryoshka parameters
    parser.add_argument("--var_flag", type=int, default=0, choices=[0, 1], help="Use learned variance (0: fixed, 1: learned)")
    parser.add_argument("--num_groups", type=int, default=3, help="Number of Matryoshka groups")
    
    # WandB parameters
    parser.add_argument("--wandb_entity", type=str, default="", help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="vsae-experiments", help="WandB project")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Generate group fractions based on num_groups
    # This creates a geometric sequence of group sizes (smaller groups first)
    group_fractions = []
    remaining = 1.0
    for i in range(args.num_groups - 1):
        # Exponentially increasing fractions
        fraction = remaining * (2 ** (i - args.num_groups + 1))
        group_fractions.append(fraction)
        remaining -= fraction
    group_fractions.append(remaining)  # Last group gets the remainder
    
    # Make weights favor later (larger) groups
    group_weights = [2 ** i for i in range(args.num_groups)]
    total_weight = sum(group_weights)
    group_weights = [w / total_weight for w in group_weights]
    
    # ========== HYPERPARAMETERS ==========
    # Define base parameters that will be used for all runs
    base_params = {
        # Model parameters
        "MODEL_NAME": args.model_name,
        "LAYER": args.layer,
        "HOOK_NAME": args.hook_name,
        
        # Training parameters        
        "TOTAL_STEPS": args.total_steps,
        "LR": args.lr,
        "KL_COEFF": args.kl_coeff,
        "AUXK_ALPHA": args.auxk_alpha,
        
        # Step fractions
        "WARMUP_FRAC": args.warmup_frac,
        "SPARSITY_WARMUP_FRAC": args.sparsity_warmup_frac,
        "DECAY_START_FRAC": args.decay_start_frac,
        
        # Buffer parameters
        "N_CTXS": args.n_ctxs,
        "CTX_LEN": args.ctx_len,
        "REFRESH_BATCH_SIZE": args.refresh_batch_size,
        "OUT_BATCH_SIZE": args.out_batch_size,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],  # Save at 50% and 100% of training
        "LOG_STEPS": args.log_steps,
        
        # Variational and Matryoshka parameters
        "VAR_FLAG": args.var_flag,
        "GROUP_FRACTIONS": group_fractions,
        "GROUP_WEIGHTS": group_weights,
        
        # WandB parameters
        "WANDB_ENTITY": args.wandb_entity or os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": args.wandb_project or os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": args.use_wandb,
    }
    
    # Print important hyperparameters
    print("\n===== CONFIGURATION =====")
    print(f"Model: {args.model_name}")
    print(f"Dictionary size multiple: {args.dict_size_multiple}x")
    print(f"Learning rate: {args.lr}")
    print(f"KL coefficient: {args.kl_coeff}")
    print(f"Auxiliary loss coefficient: {args.auxk_alpha}")
    print(f"Variational flag: {args.var_flag}")
    print(f"Group fractions: {group_fractions}")
    print(f"Group weights: {group_weights}")
    print(f"Total steps: {args.total_steps}")
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    print("\nLoading model...")
    model = HookedTransformer.from_pretrained(
        base_params["MODEL_NAME"], 
        device="cuda"
    )
    
    print("Setting up data generator...")
    # Set up data generator
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b", 
        split="train", 
        return_tokens=True
    )
    
    print("Creating activation buffer...")
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
    print(f"\nStarting training with dictionary size multiple: {args.dict_size_multiple}")
    start_time = time.time()
    
    # Run training with this dictionary size
    results = run_training(args.dict_size_multiple, model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    print(f"Completed training in {elapsed_time:.2f} seconds")
    
    # ========== PRINT RESULTS ==========
    print("\n\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Print key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive", "l2_loss", "frac_recovered"]
    dict_size = int(args.dict_size_multiple * model.cfg.d_mlp)
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    
    metric_row = f"{dict_size:<15} | "
    for metric in metrics:
        if metric in results:
            metric_row += f"{results[metric]:<22.4f}"
        else:
            metric_row += f"{'N/A':<22}"
    print(metric_row)

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()
