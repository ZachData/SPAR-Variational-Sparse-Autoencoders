import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_jump_relu import VSAEJumpReLUTrainer, VSAEJumpReLU
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time
import argparse

def run_training(args, dict_size_multiple, model, buffer, base_params):
    """Run a single training with the specified dictionary size multiple"""
    # Extract base parameters
    MODEL_NAME = base_params["MODEL_NAME"]
    LAYER = base_params["LAYER"]
    TOTAL_STEPS = base_params["TOTAL_STEPS"]
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
        "trainer": VSAEJumpReLUTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": args.lr,
        "kl_coeff": args.kl_coeff,
        "l0_coeff": args.l0_coeff,
        "target_l0": args.target_l0,
        "warmup_steps": int(args.warmup_frac * TOTAL_STEPS),
        "sparsity_warmup_steps": int(args.sparsity_warmup_frac * TOTAL_STEPS),
        "decay_start": int(args.decay_start_frac * TOTAL_STEPS),
        "var_flag": args.var_flag,
        "bandwidth": args.bandwidth,
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEJumpReLU_{MODEL_NAME}_{DICT_SIZE}_lr{args.lr}_kl{args.kl_coeff}_l0c{args.l0_coeff}_tl0{args.target_l0}",
        "dict_class": VSAEJumpReLU
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))
    
    # Set unique save directory for this run
    save_dir = f"./VSAEJumpReLU_{MODEL_NAME}_d{DICT_SIZE}_lr{args.lr}_kl{args.kl_coeff}_l0c{args.l0_coeff}_tl0{args.target_l0}"
    
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
            "experiment_type": "vsae_jump_relu",
            "kl_coefficient": args.kl_coeff,
            "l0_coefficient": args.l0_coeff,
            "target_l0": args.target_l0,
            "bandwidth": args.bandwidth,
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
    
    return eval_results, save_dir

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train VSAEJumpReLU model')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="gelu-1l", help='Model name')
    parser.add_argument('--layer', type=int, default=0, help='Layer to extract activations from')
    parser.add_argument('--dict_size_multiple', type=float, default=4.0, help='Dictionary size as multiple of d_mlp')
    
    # Training parameters
    parser.add_argument('--steps', type=int, default=30000, help='Total training steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--kl_coeff', type=float, default=5.0, help='KL divergence coefficient')
    parser.add_argument('--l0_coeff', type=float, default=1.0, help='L0 target regularization coefficient')
    parser.add_argument('--target_l0', type=float, default=20.0, help='Target L0 (avg number of active features)')
    parser.add_argument('--bandwidth', type=float, default=0.001, help='Bandwidth for JumpReLU')
    parser.add_argument('--var_flag', type=int, default=1, choices=[0, 1], help='Whether to use learned variance (1) or fixed (0)')
    
    # Step fractions
    parser.add_argument('--warmup_frac', type=float, default=0.05, help='Fraction of steps for LR warmup')
    parser.add_argument('--sparsity_warmup_frac', type=float, default=0.05, help='Fraction of steps for sparsity warmup')
    parser.add_argument('--decay_start_frac', type=float, default=0.8, help='Fraction of steps to start LR decay')
    
    # Buffer parameters
    parser.add_argument('--n_ctxs', type=int, default=3000, help='Number of contexts for buffer')
    parser.add_argument('--ctx_len', type=int, default=128, help='Context length')
    parser.add_argument('--refresh_batch', type=int, default=32, help='Refresh batch size')
    parser.add_argument('--out_batch', type=int, default=1024, help='Output batch size')
    
    # WandB parameters
    parser.add_argument('--wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb_entity', type=str, default='', help='WandB entity')
    parser.add_argument('--wandb_project', type=str, default='vsae-experiments', help='WandB project')
    
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
        
        # Buffer parameters
        "N_CTXS": args.n_ctxs,
        "CTX_LEN": args.ctx_len,
        "REFRESH_BATCH_SIZE": args.refresh_batch,
        "OUT_BATCH_SIZE": args.out_batch,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],
        "LOG_STEPS": 100,
        
        # WandB parameters
        "WANDB_ENTITY": args.wandb_entity if args.wandb_entity else os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": args.wandb_project if args.wandb_project else os.environ.get("WANDB_PROJECT", "vsae-experiments"),
        "USE_WANDB": args.wandb,
    }
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    # Load the model
    print(f"Loading model: {args.model}")
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
    print(f"\nStarting training with dictionary size multiple: {args.dict_size_multiple}")
    start_time = time.time()
    
    # Run training
    results, save_dir = run_training(args, args.dict_size_multiple, model, buffer, base_params)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Completed training in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # ========== PRINT RESULTS ==========
    print("\n\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Print key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive", "frac_recovered"]
    dict_size = int(args.dict_size_multiple * model.cfg.d_mlp)
    
    print(f"{'Dict Size':15} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (15 + 25 * len(metrics)))
    print(f"{dict_size:<15} | " + " | ".join(f"{results.get(metric, float('nan')):<22.4f}" for metric in metrics))
    
    print(f"\nModel saved to: {save_dir}")

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()
