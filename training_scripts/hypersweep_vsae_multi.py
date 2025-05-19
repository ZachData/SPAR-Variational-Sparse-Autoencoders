import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_multi import VSAEMultiGaussianTrainer, VSAEMultiGaussian
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
import multiprocessing
import os
import time
import itertools
import argparse
from datetime import datetime


def run_training(config, model, buffer, device="cuda"):
    """Run a single training with the specified hyperparameters"""
    # Extract base parameters
    MODEL_NAME = config["MODEL_NAME"]
    LAYER = config["LAYER"]
    TOTAL_STEPS = config["TOTAL_STEPS"]
    LR = config["LR"]
    KL_COEFF = config["KL_COEFF"]
    WARMUP_FRAC = config["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = config["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = config["DECAY_START_FRAC"]
    LOG_STEPS = config["LOG_STEPS"]
    CHECKPOINT_FRACS = config["CHECKPOINT_FRACS"]
    WANDB_ENTITY = config["WANDB_ENTITY"]
    WANDB_PROJECT = config["WANDB_PROJECT"]
    USE_WANDB = config["USE_WANDB"]
    DICT_SIZE_MULTIPLE = config.get("DICT_SIZE_MULTIPLE", 8)  # Set to 8x for hyperparameter sweep
    
    # VSAEMulti specific parameters
    corr_rate = config.get("CORR_RATE", 0.0)
    var_flag = config.get("VAR_FLAG", 0)
    
    # Calculate derived parameters
    d_mlp = model.cfg.d_mlp
    DICT_SIZE = int(DICT_SIZE_MULTIPLE * d_mlp)
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    
    # Create a unique name for this run based on hyperparameters
    run_name = f"VSAEMulti_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_kl{KL_COEFF}_corr{corr_rate}"
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH CONFIG:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Dictionary size: {DICT_SIZE}")
    print(f"{'='*50}\n")
    
    # Configure trainer for this run
    trainer_config = {
        "trainer": VSAEMultiGaussianTrainer,
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
        "var_flag": var_flag,
        "corr_rate": corr_rate,
        "use_april_update_mode": True,
        "device": device,
        "wandb_name": run_name,
        "dict_class": VSAEMultiGaussian
    }
    
    # Print configuration
    print("===== TRAINER CONFIGURATION =====")
    print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items() if not isinstance(v, t.Tensor)))
    
    # Create timestamp for unique subdirectory within the main folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set main save directory
    main_save_dir = "./trained_sweeps_multi_4x"
    
    # Create main directory if it doesn't exist
    os.makedirs(main_save_dir, exist_ok=True)
    
    # Create unique subdirectory for this run
    run_dir = f"{run_name}_{timestamp}"
    save_dir = os.path.join(main_save_dir, run_dir)
    
    # Add experiment metadata to run config
    run_cfg = {
        "model_type": MODEL_NAME,
        "experiment_type": "multivsae_sweep",
        "kl_coefficient": KL_COEFF,
        "dict_size_multiple": DICT_SIZE_MULTIPLE,
        "learning_rate": LR,
        "warmup_fraction": WARMUP_FRAC,
        "sparsity_warmup_fraction": SPARSITY_WARMUP_FRAC,
        "decay_start_fraction": DECAY_START_FRAC,
        "var_flag": var_flag,
        "corr_rate": corr_rate,
    }
    
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
        run_cfg=run_cfg
    )
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    from dictionary_learning.evaluation import evaluate
    
    # Load the trained dictionary
    vsae, loaded_config = load_dictionary(f"{save_dir}/trainer_0", device=device)
    
    # Evaluate on a small batch
    eval_results = evaluate(
        dictionary=vsae,
        activations=buffer,
        batch_size=64,
        max_len=config["CTX_LEN"],
        device=device,
        n_batches=10
    )
    
    # Print metrics
    print("\n===== EVALUATION RESULTS =====")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Clear CUDA cache to prevent memory issues between runs
    t.cuda.empty_cache()
    
    return {
        "save_dir": save_dir,
        "config": trainer_config,
        "eval_results": eval_results,
        "run_name": run_name
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train VSAEMultiGaussian models with hyperparameter sweep')
    parser.add_argument('--model_name', type=str, default='gelu-1l', help='Model name')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--steps', type=int, default=10000, help='Total training steps')
    parser.add_argument('--dataset', type=str, default='NeelNanda/c4-code-tokenized-2b', help='Dataset for training')
    parser.add_argument('--n_ctxs', type=int, default=3000, help='Number of contexts for buffer')
    parser.add_argument('--ctx_len', type=int, default=128, help='Context length')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity')
    parser.add_argument('--wandb_project', type=str, default='vsae-sweep', help='W&B project')
    parser.add_argument('--sweep_all', action='store_true', help='Sweep all hyperparameters (more combinations)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dict_size_multiple', type=int, default=8, help='Dictionary size multiple of model width')
    
    args = parser.parse_args()
    
    # ========== HYPERPARAMETERS ==========
    # Define base parameters that will be used for all runs
    base_params = {
        # Model parameters
        "MODEL_NAME": args.model_name,
        "LAYER": args.layer,
        "HOOK_NAME": f"blocks.{args.layer}.mlp.hook_post",
        
        # Set to 8x size multiple for hyperparameter sweep
        "DICT_SIZE_MULTIPLE": args.dict_size_multiple,
        
        # Training parameters        
        "TOTAL_STEPS": args.steps,
        
        # Buffer parameters
        "N_CTXS": args.n_ctxs,
        "CTX_LEN": args.ctx_len,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        
        # Saving parameters
        "CHECKPOINT_FRACS": [0.5, 1.0],
        "LOG_STEPS": 100,
        
        # WandB parameters
        "WANDB_ENTITY": args.wandb_entity or os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": args.wandb_project or os.environ.get("WANDB_PROJECT", "vsae-sweep"),
        "USE_WANDB": not args.no_wandb,
    }
    
    # Define hyperparameter sweep grid
    if args.sweep_all:
        # Comprehensive sweep with more values for multi
        sweep_params = {
            "LR": [5e-3, 1e-3, 5e-4, 1e-4],
            "KL_COEFF": [50, 100, 200],
            "WARMUP_FRAC": [0.05],
            "SPARSITY_WARMUP_FRAC": [0.05, 0.1],
            "DECAY_START_FRAC": [0.8],
            "VAR_FLAG": [0, 1],  # 0: fixed variance, 1: learned variance
            "CORR_RATE": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }
    else:
        # Limited sweep with fewer values for multi (faster)
        sweep_params = {
            "LR": [1e-3, 5e-4],
            "KL_COEFF": [50, 100, 200],
            "WARMUP_FRAC": [0.05],
            "SPARSITY_WARMUP_FRAC": [0.05],
            "DECAY_START_FRAC": [0.8],
            "VAR_FLAG": [0],  # Only fixed variance
            "CORR_RATE": [0.0, 0.2, 0.4]
        }
    
    # Create main save directory
    main_save_dir = "./trained_sweeps_multi_4x"
    os.makedirs(main_save_dir, exist_ok=True)
    print(f"Created main save directory: {main_save_dir}")
    
    # ========== LOAD MODEL & CREATE BUFFER ==========
    # Only load the model once for all runs
    model = HookedTransformer.from_pretrained(
        base_params["MODEL_NAME"], 
        device=args.device
    )
    
    # Set up data generator
    data_gen = hf_dataset_to_generator(
        args.dataset, 
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
        device=args.device,
    )
    
    # Generate all hyperparameter combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Starting hyperparameter sweep with {len(param_combinations)} combinations")
    print(f"Parameters being swept: {param_names}")
    
    # ========== RUN HYPERPARAMETER SWEEP ==========
    all_results = []
    
    for i, param_combo in enumerate(param_combinations):
        print(f"\n[Run {i+1}/{len(param_combinations)}] Starting training with hyperparameters:")
        config = base_params.copy()
        for name, value in zip(param_names, param_combo):
            config[name] = value
            print(f"  {name}: {value}")
        
        start_time = time.time()
        
        # Run training with this configuration
        try:
            results = run_training(config, model, buffer, device=args.device)
            all_results.append(results)
            
            elapsed_time = time.time() - start_time
            print(f"Completed training run {i+1} in {elapsed_time:.2f} seconds")
            
            # Short pause between runs
            print("Cooling down before next run...")
            time.sleep(10)
        except Exception as e:
            print(f"Error in run {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== PRINT COMPARATIVE RESULTS ==========
    print("\n\n" + "="*50)
    print("COMPARATIVE RESULTS ACROSS ALL RUNS FOR VSAEMULTI")
    print("="*50)
    
    # Print comparison table of key metrics
    metrics = ["frac_variance_explained", "l0", "frac_alive"]
    
    print(f"{'Run Name':50} | " + " | ".join(f"{metric:22}" for metric in metrics))
    print("-" * (50 + 25 * len(metrics)))
    
    for result in all_results:
        run_name = result['run_name'] 
        print(f"{run_name[:48]:50} | " + " | ".join(f"{result['eval_results'][metric]:<22.4f}" for metric in metrics))
    
    # Save all results to a file for later analysis
    import json
    serializable_results = []
    for result in all_results:
        serializable_result = {
            "run_name": result["run_name"],
            "save_dir": result["save_dir"],
            "config": {k: (v if not callable(v) else str(v)) for k, v in result["config"].items()},
            "eval_results": result["eval_results"]
        }
        serializable_results.append(serializable_result)
    
    # Save results in the main directory
    with open(os.path.join(main_save_dir, "sweep_results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nSweep completed. Results saved to {os.path.join(main_save_dir, 'sweep_results.json')}")

if __name__ == "__main__":
    # Set the start method to spawn for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()