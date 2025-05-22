import torch as t
import wandb
import os
import time
import uuid
import argparse
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_topk import VSAETopKTrainer, VSAETopK
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE, get_norm_factor
from dictionary_learning.evaluation import evaluate
import multiprocessing

def train_vsae_topk():
    """
    Training function for the VSAETopK model.
    This function will be called by the wandb.agent for each hyperparameter configuration.
    """
    # Initialize wandb run
    run_id = str(uuid.uuid4())[:8]
    
    # Set up WandB
    run = wandb.init(
        project="vsae_topk_sweep",
        group=f"dict_multiple_{wandb.config.dict_size_multiple}",
        job_type="sweep",
        name=f"sweep_{run_id}"
    )
    
    # Extract hyperparameters from wandb.config
    dict_size_multiple = wandb.config.dict_size_multiple
    k_ratio = wandb.config.k_ratio
    lr = wandb.config.lr
    kl_coeff = wandb.config.kl_coeff
    auxk_alpha = wandb.config.auxk_alpha
    var_flag = wandb.config.var_flag
    use_april_update = wandb.config.use_april_update
    n_steps = wandb.config.n_steps
    
    # Fixed parameters
    MODEL_NAME = "gelu-1l"
    LAYER = 0
    HOOK_NAME = f"blocks.{LAYER}.mlp.hook_post"
    WARMUP_FRAC = 0.05
    SPARSITY_WARMUP_FRAC = 0.05
    DECAY_START_FRAC = 0.8
    
    # Buffer parameters
    N_CTXS = 3000
    CTX_LEN = 128
    REFRESH_BATCH_SIZE = 32
    OUT_BATCH_SIZE = 1024
    
    # Set up checkpoint fractions and logging
    CHECKPOINT_FRACS = [0.5, 1.0]
    LOG_STEPS = 100
    SAVE_STEPS = [int(frac * n_steps) for frac in CHECKPOINT_FRACS]
    
    # Log all parameters to WandB
    wandb.config.update({
        "model_name": MODEL_NAME,
        "layer": LAYER,
        "warmup_frac": WARMUP_FRAC,
        "sparsity_warmup_frac": SPARSITY_WARMUP_FRAC,
        "decay_start_frac": DECAY_START_FRAC,
        "n_ctxs": N_CTXS,
        "ctx_len": CTX_LEN,
        "refresh_batch_size": REFRESH_BATCH_SIZE,
        "out_batch_size": OUT_BATCH_SIZE
    })
    
    # Load model
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, 
        device=device
    )
    
    # Calculate derived parameters
    d_mlp = model.cfg.d_mlp
    dict_size = int(dict_size_multiple * d_mlp)
    k = max(1, int(dict_size * k_ratio))
    
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH CONFIG:")
    print(f"Dictionary size multiple: {dict_size_multiple} (size: {dict_size})")
    print(f"Top-k ratio: {k_ratio} (k: {k})")
    print(f"Learning rate: {lr}")
    print(f"KL coefficient: {kl_coeff}")
    print(f"Auxiliary loss coefficient: {auxk_alpha}")
    print(f"Using {'learned' if var_flag else 'fixed'} variance")
    print(f"Using {'April update mode' if use_april_update else 'standard mode'}")
    print(f"{'='*50}\n")
    
    # Set up data generator
    data_gen = hf_dataset_to_generator(
        "NeelNanda/c4-code-tokenized-2b", 
        split="train", 
        return_tokens=True
    )
    
    # Create activation buffer
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=HOOK_NAME,
        d_submodule=d_mlp,
        n_ctxs=N_CTXS,
        ctx_len=CTX_LEN,
        refresh_batch_size=REFRESH_BATCH_SIZE,
        out_batch_size=OUT_BATCH_SIZE,
        device=device,
    )
    
    # Configure trainer
    trainer_config = {
        "trainer": VSAETopKTrainer,
        "steps": n_steps,
        "activation_dim": d_mlp,
        "dict_size": dict_size,
        "k": k,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": lr,
        "kl_coeff": kl_coeff,
        "auxk_alpha": auxk_alpha,
        "warmup_steps": int(WARMUP_FRAC * n_steps),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * n_steps),
        "decay_start": int(DECAY_START_FRAC * n_steps),
        "var_flag": var_flag,
        "use_april_update_mode": use_april_update,
        "device": device,
        "seed": 42,
        "dict_class": VSAETopK,
        "wandb_name": f"VSAETopK_{MODEL_NAME}_d{dict_size}_k{k}_lr{lr}_kl{kl_coeff}"
    }
    
    # Set unique save directory
    save_dir = (f"./VSAETopK_{MODEL_NAME}_d{dict_size}_k{k}_lr{lr}_"
                f"kl{kl_coeff}_var{var_flag}_april{int(use_april_update)}_run{run_id}")
    
    # Run training
    print(f"Starting training with run ID: {run_id}")
    start_time = time.time()
    
    try:
        # Find the norm factor for activation normalization
        norm_factor = get_norm_factor(buffer, steps=100)
        print(f"Norm factor: {norm_factor}")
        
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=n_steps,
            save_dir=save_dir,
            save_steps=SAVE_STEPS,
            log_steps=LOG_STEPS,
            verbose=True,
            normalize_activations=True,
            autocast_dtype=t.bfloat16,
            use_wandb=True,  # Always use wandb for the sweep
            wandb_entity=os.environ.get("WANDB_ENTITY", None),
            wandb_project="vsae_topk_sweep",
            run_cfg={
                "model_type": MODEL_NAME,
                "experiment_type": "vsae_topk",
                "dict_size_multiple": dict_size_multiple,
                "k_ratio": k_ratio,
                "var_flag": var_flag,
                "april_update": use_april_update,
                "run_id": run_id
            }
        )
        
        # Evaluate the trained model
        print("Evaluating trained model...")
        from dictionary_learning.utils import load_dictionary
        
        vsae_topk, config = load_dictionary(f"{save_dir}/trainer_0", device=device)
        
        eval_results = evaluate(
            dictionary=vsae_topk,
            activations=buffer,
            batch_size=64,
            max_len=CTX_LEN,
            device=device,
            n_batches=10
        )
        
        # Log evaluation metrics directly to WandB
        for metric, value in eval_results.items():
            wandb.log({f"eval/{metric}": value})
            print(f"{metric}: {value:.4f}")
        
        # Calculate validation loss for the sweep
        # Use frac_variance_explained as a positive metric (higher is better)
        # We invert it for minimization goal
        validation_loss = 1.0 - eval_results.get("frac_variance_explained", 0.0)
        wandb.log({"validation_loss": validation_loss})
        
        # Log other key metrics for the sweep
        wandb.log({
            "l0": eval_results.get("l0", 0.0),
            "frac_alive": eval_results.get("frac_alive", 0.0),
            "frac_recovered": eval_results.get("frac_recovered", 0.0),
            "training_time": time.time() - start_time
        })
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Log failure to WandB
        wandb.log({"training_failed": True, "error": str(e)})
        raise
    finally:
        # Clean up
        t.cuda.empty_cache()
        wandb.finish()

def sweep_vsae_topk(dict_size_multiple, max_sweep_runs=10):
    """
    Run a hyperparameter sweep for VSAETopK with a specific dictionary size multiple.
    
    Args:
        dict_size_multiple: Multiple of model dimension for dictionary size
        max_sweep_runs: Maximum number of runs in the sweep
    """
    # Define the sweep configuration
    sweep_config = {
        "method": "bayes",  # Use Bayesian optimization
        "metric": {
            "name": "validation_loss",  # Minimize 1 - frac_variance_explained
            "goal": "minimize"
        },
        "parameters": {
            # Fixed parameter for this sweep
            "dict_size_multiple": {"value": dict_size_multiple},
            
            # Parameters to sweep over
            "k_ratio": {
                "values": [0.05, 0.1, 0.15, 0.2]  # Different sparsity levels
            },
            "lr": {
                "distribution": "log_uniform",
                "min": -11.5,  # ~1e-5
                "max": -9.2    # ~1e-4
            },
            "kl_coeff": {
                "distribution": "log_uniform",
                "min": 0.0,    # 1
                "max": 4.6     # ~100
            },
            "auxk_alpha": {
                "values": [0.0, 0.01, 0.03125, 0.05]  # Different auxiliary loss weights
            },
            "var_flag": {
                "values": [0, 1]  # Fixed or learned variance
            },
            "use_april_update": {
                "values": [True, False]  # Whether to use April update mode
            },
            "n_steps": {
                "value": 10000  # Fixed number of steps for each sweep run
            }
        },
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 10000,
            "s": 2,
            "eta": 3
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="vsae_topk_sweep"
    )
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_vsae_topk, count=max_sweep_runs)
    
    return sweep_id

def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps for VSAETopK models")
    parser.add_argument("--dict-multiples", type=float, nargs="+", default=[4.0, 8.0, 16.0],
                        help="Dictionary size multiples to sweep over")
    parser.add_argument("--max-runs-per-sweep", type=int, default=10,
                        help="Maximum number of runs per dictionary size multiple")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up multiprocessing
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('spawn', force=True)
    
    # Run sweeps for each dictionary size multiple
    for dict_size_multiple in args.dict_multiples:
        print(f"\n\n{'='*80}")
        print(f"STARTING HYPERPARAMETER SWEEP FOR DICTIONARY SIZE MULTIPLE: {dict_size_multiple}")
        print(f"{'='*80}\n")
        
        sweep_id = sweep_vsae_topk(
            dict_size_multiple=dict_size_multiple,
            max_sweep_runs=args.max_runs_per_sweep
        )
        
        print(f"\nSweep completed for dictionary size multiple {dict_size_multiple}")
        print(f"Sweep ID: {sweep_id}")

if __name__ == "__main__":
    main()
