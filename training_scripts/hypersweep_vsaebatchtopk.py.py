"""
Hyperparameter sweep for VSAETopK on gelu-1l model.

This script runs dedicated hyperparameter sweeps for VSAETopK models on the gelu-1l model,
focusing on finding optimal configurations with multiple dictionary size multipliers.
"""

import torch as t
import os
import wandb
import random
import uuid
import json
from datetime import datetime
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_topk import VSAETopKTrainer, VSAETopK
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
import multiprocessing

def create_sweep_config(dict_size, model_name, d_mlp, steps):
    """Create a sweep configuration for a specific dictionary size."""
    sweep_config = {
        "method": "bayes",  # Use Bayesian optimization
        "metric": {
            "name": "frac_variance_explained",  # Metric to optimize
            "goal": "maximize"                  # We want to maximize this metric
        },
        "parameters": {
            # Learning rate - typically small for VSAEs
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3
            },
            # KL divergence coefficient - controls sparsity
            "kl_coeff": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 10.0
            },
            # Auxiliary loss coefficient - for dead features
            "auxk_alpha": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 0.1
            },
            # K ratio - percentage of features to select
            "k_ratio": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.1
            },
            # Variance flag - fixed or learned
            "var_flag": {
                "values": [0, 1]
            },
            # Whether to constrain weights to unit norm
            "constrain_weights": {
                "values": [True, False]
            },
            # Fixed parameters
            "dict_size": {"value": dict_size},
            "model_name": {"value": model_name},
            "d_mlp": {"value": d_mlp},
            "steps": {"value": steps},
            # Fixed fractions with reasonable defaults
            "warmup_frac": {"value": 0.05},
            "sparsity_warmup_frac": {"value": 0.05},
            "decay_start_frac": {"value": 0.8},
            "threshold_beta": {"value": 0.999}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": int(steps * 0.2),  # At least 20% of training
            "max_iter": steps,
            "s": 2,
            "eta": 3
        }
    }
    
    return sweep_config

def train_vsae_topk(config=None):
    """Training function for a single sweep run."""
    # Initialize wandb
    run_id = str(uuid.uuid4())[:8]
    variation_name = f"dict_size_{config['dict_size']}"
    
    run = wandb.init(
        project="gelu_1l_vsae_topk",        # Project for this model type
        group=variation_name,               # Group by dictionary size
        job_type="sweep",                   # Mark as sweep job
        name=f"sweep_{run_id}",             # Unique name for run
        config=config
    )
    
    # Extract parameters from config
    model_name = config["model_name"]
    dict_size = config["dict_size"]
    d_mlp = config["d_mlp"]
    steps = config["steps"]
    lr = config["lr"]
    kl_coeff = config["kl_coeff"]
    auxk_alpha = config["auxk_alpha"]
    k_ratio = config["k_ratio"]
    var_flag = config["var_flag"]
    constrain_weights = config["constrain_weights"]
    warmup_frac = config["warmup_frac"]
    sparsity_warmup_frac = config["sparsity_warmup_frac"]
    decay_start_frac = config["decay_start_frac"]
    threshold_beta = config["threshold_beta"]
    
    # Compute derived parameters
    k = max(1, int(k_ratio * d_mlp))
    warmup_steps = int(warmup_frac * steps)
    sparsity_warmup_steps = int(sparsity_warmup_frac * steps)
    decay_start = int(decay_start_frac * steps)
    threshold_start_step = 1000  # Fixed value for now
    
    # Configure trainer
    trainer_config = {
        "trainer": VSAETopKTrainer,
        "steps": steps,
        "activation_dim": d_mlp,
        "dict_size": dict_size,
        "k": k,
        "layer": 0,  # Fixed for gelu-1l
        "lm_name": model_name,
        "lr": lr,
        "kl_coeff": kl_coeff,
        "auxk_alpha": auxk_alpha,
        "warmup_steps": warmup_steps,
        "sparsity_warmup_steps": sparsity_warmup_steps,
        "decay_start": decay_start,
        "threshold_beta": threshold_beta,
        "threshold_start_step": threshold_start_step,
        "var_flag": var_flag,
        "constrain_weights": constrain_weights,
        "device": "cuda",
        "wandb_name": f"VSAETopK_{model_name}_d{dict_size}_k{k}_var{var_flag}",
        "dict_class": VSAETopK
    }
    
    # Create temporary save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./sweep_results/gelu_1l/dict{dict_size}/run_{timestamp}_{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save run configuration
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump({**config, "k": k}, f, indent=4)
    
    # Use the existing buffer from our global scope
    global buffer
    
    # Track metrics for early stopping
    best_metric = 0
    patience_counter = 0
    patience_limit = 5  # Number of logging intervals without improvement
    
    # Custom callback for logging and early stopping
    def logging_callback(step, metrics, trainer):
        nonlocal best_metric, patience_counter
        
        # Log metrics to wandb
        wandb.log(metrics)
        
        # Check for improvement
        current_metric = metrics.get("frac_variance_explained", 0)
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Check for early stopping
        if patience_counter >= patience_limit:
            return True  # Signal to stop training
        
        return False  # Continue training
    
    # Run training
    trainSAE(
        data=buffer,
        trainer_configs=[trainer_config],
        steps=steps,
        save_dir=save_dir,
        save_steps=[steps],  # Only save at the end
        log_steps=max(1, steps // 50),  # Log approximately 50 times
        verbose=False,
        normalize_activations=True,
        autocast_dtype=t.bfloat16,
        use_wandb=True,  # Already initialized
        run_cfg=None,  # No need for extra config
    )
    
    # Evaluate the model
    try:
        # Load the trained dictionary
        sae = VSAETopK.from_pretrained(
            os.path.join(save_dir, "trainer_0", "ae.pt"),
            k=k,
            var_flag=var_flag,
            device="cuda"
        )
        
        # Run evaluation
        eval_results = evaluate(
            dictionary=sae,
            activations=buffer,
            batch_size=64,
            max_len=128,
            device="cuda",
            n_batches=10
        )
        
        # Log evaluation results
        for metric, value in eval_results.items():
            wandb.log({metric: value})
            
        # Compute a composite score to help with ranking
        composite_score = (
            eval_results.get("frac_variance_explained", 0) * 0.5 + 
            eval_results.get("frac_alive", 0) * 0.25 + 
            (eval_results.get("frac_recovered", 0) if "frac_recovered" in eval_results else 0) * 0.25
        )
        wandb.log({"composite_score": composite_score})
        
        # Save evaluation results
        with open(os.path.join(save_dir, "evaluation.json"), "w") as f:
            json.dump(eval_results, f, indent=4)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        
    # Clean up
    t.cuda.empty_cache()
    
    # Finish run
    wandb.finish()

def main():
    global buffer
    
    # Define model parameters for gelu-1l
    model_params = {
        "MODEL_NAME": "gelu-1l",
        "LAYER": 0,
        "HOOK_NAME": "blocks.0.mlp.hook_post",
        "STEPS": 15000,  # Reduced for sweeps
        "SWEEP_COUNT": 20,  # Number of runs per sweep
        
        # Buffer parameters
        "N_CTXS": 3000,
        "CTX_LEN": 128,
        "REFRESH_BATCH_SIZE": 32,
        "OUT_BATCH_SIZE": 1024,
        "DATASET": "NeelNanda/c4-code-tokenized-2b",
        
        # Dictionary size multiples to try
        "DICT_SIZE_MULTIPLES": [4, 8, 16],
        
        # WandB parameters
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "WANDB_PROJECT": "gelu_1l_vsae_topk",
    }
    
    # Set random seed for reproducibility
    random.seed(42)
    t.manual_seed(42)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(42)
    
    # Make sure output directory exists
    os.makedirs("./sweep_results/gelu_1l", exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        model_params["MODEL_NAME"], 
        device="cuda"
    )
    
    # Get model dimensions
    d_mlp = model.cfg.d_mlp
    print(f"Model dimensions: d_mlp={d_mlp}")
    
    # Set up data generator
    print("Setting up data generator...")
    data_gen = hf_dataset_to_generator(
        model_params["DATASET"], 
        split="train", 
        return_tokens=True
    )
    
    # Create activation buffer (shared across all sweeps)
    print("Creating activation buffer...")
    buffer = TransformerLensActivationBuffer(
        data=data_gen,
        model=model,
        hook_name=model_params["HOOK_NAME"],
        d_submodule=d_mlp,
        n_ctxs=model_params["N_CTXS"],
        ctx_len=model_params["CTX_LEN"],
        refresh_batch_size=model_params["REFRESH_BATCH_SIZE"],
        out_batch_size=model_params["OUT_BATCH_SIZE"],
        device="cuda",
    )
    
    # Run sweeps for each dictionary size multiple
    for dict_size_multiple in model_params["DICT_SIZE_MULTIPLES"]:
        dict_size = int(dict_size_multiple * d_mlp)
        print(f"\n\n{'='*50}")
        print(f"Starting sweep for dictionary size: {dict_size} ({dict_size_multiple}x)")
        print(f"{'='*50}")
        
        # Create sweep configuration
        sweep_config = create_sweep_config(
            dict_size, 
            model_params["MODEL_NAME"], 
            d_mlp, 
            model_params["STEPS"]
        )
        
        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=model_params["WANDB_PROJECT"],
            entity=model_params["WANDB_ENTITY"]
        )
        
        print(f"Sweep created with ID: {sweep_id}")
        print(f"Running {model_params['SWEEP_COUNT']} trials...")
        
        # Run sweep agent
        wandb.agent(
            sweep_id,
            function=train_vsae_topk,
            count=model_params["SWEEP_COUNT"]
        )
        
        print(f"Completed sweep for dictionary size: {dict_size}")
    
    print("\n\nAll sweeps completed!")

if __name__ == "__main__":
    # Set the start method to spawn for compatibility with CUDA
    multiprocessing.set_start_method('spawn', force=True)
    # Run the main function
    main()
