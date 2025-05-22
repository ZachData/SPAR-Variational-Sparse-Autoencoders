import argparse
import os
import wandb
import uuid
import torch as t
import numpy as np
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.matryoshka_vsae_iso import MatryoshkaVSAEIsoTrainer, MatryoshkaVSAEIso
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_with_config(config=None):
    """
    Train a single model with the given wandb config
    
    This function is called by wandb.agent for each hyperparameter combination
    """
    with wandb.init(config=config):
        # Get the configuration for this run
        config = wandb.config
        
        # Extract parameters from config
        model_name = config.model_name
        layer = config.layer
        hook_name = config.hook_name
        dict_size_multiple = config.dict_size_multiple
        lr = config.lr
        kl_coeff = config.kl_coeff
        auxk_alpha = config.auxk_alpha
        total_steps = config.total_steps
        group_fractions = config.group_fractions
        var_flag = config.var_flag
        
        # Generate group weights based on group count
        # Makes weights higher for later (larger) groups
        num_groups = len(group_fractions)
        group_weights = [2 ** i for i in range(num_groups)]
        total_weight = sum(group_weights)
        group_weights = [w / total_weight for w in group_weights]
        
        # Load model if not already loaded or if we need a different model
        global model, buffer
        if 'model' not in globals() or model.cfg.model_name != model_name:
            logger.info(f"Loading model: {model_name}")
            model = HookedTransformer.from_pretrained(model_name, device="cuda")
            
            # Setup data generator
            logger.info("Setting up data generator")
            data_gen = hf_dataset_to_generator(
                "NeelNanda/c4-code-tokenized-2b", 
                split="train", 
                return_tokens=True
            )
            
            # Create activation buffer
            logger.info("Creating activation buffer")
            buffer = TransformerLensActivationBuffer(
                data=data_gen,
                model=model,
                hook_name=hook_name,
                d_submodule=model.cfg.d_mlp,
                n_ctxs=3000,
                ctx_len=128,
                refresh_batch_size=32,
                out_batch_size=1024,
                device="cuda",
            )
        
        # Calculate derived parameters
        d_mlp = model.cfg.d_mlp
        dict_size = int(dict_size_multiple * d_mlp)
        save_steps = [int(0.5 * total_steps), int(total_steps)]
        
        # Configure trainer for this run
        trainer_config = {
            "trainer": MatryoshkaVSAEIsoTrainer,
            "steps": total_steps,
            "activation_dim": d_mlp,
            "dict_size": dict_size,
            "layer": layer,
            "lm_name": model_name,
            "lr": lr,
            "kl_coeff": kl_coeff,
            "auxk_alpha": auxk_alpha,
            "warmup_steps": int(0.05 * total_steps),
            "sparsity_warmup_steps": int(0.05 * total_steps),
            "decay_start": int(0.8 * total_steps),
            "var_flag": var_flag,
            "group_fractions": group_fractions,
            "group_weights": group_weights,
            "device": "cuda",
            "wandb_name": f"MatryoshkaVSAE_{model_name}_{dict_size}_lr{lr}_kl{kl_coeff}",
            "dict_class": MatryoshkaVSAEIso
        }
        
        # Set unique save directory for this run
        run_id = wandb.run.id
        save_dir = f"./sweeps/MatryoshkaVSAEIso_{model_name}_d{dict_size}_{run_id}"
        
        # Log all hyperparameters to wandb
        wandb.log({
            "dict_size": dict_size,
            "dict_size_multiple": dict_size_multiple,
            "activation_dim": d_mlp,
            "group_fractions": group_fractions,
            "group_weights": group_weights,
            "var_flag": var_flag
        })
        
        # Run training
        logger.info(f"Starting training with dict_size={dict_size}, lr={lr}, kl_coeff={kl_coeff}, auxk_alpha={auxk_alpha}")
        trainSAE(
            data=buffer,
            trainer_configs=[trainer_config],
            steps=total_steps,
            save_dir=save_dir,
            save_steps=save_steps,
            log_steps=100,
            verbose=False,
            normalize_activations=True,
            autocast_dtype=t.bfloat16,
            use_wandb=True,
            wandb_entity=wandb.run.entity,
            wandb_project=wandb.run.project,
            run_cfg={
                "model_type": model_name,
                "experiment_type": "matryoshka_vsae_iso",
                "kl_coefficient": kl_coeff,
                "auxk_alpha": auxk_alpha,
                "dict_size_multiple": dict_size_multiple,
                "var_flag": var_flag,
                "group_fractions": group_fractions,
            }
        )
        
        # Evaluate the trained model
        from dictionary_learning.utils import load_dictionary
        
        # Load the trained dictionary
        try:
            sae, _ = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
            
            # Evaluate on a small batch
            eval_results = evaluate(
                dictionary=sae,
                activations=buffer,
                batch_size=64,
                max_len=128,
                device="cuda",
                n_batches=10
            )
            
            # Log evaluation metrics
            wandb.log(eval_results)
            
            # For Hyperband early stopping, use frac_variance_explained as the key metric
            # Higher is better, so we negate it for minimization
            if "frac_variance_explained" in eval_results:
                validation_loss = -eval_results["frac_variance_explained"]
                wandb.log({"validation_loss": validation_loss})
            elif "frac_recovered" in eval_results:
                validation_loss = -eval_results["frac_recovered"]
                wandb.log({"validation_loss": validation_loss})
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # Use a high validation loss as fallback
            wandb.log({"validation_loss": 999.0})
        
        # Clear CUDA cache to prevent memory issues between runs
        t.cuda.empty_cache()

def run_sweep(args):
    """Run a hyperparameter sweep using wandb"""
    
    # For each size multiple, run a separate sweep
    for dict_size_multiple in [4, 8, 16]:
        # Generate unique sweep ID for this size
        sweep_id = str(uuid.uuid4())[:8]
        
        # Group fractions examples per size
        if dict_size_multiple == 4:
            # Fewer groups for smaller dictionary
            group_fractions_options = [
                [0.2, 0.3, 0.5],  # 3 groups
                [0.1, 0.2, 0.3, 0.4]  # 4 groups
            ]
        elif dict_size_multiple == 8:
            # More groups for medium dictionary
            group_fractions_options = [
                [0.1, 0.2, 0.3, 0.4],  # 4 groups
                [0.05, 0.1, 0.15, 0.3, 0.4]  # 5 groups
            ]
        else:  # dict_size_multiple == 16
            # Most groups for largest dictionary
            group_fractions_options = [
                [0.05, 0.1, 0.15, 0.3, 0.4],  # 5 groups
                [0.05, 0.05, 0.1, 0.1, 0.2, 0.5]  # 6 groups
            ]
        
        # Configure the sweep
        sweep_config = {
            "method": "bayes",  # Bayesian optimization
            "metric": {
                "name": "validation_loss",  # We'll use -frac_variance_explained
                "goal": "minimize"
            },
            "parameters": {
                "model_name": {"value": args.model_name},
                "layer": {"value": args.layer},
                "hook_name": {"value": args.hook_name},
                "dict_size_multiple": {"value": dict_size_multiple},
                "total_steps": {"value": args.total_steps},
                
                # Parameters to sweep over
                "lr": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-4
                },
                "kl_coeff": {
                    "distribution": "log_uniform_values",
                    "min": 0.1,
                    "max": 50.0
                },
                "auxk_alpha": {
                    "distribution": "log_uniform_values",
                    "min": 0.001,
                    "max": 0.1
                },
                "var_flag": {
                    "values": [0, 1]  # Test both fixed and learned variance
                },
                "group_fractions": {
                    "values": group_fractions_options
                }
            },
            "early_terminate": {
                "type": "hyperband",
                "max_iter": args.total_steps,
                "s": 2,
                "eta": 3
            }
        }
        
        # Initialize the sweep
        variation_name = f"dict_size_{dict_size_multiple}x"
        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.wandb_project,
            entity=args.wandb_entity
        )
        
        # Start a wandb agent to run the sweep
        wandb.agent(
            sweep_id,
            function=train_model_with_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            count=args.count_per_size  # Number of runs per size multiple
        )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for MatryoshkaVSAEIso")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="gelu-1l", help="Model to use")
    parser.add_argument("--layer", type=int, default=0, help="Layer to extract activations from")
    parser.add_argument("--hook_name", type=str, default="blocks.0.mlp.hook_post", help="Hook name for activation extraction")
    
    # Training parameters
    parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps (lower for sweep)")
    parser.add_argument("--count_per_size", type=int, default=10, help="Number of runs per dictionary size")
    
    # WandB parameters
    parser.add_argument("--wandb_entity", type=str, default="", help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="matryoshka_vsae_sweep", help="WandB project")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Set default wandb entity from environment if not provided
    if not args.wandb_entity:
        args.wandb_entity = os.environ.get("WANDB_ENTITY", "")
    
    # Create directory for sweep results
    os.makedirs("./sweeps", exist_ok=True)
    
    # Run the sweep
    run_sweep(args)
