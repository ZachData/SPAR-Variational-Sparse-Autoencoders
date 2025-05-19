import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_panneal import VSAEPAnnealTrainer, VSAEPAnneal
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate
import multiprocessing
import os
import time
import argparse

def run_training(config):
    """Run the training with specified configuration"""
    # Extract parameters
    MODEL_NAME = config["MODEL_NAME"]
    LAYER = config["LAYER"]
    HOOK_NAME = config["HOOK_NAME"]
    TOTAL_STEPS = config["TOTAL_STEPS"]
    DICT_SIZE = config["DICT_SIZE"]
    LR = config["LR"]
    SPARSITY_PENALTY = config["SPARSITY_PENALTY"]
    SPARSITY_FUNCTION = config["SPARSITY_FUNCTION"]
    P_START = config["P_START"]
    P_END = config["P_END"]
    ANNEAL_START_FRAC = config["ANNEAL_START_FRAC"]
    ANNEAL_END_FRAC = config["ANNEAL_END_FRAC"]
    N_SPARSITY_UPDATES = config["N_SPARSITY_UPDATES"]
    WARMUP_FRAC = config["WARMUP_FRAC"]
    SPARSITY_WARMUP_FRAC = config["SPARSITY_WARMUP_FRAC"]
    DECAY_START_FRAC = config["DECAY_START_FRAC"]
    VAR_FLAG = config["VAR_FLAG"]
    LOG_STEPS = config["LOG_STEPS"]
    CHECKPOINT_FRACS = config["CHECKPOINT_FRACS"]
    WANDB_ENTITY = config["WANDB_ENTITY"]
    WANDB_PROJECT = config["WANDB_PROJECT"]
    USE_WANDB = config["USE_WANDB"]
    
    # Print configuration
    print(f"\n\n{'='*50}")
    print(f"STARTING TRAINING WITH THE FOLLOWING CONFIG:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(f"{'='*50}\n")
    
    # Load the model
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, 
        device="cuda"
    )
    
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
        d_submodule=model.cfg.d_mlp,
        n_ctxs=config["N_CTXS"],
        ctx_len=config["CTX_LEN"],
        refresh_batch_size=config["REFRESH_BATCH_SIZE"],
        out_batch_size=config["OUT_BATCH_SIZE"],
        device="cuda",
    )
    
    # Configure trainer
    ANNEAL_START = int(ANNEAL_START_FRAC * TOTAL_STEPS)
    ANNEAL_END = int(ANNEAL_END_FRAC * TOTAL_STEPS)
    SAVE_STEPS = [int(frac * TOTAL_STEPS) for frac in CHECKPOINT_FRACS]
    
    trainer_config = {
        "trainer": VSAEPAnnealTrainer,
        "steps": TOTAL_STEPS,
        "activation_dim": model.cfg.d_mlp,
        "dict_size": DICT_SIZE,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": LR,
        "sparsity_penalty": SPARSITY_PENALTY,
        "warmup_steps": int(WARMUP_FRAC * TOTAL_STEPS),
        "sparsity_warmup_steps": int(SPARSITY_WARMUP_FRAC * TOTAL_STEPS),
        "decay_start": int(DECAY_START_FRAC * TOTAL_STEPS),
        "var_flag": VAR_FLAG,
        "use_april_update_mode": True,
        "device": "cuda",
        "wandb_name": f"VSAEPAnneal_{MODEL_NAME}_{DICT_SIZE}_lr{LR}_sp{SPARSITY_PENALTY}_p{P_START}-{P_END}",
        "dict_class": VSAEPAnneal,
        "sparsity_function": SPARSITY_FUNCTION,
        "anneal_start": ANNEAL_START,
        "anneal_end": ANNEAL_END,
        "p_start": P_START,
        "p_end": P_END,
        "n_sparsity_updates": N_SPARSITY_UPDATES
    }
    
    # Set unique save directory for this run
    save_dir = f"./VSAEPAnneal_{MODEL_NAME}_d{DICT_SIZE}_lr{LR}_sp{SPARSITY_PENALTY}_p{P_START}-{P_END}"
    
    # Run training
    start_time = time.time()
    
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
            "experiment_type": "vsae_panneal",
            "sparsity_penalty": SPARSITY_PENALTY,
            "p_start": P_START,
            "p_end": P_END,
            "sparsity_function": SPARSITY_FUNCTION,
            "var_flag": VAR_FLAG
        }
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the trained model
    from dictionary_learning.utils import load_dictionary
    
    # Load the trained dictionary
    vsae, config = load_dictionary(f"{save_dir}/trainer_0", device="cuda")
    
    # Evaluate on a small batch
    eval_results = evaluate(
        dictionary=vsae,
        activations=buffer,
        batch_size=64,
        max_len=config["CTX_LEN"],
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
    parser = argparse.ArgumentParser(description="Train a VSAEPAnneal model")
    
    # Model parameters
    parser.add_argument("--model", default="gelu-1l", help="Model name to use")
    parser.add_argument("--layer", type=int, default=0, help="Layer to train on")
    parser.add_argument("--hook-name", default="blocks.0.mlp.hook_post", help="Hook name to extract activations from")
    parser.add_argument("--dict-size", type=int, default=None, help="Dictionary size (if None, will use 4 * d_mlp)")
    parser.add_argument("--dict-size-multiple", type=float, default=4.0, help="Dictionary size as multiple of d_mlp (only used if dict_size is None)")

    # Training parameters
    parser.add_argument("--steps", type=int, default=30000, help="Total number of training steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--sparsity-penalty", type=float, default=5.0, help="Initial sparsity penalty coefficient")
    
    # P-annealing parameters
    parser.add_argument("--sparsity-function", choices=["Lp", "Lp^p"], default="Lp", help="Type of sparsity function to use")
    parser.add_argument("--p-start", type=float, default=1.0, help="Starting p value")
    parser.add_argument("--p-end", type=float, default=0.5, help="Ending p value")
    parser.add_argument("--anneal-start-frac", type=float, default=0.1, help="Fraction of steps to start annealing p")
    parser.add_argument("--anneal-end-frac", type=float, default=0.9, help="Fraction of steps to end annealing p")
    parser.add_argument("--n-sparsity-updates", type=str, default="10", help="Number of sparsity updates (integer or 'continuous')")
    
    # VAE parameters
    parser.add_argument("--var-flag", type=int, default=0, choices=[0, 1], help="Whether to learn variance (0: fixed, 1: learned)")
    
    # Step fractions
    parser.add_argument("--warmup-frac", type=float, default=0.05, help="Fraction of steps for learning rate warmup")
    parser.add_argument("--sparsity-warmup-frac", type=float, default=0.05, help="Fraction of steps for sparsity warmup")
    parser.add_argument("--decay-start-frac", type=float, default=0.8, help="Fraction of steps to start learning rate decay")
    
    # Buffer parameters
    parser.add_argument("--n-ctxs", type=int, default=3000, help="Number of contexts in buffer")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--refresh-batch-size", type=int, default=32, help="Batch size for refreshing buffer")
    parser.add_argument("--out-batch-size", type=int, default=1024, help="Batch size for training")
    
    # Saving parameters
    parser.add_argument("--checkpoint-fracs", nargs="+", type=float, default=[0.5, 1.0], help="Fractions at which to save checkpoints")
    parser.add_argument("--log-steps", type=int, default=100, help="Steps between logging")
    
    # WandB parameters
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""), help="WandB entity")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vsae-panneal-experiments"), help="WandB project")
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB for logging")
    
    args = parser.parse_args()
    
    # Convert n_sparsity_updates to int if it's not "continuous"
    if args.n_sparsity_updates != "continuous":
        args.n_sparsity_updates = int(args.n_sparsity_updates)
    
    # Create the config dictionary
    config = {
        # Model parameters
        "MODEL_NAME": args.model,
        "LAYER": args.layer,
        "HOOK_NAME": args.hook_name,
        
        # Dictionary size - calculate if not provided
        "DICT_SIZE": args.dict_size,  # Will be set after loading model if None
        "DICT_SIZE_MULTIPLE": args.dict_size_multiple,
        
        # Training parameters
        "TOTAL_STEPS": args.steps,
        "LR": args.lr,
        "SPARSITY_PENALTY": args.sparsity_penalty,
        
        # P-annealing parameters
        "SPARSITY_FUNCTION": args.sparsity_function,
        "P_START": args.p_start,
        "P_END": args.p_end,
        "ANNEAL_START_FRAC": args.anneal_start_frac,
        "ANNEAL_END_FRAC": args.anneal_end_frac,
        "N_SPARSITY_UPDATES": args.n_sparsity_updates,
        
        # VAE parameters
        "VAR_FLAG": args.var_flag,
        
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
        "CHECKPOINT_FRACS": args.checkpoint_fracs,
        "LOG_STEPS": args.log_steps,
        
        # WandB parameters
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_PROJECT": args.wandb_project,
        "USE_WANDB": args.use_wandb,
    }
    
    # Load model just to get d_mlp
    if config["DICT_SIZE"] is None:
        temp_model = HookedTransformer.from_pretrained(config["MODEL_NAME"], device="cpu")
        config["DICT_SIZE"] = int(temp_model.cfg.d_mlp * config["DICT_SIZE_MULTIPLE"])
        del temp_model  # Free memory
        print(f"Dictionary size set to {config['DICT_SIZE']} ({config['DICT_SIZE_MULTIPLE']} * d_mlp)")
        
    # Train the model
    results, save_dir = run_training(config)
    
    # Print final results
    print("\n\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {save_dir}")
    print(f"Results: {results}")
    print("="*50)

if __name__ == "__main__":
    # Set the start method to spawn for multiprocessing compatibility
    multiprocessing.set_start_method('spawn', force=True)
    # Call the main function
    main()