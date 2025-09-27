import os
from typing import Any
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset

import sae_bench.evals.absorption.main as absorption
# Skip autointerp - requires OpenAI API
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
# Skip unlearning - requires WMDP dataset
import sae_bench.sae_bench_utils.general_utils as general_utils

# Import your custom SAE wrapper  
import sys
from pathlib import Path
# Add the SAEBench-main root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dictionary_learning_wrapper import DictionaryLearningSAEWrapper

RANDOM_SEED = 42

MODEL_CONFIGS = { #look in dictionary learning wrapper! there are hardcoded values
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "bfloat16",
        "layers": [3],
        "d_model": 512,
    },
    "gemma-2-2b": {
        "batch_size": 32, 
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
    },
    # Add your gelu-1l model
    "gelu-1l": {
        "batch_size": 64,
        "dtype": "bfloat16", 
        "layers": [0],
        "d_model": 512,
    },
}

output_folders = {
    # "absorption": "eval_results/absorption", # Skip - cannot normalize weights easily
    # "autointerp": "eval_results/autointerp",  # Skip - requires OpenAI API
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    # "unlearning": "eval_results/unlearning",  # Skip - requires WMDP dataset
    # "ravel": "eval_results/ravel", # Skip - requires architecture support
}

def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, Any]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    api_key: str | None = None,
    force_rerun: bool = False,
    save_activations: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        # Skip autointerp - requires OpenAI API
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="NeelNanda/c4-code-tokenized-2b", # "roneneldan/TinyStories", 
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
                "ravel": (
            lambda: ravel.run_eval(
                ravel.RavelEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    perform_scr=True,
                ),
                selected_saes,
                device,
                "eval_results/scr",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    perform_scr=False,
                ),
                selected_saes,
                device,
                "eval_results/tpp",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    # dataset_names=["LabHC/bias_in_bios_class_set1"]
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        # Skip unlearning - requires WMDP dataset
    }

    # Run the selected evaluations
    for eval_type in eval_types:
        if eval_type in eval_runners:
            print(f"\n{'='*50}")
            print(f"Running {eval_type.upper()} evaluation")
            print(f"{'='*50}")
            try:
                eval_runners[eval_type]()
                print(f"Completed {eval_type} evaluation successfully")
            except Exception as e:
                print(f"{eval_type} evaluation failed: {e}")
        else:
            print(f"Unknown evaluation type: {eval_type}")


def load_your_trained_saes(experiments_dir: str = "../../../experiments") -> list[tuple[str, Any]]:
    """Load your trained VSAETopK models."""
    
    experiments_path = Path(experiments_dir)
    selected_saes = []
    
    # List of your trained models (update these names to match your actual trained models)
    model_names = [
        'VSAETopK_pythia70m_d8192_k64_lr0.0008_kl1.0_aux0_fixed_var',
        'VSAETopK_pythia70m_d8192_k128_lr0.0008_kl1.0_aux0_fixed_var',
        'VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var',
        'VSAETopK_pythia70m_d8192_k512_lr0.0008_kl1.0_aux0_fixed_var',
        'TopK_SAE_pythia70m_d8192_k64_auxk0.03125_lr_auto',
        'TopK_SAE_pythia70m_d8192_k128_auxk0.03125_lr_auto',
        'TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto',
        'TopK_SAE_pythia70m_d8192_k512_auxk0.03125_lr_auto',
    ]
    
    for model_name in model_names:
        model_path = experiments_path / model_name / "trainer_0"
        
        if model_path.exists():
            try:
                print(f"Loading SAE: {model_name}")
                sae = DictionaryLearningSAEWrapper.from_pretrained(str(model_path))
                selected_saes.append((model_name, sae))
                print(f"Loaded {model_name} - d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    return selected_saes


def main():
    """Main function to run evaluations on your trained models."""
    
    # Configuration
    model_name = "pythia-70m-deduped"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MODEL_CONFIGS[model_name]
    
    # Load your trained SAEs
    print("Loading trained SAEs...")
    selected_saes = load_your_trained_saes()
    
    if not selected_saes:
        print("No SAEs loaded. Please check your model paths.")
        return
    
    print(f"\nFound {len(selected_saes)} SAEs to evaluate")
    for sae_name, _ in selected_saes:
        print(f"  - {sae_name}")
    
    # Choose which evaluations to run (excluding ones that require special setup)
    eval_types = [
        "core",           # L0/Loss Recovered - fast and fundamental
        # "sparse_probing", # Sparse Probing - relatively fast
        # #"absorption",     # Feature Absorption - moderate time
        # "scr",           # Spurious Correlation Removal - moderate time
        # "tpp",           # Targeted Probe Perturbation - moderate time
        # # "ravel",        # RAVEL - skip for now, requires model architecture support
    ]
    
    print(f"\nRunning evaluations: {', '.join(eval_types)}")
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    
    # Run evaluations
    run_evals(
        model_name=model_name,
        selected_saes=selected_saes,
        llm_batch_size=config["batch_size"],
        llm_dtype=config["dtype"],
        device=device,
        eval_types=eval_types,
        force_rerun=False,  # Set to True if you want to overwrite existing results
        save_activations=False,  # Set to True if you want to cache activations (requires ~100GB)
    )
    
    print(f"\nAll evaluations completed!")
    print(f"Results saved in eval_results/ subdirectories")


if __name__ == "__main__":
    main()