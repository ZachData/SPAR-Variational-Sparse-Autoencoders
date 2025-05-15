"""
Benchmark script for comparing VSAEIso models with standard SAE models using SAE Bench.
This script follows the SAE Bench pattern more closely by extending the BaseSAE class
and using SAE Bench's evaluation utilities.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json

# SAE Bench imports
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.evals.core.main as core
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils
from sae_bench.custom_saes.base_sae import BaseSAE

# Set random seed for reproducibility
RANDOM_SEED = 42

class VSAEIsoAdapter(BaseSAE):
    """
    Adapter class that wraps a VSAEIso model from dictionary_learning
    to match the SAE Bench BaseSAE interface.
    """
    def __init__(
        self,
        vsaeiso_path: str,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the VSAEIso adapter.
        
        Args:
            vsaeiso_path: Path to the VSAEIso model directory or file
            model_name: Name of the model the SAE was trained on
            hook_layer: Model layer where the SAE was applied
            device: Device to run on
            dtype: Data type to use
        """
        # First load the VSAEIso model
        vsaeiso, config = self._load_vsaeiso(vsaeiso_path, device, dtype)
        
        # Get dimensions from the VSAEIso model
        d_in = vsaeiso.activation_dim
        d_sae = vsaeiso.dict_size
        
        # Determine hook name based on the model config
        hook_name = config.get("hook_name", f"blocks.{hook_layer}.mlp.hook_post")
        
        # Print debug info to understand dimensions
        print(f"VSAEIso dimensions - d_in: {d_in}, d_sae: {d_sae}")
        print(f"VSAEIso encoder weight shape: {vsaeiso.encoder.weight.shape}")
        print(f"VSAEIso decoder weight shape: {vsaeiso.decoder.weight.shape}")
        
        # Initialize the BaseSAE parent class with the correct dimensions
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=hook_name,
        )
        
        # Store the VSAEIso model
        self.vsaeiso = vsaeiso
        
        # Reset parameter shapes to match the model
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        
        # Copy weights and biases from VSAEIso to match BaseSAE interface
        with torch.no_grad():
            # Set encoder weights (transpose to match BaseSAE convention)
            self.W_enc.copy_(vsaeiso.encoder.weight.T)
            
            # Set decoder weights (transpose since our model has it transposed)
            self.W_dec.copy_(vsaeiso.decoder.weight.T)
            
            # Set encoder bias
            self.b_enc.copy_(vsaeiso.encoder.bias)
            
            # Set decoder bias (may be in different attribute based on mode)
            if hasattr(vsaeiso, 'decoder') and hasattr(vsaeiso.decoder, 'bias'):
                self.b_dec.copy_(vsaeiso.decoder.bias)
            elif hasattr(vsaeiso, 'bias'):
                self.b_dec.copy_(vsaeiso.bias)
            
            # Normalize decoder weights as expected by SAE Bench
            self._normalize_decoder_weights()
            
        # Set up additional configuration
        self.cfg.architecture = "vsaeiso"
        self.cfg.training_tokens = config.get("training_tokens", 384_000_000)
        
        # Optional fields based on the config
        if "kl_coeff" in config:
            self.cfg.kl_coeff = config["kl_coeff"]
        if "dict_size_multiple" in config:
            self.cfg.dict_size_multiple = config["dict_size_multiple"]
    
    def _load_vsaeiso(self, vsaeiso_path, device, dtype):
        """
        Load a VSAEIso model from the given path.
        
        Args:
            vsaeiso_path: Path to the VSAEIso model directory or file
            device: Device to load the model to
            dtype: Data type to convert the model to
            
        Returns:
            Tuple of (vsaeiso_model, config_dict)
        """
        from dictionary_learning.utils import load_dictionary
        
        if os.path.isfile(vsaeiso_path):
            model_file = vsaeiso_path
        elif os.path.isdir(vsaeiso_path):
            # Look for ae.pt in the directory
            model_file = os.path.join(vsaeiso_path, "trainer_0", "ae.pt")
            if not os.path.exists(model_file):
                model_file = os.path.join(vsaeiso_path, "ae.pt")
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Could not find ae.pt in {vsaeiso_path}")
        else:
            raise ValueError(f"Invalid path: {vsaeiso_path}")
        
        # Load the model
        model_dir = os.path.dirname(model_file)
        config_file = os.path.join(model_dir, "config.json")
        
        # Load the VSAEIso model
        vsaeiso, config = load_dictionary(model_dir, device=device)
        vsaeiso.to(device=device, dtype=dtype)
        
        # Add default config values if missing
        if "hook_name" not in config:
            config["hook_name"] = f"blocks.{config['trainer']['layer']}.mlp.hook_post"
        
        return vsaeiso, config
    
    def _normalize_decoder_weights(self):
        """
        Normalize decoder weights to match SAE Bench expectations.
        """
        with torch.no_grad():
            # SAE Bench expects row-normalized decoder weights (not column-normalized)
            row_norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            self.W_dec.div_(row_norms)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations using the VSAEIso model.
        
        Args:
            x: Input activations [batch_size, d_in]
            
        Returns:
            Encoded features [batch_size, d_sae]
        """
        return self.vsaeiso.encode(x)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to reconstructed activations.
        
        Args:
            features: Encoded features [batch_size, d_sae]
            
        Returns:
            Reconstructed activations [batch_size, d_in]
        """
        return self.vsaeiso.decode(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VSAEIso model.
        
        Args:
            x: Input activations [batch_size, d_in]
            
        Returns:
            Reconstructed activations [batch_size, d_in]
        """
        return self.vsaeiso(x)


class StandardSAEAdapter(BaseSAE):
    """
    Adapter class that wraps a standard SAE model from dictionary_learning
    to match the SAE Bench BaseSAE interface.
    """
    def __init__(
        self,
        sae_path: str,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the Standard SAE adapter.
        
        Args:
            sae_path: Path to the Standard SAE model directory or file
            model_name: Name of the model the SAE was trained on
            hook_layer: Model layer where the SAE was applied
            device: Device to run on
            dtype: Data type to use
        """
        # First load the Standard SAE model
        sae, config = self._load_standard_sae(sae_path, device, dtype)
        
        # Get dimensions from the SAE model
        d_in = sae.activation_dim
        d_sae = sae.dict_size
        
        # Determine hook name based on the model config
        hook_name = config.get("hook_name", f"blocks.{hook_layer}.mlp.hook_post")
        
        # Print debug info to understand dimensions
        print(f"Standard SAE dimensions - d_in: {d_in}, d_sae: {d_sae}")
        print(f"Standard SAE encoder weight shape: {sae.encoder.weight.shape}")
        print(f"Standard SAE decoder weight shape: {sae.decoder.weight.shape}")
        
        # Initialize the BaseSAE parent class with the correct dimensions
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=hook_name,
        )
        
        # Store the Standard SAE model
        self.sae = sae
        
        # Reset parameter shapes to match the model
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        
        # Copy weights and biases from Standard SAE to match BaseSAE interface
        with torch.no_grad():
            # Set encoder weights (transpose to match BaseSAE convention)
            self.W_enc.copy_(sae.encoder.weight.T)
            
            # Set decoder weights (transpose since our model has it transposed)
            self.W_dec.copy_(sae.decoder.weight.T)
            
            # Set encoder bias
            self.b_enc.copy_(sae.encoder.bias)
            
            # Set decoder bias (may be in different attribute based on mode)
            if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'bias'):
                self.b_dec.copy_(sae.decoder.bias)
            elif hasattr(sae, 'bias'):
                self.b_dec.copy_(sae.bias)
            
            # Normalize decoder weights as expected by SAE Bench
            self._normalize_decoder_weights()
            
        # Set up additional configuration
        self.cfg.architecture = "standard"
        self.cfg.training_tokens = config.get("training_tokens", 384_000_000)
        
        # Optional fields based on the config
        if "l1_penalty" in config:
            self.cfg.l1_penalty = config["l1_penalty"]
        if "dict_size_multiple" in config:
            self.cfg.dict_size_multiple = config["dict_size_multiple"]
    
    def _load_standard_sae(self, sae_path, device, dtype):
        """
        Load a Standard SAE model from the given path.
        
        Args:
            sae_path: Path to the Standard SAE model directory or file
            device: Device to load the model to
            dtype: Data type to convert the model to
            
        Returns:
            Tuple of (sae_model, config_dict)
        """
        from dictionary_learning.utils import load_dictionary
        
        if os.path.isfile(sae_path):
            model_file = sae_path
        elif os.path.isdir(sae_path):
            # Look for ae.pt in the directory
            model_file = os.path.join(sae_path, "trainer_0", "ae.pt")
            if not os.path.exists(model_file):
                model_file = os.path.join(sae_path, "ae.pt")
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Could not find ae.pt in {sae_path}")
        else:
            raise ValueError(f"Invalid path: {sae_path}")
        
        # Load the model
        model_dir = os.path.dirname(model_file)
        config_file = os.path.join(model_dir, "config.json")
        
        # Load the Standard SAE model
        sae, config = load_dictionary(model_dir, device=device)
        sae.to(device=device, dtype=dtype)
        
        # Add default config values if missing
        if "hook_name" not in config:
            config["hook_name"] = f"blocks.{config['trainer']['layer']}.mlp.hook_post"
        
        return sae, config
    
    def _normalize_decoder_weights(self):
        """
        Normalize decoder weights to match SAE Bench expectations.
        """
        with torch.no_grad():
            # SAE Bench expects row-normalized decoder weights (not column-normalized)
            row_norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            self.W_dec.div_(row_norms)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations using the Standard SAE model.
        
        Args:
            x: Input activations [batch_size, d_in]
            
        Returns:
            Encoded features [batch_size, d_sae]
        """
        return self.sae.encode(x)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to reconstructed activations.
        
        Args:
            features: Encoded features [batch_size, d_sae]
            
        Returns:
            Reconstructed activations [batch_size, d_in]
        """
        return self.sae.decode(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Standard SAE model.
        
        Args:
            x: Input activations [batch_size, d_in]
            
        Returns:
            Reconstructed activations [batch_size, d_in]
        """
        return self.sae(x)


def setup_output_folders():
    """Create output folders for evaluation results and images"""
    output_folders = {
        "core": "eval_results/core",
        "sparse_probing": "eval_results/sparse_probing",
        "scr": "eval_results/scr",
        "tpp": "eval_results/tpp",
    }

    # Create output directories if they don't exist
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Create directory for images
    image_path = "./benchmark_images"
    os.makedirs(image_path, exist_ok=True)
    
    return output_folders, image_path


def run_benchmark(
    vsaeiso_path=None,
    standard_sae_path=None,
    model_name="gelu-1l",
    hook_layer=0,
    dtype=torch.float32,
    eval_types=None,
    force_rerun=False,
    save_activations=False,
):
    """
    Run benchmark comparing VSAEIso and Standard SAE models using SAE Bench.
    
    Args:
        vsaeiso_path: Path to VSAEIso model directory or file
        standard_sae_path: Path to Standard SAE model directory or file
        model_name: Name of the model the SAEs were trained on
        hook_layer: Model layer where the SAEs were applied
        dtype: Data type to use for evaluation
        eval_types: List of evaluation types to run
        force_rerun: Force rerun of evaluations even if results exist
        save_activations: Save activations for reuse
    """
    # Setup folders
    output_folders, image_path = setup_output_folders()
    
    # Setup environment
    device = general_utils.setup_environment()
    
    # Default evaluation types if not specified
    if eval_types is None:
        eval_types = ["core", "sparse_probing"]
    
    # Convert torch dtype to string format
    str_dtype = str(dtype).split(".")[-1]
    
    # Setup visualization settings
    trainer_markers = {
        "standard": "o",
        "vsaeiso": "s",
    }

    trainer_colors = {
        "standard": "blue",
        "vsaeiso": "red",
    }
    
    # Load models and create adapters
    custom_saes = []
    
    # Load VSAEIso model if provided
    if vsaeiso_path:
        try:
            vsae = VSAEIsoAdapter(
                vsaeiso_path=vsaeiso_path,
                model_name=model_name,
                hook_layer=hook_layer,
                device=device,
                dtype=dtype,
            )
            unique_vsaeiso_id = f"vsaeiso_{model_name.replace('-', '_')}_d{vsae.cfg.d_sae}"
            custom_saes.append((unique_vsaeiso_id, vsae))
            print(f"Successfully loaded VSAEIso model: {unique_vsaeiso_id}")
        except Exception as e:
            print(f"Error loading VSAEIso model: {e}")
            import traceback
            traceback.print_exc()
    
    # Load Standard SAE model if provided
    if standard_sae_path:
        try:
            standard_sae = StandardSAEAdapter(
                sae_path=standard_sae_path,
                model_name=model_name,
                hook_layer=hook_layer,
                device=device,
                dtype=dtype,
            )
            unique_standard_sae_id = f"standard_{model_name.replace('-', '_')}_d{standard_sae.cfg.d_sae}"
            custom_saes.append((unique_standard_sae_id, standard_sae))
            print(f"Successfully loaded Standard SAE model: {unique_standard_sae_id}")
        except Exception as e:
            print(f"Error loading Standard SAE model: {e}")
            import traceback
            traceback.print_exc()
    
    if not custom_saes:
        print("No models could be loaded. Exiting.")
        return
    
    # Set up configuration for SAE Bench
    selected_saes = custom_saes
    llm_batch_size = 32  # Smaller batch size to avoid OOM
    
    # Run core evaluations
    if "core" in eval_types:
        try:
            print("\nRunning core evaluations...")
            _ = core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=50,  # Reduced for faster benchmarking
                n_eval_sparsity_variance_batches=50,  # Reduced for faster benchmarking
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=output_folders["core"],
                verbose=True,
                dtype=str_dtype,
            )
            print("Core evaluations completed successfully!")
        except Exception as e:
            print(f"Error during core evaluations: {e}")
            import traceback
            traceback.print_exc()
    
    # Run sparse probing evaluations
    if "sparse_probing" in eval_types:
        try:
            print("\nRunning sparse probing evaluations...")
            dataset_names = ["LabHC/bias_in_bios_class_set1"]  # Using a subset for benchmarking
            
            _ = sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=str_dtype,
                    dataset_names=dataset_names,
                ),
                selected_saes,
                device,
                output_folders["sparse_probing"],
                force_rerun=force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
            print("Sparse probing evaluations completed successfully!")
        except Exception as e:
            print(f"Error during sparse probing evaluations: {e}")
            import traceback
            traceback.print_exc()
    
    # Run additional evaluations if requested
    import sae_bench.custom_saes.run_all_evals_custom_saes as run_all_evals
    
    if any(eval_type in ["scr", "tpp"] for eval_type in eval_types):
        additional_eval_types = [eval_type for eval_type in eval_types if eval_type in ["scr", "tpp"]]
        try:
            print(f"\nRunning additional evaluations: {additional_eval_types}")
            _ = run_all_evals.run_evals(
                model_name,
                selected_saes,
                llm_batch_size,
                str_dtype,
                device,
                additional_eval_types,
                api_key=None,
                force_rerun=force_rerun,
                save_activations=save_activations,
            )
        except Exception as e:
            print(f"Error during additional evaluations: {e}")
            import traceback
            traceback.print_exc()
    
    # Load and display results
    print("\nGenerating result summary and plots...")
    
    results_folders = ["./eval_results"]
    core_folders = [f"{folder}/core" for folder in results_folders]
    
    # Find and plot results for each evaluation type
    for eval_type in eval_types:
        eval_folders = [f"{folder}/{eval_type}" for folder in results_folders]
        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
        core_filenames = graphing_utils.find_eval_results_files(core_folders)
        
        if not eval_filenames:
            print(f"No results found for {eval_type}. Skipping plots.")
            continue
        
        try:
            # Load results
            eval_results_dict = graphing_utils.get_eval_results(eval_filenames)
            core_results_dict = graphing_utils.get_eval_results(core_filenames)
            
            # Merge results
            for sae in eval_results_dict:
                if sae in core_results_dict:
                    eval_results_dict[sae].update(core_results_dict[sae])
            
            # Print comparison of key metrics
            print(f"\n===== {eval_type.upper()} EVALUATION RESULTS =====")
            
            # Define relevant metrics based on evaluation type
            if eval_type == "core":
                key_metrics = ["frac_variance_explained", "l0", "frac_alive"]
            elif eval_type == "sparse_probing":
                key_metrics = ["sae_top_1_test_accuracy", "sae_top_10_test_accuracy", "l0"]
            elif eval_type == "scr":
                key_metrics = ["scr_acc_delta", "scr_delta_improvement", "l0"]
            elif eval_type == "tpp":
                key_metrics = ["tpp_score", "l0"]
            else:
                key_metrics = ["l0"]
            
            # Print header
            model_names = [sae_id for sae_id, _ in selected_saes]
            header = f"{'Metric':<30} | " + " | ".join(f"{name[:15]:<15}" for name in model_names)
            print(header)
            print("-" * (30 + 18 * len(model_names)))
            
            # Print metrics comparison
            for metric in key_metrics:
                values = []
                for sae_id, _ in selected_saes:
                    value = "N/A"
                    # Try finding the metric in the results
                    for key in eval_results_dict:
                        if sae_id in key or key in sae_id:
                            sae_results = eval_results_dict[key]
                            
                            # Look for the metric in different locations
                            if metric in sae_results:
                                if isinstance(sae_results[metric], (int, float)):
                                    value = f"{sae_results[metric]:.4f}"
                                else:
                                    value = str(sae_results[metric])
                            elif "sae" in sae_results and f"{metric}" in sae_results["sae"]:
                                if isinstance(sae_results["sae"][f"{metric}"], (int, float)):
                                    value = f"{sae_results['sae'][f'{metric}']:.4f}"
                                else:
                                    value = str(sae_results["sae"][f"{metric}"])
                            elif "eval_result_metrics" in sae_results:
                                if "sae" in sae_results["eval_result_metrics"] and f"sae_{metric}" in sae_results["eval_result_metrics"]["sae"]:
                                    if isinstance(sae_results["eval_result_metrics"]["sae"][f"sae_{metric}"], (int, float)):
                                        value = f"{sae_results['eval_result_metrics']['sae'][f'sae_{metric}']:.4f}"
                                    else:
                                        value = str(sae_results["eval_result_metrics"]["sae"][f"sae_{metric}"])
                            break
                    values.append(value)
                
                row = f"{metric:<30} | " + " | ".join(f"{val:<15}" for val in values)
                print(row)
            
            # Generate plots
            image_base_name = os.path.join(image_path, f"{eval_type}")
            
            try:
                graphing_utils.plot_results(
                    eval_filenames,
                    core_filenames,
                    eval_type,
                    image_base_name,
                    k=10,  # Top-k accuracy to plot for sparse probing
                    trainer_markers=trainer_markers,
                    trainer_colors=trainer_colors,
                )
                print(f"Plots saved to {image_base_name}*")
            except Exception as e:
                print(f"Error generating plots for {eval_type}: {e}")
        except Exception as e:
            print(f"Error processing results for {eval_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark VSAEIso and Standard SAE models using SAE Bench")
    parser.add_argument("--vsaeiso_path", type=str, default=None, 
                      help="Path to VSAEIso model directory or .pt file")
    parser.add_argument("--standard_sae_path", type=str, default=None,
                      help="Path to Standard SAE model directory or .pt file")
    parser.add_argument("--model_name", type=str, default="gelu-1l",
                      help="Name of the model the SAEs were trained on")
    parser.add_argument("--hook_layer", type=int, default=0,
                      help="Model layer where the SAEs were applied")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"],
                      help="Data type to use for evaluation")
    parser.add_argument("--eval_types", type=str, nargs="+", default=["core", "sparse_probing"],
                      choices=["core", "sparse_probing", "scr", "tpp"],
                      help="Evaluation types to run")
    parser.add_argument("--force_rerun", action="store_true",
                      help="Force rerun of evaluations even if results exist")
    parser.add_argument("--save_activations", action="store_true",
                      help="Save activations for reuse (requires more disk space)")
    
    args = parser.parse_args()
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Run the benchmark
    run_benchmark(
        vsaeiso_path=args.vsaeiso_path,
        standard_sae_path=args.standard_sae_path,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        dtype=dtype,
        eval_types=args.eval_types,
        force_rerun=args.force_rerun,
        save_activations=args.save_activations,
    )
