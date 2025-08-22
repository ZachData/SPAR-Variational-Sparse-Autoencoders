import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import sae_bench.custom_saes.base_sae as base_sae


class VSAETopKSAE(base_sae.BaseSAE):
    """SAEBench wrapper for VSAETopK models trained with dictionary_learning."""
    
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        var_flag: int = 0,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.k = k
        self.var_flag = var_flag
        
        # Initialize VSAETopK-specific parameters
        self.register_buffer("k_tensor", torch.tensor(k, dtype=torch.int, device=device))

    def encode(self, x: torch.Tensor):
        """Forward through VSAETopK encoder."""
        # Delegate to the underlying VSAETopK model
        return self.vsae_model.encode(x)

    def decode(self, feature_acts: torch.Tensor):
        """Forward through VSAETopK decoder."""
        # Delegate to the underlying VSAETopK model
        return self.vsae_model.decode(feature_acts)

    def forward(self, x: torch.Tensor):
        """Forward pass through VSAETopK."""
        return self.vsae_model.forward(x)


def load_dictionary_learning_vsae_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int | None = None,
    local_dir: str = "downloaded_saes",
) -> VSAETopKSAE:
    """Load a VSAETopK model trained with dictionary_learning for use with SAEBench."""
    
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    trainer_config = config["trainer"]
    k = trainer_config["k"]
    var_flag = trainer_config.get("var_flag", 0)
    
    # Load the actual VSAETopK model using dictionary_learning's loader
    from dictionary_learning.utils import load_dictionary
    
    # Get the directory path (remove /ae.pt)
    model_dir = filename.replace("/ae.pt", "")
    full_path = f"{local_dir}/{repo_id.replace('/', '_')}/{model_dir}"
    
    # Load using dictionary_learning
    vsae_model, full_config = load_dictionary(full_path, device=device)

    # Create the SAEBench wrapper
    sae = VSAETopKSAE(
        d_in=trainer_config["activation_dim"],
        d_sae=trainer_config["dict_size"],
        k=k,
        model_name=model_name,
        hook_layer=layer,  # type: ignore
        device=device,
        dtype=dtype,
        var_flag=var_flag,
    )

    # Attach the loaded VSAETopK model
    sae.vsae_model = vsae_model
    
    # Copy weight matrices for SAEBench compatibility
    if hasattr(vsae_model, 'W_dec'):
        sae.W_dec.data = vsae_model.W_dec.data.clone()
    if hasattr(vsae_model, 'W_enc'):
        sae.W_enc.data = vsae_model.W_enc.data.clone()
    if hasattr(vsae_model, 'b_enc'):
        sae.b_enc.data = vsae_model.b_enc.data.clone()
    if hasattr(vsae_model, 'b_dec'):
        sae.b_dec.data = vsae_model.b_dec.data.clone()

    sae.to(device=device, dtype=dtype)

    # Set architecture for plotting
    if config["trainer"]["trainer_class"] == "VSAETopKTrainer":
        sae.cfg.architecture = "vsae_topk"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    # Check decoder norms
    normalized = sae.check_decoder_norms()
    if not normalized:
        print("Warning: Decoder vectors are not normalized. This may affect some evaluations.")

    return sae


if __name__ == "__main__":
    # Example usage - adjust these paths to your actual model
    repo_id = "your_username/your_vsae_models"
    filename = "VSAETopK_gelu-1l_d2048_k512_lr0.0008_kl1.0_aux0.03125_fixed_var/trainer_0/ae.pt"
    layer = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    model_name = "gelu-1l"

    sae = load_dictionary_learning_vsae_topk_sae(
        repo_id,
        filename,
        model_name,
        device,  # type: ignore
        dtype,
        layer=layer,
    )
    print(f"Loaded VSAETopK SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={sae.k}")
