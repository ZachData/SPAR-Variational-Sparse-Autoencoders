from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
import torch
from nnsight import LanguageModel

# Only import the basic dictionary classes at the top
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    JumpReluAutoEncoder,
)

def hf_dataset_to_generator(dataset_name, split="train", streaming=False, return_tokens=False):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            # print(x)
            if return_tokens:
                if "tokens" in x:
                    # print({k: type(v) for k, v in x.items()})
                    yield {"tokens": x["tokens"]}
                else:
                    raise KeyError("'tokens' not found.")
            else:
                if "text" in x:
                    yield x["text"]
                else:
                    raise KeyError("'text' not found. Try return_tokens=True.")

    return gen()

def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def _convert_string_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string representation of dtype back to torch.dtype."""
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    
    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.bool": torch.bool,
    }
    
    return dtype_map.get(dtype_str, torch.float32)


def _convert_string_to_device(device_str: str) -> torch.device:
    """Convert string representation of device back to torch.device."""
    if isinstance(device_str, torch.device):
        return device_str
    
    if device_str == "None" or device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.device(device_str)


def load_dictionary(base_path: str, device: str) -> tuple:
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        from .trainers.top_k import TopKConfig, AutoEncoderTopK
        
        trainer_config = config["trainer"]
        
        # Always create a proper config from the flattened trainer config
        model_config = TopKConfig(
            activation_dim=trainer_config["activation_dim"],
            dict_size=trainer_config["dict_size"],
            k=trainer_config["k"],
            device=device
        )
        
        dictionary = AutoEncoderTopK.from_pretrained(
            ae_path, 
            config=model_config,
            device=device
        )
    elif dict_class == "BatchTopKSAE":
        from .trainers.batch_top_k import BatchTopKSAE, BatchTopKConfig
        
        # Enhanced BatchTopKSAE with configuration classes
        k = config["trainer"]["k"]
        activation_dim = config["trainer"]["activation_dim"]
        dict_size = config["trainer"]["dict_size"]
        use_april_update_mode = config["trainer"].get("use_april_update_mode", True)
        
        # Create configuration object
        batch_topk_config = BatchTopKConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            use_april_update_mode=use_april_update_mode,
            device=device
        )
        
        dictionary = BatchTopKSAE.from_pretrained(
            ae_path, 
            config=batch_topk_config,
            device=device
        )
    elif dict_class == "MatryoshkaBatchTopKSAE":
        from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE, MatryoshkaConfig
        
        # Extract configuration from trainer config
        trainer_config = config["trainer"]
        
        # Create MatryoshkaConfig from saved parameters
        matryoshka_config = MatryoshkaConfig(
            activation_dim=trainer_config["activation_dim"],
            dict_size=trainer_config["dict_size"],
            k=trainer_config["k"],
            group_fractions=trainer_config.get("group_fractions", [1.0]),  # Default to single group
            group_weights=trainer_config.get("group_weights", None),
            auxk_alpha=trainer_config.get("auxk_alpha", 1/32),
            threshold_beta=trainer_config.get("threshold_beta", 0.999),
            threshold_start_step=trainer_config.get("threshold_start_step", 1000),
            dead_feature_threshold=trainer_config.get("dead_feature_threshold", 10_000_000),
            top_k_aux_fraction=trainer_config.get("top_k_aux_fraction", 0.5),
            device=device
        )
        
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(
            ae_path, 
            config=matryoshka_config,
            device=device
        )
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "JumpReluSAE":
        from .trainers.jumprelu import JumpReluSAE
        
        # Get parameters from config
        bandwidth = config["trainer"].get("bandwidth", 0.001)
        threshold_init = config["trainer"].get("threshold_init", 0.001)
        apply_b_dec_to_input = config["trainer"].get("apply_b_dec_to_input", False)
        dictionary = JumpReluSAE.from_pretrained(
            ae_path, 
            bandwidth=bandwidth,
            threshold_init=threshold_init,
            apply_b_dec_to_input=apply_b_dec_to_input,
            device=device
        )
    elif dict_class == "VSAEIsoGaussian":
        from .trainers.vsae_iso import VSAEIsoGaussian
        dictionary = VSAEIsoGaussian.from_pretrained(ae_path, device=device)
    elif dict_class == "MatryoshkaVSAEIso":
        from .trainers.vsae_matryoshka import MatryoshkaVSAEConfig, MatryoshkaVSAEIso
        
        trainer_config = config["trainer"]
        activation_dim = trainer_config.get("activation_dim")
        dict_size = trainer_config.get("dict_size")
        
        # If dimensions not in config, detect from state dict
        if activation_dim is None or dict_size is None:
            try:
                state_dict = torch.load(ae_path, map_location=device, weights_only=False)
                if 'encoder.weight' in state_dict:
                    dict_size, activation_dim = state_dict["encoder.weight"].shape
                elif 'W_enc' in state_dict:
                    activation_dim, dict_size = state_dict["W_enc"].shape
                else:
                    raise ValueError(f"Could not determine dimensions from state dict")
            except Exception as e:
                raise ValueError(f"Could not load state dict: {e}")
        
        # Extract parameters with defaults
        k = trainer_config.get("k", 64)
        group_fractions = trainer_config.get("group_fractions", [0.25, 0.25, 0.25, 0.25])
        group_weights = trainer_config.get("group_weights", None)
        var_flag = trainer_config.get("var_flag", 0)
        use_april_update_mode = trainer_config.get("use_april_update_mode", True)
        log_var_init = trainer_config.get("log_var_init", -2.0)
        use_batch_topk = trainer_config.get("use_batch_topk", True)
        topk_mode = trainer_config.get("topk_mode", "magnitude")
        
        # Create the VSAE config
        vsae_config = MatryoshkaVSAEConfig(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            group_fractions=group_fractions,
            group_weights=group_weights,
            var_flag=var_flag,
            use_april_update_mode=use_april_update_mode,
            log_var_init=log_var_init,
            use_batch_topk=use_batch_topk,
            topk_mode=topk_mode,
            device=device
        )
        
        dictionary = MatryoshkaVSAEIso.from_pretrained(
            ae_path, config=vsae_config, device=device
        )
    elif dict_class == "VSAEPAnneal":
        from .trainers.vsae_panneal import VSAEPAnneal
        
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEPAnneal.from_pretrained(
            ae_path, 
            var_flag=var_flag,
            device=device
        )
    elif dict_class == "VSAEMultiGaussian":
        from .trainers.vsae_multi import VSAEMultiGaussian
        
        # Add parameters if they're in the config
        corr_rate = config["trainer"].get("corr_rate", 0.5)
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEMultiGaussian.from_pretrained(ae_path, device=device, corr_rate=corr_rate, var_flag=var_flag)
    elif dict_class == "VSAEMixtureGaussian":
        from .trainers.vsae_mixture import VSAEMixtureGaussian
        
        # Add parameters if they're in the config
        var_flag = config["trainer"].get("var_flag", 0)
        n_correlated_pairs = config["trainer"].get("n_correlated_pairs", 0)
        n_anticorrelated_pairs = config["trainer"].get("n_anticorrelated_pairs", 0)
        dictionary = VSAEMixtureGaussian.from_pretrained(
            ae_path, 
            device=device, 
            var_flag=var_flag,
            n_correlated_pairs=n_correlated_pairs,
            n_anticorrelated_pairs=n_anticorrelated_pairs
        )
    elif dict_class == "VSAEBatchTopK":
        from .trainers.vsae_batch_topk import VSAEBatchTopK
        
        # New improved VSAEBatchTopK model
        k = config["trainer"]["k"]
        var_flag = config["trainer"].get("var_flag", 0)
        constrain_decoder = config["trainer"].get("constrain_decoder", True)
        
        dictionary = VSAEBatchTopK.from_pretrained(
            ae_path,
            k=k,
            var_flag=var_flag,
            constrain_decoder=constrain_decoder,
            device=device
        )
    elif dict_class == "VSAEGated":
        from .trainers.vsae_gated import VSAEGated, VSAEGatedConfig
        
        # Enhanced loading for robust VSAEGated implementation
        try:
            trainer_config = config["trainer"]
            activation_dim = trainer_config.get("activation_dim")
            dict_size = trainer_config.get("dict_size")
            
            # If dimensions not in config, detect from state dict
            if activation_dim is None or dict_size is None:
                state_dict = torch.load(ae_path, map_location=device, weights_only=False)
                if 'encoder.weight' in state_dict:
                    dict_size, activation_dim = state_dict["encoder.weight"].shape
                elif 'W_enc' in state_dict:
                    activation_dim, dict_size = state_dict["W_enc"].shape
            
            vsae_config = VSAEGatedConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                var_flag=trainer_config.get("var_flag", 1),
                use_april_update_mode=trainer_config.get("use_april_update_mode", True),
                log_var_init=trainer_config.get("log_var_init", -2.0),
                device=device
            )
            
            dictionary = VSAEGated.from_pretrained(
                ae_path, config=vsae_config, device=device, strict_loading=False
            )
        except Exception as e:
            # Fallback to simple loading
            var_flag = config["trainer"].get("var_flag", 1)
            dictionary = VSAEGated.from_pretrained(ae_path, var_flag=var_flag, device=device)
    elif dict_class == "VSAEGatedAutoEncoder":
        from .trainers.vsae_gated import VSAEGated
        
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEGated.from_pretrained(
            ae_path, 
            config=None,  # Will auto-detect from state dict
            dtype=torch.float32,
            device=device,
            normalize_decoder=True,
            var_flag=var_flag
        )
    elif dict_class == "VSAEJumpReLU":
        from .trainers.vsae_jump_relu import VSAEJumpReLU
        
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 0)
        bandwidth = config["trainer"].get("bandwidth", 0.001)
        dictionary = VSAEJumpReLU.from_pretrained(
            ae_path, 
            var_flag=var_flag,
            bandwidth=bandwidth,
            device=device
        )
    elif dict_class == "VSAETopK":
        from .trainers.vsae_topk import VSAETopK
        
        k = config["trainer"]["k"]
        dictionary = VSAETopK.from_pretrained(ae_path, device=device)
    elif dict_class == "VSAEPriorsGaussian":
        from .trainers.vsae_priors import VSAEPriorsGaussian, VSAEPriorsConfig
        
        # VSAEPriors has complex configuration, so we'll reconstruct it from the trainer config
        trainer_config = config["trainer"]
        
        # Create VSAEPriorsConfig from saved parameters
        vsae_config = VSAEPriorsConfig(
            activation_dim=trainer_config["activation_dim"],
            dict_size=trainer_config["dict_size"],
            prior_types=trainer_config.get("prior_types", ["gaussian"]),
            prior_assignment_strategy=trainer_config.get("prior_assignment_strategy", "single"),
            prior_proportions=trainer_config.get("prior_proportions", None),
            prior_params=trainer_config.get("prior_params", {}),
            var_flag=trainer_config.get("var_flag", 0),
            use_april_update_mode=trainer_config.get("use_april_update_mode", True),
            log_var_init=trainer_config.get("log_var_init", -2.0),
            device=device
        )
        
        dictionary = VSAEPriorsGaussian.from_pretrained(
            ae_path, 
            config=vsae_config,
            device=device
        )
    elif dict_class == "LaplacianTopK":
        # Enhanced LaplacianTopK with configuration classes  
        try:
            from .trainers.laplace_topk import LaplacianTopK, LaplacianTopKConfig
            
            trainer_config = config["trainer"]
            activation_dim = trainer_config.get("activation_dim")
            dict_size = trainer_config.get("dict_size")
            k = trainer_config.get("k")
            
            # If dimensions not in config, detect from state dict
            if activation_dim is None or dict_size is None:
                state_dict = torch.load(ae_path, map_location=device, weights_only=False)
                if 'encoder.weight' in state_dict:
                    dict_size, activation_dim = state_dict["encoder.weight"].shape
                elif 'W_enc' in state_dict:
                    activation_dim, dict_size = state_dict["W_enc"].shape
            
            # Get k value
            if k is None:
                state_dict = torch.load(ae_path, map_location=device, weights_only=False)
                k = state_dict["k"].item() if "k" in state_dict else max(1, dict_size // 10)
            
            laplace_config = LaplacianTopKConfig(
                activation_dim=activation_dim,
                dict_size=dict_size,
                k=k,
                block_sizes=trainer_config.get("block_sizes"),
                laplacian_type=trainer_config.get("laplacian_type", "chain"),
                var_flag=trainer_config.get("var_flag", 0),
                use_april_update_mode=trainer_config.get("use_april_update_mode", True),
                log_var_init=trainer_config.get("log_var_init", -2.0),
                device=device
            )
            
            dictionary = LaplacianTopK.from_pretrained(
                ae_path, config=laplace_config, device=device, strict_loading=False
            )
        except Exception as e:
            # Fallback to simple loading if config approach fails
            dictionary = LaplacianTopK.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config


def get_submodule(model: LanguageModel, layer: int):
    """Gets the residual stream submodule"""
    model_name = model._model_key

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")