from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from nnsight import LanguageModel

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .trainers.vsae_iso import VSAEIsoGaussian
from .trainers.vsae_multi import VSAEMultiGaussian
from .trainers.vsae_mixture import VSAEMixtureGaussian
from .trainers.vsae_batch_topk import VSAEBatchTopK
from .trainers.vsae_gated import VSAEGated
from .trainers.vsae_gated_anneal import VSAEGatedAutoEncoder
from .trainers.vsae_jump_relu import VSAEJumpReLU
from .trainers.vsae_matryoshka import MatryoshkaVSAEIso
from .trainers.vsae_panneal import VSAEPAnneal
from .trainers.vsae_topk import VSAETopK
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoder,
    JumpReluAutoEncoder,
)

def hf_dataset_to_generator(dataset_name, split="train", streaming=False, return_tokens=False):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
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
    elif dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "VSAEIsoGaussian":
        dictionary = VSAEIsoGaussian.from_pretrained(ae_path, device=device)
    elif dict_class == "VSAEMultiGaussian":
        # Add parameters if they're in the config
        corr_rate = config["trainer"].get("corr_rate", 0.5)
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEMultiGaussian.from_pretrained(ae_path, device=device, corr_rate=corr_rate, var_flag=var_flag)
    elif dict_class == "VSAEMixtureGaussian":
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
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 1)
        dictionary = VSAEGated.from_pretrained(
            ae_path, 
            var_flag=var_flag,
            device=device
        )
    elif dict_class == "VSAEGatedAutoEncoder":
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEGatedAutoEncoder.from_pretrained(
            ae_path, 
            var_flag=var_flag,
            device=device
        )

    elif dict_class == "VSAEJumpReLU":
        dictionary = VSAEJumpReLU.from_pretrained(
            ae_path,
            device=device
        )

    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)

    elif dict_class == "VSAEPAnneal":
        # Get parameters from config
        var_flag = config["trainer"].get("var_flag", 0)
        dictionary = VSAEPAnneal.from_pretrained(
            ae_path, 
            var_flag=var_flag,
            device=device
        )
    elif dict_class == "VSAETopK":
        k = config["trainer"]["k"]
        dictionary = VSAETopK.from_pretrained(ae_path, device=device)   
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