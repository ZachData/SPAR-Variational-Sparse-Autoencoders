import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE

# 1. Load the model using transformer_lens
model = HookedTransformer.from_pretrained("gelu-1l", device="cuda")

# 2. Set up data generator - using Neel's C4 code dataset (instead of pile-10k to match the model)
data_gen = hf_dataset_to_generator("NeelNanda/c4-code-20k", split="train")

# 3. Create activation buffer
# The hook name for the MLP output in transformer_lens
hook_name = "blocks.0.mlp.hook_post"

buffer = TransformerLensActivationBuffer(
    data=data_gen,
    model=model,
    hook_name=hook_name,
    d_submodule=model.cfg.d_mlp,  # MLP dimension from model config
    n_ctxs=3000,
    ctx_len=128,
    refresh_batch_size=32,
    out_batch_size=1024,
    device="cuda",
)

# 4. Configure trainer
trainer_config = {
    "trainer": StandardTrainer,
    "steps": 20,
    "activation_dim": model.cfg.d_mlp,  # MLP output dimension
    "dict_size": 1 * model.cfg.d_mlp,  # 4x the MLP dimension
    "layer": 0,
    "lm_name": "gelu-1l",
    "lr": 1e-3,
    "l1_penalty": 1e-3,
    "warmup_steps": 1,
    "sparsity_warmup_steps": 2,
    "resample_steps": 3,
    "device": "cuda"
}

# 5. Run training
trainSAE(
    data=buffer,
    trainer_configs=[trainer_config],
    steps=20,
    save_dir="./trained_sae",
    log_steps=10,
    verbose=True,
    normalize_activations=True  # This helps with hyperparameter transfer
)