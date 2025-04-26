import torch as t
from transformer_lens import HookedTransformer
from dictionary_learning.trainers.vsae_iso import VSAEIsoTrainer, VSAEIsoGaussian
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE

# 1. Load the model using transformer_lens
model = HookedTransformer.from_pretrained("gelu-1l", device="cuda")

# 2. Set up data generator - using Neel's C4 code dataset to match the model
data_gen = hf_dataset_to_generator("NeelNanda/c4-tokenized-2b", split="train", return_tokens=True)

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
    "trainer": VSAEIsoTrainer,
    "steps": 20000,
    "activation_dim": model.cfg.d_mlp,  # MLP output dimension
    "dict_size":    8*model.cfg.d_mlp,  # Dictionary size
    "layer": 0,
    "lm_name": "gelu-1l",
    "lr": 5e-5,  # Learning rate recommended in April update
    "kl_coeff": 5.0,  # KL coefficient (equivalent to l1_penalty in standard SAE)
    "warmup_steps": 1000,
    "sparsity_warmup_steps": 1000,  # 5% of total steps
    "decay_start": 16000,  # 80% of total steps
    "var_flag": 0,  # Use fixed variance (set to 1 for learned variance)
    "use_april_update_mode": True,  # Use April update improvements
    "device": "cuda",
    "wandb_name": "VSAE_Iso_GELU1L",
    "dict_class": VSAEIsoGaussian  # Specify the dictionary class
}
print('\n'.join(f"{k}: {v}" for k, v in trainer_config.items()))

# 5. Run training
trainSAE(
    data=buffer,
    trainer_configs=[trainer_config],
    steps=20000,
    save_dir="./trained_vsae_iso",
    save_steps=[10000, 20000],  # Save checkpoints at these steps
    log_steps=100,
    verbose=True,
    normalize_activations=True,  # Help with hyperparameter transfer
    autocast_dtype=t.bfloat16,  # Use bfloat16 for faster training
)

# Optional: Code to evaluate the trained model
from dictionary_learning.utils import load_dictionary
from dictionary_learning.evaluation import evaluate

# Load the trained dictionary
vsae, config = load_dictionary("./trained_vsae_iso/trainer_0", device="cuda")

# Evaluate on a small batch
eval_results = evaluate(
    dictionary=vsae,
    activations=buffer,
    batch_size=64,
    max_len=128,
    device="cuda",
    n_batches=10
)

# Print metrics
print("Evaluation results:")
for metric, value in eval_results.items():
    print(f"{metric}: {value:.4f}")