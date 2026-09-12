"""
Training script for VSAEJumpReLU, mirroring train_vsae_topk.py's structure so
this arm family is directly comparable to A2 (TopK) and A3 (BatchTopK).

Written for Next steps A's discreteness companion (PROJECT.md): TopK and
BatchTopK's hard top-k selection both show sampling-induced FVE damage tracking
selection-Jaccard instability almost exactly (r=+0.9993, r=+0.9979). JumpReLU's
per-feature learned threshold is a genuinely different sparsity mechanism --
smooth in the sense that each feature's gate is its own scalar comparison, not
a joint top-k selection across the dictionary -- so it is the sharper test of
whether that coupling is about *discreteness* per se or about *hard top-k*
specifically.

No training script existed for this trainer before this one (CLAUDE.md flagged
it as "never exercised end to end"). Two real bugs turned up in
dictionary_learning/trainers/vsae_jump_relu.py while building this and are now
fixed there (see the docstring on VSAEJumpReLU.jump_relu and CLAUDE.md):
  1. `threshold` received zero gradient -- `jump_relu()` used a plain
     `(x > threshold).float()` comparison instead of the straight-through
     estimator (`JumpReLUFunction`) this repo's own non-variational
     `jumprelu.py` already defines for exactly this reason. Every prior
     "VSAEJumpReLU" would have trained as a frozen-threshold ReLU-SAE.
  2. There was no L0-target sparsity loss, so even with a working gradient
     nothing was driving the threshold toward a target sparsity. Added,
     mirroring the non-variational JumpReluTrainer's target_l0/sparsity_penalty
     term via the same StepFunction STE.

Unlike TopK/BatchTopK's hard k, target_l0 is a SOFT target reached by gradient
descent on a squared-relative-error penalty -- there is no architectural
guarantee the achieved L0 matches target_l0. That is a new researcher degree
of freedom this arm family has that A2/A3 did not: sparsity_penalty's scale
needs to actually pull L0 to the target within the training budget, not just
exist. Check achieved l0 against target_l0 in RUN_COMPLETE.json before trusting
any FVE/Jaccard comparison against A2/A3's hard-k arms.
"""

import torch
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import multiprocessing

from transformer_lens import HookedTransformer
from dictionary_learning.buffer import TransformerLensActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.evaluation import evaluate

from dictionary_learning.trainers.vsae_jump_relu import (
    VSAEJumpReLU,
    VSAEJumpReLUTrainer,
    VSAEJumpReLUConfig,
    VSAEJumpReLUTrainingConfig,
)


@dataclass
class ExperimentConfig:
    """Configuration for the VSAEJumpReLU experiment."""
    # Model configuration -- defaults match the gelu-1l residual-stream point
    # every other arm in the confirmatory battery uses (run_arm.py's BASE).
    model_name: str = "gelu-1l"
    layer: int = 0
    hook_name: str = "blocks.0.hook_resid_post"
    dict_size_multiple: float = 4.0

    # JumpReLU-specific configuration. target_l0_fraction mirrors TopK's
    # k_fraction / BatchTopK's k_ratio so the same "12.5% of dict_size active"
    # operating point is directly comparable across all three arm families.
    threshold_init: float = 0.001
    bandwidth: float = 0.001
    target_l0_fraction: float = 0.125
    sparsity_penalty: float = 1.0

    # Model-specific config (matches VSAETopKConfig's fields where they apply)
    var_flag: int = 0  # 0: fixed variance, 1: learned variance
    use_april_update_mode: bool = True
    log_var_init: float = -2.0

    # Training configuration
    total_steps: int = 10000
    lr: float = 5e-4
    kl_coeff: float = 500.0
    aux_weight: float = 0.1

    # Schedule configuration
    warmup_steps: Optional[int] = None
    sparsity_warmup_steps: Optional[int] = None
    kl_warmup_steps: Optional[int] = None
    decay_start_step: Optional[int] = None

    # Buffer configuration
    n_ctxs: int = 3000
    ctx_len: int = 128
    refresh_batch_size: int = 32
    out_batch_size: int = 1024

    # Logging and saving
    checkpoint_steps: tuple = (5000, 10000)
    log_steps: int = 100
    save_dir: str = "./experiments"

    # WandB configuration
    use_wandb: bool = True
    wandb_entity: str = "zachdata"
    wandb_project: str = "vsae-jumprelu-experiments"

    # System configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    autocast_dtype: str = "bfloat16"
    seed: Optional[int] = 42

    # Evaluation configuration
    eval_batch_size: int = 64
    eval_n_batches: int = 10

    def __post_init__(self):
        if self.warmup_steps is None:
            self.warmup_steps = max(200, int(0.02 * self.total_steps))
        if self.sparsity_warmup_steps is None:
            self.sparsity_warmup_steps = int(0.05 * self.total_steps)
        if self.kl_warmup_steps is None:
            self.kl_warmup_steps = int(0.1 * self.total_steps)
        if self.decay_start_step is None:
            decay_start = int(0.8 * self.total_steps)
            min_decay_start = max(self.warmup_steps, self.sparsity_warmup_steps) + 1
            self.decay_start_step = max(decay_start, min_decay_start)

    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return dtype_map[self.dtype]

    def get_autocast_dtype(self) -> torch.dtype:
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return dtype_map[self.autocast_dtype]

    def get_device(self) -> torch.device:
        return torch.device(self.device)


class ExperimentRunner:
    """Manages the VSAEJumpReLU training experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._dict_size: Optional[int] = None
        self.setup_logging()
        self.setup_reproducibility()

    def setup_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'vsae_jumprelu_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_reproducibility(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.logger.info(f"Set random seed to {self.config.seed}")

    def load_model(self) -> HookedTransformer:
        self.logger.info(f"Loading model: {self.config.model_name}")
        model = HookedTransformer.from_pretrained(self.config.model_name, device=self.config.device)
        self.logger.info(f"Model loaded. d_model: {model.cfg.d_model}")
        return model

    def create_buffer(self, model: HookedTransformer) -> TransformerLensActivationBuffer:
        self.logger.info("Setting up data generator and activation buffer")
        data_gen = hf_dataset_to_generator("NeelNanda/c4-code-tokenized-2b", split="train", return_tokens=True)
        buffer = TransformerLensActivationBuffer(
            data=data_gen,
            model=model,
            hook_name=self.config.hook_name,
            d_submodule=model.cfg.d_model,
            n_ctxs=self.config.n_ctxs,
            ctx_len=self.config.ctx_len,
            refresh_batch_size=self.config.refresh_batch_size,
            out_batch_size=self.config.out_batch_size,
            device=self.config.device,
        )
        return buffer

    def create_model_config(self, model: HookedTransformer) -> VSAEJumpReLUConfig:
        dict_size = int(self.config.dict_size_multiple * model.cfg.d_model)
        self._dict_size = dict_size
        return VSAEJumpReLUConfig(
            activation_dim=model.cfg.d_model,
            dict_size=dict_size,
            threshold=self.config.threshold_init,
            bandwidth=self.config.bandwidth,
            var_flag=self.config.var_flag,
            use_april_update_mode=self.config.use_april_update_mode,
            log_var_init=self.config.log_var_init,
            dtype=self.config.get_torch_dtype(),
            device=self.config.get_device(),
        )

    def create_training_config(self) -> VSAEJumpReLUTrainingConfig:
        assert self._dict_size is not None, "create_model_config must run first"
        target_l0 = self.config.target_l0_fraction * self._dict_size
        return VSAEJumpReLUTrainingConfig(
            steps=self.config.total_steps,
            lr=self.config.lr,
            kl_coeff=self.config.kl_coeff,
            kl_warmup_steps=self.config.kl_warmup_steps,
            aux_weight=self.config.aux_weight,
            sparsity_penalty=self.config.sparsity_penalty,
            target_l0=target_l0,
            warmup_steps=self.config.warmup_steps,
            sparsity_warmup_steps=self.config.sparsity_warmup_steps,
            decay_start=self.config.decay_start_step,
        )

    def create_trainer_config(self, model_config: VSAEJumpReLUConfig, training_config: VSAEJumpReLUTrainingConfig) -> Dict[str, Any]:
        return {
            "trainer": VSAEJumpReLUTrainer,
            "model_config": model_config,
            "training_config": training_config,
            "layer": self.config.layer,
            "lm_name": self.config.model_name,
            "wandb_name": self.get_experiment_name(),
            "submodule_name": self.config.hook_name,
            "seed": self.config.seed,
        }

    def get_experiment_name(self) -> str:
        var_suffix = "_learned_var" if self.config.var_flag == 1 else "_fixed_var"
        dict_size = self._dict_size or int(self.config.dict_size_multiple * 512)
        target_l0 = int(self.config.target_l0_fraction * dict_size)
        return (
            f"VSAEJumpReLU_{self.config.model_name}_"
            f"d{dict_size}_l0-{target_l0}_lr{self.config.lr}_"
            f"kl{self.config.kl_coeff}{var_suffix}"
        )

    def get_save_directory(self) -> Path:
        save_dir = Path(self.config.save_dir) / self.get_experiment_name()
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def save_config(self, save_dir: Path) -> None:
        import json
        config_path = save_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        self.logger.info(f"Saved experiment config to {config_path}")

    def run_training(self) -> Optional[Dict[str, float]]:
        start_time = time.time()
        try:
            model = self.load_model()
            buffer = self.create_buffer(model)
            model_config = self.create_model_config(model)
            training_config = self.create_training_config()
            trainer_config = self.create_trainer_config(model_config, training_config)

            save_dir = self.get_save_directory()
            self.save_config(save_dir)

            self.logger.info(f"Experiment: {self.get_experiment_name()}")
            self.logger.info(f"Dictionary size: {model_config.dict_size}")
            self.logger.info(f"Target L0: {training_config.target_l0}")
            self.logger.info(f"Save directory: {save_dir}")

            self.logger.info("Starting VSAEJumpReLU training...")
            trainSAE(
                data=buffer,
                trainer_configs=[trainer_config],
                steps=self.config.total_steps,
                save_dir=str(save_dir),
                save_steps=list(self.config.checkpoint_steps) if self.config.checkpoint_steps else None,
                log_steps=self.config.log_steps,
                verbose=True,
                normalize_activations=True,
                autocast_dtype=self.config.get_autocast_dtype(),
                use_wandb=self.config.use_wandb,
                wandb_entity=self.config.wandb_entity,
                wandb_project=self.config.wandb_project,
                run_cfg={
                    "model_type": self.config.model_name,
                    "experiment_type": "vsae_jumprelu",
                    "dict_size_multiple": self.config.dict_size_multiple,
                    "target_l0_fraction": self.config.target_l0_fraction,
                    "var_flag": self.config.var_flag,
                    **asdict(self.config),
                }
            )

            eval_results = self.evaluate_model(save_dir, buffer)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
            return eval_results

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate_model(self, save_dir: Path, buffer: TransformerLensActivationBuffer) -> Dict[str, float]:
        self.logger.info("Evaluating trained model...")
        try:
            from dictionary_learning.utils import load_dictionary

            model_path = save_dir / "trainer_0"
            vsae, config = load_dictionary(str(model_path), device=self.config.device)

            eval_results = evaluate(
                dictionary=vsae,
                activations=buffer,
                batch_size=self.config.eval_batch_size,
                max_len=self.config.ctx_len,
                device=self.config.device,
                n_batches=self.config.eval_n_batches,
            )

            self.logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                if not torch.isnan(torch.tensor(value)) and not torch.isinf(torch.tensor(value)):
                    self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.warning(f"  {metric}: {value} (invalid)")

            import json
            eval_path = save_dir / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json_results = {}
                for k, v in eval_results.items():
                    if torch.is_tensor(v):
                        json_results[k] = v.item()
                    else:
                        json_results[k] = float(v) if isinstance(v, (int, float)) else str(v)
                json.dump(json_results, f, indent=2)

            return eval_results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {}


def create_quick_test_config() -> ExperimentConfig:
    return ExperimentConfig(
        total_steps=1000,
        checkpoint_steps=(),
        log_steps=50,
        n_ctxs=500,
        refresh_batch_size=16,
        out_batch_size=128,
        eval_batch_size=32,
        eval_n_batches=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_wandb=False,
        seed=42,
    )


def create_full_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="gelu-1l",
        layer=0,
        hook_name="blocks.0.hook_resid_post",
        dict_size_multiple=4.0,
        target_l0_fraction=0.125,
        total_steps=10000,
        lr=8e-4,
        kl_coeff=0.0,
        var_flag=0,
        n_ctxs=2500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=192,
        checkpoint_steps=(10000,),
        log_steps=100,
        eval_batch_size=2,
        eval_n_batches=48,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        autocast_dtype="bfloat16",
        use_wandb=False,
        seed=42,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VSAEJumpReLU Training")
    parser.add_argument("--config", choices=["quick_test", "full"], default="quick_test")
    args = parser.parse_args()

    config_functions = {"quick_test": create_quick_test_config, "full": create_full_config}
    config = config_functions[args.config]()
    runner = ExperimentRunner(config)
    results = runner.run_training()

    if results:
        key_metrics = ["frac_variance_explained", "l0", "frac_alive", "cossim"]
        print(f"{'Metric':<25} | Value")
        print("-" * 35)
        for metric in key_metrics:
            if metric in results:
                print(f"{metric:<25} | {results[metric]:.4f}")
    else:
        print("No evaluation results available")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
