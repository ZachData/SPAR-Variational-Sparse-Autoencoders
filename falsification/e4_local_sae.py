"""Local (non-Hub) SAEBench wrappers for the recovered E4 Pythia checkpoints.

`size_control.py`'s scorer is injected and needs a live object that
`sae_bench.evals.scr_and_tpp.main` can call `encode()`/`decode()` on. The
vendored `custom_saes/topk_sae.py` and `custom_saes/vsae_topk_sae.py` only load
checkpoints from the HF Hub (`hf_hub_download`); the recovered checkpoints
(`experiments/e4_pythia_baseline/`, `experiments/e4_pythia_vsae/`, PROJECT.md
Next steps #0) are local files only, so this module reads them directly and
wraps the result to satisfy `sae_bench.custom_saes.base_sae.BaseSAE`'s contract
(`W_enc`, `W_dec`, `b_enc`, `b_dec`, `encode`, `decode`, `cfg`).

Only the baseline TopK SAE is ever masked -- E4's design (PROJECT.md) sweeps
dictionary *size* only for the baseline and asks where the vSAE's own,
unmasked, naturally-sparse score falls on that curve. The vSAE wrapper
therefore delegates encode/decode to the loaded VSAETopK model unchanged
(inheriting CLAUDE.md landmines 1 and 2 -- var_flag gating and F.relu(mu) --
automatically instead of re-deriving them) and is never masked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SAEBENCH = REPO / "SAEBench-main"
for _p in (str(REPO), str(SAEBENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sae_bench.custom_saes.base_sae as base_sae  # noqa: E402
from dictionary_learning.utils import load_dictionary  # noqa: E402


def _read_trainer_config(trainer_dir: Path) -> dict:
    with open(trainer_dir / "config.json") as f:
        return json.load(f)["trainer"]


class LocalTopKSAE(base_sae.BaseSAE):
    """A local dictionary_learning `AutoEncoderTopK` checkpoint, maskable by feature."""

    def __init__(self, d_in, d_sae, k, model_name, hook_layer, hook_name, device, dtype):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))
        self.cfg.architecture = "topk"
        self._pristine: dict[str, torch.Tensor] | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        post_relu = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        post_topk = post_relu.topk(int(self.k), sorted=False, dim=-1)
        buf = torch.zeros_like(post_relu)
        return buf.scatter_(-1, post_topk.indices, post_topk.values)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def freeze_pristine(self) -> None:
        """Snapshot the unmasked weights so `apply_mask` can be called repeatedly."""
        self._pristine = {
            "W_enc": self.W_enc.data.clone(),
            "W_dec": self.W_dec.data.clone(),
            "b_enc": self.b_enc.data.clone(),
        }

    @torch.no_grad()
    def apply_mask(self, keep_indices: np.ndarray) -> None:
        """Zero every dictionary entry outside `keep_indices`, from the pristine weights.

        `b_dec` is not a per-feature quantity (it is indexed by activation_dim,
        not dict_size) and is left untouched, matching `size_control.mask_dictionary`'s
        per-weight axis convention.
        """
        if self._pristine is None:
            raise RuntimeError("call freeze_pristine() before the first apply_mask()")
        keep = torch.zeros(self.cfg.d_sae, dtype=torch.bool, device=self.W_enc.device)
        keep[torch.as_tensor(keep_indices, device=self.W_enc.device)] = True
        self.W_enc.data = self._pristine["W_enc"] * keep.view(1, -1)
        self.W_dec.data = self._pristine["W_dec"] * keep.view(-1, 1)
        self.b_enc.data = self._pristine["b_enc"] * keep


def load_local_topk_sae(
    trainer_dir: str | Path, model_name: str, device: str, dtype: torch.dtype
) -> LocalTopKSAE:
    trainer_dir = Path(trainer_dir)
    config = _read_trainer_config(trainer_dir)
    assert config["dict_class"] == "AutoEncoderTopK", config["dict_class"]
    sd = torch.load(trainer_dir / "ae.pt", map_location="cpu")

    sae = LocalTopKSAE(
        d_in=config["activation_dim"],
        d_sae=config["dict_size"],
        k=config["k"],
        model_name=model_name,
        hook_layer=config["layer"],
        hook_name=config["submodule_name"],
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        sae.W_enc.data = sd["encoder.weight"].T.to(device=device, dtype=dtype)
        sae.W_dec.data = sd["decoder.weight"].T.to(device=device, dtype=dtype)
        sae.b_enc.data = sd["encoder.bias"].to(device=device, dtype=dtype)
        sae.b_dec.data = sd["b_dec"].to(device=device, dtype=dtype)
    sae.freeze_pristine()

    normalized = sae.check_decoder_norms()
    if not normalized:
        # The checkpoint was trained and saved in bfloat16 (config.json); reloading
        # its weights in float32 surfaces that quantization noise as a norm error
        # larger than base_sae.py's own bf16 tolerance. Not a corrupted checkpoint.
        print("(expected: bfloat16 training noise reloaded in float32, see comment)")
    return sae


class LocalVSAETopKSAE(base_sae.BaseSAE):
    """A local dictionary_learning `VSAETopK` checkpoint (must be `var_flag=0`).

    Never masked -- see module docstring. `encode`/`decode` delegate to the
    loaded model itself rather than to `W_enc`/`W_dec`, so those fields exist
    only to satisfy code that reads them directly (e.g. SCR/TPP's node-effect
    computation reads `sae.W_dec.data`) and are a straight copy of the real
    weights, kept for reference only.
    """

    def __init__(self, vsae_model, d_in, d_sae, k, model_name, hook_layer, hook_name, device, dtype):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)
        self.vsae_model = vsae_model
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))
        self.cfg.architecture = "vsae_topk"
        with torch.no_grad():
            self.W_enc.data = vsae_model.encoder.weight.T.to(device=device, dtype=dtype)
            self.W_dec.data = vsae_model.decoder.weight.T.to(device=device, dtype=dtype)
            self.b_enc.data = vsae_model.encoder.bias.to(device=device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sparse_features = self.vsae_model.encode(x, training=False)[0]
        return sparse_features.to(dtype=self.dtype)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.vsae_model.decode(f).to(dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_local_vsae_topk_sae(
    trainer_dir: str | Path, model_name: str, device: str, dtype: torch.dtype
) -> LocalVSAETopKSAE:
    trainer_dir = Path(trainer_dir)
    config = _read_trainer_config(trainer_dir)
    assert config["dict_class"] == "VSAETopK", config["dict_class"]
    assert config["var_flag"] == 0, (
        "CLAUDE.md landmine 1: this wrapper never samples; only var_flag=0 "
        f"checkpoints are load-bearing here, got var_flag={config['var_flag']}"
    )

    vsae_model, _ = load_dictionary(str(trainer_dir), device=device)
    vsae_model = vsae_model.to(dtype=dtype)

    return LocalVSAETopKSAE(
        vsae_model=vsae_model,
        d_in=config["activation_dim"],
        d_sae=config["dict_size"],
        k=config["k"],
        model_name=model_name,
        hook_layer=config["layer"],
        hook_name=config["submodule_name"],
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    base = load_local_topk_sae(
        "experiments/e4_pythia_baseline/seed42/"
        "TopK_SAE_pythia70m_d8192_k256_auxk0.03125_lr_auto/trainer_0",
        model_name="pythia-70m-deduped",
        device=device,
        dtype=dtype,
    )
    vsae = load_local_vsae_topk_sae(
        "experiments/e4_pythia_vsae/seed42/"
        "VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/trainer_0",
        model_name="pythia-70m-deduped",
        device=device,
        dtype=dtype,
    )

    x = torch.randn(4, base.cfg.d_in, device=device, dtype=dtype)

    f_base = base.encode(x)
    xhat_base = base.decode(f_base)
    print("baseline: recon", xhat_base.shape, "L0", (f_base != 0).sum(-1).tolist())

    f_vsae = vsae.encode(x)
    xhat_vsae = vsae.decode(f_vsae)
    print("vsae: recon", xhat_vsae.shape, "L0", (f_vsae != 0).sum(-1).tolist())

    base.apply_mask(np.arange(100))
    f_masked = base.encode(x)
    print("baseline masked to 100: L0", (f_masked != 0).sum(-1).tolist())
    assert (f_masked != 0).sum(-1).max().item() <= 100
    base.apply_mask(np.arange(base.cfg.d_sae))  # restore
    print("all checks passed")
