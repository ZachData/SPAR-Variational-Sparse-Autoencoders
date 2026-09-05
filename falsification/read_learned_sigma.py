"""Read the learned posterior sigma out of the var_flag=1 checkpoints.

PROJECT.md specified this diagnostic for E2 and it was never run: *if the learned
sigma collapses toward 0, the degeneracy is the optimum rather than an
implementation accident.* The var_encoder weights have been sitting in the
`e2_confirm` checkpoints since they were trained.

Three things make the reading non-obvious, and all three are reported:

* **`log_var` is clamped to [-6, 2]** in both `reparameterize` and
  `_compute_kl_loss` (vsae_topk.py:333, :856). Sigma therefore cannot go below
  exp(-3) = 0.0498 no matter what the encoder wants. A raw `log_var` sitting at or
  under the floor is the collapse signal; sigma itself can only ever *approach* it.
  Both the raw and the clamped quantity are printed.
* **Sigma alone says nothing about whether the noise matters.** What matters is
  sigma against the mu it perturbs, so the ratio is computed on the features TopK
  actually selects, where the noise can change the selection.
* **The KL pushes sigma UP, not down.** For N(mu, sigma^2) against N(0, I) the KL
  is minimised in sigma at sigma = 1, so reconstruction and the KL pull in
  opposite directions. `e2_sampling_only` (beta = 0) removes the upward pull
  entirely and is the control that isolates what reconstruction alone wants.

Every checkpoint is scored on the SAME batch of activations, drawn once, so
cross-arm differences cannot come from the data.

    python falsification/read_learned_sigma.py
    python falsification/read_learned_sigma.py --n-tokens 20000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The clamp in vsae_topk.py:333 and :856. Not a parameter -- read from the code.
LOG_VAR_MIN, LOG_VAR_MAX = -6.0, 2.0
ARMS = ("e2_confirm", "e2_sampling_only")


def checkpoints(arm: str) -> list[tuple[int, Path]]:
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*/trainer_0"):
        if (p / "ae.pt").exists():
            found[int(re.search(r"seed(\d+)", str(p)).group(1))] = p
    return [(s, found[s]) for s in sorted(found)]


def draw_activations(n_tokens: int, device: str):
    """One fixed batch of gelu-1l activations, shared by every checkpoint."""
    import torch
    from transformer_lens import HookedTransformer

    from dictionary_learning.buffer import TransformerLensActivationBuffer
    from dictionary_learning.utils import hf_dataset_to_generator

    model = HookedTransformer.from_pretrained("gelu-1l", device=device)
    buffer = TransformerLensActivationBuffer(
        data=hf_dataset_to_generator(
            "NeelNanda/c4-code-tokenized-2b", split="train", return_tokens=True
        ),
        model=model,
        hook_name="blocks.0.hook_resid_post",
        d_submodule=512,
        n_ctxs=500,
        ctx_len=128,
        refresh_batch_size=12,
        out_batch_size=2048,
        device=device,
    )
    chunks, got = [], 0
    while got < n_tokens:
        batch = next(buffer)
        chunks.append(batch.to(device))
        got += batch.shape[0]
    del model, buffer
    torch.cuda.empty_cache()
    return torch.cat(chunks)[:n_tokens]


def measure(ckpt: Path, acts, device: str) -> dict:
    import torch

    from dictionary_learning.utils import load_dictionary

    ae, _ = load_dictionary(str(ckpt), device=device)
    ae.eval()
    # var_flag=0 checkpoints have no var_encoder, so the sigma columns come back
    # empty -- but the selection diagnostics still apply, and they are what says
    # whether the two trainers' definitions of "fired" can ever disagree.
    if not hasattr(ae, "var_flag"):
        raise TypeError(
            f"{type(ae).__name__} is not a VSAETopK; this script reads the "
            f"variational encoder and the vSAE's own selection statistics. "
            f"top_k.py's AutoEncoderTopK has neither."
        )
    deterministic = ae.var_flag != 1

    n_at_floor = n_at_ceil = n_total = 0
    raw_sum = raw_sq = 0.0
    raw_min, raw_max = float("inf"), float("-inf")
    sig_sel_sum = mu_sel_sum = 0.0
    n_sel = n_noise_dominated = n_exactly_zero = 0
    n_positive_sum = 0.0
    ratios = []

    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096]
            # training=False: no sampling, but log_var is computed regardless
            # (vsae_topk.py:293), which is exactly the quantity wanted here.
            _, _, mu, log_var, top_indices, _ = ae.encode(
                x, return_topk=True, training=False
            )
            lv = (torch.zeros_like(mu) if deterministic else log_var).float()
            n_total += lv.numel()
            n_at_floor += int((lv <= LOG_VAR_MIN).sum())
            n_at_ceil += int((lv >= LOG_VAR_MAX).sum())
            raw_sum += float(lv.sum())
            raw_sq += float(lv.pow(2).sum())
            raw_min = min(raw_min, float(lv.min()))
            raw_max = max(raw_max, float(lv.max()))

            # On the selected features only: that is where the noise can flip a
            # TopK decision, so it is where sigma has consequences.
            sigma = torch.exp(0.5 * lv.clamp(LOG_VAR_MIN, LOG_VAR_MAX))
            sig_sel = torch.gather(sigma, 1, top_indices)
            mu_sel = torch.gather(mu.float(), 1, top_indices)
            sig_sel_sum += float(sig_sel.sum())
            mu_sel_sum += float(mu_sel.sum())
            n_sel += sig_sel.numel()

            # The mean of sigma/|mu| is worthless here: TopK can select a feature
            # whose mu is 0 (post-ReLU), and one such denominator swamps the
            # average. Keep a subsample for a median, and count the cases that
            # actually matter -- selections the noise dominates outright.
            n_noise_dominated += int((mu_sel.abs() < sig_sel).sum())
            n_exactly_zero += int((mu_sel == 0).sum())
            ratios.append(
                (sig_sel / mu_sel.abs().clamp(min=1e-12)).flatten()[::97].cpu()
            )

            # How many features are even available to select. TopK takes k
            # whatever happens, so if fewer than k survive the ReLU it is forced
            # to pad the selection with zeros -- where the noise is the entire
            # signal.
            n_positive_sum += float((mu.float() > 0).sum(dim=1).sum())

    raw_mean = raw_sum / n_total
    all_ratios = torch.cat(ratios)
    return {
        "ratio_median": float(all_ratios.median()),
        "frac_noise_dominated": n_noise_dominated / n_sel,
        "frac_selected_zero": n_exactly_zero / n_sel,
        "n_positive_mu": n_positive_sum / acts.shape[0],
        "log_var_mean": raw_mean,
        "log_var_sd": (raw_sq / n_total - raw_mean ** 2) ** 0.5,
        "log_var_min": raw_min,
        "log_var_max": raw_max,
        "frac_at_floor": n_at_floor / n_total,
        "frac_at_ceiling": n_at_ceil / n_total,
        "sigma_selected": sig_sel_sum / n_sel,
        "mu_selected": mu_sel_sum / n_sel,
    }


def reconstruction(ckpt: Path, acts, device: str) -> dict:
    """FVE with the sampling on and off, on identical data.

    This is not redundant with `evaluation_results.json`. `evaluation.py:51` calls
    `dictionary(x, output_features=True)` and never passes `training=`, while
    `VSAETopK.forward` defaults it to True -- so every recorded FVE for a
    var_flag=1 arm was measured WITH the reparameterisation active, at the clamped
    sigma. Turning it off separates "the trained weights are bad" from "the noise
    injected at evaluation time is what we measured".
    """
    import torch

    from dictionary_learning.utils import load_dictionary

    ae, _ = load_dictionary(str(ckpt), device=device)
    ae.eval()
    out = {}
    with torch.no_grad():
        for label, training in (("sampled", True), ("deterministic", False)):
            torch.manual_seed(0)          # same noise draw for every checkpoint
            num = den = 0.0
            for i in range(0, acts.shape[0], 4096):
                x = acts[i:i + 4096]
                x_hat = ae(x, training=training).float()
                xf = x.float()
                num += float(torch.var(xf - x_hat, dim=0).sum())
                den += float(torch.var(xf, dim=0).sum())
            out[f"fve_{label}"] = 1.0 - num / den
    out["fve_gap"] = out["fve_deterministic"] - out["fve_sampled"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=50_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--fve", action="store_true",
                    help="also measure FVE with sampling on and off")
    args = ap.parse_args()

    import math

    print(f"log_var is clamped to [{LOG_VAR_MIN}, {LOG_VAR_MAX}], so sigma is "
          f"confined to [{math.exp(LOG_VAR_MIN / 2):.4f}, "
          f"{math.exp(LOG_VAR_MAX / 2):.4f}].")
    print(f"Initialised at log_var = -2.0, i.e. sigma = {math.exp(-1.0):.4f}.")
    print(f"The KL is minimised in sigma at sigma = 1.0; reconstruction wants 0.\n")

    acts = draw_activations(args.n_tokens, args.device)
    print(f"Scoring every checkpoint on the same {acts.shape[0]:,} activations.\n")

    keys = ("log_var_mean", "sigma_selected", "mu_selected", "ratio_median",
            "frac_noise_dominated", "frac_selected_zero", "n_positive_mu")
    for arm in args.arms:
        ck = checkpoints(arm)
        if not ck:
            print(f"{arm}: no checkpoints found\n")
            continue
        rows = []
        print(f"=== {arm}  ({len(ck)} seeds) ===")
        print(f"{'seed':>5} {'log_var':>10} {'max':>8} {'@floor':>8} "
              f"{'sigma_sel':>10} {'mu_sel':>8} {'sig/mu~':>8} {'noise>mu':>9} "
              f"{'mu==0':>8} {'#mu>0':>8}")
        for seed, path in ck:
            m = measure(path, acts, args.device)
            rows.append(m)
            print(f"{seed:>5} {m['log_var_mean']:>10.3f} {m['log_var_max']:>8.2f} "
                  f"{m['frac_at_floor']:>8.4f} {m['sigma_selected']:>10.4f} "
                  f"{m['mu_selected']:>8.4f} {m['ratio_median']:>8.4f} "
                  f"{m['frac_noise_dominated']:>9.4f} {m['frac_selected_zero']:>8.4f} "
                  f"{m['n_positive_mu']:>8.1f}")
        print(f"{'mean':>5} " + " ".join(
            f"{mean(r[k] for r in rows):>10.4f}" for k in keys))
        if len(rows) > 1:
            print(f"{'sd':>5} " + " ".join(
                f"{stdev([r[k] for r in rows]):>10.4f}" for k in keys))
        print(f"      ({', '.join(keys)})\n")

        if args.fve:
            fve = [reconstruction(path, acts, args.device) for _, path in ck]
            print(f"  reconstruction on the same batch, {len(fve)} seeds:")
            for label in ("fve_sampled", "fve_deterministic", "fve_gap"):
                v = [f[label] for f in fve]
                print(f"    {label:<18} {mean(v):>9.6f} ± {stdev(v):.6f}"
                      if len(v) > 1 else f"    {label:<18} {v[0]:>9.6f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
