"""A3 -- Claim #3's discreteness companion: does TopK's near-perfect FVE-vs-
selection-Jaccard coupling (RESULTS addendum 17, r=+0.9993) hold just as
tightly for BatchTopK's GLOBAL top-(k*batch_size) selection over the whole
flattened batch, or does a more elastic, less-per-token-rigid selection budget
decouple noise-induced churn from reconstruction damage?

Companion to `falsification/read_a2_dose_response.py` -- same method (FVE off
each run's own RUN_COMPLETE.json, convergence-checkpoint selection Jaccard on a
shared fixed activation batch), same sigma grid (log_var_init in
{-1,-2,-3,-4,-5,-8}, reparameterize()'s clamp is IDENTICAL between the two
trainers so sigma values line up point-for-point), same 13 seeds/point. Run
via `falsification/run_arm.py`'s `a3_batchtopk_*` arms /
`falsification/run_a3_sweep.sh`.

Two differences from the TopK reader, both load-bearing:
  * `VSAEBatchTopK.encode()` returns a dict with a `selection_mask` boolean
    tensor [n_tokens, dict_size] directly -- no index-scatter needed, unlike
    `VSAETopK.encode(..., return_topk=True)`'s index-tuple API.
  * BatchTopK's global budget does NOT guarantee exactly k active features per
    token (only k averaged over the whole batch), so Jaccard's union must be
    computed directly via boolean OR per token, not inferred as `2k - inter`
    the way `read_a2_dose_response.py`'s TopK-specific shortcut does.

All a3_batchtopk_* checkpoints were trained today (2026-09-11) with the fixed
`scale_biases` (see the fix's comment in `vsae_batch_topk.py` and RESULTS
addendum 18) -- no bias correction needed, unlike `read_a2_dose_response.py`'s
two reused pre-fix TopK arms.

    python falsification/read_a3_dose_response.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# (log_var_init, arm). Ordered by log_var_init descending (most noise first).
ARMS = [
    (-1.0, "a3_batchtopk_sigma_init_m1"),
    (-2.0, "a3_batchtopk_sampling_only"),
    (-3.0, "a3_batchtopk_sigma_init_m3"),
    (-4.0, "a3_batchtopk_sigma_init_m4"),
    (-5.0, "a3_batchtopk_sigma_init_m5"),
    (-8.0, "a3_batchtopk_sigma_low_init"),
]
BASELINE_ARM = "a3_batchtopk_baseline"  # var_flag=0, deterministic reference


def checkpoints(arm: str) -> list[tuple[int, Path]]:
    found = {}
    for p in (REPO / "experiments" / arm).glob("seed*/*/trainer_0"):
        if (p / "ae.pt").exists():
            found[int(re.search(r"seed(\d+)", str(p)).group(1))] = p
    return [(s, found[s]) for s in sorted(found)]


def fve_for(arm: str, seed: int) -> float | None:
    p = REPO / "experiments" / arm / f"seed{seed}" / "RUN_COMPLETE.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["results"]["frac_variance_explained"]


def draw_activations(n_tokens: int, device: str):
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


def mean_jaccard(ckpt_dir: Path, acts, device: str) -> float:
    import torch

    from dictionary_learning.utils import load_dictionary

    ae, _ = load_dictionary(str(ckpt_dir), device=device)
    ae.eval()

    overlaps = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096].to(ae.encoder.weight.dtype)
            mask1 = ae.encode(x)["selection_mask"]
            mask2 = ae.encode(x)["selection_mask"]
            inter = (mask1 & mask2).sum(dim=1).float()
            union = (mask1 | mask2).sum(dim=1).float()
            jaccard = torch.where(union > 0, inter / union, torch.ones_like(union))
            overlaps.append(jaccard.cpu())
    return float(torch.cat(overlaps).mean())


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    return cov / (sx * sy)


def figure(rows: list[dict], baseline_fve: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: -r["sigma"])
    sigma = [r["sigma"] for r in rows]
    fve = [r["fve_mean"] for r in rows]
    fve_sd = [r["fve_std"] for r in rows]
    jac = [r["jaccard_mean"] for r in rows]
    jac_sd = [r["jaccard_std"] for r in rows]
    r_fj = pearson(fve, jac)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    ax.errorbar(sigma, fve, yerr=fve_sd, fmt="o-", color="#c0392b", label="FVE", ms=6)
    ax.errorbar(sigma, jac, yerr=jac_sd, fmt="s-", color="#2471a3", label="Jaccard", ms=6)
    if baseline_fve is not None:
        ax.axhline(baseline_fve, color="#52514e", lw=0.8, ls=":")
        ax.annotate("BatchTopK baseline FVE", (sigma[0], baseline_fve), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5, color="#52514e")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("sigma at convergence (log scale, decreasing noise -->)")
    ax.set_ylabel("value")
    ax.set_title("BatchTopK: FVE and Jaccard vs. sigma", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.errorbar(jac, fve, xerr=jac_sd, yerr=fve_sd, fmt="o", color="#1e8449", ms=7)
    for r in rows:
        ax.annotate(f"log_var_init={r['log_var_init']:.0f}", (r["jaccard_mean"], r["fve_mean"]),
                    xytext=(5, -3), textcoords="offset points", fontsize=7, color="#52514e")
    if baseline_fve is not None:
        ax.axhline(baseline_fve, color="#52514e", lw=0.8, ls=":")
    ax.set_xlabel("selection Jaccard at convergence")
    ax.set_ylabel("fraction of variance explained")
    ax.annotate(f"Pearson r = {r_fj:+.4f}  (n = {len(rows)} grid points)", (0.03, 0.93),
                xycoords="axes fraction", fontsize=8.5, color="#0b0b0b")
    ax.set_title("BatchTopK: FVE vs. selection stability", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A3: TopK vs. BatchTopK -- does global selection decouple\n"
                  "FVE damage from selection churn?", fontsize=11, y=1.04)
    fig.tight_layout()
    out = REPO / "workshop" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "a3_batchtopk_dose_response.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out / "a3_batchtopk_dose_response.png", dpi=180, bbox_inches="tight")
    print(f"\nWrote {out / 'a3_batchtopk_dose_response.pdf'} and .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "falsification/a3_dose_response_results.json"))
    ap.add_argument("--fig-only", action="store_true")
    args = ap.parse_args()

    if args.fig_only:
        data = json.loads(Path(args.out).read_text())
        figure(data["rows"], data.get("baseline_fve_mean"))
        return 0

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    acts = draw_activations(args.n_tokens, device)
    print(f"Scoring convergence Jaccard on {acts.shape[0]} fixed activations.\n")

    baseline_fves = [fve_for(BASELINE_ARM, s) for s, _ in checkpoints(BASELINE_ARM)]
    baseline_fves = [f for f in baseline_fves if f is not None]
    baseline_fve_mean = mean(baseline_fves) if baseline_fves else None
    print(f"=== {BASELINE_ARM} (var_flag=0, deterministic) ===")
    print(f"  n={len(baseline_fves)}  FVE {baseline_fve_mean:.4f} +/- "
          f"{stdev(baseline_fves) if len(baseline_fves) > 1 else 0.0:.4f}\n")

    rows = []
    for log_var_init, arm in ARMS:
        ck = checkpoints(arm)
        clamped_log_var = max(log_var_init, -6.0)
        sigma = float(torch.exp(torch.tensor(0.5 * clamped_log_var)))
        print(f"=== {arm} (log_var_init={log_var_init}, sigma={sigma:.4f}, {len(ck)} seeds) ===")
        fves, jaccards = [], []
        for seed, trainer_dir in ck:
            fve = fve_for(arm, seed)
            j = mean_jaccard(trainer_dir, acts, device)
            if fve is not None:
                fves.append(fve)
            jaccards.append(j)
            print(f"  seed {seed:>2}: FVE={fve if fve is not None else float('nan'):.4f}  jaccard={j:.4f}")
        row = {
            "log_var_init": log_var_init,
            "arm": arm,
            "sigma": sigma,
            "n_seeds": len(ck),
            "fve_mean": mean(fves) if fves else None,
            "fve_std": stdev(fves) if len(fves) > 1 else 0.0,
            "jaccard_mean": mean(jaccards) if jaccards else None,
            "jaccard_std": stdev(jaccards) if len(jaccards) > 1 else 0.0,
            "fve_per_seed": fves,
            "jaccard_per_seed": jaccards,
        }
        rows.append(row)
        print(f"  -> FVE {row['fve_mean']:.4f} +/- {row['fve_std']:.4f}   "
              f"Jaccard {row['jaccard_mean']:.4f} +/- {row['jaccard_std']:.4f}\n")

    print(f"{'log_var_init':>13} {'sigma':>8} {'n':>3} {'FVE':>18} {'Jaccard':>18}")
    for row in rows:
        print(f"{row['log_var_init']:>13.1f} {row['sigma']:>8.4f} {row['n_seeds']:>3d} "
              f"{row['fve_mean']:>8.4f}+/-{row['fve_std']:<6.4f} "
              f"{row['jaccard_mean']:>8.4f}+/-{row['jaccard_std']:<6.4f}")

    with open(args.out, "w") as f:
        json.dump({"baseline_fve_mean": baseline_fve_mean,
                   "baseline_fve_per_seed": baseline_fves, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")

    fve_means = [row["fve_mean"] for row in rows]
    jac_means = [row["jaccard_mean"] for row in rows]
    r_fj = pearson(fve_means, jac_means)
    print(f"\nPearson r(FVE, Jaccard) across the {len(rows)} grid points: {r_fj:+.4f}")
    print("(TopK's A2 comparison figure: r = +0.9993, RESULTS addendum 17)")

    figure(rows, baseline_fve_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
