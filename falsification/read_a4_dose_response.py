"""A4 -- Claim #3's SHARPEST discreteness test: does TopK's near-perfect
FVE-vs-selection-Jaccard coupling (RESULTS addendum 17, r=+0.9993; A3's
BatchTopK analog, r=+0.9979) hold for a per-feature LEARNED THRESHOLD
(JumpReLU) with no k at all, or is the coupling specific to hard top-k
selection (picking exactly k, however the budget is scoped)?

Companion to `falsification/read_a2_dose_response.py` /
`read_a3_dose_response.py` -- same method (FVE off each run's own
RUN_COMPLETE.json, convergence-checkpoint selection Jaccard on a shared fixed
activation batch), same sigma grid (log_var_init in {-1,-2,-3,-4,-5,-8}, the
SAME reparameterize() clamp as the other two trainers so sigma values line up
point-for-point), same 13 seeds/point. Run via `falsification/run_arm.py`'s
`a4_jumprelu_*` arms / `falsification/run_a4_sweep.sh`.

Differences from the other two readers, all load-bearing:
  * `VSAEJumpReLU.encode(x)` returns the raw, UNGATED pre-activation as `mu`
    (CLAUDE.md: the gate used to run before sampling, which made selection
    churn structurally impossible -- fixed 2026-09-12 to gate after noise).
    Getting the SAME selection two independent forward passes would sample
    requires calling `reparameterize` then `select` explicitly, not just
    `encode`.
  * There is no `2k - intersection` shortcut (TopK) or an exposed boolean
    mask attribute (BatchTopK's `selection_mask`) -- the mask here is derived
    directly as `select(z) > 0`, computed twice per token on independent noise
    draws, mirroring BatchTopK's direct-boolean-OR approach since JumpReLU
    likewise does not guarantee any fixed per-token or per-batch active count.
  * All a4_jumprelu_* checkpoints were trained AFTER all three vsae_jump_relu.py
    fixes (STE gradient, L0-target loss, gate-after-noise order) landed --
    like A3's checkpoints, no addendum-8-style bias correction is needed.

    python falsification/read_a4_dose_response.py
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
    (-1.0, "a4_jumprelu_sigma_init_m1"),
    (-2.0, "a4_jumprelu_sampling_only"),
    (-3.0, "a4_jumprelu_sigma_init_m3"),
    (-4.0, "a4_jumprelu_sigma_init_m4"),
    (-5.0, "a4_jumprelu_sigma_init_m5"),
    (-8.0, "a4_jumprelu_sigma_low_init"),
]
BASELINE_ARM = "a4_jumprelu_baseline"  # var_flag=0, deterministic reference


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


def l0_for(arm: str, seed: int) -> float | None:
    p = REPO / "experiments" / arm / f"seed{seed}" / "RUN_COMPLETE.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["results"].get("l0")


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
            x = acts[i:i + 4096].to(ae.W_enc.dtype)
            mu, log_var = ae.encode(x)
            mask1 = ae.select(ae.reparameterize(mu, log_var)) > 0
            mask2 = ae.select(ae.reparameterize(mu, log_var)) > 0
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
        ax.annotate("JumpReLU baseline FVE", (sigma[0], baseline_fve), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5, color="#52514e")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("sigma at convergence (log scale, decreasing noise -->)")
    ax.set_ylabel("value")
    ax.set_title("JumpReLU: FVE and Jaccard vs. sigma", fontsize=10)
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
    ax.set_title("JumpReLU: FVE vs. selection stability", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A4: does a learned threshold (no k) decouple\n"
                  "FVE damage from selection churn?", fontsize=11, y=1.04)
    fig.tight_layout()
    out = REPO / "workshop" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "a4_jumprelu_dose_response.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out / "a4_jumprelu_dose_response.png", dpi=180, bbox_inches="tight")
    print(f"\nWrote {out / 'a4_jumprelu_dose_response.pdf'} and .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "falsification/a4_dose_response_results.json"))
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
    baseline_l0s = [l0_for(BASELINE_ARM, s) for s, _ in checkpoints(BASELINE_ARM)]
    baseline_l0s = [l for l in baseline_l0s if l is not None]
    baseline_fve_mean = mean(baseline_fves) if baseline_fves else None
    print(f"=== {BASELINE_ARM} (var_flag=0, deterministic) ===")
    l0_note = f"  L0 {mean(baseline_l0s):.1f} (target 256)" if baseline_l0s else ""
    print(f"  n={len(baseline_fves)}  FVE {baseline_fve_mean:.4f} +/- "
          f"{stdev(baseline_fves) if len(baseline_fves) > 1 else 0.0:.4f}{l0_note}\n")

    rows = []
    for log_var_init, arm in ARMS:
        ck = checkpoints(arm)
        clamped_log_var = max(log_var_init, -6.0)
        sigma = float(torch.exp(torch.tensor(0.5 * clamped_log_var)))
        print(f"=== {arm} (log_var_init={log_var_init}, sigma={sigma:.4f}, {len(ck)} seeds) ===")
        fves, jaccards, l0s = [], [], []
        for seed, trainer_dir in ck:
            fve = fve_for(arm, seed)
            l0 = l0_for(arm, seed)
            j = mean_jaccard(trainer_dir, acts, device)
            if fve is not None:
                fves.append(fve)
            if l0 is not None:
                l0s.append(l0)
            jaccards.append(j)
            print(f"  seed {seed:>2}: FVE={fve if fve is not None else float('nan'):.4f}  "
                  f"L0={l0 if l0 is not None else float('nan'):.1f}  jaccard={j:.4f}")
        row = {
            "log_var_init": log_var_init,
            "arm": arm,
            "sigma": sigma,
            "n_seeds": len(ck),
            "fve_mean": mean(fves) if fves else None,
            "fve_std": stdev(fves) if len(fves) > 1 else 0.0,
            "l0_mean": mean(l0s) if l0s else None,
            "jaccard_mean": mean(jaccards) if jaccards else None,
            "jaccard_std": stdev(jaccards) if len(jaccards) > 1 else 0.0,
            "fve_per_seed": fves,
            "jaccard_per_seed": jaccards,
        }
        rows.append(row)
        print(f"  -> FVE {row['fve_mean']:.4f} +/- {row['fve_std']:.4f}   "
              f"L0 {row['l0_mean']:.1f}   "
              f"Jaccard {row['jaccard_mean']:.4f} +/- {row['jaccard_std']:.4f}\n")

    print(f"{'log_var_init':>13} {'sigma':>8} {'n':>3} {'FVE':>18} {'L0':>8} {'Jaccard':>18}")
    for row in rows:
        print(f"{row['log_var_init']:>13.1f} {row['sigma']:>8.4f} {row['n_seeds']:>3d} "
              f"{row['fve_mean']:>8.4f}+/-{row['fve_std']:<6.4f} "
              f"{row['l0_mean']:>8.1f} "
              f"{row['jaccard_mean']:>8.4f}+/-{row['jaccard_std']:<6.4f}")

    with open(args.out, "w") as f:
        json.dump({"baseline_fve_mean": baseline_fve_mean,
                   "baseline_fve_per_seed": baseline_fves,
                   "baseline_l0_mean": mean(baseline_l0s) if baseline_l0s else None,
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")

    fve_means = [row["fve_mean"] for row in rows]
    jac_means = [row["jaccard_mean"] for row in rows]
    r_fj = pearson(fve_means, jac_means)
    print(f"\nPearson r(FVE, Jaccard) across the {len(rows)} grid points: {r_fj:+.4f}")
    print("(TopK's A2: r = +0.9993, addendum 17; BatchTopK's A3: r = +0.9979, addendum 18)")

    figure(rows, baseline_fve_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
