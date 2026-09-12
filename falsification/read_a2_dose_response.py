"""A2 -- the sigma_init dose-response: FVE vs. converged-checkpoint selection
Jaccard, across `falsification/run_arm.py`'s `a2_sigma_init_*` sweep plus the
two pre-existing endpoints `e2_sampling_only` (log_var_init=-2.0) and
`e2_sigma_low_init` (log_var_init=-8.0, which reparameterize()'s clamp pins to
an effective -6.0 from step 0 -- CLAUDE.md, PROJECT.md Next steps A2).

RESULTS addendum 16's pre-flight (`read_preact_gap.py`) found the k/(k+1)
pre-activation gap sits 80-550x below even the clamp-floor sigma, refining the
original "FVE and Jaccard knee together" prediction to "FVE should be far less
dose-sensitive than Jaccard across this grid." This script produces the actual
curve so that prediction can be checked against data rather than argued from
the gap measurement alone.

FVE per seed comes straight from each run's own `RUN_COMPLETE.json` (the
trainer's own `evaluate()`, matching the confirmatory battery -- no re-derivation).
Jaccard comes from `mean_jaccard`: two independent stochastic TopK selections
on the SAME fixed batch of activations, at the FINAL (`ae.pt`, step 10000)
checkpoint of every seed -- addendum 8 already established convergence Jaccard
is representative (its own dense-schedule read showed the early-training
transient is gone well before step 10000), so a dense schedule is not required
to answer the dose-response question, only a bonus if one wants the
early-training curve too (the four new arms carry the same dense schedule as
`e2_sampling_only_early` / `e2_sigma_low_init_early` for exactly that purpose,
unused by this script).

CLAUDE.md's `scale_biases` correction applies ONLY to checkpoints saved before
the 2026-09-04 fix: `e2_sampling_only` and `e2_sigma_low_init` predate it and
need `var_encoder.bias`/`.weight` divided by `NORM_FACTOR` before `encode()` is
called (exactly `read_selection_jaccard.py`'s correction); the four new
`a2_sigma_init_*` arms were trained today with the fixed code and must NOT be
corrected -- verified directly (a2_sigma_init_m3 seed1's saved `var_encoder.bias`
is exactly -3.0, matching `log_var_init` with no norm_factor multiplication).

    python falsification/read_a2_dose_response.py
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

NORM_FACTOR = 25.75  # falsification/read_selection_jaccard.py

# (log_var_init, arm, needs_bias_correction). Ordered by log_var_init descending
# (least negative / most noise first). -8.0 is reported at its true value but is
# clamp-equivalent to -6.0 for sigma purposes (reparameterize() clamps to [-6,2]).
ARMS = [
    (-1.0, "a2_sigma_init_m1", False),
    (-2.0, "e2_sampling_only", True),
    (-3.0, "a2_sigma_init_m3", False),
    (-4.0, "a2_sigma_init_m4", False),
    (-5.0, "a2_sigma_init_m5", False),
    (-8.0, "e2_sigma_low_init", True),
]

D, K = 2048, 256
CHANCE_JACCARD = K / (2 * D - K)


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


def mean_jaccard(ckpt: Path, acts, needs_correction: bool, device: str) -> float:
    import torch

    from dictionary_learning.trainers.vsae_topk import VSAETopK

    ae = VSAETopK.from_pretrained(str(ckpt), device=device, var_flag=1)
    ae.eval()
    k = int(ae.k.item())
    if needs_correction:
        with torch.no_grad():
            ae.var_encoder.bias.div_(NORM_FACTOR)
            ae.var_encoder.weight.div_(NORM_FACTOR)

    overlaps = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], 4096):
            x = acts[i:i + 4096].to(ae.encoder.weight.dtype)
            _, _, _, _, idx1, _ = ae.encode(x, return_topk=True, training=True)
            _, _, _, _, idx2, _ = ae.encode(x, return_topk=True, training=True)
            i1 = torch.zeros(x.shape[0], ae.dict_size, dtype=torch.bool, device=device)
            i2 = torch.zeros_like(i1)
            i1.scatter_(1, idx1, True)
            i2.scatter_(1, idx2, True)
            inter = (i1 & i2).sum(dim=1).float()
            union = 2 * k - inter
            overlaps.append((inter / union).cpu())
    return float(torch.cat(overlaps).mean())


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    return cov / (sx * sy)


def figure(rows: list[dict]) -> None:
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
    ax.axhline(0.900159, color="#52514e", lw=0.8, ls=":")
    ax.annotate("baseline FVE", (sigma[0], 0.900159), xytext=(4, 3),
                textcoords="offset points", fontsize=7.5, color="#52514e")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("sigma at convergence (log scale, decreasing noise -->)")
    ax.set_ylabel("value")
    ax.set_title("Smooth, no knee: both rise together\nas sigma falls", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.errorbar(jac, fve, xerr=jac_sd, yerr=fve_sd, fmt="o", color="#6c3483", ms=7)
    for r in rows:
        ax.annotate(f"log_var_init={r['log_var_init']:.0f}", (r["jaccard_mean"], r["fve_mean"]),
                    xytext=(5, -3), textcoords="offset points", fontsize=7, color="#52514e")
    ax.axhline(0.900159, color="#52514e", lw=0.8, ls=":")
    ax.set_xlabel("selection Jaccard at convergence")
    ax.set_ylabel("fraction of variance explained")
    ax.annotate(f"Pearson r = {r_fj:+.4f}  (n = 6 grid points)", (0.03, 0.93),
                xycoords="axes fraction", fontsize=8.5, color="#0b0b0b")
    ax.set_title("FVE tracks selection stability almost\nlinearly across the whole grid", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A2: the sigma_init dose-response is smooth, not a threshold —\n"
                  "but FVE damage tracks TopK selection churn almost exactly",
                  fontsize=11, y=1.04)
    fig.tight_layout()
    out = REPO / "workshop" / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "a2_dose_response.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(out / "a2_dose_response.png", dpi=180, bbox_inches="tight")
    print(f"\nWrote {out / 'a2_dose_response.pdf'} and {out / 'a2_dose_response.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "falsification/a2_dose_response_results.json"))
    ap.add_argument("--fig-only", action="store_true",
                     help="skip measurement, just re-plot from an existing --out JSON")
    args = ap.parse_args()

    if args.fig_only:
        data = json.loads(Path(args.out).read_text())
        figure(data["rows"])
        return 0

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    acts = draw_activations(args.n_tokens, device)
    print(f"Scoring convergence Jaccard on {acts.shape[0]} fixed activations.")
    print(f"Chance-level Jaccard for k={K}, d={D}: {CHANCE_JACCARD:.4f}\n")

    rows = []
    for log_var_init, arm, needs_correction in ARMS:
        ck = checkpoints(arm)
        # reparameterize() clamps log_var to [-6, 2] (vsae_topk.py:390) before
        # exp() -- -8.0 (e2_sigma_low_init) is clamped to -6.0 from step 0, so
        # its EFFECTIVE sigma is exp(-3)=0.0498, not exp(-4)=0.0183 from the raw
        # log_var_init. Report the effective (clamped) value.
        clamped_log_var = max(log_var_init, -6.0)
        sigma = float(torch.exp(torch.tensor(0.5 * clamped_log_var)))
        print(f"=== {arm} (log_var_init={log_var_init}, sigma={sigma:.4f}, "
              f"{len(ck)} seeds, bias_correction={needs_correction}) ===")
        fves, jaccards = [], []
        for seed, trainer_dir in ck:
            fve = fve_for(arm, seed)
            j = mean_jaccard(trainer_dir / "ae.pt", acts, needs_correction, device)
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
        json.dump({"chance_jaccard": CHANCE_JACCARD, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")

    fve_means = [row["fve_mean"] for row in rows]
    jac_means = [row["jaccard_mean"] for row in rows]
    print(f"\nPearson r(FVE, Jaccard) across the {len(rows)} grid points: "
          f"{pearson(fve_means, jac_means):+.4f}")
    figure(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
