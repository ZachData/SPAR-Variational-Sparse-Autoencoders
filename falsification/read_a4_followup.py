"""A4 follow-up: densify the L0-matched region RESULTS addendum 20 flagged as
too sparse (n=4) to distinguish "the FVE-vs-Jaccard coupling genuinely
weakens once achieved sparsity is controlled" from "this design cannot see
it". Adds four new log_var_init points (-3.5, -4.5, -5.5, -6.0) between the
original grid's -3 and -8, filling exactly the gap the original 1-unit
spacing left unsampled in the region where achieved L0 sits near the 256
target. Run at 5 seeds/arm rather than the main battery's 13 -- a
deliberately scoped-down follow-up (lower per-point precision, honestly
reported as such) whose purpose is grid DENSITY for the arm-level
correlation, not per-point statistical power (each point's own FVE/L0/Jaccard
mean is still precise to 3-4 decimals at n=5, as the printed per-point SDs
below show).

Reuses read_a4_dose_response.py's checkpoint discovery, FVE/L0 readers,
Jaccard measurement, and Pearson correlation -- imported, not reimplemented.
Only the four new arms are freshly measured; the original six points are
read from the existing falsification/a4_dose_response_results.json rather
than re-running Jaccard on 78 already-measured checkpoints.

    python falsification/read_a4_followup.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.read_a4_dose_response import (  # noqa: E402
    checkpoints,
    draw_activations,
    fve_for,
    l0_for,
    mean_jaccard,
    pearson,
)

ORIGINAL_RESULTS = REPO / "falsification" / "a4_dose_response_results.json"
OUT = REPO / "falsification" / "a4_followup_results.json"

NEW_ARMS = [
    (-3.5, "a4_jumprelu_sigma_init_m3_5"),
    (-4.5, "a4_jumprelu_sigma_init_m4_5"),
    (-5.5, "a4_jumprelu_sigma_init_m5_5"),
    (-6.0, "a4_jumprelu_sigma_init_m6"),
]

# The band addendum 20 used to define "L0-matched": achieved L0 in a
# comparable range to the 256 target, the closest this soft-target design
# gets to A2/A3's exactly-fixed sparsity.
L0_BAND = (240.0, 320.0)


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts = draw_activations(8192, device)
    print(f"Scoring convergence Jaccard on {acts.shape[0]} fixed activations "
          f"(same method as read_a4_dose_response.py).\n")

    new_rows = []
    for log_var_init, arm in NEW_ARMS:
        ck = checkpoints(arm)
        clamped = max(log_var_init, -6.0)
        sigma = float(torch.exp(torch.tensor(0.5 * clamped)))
        print(f"=== {arm} (log_var_init={log_var_init}, sigma={sigma:.4f}, "
              f"{len(ck)} seeds) ===")
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
            print(f"  seed {seed:>2}: FVE={fve:.4f}  L0={l0:.1f}  jaccard={j:.4f}")
        row = {
            "log_var_init": log_var_init, "arm": arm, "sigma": sigma,
            "n_seeds": len(ck),
            "fve_mean": mean(fves), "fve_std": stdev(fves) if len(fves) > 1 else 0.0,
            "l0_mean": mean(l0s), "l0_std": stdev(l0s) if len(l0s) > 1 else 0.0,
            "jaccard_mean": mean(jaccards),
            "jaccard_std": stdev(jaccards) if len(jaccards) > 1 else 0.0,
            "fve_per_seed": fves, "jaccard_per_seed": jaccards,
        }
        new_rows.append(row)
        print(f"  -> FVE {row['fve_mean']:.4f}+/-{row['fve_std']:.4f}  "
              f"L0 {row['l0_mean']:.1f}+/-{row['l0_std']:.1f}  "
              f"Jaccard {row['jaccard_mean']:.4f}+/-{row['jaccard_std']:.4f}\n")

    original = json.loads(ORIGINAL_RESULTS.read_text())
    all_rows = original["rows"] + new_rows
    all_rows.sort(key=lambda r: -r["log_var_init"])

    print(f"\n{'log_var_init':>13} {'sigma':>8} {'n':>3} {'FVE':>18} {'L0':>8} {'Jaccard':>18}")
    for r in all_rows:
        print(f"{r['log_var_init']:>13.1f} {r['sigma']:>8.4f} {r['n_seeds']:>3d} "
              f"{r['fve_mean']:>8.4f}+/-{r['fve_std']:<6.4f} "
              f"{r['l0_mean']:>8.1f} "
              f"{r['jaccard_mean']:>8.4f}+/-{r['jaccard_std']:<6.4f}")

    fve_all = [r["fve_mean"] for r in all_rows]
    jac_all = [r["jaccard_mean"] for r in all_rows]
    l0_all = [r["l0_mean"] for r in all_rows]
    r_fj_all = pearson(fve_all, jac_all)
    r_fl_all = pearson(fve_all, l0_all)
    print(f"\nFull {len(all_rows)}-point grid: "
          f"r(FVE, Jaccard) = {r_fj_all:+.4f}   r(FVE, L0) = {r_fl_all:+.4f}")

    matched = [r for r in all_rows if L0_BAND[0] <= r["l0_mean"] <= L0_BAND[1]]
    fve_m = [r["fve_mean"] for r in matched]
    jac_m = [r["jaccard_mean"] for r in matched]
    l0_m = [r["l0_mean"] for r in matched]
    r_fj_m = pearson(fve_m, jac_m) if len(matched) > 2 else None
    r_fl_m = pearson(fve_m, l0_m) if len(matched) > 2 else None
    print(f"\nL0-matched subset (L0 in {L0_BAND}, n={len(matched)}): "
          f"log_var_init = {[r['log_var_init'] for r in matched]}")
    print(f"  r(FVE, Jaccard) = {r_fj_m:+.4f}   r(FVE, L0) = {r_fl_m:+.4f}")
    print("  (addendum 20's original 4-point subsample: r(FVE,Jaccard)=+0.6297, "
          "r(FVE,L0)=-0.5467)")

    with open(OUT, "w") as f:
        json.dump({
            "baseline_fve_mean": original["baseline_fve_mean"],
            "baseline_l0_mean": original["baseline_l0_mean"],
            "rows": all_rows,
            "new_rows": new_rows,
            "l0_band": L0_BAND,
            "full_grid_r_fve_jaccard": r_fj_all,
            "full_grid_r_fve_l0": r_fl_all,
            "l0_matched_n": len(matched),
            "l0_matched_log_var_inits": [r["log_var_init"] for r in matched],
            "l0_matched_r_fve_jaccard": r_fj_m,
            "l0_matched_r_fve_l0": r_fl_m,
        }, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
