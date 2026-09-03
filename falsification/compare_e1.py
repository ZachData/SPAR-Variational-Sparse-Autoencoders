"""E1: is a fixed-variance vSAE the same thing as a TopK SAE with an L2 penalty?

Compares `e1_penalty` (TopK + activation_penalty=1.0) against `e1_vsae_ref`
(vSAE, var_flag=0, kl_coeff=1.0). At sigma=1 the vSAE's KL reduces to
0.5*||mu||^2, so under the degeneracy (CLAUDE.md landmine 1) the two should be
indistinguishable.

Both arms now apply their penalty at constant full strength from step 0
(`kl_warmup_steps=0`). Before that fix the vSAE ramped its KL over the first 1000
steps while the penalty arm did not, so the comparison differed in the schedule of
the quantity it was supposed to hold fixed. Pass --archive to score the
pre-correction runs and see how much that confound was worth.

    python falsification/compare_e1.py
    python falsification/compare_e1.py --archive
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.permutation import seed_permutation_test  # noqa: E402
from falsification.report_summaries import liveness  # noqa: E402

EVAL_KEYS = ("frac_variance_explained", "frac_recovered", "frac_alive")


def _by_seed(root: Path, pattern: str):
    out = {}
    for f in glob.glob(str(root / pattern)):
        out[int(re.search(r"seed(\d+)", f).group(1))] = f
    return [out[s] for s in sorted(out)]


def arm_values(root: Path):
    """Per-seed metric values for one arm directory."""
    vals: dict[str, list[float]] = {k: [] for k in EVAL_KEYS}
    vals["frac_below_0.1x"] = []
    for f in _by_seed(root, "seed*/*/evaluation_results.json"):
        d = json.load(open(f))
        for k in EVAL_KEYS:
            vals[k].append(d[k])
    for f in _by_seed(root, "seed*/*/comprehensive_summary_*.json"):
        lv = liveness(Path(f))
        if lv:
            vals["frac_below_0.1x"].append(lv["below_0.1x"])
    return {k: v for k, v in vals.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="store_true",
                    help="score the pre-correction (kl_warmup_steps=1000) vSAE runs")
    args = ap.parse_args()

    pen = REPO / "experiments" / "e1_penalty"
    ref = (REPO / "archive" / "e1_vsae_ref_klwarmup1000") if args.archive \
        else (REPO / "experiments" / "e1_vsae_ref")

    label = "kl_warmup_steps=1000 (PRE-CORRECTION, confounded)" if args.archive \
        else "kl_warmup_steps=0 (schedule-matched)"
    print(f"E1: e1_penalty  vs  e1_vsae_ref  [{label}]")
    print("Degeneracy predicts: indistinguishable.\n")

    a, b = arm_values(pen), arm_values(ref)
    print(f"{'metric':<24} {'e1_penalty':>20} {'e1_vsae_ref':>20} "
          f"{'diff':>10} {'d':>9} {'p':>9}")
    for k in sorted(set(a) & set(b)):
        va, vb = a[k], b[k]
        if len(va) < 2 or len(vb) < 2:
            continue
        sa, sb = stdev(va), stdev(vb)
        pooled = (sa + sb) / 2
        diff = mean(va) - mean(vb)
        d = diff / pooled if pooled > 0 else float("inf")
        res = seed_permutation_test(va, vb, alternative="two-sided")
        print(f"{k:<24} {mean(va):>13.6f}±{sa:<6.4f} {mean(vb):>13.6f}±{sb:<6.4f} "
              f"{diff:>+10.4f} {d:>+9.1f} {res['p_value']:>9.5f}")

    print("\nEvery p at 0.00216 is the exact two-sided minimum for 6v6 -- the design "
          "floor,\nnot a measure of how large the effect is. Read the d column for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
