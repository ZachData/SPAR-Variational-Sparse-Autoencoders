"""E1: is a fixed-variance vSAE the same thing as a TopK SAE with an L2 penalty?

Compares `e1_penalty` (TopK + activation_penalty=1.0) against `e1_vsae_ref`
(vSAE, var_flag=0, kl_coeff=1.0). At sigma=1 the vSAE's KL reduces to
0.5*||mu||^2, so under the degeneracy (CLAUDE.md landmine 1) the two should be
indistinguishable.

Two confounds have been removed from this comparison in turn, and each
pre-correction generation is archived so its cost stays measurable rather than
merely asserted. Use --ref to score any of the three:

  gradproj  everything below, plus the decoder-gradient projection: the last
            identified asymmetry, run as a factor (see run_arm.py)
  unitinit  schedule- and parameterisation-matched AND started at the same decoder
            norm; this is the generation E1's pre-registered claim is scored on
  current   both arms apply their penalty at constant full strength from step 0
            AND share the TopK bias form (`use_april_update_mode=False`), but the
            vSAE still starts its decoder 10x smaller
  aprilmode schedule-matched, but the vSAE still had no pre-bias and an untied
            decoder.bias while the penalty arm centred its input on a tied b_dec
  klwarmup  neither matched: the vSAE ramped its KL over the first 1000 steps
            while the penalty arm applied its penalty flat from step 0

    python falsification/compare_e1.py
    python falsification/compare_e1.py --ref aprilmode
    python falsification/compare_e1.py --ref klwarmup
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

from falsification.permutation import min_p_floor, seed_permutation_test  # noqa: E402
from falsification.report_summaries import liveness  # noqa: E402

EVAL_KEYS = ("frac_variance_explained", "frac_recovered", "frac_alive")

# Each generation of the vSAE arm, newest first, with the confound it still carries.
REF_ARMS = {
    "gradproj": (
        "experiments/e1_vsae_ref_gradproj",
        "init 1.0 AND the decoder-gradient projection top_k.py applies",
    ),
    "unitinit": (
        "experiments/e1_vsae_ref_unitinit",
        "decoder_init_scale=1.0, matching top_k.py's unit-norm init",
    ),
    "current": (
        "experiments/e1_vsae_ref",
        "schedule- and parameterisation-matched, but decoder_init_scale=0.1",
    ),
    "aprilmode": (
        "archive/e1_vsae_ref_aprilmode",
        "use_april_update_mode=True (PRE-CORRECTION: bias form unmatched)",
    ),
    "klwarmup": (
        "archive/e1_vsae_ref_klwarmup1000",
        "kl_warmup_steps=1000 (PRE-CORRECTION: schedule unmatched too)",
    ),
}


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
    ap.add_argument("--ref", default="current",
                    choices=sorted(REF_ARMS),
                    help="which generation of the vSAE arm to score")
    ap.add_argument("--archive", action="store_true",
                    help=argparse.SUPPRESS)  # back-compat: == --ref klwarmup
    args = ap.parse_args()

    which = "klwarmup" if args.archive else args.ref
    rel, label = REF_ARMS[which]

    pen = REPO / "experiments" / "e1_penalty"
    ref = REPO / rel
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

    # This ran at 6 seeds, where every p sat at 0.00216 -- the exact two-sided
    # floor for 6v6, 3.07 sigma, reachable by any effect large enough. The arms are
    # at 13 seeds now, so the floor moved and the p-values are no longer pinned to
    # it, but the caution is the same one and still applies: check the floor before
    # reading a p as evidence.
    n = min(len(v) for v in list(a.values()) + list(b.values()))
    floor = min_p_floor(n, n, "two-sided")
    print(f"\nSmallest attainable p at {n} seeds/group: {floor:.3g} (exact "
          f"enumeration).\nA p sitting there means the design ran out, not the "
          f"evidence. Read the d column\nfor effect size -- and note the default "
          f"n_perm caps Monte Carlo p at 1e-5;\nfalsification/compare_arms.py runs "
          f"the same test at 4e6 draws.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
