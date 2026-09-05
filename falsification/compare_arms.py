"""Compare any two arms on the reconstruction and pre-registered liveness metrics.

`compare_e1.py` scores one fixed pair (`e1_penalty` against a generation of the
vSAE reference arm). This is the general form: it takes two arm directories and
runs the same seed-level permutation test over every metric they share.

Two defaults here are load-bearing and are the reason this exists as a script
rather than being retyped per comparison:

* `--n-perm 4_000_000`. At 13 seeds/group `C(26,13)` = 10.4M exceeds
  `_EXACT_ENUMERATION_LIMIT`, so the test falls back to Monte Carlo, whose floor
  is `1/(n_perm+1)`. The 100k library default caps evidence at 4.42 sigma
  regardless of effect size -- an underpowered *analysis* of a fully powered
  design. 4M draws take about a second since the MC branch was vectorised and put
  the floor at 5.16 sigma.
* Both pre-registered liveness thresholds are always printed. A result counts as
  robust only if they agree; a disagreement between them is itself the finding
  (REMEDIATION.md F8b) and has twice caught a distribution *shape* change that one
  threshold alone would have reported as a clean one-directional effect.

`p_floor` and `exact` are printed for every row because a p sitting exactly at the
floor means the design ran out, not the evidence (CLAUDE.md).

    python falsification/compare_arms.py e1_vsae_ref_unitinit e1_vsae_ref_gradproj
    python falsification/compare_arms.py e1_penalty e1_vsae_ref_gradproj
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
from falsification.report_summaries import (  # noqa: E402
    PREREGISTERED_LIVENESS_THRESHOLDS,
    liveness,
)

EVAL_KEYS = ("frac_variance_explained", "frac_recovered", "frac_alive")
LIVENESS_KEYS = tuple(f"below_{rel}x" for rel in PREREGISTERED_LIVENESS_THRESHOLDS)


def _sigma(p: float) -> float:
    """Two-sided p as a normal-deviate sigma, the unit the results are written in."""
    from scipy.stats import norm

    return float(norm.isf(p / 2.0))


def effective_floor(res: dict) -> float:
    """The smallest p this run of the test could have produced.

    `seed_permutation_test` returns `p_floor` as the DESIGN floor -- the
    combinatorial `2/C(2n,n)` an exact enumeration would reach. In the Monte Carlo
    branch the analysis has a second floor, `1/(n_perm+1)`, and the binding one is
    whichever is larger. At 13v13 with 4M draws that is the Monte Carlo floor
    (2.5e-07 against the design's 1.9e-07), so reporting `p_floor` alone would
    describe evidence as short of a limit it had actually hit.
    """
    floor = res["p_floor"]
    if not res["exact"]:
        floor = max(floor, 1.0 / (res["n_draws"] + 1))
    return float(floor)


def _by_seed(root: Path, pattern: str) -> list[tuple[int, str]]:
    out = {}
    for f in glob.glob(str(root / pattern)):
        out[int(re.search(r"seed(\d+)", f).group(1))] = f
    return [(s, out[s]) for s in sorted(out)]


def _corrected_or_original(f: str) -> str:
    """Prefer evaluation_results_corrected.json when present.

    RESULTS addendum 8 / reeval_var_flag1.py: every var_flag=1 checkpoint saved
    before the scale_biases fix has its official evaluation_results.json computed
    against a corrupted var_encoder.bias, which understates sampling noise. The
    corrected re-evaluation is written alongside it rather than overwriting it
    (the original stays as the historical record), so this is the one place that
    needs to know to prefer the corrected file.
    """
    corrected = f.replace("evaluation_results.json", "evaluation_results_corrected.json")
    return corrected if Path(corrected).exists() else f


def arm_values(root: Path) -> tuple[dict[str, dict[int, float]], list[int]]:
    """Per-seed metric values, keyed by seed so two arms can be paired honestly.

    Keying by seed rather than appending to a list matters: an arm missing one
    seed's analysis would otherwise silently shift every later value against the
    other arm's, and the permutation test would compare misaligned runs.
    """
    vals: dict[str, dict[int, float]] = {k: {} for k in EVAL_KEYS + LIVENESS_KEYS}
    seeds: set[int] = set()
    for seed, f in _by_seed(root, "seed*/*/evaluation_results.json"):
        d = json.load(open(_corrected_or_original(f)))
        seeds.add(seed)
        for k in EVAL_KEYS:
            if k in d:
                vals[k][seed] = d[k]
    for seed, f in _by_seed(root, "seed*/*/comprehensive_summary_*.json"):
        lv = liveness(Path(f))
        if lv:
            seeds.add(seed)
            for k in LIVENESS_KEYS:
                vals[k][seed] = lv[k]
    return {k: v for k, v in vals.items() if v}, sorted(seeds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--n-perm", type=int, default=4_000_000,
                    help="Monte Carlo draws when exact enumeration is impossible "
                         "(default 4e6, floor 5.16 sigma)")
    args = ap.parse_args()

    roots = {}
    for arm in (args.arm_a, args.arm_b):
        root = REPO / "experiments" / arm
        if not root.is_dir():
            root = REPO / "archive" / arm          # retired pre-correction generations
        if not root.is_dir():
            print(f"No such arm directory: {arm}")
            return 1
        roots[arm] = root

    a, seeds_a = arm_values(roots[args.arm_a])
    b, seeds_b = arm_values(roots[args.arm_b])
    print(f"{args.arm_a}  vs  {args.arm_b}")
    print(f"  seeds: {len(seeds_a)} vs {len(seeds_b)}"
          + ("" if seeds_a == seeds_b else "   (SEED SETS DIFFER)"))
    print(f"  n_perm: {args.n_perm:,}\n")

    header = (f"{'metric':<24} {args.arm_a[:20]:>20} {args.arm_b[:20]:>20} "
              f"{'diff':>10} {'d':>8} {'p':>11} {'sigma':>7} {'floor':>7}")
    print(header)
    print("-" * len(header))
    for k in EVAL_KEYS + LIVENESS_KEYS:
        if k not in a or k not in b:
            continue
        va = [a[k][s] for s in sorted(a[k])]
        vb = [b[k][s] for s in sorted(b[k])]
        if len(va) < 2 or len(vb) < 2:
            continue
        sa, sb = stdev(va), stdev(vb)
        pooled = (sa + sb) / 2
        diff = mean(va) - mean(vb)
        # Across-seed SDs here are ~1e-3, so d is large for differences that may or
        # may not matter. It is an effect size, not a verdict; E1's equivalence
        # margin is the open decision that would turn one into the other.
        d = diff / pooled if pooled > 0 else float("inf")
        res = seed_permutation_test(va, vb, alternative="two-sided",
                                    n_perm=args.n_perm)
        p = res["p_value"]
        floor = effective_floor(res)
        at_floor = "*" if p <= floor else " "
        print(f"{k:<24} {mean(va):>13.6f}±{sa:<6.4f} {mean(vb):>13.6f}±{sb:<6.4f} "
              f"{diff:>+10.4f} {d:>+8.1f} {p:>10.3e}{at_floor} "
              f"{_sigma(p):>7.2f} {_sigma(floor):>7.2f}")

    print("\n  * = p is AT the design floor: the permutation budget ran out, not the")
    print("    evidence. sigma is then a lower bound on the evidence, not a measure")
    print("    of it. Raise --n-perm (vectorised; 4M draws take ~1s).")
    print("  Both liveness thresholds are pre-registered: robust only if they agree,")
    print("  and a disagreement in direction is itself the finding (F8b).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
