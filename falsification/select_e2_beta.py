"""E2 stage 1: apply the pre-registered beta-selection rule to the pilot.

Why E2 needs a beta pilot at all
--------------------------------
`kl_coeff` is NOT a shared scale across `var_flag`. At `var_flag=0` the KL reduces
to `0.5*||mu||^2`, a mild L2 penalty on the activations. At `var_flag=1` the
variance term enters and dominates -- it contributed ~220 of a ~225 total loss in
the pilot. So "E2 vs e1_vsae_ref at matched beta" compares two different
interventions, not one intervention under two settings.

The consequence was not subtle: at `beta = 1.0` the model posterior-collapses in
all six seeds (mu -> 1e-3, FVE = 0.0001, `frac_recovered` ~ -0.74, i.e. worse than
zero-ablating the activation). A model in that state has no feature structure, so
E2 as first run could not distinguish "a genuinely variational SAE degenerates"
from "beta = 1.0 is far too large once sampling is switched on"
(falsification/FINDINGS_2026-09-02.md item 6, REMEDIATION.md F6).

The rule, fixed before any pilot result existed
-----------------------------------------------
    Select the LARGEST beta whose frac_variance_explained is within
    E2_SELECTION_FVE_MARGIN (0.02) of the baseline arm's mean FVE.
    If no beta qualifies, select the SMALLEST beta tried.

"Largest that still reconstructs" is the point: E2 asks whether a working
variational SAE degenerates, so it wants the strongest KL pressure that leaves a
model worth measuring. The fallback exists so the rule always returns something
rather than leaving the choice to whoever reads the table.

This is a script and not a paragraph in a document because a rule a human applies
after seeing the numbers is not a pre-registered rule. Stage 2 then trains six
confirmatory seeds at the selected beta, seeds DISJOINT from the pilot's, so no
checkpoint contributes to both selection and inference.

    python falsification/select_e2_beta.py            # apply the rule, print the pick
    python falsification/select_e2_beta.py --quiet    # print only the beta, for scripts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falsification.run_arm import (  # noqa: E402
    E2_PILOT_BETAS,
    E2_PILOT_SEED,
    E2_SELECTION_FVE_MARGIN,
)

METRIC = "frac_variance_explained"


def _fve(run: Path) -> float | None:
    marker = run / "RUN_COMPLETE.json"
    if not marker.exists():
        return None
    results = json.loads(marker.read_text()).get("results") or {}
    value = results.get(METRIC)
    return float(value) if isinstance(value, (int, float)) else None


def baseline_fve() -> float:
    """Mean FVE over the baseline arm's seeds -- the reference the margin is around."""
    values = [
        v for run in sorted((REPO / "experiments" / "baseline").glob("seed*"))
        if (v := _fve(run)) is not None
    ]
    if not values:
        raise SystemExit(
            "No baseline runs found. The selection rule is defined relative to the "
            "baseline's mean FVE, so the baseline arm must be trained first."
        )
    return mean(values)


def pilot_fves() -> dict[float, float | None]:
    return {
        beta: _fve(REPO / "experiments" / f"e2_beta_pilot_{beta:g}" / f"seed{E2_PILOT_SEED}")
        for beta in E2_PILOT_BETAS
    }


def select(reference: float, fves: dict[float, float | None]) -> tuple[float, str]:
    """The pre-registered rule. Returns (beta, why)."""
    qualifying = [
        beta for beta, fve in fves.items()
        if fve is not None and abs(fve - reference) <= E2_SELECTION_FVE_MARGIN
    ]
    if qualifying:
        return max(qualifying), "largest beta within the FVE margin"
    return min(fves), "fallback: no beta qualified, so the smallest tried"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true",
                    help="print only the selected beta (for shell substitution)")
    args = ap.parse_args()

    reference = baseline_fve()
    fves = pilot_fves()

    missing = [f"{b:g}" for b, v in fves.items() if v is None]
    if missing:
        raise SystemExit(
            f"Pilot incomplete: no result for beta {', '.join(missing)}. "
            f"Run every e2_beta_pilot_* arm at seed {E2_PILOT_SEED} before selecting, "
            "so the rule sees the whole grid it was written against."
        )

    beta, why = select(reference, fves)

    if args.quiet:
        print(f"{beta:g}")
        return 0

    print(f"E2 stage-1 beta selection (pilot seed {E2_PILOT_SEED})")
    print(f"Rule (pre-registered): largest beta with |FVE - baseline| <= "
          f"{E2_SELECTION_FVE_MARGIN}\n")
    print(f"baseline mean {METRIC}: {reference:.6f}\n")
    print(f"{'beta':>10} {METRIC:>26} {'|diff|':>10} {'qualifies':>10}")
    for b in sorted(fves):
        v = fves[b]
        diff = abs(v - reference)
        print(f"{b:>10g} {v:>26.6f} {diff:>10.4f} "
              f"{'yes' if diff <= E2_SELECTION_FVE_MARGIN else 'no':>10}")
    print(f"\nSELECTED beta = {beta:g}  ({why})")
    print(f"\nStage 2: 6 confirmatory seeds at kl_coeff={beta:g}, var_flag=1, "
          f"seeds 1-6 --\n  disjoint from the pilot's seed {E2_PILOT_SEED}, so no "
          "checkpoint informs both\n  selection and inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
