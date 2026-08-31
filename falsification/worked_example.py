"""Apply the framework to the data already in this repository.

Run: python falsification/worked_example.py

Demonstrates the framework on the beta dose-response measured in
comprehensive_histogram_analysis/, and on the SCR/TPP claim as the preprint
made it. The point of this script is to show what the existing evidence does and
does not license -- it is the empirical motivation for the seed budget in
PROJECT.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falsification.evalues import FalsificationTest, SequentialFalsifier
from falsification.permutation import monotone_trend_test

SWEEP_TAG = "gelu-1l_d2048_k256"


def load_beta_sweep() -> tuple[list[float], list[float]]:
    betas, alive = [], []
    pattern = "comprehensive_histogram_analysis/*/comprehensive_summary_*.json"
    for path in sorted(glob.glob(pattern)):
        if SWEEP_TAG not in os.path.basename(os.path.dirname(path)):
            continue
        with open(path) as handle:
            data = json.load(handle)
        if data["model_info"].get("dictionary_type") != "VSAETopK":
            continue
        usage = data["feature_usage_summary"]
        betas.append(float(data["model_info"]["kl_coeff"]))
        alive.append(usage["features_used"] / usage["total_features"])
    order = sorted(range(len(betas)), key=lambda i: betas[i])
    return [betas[i] for i in order], [alive[i] for i in order]


def main() -> None:
    betas, alive = load_beta_sweep()
    print("Beta sweep already in the repo (gelu-1l, d=2048, k=256, auxk=1/32):")
    for b, a in zip(betas, alive):
        print(f"  beta = {b:<10g}  live features = {a:.1%}")

    print("\n" + "=" * 72)
    print("H1: 'Increasing the KL coefficient reduces the number of live features.'")
    print("We believe this is TRUE -- it is the cleanest result in the preprint.")
    print("=" * 72)

    trend = monotone_trend_test(betas, alive, direction="decreasing")
    f1 = SequentialFalsifier(main_hypothesis="Increasing beta reduces live features")
    f1.add(
        FalsificationTest(
            name="Monotone dose-response across the beta sweep",
            null_hypothesis="live-feature count is unrelated to beta",
            alt_hypothesis="live-feature count decreases in beta",
            p_value=trend["p_value"],
            unit_of_analysis=trend["unit_of_analysis"],
            n_units=trend["n_units"],
            confounders_controlled=("auxk_alpha", "k", "dict_size", "lr", "steps"),
        )
    )
    print(f1.report())
    print(
        f"\n  p_floor for a {len(betas)}-point sweep = {trend['p_floor']:.4f} "
        f"(= 1/{len(betas)}!)"
    )
    print(
        "  Even a PERFECT monotone trend over six orders of magnitude cannot\n"
        "  reach the threshold with one run per condition. Replication across\n"
        "  seeds -- not more sweep points -- is what buys evidence."
    )

    print("\n" + "=" * 72)
    print("H2: 'The vSAE learns a better-organised feature space than the SAE.'")
    print("This is the preprint's claim. Watch the implication check do its work.")
    print("=" * 72)

    f2 = SequentialFalsifier(
        main_hypothesis="vSAE learns a better-organised feature space than the SAE"
    )
    # As the preprint ran it: a strong SCR result, but SCR also rewards small
    # dictionaries, and the vSAE has 18% of its features alive against 90%.
    f2.add(
        FalsificationTest(
            name="SCR score, vSAE vs SAE (as reported in the preprint)",
            null_hypothesis="vSAE does not outscore SAE on SCR",
            alt_hypothesis="vSAE outscores SAE on SCR",
            p_value=1e-4,
            unit_of_analysis="single checkpoint pair",
            n_units=2,
            confounders_uncontrolled=("live feature count", "training seed"),
            notes="SCR rewards selective ablation, which is easier with fewer live features.",
        )
    )
    f2.add(
        FalsificationTest(
            name="TPP score, vSAE vs SAE (as reported in the preprint)",
            null_hypothesis="vSAE does not outscore SAE on TPP",
            alt_hypothesis="vSAE outscores SAE on TPP",
            p_value=1e-3,
            unit_of_analysis="single checkpoint pair",
            n_units=2,
            confounders_uncontrolled=("live feature count", "training seed"),
        )
    )
    print(f2.report())
    print(
        "\n  Both tests are excluded before any evidence accrues, because neither\n"
        "  sub-null is implied by the main null (POPPER Assumption 1). The preprint\n"
        "  read exactly these two results as support for better organisation, then\n"
        "  contradicted itself in its own conclusion. The framework declines to\n"
        "  validate on this evidence -- which is the correct answer."
    )


if __name__ == "__main__":
    main()
