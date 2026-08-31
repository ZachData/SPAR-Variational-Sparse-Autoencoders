"""Monte Carlo validation of the falsification protocol itself.

Two questions that cannot be answered by staring at the theory, and that decide
how the seeded experiments in PROJECT.md must be designed:

1. **Dependence.** POPPER's Assumption 2 requires E[e_i | D_{i-1}] <= 1, i.e. each
   e-value is valid *conditional on the previous ones*. POPPER gets this nearly for
   free because each falsification test queries a different database. We do not:
   if three falsification tests all compute metrics from the same ten training
   runs, their p-values are dependent. Does the product of e-values still control
   Type-I error?

2. **Adaptive ordering.** The framework lets the analyst stop whenever they like.
   What if they also *choose* which test to run next after peeking at all the
   p-values? That violates Assumption 2 outright. How bad is it?

Run: python falsification/simulate.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falsification.evalues import FalsificationTest, SequentialFalsifier
from falsification.permutation import seed_permutation_test

Selection = Literal["prespecified", "cherry_pick"]


def _correlated_metrics(
    n_runs: int, n_metrics: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Metrics for each run, sharing a common factor of correlation `rho`.

    rho = 0  -> metrics are independent measurements of the same runs
    rho = 1  -> every metric is the same measurement (maximal duplication)

    All are pure noise: there is no group effect, so every null is true.
    """
    common = rng.normal(size=(n_runs, 1))
    idiosyncratic = rng.normal(size=(n_runs, n_metrics))
    return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic


def simulate_trial(
    n_per_group: int,
    n_metrics: int,
    rho: float,
    rng: np.random.Generator,
    alpha: float = 0.1,
    kappa: float = 0.5,
    selection: Selection = "prespecified",
) -> bool:
    """One trial under a TRUE null. Returns whether the framework falsely validated."""
    n_runs = 2 * n_per_group
    metrics = _correlated_metrics(n_runs, n_metrics, rho, rng)

    p_values = []
    for j in range(n_metrics):
        res = seed_permutation_test(
            metrics[:n_per_group, j], metrics[n_per_group:, j], alternative="greater"
        )
        p_values.append(res["p_value"])

    if selection == "cherry_pick":
        # The abuse: peek at every p-value, then feed the most significant first.
        p_values.sort()

    f = SequentialFalsifier(
        main_hypothesis="null", alpha=alpha, kappa=kappa, max_tests=n_metrics
    )
    for p in p_values:
        decision = f.add(
            FalsificationTest(
                name="t",
                null_hypothesis="no effect",
                alt_hypothesis="effect",
                p_value=p,
                unit_of_analysis="training run (seed)",
                n_units=n_runs,
                confounders_controlled=("config",),
            )
        )
        if decision == "reject_null":
            break
    return f.validated()


def type_i_error(
    n_per_group: int,
    n_metrics: int,
    rho: float,
    n_trials: int = 4000,
    alpha: float = 0.1,
    kappa: float = 0.5,
    selection: Selection = "prespecified",
    seed: int = 0,
) -> tuple[float, float]:
    """Empirical false-validation rate and its Monte Carlo standard error."""
    rng = np.random.default_rng(seed)
    hits = sum(
        simulate_trial(n_per_group, n_metrics, rho, rng, alpha, kappa, selection)
        for _ in range(n_trials)
    )
    rate = hits / n_trials
    stderr = float(np.sqrt(max(rate * (1 - rate), 1e-12) / n_trials))
    return rate, stderr


@dataclass
class PowerResult:
    n_per_group: int
    effect_size: float
    power: float


def power_curve(
    effect_size: float,
    n_per_group_values: tuple[int, ...] = (4, 5, 6, 8, 10),
    n_metrics: int = 2,
    n_trials: int = 2000,
    alpha: float = 0.1,
    kappa: float = 0.5,
    seed: int = 0,
) -> list[PowerResult]:
    """Probability of validating a TRUE alternative, as a function of seed count.

    `effect_size` is the mean shift between architectures in units of the
    across-seed standard deviation of the metric. This is the number that converts
    "how many seeds should we train?" into an answerable question.
    """
    results = []
    for n in n_per_group_values:
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_trials):
            a = rng.normal(loc=effect_size, size=(n, n_metrics))
            b = rng.normal(size=(n, n_metrics))
            f = SequentialFalsifier(
                main_hypothesis="alt", alpha=alpha, kappa=kappa, max_tests=n_metrics
            )
            for j in range(n_metrics):
                res = seed_permutation_test(a[:, j], b[:, j], alternative="greater")
                decision = f.add(
                    FalsificationTest(
                        name="t",
                        null_hypothesis="no effect",
                        alt_hypothesis="effect",
                        p_value=res["p_value"],
                        unit_of_analysis="training run (seed)",
                        n_units=2 * n,
                        confounders_controlled=("config",),
                    )
                )
                if decision == "reject_null":
                    break
            hits += f.validated()
        results.append(PowerResult(n, effect_size, hits / n_trials))
    return results


def main() -> None:
    alpha = 0.1
    print("=" * 74)
    print("1. Does reusing the SAME training runs across tests break error control?")
    print(f"   All nulls true; 5 seeds/group; alpha = {alpha}")
    print("=" * 74)
    print(f"{'metrics':>8} {'rho':>6} {'Type-I':>9} {'+/- se':>8}   verdict")
    for n_metrics in (2, 3, 5):
        for rho in (0.0, 0.5, 0.95, 1.0):
            rate, se = type_i_error(5, n_metrics, rho, alpha=alpha)
            verdict = "OK" if rate <= alpha + 2 * se else "INFLATED"
            print(f"{n_metrics:>8} {rho:>6.2f} {rate:>9.4f} {se:>8.4f}   {verdict}")

    print()
    print("=" * 74)
    print("2. What if the analyst peeks at all p-values and runs the best one first?")
    print("=" * 74)
    print(f"{'metrics':>8} {'rho':>6} {'prespec':>9} {'cherry':>9}   inflation")
    for n_metrics in (3, 5, 10):
        rate_pre, _ = type_i_error(5, n_metrics, 0.0, alpha=alpha, selection="prespecified")
        rate_cp, se_cp = type_i_error(5, n_metrics, 0.0, alpha=alpha, selection="cherry_pick")
        flag = "" if rate_cp <= alpha + 2 * se_cp else "  <-- VIOLATION"
        print(f"{n_metrics:>8} {0.0:>6.2f} {rate_pre:>9.4f} {rate_cp:>9.4f}{flag}")

    print()
    print("=" * 74)
    print("3. Power: probability of validating a TRUE effect (2 pre-specified tests)")
    print("=" * 74)
    print(f"{'seeds/grp':>10} " + " ".join(f"{f'd={d}':>8}" for d in (0.5, 1.0, 1.5, 2.0)))
    by_n: dict[int, list[float]] = {}
    for d in (0.5, 1.0, 1.5, 2.0):
        for r in power_curve(d, n_trials=1500):
            by_n.setdefault(r.n_per_group, []).append(r.power)
    for n, powers in sorted(by_n.items()):
        print(f"{n:>10} " + " ".join(f"{p:>8.2f}" for p in powers))


if __name__ == "__main__":
    main()
