"""E-values and sequential falsification for claims about sparse autoencoders.

Implements the p-to-e calibration and sequential aggregation of Vovk & Wang
(2021) / Grunwald et al. (2020), as used by POPPER (Huang et al., 2025), for
hypotheses of the form "architecture A yields better <property> than architecture B".

The single most important property: aggregated evidence E = prod_i e_i is a
non-negative supermartingale under the null, so the decision rule E >= 1/alpha
controls Type-I error at alpha *even when the analyst chooses adaptively how many
tests to run and when to stop*. That is what licenses "keep testing until the
evidence is convincing" without the usual multiple-comparisons penalty.

Correctness notes that are easy to get wrong:

* The calibrator e = kappa * p^(kappa-1) satisfies E[e] = 1 exactly for any
  kappa in (0,1) when p ~ Uniform[0,1]. Monte Carlo estimates of this integral
  are biased low for small kappa because the density diverges at p -> 0; that is
  a sampling artifact, not an invalidity. `analytic_null_expectation` documents this.
* Every p-value fed in must be *conditionally* valid given prior tests
  (Assumption 2 of POPPER). In practice: choose the next test without looking at
  the data it will be computed on.
* Monte Carlo permutation p-values MUST use the (count + 1) / (n_perm + 1) form.
  The naive count / n_perm is anti-conservative and can report p = 0, which maps
  to an infinite e-value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

Decision = Literal["reject_null", "continue", "exhausted"]


def calibrate_p_to_e(p: float | np.ndarray, kappa: float = 0.5) -> float | np.ndarray:
    """Convert a p-value to an e-value via e = kappa * p^(kappa - 1).

    Smaller kappa is more aggressive: it rewards very small p-values more and
    penalises large ones less. kappa = 0.5 is a reasonable default; kappa = 0.3
    buys roughly 1.6x more evidence at p = 1e-2 and is worth considering when the
    number of training seeds caps how small p can get (see `min_attainable_p`).
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must lie in (0, 1), got {kappa}")
    p_arr = np.asarray(p, dtype=float)
    if np.any(p_arr < 0.0) or np.any(p_arr > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    if np.any(p_arr == 0.0):
        raise ValueError(
            "p = 0 gives an infinite e-value. Monte Carlo permutation p-values "
            "must use the (count + 1) / (n_perm + 1) correction so they are "
            "strictly positive."
        )
    return kappa * np.power(p_arr, kappa - 1.0)


def analytic_null_expectation(kappa: float = 0.5) -> float:
    """E[e] under the null, computed analytically. Always exactly 1.

    integral_0^1 kappa * p^(kappa-1) dp = kappa * [p^kappa / kappa]_0^1 = 1.
    Provided so tests assert the exact identity rather than a noisy MC estimate.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must lie in (0, 1), got {kappa}")
    return 1.0


@dataclass(frozen=True)
class FalsificationTest:
    """One falsification experiment and its outcome.

    `confounders_controlled` and `confounders_uncontrolled` are the domain-specific
    part of this framework. POPPER's relevance checker enforces its Assumption 1
    (main null implies sub-null). For SAE claims the way that assumption usually
    fails is a *nuisance variable*: a sub-hypothesis about SCR or TPP is not implied
    by a null about feature quality, because those metrics also move with the number
    of live features. A test with uncontrolled confounders breaks the implication
    and must not contribute evidence -- see `SequentialFalsifier.add`.
    """

    name: str
    null_hypothesis: str
    alt_hypothesis: str
    p_value: float
    unit_of_analysis: str
    n_units: int
    confounders_controlled: tuple[str, ...] = ()
    confounders_uncontrolled: tuple[str, ...] = ()
    notes: str = ""

    @property
    def implication_holds(self) -> bool:
        """Whether this test may contribute evidence (POPPER Assumption 1)."""
        return not self.confounders_uncontrolled


@dataclass
class SequentialFalsifier:
    """Accumulate falsification evidence against a main null hypothesis.

    Usage::

        f = SequentialFalsifier(main_hypothesis="...", alpha=0.1)
        decision = f.add(test1)
        decision = f.add(test2)   # stop as soon as decision == "reject_null"

    Stopping whenever you like is valid; that is the point of the e-value
    construction. What is *not* valid is dropping a completed test because you
    dislike its p-value -- that breaks the supermartingale. Tests excluded for a
    failed implication check are recorded but never computed on, so excluding them
    is a design-time decision, not a data-dependent one.
    """

    main_hypothesis: str
    alpha: float = 0.1
    kappa: float = 0.5
    max_tests: int = 5
    tests: list[FalsificationTest] = field(default_factory=list)
    excluded: list[FalsificationTest] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")

    @property
    def threshold(self) -> float:
        """Reject the null once aggregated evidence reaches 1 / alpha."""
        return 1.0 / self.alpha

    @property
    def e_values(self) -> list[float]:
        return [float(calibrate_p_to_e(t.p_value, self.kappa)) for t in self.tests]

    @property
    def cumulative_e(self) -> float:
        """E = prod_i e_i, the aggregated evidence."""
        e = 1.0
        for value in self.e_values:
            e *= value
        return e

    def add(self, test: FalsificationTest) -> Decision:
        """Record one falsification experiment and return the current decision."""
        if not test.implication_holds:
            self.excluded.append(test)
            return self.decision()
        self.tests.append(test)
        return self.decision()

    def decision(self) -> Decision:
        if self.cumulative_e >= self.threshold:
            return "reject_null"
        if len(self.tests) >= self.max_tests:
            return "exhausted"
        return "continue"

    def validated(self) -> bool:
        """True iff the main hypothesis is validated (its null is rejected)."""
        return self.cumulative_e >= self.threshold

    def evidence_still_needed(self) -> float:
        """Multiplicative evidence still required to reach the threshold."""
        return max(1.0, self.threshold / max(self.cumulative_e, 1e-300))

    def report(self) -> str:
        lines = [
            f"Main hypothesis: {self.main_hypothesis}",
            f"alpha = {self.alpha}  (reject null at E >= {self.threshold:.1f})"
            f"   kappa = {self.kappa}",
            "",
        ]
        running = 1.0
        for i, (test, e) in enumerate(zip(self.tests, self.e_values), start=1):
            running *= e
            lines.append(
                f"  {i}. {test.name}\n"
                f"     p = {test.p_value:.3e}   e = {e:.2f}   cumulative E = {running:.2f}\n"
                f"     unit of analysis: {test.unit_of_analysis} (n = {test.n_units})"
            )
            if test.confounders_controlled:
                lines.append(
                    f"     controlled: {', '.join(test.confounders_controlled)}"
                )
        for test in self.excluded:
            lines.append(
                f"  [EXCLUDED] {test.name}\n"
                f"     implication fails; uncontrolled: "
                f"{', '.join(test.confounders_uncontrolled)}"
            )
        decision = self.decision()
        lines += [
            "",
            f"Aggregated evidence E = {self.cumulative_e:.3f}",
            f"Decision: {decision}",
        ]
        if decision != "reject_null":
            lines.append(
                f"Not validated. Would need {self.evidence_still_needed():.1f}x "
                f"more evidence to reach threshold."
            )
        return "\n".join(lines)


def min_attainable_p(n_per_group: int, two_sided: bool = False) -> float:
    """Smallest p-value an exact two-group permutation test can produce.

    This is the binding power constraint when the unit of analysis is a training
    run: with n seeds per group there are only C(2n, n) label assignments, so p
    cannot go below their reciprocal no matter how large the effect.
    """
    from math import comb

    if n_per_group < 1:
        raise ValueError("n_per_group must be >= 1")
    n_assignments = comb(2 * n_per_group, n_per_group)
    # A two-sided test cannot go below 2/N: an assignment and its complement always
    # tie on |statistic|, so the count is never 1. A one-sided test can reach 1/N,
    # where only the observed assignment matches or exceeds itself. This was
    # previously inverted, making every one-sided floor 2x too pessimistic.
    return (2.0 if two_sided else 1.0) / n_assignments


def seeds_required(alpha: float = 0.1, kappa: float = 0.5, n_tests: int = 1) -> int:
    """Seeds per group needed for `n_tests` maximally-significant tests to validate.

    Answers the practical question "how many training runs must I budget?".
    Assumes every test attains its minimum p-value, so this is a floor: real
    effects will need more.
    """
    threshold = 1.0 / alpha
    for n in range(2, 65):
        e = calibrate_p_to_e(min_attainable_p(n), kappa)
        if float(e) ** n_tests >= threshold:
            return n
    raise ValueError("No feasible seed count below 64; loosen alpha or kappa.")
