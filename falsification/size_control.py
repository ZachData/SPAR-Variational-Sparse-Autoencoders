"""Is a metric advantage explained by dictionary size alone? (E4)

The preprint's vSAE outscored its baseline on SCR and TPP while holding 1474 live
features against 7379. Those metrics reward *selective* ablation, and selectivity
gets easier as a dictionary shrinks, so the comparison is confounded and the
result carries no evidence about feature quality.

The obvious control -- restrict the baseline to a random 1474 of its 8192 features
and compare -- is worse than no control at all. A random 18% subset of a
dictionary trained to work as a whole reconstructs badly, while the vSAE's 1474
features were learned together. The vSAE beats that null trivially, so the test is
rigged in its favour and would falsely confirm the hypothesis it exists to check.

What this module does instead: measure the metric as a FUNCTION of dictionary
size for the baseline, then ask where the vSAE's score falls on that curve.

    - `top_usage`   the baseline's N most-used features: its best foot forward,
                    and the strongest thing size alone can buy. This is the
                    conservative comparison and the one that decides the claim.
    - `random`      a random N: the weak lower bound above. Reported only to
                    bracket the answer, never on its own.

If the vSAE's score sits on or below the `top_usage` curve at its own live count,
the advantage is explained by size. If it sits clearly above, something other than
size is going on.

The scorer is injected, so everything here is testable without torch or SAEBench.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

Strategy = Literal["top_usage", "random"]

# A scorer maps a set of kept feature indices to a metric value (e.g. an SCR score).
# In production this closes over a trained SAE and calls SAEBench; in tests it is
# a synthetic function with known ground truth.
Scorer = Callable[[np.ndarray], float]


def select_features(
    usage_counts: Sequence[float],
    n_features: int,
    strategy: Strategy,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Choose which `n_features` dictionary entries to keep."""
    usage = np.asarray(usage_counts, dtype=float)
    total = len(usage)
    if not 1 <= n_features <= total:
        raise ValueError(f"n_features must be in [1, {total}], got {n_features}")

    if strategy == "top_usage":
        # argsort is ascending; take the tail and sort for a deterministic result.
        return np.sort(np.argsort(usage, kind="stable")[-n_features:])
    if strategy == "random":
        if rng is None:
            raise ValueError("strategy='random' requires an rng")
        # Only ever sample among features that are actually alive: a dead feature
        # contributes nothing, so including them would understate the subset size.
        alive = np.flatnonzero(usage > 0)
        if len(alive) < n_features:
            raise ValueError(
                f"only {len(alive)} live features available, cannot draw {n_features}"
            )
        return np.sort(rng.choice(alive, size=n_features, replace=False))
    raise ValueError(f"unknown strategy: {strategy}")


@dataclass
class SizeResponsePoint:
    n_features: int
    strategy: Strategy
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0


def size_response_curve(
    usage_counts: Sequence[float],
    scorer: Scorer,
    n_grid: Sequence[int],
    strategies: Sequence[Strategy] = ("top_usage", "random"),
    n_draws: int = 20,
    seed: int = 0,
) -> list[SizeResponsePoint]:
    """Metric as a function of dictionary size, for each selection strategy.

    `top_usage` is deterministic, so it is evaluated once per grid point regardless
    of `n_draws`; `random` is averaged over `n_draws` draws.
    """
    rng = np.random.default_rng(seed)
    points: list[SizeResponsePoint] = []
    for strategy in strategies:
        for n in n_grid:
            draws = 1 if strategy == "top_usage" else n_draws
            scores = [
                scorer(select_features(usage_counts, n, strategy, rng))
                for _ in range(draws)
            ]
            points.append(SizeResponsePoint(n, strategy, scores))
    return points


@dataclass
class SizeVerdict:
    observed_score: float
    observed_n: int
    reference_score: float
    reference_strategy: Strategy
    explained_by_size: bool
    margin: float

    def __str__(self) -> str:
        head = (
            "EXPLAINED BY SIZE"
            if self.explained_by_size
            else "NOT explained by size"
        )
        return (
            f"{head}: score {self.observed_score:.4f} at n={self.observed_n} vs "
            f"{self.reference_strategy} baseline {self.reference_score:.4f} "
            f"(margin {self.margin:+.4f})"
        )


def interpolate_curve(
    points: Sequence[SizeResponsePoint], n_features: int, strategy: Strategy
) -> float:
    """Baseline score at `n_features`, linearly interpolated along the curve."""
    selected = sorted(
        (p for p in points if p.strategy == strategy), key=lambda p: p.n_features
    )
    if not selected:
        raise ValueError(f"no curve points for strategy {strategy!r}")
    xs = np.array([p.n_features for p in selected], dtype=float)
    ys = np.array([p.mean for p in selected], dtype=float)
    return float(np.interp(float(n_features), xs, ys))


def verdict(
    observed_score: float,
    observed_n: int,
    points: Sequence[SizeResponsePoint],
    reference_strategy: Strategy = "top_usage",
) -> SizeVerdict:
    """Decide whether an observed advantage is attributable to dictionary size.

    Compared against `top_usage` by default: the baseline's own best N features are
    the strongest size-matched comparison available, so clearing that bar is the
    demanding test. Beating only the `random` curve means nothing.
    """
    reference = interpolate_curve(points, observed_n, reference_strategy)
    margin = observed_score - reference
    return SizeVerdict(
        observed_score=observed_score,
        observed_n=observed_n,
        reference_score=reference,
        reference_strategy=reference_strategy,
        explained_by_size=margin <= 0.0,
        margin=margin,
    )


def mask_dictionary(weights: dict, keep_indices: np.ndarray, dict_axis: dict) -> dict:
    """Zero out every dictionary entry outside `keep_indices`.

    `weights` maps a name to an array; `dict_axis` maps the same name to the axis
    indexing dictionary features (encoder rows and decoder columns differ, which is
    an easy and silent mistake). Returns copies -- the originals are untouched.
    """
    masked = {}
    for name, array in weights.items():
        if name not in dict_axis:
            raise ValueError(f"no dictionary axis specified for weight {name!r}")
        axis = dict_axis[name]
        size = array.shape[axis]
        keep = np.zeros(size, dtype=bool)
        keep[keep_indices] = True
        shape = [1] * array.ndim
        shape[axis] = size
        masked[name] = array * keep.reshape(shape)
    return masked
