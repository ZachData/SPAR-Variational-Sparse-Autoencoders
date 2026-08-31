"""Permutation tests that produce conditionally valid p-values for SAE claims.

The hard part of applying sequential falsification to interpretability is not the
e-value algebra -- it is producing a p-value that means anything. This module
implements the three sources of legitimate randomness available when comparing
sparse autoencoders, and is explicit about which claim each one can support.

    1. `seed_permutation_test` -- the unit of analysis is a TRAINING RUN.
       This is the only test that supports a claim about an ARCHITECTURE.
    2. `subsample_null_test` -- the unit is a random size-matched sub-dictionary.
       Controls the live-feature-count confounder without retraining anything.
    3. `paired_token_test` -- the unit is an evaluation token.
       Supports claims about TWO SPECIFIC CHECKPOINTS, never about architectures.

The distinction between (1) and (3) is the most common inferential error in SAE
papers. A difference that is overwhelming across a million eval tokens can vanish
entirely across five training seeds, because token-level tests estimate evaluation
noise while the claim is about training. `paired_token_test` therefore refuses to
be used for architecture-level claims unless you explicitly acknowledge the scope.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Literal, Sequence

import numpy as np

Alternative = Literal["greater", "less", "two-sided"]

# Above this many label assignments, enumerate by Monte Carlo instead of exactly.
_EXACT_ENUMERATION_LIMIT = 200_000


def _one_sided_count(observed: float, null_stats: np.ndarray, alternative: Alternative) -> int:
    if alternative == "greater":
        return int(np.sum(null_stats >= observed))
    if alternative == "less":
        return int(np.sum(null_stats <= observed))
    return int(np.sum(np.abs(null_stats) >= abs(observed)))


def seed_permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alternative: Alternative = "greater",
    n_perm: int = 100_000,
    seed: int = 0,
) -> dict:
    """Exact-when-possible permutation test over independent training runs.

    `group_a` and `group_b` hold one metric value per training seed, for two
    architectures trained with otherwise identical configuration. Under the null
    that the architecture does not affect the metric, the group labels are
    exchangeable, so permuting them gives the exact null distribution of the mean
    difference.

    Returns a dict with the p-value, whether enumeration was exact, and
    `p_floor` -- the smallest p-value this design could possibly produce. Always
    report `p_floor`: if it exceeds what you need, the experiment was
    underpowered by construction and no result can rescue it.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("groups must be 1-D sequences of per-seed metric values")
    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "need at least 2 seeds per group; a claim about an architecture "
            "cannot rest on a single training run"
        )

    pooled = np.concatenate([a, b])
    n_a, n_total = len(a), len(pooled)
    observed = a.mean() - b.mean()

    n_assignments = comb(n_total, n_a)
    exact = n_assignments <= _EXACT_ENUMERATION_LIMIT

    if exact:
        null_stats = np.empty(n_assignments, dtype=float)
        total_sum = pooled.sum()
        for i, idx in enumerate(combinations(range(n_total), n_a)):
            sum_a = pooled[list(idx)].sum()
            null_stats[i] = sum_a / n_a - (total_sum - sum_a) / (n_total - n_a)
        count = _one_sided_count(observed, null_stats, alternative)
        # Exact enumeration includes the observed assignment, so this is already valid.
        p_value = count / n_assignments
        n_draws = n_assignments
    else:
        rng = np.random.default_rng(seed)
        null_stats = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = rng.permutation(pooled)
            null_stats[i] = perm[:n_a].mean() - perm[n_a:].mean()
        count = _one_sided_count(observed, null_stats, alternative)
        # (count + 1) / (n + 1) keeps the Monte Carlo p-value valid and non-zero.
        p_value = (count + 1) / (n_perm + 1)
        n_draws = n_perm

    return {
        "p_value": float(p_value),
        "observed_difference": float(observed),
        "exact": exact,
        "n_draws": int(n_draws),
        "p_floor": float(min_p_floor(len(a), len(b), alternative)),
        "unit_of_analysis": "training run (seed)",
        "n_units": int(n_total),
    }


def min_p_floor(n_a: int, n_b: int, alternative: Alternative = "greater") -> float:
    """Smallest attainable p-value for an exact permutation test of these group sizes."""
    n_assignments = comb(n_a + n_b, n_a)
    if alternative == "two-sided":
        return 1.0 / n_assignments
    return 1.0 / max(n_assignments // 2, 1)


def subsample_null_test(
    observed_score: float,
    baseline_scores_subsampled: Sequence[float],
    alternative: Alternative = "greater",
) -> dict:
    """Test a score against a size-matched random-sub-dictionary null.

    This is the control the preprint was missing. Metrics such as SCR and TPP
    reward *selective* ablation, and selectivity is easier when fewer features are
    live -- so a variational SAE with 18% of its dictionary alive can outscore a
    baseline with 90% alive for reasons that have nothing to do with feature
    quality. Repeatedly restricting the baseline dictionary to a random subset of
    the same size as the comparison model's live count, and recomputing the metric,
    gives the null distribution of "what does a random dictionary of this size
    score?".

    `baseline_scores_subsampled` are those recomputed scores, one per random subset.
    Uses the (count + 1) / (n + 1) correction, so p is never zero.
    """
    null_stats = np.asarray(baseline_scores_subsampled, dtype=float)
    if null_stats.size < 20:
        raise ValueError(
            f"need >= 20 subsamples for a usable null distribution, got {null_stats.size}"
        )
    count = _one_sided_count(observed_score, null_stats, alternative)
    p_value = (count + 1) / (null_stats.size + 1)
    return {
        "p_value": float(p_value),
        "observed_score": float(observed_score),
        "null_mean": float(null_stats.mean()),
        "null_std": float(null_stats.std(ddof=1)),
        "n_draws": int(null_stats.size),
        "p_floor": 1.0 / (null_stats.size + 1),
        "unit_of_analysis": "random size-matched sub-dictionary",
        "n_units": int(null_stats.size),
    }


def paired_token_test(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    alternative: Alternative = "greater",
    n_boot: int = 10_000,
    seed: int = 0,
    acknowledge_checkpoint_scope: bool = False,
) -> dict:
    """Paired bootstrap over evaluation tokens. Scope: TWO CHECKPOINTS ONLY.

    Valid for "this trained vSAE reconstructs held-out tokens worse than this
    trained SAE". NOT valid for "vSAEs reconstruct worse than SAEs" -- that is a
    claim about architectures, whose unit of analysis is the training run. With a
    million eval tokens this test will report p ~ 1e-300 for differences that are
    pure seed noise, which is exactly how underpowered architecture comparisons get
    published as decisive.

    Set `acknowledge_checkpoint_scope=True` to confirm the narrower claim is what
    you intend.
    """
    if not acknowledge_checkpoint_scope:
        raise ValueError(
            "paired_token_test measures evaluation noise between two specific "
            "checkpoints, not variation between architectures. Use "
            "seed_permutation_test for architecture-level claims, or pass "
            "acknowledge_checkpoint_scope=True if a checkpoint-level claim is "
            "genuinely what you mean."
        )
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired test requires equal-length, aligned per-token metrics")

    diff = a - b
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    # Null: centre the differences, then bootstrap.
    centred = diff - observed
    boot = np.empty(n_boot, dtype=float)
    n = len(diff)
    for i in range(n_boot):
        boot[i] = centred[rng.integers(0, n, size=n)].mean()
    count = _one_sided_count(observed, boot, alternative)
    return {
        "p_value": float((count + 1) / (n_boot + 1)),
        "observed_difference": float(observed),
        "n_draws": int(n_boot),
        "p_floor": 1.0 / (n_boot + 1),
        "unit_of_analysis": "evaluation token (CHECKPOINT-LEVEL CLAIM ONLY)",
        "n_units": int(n),
    }


def monotone_trend_test(
    condition_values: Sequence[float],
    outcome_values: Sequence[float],
    direction: Literal["decreasing", "increasing"] = "decreasing",
) -> dict:
    """Exact permutation test for a monotone dose-response across conditions.

    For a sweep with ONE run per condition (the common case when each condition
    costs a training run), the only exchangeable quantity under the null "the
    condition does not affect the outcome" is the assignment of outcomes to
    conditions. Enumerating all k! assignments gives an exact p-value.

    The test statistic is Spearman rank correlation between condition and outcome.
    A statistic must depend on the full ordering: an earlier version here summed
    successive differences, which telescopes to (last - first) and therefore reads
    only the endpoints, making a perfect monotone trend look twice as likely under
    the null as it is.

    The floor is harsh and worth internalising: with k conditions the smallest
    attainable p is 1/k!, so a 4-point sweep cannot go below 1/24 ~ 0.042 no
    matter how clean the trend. Replication across seeds, not more sweep points,
    is what buys evidence.
    """
    from itertools import permutations as iter_permutations

    cond = np.asarray(condition_values, dtype=float)
    out = np.asarray(outcome_values, dtype=float)
    if cond.shape != out.shape:
        raise ValueError("condition and outcome arrays must align")
    k = len(cond)
    if k < 3:
        raise ValueError("need at least 3 conditions for a trend test")
    if k > 9:
        raise ValueError("exact enumeration beyond 9 conditions is impractical")

    sign = -1.0 if direction == "decreasing" else 1.0
    cond_ranks = _rankdata(cond)
    cond_centred = cond_ranks - cond_ranks.mean()

    def spearman_like(values: np.ndarray) -> float:
        """Signed rank covariance; monotone in Spearman rho for fixed rank sets."""
        v = _rankdata(values)
        return float(sign * np.dot(cond_centred, v - v.mean()))

    observed = spearman_like(out)
    null_stats = np.array(
        [spearman_like(np.asarray(perm)) for perm in iter_permutations(out)]
    )
    # Exact enumeration includes the observed assignment, so this is valid as-is.
    count = int(np.sum(null_stats >= observed - 1e-12))
    return {
        "p_value": count / null_stats.size,
        "observed_statistic": observed,
        "exact": True,
        "n_draws": int(null_stats.size),
        "p_floor": 1.0 / null_stats.size,
        "unit_of_analysis": "sweep condition (1 run each)",
        "n_units": k,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (equivalent to scipy.stats.rankdata)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average ranks within tie groups.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks
