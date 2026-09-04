"""Contracts for the permutation tests that produce the p-values.

The framework's validity rests entirely on these: a wrong p-value here is
laundered into an authoritative-looking e-value downstream. Several assertions
guard specific bugs called out in the module's own docstrings.
"""

from math import comb

import numpy as np
import pytest

from falsification.permutation import (
    min_p_floor,
    monotone_trend_test,
    paired_token_test,
    seed_permutation_test,
    subsample_null_test,
)


# --- seed_permutation_test: the only architecture-level instrument ---------


def test_exact_enumeration_for_small_designs():
    res = seed_permutation_test([3.0, 4.0, 5.0], [1.0, 2.0, 2.5])
    assert res["exact"] is True
    assert res["n_draws"] == comb(6, 3)
    assert res["unit_of_analysis"] == "training run (seed)"
    assert res["n_units"] == 6


def test_perfect_separation_attains_one_over_the_assignment_count():
    """Maximal separation leaves only the observed assignment at or above itself."""
    res = seed_permutation_test([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
    assert res["p_value"] == pytest.approx(1.0 / comb(6, 3))


def test_observed_assignment_is_always_counted():
    """The ULP bug guard.

    The docstring records a real failure: computing the observed statistic as
    a.mean() - b.mean() while the null statistics come from a different but
    algebraically identical expression let the observed assignment fail its own
    >= comparison, giving count = 0 and p = 0 -- an infinite e-value, i.e.
    automatic validation. p must never be zero for any input.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.normal(size=4) * rng.choice([1e-8, 1.0, 1e8])
        b = rng.normal(size=4) * rng.choice([1e-8, 1.0, 1e8])
        for alt in ("greater", "less", "two-sided"):
            p = seed_permutation_test(a, b, alternative=alt)["p_value"]
            assert p > 0.0
            assert p <= 1.0


def test_identical_groups_give_no_evidence():
    res = seed_permutation_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert res["p_value"] > 0.5


def test_reversed_effect_direction_is_not_significant():
    small = seed_permutation_test([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], alternative="greater")
    assert small["p_value"] == pytest.approx(1.0)


def test_alternative_less_mirrors_alternative_greater():
    a, b = [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]
    assert seed_permutation_test(a, b, alternative="less")["p_value"] == pytest.approx(
        1.0 / comb(6, 3)
    )


def test_two_sided_counts_both_tails():
    a, b = [10.0, 11.0, 12.0], [1.0, 2.0, 3.0]
    one = seed_permutation_test(a, b, alternative="greater")["p_value"]
    two = seed_permutation_test(a, b, alternative="two-sided")["p_value"]
    assert two == pytest.approx(2.0 * one)


def test_a_single_run_per_group_is_refused():
    """CLAUDE.md: a claim about an architecture cannot rest on one training run."""
    with pytest.raises(ValueError, match="at least 2 seeds"):
        seed_permutation_test([1.0], [2.0])


def test_non_1d_input_is_refused():
    with pytest.raises(ValueError, match="1-D"):
        seed_permutation_test([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])


def test_large_designs_fall_back_to_monte_carlo_and_stay_positive():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=12), rng.normal(size=12)
    res = seed_permutation_test(a, b, n_perm=2000)
    assert res["exact"] is False
    assert res["n_draws"] == 2000
    # CLAUDE.md: Monte Carlo p-values must use (count + 1) / (n_perm + 1).
    assert res["p_value"] >= 1.0 / 2001
    assert res["p_value"] > 0.0


def test_monte_carlo_floor_uses_the_plus_one_correction():
    """Even a null that is never exceeded cannot report p = 0."""
    a = np.arange(12, dtype=float) + 1000.0
    b = np.arange(12, dtype=float)
    res = seed_permutation_test(a, b, n_perm=500)
    assert res["exact"] is False
    assert res["p_value"] == pytest.approx(1.0 / 501)


def test_p_floor_is_reported_for_every_result():
    """An underpowered design cannot be rescued by its result, so always report it."""
    res = seed_permutation_test([3.0, 4.0, 5.0], [1.0, 2.0, 2.5])
    assert "p_floor" in res and res["p_floor"] > 0.0


def test_p_value_never_falls_below_its_own_reported_floor():
    """Regression: min_p_floor had one-sided and two-sided swapped until 2026-09-02,
    so a one-sided test could return a p-value at half its own reported floor."""
    for alt in ("greater", "less", "two-sided"):
        a, b = ([10.0, 11.0, 12.0], [1.0, 2.0, 3.0]) if alt != "less" else (
            [1.0, 2.0, 3.0], [10.0, 11.0, 12.0])
        res = seed_permutation_test(a, b, alternative=alt)
        assert res["p_value"] >= res["p_floor"], alt


def test_one_sided_floor_is_one_over_the_assignment_count():
    assert min_p_floor(3, 3, "greater") == pytest.approx(1.0 / comb(6, 3))
    assert min_p_floor(6, 6, "greater") == pytest.approx(1.0 / comb(12, 6))


def test_two_sided_floor_is_twice_the_one_sided_floor():
    """An assignment and its complement tie on |stat|, so the count is never 1."""
    for n in (3, 4, 6):
        assert min_p_floor(n, n, "two-sided") == pytest.approx(
            2.0 * min_p_floor(n, n, "greater")
        )


def test_floors_are_exactly_attained_by_maximal_separation():
    """The floor is a floor *and* it is reachable -- not merely a loose bound."""
    for n in (3, 4):
        a = [100.0 + i for i in range(n)]
        b = [float(i) for i in range(n)]
        for alt in ("greater", "two-sided"):
            res = seed_permutation_test(a, b, alternative=alt)
            assert res["p_value"] == pytest.approx(res["p_floor"]), (n, alt)


def test_min_p_floor_is_self_consistent_with_seed_permutation_test():
    res = seed_permutation_test([3.0, 4.0, 5.0], [1.0, 2.0, 2.5])
    assert res["p_floor"] == pytest.approx(min_p_floor(3, 3, "greater"))


# --- subsample_null_test ---------------------------------------------------


def test_subsample_null_requires_enough_draws():
    with pytest.raises(ValueError, match=">= 20 subsamples"):
        subsample_null_test(1.0, [0.1] * 19)


def test_subsample_null_uses_the_plus_one_correction():
    res = subsample_null_test(10.0, list(np.linspace(0.0, 1.0, 50)))
    assert res["p_value"] == pytest.approx(1.0 / 51)
    assert res["p_floor"] == pytest.approx(1.0 / 51)
    assert res["p_value"] > 0.0


def test_subsample_null_reports_the_null_distribution():
    draws = list(np.linspace(0.0, 1.0, 40))
    res = subsample_null_test(0.5, draws)
    assert res["null_mean"] == pytest.approx(float(np.mean(draws)))
    assert res["null_std"] == pytest.approx(float(np.std(draws, ddof=1)))
    assert res["unit_of_analysis"] == "random size-matched sub-dictionary"


def test_subsample_null_is_unconvinced_by_a_typical_score():
    draws = list(np.linspace(0.0, 1.0, 50))
    assert subsample_null_test(0.5, draws)["p_value"] > 0.1


# --- paired_token_test: scope enforcement ----------------------------------


def test_token_test_refuses_architecture_level_use_by_default():
    """The most common inferential error in SAE papers, refused at the API."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="seed_permutation_test"):
        paired_token_test(rng.normal(size=50), rng.normal(size=50))


def test_token_test_runs_once_the_narrower_scope_is_acknowledged():
    rng = np.random.default_rng(0)
    res = paired_token_test(
        rng.normal(size=200) + 0.5,
        rng.normal(size=200),
        n_boot=500,
        acknowledge_checkpoint_scope=True,
    )
    assert res["p_value"] > 0.0
    assert "CHECKPOINT-LEVEL CLAIM ONLY" in res["unit_of_analysis"]


def test_token_test_requires_aligned_pairs():
    with pytest.raises(ValueError, match="aligned"):
        paired_token_test(
            np.zeros(10), np.zeros(11), acknowledge_checkpoint_scope=True
        )


def test_token_test_p_value_is_never_zero():
    a = np.arange(500, dtype=float) + 1e6
    res = paired_token_test(
        a, np.arange(500, dtype=float), n_boot=300, acknowledge_checkpoint_scope=True
    )
    assert res["p_value"] == pytest.approx(1.0 / 301)


# --- monotone_trend_test ---------------------------------------------------


def test_perfect_monotone_trend_attains_one_over_k_factorial():
    """The floor that makes the beta dose-response non-validating on its own."""
    res = monotone_trend_test([1.0, 2.0, 3.0, 4.0], [10.0, 8.0, 6.0, 4.0])
    assert res["p_value"] == pytest.approx(1.0 / 24)
    assert res["p_floor"] == pytest.approx(1.0 / 24)
    assert res["n_draws"] == 24


def test_trend_statistic_reads_the_whole_ordering_not_just_the_endpoints():
    """Guards the telescoping bug recorded in the docstring.

    A statistic that summed successive differences collapses to (last - first),
    so it cannot tell a perfectly monotone sequence from one with the same
    endpoints and a scrambled interior. The Spearman statistic can.
    """
    monotone = monotone_trend_test([1.0, 2.0, 3.0, 4.0], [10.0, 8.0, 6.0, 4.0])
    scrambled = monotone_trend_test([1.0, 2.0, 3.0, 4.0], [10.0, 6.0, 8.0, 4.0])
    assert monotone["p_value"] < scrambled["p_value"]


def test_trend_test_needs_at_least_three_conditions():
    with pytest.raises(ValueError, match="at least 3"):
        monotone_trend_test([1.0, 2.0], [2.0, 1.0])


def test_trend_test_refuses_impractical_enumeration():
    with pytest.raises(ValueError, match="beyond 9"):
        monotone_trend_test(list(range(10)), list(range(10)))


def test_trend_test_requires_aligned_arrays():
    with pytest.raises(ValueError, match="align"):
        monotone_trend_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_increasing_direction_is_supported():
    res = monotone_trend_test([1.0, 2.0, 3.0, 4.0], [4.0, 6.0, 8.0, 10.0], direction="increasing")
    assert res["p_value"] == pytest.approx(1.0 / 24)


def test_a_flat_response_is_not_a_trend():
    res = monotone_trend_test([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0])
    assert res["p_value"] == pytest.approx(1.0)
