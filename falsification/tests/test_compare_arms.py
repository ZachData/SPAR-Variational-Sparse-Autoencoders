"""Contracts for the general two-arm comparison.

The two things here that can silently corrupt a result are seed alignment and the
floor. Both are cheap to get wrong and expensive to notice: a misaligned pair still
prints a plausible table, and a p reported against the design floor when Monte
Carlo is the binding one understates how much the analysis, not the experiment,
limited the evidence.
"""

import json
from pathlib import Path

import pytest

from falsification.compare_arms import _sigma, arm_values, effective_floor


def _write_run(root: Path, seed: int, fve: float):
    d = root / f"seed{seed}" / "ckpt"
    d.mkdir(parents=True)
    (d / "evaluation_results.json").write_text(json.dumps({
        "frac_variance_explained": fve,
        "frac_recovered": fve + 0.05,
        "frac_alive": 1.0,
    }))


def test_values_are_ordered_by_seed_not_by_glob(tmp_path):
    """Glob order is filesystem order: seed10 sorts before seed2 as a string.

    Appending in that order pairs seed 10 of one arm against seed 2 of the other,
    which is still a valid permutation test -- of the wrong thing.
    """
    root = tmp_path / "arm"
    for seed, fve in [(2, 0.2), (10, 0.9), (1, 0.1)]:
        _write_run(root, seed, fve)

    vals, seeds = arm_values(root)
    assert seeds == [1, 2, 10]
    assert vals["frac_variance_explained"] == {1: 0.1, 2: 0.2, 10: 0.9}


def test_missing_seed_does_not_shift_the_other_values(tmp_path):
    """An arm with a gap must keep its remaining values on their own seeds."""
    root = tmp_path / "arm"
    for seed, fve in [(1, 0.1), (3, 0.3)]:
        _write_run(root, seed, fve)

    vals, seeds = arm_values(root)
    assert seeds == [1, 3]
    assert vals["frac_variance_explained"][3] == pytest.approx(0.3)


def test_exact_result_uses_the_design_floor():
    res = {"exact": True, "p_floor": 2 / 924, "n_draws": 924}
    assert effective_floor(res) == pytest.approx(2 / 924)


def test_monte_carlo_floor_binds_when_it_is_the_larger():
    """13v13 at 4M draws: the MC floor 2.5e-07 exceeds the design's 1.9e-07."""
    res = {"exact": False, "p_floor": 2 / 10_400_600, "n_draws": 4_000_000}
    assert effective_floor(res) == pytest.approx(2.5e-07, rel=1e-6)


def test_design_floor_binds_when_permutations_are_plentiful():
    """A huge n_perm cannot buy evidence the design cannot express."""
    res = {"exact": False, "p_floor": 2 / 10_400_600, "n_draws": 10**9}
    assert effective_floor(res) == pytest.approx(2 / 10_400_600)


@pytest.mark.parametrize("p, expected", [
    (2.5e-07, 5.16),    # 13v13 Monte Carlo floor at n_perm = 4e6
    (2 / 924, 3.07),    # 6v6 exact two-sided floor
    (0.05, 1.96),
])
def test_sigma_matches_the_figures_the_results_are_written_in(p, expected):
    pytest.importorskip("scipy")
    assert _sigma(p) == pytest.approx(expected, abs=0.005)
