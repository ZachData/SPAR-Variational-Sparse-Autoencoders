"""The three simulation findings that decide the experimental design.

PROJECT.md states these as settled facts and pre-registers the protocol on top of
them ("What the simulation settled"). They are locked in here so a change to the
framework cannot silently invalidate the design.

Trial counts are kept small: these assert the qualitative findings, not the exact
rates in the tables, which are quoted from full-length runs of `simulate.main()`.
"""

import numpy as np
import pytest

from falsification.simulate import (
    _correlated_metrics,
    power_curve,
    simulate_trial,
    type_i_error,
)


# --- finding 1: reusing runs across tests stays valid ----------------------


def test_type_i_error_is_controlled_with_independent_tests():
    rate, se = type_i_error(n_per_group=5, n_metrics=2, rho=0.0, n_trials=400, alpha=0.1)
    assert rate <= 0.1 + 3 * se


def test_type_i_error_is_controlled_even_with_fully_redundant_tests():
    """The worst case POPPER's Assumption 2 warns about: every test on the same runs."""
    rate, se = type_i_error(n_per_group=5, n_metrics=5, rho=1.0, n_trials=400, alpha=0.1)
    assert rate <= 0.1 + 3 * se


def test_redundant_tests_spend_more_of_the_error_budget():
    """Design rule: prefer few genuinely different tests over many correlated ones."""
    independent, _ = type_i_error(5, 2, rho=0.0, n_trials=600, alpha=0.1, seed=1)
    redundant, _ = type_i_error(5, 5, rho=1.0, n_trials=600, alpha=0.1, seed=1)
    assert redundant > independent


# --- finding 2: peeking at test order breaks control -----------------------


def test_cherry_picking_test_order_inflates_type_i_error():
    """The optional-stopping guarantee covers WHEN you stop, never WHICH test you pick."""
    fixed, _ = type_i_error(
        5, 10, rho=0.0, n_trials=400, alpha=0.1, selection="prespecified", seed=3
    )
    peeked, _ = type_i_error(
        5, 10, rho=0.0, n_trials=400, alpha=0.1, selection="cherry_pick", seed=3
    )
    assert peeked > fixed


# --- finding 3: power is the binding constraint ----------------------------


def test_power_increases_with_seeds():
    results = power_curve(1.5, n_per_group_values=(4, 8), n_trials=200, kappa=0.3)
    assert results[1].power > results[0].power


def test_power_increases_with_effect_size():
    weak = power_curve(0.5, n_per_group_values=(6,), n_trials=200, kappa=0.3)[0].power
    strong = power_curve(2.0, n_per_group_values=(6,), n_trials=200, kappa=0.3)[0].power
    assert strong > weak


def test_a_one_sd_effect_is_not_reliably_detectable_even_at_ten_seeds():
    """PROJECT.md's headline warning, and the reason subtle claims are out of reach."""
    power = power_curve(1.0, n_per_group_values=(10,), n_trials=300, kappa=0.3)[0].power
    assert power < 0.8


def test_no_effect_yields_power_at_or_below_alpha():
    power = power_curve(0.0, n_per_group_values=(6,), n_trials=400, kappa=0.3)[0].power
    assert power <= 0.15


# --- the correlated-metric generator --------------------------------------


def test_rho_zero_gives_uncorrelated_metrics():
    rng = np.random.default_rng(0)
    m = _correlated_metrics(4000, 2, rho=0.0, rng=rng)
    assert abs(np.corrcoef(m[:, 0], m[:, 1])[0, 1]) < 0.1


def test_rho_one_duplicates_a_single_measurement():
    rng = np.random.default_rng(0)
    m = _correlated_metrics(500, 3, rho=1.0, rng=rng)
    assert np.allclose(m[:, 0], m[:, 1])


def test_generated_metrics_carry_no_group_effect():
    """Every null is true by construction, which is what makes Type-I measurable."""
    rng = np.random.default_rng(0)
    m = _correlated_metrics(4000, 1, rho=0.5, rng=rng)
    assert abs(m[:2000].mean() - m[2000:].mean()) < 0.15


def test_a_single_trial_returns_a_boolean_verdict():
    rng = np.random.default_rng(0)
    assert isinstance(bool(simulate_trial(5, 2, 0.0, rng)), bool)
