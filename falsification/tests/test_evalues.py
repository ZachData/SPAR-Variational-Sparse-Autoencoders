"""Contracts for the e-value calibration and sequential aggregation.

Every assertion here restates a property claimed in `falsification/evalues.py`'s
docstrings, in PROJECT.md, or in CLAUDE.md's "Conventions" section. If one of
these fails, the corresponding written claim is wrong.
"""

from math import comb

import numpy as np
import pytest

from falsification.evalues import (
    FalsificationTest,
    SequentialFalsifier,
    analytic_null_expectation,
    calibrate_p_to_e,
    min_attainable_p,
    seeds_required,
)


def _test(p, **kw):
    """A FalsificationTest with sane defaults, so cases state only what matters."""
    defaults = dict(
        name="t",
        null_hypothesis="no effect",
        alt_hypothesis="effect",
        p_value=p,
        unit_of_analysis="training run (seed)",
        n_units=12,
        confounders_controlled=("k", "dict_size"),
    )
    defaults.update(kw)
    return FalsificationTest(**defaults)


# --- calibration -----------------------------------------------------------


def test_calibrator_matches_the_documented_formula():
    for kappa in (0.2, 0.3, 0.5, 0.8):
        for p in (1e-4, 1e-2, 0.5, 1.0):
            assert calibrate_p_to_e(p, kappa) == pytest.approx(kappa * p ** (kappa - 1))


def test_null_expectation_is_exactly_one():
    # E[e] = 1 for any kappa in (0,1); the docstring promises the exact identity,
    # not a Monte Carlo approximation.
    for kappa in (0.1, 0.3, 0.5, 0.9):
        assert analytic_null_expectation(kappa) == 1.0


def test_calibrator_is_decreasing_in_p():
    ps = np.array([1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0])
    es = calibrate_p_to_e(ps, 0.3)
    assert np.all(np.diff(es) < 0)


def test_p_equal_to_one_gives_evidence_below_one():
    # A maximally unconvincing test must *reduce* aggregated evidence.
    assert calibrate_p_to_e(1.0, 0.3) == pytest.approx(0.3)
    assert calibrate_p_to_e(1.0, 0.3) < 1.0


def test_zero_p_value_is_refused():
    # CLAUDE.md: p = 0 maps to an infinite e-value and would validate anything.
    with pytest.raises(ValueError, match="infinite e-value"):
        calibrate_p_to_e(0.0, 0.3)


def test_p_values_outside_the_unit_interval_are_refused():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="must lie in"):
            calibrate_p_to_e(bad, 0.3)


def test_kappa_outside_the_open_unit_interval_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="kappa"):
            calibrate_p_to_e(0.05, bad)
        with pytest.raises(ValueError, match="kappa"):
            analytic_null_expectation(bad)


def test_smaller_kappa_rewards_small_p_more():
    # PROJECT.md's justification for moving from kappa=0.5 to kappa=0.3.
    assert calibrate_p_to_e(1e-2, 0.3) > calibrate_p_to_e(1e-2, 0.5)


# --- the implication check (POPPER Assumption 1) ---------------------------


def test_uncontrolled_confounder_breaks_the_implication():
    assert _test(0.01).implication_holds
    assert not _test(0.01, confounders_uncontrolled=("live_feature_count",)).implication_holds


def test_excluded_tests_contribute_no_evidence_at_all():
    """CLAUDE.md: excluded, not down-weighted. The implication assumption is binary."""
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    f.add(_test(1e-9, confounders_uncontrolled=("live_feature_count",)))
    assert f.cumulative_e == 1.0
    assert f.tests == []
    assert len(f.excluded) == 1
    assert not f.validated()


def test_the_preprint_scenario_declines_to_validate():
    """The worked example: both headline wins are size-confounded, so nothing accrues."""
    f = SequentialFalsifier(main_hypothesis="vSAE is better organised", alpha=0.1, kappa=0.3)
    f.add(_test(1e-6, name="SCR", confounders_uncontrolled=("live_feature_count",)))
    f.add(_test(1e-6, name="TPP", confounders_uncontrolled=("live_feature_count",)))
    assert not f.validated()
    assert f.cumulative_e == 1.0
    assert "EXCLUDED" in f.report()


# --- aggregation and the decision rule -------------------------------------


def test_threshold_is_one_over_alpha():
    assert SequentialFalsifier(main_hypothesis="h", alpha=0.1).threshold == 10.0
    assert SequentialFalsifier(main_hypothesis="h", alpha=0.05).threshold == 20.0


def test_alpha_outside_the_open_unit_interval_is_refused():
    for bad in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="alpha"):
            SequentialFalsifier(main_hypothesis="h", alpha=bad)


def test_cumulative_evidence_is_the_product_of_e_values():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    for p in (0.2, 0.1, 0.05):
        f.add(_test(p))
    assert f.cumulative_e == pytest.approx(float(np.prod(f.e_values)))


def test_evidence_accumulates_until_the_threshold_is_reached():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    assert f.add(_test(0.5)) == "continue"
    assert not f.validated()
    assert f.add(_test(1e-8)) == "reject_null"
    assert f.validated()


def test_decision_is_exhausted_once_max_tests_is_reached():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3, max_tests=2)
    f.add(_test(0.9))
    assert f.add(_test(0.9)) == "exhausted"
    assert not f.validated()


def test_evidence_still_needed_is_one_once_validated():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    f.add(_test(1e-9))
    assert f.validated()
    assert f.evidence_still_needed() == 1.0


def test_evidence_still_needed_reports_the_multiplicative_shortfall():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    f.add(_test(0.5))
    assert f.evidence_still_needed() == pytest.approx(f.threshold / f.cumulative_e)


def test_report_names_the_decision_and_the_unit_of_analysis():
    f = SequentialFalsifier(main_hypothesis="h", alpha=0.1, kappa=0.3)
    f.add(_test(0.05))
    text = f.report()
    assert "training run (seed)" in text
    assert "Decision:" in text
    assert "kappa = 0.3" in text


# --- design arithmetic -----------------------------------------------------


def test_min_attainable_p_is_a_reciprocal_of_a_binomial_count():
    for n in (3, 4, 5, 6, 8):
        assert min_attainable_p(n) == pytest.approx(1.0 / comb(2 * n, n))


def test_min_attainable_p_two_sided_is_twice_the_one_sided_value():
    for n in (3, 5, 6):
        assert min_attainable_p(n, two_sided=True) == pytest.approx(
            2.0 * min_attainable_p(n)
        )


def test_five_seeds_do_validate_at_the_pre_registered_kappa():
    """Corrected-floor consequence, recorded because it reverses a planning claim.

    PROJECT.md says 5 seeds/group cannot validate on a single test. That held under
    the inverted floor (e = 8.85) and at the superseded kappa = 0.5, but not at the
    pre-registered kappa = 0.3 with the correct one-sided floor.
    """
    e5 = float(calibrate_p_to_e(min_attainable_p(5), 0.3))
    assert e5 == pytest.approx(14.4, abs=0.1)
    assert e5 >= 10.0


def test_min_attainable_p_is_monotone_in_seed_count():
    values = [min_attainable_p(n) for n in range(2, 12)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_min_attainable_p_refuses_degenerate_group_sizes():
    with pytest.raises(ValueError):
        min_attainable_p(0)


def test_five_seeds_cannot_validate_on_a_single_test_at_kappa_half():
    """PROJECT.md's planning fact, with the corrected floor.

    The claim survives at kappa = 0.5, but the value does not: the table says
    e = 5.61, computed from the inverted floor 1/126. The correct one-sided floor
    is 1/252, giving e = 7.94 -- still short of the threshold of 10.
    See test_five_seeds_do_validate_at_the_pre_registered_kappa for kappa = 0.3,
    where the conclusion reverses.
    """
    e5 = float(calibrate_p_to_e(min_attainable_p(5), 0.5))
    assert e5 == pytest.approx(7.94, abs=0.05)
    assert e5 < 10.0


def test_six_seeds_can_validate_on_a_single_test_at_kappa_half():
    """Also shifted by the floor correction: 10.75 in the table, 15.20 corrected."""
    e6 = float(calibrate_p_to_e(min_attainable_p(6), 0.5))
    assert e6 == pytest.approx(15.20, abs=0.05)
    assert e6 >= 10.0


def test_seeds_required_agrees_with_the_planning_table():
    assert seeds_required(alpha=0.1, kappa=0.5, n_tests=1) == 6
    assert seeds_required(alpha=0.1, kappa=0.5, n_tests=2) < 6


def test_seeds_required_decreases_as_more_tests_are_budgeted():
    one = seeds_required(alpha=0.1, kappa=0.3, n_tests=1)
    two = seeds_required(alpha=0.1, kappa=0.3, n_tests=2)
    assert two <= one
