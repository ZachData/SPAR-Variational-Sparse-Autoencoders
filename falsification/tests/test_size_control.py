"""Contracts for the E4 size-artifact control.

The central claim this module exists to support: the preprint's SCR/TPP advantage
may be an artifact of the vSAE having fewer live features. The test that matters
most here is the one demonstrating that the ORIGINALLY PLANNED version of the
experiment would have manufactured a false positive.
"""

import numpy as np
import pytest

from falsification.size_control import (
    interpolate_curve,
    mask_dictionary,
    select_features,
    size_response_curve,
    verdict,
)


@pytest.fixture
def usage():
    """Usage counts for a 100-feature dictionary; features 0..79 live, 80..99 dead."""
    counts = np.zeros(100)
    counts[:80] = np.arange(80, 0, -1, dtype=float)
    return counts


# --- feature selection -----------------------------------------------------


def test_top_usage_picks_the_most_used_features(usage):
    keep = select_features(usage, 10, "top_usage")
    assert len(keep) == 10
    assert set(keep) == set(range(10))  # counts descend from index 0


def test_top_usage_is_deterministic(usage):
    a = select_features(usage, 25, "top_usage")
    b = select_features(usage, 25, "top_usage")
    assert np.array_equal(a, b)


def test_selection_indices_are_sorted(usage):
    rng = np.random.default_rng(0)
    assert np.all(np.diff(select_features(usage, 20, "top_usage")) > 0)
    assert np.all(np.diff(select_features(usage, 20, "random", rng)) > 0)


def test_random_selection_requires_an_rng(usage):
    with pytest.raises(ValueError, match="requires an rng"):
        select_features(usage, 10, "random")


def test_random_selection_never_returns_a_dead_feature(usage):
    """A dead feature contributes nothing, so counting it would overstate the size."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        keep = select_features(usage, 40, "random", rng)
        assert np.all(usage[keep] > 0)


def test_random_selection_refuses_to_exceed_the_live_count(usage):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="only 80 live features"):
        select_features(usage, 90, "random", rng)


def test_out_of_range_sizes_are_refused(usage):
    for bad in (0, 101):
        with pytest.raises(ValueError, match="n_features must be in"):
            select_features(usage, bad, "top_usage")


def test_unknown_strategy_is_refused(usage):
    with pytest.raises(ValueError, match="unknown strategy"):
        select_features(usage, 10, "middle_out")


# --- the size-response curve ----------------------------------------------


def test_top_usage_is_evaluated_once_and_random_is_averaged(usage):
    calls = []

    def scorer(keep):
        calls.append(len(keep))
        return float(len(keep))

    points = size_response_curve(usage, scorer, n_grid=(10, 20), n_draws=7)
    top = [p for p in points if p.strategy == "top_usage"]
    rnd = [p for p in points if p.strategy == "random"]
    assert all(len(p.scores) == 1 for p in top)
    assert all(len(p.scores) == 7 for p in rnd)
    assert len(calls) == 2 * 1 + 2 * 7


def test_curve_point_reports_mean_and_std():
    scores = [1.0, 2.0, 3.0]
    from falsification.size_control import SizeResponsePoint

    point = SizeResponsePoint(10, "random", scores)
    assert point.mean == pytest.approx(2.0)
    assert point.std == pytest.approx(np.std(scores, ddof=1))


def test_single_score_has_zero_std():
    from falsification.size_control import SizeResponsePoint

    assert SizeResponsePoint(10, "top_usage", [1.5]).std == 0.0


def test_interpolation_is_linear_between_grid_points(usage):
    points = size_response_curve(
        usage, lambda keep: float(len(keep)), n_grid=(10, 30), strategies=("top_usage",)
    )
    assert interpolate_curve(points, 20, "top_usage") == pytest.approx(20.0)


def test_interpolating_a_missing_strategy_is_refused(usage):
    points = size_response_curve(
        usage, lambda keep: 1.0, n_grid=(10,), strategies=("top_usage",)
    )
    with pytest.raises(ValueError, match="no curve points"):
        interpolate_curve(points, 10, "random")


# --- the verdict -----------------------------------------------------------


def test_score_below_the_top_usage_curve_is_explained_by_size(usage):
    points = size_response_curve(
        usage, lambda keep: float(len(keep)), n_grid=(10, 50), strategies=("top_usage",)
    )
    v = verdict(observed_score=15.0, observed_n=30, points=points)
    assert v.explained_by_size
    assert v.margin < 0
    assert "EXPLAINED BY SIZE" in str(v)


def test_score_above_the_top_usage_curve_is_not_explained_by_size(usage):
    points = size_response_curve(
        usage, lambda keep: float(len(keep)), n_grid=(10, 50), strategies=("top_usage",)
    )
    v = verdict(observed_score=99.0, observed_n=30, points=points)
    assert not v.explained_by_size
    assert v.margin > 0
    assert "NOT explained by size" in str(v)


def test_verdict_defaults_to_the_top_usage_reference(usage):
    points = size_response_curve(usage, lambda keep: float(len(keep)), n_grid=(10, 50))
    assert verdict(1.0, 30, points).reference_strategy == "top_usage"


def test_random_subset_null_would_falsely_confirm_the_hypothesis(usage):
    """The design flaw that PROJECT.md records as caught before it did damage.

    Scoring model: a dictionary scores well when its kept features are the
    baseline's most-used ones. A random subset therefore scores badly, while the
    top-usage subset scores well. A vSAE with NO genuine advantage -- exactly the
    top-usage score at its own size -- still clears the random null comfortably.
    Judged against `random` it looks confirmed; judged against `top_usage` it is
    correctly called an artifact of size.
    """
    def scorer(keep):
        # Reward keeping high-usage features: mean usage rank of the kept set.
        return float(np.mean(usage[keep]))

    points = size_response_curve(usage, scorer, n_grid=(20, 40, 60), n_draws=30, seed=0)
    no_advantage = interpolate_curve(points, 40, "top_usage")

    against_top = verdict(no_advantage, 40, points, reference_strategy="top_usage")
    against_random = verdict(no_advantage, 40, points, reference_strategy="random")

    assert against_top.explained_by_size, "top-usage reference must not be beaten"
    assert not against_random.explained_by_size, (
        "the random reference is the weak null that manufactures a false positive"
    )


# --- weight masking --------------------------------------------------------


def test_masking_zeroes_everything_outside_the_kept_set():
    weights = {"W_enc": np.ones((6, 3)), "W_dec": np.ones((3, 6))}
    axes = {"W_enc": 0, "W_dec": 1}
    masked = mask_dictionary(weights, np.array([0, 2]), axes)
    assert np.array_equal(masked["W_enc"][:, 0], [1, 0, 1, 0, 0, 0])
    assert np.array_equal(masked["W_dec"][0, :], [1, 0, 1, 0, 0, 0])


def test_masking_respects_a_different_axis_per_weight():
    """Encoder rows and decoder columns index features on different axes."""
    weights = {"W_enc": np.arange(12).reshape(4, 3), "W_dec": np.arange(12).reshape(3, 4)}
    masked = mask_dictionary(weights, np.array([1]), {"W_enc": 0, "W_dec": 1})
    assert masked["W_enc"].shape == (4, 3)
    assert masked["W_dec"].shape == (3, 4)
    assert np.count_nonzero(masked["W_enc"][0]) == 0
    assert np.count_nonzero(masked["W_dec"][:, 0]) == 0


def test_masking_does_not_mutate_the_originals():
    original = np.ones((4, 2))
    masked = mask_dictionary({"W": original}, np.array([0]), {"W": 0})
    assert np.all(original == 1.0)
    assert masked["W"] is not original


def test_masking_requires_an_axis_for_every_weight():
    with pytest.raises(ValueError, match="no dictionary axis"):
        mask_dictionary({"W_enc": np.ones((4, 2))}, np.array([0]), {})
