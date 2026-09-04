"""Contracts for the arm table.

`run_arm.py --check` already validates every arm's overrides against its training
script's dataclass. What it cannot check is whether an arm means what its name and
comment say. A factor arm is only interpretable if it differs from its reference in
EXACTLY the factor being measured -- that is the whole design, and it is the thing
a careless edit silently breaks.
"""

from falsification.run_arm import ARMS


def _diff(a: str, b: str) -> dict:
    """Overrides that differ between two arms, as {field: (a_value, b_value)}."""
    left, right = ARMS[a]["overrides"], ARMS[b]["overrides"]
    return {
        field: (left.get(field), right.get(field))
        for field in set(left) | set(right)
        if left.get(field) != right.get(field)
    }


def test_gradproj_arm_isolates_exactly_one_factor():
    """e1_vsae_ref_gradproj must be e1_vsae_ref_unitinit plus the projection.

    Anything else in the diff confounds the projection with that difference, which
    is precisely the failure mode E3's ReLU and E1's decoder init already produced.
    """
    assert _diff("e1_vsae_ref_unitinit", "e1_vsae_ref_gradproj") == {
        "project_decoder_grad": (None, True)
    }


def test_gradproj_arm_keeps_the_matched_init():
    """The projection is measured on top of the MATCHED configuration, not the
    historical one -- the reconstruction gap it targets is the one that survives
    matching kl_warmup, the bias form and the decoder init."""
    overrides = ARMS["e1_vsae_ref_gradproj"]["overrides"]
    assert overrides["decoder_init_scale"] == 1.0
    assert overrides["kl_warmup_steps"] == 0
    assert overrides["use_april_update_mode"] is False


# The two arms that deliberately carry the projection. Everything else must keep
# the historical behaviour; adding a name here is a claim that the arm exists to
# measure the projection or something layered on top of it.
PROJECTING_ARMS = {"e1_vsae_ref_gradproj", "e1_vsae_ref_fullmatch"}


def test_reference_arms_do_not_project():
    """Every pre-existing arm must keep the historical behaviour, or the committed
    13-seed results stop being comparable to anything run later."""
    for arm, spec in ARMS.items():
        if arm in PROJECTING_ARMS:
            continue
        assert spec["overrides"].get("project_decoder_grad", False) is False, arm


def test_reference_arms_keep_the_gaussian_init_draw():
    """Same contract for the init distribution: only the closing arm changes it, so
    every committed checkpoint stays comparable."""
    for arm, spec in ARMS.items():
        if arm == "e1_vsae_ref_fullmatch":
            continue
        assert "decoder_init_dist" not in spec["overrides"], arm


def test_fullmatch_arm_isolates_exactly_one_factor():
    """e1_vsae_ref_fullmatch must be e1_vsae_ref_gradproj plus the init draw.

    It is the closing arm of E1's decomposition: the last item on the frozen code
    diff (RESULTS addendum 4). If it differs from gradproj in anything else, the
    init draw's measured effect is confounded and the decomposition does not close.
    """
    assert _diff("e1_vsae_ref_gradproj", "e1_vsae_ref_fullmatch") == {
        "decoder_init_dist": (None, "uniform")
    }


def test_fullmatch_arm_matches_every_enumerated_factor():
    """The frozen list, pinned. This arm's whole claim is that NOTHING on the E1
    code diff is left unmatched in it, so each matched factor is asserted here
    rather than left to the arm's comment."""
    overrides = ARMS["e1_vsae_ref_fullmatch"]["overrides"]
    assert overrides["kl_warmup_steps"] == 0            # factor 1
    assert overrides["use_april_update_mode"] is False  # factor 2
    assert overrides["decoder_init_scale"] == 1.0       # factor 3
    assert overrides["project_decoder_grad"] is True    # factor 4
    assert overrides["decoder_init_dist"] == "uniform"  # factor 7, the closing one


def test_init_factor_still_isolates_one_factor():
    """The precedent this arm copies; pinned so the pair stays interpretable."""
    assert _diff("e1_vsae_ref", "e1_vsae_ref_unitinit") == {
        "decoder_init_scale": (None, 1.0)
    }


def test_relu_factor_still_isolates_one_factor():
    assert _diff("e3_masked_kl", "e3_masked_kl_relu") == {"relu_mu": (False, True)}
