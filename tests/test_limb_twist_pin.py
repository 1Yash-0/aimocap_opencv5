import numpy as np
import pytest

from aimocap.retarget.mocap_ik import (
    LIMB_TWIST_PIN_THETA_HI_DEG,
    LIMB_TWIST_PIN_THETA_LO_DEG,
    limb_twist_pin_multiplier,
)
from tests.test_limb_twist_anchor import _measured, _pin_context, _solver


from tests._assets import requires_manny_fbx
def _bend(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0 + np.cos(theta), np.sin(theta), 0.0],
    ])


def test_fade_has_exact_endpoints_and_c1_midpoint():
    m_lo, _ = limb_twist_pin_multiplier(_bend(LIMB_TWIST_PIN_THETA_LO_DEG), 0, 1, 2)
    m_mid, _ = limb_twist_pin_multiplier(_bend((LIMB_TWIST_PIN_THETA_LO_DEG + LIMB_TWIST_PIN_THETA_HI_DEG) / 2.0), 0, 1, 2)
    m_hi, _ = limb_twist_pin_multiplier(_bend(LIMB_TWIST_PIN_THETA_HI_DEG), 0, 1, 2)
    assert m_lo == 1.0
    assert m_mid == pytest.approx(0.5)
    assert m_hi == 0.0


@requires_manny_fbx
def test_always_matches_default_and_off_removes_term():
    measured = _measured(_solver())
    default = _solver()
    explicit = _solver()
    explicit.limb_twist_pin = "always"
    a = _pin_context(default, measured)
    b = _pin_context(explicit, measured)
    assert len(a["limb_pinned_info"]) == len(b["limb_pinned_info"])
    for left, right in zip(a["limb_pinned_info"], b["limb_pinned_info"]):
        assert left["twist_target"] == right["twist_target"]
        assert left["twist_weight"] == right["twist_weight"]
    off = _solver()
    off.limb_twist_pin = "off"
    assert _pin_context(off, measured)["limb_pinned_info"] == []
