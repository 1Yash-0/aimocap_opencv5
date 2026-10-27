"""Regression tests for the opt-in ankle twist clamp override."""
import numpy as np

from aimocap.retarget.mocap_ik import MocapIKSolver
from tests.test_twist_limit_mode import _fixture


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_ankle_twist_limit_none_is_bit_identical_to_explicit_default():
    skel, targets = _fixture(frames=3)
    implicit = MocapIKSolver(skel, target_sequence=targets)
    explicit = MocapIKSolver(
        skel, target_sequence=targets, ankle_twist_limit_deg=None
    )
    assert implicit._twist_limits == explicit._twist_limits
    implicit_states = []
    explicit_states = []
    prev_implicit = prev_explicit = None
    for frame in range(3):
        measured = {key: value[frame].copy() for key, value in targets.items()}
        implicit_state = implicit.solve_frame(
            measured, prev_x=prev_implicit, temporal_weight=0.0, frame_idx=frame
        )
        explicit_state = explicit.solve_frame(
            measured, prev_x=prev_explicit, temporal_weight=0.0, frame_idx=frame
        )
        implicit_states.append(implicit_state)
        explicit_states.append(explicit_state)
        prev_implicit, prev_explicit = implicit_state, explicit_state
    assert np.array_equal(np.asarray(implicit_states), np.asarray(explicit_states))


@requires_manny_fbx
def test_ankle_twist_override_changes_only_ankle_limit_entries():
    skel, targets = _fixture(frames=1)
    base = MocapIKSolver(skel, target_sequence=targets)
    override = MocapIKSolver(
        skel, target_sequence=targets, ankle_twist_limit_deg=30.0
    )
    assert override._twist_limits["ankle_l"] == 30.0
    assert override._twist_limits["ankle_r"] == 30.0
    changed = {
        name for name in base._twist_limits
        if base._twist_limits[name] != override._twist_limits[name]
    }
    assert changed == {"ankle_l", "ankle_r"}
    for name in base._twist_limits:
        if name not in {"ankle_l", "ankle_r"}:
            assert base._twist_limits[name] == override._twist_limits[name]


@requires_manny_fbx
def test_ankle_twist_override_rejects_invalid_values():
    skel, targets = _fixture(frames=1)
    for value in (-1.0, float("nan"), float("inf"), True):
        try:
            MocapIKSolver(
                skel, target_sequence=targets, ankle_twist_limit_deg=value
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ankle limit accepted: {value!r}")
