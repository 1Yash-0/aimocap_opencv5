"""CPU-only regression for deterministic frontal-angle-matched foot locking."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.foot_lock import (
    _fk_sequence,
    _frontal_valgus_point,
    apply_foot_lock,
)


from tests._assets import requires_manny_fbx
def _positions(skel, animation):
    eulers = animation[:, 3:].reshape(len(animation), skel.num_joints, 3)
    quats = Rotation.from_euler("xyz", eulers.reshape(-1, 3)).as_quat()
    quats = quats.reshape(len(animation), skel.num_joints, 4)
    return _fk_sequence(skel, animation[:, :3], quats)[0]


def _left_valgus(pos, skel):
    ids = skel.name_to_idx
    values = [
        _frontal_valgus_point(
            pos[f, ids["thigh_l"]], pos[f, ids["thigh_r"]],
            pos[f, ids["thigh_l"]], pos[f, ids["calf_l"]],
            pos[f, ids["foot_l"]], "l",
        )
        for f in range(len(pos))
    ]
    return np.asarray([v for v in values if v is not None])


@requires_manny_fbx
def test_frontal_match_foot_lock_is_deterministic_and_preserves_knee_line():
    skel = Skeleton(str(ROOT / "Manny.FBX"))
    frames = 40
    rest_eulers = Rotation.from_quat(np.asarray(skel.rest_rotations)).as_euler(
        "xyz", degrees=False)
    animation = np.zeros((frames, 3 + skel.num_joints * 3), dtype=float)
    animation[:, 3:] = rest_eulers.reshape(-1)
    animation[:, 0] = 6.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, frames))
    before_pos = _positions(skel, animation)

    source = np.full((frames, 133, 3), np.nan, dtype=float)
    left_toe = 0.5 * (
        before_pos[0, skel.name_to_idx["foot_l"]]
        + before_pos[0, skel.name_to_idx["ball_l"]]
    )
    source[:, [17, 18, 19], :] = left_toe
    right_toe = 0.5 * (
        before_pos[0, skel.name_to_idx["foot_r"]]
        + before_pos[0, skel.name_to_idx["ball_r"]]
    )
    source[:, [20, 21, 22], :] = right_toe + np.array([0.0, 0.0, 100.0])

    locked_a, diagnostics_a = apply_foot_lock(skel, animation.copy(), source, fps=30.0)
    locked_b, diagnostics_b = apply_foot_lock(skel, animation.copy(), source, fps=30.0)
    assert np.array_equal(locked_a, locked_b)
    assert diagnostics_a["contacts"] == diagnostics_b["contacts"]
    assert diagnostics_a["contacts"]["left"]["segments"]
    assert diagnostics_a["after"]["left"]["foot_drift_cm"] < diagnostics_a[
        "before"]["left"]["foot_drift_cm"]
    source_p95 = float(np.percentile(_left_valgus(before_pos, skel), 95))
    after_p95 = float(np.percentile(_left_valgus(_positions(skel, locked_a), skel), 95))
    assert abs(after_p95 - source_p95) < 4.0
    assert np.isfinite(locked_a).all()
