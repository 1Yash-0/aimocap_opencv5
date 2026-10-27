"""Regression tests for opt-in hard/soft/off swing-twist limit modes."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.data.wholebody_layout import COCO_ANCHORS
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.ik_backend import _solve_scipy_precomputed
from aimocap.retarget.mocap_ik import (
    MocapIKSolver,
    TWIST_SOFT_BAND_RAD,
    soft_twist_hinge,
)
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
from aimocap.retarget.swing_twist import signed_twist_angle, TWIST_LIMITS

ROOT = Path(__file__).resolve().parents[1]


from tests._assets import requires_manny_fbx
def _fixture(frames=3):
    points = np.tile(np.array([0.0, 0.0, 80.0]), (frames, 133, 1)).astype(np.float64)
    anchors = {
        "pelvis_l": (-10.0, 0.0, 100.0), "pelvis_r": (10.0, 0.0, 100.0),
        "shoulder_l": (-20.0, 0.0, 140.0), "shoulder_r": (20.0, 0.0, 140.0),
        "elbow_l": (-42.0, 0.0, 105.0), "elbow_r": (42.0, 0.0, 105.0),
        "wrist_l": (-64.0, 0.0, 90.0), "wrist_r": (64.0, 0.0, 90.0),
        "knee_l": (-12.0, 0.0, 55.0), "knee_r": (12.0, 0.0, 55.0),
        "ankle_l": (-12.0, 0.0, 10.0), "ankle_r": (12.0, 0.0, 10.0),
        "big_toe_l": (-12.0, 15.0, 9.0), "big_toe_r": (12.0, 15.0, 9.0),
        "heel_l": (-12.0, -3.0, 9.0), "heel_r": (12.0, -3.0, 9.0),
        "small_toe_l": (-14.0, 14.0, 9.0), "small_toe_r": (14.0, 14.0, 9.0),
        "left_ear": (9.0, 2.0, 145.0), "right_ear": (-9.0, 2.0, 145.0),
        "left_eye": (2.0, 8.0, 146.0), "right_eye": (-2.0, 8.0, 146.0),
        "nose": (0.0, 12.0, 144.0),
    }
    for name, xyz in anchors.items():
        if name in COCO_ANCHORS:
            points[:, COCO_ANCHORS[name], :] = xyz
    confidence = np.ones((frames, 133), dtype=np.float64)
    rig = Skeleton(str(ROOT / "Manny.FBX"))
    skel = MocapSkeleton(points, confidence, fbx_skel=rig)
    targets = extract_mocap_points(points)
    return skel, targets


def _solve_isolated_twist(mode):
    skel, targets = _fixture(frames=1)
    solver = MocapIKSolver(skel, target_sequence=targets, twist_limit_mode=mode)
    measured = {key: value[0].copy() for key, value in targets.items()}
    ctx = solver._precompute_frame(measured, temporal_weight=0.0)
    name = "knee_l"
    ji = solver._twist_joint_indices[name]
    axis = solver._twist_bone_axes[name]
    # Isolate the limited local DOF: this synthetic target asks for 2x the
    # limit while all positional terms remain unchanged by axial twist.
    ctx["limb_pinned_info"] = []
    ctx["init_x"] = None
    ctx["prev_x"] = None
    ctx["temporal_weight"] = 0.0
    ctx["twist_limits"] = {name: TWIST_LIMITS[name]}
    ctx["twist_joint_indices"] = {name: ji}
    ctx["twist_bone_axes"] = {name: axis}
    x0 = np.zeros(solver.num_vars, dtype=np.float64)
    x0[3 + ji * 3:3 + ji * 3 + 3] = 2.0 * np.deg2rad(TWIST_LIMITS[name]) * axis
    result = _solve_scipy_precomputed(solver, ctx, x0)
    angle = abs(np.degrees(signed_twist_angle(
        Rotation.from_rotvec(result[3 + ji * 3:3 + ji * 3 + 3]), axis)))
    return angle, np.asarray(result), float(TWIST_LIMITS[name])


@requires_manny_fbx
def test_clamp_default_is_bit_identical_to_explicit_clamp_multi_frame():
    skel, targets = _fixture(frames=3)
    absent = MocapIKSolver(skel, target_sequence=targets)
    explicit = MocapIKSolver(skel, target_sequence=targets, twist_limit_mode="clamp")
    a_prev = b_prev = None
    a_out, b_out = [], []
    for frame in range(3):
        measured = {key: value[frame].copy() for key, value in targets.items()}
        a = absent.solve_frame(measured, prev_x=a_prev, temporal_weight=0.0, frame_idx=frame)
        b = explicit.solve_frame(measured, prev_x=b_prev, temporal_weight=0.0, frame_idx=frame)
        a_out.append(a)
        b_out.append(b)
        a_prev, b_prev = a, b
    assert np.array_equal(np.asarray(a_out), np.asarray(b_out))


@requires_manny_fbx
def test_soft_stays_within_limit_plus_three_degrees_for_2x_target():
    angle, _, limit = _solve_isolated_twist("soft")
    assert angle <= limit + 3.0


@requires_manny_fbx
def test_off_allows_twist_exceedance():
    angle, _, limit = _solve_isolated_twist("off")
    assert angle > limit + 2.0


def test_soft_residual_is_c1_at_band_edges():
    h = 1e-7
    for edge in (-TWIST_SOFT_BAND_RAD, 0.0):
        left_slope = float((soft_twist_hinge(edge) - soft_twist_hinge(edge - h)) / h)
        right_slope = float((soft_twist_hinge(edge + h) - soft_twist_hinge(edge)) / h)
        assert np.isclose(left_slope, right_slope, rtol=0.0, atol=2e-5)


if __name__ == "__main__":
    test_clamp_default_is_bit_identical_to_explicit_clamp_multi_frame()
    test_soft_stays_within_limit_plus_three_degrees_for_2x_target()
    test_off_allows_twist_exceedance()
    test_soft_residual_is_c1_at_band_edges()
    print("twist limit mode tests passed")



@requires_manny_fbx
def test_hybrid_keeps_non_soft_knee_hard_clamp():
    angle, _, limit = _solve_isolated_twist("hybrid")
    assert angle <= limit + 1e-6


@requires_manny_fbx
def test_hybrid_constant_target_has_no_iterative_twist_drift():
    skel, targets = _fixture(frames=1)
    solver = MocapIKSolver(skel, target_sequence=targets, twist_limit_mode="hybrid")
    measured = {key: value[0].copy() for key, value in targets.items()}
    ctx = solver._precompute_frame(measured, temporal_weight=0.0)
    name = "clavicle_l"
    ji = solver._twist_joint_indices[name]
    axis = solver._twist_bone_axes[name]
    ctx["limb_pinned_info"] = []
    ctx["init_x"] = None
    ctx["prev_x"] = None
    ctx["temporal_weight"] = 0.0
    ctx["twist_limits"] = {name: TWIST_LIMITS[name]}
    ctx["twist_joint_indices"] = {name: ji}
    ctx["twist_bone_axes"] = {name: axis}
    ctx["twist_soft_joints"] = {name}
    x = np.zeros(skel.num_joints * 3 + 3, dtype=np.float64)
    x[3 + ji * 3:3 + ji * 3 + 3] = 2.0 * np.deg2rad(TWIST_LIMITS[name]) * axis
    solved = []
    for _ in range(4):
        x = _solve_scipy_precomputed(solver, ctx, x)
        solved.append(x.copy())
    steps = []
    for a, b in zip(solved[:-1], solved[1:]):
        qa = Rotation.from_rotvec(a[3 + ji * 3:3 + ji * 3 + 3])
        qb = Rotation.from_rotvec(b[3 + ji * 3:3 + ji * 3 + 3])
        steps.append(np.degrees((qa.inv() * qb).magnitude()))
    assert max(steps, default=0.0) < 1e-4
    from aimocap.retarget.swing_twist import signed_twist_angle
    final_angle = abs(np.degrees(signed_twist_angle(
        Rotation.from_rotvec(solved[-1][3 + ji * 3:3 + ji * 3 + 3]), axis)))
    assert final_angle <= TWIST_LIMITS[name] + 10.0 + 1e-5
