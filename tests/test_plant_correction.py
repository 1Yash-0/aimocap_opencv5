from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.plant_correction import correct_planted_feet


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


from tests._assets import requires_manny_fbx
def _animation(n: int = 30) -> tuple[Skeleton, np.ndarray, dict[str, np.ndarray]]:
    skeleton = Skeleton(str(ROOT / "Manny.FBX"))
    rest = Rotation.from_quat(np.asarray(skeleton.rest_rotations)).as_euler("xyz")
    animation = np.zeros((n, 3 + 3 * skeleton.num_joints), dtype=float)
    animation[:, 3:] = rest.reshape(-1)
    thigh = skeleton.name_to_idx["thigh_l"]
    animation[:, 3 + 3 * thigh] += np.linspace(-0.12, 0.12, n)
    local = Rotation.from_euler("xyz", animation[:, 3:].reshape(-1, 3)).as_quat()
    local = local.reshape(n, skeleton.num_joints, 4)
    positions = np.asarray([
        skeleton.get_forward_kinematics(local[frame], animation[frame, :3])[0]
        for frame in range(n)
    ])
    foot = positions[0, skeleton.name_to_idx["foot_l"]]
    ball = positions[0, skeleton.name_to_idx["ball_l"]]
    targets = {
        "ankle_l": np.repeat(foot[None], n, axis=0),
        "heel_l": np.repeat(foot[None], n, axis=0),
        "big_toe_l": np.repeat(ball[None], n, axis=0),
        "ankle_r": np.repeat(positions[0, skeleton.name_to_idx["foot_r"]][None], n, axis=0),
        "heel_r": np.repeat(positions[0, skeleton.name_to_idx["foot_r"]][None], n, axis=0),
        "big_toe_r": np.repeat(positions[0, skeleton.name_to_idx["ball_r"]][None], n, axis=0),
    }
    return skeleton, animation, targets


@requires_manny_fbx
def test_static_targets_are_detected_and_result_is_finite_and_root_unchanged():
    skeleton, animation, targets = _animation()
    corrected, diagnostics = correct_planted_feet(skeleton, animation, targets, 30.0)
    assert diagnostics["plants_detected"] >= 2
    assert diagnostics["finite"] is True
    assert np.isfinite(corrected).all()
    np.testing.assert_array_equal(corrected[:, :3], animation[:, :3])


@requires_manny_fbx
def test_correction_is_deterministic_and_has_no_root_change():
    skeleton, animation, targets = _animation()
    first, diagnostics_a = correct_planted_feet(skeleton, animation, targets, 30.0, k=3.0)
    second, diagnostics_b = correct_planted_feet(skeleton, animation, targets, 30.0, k=3.0)
    np.testing.assert_array_equal(first, second)
    assert diagnostics_a == diagnostics_b
    np.testing.assert_array_equal(first[:, :3], animation[:, :3])
    assert diagnostics_a["frames_corrected"] > 0
    assert diagnostics_a["skipped_unreachable"] == 0


@requires_manny_fbx
def test_orientation_hold_uses_robust_plant_mean_for_foot_and_ball():
    skeleton, animation, targets = _animation(30)
    foot_idx = skeleton.name_to_idx["foot_l"]
    ball_idx = skeleton.name_to_idx["ball_l"]
    animation[:, 3 + 3 * foot_idx] = np.linspace(-0.35, 0.35, len(animation))
    local_before = Rotation.from_euler(
        "xyz", animation[:, 3:].reshape(-1, 3), degrees=False
    ).as_quat().reshape(len(animation), skeleton.num_joints, 4)
    before_world = np.asarray([
        skeleton.get_forward_kinematics(local_before[frame], animation[frame, :3])[1]
        for frame in range(len(animation))
    ])
    expected_foot = Rotation.from_quat(before_world[:, foot_idx]).mean()
    expected_ball = Rotation.from_quat(before_world[:, ball_idx]).mean()

    corrected, diagnostics = correct_planted_feet(skeleton, animation, targets, 30.0)
    local_after = Rotation.from_euler(
        "xyz", corrected[:, 3:].reshape(-1, 3), degrees=False
    ).as_quat().reshape(len(corrected), skeleton.num_joints, 4)
    after_world = np.asarray([
        skeleton.get_forward_kinematics(local_after[frame], corrected[frame, :3])[1]
        for frame in range(len(corrected))
    ])

    assert diagnostics["sides"]["left"]["orientation_hold"]
    foot_error = (Rotation.from_quat(after_world[15, foot_idx]).inv() * expected_foot).magnitude()
    ball_error = (Rotation.from_quat(after_world[15, ball_idx]).inv() * expected_ball).magnitude()
    assert foot_error < 1e-7
    assert ball_error < 1e-7

