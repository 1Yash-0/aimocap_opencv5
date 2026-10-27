"""Tests for Phase 2.2: Preservation of the Full 133-Joint Canonical Representation.

Verifies:
- Test A: Canonical reconstruction shape is strictly (T, 133, 3) in production mode.
- Test B: Canonical 133-joint ordering matches the repository's authoritative WholeBody layout.
- Test C: Diagnostics arrays (confidence, reprojection, inliers, ray angles, depth) remain
          aligned to the 133-joint dimension.
- Test D: Explicit retarget adapter (WholeBody133ToManny17Adapter) produces exact 17-joint proxy.
- Test E: Explicit retarget adapter rejects invalid/ambiguous joint counts loudly.
- Test F: Downstream retarget_to_fbx accepts canonical 133-joint and adapted 17-joint data,
          and rejects invalid joint dimensions (e.g. 19, 25).
- Test G: Regression proving production reconstruction does NOT prematurely reduce to 17 joints
          or overwrite body coordinates with an old rigid stick model.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from aimocap.cli import main
from aimocap.calib.io import save_calibration
from aimocap.data.wholebody_layout import (
    ALL_NAMES,
    BODY_NAMES,
    FACE_NAMES,
    FOOT_NAMES,
    LEFT_HAND_NAMES,
    N_KEYPOINTS,
    RIGHT_HAND_NAMES,
)
from aimocap.pose.keypoints import (
    KEYPOINT_NAMES_133,
    REGIONS_133,
    SKELETON_133,
)
from aimocap.retarget.adapter import (
    MANNY_17_JOINT_NAMES,
    MANNY_17_NUM_JOINTS,
    WHOLEBODY_133_NUM_JOINTS,
    WholeBody133ToManny17Adapter,
    wholebody133_to_manny17,
)
from aimocap.retarget.engine import retarget_to_fbx


def _create_mock_calibration_and_poses(
    tmp_path: Path,
    num_frames: int = 6,
    num_cams: int = 2,
    num_joints: int = 133,
) -> tuple[Path, Path]:
    """Create matched calibration and multi-view 2D pose test fixtures."""
    calib_path = tmp_path / "mock_calib.json"
    pose_path = tmp_path / "mock_poses.npz"

    K0 = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K1 = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros((3, 1), dtype=np.float64)
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.array([[-1.0], [0.0], [0.0]], dtype=np.float64)

    save_calibration(
        calib_path,
        [K0, K1],
        [(R0, t0), (R1, t1)],
        image_sizes=[(1920, 1080), (1920, 1080)],
        camera_names=["cam0", "cam1"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
    )

    # Build synthetic 3D points
    rng = np.random.default_rng(1234)
    pts3d = rng.uniform(-0.5, 0.5, (num_frames, num_joints, 3)).astype(np.float64)
    pts3d[..., 2] += 3.0  # 3m depth

    # Project to cam0 and cam1
    kpts2d = np.zeros((num_frames, num_cams, num_joints, 2), dtype=np.float32)
    scores = np.ones((num_frames, num_cams, num_joints), dtype=np.float32)

    for c_idx, (K, R, t) in enumerate([(K0, R0, t0), (K1, R1, t1)]):
        for f in range(num_frames):
            p_cam = (R @ pts3d[f].T + t).T
            p_pix = (K @ (p_cam / p_cam[:, 2:3]).T).T[:, :2]
            kpts2d[f, c_idx] = p_pix.astype(np.float32)

    np.savez(
        pose_path,
        keypoints=kpts2d,
        scores=scores,
        fps=np.asarray(30.0),
        image_sizes=np.array([[1920, 1080], [1920, 1080]]),
        camera_names=np.array(["cam0", "cam1"]),
        sync_status="VERIFIED_AUDIO_CLAP",
    )

    return calib_path, pose_path


# ── Test A: Canonical shape ──────────────────────────────────────────────────

def test_a_canonical_shape_preserved_in_production(tmp_path: Path):
    """Test A: Standard (non-raw) production triangulation preserves (T, 133, 3) shape."""
    calib_path, pose_path = _create_mock_calibration_and_poses(tmp_path, num_frames=8)
    out_path = tmp_path / "canonical_skeleton.npz"

    ret = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path),
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret == 0, "triangulate CLI command failed"
    assert out_path.exists(), "Output skeleton3d.npz was not generated"

    data = np.load(out_path)
    assert "skeleton3d" in data, "skeleton3d key missing from output"
    skel = data["skeleton3d"]
    assert skel.shape == (8, 133, 3), f"Expected shape (8, 133, 3), got {skel.shape}"
    assert np.all(np.isfinite(skel)), "Output 3D coordinates contain non-finite values"


# ── Test B: Joint ordering ───────────────────────────────────────────────────

def test_b_canonical_133_joint_ordering():
    """Test B: Canonical 133-joint layout matches repository's authoritative definition."""
    assert N_KEYPOINTS == 133
    assert WHOLEBODY_133_NUM_JOINTS == 133
    assert len(ALL_NAMES) == 133
    assert len(KEYPOINT_NAMES_133) == 133

    # Check region boundaries match
    assert REGIONS_133["body"].start == 0 and REGIONS_133["body"].end == 17
    assert REGIONS_133["feet"].start == 17 and REGIONS_133["feet"].end == 23
    assert REGIONS_133["face"].start == 23 and REGIONS_133["face"].end == 91
    assert REGIONS_133["left_hand"].start == 91 and REGIONS_133["left_hand"].end == 112
    assert REGIONS_133["right_hand"].start == 112 and REGIONS_133["right_hand"].end == 133

    # First 17 are body
    assert len(BODY_NAMES) == 17
    assert BODY_NAMES[0] == "nose"
    assert BODY_NAMES[5] == "left_shoulder"
    assert BODY_NAMES[6] == "right_shoulder"
    assert BODY_NAMES[15] == "left_ankle"
    assert BODY_NAMES[16] == "right_ankle"

    # Next 6 are feet
    assert len(FOOT_NAMES) == 6
    assert FOOT_NAMES[0] == "big_toe_l"
    assert FOOT_NAMES[3] == "big_toe_r"

    # Face has 68
    assert len(FACE_NAMES) == 68

    # Hands have 21 each
    assert len(LEFT_HAND_NAMES) == 21
    assert len(RIGHT_HAND_NAMES) == 21

    # Deterministic sequence check
    expected_all = BODY_NAMES + FOOT_NAMES + FACE_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES
    assert ALL_NAMES == expected_all


# ── Test C: Diagnostics alignment ────────────────────────────────────────────

def test_c_diagnostics_joint_dimension_alignment(tmp_path: Path):
    """Test C: All diagnostic arrays match the 133-joint dimension."""
    calib_path, pose_path = _create_mock_calibration_and_poses(tmp_path, num_frames=6)
    out_path = tmp_path / "diagnostics_test.npz"

    ret = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path),
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret == 0

    data = np.load(out_path)
    F = 6
    J = 133

    assert data["skeleton3d"].shape == (F, J, 3)
    assert data["confidence"].shape == (F, J)
    assert data["triangulation_reprojection_error_px"].shape == (F, J, 2)
    assert data["triangulation_reprojection_error_px"].shape[1] == J
    assert data["triangulation_num_inliers"].shape == (F, J)
    assert data["triangulation_ray_angle_deg"].shape == (F, J)
    assert data["triangulation_observed_camera_count"].shape == (F, J)
    assert data["triangulation_positive_depth_mask"].shape == (F, J, 2)
    assert data["triangulation_positive_depth_mask"].shape[1] == J
    assert data["triangulation_observation_provenance"].shape == (F, J)
    assert data["triangulation_inlier_mask"].shape == (F, J, 2)
    assert data["triangulation_inlier_mask"].shape[1] == J
    assert data["triangulation_candidate_inlier_mask"].shape == (F, J, 2)
    assert data["triangulation_candidate_inlier_mask"].shape[1] == J


# ── Test D: Explicit retarget mapping ────────────────────────────────────────

def test_d_explicit_retarget_adapter_mapping():
    """Test D: WholeBody133ToManny17Adapter maps known 133-joint input to 17 joints accurately."""
    rng = np.random.default_rng(42)
    skel_133 = rng.normal(size=(5, 133, 3)).astype(np.float64)
    conf_133 = rng.uniform(0.1, 1.0, size=(5, 133)).astype(np.float32)

    adapter = WholeBody133ToManny17Adapter()
    adapted_skel, adapted_conf = adapter.adapt(skel_133, conf_133)

    assert adapted_skel.shape == (5, 17, 3)
    assert adapted_conf.shape == (5, 17)

    # Verify exact coordinate equivalence for the 17 body joints
    np.testing.assert_array_equal(adapted_skel, skel_133[:, :17, :])
    np.testing.assert_array_equal(adapted_conf, conf_133[:, :17])

    # Test convenience function
    func_skel, func_conf = wholebody133_to_manny17(skel_133, conf_133)
    np.testing.assert_array_equal(func_skel, adapted_skel)
    np.testing.assert_array_equal(func_conf, adapted_conf)


# ── Test E: Wrong input rejection ────────────────────────────────────────────

def test_e_adapter_rejects_wrong_joint_counts():
    """Test E: Adapter rejects incorrectly sized inputs loudly instead of silently dropping data."""
    adapter = WholeBody133ToManny17Adapter()

    # Wrong joint counts
    for bad_j in (16, 18, 19, 23, 68, 132, 134):
        bad_skel = np.zeros((4, bad_j, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="WholeBody133ToManny17Adapter expected joint dimension 133"):
            adapter.adapt(bad_skel)

    # 17 joints rejected when allow_passthrough_17=False
    skel_17 = np.zeros((4, 17, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="allow_passthrough_17=False"):
        adapter.adapt(skel_17)

    # 17 joints accepted when allow_passthrough_17=True
    adapter_passthrough = WholeBody133ToManny17Adapter(allow_passthrough_17=True)
    out_17, _ = adapter_passthrough.adapt(skel_17)
    assert out_17.shape == (4, 17, 3)

    # Wrong dimensionality
    with pytest.raises(ValueError, match="ndim >= 2"):
        adapter.adapt(np.zeros((3,)))
    with pytest.raises(ValueError, match="3D coordinates"):
        adapter.adapt(np.zeros((4, 133, 2)))

    # Mismatched confidence shape
    with pytest.raises(ValueError, match="Confidence shape"):
        adapter.adapt(np.zeros((4, 133, 3)), np.zeros((4, 17)))


# ── Test F: Downstream compatibility with retarget_to_fbx ─────────────────────

def test_f_downstream_retarget_compatibility(tmp_path: Path):
    """Test F: retarget_to_fbx accepts canonical 133-joint input and adapted 17-joint input,

    and rejects invalid joint topologies.
    """
    fbx_rig = Path("Manny.FBX")
    if not fbx_rig.exists():
        pytest.skip("Manny.FBX not available for rig testing")

    # 1. Test canonical 133-joint input
    skel_133 = np.zeros((10, 133, 3), dtype=np.float64)
    # Valid core positions for MocapSkeleton
    skel_133[:, 5] = [-20, 140, 0]   # shoulder_l
    skel_133[:, 6] = [20, 140, 0]    # shoulder_r
    skel_133[:, 11] = [-10, 90, 0]   # hip_l
    skel_133[:, 12] = [10, 90, 0]    # hip_r
    skel_133[:, 13] = [-10, 50, 0]   # knee_l
    skel_133[:, 14] = [10, 50, 0]    # knee_r
    skel_133[:, 15] = [-10, 10, 0]   # ankle_l
    skel_133[:, 16] = [10, 10, 0]    # ankle_r
    skel_133[:, 17] = [-10, 5, 10]   # big_toe_l
    skel_133[:, 20] = [10, 5, 10]    # big_toe_r

    npz_133 = tmp_path / "valid_133.npz"
    np.savez(npz_133, skeleton3d=skel_133, units="cm")

    bvh_out_133 = tmp_path / "out_133.bvh"
    retarget_to_fbx(
        str(npz_133),
        str(fbx_rig),
        str(bvh_out_133),
        max_frames=5,
        mocap_target_stabilization=False,
        leg_stabilization=False,
        foot_lock=False,
        export_fbx=False,
    )
    assert bvh_out_133.exists()

    # 2. Test explicit 17-joint input (adapted)
    adapter = WholeBody133ToManny17Adapter(allow_passthrough_17=False)
    skel_17, _ = adapter.adapt(skel_133)
    npz_17 = tmp_path / "adapted_17.npz"
    np.savez(npz_17, skeleton3d=skel_17, units="cm")

    bvh_out_17 = tmp_path / "out_17.bvh"
    retarget_to_fbx(
        str(npz_17),
        str(fbx_rig),
        str(bvh_out_17),
        max_frames=5,
        mocap_target_stabilization=False,
        leg_stabilization=False,
        foot_lock=False,
        export_fbx=False,
    )
    assert bvh_out_17.exists()

    # 3. Test invalid joint dimension rejected
    npz_invalid = tmp_path / "invalid_19.npz"
    np.savez(npz_invalid, skeleton3d=np.zeros((10, 19, 3)), units="cm")
    with pytest.raises(ValueError, match="must have 133 joints.*or 17 joints"):
        retarget_to_fbx(
            str(npz_invalid),
            str(fbx_rig),
            str(tmp_path / "invalid.bvh"),
        )


# ── Test G: No premature reduction regression ────────────────────────────────

def test_g_no_premature_reduction_regression(tmp_path: Path):
    """Test G: Regression proving production triangulation does NOT overwrite canonical

    triangulation points with a 17-joint kinematic stick figure.
    """
    calib_path, pose_path = _create_mock_calibration_and_poses(tmp_path, num_frames=6)
    out_path_prod = tmp_path / "prod_tri.npz"
    out_path_raw = tmp_path / "raw_tri.npz"

    # Run raw triangulation
    ret_raw = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path_raw),
        "--raw",
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret_raw == 0

    # Run standard production triangulation
    ret_prod = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path_prod),
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret_prod == 0

    raw_data = np.load(out_path_raw)
    prod_data = np.load(out_path_prod)

    # In production checkerboard calibration, scale is preserved (scale_factor=1.0)
    # and canonical 133-joint 3D coordinates must match raw triangulation exactly.
    assert prod_data["skeleton3d"].shape == (6, 133, 3)
    np.testing.assert_allclose(
        prod_data["skeleton3d"],
        raw_data["skeleton3d"],
        atol=1e-6,
        err_msg="Production skeleton3d was altered or reduced from canonical triangulation!",
    )
