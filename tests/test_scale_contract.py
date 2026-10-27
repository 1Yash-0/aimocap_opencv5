"""Tests for V1 metric-scale handling and scale contract validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aimocap.calib.io import load_calibration_bundle, save_calibration
from aimocap.calib.scale import (
    ScaleContractError,
    ScalePolicy,
    apply_metric_scale,
    resolve_scale_policy,
)
from aimocap.cli import main


import cv2


def _make_test_skeleton(femur_len: float = 0.40, depth: float = 3.0) -> np.ndarray:
    """Create a synthetic 133-joint skeleton in front of the camera (OpenCV Y-down, Z-forward)."""
    skel = np.zeros((10, 133, 3), dtype=np.float64)
    for f in range(10):
        # Pelvis / hips at depth Z=3.0
        skel[f, 11, :] = np.array([0.14, 0.0, depth])   # hip_l
        skel[f, 12, :] = np.array([-0.14, 0.0, depth])  # hip_r
        # Knees below hips (positive Y)
        skel[f, 13, :] = np.array([0.14, femur_len, depth])   # knee_l
        skel[f, 14, :] = np.array([-0.14, femur_len, depth])  # knee_r
        # Ankles below knees
        skel[f, 15, :] = np.array([0.14, femur_len + 0.40, depth])  # ankle_l
        skel[f, 16, :] = np.array([-0.14, femur_len + 0.40, depth]) # ankle_r
        # Shoulders above hips (negative Y)
        skel[f, 5, :] = np.array([0.18, -0.45, depth])  # shoulder_l
        skel[f, 6, :] = np.array([-0.18, -0.45, depth]) # shoulder_r
        # Elbows
        skel[f, 7, :] = np.array([0.18, -0.15, depth])  # elbow_l
        skel[f, 8, :] = np.array([-0.18, -0.15, depth]) # elbow_r
        # Wrists
        skel[f, 9, :] = np.array([0.18, 0.10, depth])   # wrist_l
        skel[f, 10, :] = np.array([-0.18, 0.10, depth])  # wrist_r
        # Head / face
        skel[f, 0, :] = np.array([0.0, -0.65, depth])   # nose
        skel[f, 1, :] = np.array([0.03, -0.68, depth])  # eye_l
        skel[f, 2, :] = np.array([-0.03, -0.68, depth]) # eye_r
        skel[f, 3, :] = np.array([0.07, -0.68, depth])  # ear_l
        skel[f, 4, :] = np.array([-0.07, -0.68, depth]) # ear_r
    return skel


def _project_skeleton(skel3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    num_frames, num_kpts, _ = skel3d.shape
    rvec, _ = cv2.Rodrigues(R)
    out2d = np.zeros((num_frames, num_kpts, 2), dtype=np.float32)
    for f in range(num_frames):
        pts2d, _ = cv2.projectPoints(skel3d[f], rvec, t, K, None)
        out2d[f] = pts2d.reshape(-1, 2)
    return out2d


def test_production_checkerboard_scale_unchanged():
    """Production checkerboard calibration must strictly preserve metric scale without anthropometric scaling."""
    metadata = {
        "accuracy_class": "production",
        "calibration_source": "checkerboard",
        "scale_source": "checkerboard",
        "units": "meters",
    }
    policy = resolve_scale_policy(metadata, allow_exploratory=False)
    assert policy.action == "PRESERVE_METRIC"
    assert policy.is_metric is True
    assert policy.scale_factor == 1.0
    assert policy.scale_source == "checkerboard"

    # Even if allow_exploratory is True, production checkerboard calibration must remain preserved
    policy_override = resolve_scale_policy(metadata, allow_exploratory=True)
    assert policy_override.action == "PRESERVE_METRIC"
    assert policy_override.is_metric is True
    assert policy_override.scale_factor == 1.0


def test_exploratory_calibration_scaling_allowed_only_with_override():
    """Exploratory calibration must be rejected unless explicitly permitted."""
    metadata = {
        "accuracy_class": "exploratory",
        "calibration_source": "markerless_pose_extrinsics",
        "scale_source": "unscaled",
        "units": "arbitrary",
    }

    # Without override -> must raise ScaleContractError
    with pytest.raises(ScaleContractError) as exc_info:
        resolve_scale_policy(metadata, allow_exploratory=False)
    assert "Pass --allow-exploratory-calibration" in str(exc_info.value)

    # With override -> returns ESTIMATE_ANTHROPOMETRIC
    policy = resolve_scale_policy(metadata, allow_exploratory=True)
    assert policy.action == "ESTIMATE_ANTHROPOMETRIC"
    assert policy.is_metric is False
    assert policy.scale_source == "anthropometric"

    # Verify anthropometric scaling actually estimates scale factor
    skel_1_5x = _make_test_skeleton(femur_len=0.40) * 1.5
    extrinsics = [(np.eye(3), np.zeros((3, 1)))]
    scaled_skel, scaled_ext, factor = apply_metric_scale(skel_1_5x, extrinsics)
    assert np.isclose(factor, 1.0 / 1.5, atol=0.05)
    assert not np.allclose(scaled_skel, skel_1_5x)


def test_missing_or_invalid_scale_metadata_raises_error():
    """Missing or contradictory scale metadata must raise ScaleContractError."""
    # 1. Empty metadata
    with pytest.raises(ScaleContractError) as exc:
        resolve_scale_policy({})
    assert "Missing or invalid scale metadata" in str(exc.value)

    # 2. Production accuracy class but arbitrary units (contradiction)
    with pytest.raises(ScaleContractError) as exc:
        resolve_scale_policy({
            "accuracy_class": "production",
            "calibration_source": "checkerboard",
            "units": "arbitrary",
        })
    assert "Production calibration contract violation" in str(exc.value)

    # 3. Missing units in production
    with pytest.raises(ScaleContractError) as exc:
        resolve_scale_policy({
            "accuracy_class": "production",
            "calibration_source": "checkerboard",
            "units": "unknown",
        })
    assert "Production calibration contract violation" in str(exc.value)


def test_panoptic_studio_calibration_preserves_metric():
    """Panoptic studio calibration must be inferred as production metric."""
    metadata = {
        "calibDataSource": "cmu_panoptic",
        "units": "cm",
    }
    policy = resolve_scale_policy(metadata, allow_exploratory=False)
    assert policy.action == "PRESERVE_METRIC"
    assert policy.is_metric is True
    assert policy.scale_factor == 1.0


def test_save_and_load_calibration_bundle_preserves_scale_source(tmp_path):
    """save_calibration and load_calibration_bundle must preserve scale_source."""
    path = tmp_path / "calib_prod.json"
    K = np.eye(3) * 1000.0
    K[2, 2] = 1.0
    R = np.eye(3)
    t = np.array([[0.0], [0.0], [0.0]])

    save_calibration(
        path,
        [K],
        [(R, t)],
        image_sizes=[(1920, 1080)],
        camera_names=["cam0"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
    )

    bundle = load_calibration_bundle(path, require_metadata=True)
    assert bundle.metadata["scale_source"] == "checkerboard"
    assert bundle.metadata["accuracy_class"] == "production"
    assert bundle.metadata["units"] == "meters"

    policy = resolve_scale_policy(bundle.metadata, allow_exploratory=False)
    assert policy.action == "PRESERVE_METRIC"
    assert policy.scale_factor == 1.0


def test_cli_triangulate_scale_contract_integration(tmp_path):
    """Integration test: CLI triangulate preserves metric scale from checkerboard."""
    calib_path = tmp_path / "checkerboard_calib.json"
    exploratory_calib_path = tmp_path / "exploratory_calib.json"
    pose_path = tmp_path / "poses.npz"
    out_path_raw = tmp_path / "skeleton_raw.npz"
    out_path_prod = tmp_path / "skeleton_prod.npz"
    out_path_expl = tmp_path / "skeleton_expl.npz"

    # 1. Write production checkerboard calibration
    K0 = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K1 = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    R1 = np.eye(3)
    t1 = np.array([[-1.0], [0.0], [0.0]])  # 1 meter baseline

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

    # Write exploratory unscaled calibration
    save_calibration(
        exploratory_calib_path,
        [K0, K1],
        [(R0, t0), (R1, t1)],
        image_sizes=[(1920, 1080), (1920, 1080)],
        camera_names=["cam0", "cam1"],
        units="arbitrary",
        calibration_source="markerless_pose_extrinsics",
        accuracy_class="exploratory",
        scale_source="unscaled",
    )

    # 2. Write synthetic 2D pose observations projected from a known 3D skeleton
    test_skel = _make_test_skeleton(femur_len=0.40, depth=3.0)
    num_frames = 10
    num_cams = 2
    num_kpts = 133
    keypoints = np.zeros((num_frames, num_cams, num_kpts, 2), dtype=np.float32)
    keypoints[:, 0] = _project_skeleton(test_skel, K0, R0, t0)
    keypoints[:, 1] = _project_skeleton(test_skel, K1, R1, t1)
    scores = np.ones((num_frames, num_cams, num_kpts), dtype=np.float32)

    np.savez(
        pose_path,
        keypoints=keypoints,
        scores=scores,
        fps=np.asarray(30.0),
        image_sizes=np.array([[1920, 1080], [1920, 1080]]),
        camera_names=np.array(["cam0", "cam1"]),
        sync_status="VERIFIED_AUDIO_CLAP",
    )

    # 3. CLI raw triangulation with production checkerboard -> scale factor is 1.0
    ret_raw = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path_raw),
        "--raw",
        "--allow-unverified-sync",
    ])
    assert ret_raw == 0
    assert out_path_raw.exists()
    raw_data = np.load(out_path_raw)
    assert str(raw_data["units"]) == "meters"
    assert float(raw_data["scale_factor"]) == 1.0

    # 4. CLI standard (non-raw) triangulation with production checkerboard -> scale MUST BE PRESERVED (1.0)
    ret_prod = main([
        "triangulate",
        str(pose_path),
        str(calib_path),
        "-o", str(out_path_prod),
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret_prod == 0
    assert out_path_prod.exists()
    prod_data = np.load(out_path_prod)
    assert str(prod_data["scale_action"]) == "PRESERVE_METRIC"
    assert float(prod_data["scale_factor"]) == 1.0
    assert str(prod_data["scale_source"]) == "checkerboard"
    assert str(prod_data["units"]) == "meters"

    # 5. CLI triangulation on exploratory calibration WITHOUT --allow-exploratory-calibration -> REJECTED (exit code 3)
    ret_expl_fail = main([
        "triangulate",
        str(pose_path),
        str(exploratory_calib_path),
        "-o", str(out_path_expl),
        "--allow-unverified-sync",
        "--allow-unsafe-geometry",
    ])
    assert ret_expl_fail == 3

    # 6. CLI triangulation on exploratory calibration WITH --allow-exploratory-calibration -> ALLOWED
    ret_expl_ok = main([
        "triangulate",
        str(pose_path),
        str(exploratory_calib_path),
        "-o", str(out_path_expl),
        "--allow-unverified-sync",
        "--allow-exploratory-calibration",
        "--allow-unsafe-geometry",
    ])
    assert ret_expl_ok == 0
    assert out_path_expl.exists()
    expl_data = np.load(out_path_expl)
    assert str(expl_data["scale_action"]) == "ESTIMATE_ANTHROPOMETRIC"
    assert str(expl_data["scale_source"]) == "anthropometric"
