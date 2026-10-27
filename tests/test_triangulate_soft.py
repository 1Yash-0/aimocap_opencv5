"""Soft per-camera trust in triangulation (P3.3).

``GeometryConfig.camera_policy`` selects the camera-trust policy:

* ``"legacy"`` is the historical hard-consensus subset search and must stay
  bit-identical (pinned by ``fixtures/triangulate_soft_legacy.npz``, generated
  before the soft path existed).
* ``"soft"`` runs an IRLS robust solve, continuous leave-one-out down-weighting
  and a temporal EMA, and reports per-camera trust instead of hard in/out.

These tests pin the legacy contract, the outlier recovery of the soft solver,
trust stability on a static noisy sequence, the hand finger-chain grouping and
NaN safety.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aimocap.math.coords import opencv_to_internal
from aimocap.math.triangulate import project_points
from aimocap.triangulate import engine as engine_module
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import (
    _soft_prior_solve,
    _soft_support_decision,
    _SoftObservation,
    soft_trust_groups,
    triangulate_sequence_with_diagnostics,
)

FIXTURE = Path(__file__).parent / "fixtures" / "triangulate_soft_legacy.npz"

BASE_CONFIG = dict(
    min_detection_conf=0.5,
    max_reprojection_error_px=5.0,
    hard_reprojection_error_px=12.0,
    min_ray_angle_deg=0.0,
    min_depth_m=0.0,
    f_scale=10.0,
    min_inliers=2,
    hysteresis_margin_px=5.0,
    w_temporal=0.35,
    w_geom=2.0,
    w_anat=2.0,
)


def _camera_at(position, target=np.array([0.0, -1.0, 0.0]), focal=1400.0):
    """OpenCV world-to-camera model (x right, y down, z forward)."""
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, -true_up, forward])
    translation = (-rotation @ position).reshape(3, 1)
    intrinsic = np.array(
        [[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]]
    )
    return intrinsic, rotation, translation


def _cameras():
    return [
        _camera_at([4.0, -1.0, 0.0]),
        _camera_at([0.0, -1.0, 4.0]),
        _camera_at([-4.0, -1.0, 0.0]),
    ]


def _synthetic_sequence():
    """Deterministic 6-frame, 3-camera COCO-WholeBody sequence in OpenCV world."""
    rng = np.random.RandomState(0)
    base = np.zeros((133, 3), dtype=np.float64)
    base[0] = (0.0, -1.62, 0.0)
    base[1] = (-0.03, -1.63, -0.04)
    base[2] = (0.03, -1.63, -0.04)
    base[3] = (-0.08, -1.62, 0.0)
    base[4] = (0.08, -1.62, 0.0)
    base[5] = (-0.18, -1.42, 0.0)
    base[6] = (0.18, -1.42, 0.0)
    base[7] = (-0.18, -1.126, 0.0)
    base[8] = (0.18, -1.12, 0.0)
    base[9] = (-0.18, -0.867, 0.0)
    base[10] = (0.18, -0.856, 0.0)
    base[11] = (-0.09, -1.0, 0.0)
    base[12] = (0.09, -1.0, 0.0)
    base[13] = (-0.09, -0.584, 0.0)
    base[14] = (0.09, -0.605, 0.0)
    base[15] = (-0.09, -0.168, 0.0)
    base[16] = (0.09, -0.230, 0.0)
    base[17] = (-0.09, -0.10, 0.05)
    base[18] = (-0.12, -0.12, 0.0)
    base[19] = (-0.06, -0.12, 0.02)
    base[20] = (0.09, -0.16, 0.05)
    base[21] = (0.12, -0.18, 0.0)
    base[22] = (0.06, -0.18, 0.02)
    base[23:] = np.array([0.0, -1.62, 0.0]) + rng.randn(110, 3) * 0.12
    base[91:112] = base[9] + rng.randn(21, 3) * 0.03
    base[112:133] = base[10] + rng.randn(21, 3) * 0.03

    num_frames = 6
    cameras = _cameras()
    keypoints = np.zeros((num_frames, 3, 133, 2), dtype=np.float64)
    for frame in range(num_frames):
        phase = 2.0 * np.pi * frame / num_frames
        points = base.copy()
        points[:, 1] += 0.015 * np.sin(phase)
        points[[13, 15], 0] += 0.05 * np.sin(phase)
        points[[14, 16], 0] -= 0.05 * np.sin(phase)
        points[[7, 9], 0] += 0.04 * np.sin(phase + np.pi / 3.0)
        points[[8, 10], 0] -= 0.04 * np.sin(phase + np.pi / 3.0)
        for camera, (intrinsic, rotation, translation) in enumerate(cameras):
            projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
            keypoints[frame, camera] = project_points(points, projection)
            keypoints[frame, camera] += rng.randn(133, 2) * 1.2
    keypoints[2, 2, 12] += np.array([6.0, -4.0])
    keypoints[3, 1, 14] += np.array([-7.0, 5.0])
    keypoints[4, 2, 7] += np.array([5.0, 6.0])
    scores = np.ones((num_frames, 3, 133), dtype=np.float64)
    scores[4, 0, 20] = 0.2
    scores[3, 1, 25] = 0.0

    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    return keypoints, scores, intrinsics, extrinsics


def _run(config: GeometryConfig):
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    return triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )


def _soft_config(**overrides) -> GeometryConfig:
    values = dict(BASE_CONFIG)
    values.update(
        camera_policy="soft",
        soft_scale_px=6.0,
        soft_loo_ratio=2.0,
        soft_ema_alpha=0.5,
        soft_iterations=3,
    )
    values.update(overrides)
    return GeometryConfig(**values)


def test_legacy_camera_policy_is_bit_identical_to_previous_engine():
    expected = np.load(FIXTURE)
    diag = _run(GeometryConfig(**BASE_CONFIG))
    assert GeometryConfig(**BASE_CONFIG).camera_policy == "legacy"
    np.testing.assert_array_equal(diag.points3d, expected["points3d"])
    np.testing.assert_array_equal(diag.candidate_points3d, expected["candidate_points3d"])
    np.testing.assert_array_equal(diag.confidence, expected["confidence"])
    np.testing.assert_array_equal(diag.inlier_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(
        diag.reprojection_error_px, expected["reprojection_error_px"]
    )
    np.testing.assert_array_equal(diag.camera_trust, expected["camera_trust"])


def test_legacy_preset_reproduces_p31_fixture_bit_identically():
    expected = np.load(FIXTURE)
    config = GeometryConfig.triangulation_preset("legacy", **BASE_CONFIG)
    assert config.camera_policy == "legacy"
    assert config.temporal_filter == "none"
    assert config.confidence3d is False
    assert config.legacy_unit_behaviour is False

    diag = _run(config)
    np.testing.assert_array_equal(diag.points3d, expected["points3d"])
    np.testing.assert_array_equal(diag.candidate_points3d, expected["candidate_points3d"])
    np.testing.assert_array_equal(diag.confidence, expected["confidence"])
    np.testing.assert_array_equal(diag.inlier_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(diag.support_mask, expected["inlier_mask"])
    np.testing.assert_array_equal(
        diag.reprojection_error_px, expected["reprojection_error_px"]
    )
    np.testing.assert_array_equal(diag.camera_trust, expected["camera_trust"])


def test_production_preset_selects_p37_geometry_defaults():
    config = GeometryConfig.triangulation_preset("production")

    assert config.camera_policy == "soft"
    assert config.soft_scale_px == 3.0
    assert config.soft_loo_ratio == 4.0
    assert config.soft_ema_alpha == 0.5
    assert config.temporal_filter == "robust_rts"
    assert config.confidence3d is True
    assert config.soft_support_min_trust == 0.25
    assert config.lr_swap_fix is False
    # P3.2 per-part cutoffs stay opt-in and default to 0.5 / gamma 1.
    assert config.body_confidence is None
    assert config.feet_confidence is None
    assert config.face_confidence is None
    assert config.hands_confidence is None


def test_triangulation_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown triangulation preset"):
        GeometryConfig.triangulation_preset("experimental")


def _three_camera_frame(outlier_px: float = 60.0, noise: float = 0.5):
    """One static 3D point seen by three consistent cameras, one corrupted."""
    rng = np.random.RandomState(3)
    cameras = _cameras()
    points3d = np.array([[0.05, -1.10, 0.10]])
    observations = []
    for intrinsic, rotation, translation in cameras:
        projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
        observations.append(project_points(points3d, projection)[0])
    observations = np.asarray(observations, dtype=np.float64)
    observations += rng.randn(3, 2) * noise
    observations[2] += np.array([outlier_px, -0.5 * outlier_px])
    return observations, points3d[0]


def test_soft_solve_recovers_point_with_one_large_outlier_camera():
    observations, truth = _three_camera_frame()
    cameras = _cameras()
    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    keypoints = np.zeros((1, 3, 133, 2), dtype=np.float64)
    keypoints[0, :, 0] = observations
    # Neutralise the other joints so only joint 0 is informative.
    keypoints[0, :, 1:] = observations[:, None, :].mean(axis=0)
    scores = np.ones((1, 3, 133), dtype=np.float64)
    config = _soft_config(min_ray_angle_deg=0.0)

    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    error_m = float(
        np.linalg.norm(diag.points3d[0, 0] - opencv_to_internal(truth[None, :])[0])
    )
    assert error_m < 0.02
    assert diag.camera_trust[0, 0, 2] < 0.5
    assert diag.inlier_mask[0, 0, 2] == np.bool_(False)
    assert diag.inlier_mask[0, 0, 0] and diag.inlier_mask[0, 0, 1]


def test_soft_points_emit_when_mask_has_fewer_than_two_cameras():
    """Point emission is decoupled from the >=0.5 trust mask (P3.3)."""
    observations, truth = _three_camera_frame(outlier_px=60.0, noise=3.0)
    cameras = _cameras()
    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    keypoints = np.zeros((1, 3, 133, 2), dtype=np.float64)
    keypoints[0, :, 0] = observations
    keypoints[0, :, 1:] = observations[:, None, :].mean(axis=0)
    scores = np.ones((1, 3, 133), dtype=np.float64)
    config = _soft_config(soft_scale_px=0.5, min_ray_angle_deg=0.0)

    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    assert int(diag.inlier_mask[0, 0].sum()) < 2
    assert np.isfinite(diag.points3d[0, 0]).all()
    assert diag.confidence[0, 0] > 0.0
    error_m = float(
        np.linalg.norm(diag.points3d[0, 0] - opencv_to_internal(truth[None, :])[0])
    )
    assert error_m < 0.12


def test_soft_trust_does_not_flip_flop_on_static_noisy_sequence():
    rng = np.random.RandomState(11)
    cameras = _cameras()
    intrinsics = [item[0] for item in cameras]
    extrinsics = [(item[1], item[2]) for item in cameras]
    points3d = np.array([[0.0, -1.20, 0.05]])
    clean = []
    for intrinsic, rotation, translation in cameras:
        projection = intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
        clean.append(project_points(points3d, projection)[0])
    clean = np.asarray(clean)

    frames = 25
    keypoints = np.zeros((frames, 3, 133, 2), dtype=np.float64)
    for frame in range(frames):
        observed = clean + rng.randn(3, 2) * 0.8
        keypoints[frame, :, 0] = observed
        keypoints[frame, :, 1:] = observed[:, None, :].mean(axis=0)
    scores = np.ones((frames, 3, 133), dtype=np.float64)
    config = _soft_config(soft_ema_alpha=0.8, min_ray_angle_deg=0.0)
    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    trust = diag.camera_trust[:, 0, :]
    assert np.isfinite(trust).all()
    assert float(np.max(np.abs(np.diff(trust, axis=0)))) < 0.35
    switches = np.abs(np.diff(diag.inlier_mask[:, 0, :].astype(np.int8), axis=0)).sum()
    assert int(switches) <= 2


def test_hand_trust_groups_follow_finger_chains():
    groups, assignment = soft_trust_groups(133)
    # Body, feet and face joints stay independent.
    for joint in range(91):
        assert groups[assignment[joint]] == (joint,)
    # Right hand: wrist/palm plus five independent finger chains.
    right = {group for group in groups if any(91 <= joint < 112 for joint in group)}
    assert (91,) in right
    for offsets in ((91, 92, 93, 94, 95), (91, 96, 97, 98, 99),
                    (91, 100, 101, 102, 103), (91, 104, 105, 106, 107),
                    (91, 108, 109, 110, 111)):
        assert offsets in right
    # The finger joints take their own chain's trust; the wrist keeps the palm.
    assert assignment[92] == assignment[95]
    assert assignment[92] != assignment[96]
    assert assignment[91] == next(index for index, group in enumerate(groups) if group == (91,))
    # Left hand chains are separate from right-hand chains.
    assert assignment[92] != assignment[113]


def test_soft_path_is_nan_safe():
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    scores[0, 0, 7] = np.nan
    scores[0, 1, 95] = np.nan
    keypoints[1, 0, 40] = np.nan
    scores[2, 2, 120] = 0.0
    config = _soft_config()
    diag = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    assert diag.points3d.shape == (6, 133, 3)
    assert diag.camera_trust.shape == (6, 133, 3)
    assert np.isfinite(diag.points3d[0, 7]).all()
    assert np.isfinite(diag.points3d[0, 95]).all()
    assert np.isfinite(diag.points3d[1, 40]).all()
    assert np.isfinite(diag.confidence[:, [7, 40, 95]]).all()
    assert np.all(diag.camera_trust >= 0.0)
    assert np.all(diag.camera_trust <= 1.0)


def test_support_decision_requires_two_supporting_and_fitting_cameras():
    off = _soft_config()
    on = _soft_config(soft_support_gate=True)
    trust = np.array([1.0, 0.05, 0.05])
    err = np.array([1.0, 1.0, 1.0])

    emit, support, fit = _soft_support_decision(trust, err, off)
    assert emit
    assert support.tolist() == [True, False, False]
    assert fit.tolist() == [True, False, False]

    emit, support, fit = _soft_support_decision(trust, err, on)
    assert not emit
    assert int(support.sum()) == 1

    # Two trusted cameras but only one of them actually fits the point.
    emit, support, fit = _soft_support_decision(
        np.array([1.0, 1.0, 1.0]), np.array([1.0, 20.0, 20.0]), on
    )
    assert not emit
    assert int(fit.sum()) == 1


def test_support_gate_emits_nan_when_only_one_camera_supports(monkeypatch):
    original = engine_module._soft_group_trust

    def patched(items, **kwargs):
        first, trust = original(items, **kwargs)
        if kwargs.get("frame") == 0 and any(int(joint) == 0 for joint, _ in items):
            trust[1] = 0.05
            trust[2] = 0.05
        return first, trust

    monkeypatch.setattr(engine_module, "_soft_group_trust", patched)
    diag = _run(_soft_config(soft_support_gate=True))
    assert np.isnan(diag.points3d[0, 0]).all()
    assert diag.observation_provenance[0, 0] == "insufficient_support"
    assert np.isfinite(diag.points3d[0, 1]).all()

    monkeypatch.undo()
    diag_off = _run(_soft_config())
    assert np.isfinite(diag_off.points3d[0, 0]).all()


def _projection_matrices():
    return [
        intrinsic @ np.hstack((rotation, translation.reshape(3, 1)))
        for intrinsic, rotation, translation in _cameras()
    ]


def test_temporal_prior_picks_correct_pair_when_reprojection_picks_biased():
    """Camera A is biased, B and C are correct; the prior must pick B+C."""
    point = np.array([0.05, -1.10, 0.10])
    projections = _projection_matrices()
    clean = np.array([project_points(point[None, :], P)[0] for P in projections])
    observed = clean.copy()
    observed[0] += np.array([60.0, 0.0])
    observation = _SoftObservation(
        [0, 1, 2], observed, observed, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], None
    )
    config = _soft_config(
        soft_scale_px=3.0,
        soft_temporal_prior=True,
        soft_prior_w_temporal=0.2,
        soft_prior_w_anat=0.0,
    )

    solved, weights, chosen, _, reprojection_best = _soft_prior_solve(
        observation, projections, 0.0, config, point, None, None
    )

    assert reprojection_best == (0, 1)
    assert chosen == (1, 2)
    assert weights[0] == pytest.approx(0.05)
    assert float(np.linalg.norm(solved - point)) < 1e-3


def test_temporal_prior_lockin_escape_reselects_without_prior(monkeypatch):
    projections = _projection_matrices()
    point = np.array([0.0, -1.10, 0.0])
    observation = _SoftObservation(
        [0, 1, 2],
        np.array([project_points(point[None, :], P)[0] for P in projections]),
        np.array([project_points(point[None, :], P)[0] for P in projections]),
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        None,
    )
    config = _soft_config(soft_temporal_prior=True, soft_prior_lockin_frames=2)
    calls: list[bool] = []

    def fake_solve(observation, P_list, min_depth, cfg, prior=None, **kwargs):
        calls.append(prior is not None)
        return (
            point.copy(),
            np.zeros(len(observation.cams)),
            (0, 1),
            100.0,
            (0, 1, 2),
        )

    monkeypatch.setattr(engine_module, "_soft_observation_solve", fake_solve)
    pts3d = np.zeros((4, 133, 3), dtype=np.float64)
    pts3d[:, 0] = point
    state = np.zeros(133, dtype=np.int64)
    for frame in (2, 3, 4):
        engine_module._soft_group_first_pass(
            [(0, observation)],
            projections,
            0.0,
            config,
            pts3d=pts3d,
            frame=frame,
            lockin_state=state,
        )

    assert calls == [True, True, False]
    assert int(state[0]) == 0


def _write_fixture() -> None:
    """Regenerate the pre-P3.3 legacy fixture (run against the old engine)."""
    diag = _run(GeometryConfig(**BASE_CONFIG))
    np.savez(
        FIXTURE,
        points3d=diag.points3d,
        candidate_points3d=diag.candidate_points3d,
        confidence=diag.confidence,
        inlier_mask=diag.inlier_mask,
        reprojection_error_px=diag.reprojection_error_px,
        camera_trust=diag.inlier_mask.astype(np.float32),
    )
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    _write_fixture()
