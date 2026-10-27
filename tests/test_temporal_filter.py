"""Contract tests for the opt-in P3.5 3D temporal smoother."""

from __future__ import annotations

import numpy as np
import pytest

from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import TriangulationDiagnostics, _apply_temporal_filter
from aimocap.triangulate.temporal_filter import filter_skeleton3d_rts


def _diagnostics(points: np.ndarray, num_cameras: int = 2) -> TriangulationDiagnostics:
    frames, joints, _ = points.shape
    return TriangulationDiagnostics(
        points3d=points,
        candidate_points3d=points.copy(),
        confidence=np.ones((frames, joints), dtype=np.float32),
        candidate_inlier_mask=np.ones((frames, joints, num_cameras), dtype=bool),
        inlier_mask=np.ones((frames, joints, num_cameras), dtype=bool),
        support_mask=np.ones((frames, joints, num_cameras), dtype=bool),
        camera_trust=np.ones((frames, joints, num_cameras), dtype=np.float32),
        reprojection_error_px=np.ones(
            (frames, joints, num_cameras), dtype=np.float32
        ),
        candidate_num_inliers=np.full((frames, joints), num_cameras, dtype=np.uint8),
        num_inliers=np.full((frames, joints), num_cameras, dtype=np.uint8),
        low_quality_mask=np.zeros(frames, dtype=bool),
        ray_angle_deg=np.full((frames, joints), 60.0, dtype=np.float32),
        observed_camera_count=np.full(
            (frames, joints), num_cameras, dtype=np.uint8
        ),
        positive_depth_mask=np.ones(
            (frames, joints, num_cameras), dtype=bool
        ),
        observation_provenance=np.full((frames, joints), "observed_2plus"),
        camera_rejection_reason=np.full(
            (frames, joints, num_cameras), "inlier"
        ),
        lr_swap_mask=np.zeros((frames, num_cameras, 3), dtype=bool),
    )


def test_step_peak_preserved_without_overshoot():
    frames = 200
    points = np.zeros((frames, 1, 3), dtype=np.float64)
    t = np.clip((np.arange(frames) - 90) / 20.0, 0.0, 1.0)
    points[:, 0, 0] = t * t * (3.0 - 2.0 * t)

    filtered = filter_skeleton3d_rts(points, 30.0)

    assert filtered[:, 0, 0].max() <= 1.01
    assert filtered[:, 0, 0].min() >= -0.01
    assert abs(filtered[-1, 0, 0] - 1.0) < 1e-3

    peak = np.zeros((120, 1, 3), dtype=np.float64)
    peak[50:70, 0, 0] = np.concatenate(
        [np.linspace(0.0, 0.5, 10), np.linspace(0.5, 0.0, 10)]
    )
    smoothed_peak = filter_skeleton3d_rts(peak, 30.0)
    assert 0.45 <= smoothed_peak[:, 0, 0].max() <= 0.52


def test_noisy_constant_position_reduces_variance_and_keeps_mean():
    rng = np.random.RandomState(0)
    frames = 300
    points = np.zeros((frames, 1, 3), dtype=np.float64)
    points[:, 0, 0] = 0.15 + rng.normal(0.0, 0.01, frames)

    filtered = filter_skeleton3d_rts(points, 30.0)

    raw_var = float(np.var(points[:, 0, 0]))
    filtered_var = float(np.var(filtered[:, 0, 0]))
    assert filtered_var < 0.8 * raw_var
    assert abs(float(filtered[:, 0, 0].mean()) - 0.15) < 0.002


def test_nan_gaps_are_bridged_only_up_to_the_limit():
    frames = 60
    points = np.zeros((frames, 1, 3), dtype=np.float64)
    points[:, 0, 0] = 0.1
    observed = np.ones(frames, dtype=bool)
    observed[0] = False
    observed[-1] = False
    observed[20:22] = False
    observed[40:50] = False
    points[~observed] = np.nan

    filtered = filter_skeleton3d_rts(points, 30.0, max_gap_s=0.2)

    assert np.isnan(filtered[0, 0, 0])
    assert np.isnan(filtered[-1, 0, 0])
    assert np.isfinite(filtered[20, 0, 0])
    assert np.isfinite(filtered[21, 0, 0])
    assert np.isnan(filtered[45, 0, 0])
    assert np.isfinite(filtered[39, 0, 0])
    assert np.isfinite(filtered[50, 0, 0])


def test_none_filter_is_bit_identical():
    rng = np.random.RandomState(1)
    points = 0.2 + rng.normal(0.0, 0.01, (40, 2, 3))
    diagnostics = _diagnostics(points)
    config = GeometryConfig(temporal_filter="none")

    result = _apply_temporal_filter(diagnostics, config, 30.0)

    assert result is diagnostics
    np.testing.assert_array_equal(result.points3d, points)


def test_enabled_filter_requires_fps():
    diagnostics = _diagnostics(np.zeros((10, 1, 3), dtype=np.float64))
    config = GeometryConfig(temporal_filter="robust_rts")

    with pytest.raises(ValueError, match="fps"):
        _apply_temporal_filter(diagnostics, config, None)


def test_enabled_filter_changes_points_and_respects_nan_mask():
    rng = np.random.RandomState(2)
    frames = 60
    points = np.zeros((frames, 1, 3), dtype=np.float64)
    points[:, 0, 0] = 0.3 + 0.02 * np.sin(np.arange(frames) / 5.0)
    points[:, 0, 0] += rng.normal(0.0, 0.002, frames)
    points[30, 0, :] = np.nan

    diagnostics = _diagnostics(points)
    config = GeometryConfig(temporal_filter="robust_rts")
    result = _apply_temporal_filter(diagnostics, config, 60.0)

    assert result is not diagnostics
    assert result.points3d.shape == points.shape
    assert np.isfinite(result.points3d[30, 0]).all()
    assert np.isfinite(result.points3d[0, 0]).all()
    assert not np.array_equal(result.points3d, points)


def test_spike_guard_rejects_one_frame_spike_but_keeps_step():
    rng = np.random.RandomState(0)
    frames = 120
    base = np.zeros((frames, 1, 3), dtype=np.float64)
    base[:, 0, 0] = 0.2
    base += rng.normal(0.0, 0.002, base.shape)
    spike = base.copy()
    spike[60, 0, 0] += 0.5
    step = base.copy()
    step[60:, 0, 0] += 0.1

    ungated = filter_skeleton3d_rts(spike, 30.0)
    gated = filter_skeleton3d_rts(spike, 30.0, spike_gate=True)
    assert abs(float(ungated[60, 0, 0]) - 0.2) > 0.3
    assert abs(float(gated[60, 0, 0]) - 0.2) < 0.05

    step_gated = filter_skeleton3d_rts(step, 30.0, spike_gate=True)
    assert abs(float(step_gated[-1, 0, 0]) - 0.3) < 0.01
    assert float(step_gated[60:64, 0, 0].max()) > 0.25


def test_fps_scaling_is_consistent():
    rng = np.random.RandomState(3)

    def sample(fps: float, seconds: float = 6.0) -> np.ndarray:
        frames = int(fps * seconds)
        t = np.arange(frames) / fps
        points = np.zeros((frames, 2, 3), dtype=np.float64)
        for joint in range(2):
            points[:, joint, 0] = 0.2 * np.sin(
                2.0 * np.pi * (0.5 + 0.3 * joint) * t + 0.2 * joint
            ) + 0.05 * np.sin(2.0 * np.pi * 1.5 * t)
            points[:, joint, 1] = 0.1 * np.cos(2.0 * np.pi * 0.8 * t + joint)
            points[:, joint, 2] = 0.3 * t
        return points + rng.normal(0.0, 0.004, points.shape)

    at_30 = filter_skeleton3d_rts(sample(30.0), 30.0)
    at_60 = filter_skeleton3d_rts(sample(60.0), 60.0)
    difference = np.abs(at_60[::2] - at_30)

    assert np.nanmedian(difference) < 0.003
    assert np.nanmax(difference) < 0.02
