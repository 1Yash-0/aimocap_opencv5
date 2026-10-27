from types import SimpleNamespace

import numpy as np
import pytest

from aimocap.pipeline.gates import (
    GateFailureReport,
    _weighted_percentile,
    check_raw_geometry_gates,
)


def _diagnostics(*, finite=True, reprojection=1.0, ray_angle=20.0):
    points = np.zeros((10, 23, 3), dtype=np.float64)
    if not finite:
        points[:, :3] = np.nan
    inliers = np.ones((10, 23, 3), dtype=bool)
    reproj = np.full((10, 23, 3), reprojection, dtype=np.float32)
    positive = np.ones((10, 23, 3), dtype=bool)
    return SimpleNamespace(
        points3d=points,
        inlier_mask=inliers,
        num_inliers=np.full((10, 23), 3, dtype=np.uint8),
        reprojection_error_px=reproj,
        positive_depth_mask=positive,
        ray_angle_deg=np.full((10, 23), ray_angle, dtype=np.float32),
        confidence=np.ones((10, 23), dtype=np.float32),
    )


def test_raw_geometry_gate_accepts_observable_geometry():
    result = check_raw_geometry_gates(_diagnostics(), "calib", "input")

    assert not isinstance(result, GateFailureReport)
    assert result.status == "PASS"


def test_raw_geometry_gate_rejects_low_core_coverage():
    result = check_raw_geometry_gates(_diagnostics(finite=False), "calib", "input")

    assert isinstance(result, GateFailureReport)
    assert result.failing_gate == "core_finite_coverage"


def _soft_diagnostics() -> SimpleNamespace:
    """Soft diagnostics where support != inlier and residual trust varies."""
    diagnostics = _diagnostics()
    support = np.ones((10, 23, 3), dtype=bool)
    support[..., 2] = False
    trust = np.zeros((10, 23, 3), dtype=np.float32)
    trust[..., 0] = 0.9
    trust[..., 1] = 0.4
    trust[..., 2] = 0.2
    reprojection = np.empty((10, 23, 3), dtype=np.float32)
    reprojection[..., 0] = 1.0
    reprojection[..., 1] = 5.0
    reprojection[..., 2] = 50.0
    diagnostics.support_mask = support
    diagnostics.camera_trust = trust
    diagnostics.reprojection_error_px = reprojection
    return diagnostics


def test_soft_gate_uses_support_and_trust_weighted_residuals():
    result = check_raw_geometry_gates(
        _soft_diagnostics(), "calib", "input", camera_policy="soft"
    )

    assert not isinstance(result, GateFailureReport)
    gates = result.gate_results
    assert gates["core_two_camera_coverage"]["value"] == 1.0
    # Equal-count 1 px / 5 px support residuals.  The 0.9-vs-0.4 trust weights
    # pull the pooled median into the 1 px block; unweighted it would be 3 px.
    assert gates["reprojection_median"]["value"] == 1.0
    assert gates["reprojection_p95"]["value"] == pytest.approx(5.0)
    # The informational legacy number uses the historical 12 px hard-inlier
    # threshold, not the soft trust>0.5 mask: cameras 0 (1 px) and 1 (5 px).
    assert gates["legacy_two_camera_coverage"]["value"] == 1.0
    assert gates["legacy_two_camera_coverage"]["hard_inlier_threshold_px"] == 12.0


def test_soft_gate_legacy_two_camera_uses_hard_12px_inliers():
    diagnostics = _soft_diagnostics()
    diagnostics.reprojection_error_px = np.full((10, 23, 3), 20.0, dtype=np.float32)
    diagnostics.reprojection_error_px[..., 0] = 1.0

    result = check_raw_geometry_gates(
        diagnostics, "calib", "input", camera_policy="soft"
    )

    assert isinstance(result, GateFailureReport)
    legacy = result.evidence.gate_results["legacy_two_camera_coverage"]
    assert legacy["value"] == 0.0
    assert legacy["hard_inlier_threshold_px"] == 12.0


def test_weighted_percentile_interpolates_with_weights():
    value = _weighted_percentile(
        np.array([1.0, 5.0]), np.array([0.9, 0.4]), 50.0
    )

    assert value == pytest.approx(2.2308, abs=1e-3)


def test_legacy_gate_keeps_hard_inlier_residuals():
    result = check_raw_geometry_gates(_soft_diagnostics(), "calib", "input")

    assert isinstance(result, GateFailureReport)
    assert result.failing_gate == "reprojection_median"
    assert result.evidence.gate_results["reprojection_median"]["value"] == 5.0


def test_soft_gate_falls_back_to_trust_threshold_without_support_mask():
    diagnostics = _soft_diagnostics()
    del diagnostics.support_mask

    result = check_raw_geometry_gates(
        diagnostics, "calib", "input", camera_policy="soft"
    )

    assert not isinstance(result, GateFailureReport)
    assert result.gate_results["core_two_camera_coverage"]["value"] == 1.0
