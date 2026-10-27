"""Per-point 3D confidence (P3.6).

Covered contracts:
* the shipped model has the documented monotone sign constraints;
* confidence is a finite probability in [0, 1] and NaN-safe;
* it is monotone in trusted-camera count (increasing) and residual (decreasing);
* enabling it does not change the triangulated points or any P3.5 diagnostic;
* the default/off path remains bit-identical to the pinned legacy fixture and
  the confidence3d key is absent unless explicitly requested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aimocap.triangulate import confidence3d
from aimocap.triangulate.config import GeometryConfig
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from tests.test_triangulate_soft import BASE_CONFIG, _soft_config, _synthetic_sequence

FIXTURE = Path(__file__).parent / "fixtures" / "triangulate_soft_legacy.npz"


def _features(trust, residual, ray=60.0, innovation=None, score=0.9):
    trust = np.asarray(trust, dtype=np.float64)[None, None, :]
    error = np.asarray(residual, dtype=np.float64)[None, None, :]
    ray = np.asarray([[ray]], dtype=np.float64)
    innovation = (
        None if innovation is None else np.asarray([[innovation]], dtype=np.float64)
    )
    scores = np.full((1, trust.shape[2], 1), score, dtype=np.float64)
    return confidence3d.build_features(
        trust,
        error,
        ray,
        scores=scores,
        innovation_m=innovation,
    )


def test_shipped_model_is_loaded_with_expected_sign_constraints():
    model = confidence3d.load_model()
    assert tuple(model["feature_names"]) == confidence3d.FEATURE_NAMES
    for name, value in zip(confidence3d.FEATURE_NAMES, model["coefficients"]):
        sign = confidence3d.SIGN_CONSTRAINTS[name]
        assert np.isfinite(value)
        assert value >= 0.0 if sign == "increasing" else value <= 0.0
    assert model["fit"]["body_threshold_mm"] == 30.0
    assert model["fit"]["hands_threshold_mm"] == 15.0
    assert model["fit"]["fit_clip"] == "clip1"


def test_confidence_is_a_finite_probability_and_nan_safe():
    trust = np.array([[[1.0, 0.0, np.nan], [0.2, np.nan, np.nan]]], dtype=np.float64)
    error = np.array([[[2.0, 40.0, np.nan], [np.nan, np.nan, np.nan]]], dtype=np.float64)
    ray = np.array([[45.0, np.nan]], dtype=np.float64)
    scores = np.array([[[0.9, np.nan, 0.1], [0.4, np.nan, np.nan]]], dtype=np.float64)
    features = confidence3d.build_features(
        trust,
        error,
        ray,
        scores=scores,
        innovation_m=np.array([[0.001, np.nan]], dtype=np.float64),
    )
    assert np.isfinite(features).all()
    valid = np.array([[True, False]], dtype=bool)
    values = confidence3d.predict_confidence3d(features, valid=valid)
    assert values.shape == (1, 2)
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0) and np.all(values <= 1.0)
    assert values[0, 1] == 0.0

    # Direct NaN input is replaced by the neutral feature mean, never NaN.
    raw = np.full((1, 2, len(confidence3d.FEATURE_NAMES)), np.nan)
    direct = confidence3d.predict_confidence3d(raw)
    assert np.isfinite(direct).all()
    assert np.all(direct >= 0.0) and np.all(direct <= 1.0)


def test_confidence_is_monotone_in_trusted_camera_count():
    values = []
    for trust in ([0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.9, 0.9, 0.9]):
        features = _features(trust, [5.0, 5.0, 5.0])
        values.append(float(confidence3d.predict_confidence3d(features)[0, 0]))
    assert values[0] <= values[1] <= values[2]


def test_confidence_is_monotone_in_residual_and_smoother_innovation():
    residual_values = []
    for residual in (0.5, 3.0, 5.0, 30.0):
        features = _features([1.0, 1.0, 1.0], [residual] * 3)
        residual_values.append(float(confidence3d.predict_confidence3d(features)[0, 0]))
    assert all(a >= b for a, b in zip(residual_values, residual_values[1:]))

    innovation_values = []
    for innovation in (0.0, 0.002, 0.02, 0.2):
        features = _features(
            [1.0, 1.0, 1.0], [3.0] * 3, innovation=innovation
        )
        innovation_values.append(float(confidence3d.predict_confidence3d(features)[0, 0]))
    assert all(a >= b for a, b in zip(innovation_values, innovation_values[1:]))


def test_confidence_does_not_change_the_triangulation_off_or_on():
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    off = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(),
    )
    assert off.confidence3d is None
    assert off.confidence3d_features is None
    assert off.smoother_innovation_m is None

    on = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        intrinsics,
        extrinsics,
        geometry_config=_soft_config(confidence3d=True),
    )
    np.testing.assert_array_equal(on.points3d, off.points3d)
    np.testing.assert_array_equal(on.confidence, off.confidence)
    np.testing.assert_array_equal(on.camera_trust, off.camera_trust)
    np.testing.assert_array_equal(on.inlier_mask, off.inlier_mask)
    assert on.confidence3d is not None
    assert on.confidence3d.shape == on.points3d.shape[:2]
    assert np.isfinite(on.confidence3d).all()
    assert np.all(on.confidence3d >= 0.0) and np.all(on.confidence3d <= 1.0)
    assert np.all(on.confidence3d[~np.isfinite(on.points3d).all(axis=-1)] == 0.0)


def test_legacy_output_is_bit_identical_and_confidence3d_is_opt_in():
    expected = np.load(FIXTURE)
    config = GeometryConfig(**BASE_CONFIG)
    assert config.confidence3d is False
    keypoints, scores, intrinsics, extrinsics = _synthetic_sequence()
    diagnostics = triangulate_sequence_with_diagnostics(
        keypoints, scores, intrinsics, extrinsics, geometry_config=config
    )
    np.testing.assert_array_equal(diagnostics.points3d, expected["points3d"])
    np.testing.assert_array_equal(diagnostics.confidence, expected["confidence"])
    np.testing.assert_array_equal(diagnostics.camera_trust, expected["camera_trust"])
    assert diagnostics.confidence3d is None


def test_confidence3d_requires_soft_camera_policy():
    with pytest.raises(ValueError, match="camera_policy='soft'"):
        GeometryConfig(confidence3d=True)


def test_fit_returns_json_ready_sign_constrained_model():
    rng = np.random.RandomState(4)
    features = rng.randn(2, 40, len(confidence3d.FEATURE_NAMES))
    labels = features[..., 0] > 0.0
    model = confidence3d.fit_confidence3d(
        features,
        labels,
        is_hand=np.arange(40) >= 20,
        l2=0.05,
    )
    for name, value in zip(confidence3d.FEATURE_NAMES, model["coefficients"]):
        sign = confidence3d.SIGN_CONSTRAINTS[name]
        assert value >= 0.0 if sign == "increasing" else value <= 0.0
    assert model["fit"]["samples"] == 80
    assert "feature_mean" in model and "feature_scale" in model
