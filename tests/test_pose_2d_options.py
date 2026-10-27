"""CPU-only tests for opt-in 2D pose options."""

from __future__ import annotations

import numpy as np

from aimocap.pose import infer
from aimocap.pose.multi_video import smooth_boxes
from aimocap.pose.keypoints import N_KEYPOINTS


def _simcc_peak(frac: float, bins: int = 32):
    x = np.arange(bins, dtype=np.float64)
    row = -(x - frac) ** 2
    values = np.zeros((1, N_KEYPOINTS, bins), dtype=np.float64)
    values[:] = row
    return values, values.copy()


def test_subpixel_equals_argmax_for_symmetric_peak():
    simcc_x, simcc_y = _simcc_peak(12.0)
    decoded, scores = infer._decode_simcc_subpixel(simcc_x, simcc_y, 2.0)
    assert np.all(decoded == np.array([6.0, 6.0]))
    assert np.all(scores == 0.0)


def test_subpixel_recovers_fractional_parabola_peak():
    target = 12.35
    simcc_x, simcc_y = _simcc_peak(target)
    decoded, _ = infer._decode_simcc_subpixel(simcc_x, simcc_y, 1.0)
    assert abs(float(decoded[0, 0]) - target) < 0.05
    assert abs(float(decoded[0, 1]) - target) < 0.05


def test_smooth_boxes_are_safe_and_finite_with_missing_boxes():
    aspect = 2.0
    raw = np.array(
        [
            [10.0, 20.0, 30.0, 80.0],
            [np.nan, np.nan, np.nan, np.nan],
            [4.0, 10.0, 44.0, 100.0],
            [15.0, 25.0, 35.0, 70.0],
        ]
    )
    crops = smooth_boxes(raw, aspect, pad=1.25, win=1, extra=1.05)
    assert len(crops) == len(raw)
    for i, (_, crop) in enumerate(crops):
        assert np.isfinite(crop).all()
        width = crop[2] - crop[0]
        height = crop[3] - crop[1]
        for j in range(max(0, i - 1), min(len(raw), i + 2)):
            if np.isfinite(raw[j]).all():
                raw_width = raw[j, 2] - raw[j, 0]
                raw_height = raw[j, 3] - raw[j, 1]
                assert width >= raw_width
                assert height >= raw_height


def test_default_estimator_path_calls_cigpose_decode(monkeypatch):
    estimator = infer.PoseEstimator.__new__(infer.PoseEstimator)
    estimator.input_w = 288
    estimator.input_h = 384
    estimator.split_ratio = 2.0
    estimator.simcc_decode = "argmax"
    estimator.pose_session = type(
        "Session",
        (),
        {"run": lambda self, _outputs, _feeds: _simcc_peak(12.0)},
    )()
    calls = []

    def fake_preprocess(*_args):
        return np.zeros((1, 3, 384, 288), dtype=np.float32), None

    def fake_decode(*args):
        calls.append(args)
        return np.zeros((N_KEYPOINTS, 2)), np.ones(N_KEYPOINTS)

    monkeypatch.setattr(infer, "preprocess_person", fake_preprocess)
    monkeypatch.setattr(infer, "decode_simcc", fake_decode)
    pose = estimator._estimate_one(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.array([10.0, 10.0, 80.0, 90.0], dtype=np.float32),
    )
    assert calls
    assert pose.keypoints.shape == (N_KEYPOINTS, 2)
