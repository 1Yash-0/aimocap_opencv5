"""CPU-only tests for the opt-in RTMW-x PoseEstimator backend."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from aimocap import cli
from aimocap.pose import infer


class _FakeRTMWSession:
    def __init__(self):
        self.feed = None

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=["batch", 3, 384, 288])]

    def get_outputs(self):
        return [
            SimpleNamespace(name="simcc_x", shape=["batch", 133, 576]),
            SimpleNamespace(name="simcc_y", shape=["batch", 133, 768]),
        ]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, feeds):
        self.feed = feeds["input"]
        return [
            np.zeros((1, 133, 576), dtype=np.float32),
            np.zeros((1, 133, 768), dtype=np.float32),
        ]


def test_rtmw_normalization_constants_are_used(monkeypatch):
    session = _FakeRTMWSession()
    fake_ort = SimpleNamespace(
        InferenceSession=lambda _path, providers: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(infer.Path, "exists", lambda _self: True)

    estimator = infer.PoseEstimator(
        pose_model="mock-rtmw.onnx",
        model_family="rtmw",
        detector=None,
        providers=["CPUExecutionProvider"],
    )
    frame = np.empty((384, 288, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)  # BGR; RTMW preprocessing is RGB.
    estimator.estimate_with_crop(frame, [0, 0, 288, 384])

    expected = (np.asarray([30.0, 20.0, 10.0]) - infer._RTMW_MEAN) / infer._RTMW_STD
    assert session.feed.shape == (1, 3, 384, 288)
    np.testing.assert_allclose(session.feed[0, :, 0, 0], expected, rtol=0, atol=1e-6)
    assert estimator.input_w == 288
    assert estimator.input_h == 384
    assert estimator.split_ratio == 2.0


def test_default_estimator_family_remains_cigpose(monkeypatch):
    class _CigSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(infer.Path, "exists", lambda _self: True)
    monkeypatch.setattr(
        infer,
        "load_pose_model",
        lambda _path, providers: (_CigSession(), 288, 384, 2.0),
    )
    estimator = infer.PoseEstimator(pose_model="mock-cigpose.onnx", detector=None)
    assert estimator.model_family == "cigpose"


def test_cli_accepts_rtmw_backend():
    args = cli.build_parser().parse_args(
        ["multipose", "a.mp4", "-o", "b.npz", "--pose-backend", "rtmw"]
    )
    assert args.pose_backend == "rtmw"
