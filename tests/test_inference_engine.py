"""Parity and selection tests for the optional OpenCV 5 DNN backend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aimocap import cli
from aimocap.pose.inference_session import InferenceSession


MODEL_ROOT = Path(
    r"E:\Chaos\Hackathons\OPENCVxAIMOCAP\models"
)
RTMW = MODEL_ROOT / "rtmw" / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
HAND = MODEL_ROOT / "rtmpose_hand" / "20230831" / "rtmpose_onnx" / "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320" / "end2end.onnx"
YOLOX = MODEL_ROOT / "yolox_nano.onnx"


def _require_models(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip("real parity model files missing: " + ", ".join(missing))


def test_default_engine_is_ort():
    args = cli.build_parser().parse_args(["multipose", "a.mp4", "-o", "b.npz"])
    assert args.inference_engine == "ort"




def test_pose_estimator_default_engine_is_ort(monkeypatch):
    from aimocap.pose import infer

    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(infer.Path, "exists", lambda _self: True)
    monkeypatch.setattr(
        infer,
        "load_pose_model",
        lambda _path, providers: (FakeSession(), 288, 384, 2.0),
    )
    estimator = infer.PoseEstimator(pose_model="mock.onnx", detector=None)
    assert estimator.inference_engine == "ort"
    assert estimator.active_providers == ["CPUExecutionProvider"]


def test_cli_accepts_cv2dnn_engine():
    args = cli.build_parser().parse_args(
        ["multipose", "a.mp4", "-o", "b.npz", "--inference-engine", "cv2dnn"]
    )
    assert args.inference_engine == "cv2dnn"


def _assert_model_parity(model: Path, shape: tuple[int, ...]) -> None:
    _require_models(model)
    rng = np.random.default_rng(20261025)
    tensor = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    ort_session = InferenceSession(model, engine="ort", providers=["CPUExecutionProvider"])
    dnn_session = InferenceSession(model, engine="cv2dnn", target="cpu", input_shape=shape)
    ort_outputs = ort_session.run(None, {ort_session.input_name: tensor})
    dnn_outputs = dnn_session.run(None, {dnn_session.input_name: tensor})
    assert len(ort_outputs) == len(dnn_outputs)
    maximum = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in zip(ort_outputs, dnn_outputs)
    )
    assert maximum <= 1e-3, f"{model.name} max abs output diff={maximum:.8g}"


def test_rtmw_cv2dnn_matches_ort():
    _assert_model_parity(RTMW, (1, 3, 384, 288))


def test_hand_cv2dnn_matches_ort():
    _assert_model_parity(HAND, (1, 3, 256, 256))


def test_yolox_cv2dnn_matches_ort():
    _assert_model_parity(YOLOX, (1, 3, 416, 416))
