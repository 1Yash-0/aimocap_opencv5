"""Small common inference-session adapter for ORT and OpenCV DNN.

The adapter deliberately keeps preprocessing and postprocessing outside the
session.  Its only backend-specific operation is loading a graph and running
one tensor through it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


INFERENCE_ENGINES = ("ort", "cv2dnn")
DNN_TARGETS = ("cpu", "opencl")


def _shape_ints(shape: Sequence[Any] | None) -> tuple[int | None, ...] | None:
    if shape is None:
        return None
    result: list[int | None] = []
    for value in shape:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            result.append(None)
    return tuple(result)


class InferenceSession:
    """Backend-neutral ONNX inference session.

    ``ort`` is the production-compatible default.  ``cv2dnn`` uses OpenCV's
    CPU or OpenCL target and returns outputs in the requested/name order.
    ``input_shape`` is only needed for OpenCV because Net has no input metadata
    API; it is also exposed for callers that need preprocessing dimensions.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        engine: str = "ort",
        providers: list[str] | None = None,
        target: str = "cpu",
        input_shape: Sequence[Any] | None = None,
    ) -> None:
        if engine not in INFERENCE_ENGINES:
            raise ValueError(f"engine must be one of {INFERENCE_ENGINES}, got {engine!r}")
        if target not in DNN_TARGETS:
            raise ValueError(f"target must be one of {DNN_TARGETS}, got {target!r}")
        self.engine = engine
        self.target = target
        self.model_path = Path(model_path)
        self._session: Any
        self.input_shape: tuple[int | None, ...] | None
        if engine == "ort":
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            input_meta = self._session.get_inputs()[0]
            self.input_name = str(input_meta.name)
            self.input_shape = _shape_ints(input_meta.shape)
            self.output_names = [str(item.name) for item in self._session.get_outputs()]
            self.active_providers = list(self._session.get_providers())
        else:
            net = cv2.dnn.readNetFromONNX(str(self.model_path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(
                cv2.dnn.DNN_TARGET_OPENCL if target == "opencl" else cv2.dnn.DNN_TARGET_CPU
            )
            self._session = net
            self.input_name = "input"
            self.input_shape = _shape_ints(input_shape)
            self.output_names = [str(name) for name in net.getUnconnectedOutLayersNames()]
            self.active_providers = [f"OpenCV-DNN-{target.upper()}"]

    def get_inputs(self) -> list[Any]:
        if self.engine == "ort":
            return list(self._session.get_inputs())
        return [type("Input", (), {"name": self.input_name, "shape": self.input_shape})()]

    def get_outputs(self) -> list[Any]:
        if self.engine == "ort":
            return list(self._session.get_outputs())
        return [type("Output", (), {"name": name, "shape": None})() for name in self.output_names]

    def get_providers(self) -> list[str]:
        return list(self.active_providers)

    def run(self, output_names: Sequence[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.engine == "ort":
            return list(self._session.run(output_names, feeds))
        if not feeds:
            raise ValueError("cv2dnn inference requires one input tensor")
        tensor = next(iter(feeds.values()))
        self._session.setInput(np.asarray(tensor))
        requested = list(output_names) if output_names is not None else list(self.output_names)
        # Some exported graphs use a name that OpenCV normalizes.  In that
        # case, forwarding all unconnected outputs still preserves graph order.
        if not requested or not set(requested).issubset(set(self.output_names)):
            requested = list(self.output_names)
        result = self._session.forward(requested)
        if isinstance(result, np.ndarray):
            return [result]
        return [np.asarray(value) for value in result]


class YOLOXDNNDetector:
    """YOLOX-Nano detector with the same math as cigpose's detector."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 416,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
        engine: str = "cv2dnn",
        providers: list[str] | None = None,
        target: str = "cpu",
    ) -> None:
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.session = InferenceSession(
            model_path,
            engine=engine,
            providers=providers,
            target=target,
            input_shape=(1, 3, input_size, input_size),
        )
        self.input_name = self.session.input_name
        self._grids, self._strides = self._build_grids()

    def _build_grids(self):
        grids, strides = [], []
        for stride in (8, 16, 32):
            grid = self.input_size // stride
            yv, xv = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
            grids.append(np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32))
            strides.append(np.full((grid * grid, 1), stride, dtype=np.float32))
        return np.concatenate(grids), np.concatenate(strides)

    def _letterbox(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        new_width, new_height = int(width * ratio), int(height * ratio)
        resized = cv2.resize(frame, (new_width, new_height))
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = resized
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        return blob[np.newaxis], ratio

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        blob, ratio = self._letterbox(frame)
        raw = self.session.run(None, {self.input_name: blob})[0][0]
        decoded = raw.copy()
        decoded[:, :2] = (raw[:, :2] + self._grids) * self._strides
        decoded[:, 2:4] = np.exp(raw[:, 2:4]) * self._strides
        scores = decoded[:, 4] * decoded[:, 5]
        keep = scores >= self.conf_thresh
        if not np.any(keep):
            return []
        boxes, scores = decoded[keep, :4], scores[keep]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )
        if len(indices) == 0:
            return []
        return [
            [float(x1[i] / ratio), float(y1[i] / ratio), float(x2[i] / ratio), float(y2[i] / ratio)]
            for i in np.asarray(indices).reshape(-1)
        ]


__all__ = ["DNN_TARGETS", "INFERENCE_ENGINES", "InferenceSession", "YOLOXDNNDetector"]
