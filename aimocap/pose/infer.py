"""2D whole-body pose inference via CIGPose (ONNX Runtime, GPU).

Wraps cigpose's low-level API so we get raw keypoint data (not frame drawings).
The high-level ``cigpose.infer_persons`` draws onto the frame — useless for a
data pipeline — so we drive the pieces ourselves:

    frame -> YOLOX.detect -> [bbox]
           -> preprocess_person -> pose_session.run -> decode_simcc
           -> remap_to_frame  ->  Pose2D(keypoints, scores, bbox)

A ``PoseEstimator`` holds the loaded ONNX sessions (detector + pose model) and
the cigpose bookkeeping (input_w/h, split_ratio). Construct once, call
``.estimate(frame)`` per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# cigpose imports onnxruntime; importing aimocap first ensures CUDA DLLs load.
import aimocap  # noqa: F401
try:
    from cigpose import (
        YOLOXDetector,
        decode_simcc,
        load_pose_model,
        preprocess_person,
    )
    _CIGPOSE_AVAILABLE = True
except ImportError:
    _CIGPOSE_AVAILABLE = False
    YOLOXDetector = None
    decode_simcc = None
    load_pose_model = None
    preprocess_person = None


def _decode_simcc_subpixel(simcc_x, simcc_y, split_ratio: float):
    """Decode SimCC with a clipped three-bin parabola refinement."""
    outputs = []
    for values in (np.asarray(simcc_x)[0], np.asarray(simcc_y)[0]):
        peak = np.argmax(values, axis=-1)
        location = peak.astype(np.float64)
        bins = values.shape[-1]
        if bins >= 3:
            center = np.clip(peak, 1, bins - 2)
            left = np.take_along_axis(values, (center - 1)[:, None], axis=1)[:, 0]
            middle = np.take_along_axis(values, center[:, None], axis=1)[:, 0]
            right = np.take_along_axis(values, (center + 1)[:, None], axis=1)[:, 0]
            denominator = left - 2.0 * middle + right
            safe_denominator = np.where(
                np.abs(denominator) > 1e-12, denominator, 1.0
            )
            offset = 0.5 * (left - right) / safe_denominator
            offset = np.where(
                (peak > 0) & (peak < bins - 1),
                np.clip(offset, -0.5, 0.5),
                0.0,
            )
            location = location + offset
        outputs.append(location / split_ratio)
    scores = np.minimum(np.asarray(simcc_x)[0].max(-1), np.asarray(simcc_y)[0].max(-1))
    return np.stack(outputs, axis=-1), scores

from aimocap.pose.transforms import get_crop_affine_transform, apply_affine_transform
from aimocap.pose.inference_session import InferenceSession, YOLOXDNNDetector

from aimocap.pose.keypoints import N_KEYPOINTS

# Default model + detector paths (relative to project root).
_DEFAULT_POSE_MODEL = "models/cigpose-x_coco-wholebody_384x288.onnx"
_DEFAULT_RTMW_POSE_MODEL = (
    "models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
)


def _preprocess_person_with_stats(
    frame: np.ndarray,
    bbox: list[float],
    input_w: int,
    input_h: int,
    mean: np.ndarray,
    std: np.ndarray,
):
    """CIGPose-compatible crop/resize with caller-selected normalization."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    aspect = input_w / input_h
    if bw / max(bh, 1) > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    bw *= 1.25
    bh *= 1.25

    sx1 = int(max(0, cx - bw / 2))
    sy1 = int(max(0, cy - bh / 2))
    sx2 = int(min(frame.shape[1], cx + bw / 2))
    sy2 = int(min(frame.shape[0], cy + bh / 2))
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        crop = frame

    resized = cv2.resize(crop, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis]
    return tensor, (sx1, sy1, sx2, sy2)


def _load_rtmw_pose_model(
    path: str,
    providers: list[str],
    *,
    inference_engine: str = "ort",
    dnn_target: str = "cpu",
):
    """Load RTMW and derive its input/output geometry from model metadata."""
    session = InferenceSession(
        path,
        engine=inference_engine,
        providers=providers,
        target=dnn_target,
        input_shape=(1, 3, 384, 288),
    )
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape
    if input_shape is None or len(input_shape) != 4:
        raise ValueError(f"RTMW input must be rank 4, got {input_shape!r}")
    try:
        input_h = int(input_shape[-2])
        input_w = int(input_shape[-1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RTMW input spatial shape must be static, got {input_shape!r}") from exc

    output_meta = list(session.get_outputs())
    output_names = [str(meta.name) for meta in output_meta]
    if len(output_names) < 2:
        raise ValueError("RTMW model must expose simcc_x and simcc_y outputs")
    try:
        x_index = next(i for i, name in enumerate(output_names) if name == "simcc_x")
        y_index = next(i for i, name in enumerate(output_names) if name == "simcc_y")
    except StopIteration as exc:
        raise ValueError(f"RTMW outputs must include simcc_x/simcc_y, got {output_names!r}") from exc

    def _last_dim(meta, fallback):
        shape = getattr(meta, "shape", None)
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            return fallback

    x_bins = _last_dim(output_meta[x_index], input_w * 2)
    y_bins = _last_dim(output_meta[y_index], input_h * 2)
    split_x = x_bins / float(input_w)
    split_y = y_bins / float(input_h)
    if not np.isclose(split_x, split_y):
        raise ValueError(f"RTMW SimCC split differs for x/y: {split_x}, {split_y}")
    return (
        session,
        str(input_meta.name),
        input_w,
        input_h,
        float(split_x),
        output_names,
        x_index,
        y_index,
    )


_DEFAULT_DETECTOR = "models/yolox_nano.onnx"
_RTMW_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_RTMW_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass(slots=True)
class Pose2D:
    """One person's 2D pose in a single frame.

    keypoints: (133, 2) float32, pixel coords in the original frame.
    scores:    (133,)   float32, per-keypoint confidence in [0, ~1].
    bbox:      (4,)     float32, [x1, y1, x2, y2] person box in pixels.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray

    def __post_init__(self) -> None:
        if self.keypoints.shape != (N_KEYPOINTS, 2):
            raise ValueError(
                f"keypoints shape {self.keypoints.shape}, expected ({N_KEYPOINTS}, 2)"
            )
        if self.scores.shape != (N_KEYPOINTS,):
            raise ValueError(
                f"scores shape {self.scores.shape}, expected ({N_KEYPOINTS},)"
            )
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox shape {self.bbox.shape}, expected (4,)")

    @property
    def area(self) -> float:
        """Bounding-box area in pixels^2 (used to pick the largest person)."""
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    def mean_score(self, mask: np.ndarray | slice | None = None) -> float:
        """Mean confidence over the given keypoint subset (default: all)."""
        s = self.scores if mask is None else self.scores[mask]
        return float(s.mean()) if s.size else 0.0


class PoseEstimator:
    """Loads a CIGPose or opt-in RTMW detector + pose model and runs inference.

    Parameters
    ----------
    pose_model : path to the whole-body ONNX pose model. If omitted, the
                 model-family default is used.
    detector   : path to the YOLOX-Nano ONNX detector (None = no detection;
                 the full frame is treated as one person — useful for pre-cropped
                 single-subject inputs).
    providers  : ONNX Runtime provider list. Defaults to CUDA-first.
    det_threshold / nms_threshold : YOLOX detection tuning.
    simcc_decode : ``argmax`` or the opt-in ``subpixel`` decoder.
    model_family : ``cigpose`` (default) or ``rtmw``.
    """

    def __init__(
        self,
        pose_model: str | Path | None = None,
        detector: str | Path | None = _DEFAULT_DETECTOR,
        providers: list[str] | None = None,
        det_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        simcc_decode: str = "argmax",
        model_family: str = "cigpose",
        score_calibration: str = "auto",
        inference_engine: str = "ort",
        dnn_target: str = "cpu",
    ) -> None:
        # score_calibration: "auto" = calibrated probabilities for rtmw, raw
        # scores for cigpose (unchanged); "off" = always raw scores.
        if score_calibration not in {"auto", "off"}:
            raise ValueError(f"score_calibration must be 'auto' or 'off', got {score_calibration!r}")
        self.calibrate = score_calibration == "auto" and model_family == "rtmw"
        if self.calibrate:
            from aimocap.pose.score_calibration import load_calibration

            self.score_calibration_table = load_calibration()
        else:
            self.score_calibration_table = None
        if model_family not in {"cigpose", "rtmw"}:
            raise ValueError(
                f"model_family must be 'cigpose' or 'rtmw', got {model_family!r}"
            )
        if simcc_decode not in {"argmax", "subpixel"}:
            raise ValueError(
                f"simcc_decode must be 'argmax' or 'subpixel', got {simcc_decode!r}"
            )
        if inference_engine not in {"ort", "cv2dnn"}:
            raise ValueError(
                f"inference_engine must be 'ort' or 'cv2dnn', got {inference_engine!r}"
            )
        if dnn_target not in {"cpu", "opencl"}:
            raise ValueError(f"dnn_target must be 'cpu' or 'opencl', got {dnn_target!r}")
        self.model_family = model_family
        self.simcc_decode = simcc_decode
        self.inference_engine = inference_engine
        self.dnn_target = dnn_target
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        if pose_model is None:
            pose_model = (
                _DEFAULT_POSE_MODEL
                if model_family == "cigpose"
                else _DEFAULT_RTMW_POSE_MODEL
            )
        pose_path = str(pose_model)
        if not Path(pose_path).exists():
            raise FileNotFoundError(f"pose model not found: {pose_path}")

        self._pose_output_names = None
        self._pose_x_index = 0
        self._pose_y_index = 1
        if model_family == "cigpose" and inference_engine == "ort":
            (
                self.pose_session,
                self.input_w,
                self.input_h,
                self.split_ratio,
            ) = load_pose_model(pose_path, providers=self.providers)
            # Keep CIGPose's exact preprocessing constants and call path.
            from cigpose import inference as cigpose_inference

            self._mean = cigpose_inference._MEAN
            self._std = cigpose_inference._STD
            self.input_name = "input"
        elif model_family == "cigpose":
            self.pose_session = InferenceSession(
                pose_path,
                engine=inference_engine,
                providers=self.providers,
                target=dnn_target,
            )
            input_shape = self.pose_session.input_shape
            if input_shape is None or len(input_shape) != 4:
                raise ValueError(f"CIGPose input must be rank 4, got {input_shape!r}")
            self.input_h, self.input_w = int(input_shape[-2]), int(input_shape[-1])
            self.input_name = self.pose_session.input_name
            output_names = self.pose_session.output_names
            self._pose_output_names = output_names
            self._pose_x_index = 0
            self._pose_y_index = 1
            from cigpose import inference as cigpose_inference

            self._mean = cigpose_inference._MEAN
            self._std = cigpose_inference._STD
            self.split_ratio = 2.0
        else:
            (
                self.pose_session,
                self.input_name,
                self.input_w,
                self.input_h,
                self.split_ratio,
                self._pose_output_names,
                self._pose_x_index,
                self._pose_y_index,
            ) = _load_rtmw_pose_model(
                pose_path,
                self.providers,
                inference_engine=inference_engine,
                dnn_target=dnn_target,
            )
            self._mean = _RTMW_MEAN
            self._std = _RTMW_STD

        if detector is None:
            self.detector = None
        else:
            det_path = str(detector)
            if not Path(det_path).exists():
                raise FileNotFoundError(f"detector not found: {det_path}")
            if inference_engine == "ort":
                self.detector = YOLOXDetector(
                    det_path,
                    conf_thresh=det_threshold,
                    nms_thresh=nms_threshold,
                    providers=self.providers,
                )
            else:
                self.detector = YOLOXDNNDetector(
                    det_path,
                    conf_thresh=det_threshold,
                    nms_thresh=nms_threshold,
                    engine=inference_engine,
                    providers=self.providers,
                    target=dnn_target,
                )

        # Surface which provider/backend actually got used (for logging).
        self.active_providers = self.pose_session.get_providers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, pick: str = "largest") -> list[Pose2D]:
        """Estimate poses for all detected persons in ``frame``.

        ``pick`` only affects the return when a detector is configured:
          - "all"      : return every detected person
          - "largest"  : return only the largest-area person (list of len 1)
          - "best"     : return only the highest mean-confidence person
        Without a detector, returns a single Pose2D for the whole frame.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 BGR, got {frame.shape}")

        bboxes = self._detect(frame)
        poses = [self._estimate_one(frame, np.asarray(b, dtype=np.float32))
                 for b in bboxes]

        if pick == "all" or not poses:
            return poses
        if pick == "largest":
            return [max(poses, key=lambda p: p.area)]
        if pick == "best":
            return [max(poses, key=lambda p: p.mean_score())]
        raise ValueError(f"unknown pick={pick!r}; use 'all'|'largest'|'best'")

    def estimate_with_crop(
        self, frame: np.ndarray, crop_xyxy: np.ndarray | list[float]
    ) -> Pose2D:
        """Estimate one pose from a uniform-scale crop in frame coordinates."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be HxWx3 BGR, got {frame.shape}")
        crop = np.asarray(crop_xyxy, dtype=np.float32).reshape(-1)
        if crop.size != 4 or not np.isfinite(crop).all():
            raise ValueError("crop_xyxy must contain four finite coordinates")
        x1, y1, x2, y2 = (float(value) for value in crop)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"crop_xyxy must have positive size, got {crop_xyxy!r}")

        scale = self.input_w / (x2 - x1)
        scale_x = scale_y = scale
        matrix = np.array(
            [[scale_x, 0.0, -x1 * scale_x], [0.0, scale_y, -y1 * scale_y]],
            dtype=np.float64,
        )
        resized = cv2.warpAffine(
            frame,
            matrix,
            (self.input_w, self.input_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - self._mean) / self._std).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._run_pose(tensor)
        if self.simcc_decode == "argmax":
            kpts, scores = decode_simcc(
                simcc_x, simcc_y, self.input_w, self.input_h, self.split_ratio
            )
        else:
            kpts, scores = _decode_simcc_subpixel(
                simcc_x, simcc_y, self.split_ratio
            )
        kpts = np.asarray(kpts, dtype=np.float64)
        kpts[:, 0] = kpts[:, 0] / scale_x + x1
        kpts[:, 1] = kpts[:, 1] / scale_y + y1
        return Pose2D(
            keypoints=kpts.astype(np.float32),
            scores=self._final_scores(scores),
            bbox=crop.astype(np.float32),
        )

    def _final_scores(self, scores) -> np.ndarray:
        """Raw scores (cigpose default) or calibrated probabilities (rtmw)."""
        if not getattr(self, "calibrate", False):
            return np.asarray(scores).astype(np.float32)
        from aimocap.pose.score_calibration import calibrate_scores

        return calibrate_scores(np.asarray(scores), self.score_calibration_table).astype(np.float32)

    def _run_pose(self, tensor: np.ndarray):
        """Run the selected pose model and return x/y SimCC tensors."""
        if getattr(self, "model_family", "cigpose") == "cigpose":
            return self.pose_session.run(None, {"input": tensor})
        outputs = self.pose_session.run(self._pose_output_names, {self.input_name: tensor})
        return outputs[self._pose_x_index], outputs[self._pose_y_index]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> list[list[float]]:
        if self.detector is None:
            h, w = frame.shape[:2]
            return [[0.0, 0.0, float(w), float(h)]]
        return self.detector.detect(frame)

    def _estimate_one(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Pose2D:
        if getattr(self, "model_family", "cigpose") == "cigpose":
            tensor, crop_region = preprocess_person(
                frame, bbox.tolist(), self.input_w, self.input_h
            )
        else:
            tensor, crop_region = _preprocess_person_with_stats(
                frame,
                bbox.tolist(),
                self.input_w,
                self.input_h,
                self._mean,
                self._std,
            )
        simcc_x, simcc_y = self._run_pose(tensor)
        if self.simcc_decode == "argmax":
            kpts, scores = decode_simcc(
                simcc_x, simcc_y, self.input_w, self.input_h, self.split_ratio
            )
        else:
            kpts, scores = _decode_simcc_subpixel(
                simcc_x, simcc_y, self.split_ratio
            )
        
        # Rigorous geometric un-crop using explicit affine inverse
        h, w = frame.shape[:2]
        M, M_inv, _ = get_crop_affine_transform(bbox, self.input_w, self.input_h, w, h)
        kpts = apply_affine_transform(kpts, M_inv)
        
        return Pose2D(
            keypoints=kpts.astype(np.float32),
            scores=self._final_scores(scores),
            bbox=bbox,
        )


__all__ = ["Pose2D", "PoseEstimator"]
