"""Opt-in hand and face refinement for COCO-WholeBody 2D poses.

The crop geometry intentionally mirrors the accepted offline experiments:
centres use a centred +/-3-frame median and crop sizes use a +/-5-frame
rolling maximum.  The whole-body pass produces body-calibrated ``scores``;
refinement then maps its model-native hand/face scores through
``aimocap.pose.score_calibration.calibrate_refined_part_scores`` so refined
points also carry calibrated probabilities.  The model-native values remain
available in ``hand_scores_raw`` and ``face_scores_raw``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aimocap.pose.infer import _decode_simcc_subpixel
from aimocap.pose.inference_session import InferenceSession

FACE0 = 23
FACE_COUNT = 68
LEFT_HAND0 = 91
RIGHT_HAND0 = 112
HAND_BASE = {"left": LEFT_HAND0, "right": RIGHT_HAND0}
BODY_ELBOW = {"left": 7, "right": 8}
BODY_WRIST = {"left": 9, "right": 10}
HANDS = ("left", "right")
VISIBILITY = 0.5
H4_SCORE = 0.10
HAND_CROP_FACTOR = 1.6
_DEFAULT_HAND_DIR = Path("models/rtmpose_hand")
_DEFAULT_FACE_MODEL = Path(
    "models/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
)


def _centered_median(values: np.ndarray, radius: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(len(values)):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        with np.errstate(all="ignore"):
            out[i] = np.nanmedian(values[lo:hi], axis=0)
    return out


def _rolling_max(values: np.ndarray, radius: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(len(values)):
        lo, hi = max(0, i - radius), min(len(values), i + radius + 1)
        with np.errstate(all="ignore"):
            out[i] = np.nanmax(values[lo:hi], axis=0)
    return out


def _make_crop(center: np.ndarray, side: float) -> tuple[float, float, float, float]:
    x1, y1 = np.asarray(center, dtype=np.float64) - float(side) / 2.0
    return float(x1), float(y1), float(x1 + side), float(y1 + side)


def _normalise_crop(crop: Any) -> tuple[float, float, float, float]:
    """Accept XYXY crops and the experiment's ``(center, side)`` form.

    ``hand_crop_boxes`` returns batched XYXY arrays, but the frozen experiment
    helpers historically exposed a single crop as ``(center, side)``.  The
    refinement boundary is intentionally tolerant of both forms so a caller
    cannot accidentally unpack a two-item crop as four coordinates.
    """
    try:
        values = list(crop)
    except TypeError as exc:
        raise ValueError(f"crop must be XYXY or (center, side), got {crop!r}") from exc
    if len(values) == 4:
        try:
            xyxy = np.asarray(values, dtype=np.float64).reshape(4)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid XYXY crop: {crop!r}") from exc
        if not np.isfinite(xyxy).all():
            raise ValueError(f"crop must contain finite coordinates, got {crop!r}")
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            raise ValueError(f"crop must have positive width and height, got {crop!r}")
        return tuple(float(value) for value in xyxy)
    if len(values) == 2:
        try:
            center = np.asarray(values[0], dtype=np.float64).reshape(2)
            side = float(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid (center, side) crop: {crop!r}") from exc
        if not np.isfinite(center).all() or not np.isfinite(side) or side <= 0.0:
            raise ValueError(f"crop must contain finite positive geometry, got {crop!r}")
        return _make_crop(center, side)
    raise ValueError(f"crop must be XYXY or (center, side), got shape ({len(values)},)")


def _normalise_sequence(kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(kp, dtype=np.float64)
    scores = np.asarray(sc, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError(f"kp must have shape (frames, 133, 2), got {keypoints.shape}")
    if scores.shape != (len(keypoints), 133):
        raise ValueError(f"sc must have shape ({len(keypoints)}, 133), got {scores.shape}")
    return keypoints, scores


def hand_crop_boxes(
    kp: np.ndarray,
    sc: np.ndarray,
    hand: str | None = None,
    *,
    factor: float = HAND_CROP_FACTOR,
    visibility: float = VISIBILITY,
) -> np.ndarray:
    """Return H4 square hand crops, exactly using the offline H4 geometry.

    ``hand`` may be ``left`` or ``right`` and returns ``(frames, 4)``.  When
    omitted, both hands are returned as ``(frames, 2, 4)`` in left/right order.
    The input is the complete 133-point body output; existing whole-body hand
    points are used for crop geometry, with elbow/wrist fallback as in H4.
    """
    keypoints, scores = _normalise_sequence(kp, sc)
    if hand is not None and hand not in HANDS:
        raise ValueError(f"hand must be one of {HANDS!r}, got {hand!r}")
    if factor <= 0.0:
        raise ValueError("factor must be positive")
    selected_hands = (hand,) if hand is not None else HANDS
    result = []
    for side_name in selected_hands:
        t = len(keypoints)
        centers = np.full((t, 2), np.nan, dtype=np.float64)
        side_raw = np.full(t, np.nan, dtype=np.float64)
        hand_size = np.full(t, np.nan, dtype=np.float64)
        h0 = HAND_BASE[side_name]
        elbow_i, wrist_i = BODY_ELBOW[side_name], BODY_WRIST[side_name]
        for i in range(t):
            points = keypoints[i, h0:h0 + 21]
            point_scores = scores[i, h0:h0 + 21]
            good = (point_scores >= visibility) & np.isfinite(points).all(axis=1)
            wrist = keypoints[i, wrist_i]
            elbow = keypoints[i, elbow_i]
            forearm = (
                float(np.linalg.norm(wrist - elbow))
                if np.isfinite(wrist).all() and np.isfinite(elbow).all()
                else np.nan
            )
            if int(good.sum()) >= 8:
                selected = points[good]
                lo, hi = selected.min(axis=0), selected.max(axis=0)
                bbox_side = float(np.max(hi - lo))
                centers[i] = (lo + hi) / 2.0
                hand_size[i] = max(bbox_side, 1.0)
                side_raw[i] = max(
                    factor * bbox_side,
                    0.8 * forearm if np.isfinite(forearm) else 0.0,
                )
            elif np.isfinite(wrist).all() and np.isfinite(elbow).all():
                centers[i] = wrist + 0.3 * (wrist - elbow)
                hand_size[i] = max(0.5 * forearm, 1.0) if np.isfinite(forearm) else np.nan
                side_raw[i] = 1.2 * forearm if np.isfinite(forearm) else np.nan
            elif np.isfinite(points).all(axis=1).any():
                selected = points[np.isfinite(points).all(axis=1)]
                centers[i] = np.mean(selected, axis=0)
                hand_size[i] = max(float(np.ptp(selected, axis=0).max()), 1.0)
                side_raw[i] = factor * hand_size[i]
        centers = _centered_median(centers, 3)
        sides = _rolling_max(side_raw, 5)
        finite_sides = sides[np.isfinite(sides) & (sides > 1.0)]
        fallback_side = float(np.median(finite_sides)) if len(finite_sides) else 100.0
        sides[~np.isfinite(sides) | (sides <= 1.0)] = fallback_side
        finite_sizes = hand_size[np.isfinite(hand_size) & (hand_size > 1.0)]
        fallback_size = float(np.median(finite_sizes)) if len(finite_sizes) else fallback_side / factor
        hand_size[~np.isfinite(hand_size) | (hand_size <= 1.0)] = fallback_size
        fallback_center = np.nanmedian(centers, axis=0)
        if not np.isfinite(fallback_center).all():
            fallback_center = np.asarray([960.0, 540.0])
        centers[~np.isfinite(centers).all(axis=1)] = fallback_center
        result.append(np.asarray([_make_crop(c, s) for c, s in zip(centers, sides)], dtype=np.float32))
    return result[0] if hand is not None else np.stack(result, axis=1)


def head_crop_boxes(
    kp: np.ndarray,
    sc: np.ndarray,
    *,
    visibility: float = VISIBILITY,
) -> np.ndarray:
    """Return F2 square head crops with the experiment's offline smoothing."""
    body, scores = _normalise_sequence(kp, sc)
    n = len(body)
    centers = np.full((n, 2), np.nan, dtype=np.float64)
    sides = np.full(n, np.nan, dtype=np.float64)
    anchors = np.arange(5)
    face = np.arange(FACE0, FACE0 + FACE_COUNT)
    for i in range(n):
        point_list = []
        anchor_good = (scores[i, anchors] >= visibility) & np.isfinite(body[i, anchors]).all(axis=1)
        face_good = (scores[i, face] >= visibility) & np.isfinite(body[i, face]).all(axis=1)
        if np.any(anchor_good):
            point_list.append(body[i, anchors[anchor_good]])
        if np.any(face_good):
            point_list.append(body[i, face[face_good]])
        if point_list:
            centers[i] = np.concatenate(point_list, axis=0).mean(axis=0)
        if np.any(face_good):
            sides[i] = 1.8 * max(float(np.ptp(body[i, face[face_good]], axis=0).max()), 1.0)
        distances = []
        for a, b in ((1, 3), (2, 4)):
            if scores[i, a] >= visibility and scores[i, b] >= visibility and np.isfinite(body[i, [a, b]]).all():
                distances.append(float(np.linalg.norm(body[i, a] - body[i, b])))
        if distances:
            sides[i] = max(sides[i] if np.isfinite(sides[i]) else 0.0, 2.5 * float(np.mean(distances)))
    centers = _centered_median(centers, 3)
    sides = _rolling_max(sides, 5)
    good_side = np.isfinite(sides) & (sides > 2.0)
    fallback_side = float(np.median(sides[good_side])) if np.any(good_side) else 160.0
    sides[~good_side] = fallback_side
    good_center = np.isfinite(centers).all(axis=1)
    fallback_center = np.nanmedian(centers, axis=0)
    if not np.isfinite(fallback_center).all():
        fallback_center = np.asarray([960.0, 540.0])
    centers[~good_center] = fallback_center
    return np.asarray([_make_crop(c, s) for c, s in zip(centers, sides)], dtype=np.float32)


def h4_switch(
    raw_points: np.ndarray,
    raw_scores: np.ndarray,
    candidate: np.ndarray,
    candidate_scores: np.ndarray,
    crop: tuple[float, float, float, float],
    *,
    threshold: float = H4_SCORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge a hand candidate using the accepted H4 score/distance gate."""
    crop = _normalise_crop(crop)
    result = np.asarray(raw_points, dtype=np.float32).copy()
    scores = np.asarray(raw_scores, dtype=np.float32).copy()
    source = np.zeros(21, dtype=np.uint8)
    side = max(float(crop[2] - crop[0]), 1.0)
    for index in range(21):
        if (
            candidate_scores[index] < threshold
            or not np.isfinite(candidate[index]).all()
            or not np.isfinite(raw_points[index]).all()
        ):
            continue
        if float(np.linalg.norm(candidate[index] - raw_points[index])) <= 0.5 * side:
            result[index] = candidate[index]
            scores[index] = candidate_scores[index]
            source[index] = 1
    return result, scores, source


HAND_WRIST = 0
HAND_MIDDLE_TIP = 12
HAND_FINGERTIPS = (4, 8, 12, 16, 20)
FOREARM_JOINTS = {"left": (7, 9), "right": (8, 10)}


def hand_size_primary(
    points: np.ndarray,
    scores: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Return the per-frame hand size from confident RTMW hand points.

    ``points`` and ``scores`` cover one hand only: ``(frames, 21, 2)`` and
    ``(frames, 21)``.  The primary estimate is the wrist-to-middle-tip
    distance, which matches the hand-size convention used by the offline
    labels.  When the middle tip is unconfident, the median of the other
    confident wrist-to-fingertip distances is used; frames with fewer than
    two usable distances stay NaN so the caller can apply its fallback.
    """
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 2):
        raise ValueError(f"hand points must have shape (frames, 21, 2), got {points.shape}")
    if scores.shape != points.shape[:2]:
        raise ValueError(f"hand scores must have shape {points.shape[:2]}, got {scores.shape}")
    threshold = float(config["confidence_threshold"])
    wrist = int(config.get("wrist", HAND_WRIST))
    middle_tip = int(config.get("middle_tip", HAND_MIDDLE_TIP))
    fingertips = tuple(int(value) for value in config.get("fingertips", HAND_FINGERTIPS))
    min_size = float(config.get("min_size_px", 1e-3))
    confident = np.isfinite(points).all(axis=-1) & (scores >= threshold)
    wrist_tip = np.where(
        confident[:, wrist] & confident[:, middle_tip],
        np.linalg.norm(points[:, wrist] - points[:, middle_tip], axis=-1),
        np.nan,
    )
    distances = np.stack(
        [
            np.where(
                confident[:, wrist] & confident[:, tip],
                np.linalg.norm(points[:, wrist] - points[:, tip], axis=-1),
                np.nan,
            )
            for tip in fingertips
        ],
        axis=1,
    )
    with np.errstate(all="ignore"):
        tip_median = np.full(len(distances), np.nan, dtype=np.float64)
        rows = np.isfinite(distances).any(axis=1)
        if rows.any():
            tip_median[rows] = np.nanmedian(distances[rows], axis=1)
    tip_median = np.where(np.isfinite(distances).sum(axis=1) >= 2, tip_median, np.nan)
    size = np.where(np.isfinite(wrist_tip), wrist_tip, tip_median)
    return np.where(np.isfinite(size) & (size > min_size), size, np.nan)


def hand_forearm_lengths(
    body_points: np.ndarray,
    joints: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """Return ``(frames, 2)`` elbow-to-wrist lengths in left/right order."""
    body = np.asarray(body_points, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (133, 2):
        raise ValueError(f"body points must have shape (frames, 133, 2), got {body.shape}")
    mapping = joints or FOREARM_JOINTS
    out = np.full((len(body), len(HANDS)), np.nan, dtype=np.float64)
    for index, side in enumerate(HANDS):
        elbow, wrist = mapping[side]
        out[:, index] = np.linalg.norm(body[:, wrist] - body[:, elbow], axis=-1)
    return out


def estimate_hand_sizes(
    body_points: np.ndarray,
    body_scores: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``(frames, 2)`` hand sizes and flag forearm fallback frames.

    Hand points come only from the RTMW body pass (``keypoints_body_raw``);
    no refined H4 point and no ground truth is used.  The confident estimate
    is smoothed by a centred median of +/- ``smooth_radius_frames`` and
    missing frames fall back to ``forearm_length * forearm_scale_k``.  A final
    sequence-median fill keeps the array finite for degenerate hands.
    """
    body = np.asarray(body_points, dtype=np.float64)
    scores = np.asarray(body_scores, dtype=np.float64)
    if body.ndim != 3 or body.shape[1:] != (133, 2):
        raise ValueError(f"body points must have shape (frames, 133, 2), got {body.shape}")
    if scores.shape != body.shape[:2]:
        raise ValueError(f"body scores must have shape {body.shape[:2]}, got {scores.shape}")
    radius = int(config["smooth_radius_frames"])
    scale = float(config["forearm_scale_k"])
    min_size = float(config.get("min_size_px", 1e-3))
    forearms = hand_forearm_lengths(body, config.get("forearm_joints"))
    forearms = _centered_median(forearms, radius) if np.isfinite(forearms).any() else forearms
    sizes = np.full((len(body), len(HANDS)), np.nan, dtype=np.float64)
    fallback = np.zeros_like(sizes, dtype=bool)
    for index, side in enumerate(HANDS):
        base = HAND_BASE[side]
        primary = hand_size_primary(
            body[:, base:base + 21],
            scores[:, base:base + 21],
            config,
        )
        smoothed = _centered_median(primary, radius) if np.isfinite(primary).any() else primary
        with np.errstate(all="ignore"):
            from_forearm = forearms[:, index] * scale
        usable = np.isfinite(smoothed) & (smoothed > min_size)
        values = np.where(usable, smoothed, from_forearm)
        values = np.where(np.isfinite(values) & (values > min_size), values, np.nan)
        finite = values[np.isfinite(values)]
        fallback_value = float(np.median(finite)) if len(finite) else 1.0
        fallback[:, index] = ~usable & np.isfinite(from_forearm) & (from_forearm > min_size)
        sizes[:, index] = np.where(np.isfinite(values), values, fallback_value)
    return sizes, fallback


def hand_agreement_distances(
    final_points: np.ndarray,
    body_points: np.ndarray,
    hand_source: np.ndarray,
    hand_size_px: np.ndarray,
) -> np.ndarray:
    """Return ``(frames, 133)`` cross-model disagreement ``d`` for hands.

    ``d = ||final H4 point - RTMW body point|| / hand_size`` where
    ``hand_source == 1``.  Points that fell back to RTMW are assigned
    ``d = 0`` by definition (they must not inherit top hand-model trust);
    every non-hand index is NaN.
    """
    final = np.asarray(final_points, dtype=np.float64)
    body = np.asarray(body_points, dtype=np.float64)
    source = np.asarray(hand_source)
    sizes = np.asarray(hand_size_px, dtype=np.float64)
    if final.ndim != 3 or final.shape[1:] != (133, 2):
        raise ValueError(f"final points must have shape (frames, 133, 2), got {final.shape}")
    if body.shape != final.shape:
        raise ValueError(f"body points must have shape {final.shape}, got {body.shape}")
    if source.shape != final.shape[:2]:
        raise ValueError(f"hand_source must have shape {final.shape[:2]}, got {source.shape}")
    if sizes.shape != (len(final), len(HANDS)):
        raise ValueError(f"hand_size_px must have shape ({len(final)}, {len(HANDS)}), got {sizes.shape}")
    distances = np.full(final.shape[:2], np.nan, dtype=np.float64)
    for index, side in enumerate(HANDS):
        base = HAND_BASE[side]
        selection = slice(base, base + 21)
        gap = np.linalg.norm(final[:, selection] - body[:, selection], axis=-1)
        size = sizes[:, index:index + 1]
        valid = np.isfinite(size) & (size > 0.0)
        ratio = np.where(valid, gap / np.where(valid, size, 1.0), np.nan)
        distances[:, selection] = np.where(source[:, selection] == 1, ratio, 0.0)
    return distances


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hand_model_path(path: str | Path | None = None) -> Path:
    if path is not None:
        model = Path(path)
    else:
        candidates = sorted(_DEFAULT_HAND_DIR.rglob("end2end.onnx"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one RTMPose hand ONNX, got {candidates}")
        model = candidates[0]
    if not model.exists():
        raise FileNotFoundError(f"hand model not found: {model}")
    return model


def face_model_path(path: str | Path | None = None) -> Path:
    model = Path(path) if path is not None else _DEFAULT_FACE_MODEL
    if not model.exists():
        raise FileNotFoundError(f"face model not found: {model}")
    return model


def model_hashes(part_refine: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if part_refine in {"hands", "hands_face"}:
        result["hand_model_sha256"] = _sha256(hand_model_path())
    if part_refine in {"face", "hands_face"}:
        result["face_model_sha256"] = _sha256(face_model_path())
    return result


class HandRefiner:
    """RTMPose-m hand5 inference plus the H4 safety switch."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        *,
        threshold: float = H4_SCORE,
        inference_engine: str = "ort",
        dnn_target: str = "cpu",
    ) -> None:
        self.model_path = hand_model_path(model_path)
        self.providers_requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.inference_engine = inference_engine
        self.dnn_target = dnn_target
        self.session = InferenceSession(
            self.model_path,
            engine=inference_engine,
            providers=self.providers_requested,
            target=dnn_target,
            input_shape=(1, 3, 256, 256),
        )
        self.active_providers = self.session.get_providers()
        input_meta = self.session.get_inputs()[0]
        self.input_name = str(input_meta.name)
        self.height, self.width = int(input_meta.shape[-2]), int(input_meta.shape[-1])
        outputs = self.session.get_outputs()
        names = [str(item.name) for item in outputs]
        self.x_index, self.y_index = names.index("simcc_x"), names.index("simcc_y")
        try:
            self.split_ratio = float(outputs[self.x_index].shape[-1]) / self.width
        except (TypeError, ValueError, IndexError):
            self.split_ratio = 2.0
        self.threshold = float(threshold)
        self.mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
        self.sha256 = _sha256(self.model_path)
        self.last_raw_scores = np.full(21, np.nan, dtype=np.float32)

    def _predict(self, frame: np.ndarray, crop: tuple[float, float, float, float]):
        x1, y1, x2, y2 = _normalise_crop(crop)
        side = max(float(x2 - x1), 1.0)
        matrix = np.asarray([[self.width / side, 0.0, -x1 * self.width / side],
                             [0.0, self.height / side, -y1 * self.height / side]], dtype=np.float32)
        patch = cv2.warpAffine(frame, matrix, (self.width, self.height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - self.mean) / self.std).transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {self.input_name: tensor})
        points, scores = _decode_simcc_subpixel(outputs[self.x_index], outputs[self.y_index], self.split_ratio)
        points[:, 0] = points[:, 0] * side / self.width + x1
        points[:, 1] = points[:, 1] * side / self.height + y1
        return points.astype(np.float32), scores.astype(np.float32)

    def refine(self, frame: np.ndarray, body_points: np.ndarray, body_scores: np.ndarray,
               hand: str, crop: tuple[float, float, float, float]):
        crop = _normalise_crop(crop)
        h0 = HAND_BASE[hand]
        raw_points = np.asarray(body_points[h0:h0 + 21], dtype=np.float32)
        raw_scores = np.asarray(body_scores[h0:h0 + 21], dtype=np.float32)
        if not np.isfinite(raw_points).all():
            self.last_raw_scores = np.full(21, np.nan, dtype=np.float32)
            return raw_points.copy(), raw_scores.copy(), np.zeros(21, dtype=np.uint8)
        candidate, candidate_scores = self._predict(frame, crop)
        self.last_raw_scores = candidate_scores.copy()
        merged, merged_scores, source = h4_switch(raw_points, raw_scores, candidate, candidate_scores, crop,
                                                   threshold=self.threshold)
        return merged, merged_scores, source


class FaceRefiner:
    """RTMW-x inference on the F2 padded 3:4 head crop."""

    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None,
                 *, estimator: Any | None = None, inference_engine: str = "ort",
                 dnn_target: str = "cpu") -> None:
        from aimocap.pose.infer import PoseEstimator

        self.model_path = face_model_path(model_path)
        self.estimator = estimator or PoseEstimator(
            pose_model=self.model_path, detector=None,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
            simcc_decode="subpixel", model_family="rtmw",
            inference_engine=inference_engine, dnn_target=dnn_target,
        )
        self.active_providers = list(getattr(self.estimator, "active_providers", []))
        self.sha256 = _sha256(self.model_path)

    def refine(self, frame: np.ndarray, crop: tuple[float, float, float, float]):
        x1, y1, x2, y2 = _normalise_crop(crop)
        side = max(float(x2 - x1), 1.0)
        content = min(self.estimator.input_w, self.estimator.input_h)
        offset_x = (self.estimator.input_w - content) / 2.0
        scale = content / side
        matrix = np.asarray([[scale, 0.0, offset_x - x1 * scale],
                             [0.0, scale, -y1 * scale]], dtype=np.float32)
        patch = cv2.warpAffine(frame, matrix, (self.estimator.input_w, self.estimator.input_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - self.estimator._mean) / self.estimator._std).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.estimator._run_pose(tensor)
        points, raw_scores = _decode_simcc_subpixel(simcc_x, simcc_y, self.estimator.split_ratio)
        points[:, 0] = (points[:, 0] - offset_x) / scale + x1
        points[:, 1] = points[:, 1] / scale + y1
        return (points[FACE0:FACE0 + FACE_COUNT].astype(np.float32),
                np.asarray(raw_scores[FACE0:FACE0 + FACE_COUNT], dtype=np.float32))


__all__ = [
    "FACE0", "FACE_COUNT", "HAND_BASE", "HANDS", "HandRefiner", "FaceRefiner",
    "hand_crop_boxes", "head_crop_boxes", "h4_switch", "hand_model_path",
    "face_model_path", "model_hashes", "estimate_hand_sizes",
    "hand_agreement_distances", "hand_forearm_lengths", "hand_size_primary",
]
