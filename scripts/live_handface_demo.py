"""Live RTMW + RTMPose hand + MediaPipe face/blendshape demo.

This is an experiment/demo script. It intentionally does not modify the
production package.  The live path is causal: hand crop centers use EMA and
crop sizes use a five-frame rolling maximum.  The H4 hand safety switch starts
with calibrated RTMW whole-body hands and accepts RTMPose-m points only when
score >= 0.10 and the point is no farther than half the crop side from RTMW.

Keys in the display window:
    q       quit
    m       toggle MediaPipe Face Landmarker/blendshapes
    h       show raw RTMW hands beside H4 hands
    r       start/stop a recording in outputs/visual_tests/live_<timestamp>.mp4

Headless mode never creates a window.  It records automatically to a new MP4
under outputs/visual_tests and prints measured FPS and active ONNX providers.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RTMW_PATH = ROOT / "models" / "rtmw" / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
DETECTOR_PATH = ROOT / "models" / "yolox_nano.onnx"
FACE_TASK = ROOT / "models" / "mediapipe" / "face_landmarker.task"
HAND_DIR = ROOT / "models" / "rtmpose_hand"

# COCO-WholeBody layout used by RTMW.
FACE0 = 23
LEFT_HAND0 = 91
RIGHT_HAND0 = 112
BODY_ELBOW = {"left": 7, "right": 8}
BODY_WRIST = {"left": 9, "right": 10}
HAND0 = {"left": LEFT_HAND0, "right": RIGHT_HAND0}
HANDS = ("left", "right")
VISIBILITY = 0.5
H4_SCORE = 0.10
HAND_CROP_FACTOR = 1.6

BODY_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)
HAND_FINGER_EDGES = {
    "thumb": ((0, 1), (1, 2), (2, 3), (3, 4)),
    "index": ((0, 5), (5, 6), (6, 7), (7, 8)),
    "middle": ((0, 9), (9, 10), (10, 11), (11, 12)),
    "ring": ((0, 13), (13, 14), (14, 15), (15, 16)),
    "pinky": ((0, 17), (17, 18), (18, 19), (19, 20)),
}
FINGER_COLORS = {
    "thumb": (0, 180, 255),
    "index": (0, 255, 100),
    "middle": (255, 190, 0),
    "ring": (255, 80, 220),
    "pinky": (100, 120, 255),
}
KEY_BLENDSHAPES = (
    "jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft",
    "mouthSmileRight", "browInnerUp", "browDownLeft",
)


def make_crop(center: np.ndarray, side: float) -> tuple[float, float, float, float]:
    x1, y1 = np.asarray(center, dtype=float) - float(side) / 2.0
    return float(x1), float(y1), float(x1 + side), float(y1 + side)


def finite_xy(value: np.ndarray) -> bool:
    return np.asarray(value).shape == (2,) and bool(np.isfinite(value).all())


def active_providers(obj: Any) -> list[str]:
    """Read an ONNX session provider list from an object when available."""
    for candidate in (obj, getattr(obj, "session", None), getattr(obj, "pose_session", None)):
        getter = getattr(candidate, "get_providers", None)
        if getter is not None:
            return [str(item) for item in getter()]
    return []


def unique_record_path() -> Path:
    directory = ROOT / "outputs" / "visual_tests"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = directory / f"live_{stamp}.mp4"
    serial = 1
    while candidate.exists():
        candidate = directory / f"live_{stamp}_{serial}.mp4"
        serial += 1
    return candidate


def hand_model_path() -> Path:
    candidates = sorted(HAND_DIR.rglob("end2end.onnx"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one RTMPose hand end2end.onnx, got {candidates}")
    return candidates[0]


@dataclass
class HandCropState:
    """Causal crop state for one hand."""

    center: np.ndarray | None = None
    sizes: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, body: np.ndarray, scores: np.ndarray, hand: str) -> tuple[float, float, float, float] | None:
        first = HAND0[hand]
        points = np.asarray(body[first:first + 21], dtype=np.float32)
        point_scores = np.asarray(scores[first:first + 21], dtype=np.float32)
        valid = (point_scores >= VISIBILITY) & np.isfinite(points).all(axis=1)
        wrist_i, elbow_i = BODY_WRIST[hand], BODY_ELBOW[hand]
        wrist, elbow = body[wrist_i], body[elbow_i]
        current_center: np.ndarray | None = None
        current_side: float | None = None
        if int(valid.sum()) >= 8:
            selected = points[valid]
            lo, hi = selected.min(axis=0), selected.max(axis=0)
            span = max(float(np.max(hi - lo)), 1.0)
            forearm = float(np.linalg.norm(wrist - elbow)) if finite_xy(wrist) and finite_xy(elbow) else 0.0
            current_center = (lo + hi) / 2.0
            current_side = max(HAND_CROP_FACTOR * span, 0.8 * forearm, 16.0)
        elif (scores[wrist_i] >= VISIBILITY and scores[elbow_i] >= VISIBILITY
              and finite_xy(wrist) and finite_xy(elbow)):
            forearm = max(float(np.linalg.norm(wrist - elbow)), 1.0)
            current_center = wrist + 0.3 * (wrist - elbow)
            current_side = max(1.2 * forearm, 16.0)
        else:
            # Do not invent a new crop when neither the hand nor the forearm is
            # observable.  Keeping the previous state avoids crop teleporting,
            # but the caller skips inference for this frame as required.
            return None

        if self.center is None:
            self.center = current_center.astype(np.float32)
        else:
            self.center = (0.5 * self.center + 0.5 * current_center).astype(np.float32)
        self.sizes.append(float(current_side))
        side = max(self.sizes)
        return make_crop(self.center, side)


@dataclass
class HeadCropState:
    center: np.ndarray | None = None
    sizes: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, body: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, int, int]) -> tuple[float, float, float, float]:
        face = body[FACE0:FACE0 + 68]
        face_scores = scores[FACE0:FACE0 + 68]
        face_good = (face_scores >= VISIBILITY) & np.isfinite(face).all(axis=1)
        anchors_good = (scores[:5] >= VISIBILITY) & np.isfinite(body[:5]).all(axis=1)
        selected = []
        if face_good.any():
            selected.append(face[face_good].astype(np.float32))
        if anchors_good.any():
            selected.append(body[:5][anchors_good].astype(np.float32))
        if selected:
            current_center = np.concatenate(selected, axis=0).mean(axis=0)
        elif self.center is not None:
            current_center = self.center
        else:
            h, w = frame_shape[:2]
            current_center = np.asarray([w / 2.0, h * 0.22], dtype=np.float32)

        spans: list[float] = []
        if face_good.any():
            spans.append(1.8 * max(float(np.ptp(face[face_good], axis=0).max()), 1.0))
        for a, b in ((1, 3), (2, 4)):
            if scores[a] >= VISIBILITY and scores[b] >= VISIBILITY and np.isfinite(body[[a, b]]).all():
                spans.append(2.5 * float(np.linalg.norm(body[a] - body[b])))
        if not spans:
            h, w = frame_shape[:2]
            spans.append(0.24 * min(h, w))
        current_side = max(spans + ([max(self.sizes)] if self.sizes else []))
        self.center = current_center if self.center is None else (0.5 * self.center + 0.5 * current_center)
        self.center = np.asarray(self.center, dtype=np.float32)
        self.sizes.append(float(current_side))
        return make_crop(self.center, max(self.sizes))


def h4_switch(raw_points: np.ndarray, raw_scores: np.ndarray,
              candidate: np.ndarray, candidate_scores: np.ndarray,
              crop: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Apply the accepted H4 per-point score and distance safety gate."""
    result = np.array(raw_points, dtype=np.float32, copy=True)
    scores = np.array(raw_scores, dtype=np.float32, copy=True)
    side = max(float(crop[2] - crop[0]), 1.0)
    for index in range(21):
        if (candidate_scores[index] < H4_SCORE or not finite_xy(candidate[index])
                or not finite_xy(raw_points[index])):
            continue
        if float(np.linalg.norm(candidate[index] - raw_points[index])) <= 0.5 * side:
            result[index] = candidate[index]
            scores[index] = candidate_scores[index]
    return result, scores


def safe_crop(frame: np.ndarray, crop: tuple[float, float, float, float], size: int) -> np.ndarray:
    x1, y1, x2, y2 = crop
    side = max(float(x2 - x1), 1.0)
    matrix = np.asarray([[size / side, 0.0, -x1 * size / side],
                         [0.0, size / side, -y1 * size / side]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def map_to_crop(points: np.ndarray, crop: tuple[float, float, float, float], size: int) -> np.ndarray:
    x1, y1, x2, _ = crop
    scale = size / max(float(x2 - x1), 1.0)
    mapped = (np.asarray(points, dtype=np.float32) - np.asarray([x1, y1], dtype=np.float32)) * scale
    return mapped


def draw_body(image: np.ndarray, points: np.ndarray, scores: np.ndarray, offset=(0, 0), scale=1.0) -> None:
    ox, oy = offset
    def pt(index: int) -> tuple[int, int] | None:
        if index >= len(points) or scores[index] < VISIBILITY or not finite_xy(points[index]):
            return None
        return (int(round(ox + points[index, 0] * scale)), int(round(oy + points[index, 1] * scale)))
    for a, b in BODY_EDGES:
        p1, p2 = pt(a), pt(b)
        if p1 and p2:
            cv2.line(image, p1, p2, (80, 220, 255), 2, cv2.LINE_AA)
    for index in range(min(17, len(points))):
        point = pt(index)
        if point:
            cv2.circle(image, point, 4, (255, 240, 180), -1, cv2.LINE_AA)


def draw_hand(image: np.ndarray, points: np.ndarray, scores: np.ndarray,
              offset=(0, 0), scale=1.0, muted=False) -> None:
    ox, oy = offset
    def pt(index: int) -> tuple[int, int] | None:
        if index >= len(points) or scores[index] <= 0.0 or not finite_xy(points[index]):
            return None
        return (int(round(ox + points[index, 0] * scale)), int(round(oy + points[index, 1] * scale)))
    for finger, edges in HAND_FINGER_EDGES.items():
        color = (100, 100, 100) if muted else FINGER_COLORS[finger]
        for a, b in edges:
            p1, p2 = pt(a), pt(b)
            if p1 and p2:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    for index in range(min(21, len(points))):
        point = pt(index)
        if point:
            cv2.circle(image, point, 3, (150, 150, 150) if muted else (245, 245, 245), -1, cv2.LINE_AA)


def draw_face(image: np.ndarray, points: np.ndarray, scores: np.ndarray,
              offset=(0, 0), scale=1.0, color=(0, 180, 255), radius=2) -> None:
    ox, oy = offset
    for index, point in enumerate(np.asarray(points)):
        if index < len(scores) and scores[index] > 0.0 and finite_xy(point):
            pos = (int(round(ox + point[0] * scale)), int(round(oy + point[1] * scale)))
            cv2.circle(image, pos, radius, color, -1, cv2.LINE_AA)


def draw_crop_inset(panel: np.ndarray, frame: np.ndarray, crop: tuple[float, float, float, float] | None,
                    raw: np.ndarray | None, raw_scores: np.ndarray | None,
                    chosen: np.ndarray | None, chosen_scores: np.ndarray | None,
                    origin: tuple[int, int], size: int, title: str, compare: bool) -> None:
    x, y = origin
    if crop is None:
        cv2.rectangle(panel, (x, y), (x + size, y + size), (45, 45, 45), 1)
        cv2.putText(panel, f"{title}: no crop", (x, y + size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        return
    patch = safe_crop(frame, crop, size)
    if compare and raw is not None and raw_scores is not None:
        draw_hand(patch, map_to_crop(raw, crop, size), raw_scores, muted=True)
    if chosen is not None and chosen_scores is not None:
        draw_hand(patch, map_to_crop(chosen, crop, size), chosen_scores)
    panel[y:y + size, x:x + size] = patch
    cv2.rectangle(panel, (x, y), (x + size, y + size), (90, 90, 90), 1)
    cv2.putText(panel, title, (x + 4, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)


def render(frame: np.ndarray, pose: Any, head_crop: tuple[float, float, float, float],
           hand_crops: dict[str, tuple[float, float, float, float] | None],
           raw_hands: dict[str, tuple[np.ndarray, np.ndarray]],
           chosen_hands: dict[str, tuple[np.ndarray, np.ndarray]],
           face_points: np.ndarray, face_scores: np.ndarray,
           mp_points: np.ndarray, mp_scores: np.ndarray, blend_values: dict[str, float],
           timings: dict[str, float], fps: float, mp_enabled: bool, compare_hands: bool) -> np.ndarray:
    height, width = 720, 1280
    main_width, panel_width = 960, 320
    annotated = frame.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in pose.bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 230, 100), 2)
    draw_body(annotated, pose.keypoints, pose.scores)
    for hand in HANDS:
        raw, raw_scores = raw_hands[hand]
        chosen, chosen_scores = chosen_hands[hand]
        if compare_hands:
            draw_hand(annotated, raw, raw_scores, muted=True)
        draw_hand(annotated, chosen, chosen_scores)
    draw_face(annotated, face_points, face_scores, color=(0, 180, 255))
    if mp_enabled:
        draw_face(annotated, mp_points, mp_scores, color=(255, 100, 80), radius=2)
    scale = min(main_width / max(annotated.shape[1], 1), height / max(annotated.shape[0], 1))
    shown_w, shown_h = int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)
    main = cv2.resize(annotated, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    main_x, main_y = (main_width - shown_w) // 2, (height - shown_h) // 2
    canvas[main_y:main_y + shown_h, main_x:main_x + shown_w] = main
    panel = canvas[:, main_width:]
    cv2.putText(panel, "LIVE HAND + FACE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, f"FPS {fps:5.1f}   MP {'ON' if mp_enabled else 'OFF'}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 230, 160), 1, cv2.LINE_AA)
    cv2.putText(panel, "H4 + raw" if compare_hands else "H4 hands", (190, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 255), 1, cv2.LINE_AA)
    inset = 130
    draw_crop_inset(panel, frame, hand_crops["left"], raw_hands["left"][0], raw_hands["left"][1], chosen_hands["left"][0], chosen_hands["left"][1], (12, 66), inset, "LEFT", compare_hands)
    draw_crop_inset(panel, frame, hand_crops["right"], raw_hands["right"][0], raw_hands["right"][1], chosen_hands["right"][0], chosen_hands["right"][1], (170, 66), inset, "RIGHT", compare_hands)
    face_y, face_size = 214, 150
    face_patch = safe_crop(frame, head_crop, face_size)
    draw_face(face_patch, map_to_crop(face_points, head_crop, face_size), face_scores, color=(0, 180, 255), radius=2)
    if mp_enabled:
        draw_face(face_patch, map_to_crop(mp_points, head_crop, face_size), mp_scores, color=(255, 100, 80), radius=1)
    panel[face_y:face_y + face_size, 12:12 + face_size] = face_patch
    cv2.rectangle(panel, (12, face_y), (12 + face_size, face_y + face_size), (90, 90, 90), 1)
    cv2.putText(panel, "HEAD / F2 + MP", (16, face_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "Blendshapes", (175, face_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    bar_x, bar_y, bar_w = 175, face_y + 34, 132
    for row, name in enumerate(KEY_BLENDSHAPES):
        value = float(blend_values.get(name, np.nan))
        value = np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0)
        yy = bar_y + row * 22
        cv2.putText(panel, name, (bar_x, yy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (205, 205, 205), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_x, yy + 14), (bar_x + bar_w, yy + 20), (55, 55, 55), -1)
        cv2.rectangle(panel, (bar_x, yy + 14), (bar_x + int(bar_w * value), yy + 20), (100, 190, 255), -1)
    cv2.putText(panel, "Stage ms", (12, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    for row, name in enumerate(("body", "hands", "face", "mediapipe")):
        cv2.putText(panel, f"{name:9s} {timings.get(name, 0.0):6.1f}", (12, 412 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 210, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "q quit  m MediaPipe  h compare  r record", (12, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, float]:
    if args.video:
        source = ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
        if not source.exists():
            raise FileNotFoundError(f"video not found: {source}")
        capture = cv2.VideoCapture(str(source))
    else:
        capture = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(args.index)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {'video ' + str(args.video) if args.video else 'webcam index ' + str(args.index)}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    return capture, fps


def load_models() -> tuple[Any, Any, Any, Any, dict[str, list[str]]]:
    import onnxruntime as ort
    from aimocap.pose.infer import PoseEstimator
    from scripts.phase1b_face_ab2 import RTMWCropModel, make_landmarker
    from scripts.phase1b_hand_ab3 import SimCCHandModel

    available = [str(item) for item in ort.get_available_providers()]
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"ONNX Runtime {ort.__version__}; available providers: {available}", flush=True)
    print(f"requested providers: {providers}", flush=True)
    body = PoseEstimator(pose_model=RTMW_PATH, detector=DETECTOR_PATH,
                         providers=providers, simcc_decode="subpixel",
                         model_family="rtmw")
    head = RTMWCropModel(RTMW_PATH, providers)
    hand = SimCCHandModel(hand_model_path(), providers)
    mp, landmarker = make_landmarker()
    active = {
        "body_pose": active_providers(body),
        "body_detector": active_providers(body.detector),
        "head_rtmw": active_providers(head.model),
        "rtmpose_hand": active_providers(hand),
    }
    print(f"active providers: {active}", flush=True)
    for name, values in active.items():
        if values and "CUDAExecutionProvider" not in values:
            raise RuntimeError(f"{name} did not activate CUDAExecutionProvider: {values}")
    return body, head, hand, (mp, landmarker), active


def process(args: argparse.Namespace) -> int:
    capture, source_fps = open_capture(args)
    body_model = head_model = hand_model = None
    mp_bundle = None
    writer = None
    output_path: Path | None = None
    try:
        body_model, head_model, hand_model, mp_bundle, providers = load_models()
        mp, landmarker = mp_bundle
        hand_states = {hand: HandCropState() for hand in HANDS}
        head_state = HeadCropState()
        mp_enabled = True
        compare_hands = False
        total_frames = 0
        start_time = time.perf_counter()
        last_fps = 0.0
        blend_values: dict[str, float] = {}
        max_frames = args.max_frames if args.max_frames is not None else 0
        if args.headless:
            output_path = unique_record_path()
            print(f"headless recording: {output_path.relative_to(ROOT)}", flush=True)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames and total_frames >= max_frames:
                break
            frame_start = time.perf_counter()
            height, width = frame.shape[:2]

            stage_start = time.perf_counter()
            poses = body_model.estimate(frame, pick="largest")
            if poses:
                pose = poses[0]
            else:
                # Required detector-loss behavior: process the full frame as
                # the person crop instead of dropping the frame.
                pose = body_model.estimate_with_crop(frame, [0.0, 0.0, float(width), float(height)])
            body_ms = (time.perf_counter() - stage_start) * 1000.0
            body, body_scores = pose.keypoints, pose.scores

            hand_crops: dict[str, tuple[float, float, float, float] | None] = {}
            raw_hands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            chosen_hands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            hand_ms = 0.0
            for hand_name in HANDS:
                first = HAND0[hand_name]
                raw = np.asarray(body[first:first + 21], dtype=np.float32)
                raw_score = np.asarray(body_scores[first:first + 21], dtype=np.float32)
                crop = hand_states[hand_name].update(body, body_scores, hand_name)
                hand_crops[hand_name] = crop
                candidate = np.full((21, 2), np.nan, dtype=np.float32)
                candidate_score = np.zeros(21, dtype=np.float32)
                if crop is not None:
                    stage_start = time.perf_counter()
                    candidate, candidate_score, _ = hand_model.run(frame, crop)
                    hand_ms += (time.perf_counter() - stage_start) * 1000.0
                chosen, chosen_score = h4_switch(raw, raw_score, candidate, candidate_score, crop or (0, 0, 1, 1))
                raw_hands[hand_name] = (raw, raw_score)
                chosen_hands[hand_name] = (chosen, chosen_score)

            stage_start = time.perf_counter()
            head_crop = head_state.update(body, body_scores, frame.shape)
            face_full, face_full_scores, _ = head_model.run(frame, head_crop)
            face_points = face_full[0, :68] if face_full.ndim == 3 else face_full[:68]
            face_scores = face_full_scores[0, :68] if face_full_scores.ndim == 2 else face_full_scores[:68]
            face_ms = (time.perf_counter() - stage_start) * 1000.0

            mp_points = np.full((68, 2), np.nan, dtype=np.float32)
            mp_scores = np.zeros(68, dtype=np.float32)
            mp_ms = 0.0
            if mp_enabled:
                stage_start = time.perf_counter()
                mp_points, mp_scores, blend, _, mp_ms, names = __import__(
                    "scripts.phase1b_face_ab2", fromlist=["mp_predict"]
                ).mp_predict(mp, landmarker, frame, head_crop)
                if names:
                    blend_values = {name: float(blend[index]) for index, name in enumerate(names) if index < len(blend)}
                mp_ms = (time.perf_counter() - stage_start) * 1000.0

            total_frames += 1
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            last_fps = total_frames / elapsed
            timings = {"body": body_ms, "hands": hand_ms, "face": face_ms, "mediapipe": mp_ms}
            shown = render(frame, pose, head_crop, hand_crops, raw_hands, chosen_hands,
                           face_points, face_scores, mp_points, mp_scores, blend_values,
                           timings, last_fps, mp_enabled, compare_hands)
            if args.headless:
                if writer is None:
                    output_path = output_path or unique_record_path()
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(source_fps, 1.0), (shown.shape[1], shown.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create recording: {output_path}")
                writer.write(shown)
            else:
                cv2.imshow("aimocap live hand + face", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    mp_enabled = not mp_enabled
                    print(f"MediaPipe {'enabled' if mp_enabled else 'disabled'}", flush=True)
                elif key == ord("h"):
                    compare_hands = not compare_hands
                    print(f"hand comparison {'enabled' if compare_hands else 'disabled'}", flush=True)
                elif key == ord("r"):
                    if writer is None:
                        output_path = unique_record_path()
                        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 max(source_fps, 1.0), (shown.shape[1], shown.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"could not create recording: {output_path}")
                        print(f"recording: {output_path.relative_to(ROOT)}", flush=True)
                    else:
                        writer.release()
                        writer = None
                        print(f"recording stopped: {output_path.relative_to(ROOT)}", flush=True)
                if writer is not None:
                    writer.write(shown)
            if total_frames % 10 == 0:
                print(f"frame={total_frames} fps={last_fps:.2f} body_ms={body_ms:.1f} hands_ms={hand_ms:.1f} face_ms={face_ms:.1f} mp_ms={mp_ms:.1f}", flush=True)

        if total_frames == 0:
            raise RuntimeError("capture ended before any frame was processed")
        print(f"SUMMARY frames={total_frames} fps={last_fps:.2f} source_fps={source_fps:.2f} providers={providers}", flush=True)
        if output_path is not None:
            print(f"OUTPUT {output_path}", flush=True)
        return 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if mp_bundle is not None:
            try:
                mp_bundle[1].close()
            except Exception:
                pass
        if not args.headless:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="video path instead of webcam")
    parser.add_argument("--index", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="disable the window; record automatically")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return process(args)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        return 0
    except RuntimeError as exc:
        message = str(exc)
        if not args.video and message.startswith("could not open webcam"):
            print(f"CAMERA_UNAVAILABLE index={args.index}: {message}", flush=True)
            return 0
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
