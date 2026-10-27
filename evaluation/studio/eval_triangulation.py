"""Phase-3 reusable 3D evaluation harness.

Evaluates a ``skeleton3d.npz`` produced by ``python -m aimocap.cli triangulate``
against dataset 3D ground truth in the same world frame as the calibration.

CLI:
    python -m evaluation.studio.eval_triangulation --clip clip1 --skel <skeleton3d.npz> --out <json>

Library:
    from evaluation.studio.eval_triangulation import evaluate
    result = evaluate(skeleton3d_npz, clip_cfg)

Frame policy:
* skeleton3d is the production internal frame ``internal_y_up_right_handed_z_back``
  in metres. The CLI is run with the default identity floor transform and
  metric scale preservation (scale_factor == 1.0), so no transform is inverted.
* Panoptic/AIST++ GT is in centimetres in the calibration world frame. The
  conversion to the internal frame is ``(x, -y, -z) / 100`` for both datasets.
* No per-frame Procrustes/alignment is used. ``align="rigid"`` is available for
  the explicit fallback of one global rigid transform (no scale) per clip and
  is recorded in the output when used.

Metrics are defined in the report emitted by this script; values are millimetres
unless stated otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS_PATH = ROOT / "evaluation" / "studio" / "clips.json"

BODY = list(range(0, 17))
FEET = list(range(17, 23))
FACE = list(range(23, 91))
LEFT_HAND = list(range(91, 112))
RIGHT_HAND = list(range(112, 133))
FINGERTIPS = [95, 99, 103, 107, 111, 116, 120, 124, 128, 132]

GROUPS: dict[str, list[int]] = {
    "body": BODY,
    "body_no_hips": [joint for joint in BODY if joint not in (11, 12)],
    "feet": FEET,
    "face": FACE,
    "left_hand": LEFT_HAND,
    "right_hand": RIGHT_HAND,
    "fingertips": FINGERTIPS,
}
ALL_JOINTS = list(range(133))

JOINT_NAMES = (
    [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
    ]
    + [f"face{i:02d}" for i in range(68)]
    + [f"left_hand{i:02d}" for i in range(21)]
    + [f"right_hand{i:02d}" for i in range(21)]
)
assert len(JOINT_NAMES) == 133

# Main limb bones: proximal -> distal, from COCO-WholeBody body joints.
BONES: dict[str, tuple[int, int]] = {
    "left_upper_arm": (5, 7),
    "right_upper_arm": (6, 8),
    "left_forearm": (7, 9),
    "right_forearm": (8, 10),
    "left_thigh": (11, 13),
    "right_thigh": (12, 14),
    "left_shin": (13, 15),
    "right_shin": (14, 16),
}

# Panoptic COCO-19 -> COCO-17 (same table as scripts/phase1_2d_experiment.py).
COCO17_TO_PAN19 = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
NOT_OBSERVED_REASONS = frozenset(
    {"not_observed", "low_confidence", "nonfinite_2d", "outside_image"}
)


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return value


def load_clip_config(clip_id: str, path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else CLIPS_PATH
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", {})
    if clip_id not in clips:
        raise KeyError(f"clip {clip_id!r} not found in {cfg_path}")
    cfg = dict(clips[clip_id])
    cfg["clip_id"] = clip_id
    return cfg


def _to_internal_m(points_cm: np.ndarray) -> np.ndarray:
    """Centimetres in a calibration world frame -> internal Y-up metres."""
    points = np.asarray(points_cm, dtype=np.float64) / 100.0
    points[..., 1] *= -1.0
    points[..., 2] *= -1.0
    return points


def _normalise_name(value: str) -> str:
    stem = Path(str(value)).stem
    return stem[3:] if stem.startswith("hd_") else stem


def _global_frames(
    data: Mapping[str, Any], clip_cfg: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Return the dataset global frame index for every output frame.

    ``triangulate`` artifacts drop the ``sync_manifest`` blob, so fall back to
    the manifest path recorded in the clip configuration. The offset is the
    reference camera's ``global_start_frame`` (Panoptic: window start,
    AIST++: +1 video/annotation frame offset).
    """
    source = np.asarray(data["source_frame_indices"], dtype=np.int64)
    if source.ndim != 2:
        source = source[:, None]
    manifest: dict[str, Any] = {}
    if "sync_manifest" in data:
        try:
            manifest = json.loads(str(data["sync_manifest"]))
        except (TypeError, json.JSONDecodeError):
            manifest = {}
    if not manifest and clip_cfg is not None and clip_cfg.get("sync_manifest"):
        fallback = _resolve_path(str(clip_cfg["sync_manifest"]))
        if fallback.exists():
            manifest = json.loads(fallback.read_text(encoding="utf-8"))
    cameras = manifest.get("cameras", {}) if isinstance(manifest, dict) else {}
    reference_id = None
    if "sync_reference_camera" in data:
        reference_id = str(_scalar(data["sync_reference_camera"]))
    sync_ids = None
    if "sync_camera_ids" in data:
        sync_ids = [str(x) for x in np.asarray(data["sync_camera_ids"]).reshape(-1)]
    column = 0
    if reference_id is not None and sync_ids and reference_id in sync_ids:
        column = sync_ids.index(reference_id)
    offset = 0
    if reference_id is not None and isinstance(cameras, dict):
        entry = cameras.get(reference_id)
        if isinstance(entry, dict) and "global_start_frame" in entry:
            offset = int(entry["global_start_frame"])
    elif isinstance(cameras, dict) and cameras:
        first = next(iter(cameras.values()))
        if isinstance(first, dict):
            offset = int(first.get("global_start_frame", 0))
    return source[:, column] + offset


def _load_panoptic_body(
    seq_dir: Path, gt_cfg: Mapping[str, Any], frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = len(frames)
    gt = np.full((count, 133, 3), np.nan, dtype=np.float64)
    valid = np.zeros((count, 133), dtype=bool)
    body_dir = seq_dir / str(gt_cfg.get("body_dir", "hdPose3d_stage1_coco19"))
    confidence_min = float(gt_cfg.get("joints19_confidence_min", 0.0))
    for index, global_frame in enumerate(frames):
        path = body_dir / f"body3DScene_{int(global_frame):08d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        bodies = record.get("bodies") or []
        if not bodies:
            continue
        joints = np.asarray(bodies[0].get("joints19", []), dtype=np.float64).reshape(-1, 4)
        if joints.shape[0] < 19:
            continue
        for coco17, pan19 in enumerate(COCO17_TO_PAN19):
            xyz = joints[pan19, :3]
            confidence = joints[pan19, 3]
            if np.isfinite(xyz).all() and np.isfinite(confidence) and confidence >= confidence_min:
                gt[index, coco17] = _to_internal_m(xyz)
                valid[index, coco17] = True
    return gt, valid


def _load_panoptic_hands(
    seq_dir: Path,
    gt_cfg: Mapping[str, Any],
    frames: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> None:
    hand_dir = seq_dir / str(gt_cfg.get("hand_dir", "hdHand3d"))
    for index, global_frame in enumerate(frames):
        path = hand_dir / f"handRecon3D_hd{int(global_frame):08d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        people = record.get("people") or []
        for field, base in (("left_hand", 91), ("right_hand", 112)):
            item = next(
                (person.get(field) for person in people if isinstance(person.get(field), dict)),
                None,
            )
            if not item:
                continue
            landmarks = np.asarray(item.get("landmarks", []), dtype=np.float64)
            if landmarks.size != 21 * 3:
                continue
            points = landmarks.reshape(21, 3)
            validity = np.asarray(item.get("validity", np.ones(21)), dtype=np.float64)
            if validity.size != 21:
                validity = np.ones(21, dtype=np.float64)
            ok = (validity > 0.0) & np.isfinite(points).all(axis=1)
            if not ok.any():
                continue
            converted = _to_internal_m(points)
            gt[index, base : base + 21][ok] = converted[ok]
            valid[index, base : base + 21][ok] = True


def _load_panoptic_face(
    seq_dir: Path,
    gt_cfg: Mapping[str, Any],
    frames: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> None:
    face_dir = seq_dir / str(gt_cfg.get("face_dir", "hdFace3d"))
    for index, global_frame in enumerate(frames):
        path = face_dir / f"faceRecon3D_hd{int(global_frame):08d}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        people = record.get("people") or []
        item = next(
            (person.get("face70") for person in people if isinstance(person.get("face70"), dict)),
            None,
        )
        if not item:
            continue
        landmarks = np.asarray(item.get("landmarks", []), dtype=np.float64)
        if landmarks.size != 70 * 3:
            continue
        points = landmarks.reshape(70, 3)[:68]
        validity = np.asarray(item.get("validity", []), dtype=np.float64).reshape(-1)
        if validity.size >= 68:
            validity = validity[:68]
        else:
            validity = np.ones(68, dtype=np.float64)
        ok = (validity > 0.0) & np.isfinite(points).all(axis=1)
        if not ok.any():
            continue
        converted = _to_internal_m(points)
        gt[index, 23:91][ok] = converted[ok]
        valid[index, 23:91][ok] = True


def _load_gt(
    clip_cfg: Mapping[str, Any], frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dataset = str(clip_cfg.get("dataset", "")).lower()
    count = len(frames)
    gt = np.full((count, 133, 3), np.nan, dtype=np.float64)
    valid = np.zeros((count, 133), dtype=bool)
    if dataset == "panoptic":
        seq_dir = ROOT / str(clip_cfg["sequence_dir"])
        gt_cfg = dict(clip_cfg.get("gt", {}))
        body_gt, body_valid = _load_panoptic_body(seq_dir, gt_cfg, frames)
        gt[:, :17] = body_gt[:, :17]
        valid[:, :17] = body_valid[:, :17]
        _load_panoptic_hands(seq_dir, gt_cfg, frames, gt, valid)
        _load_panoptic_face(seq_dir, gt_cfg, frames, gt, valid)
        return gt, valid
    if dataset == "aistpp":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.datasets.aistpp_subset import load_aistpp

        loaded = load_aistpp(str(clip_cfg["sequence"]), list(clip_cfg["cameras"]))
        all_gt = _to_internal_m(np.asarray(loaded["gt3d"], dtype=np.float64))
        for index, global_frame in enumerate(frames):
            frame = int(global_frame)
            if frame < 0 or frame >= len(all_gt):
                continue
            points = all_gt[frame]
            ok = np.isfinite(points).all(axis=1)
            if not ok.any():
                continue
            gt[index, :17][ok] = points[ok]
            valid[index, :17][ok] = True
        return gt, valid
    raise ValueError(f"unsupported dataset {dataset!r}")


def _rigid_alignment(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """One global rigid transform (rotation + translation, no scale)."""
    indices = np.asarray(BODY, dtype=int)
    mask = valid[:, indices] & np.isfinite(pred[:, indices]).all(axis=-1)
    if int(mask.sum()) < 3:
        raise ValueError("not enough valid body correspondences for a rigid alignment")
    source = pred[:, indices][mask]
    target = gt[:, indices][mask]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = target_mean - rotation @ source_mean
    residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, float(np.sqrt(np.mean(residuals ** 2)) * 1000.0)


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "mean": None, "p95": None, "p99": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _bone_summary(pred: np.ndarray) -> dict[str, Any]:
    bones: dict[str, dict[str, Any]] = {}
    for name, (left, right) in BONES.items():
        mask = np.isfinite(pred[:, left]).all(axis=1) & np.isfinite(pred[:, right]).all(axis=1)
        if int(mask.sum()) < 2:
            continue
        lengths = np.linalg.norm(pred[mask, left] - pred[mask, right], axis=1) * 1000.0
        bones[name] = {
            "mean_mm": float(np.mean(lengths)),
            "std_mm": float(np.std(lengths, ddof=0)),
            "frames": int(mask.sum()),
        }
    stds = [item["std_mm"] for item in bones.values()]
    return {
        "median_std_mm": float(np.median(stds)) if stds else None,
        "bones": bones,
    }


def _group_block(
    indices: Sequence[int],
    error: np.ndarray,
    covered: np.ndarray,
    gt_valid: np.ndarray,
    error_pelvis: np.ndarray,
    pelvis_ok: np.ndarray,
    accel_error: np.ndarray,
    accel_valid: np.ndarray,
    gt_speed: np.ndarray,
    pred_speed: np.ndarray,
    bone_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    idx = np.asarray(list(indices), dtype=int)
    gt_points = int(gt_valid[:, idx].sum())
    covered_points = int(covered[:, idx].sum())
    errors = error[:, idx][covered[:, idx]]
    pelvis_mask = covered[:, idx] & pelvis_ok[:, None]
    pelvis_errors = error_pelvis[:, idx][pelvis_mask]
    accel_values = accel_error[:, idx][accel_valid[:, idx]]

    fast_accel: list[np.ndarray] = []
    fast_ratios: list[np.ndarray] = []
    fast_pred_medians: list[np.ndarray] = []
    fast_gt_medians: list[np.ndarray] = []
    for joint in idx:
        usable = (
            accel_valid[:, joint]
            & np.isfinite(gt_speed[:, joint])
            & np.isfinite(pred_speed[:, joint])
        )
        if int(usable.sum()) < 5:
            continue
        speed_values = gt_speed[usable, joint]
        threshold = float(np.percentile(speed_values, 80))
        fast = usable & (gt_speed[:, joint] >= threshold)
        if not fast.any():
            continue
        fast_accel.append(accel_error[fast, joint])
        numerator = pred_speed[fast, joint]
        denominator = gt_speed[fast, joint]
        ratio_mask = denominator > 1e-9
        if ratio_mask.any():
            fast_ratios.append(numerator[ratio_mask] / denominator[ratio_mask])
            fast_pred_medians.append(numerator[ratio_mask])
            fast_gt_medians.append(denominator[ratio_mask])

    fast_jitter = _stats(np.concatenate(fast_accel)) if fast_accel else _stats(np.empty(0))
    if fast_ratios:
        ratios = np.concatenate(fast_ratios)
        pred_values = np.concatenate(fast_pred_medians)
        gt_values = np.concatenate(fast_gt_medians)
        median_of_ratios = float(np.median(ratios))
        ratio_of_medians = (
            float(np.median(pred_values) / np.median(gt_values))
            if np.median(gt_values) > 1e-9
            else None
        )
    else:
        median_of_ratios = None
        ratio_of_medians = None

    return {
        "gt_available": gt_points > 0,
        "gt_valid_points": gt_points,
        "covered_points": covered_points,
        "coverage": _rate(covered_points, gt_points),
        "mpjpe_mm": _stats(errors),
        "count_gt_50mm": int(np.count_nonzero(errors > 50.0)),
        "count_gt_25mm": int(np.count_nonzero(errors > 25.0)),
        "pelvis_aligned_median_mm": (
            float(np.median(pelvis_errors)) if pelvis_errors.size else None
        ),
        "jitter_mm_per_frame2": _stats(accel_values),
        "fast_frame_jitter_mm_per_frame2": fast_jitter,
        "fast_frame_speed_ratio_median": ratio_of_medians,
        "fast_frame_speed_ratio_median_of_ratios": median_of_ratios,
        "bone_length_std_mm": dict(bone_summary) if bone_summary is not None else None,
    }


def _switch_arrays(
    inlier_mask: np.ndarray, finite_pred: np.ndarray, indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(list(indices), dtype=int)
    sub = np.asarray(inlier_mask, dtype=bool)[:, idx, :]
    frames, joints, cameras = sub.shape
    masks = np.zeros((frames, joints), dtype=np.int64)
    for camera in range(cameras):
        masks |= sub[:, :, camera].astype(np.int64) << camera
    counts = sub.sum(axis=2)
    switch = np.zeros((frames, joints), dtype=bool)
    defined = np.zeros((frames, joints), dtype=bool)
    if frames > 1:
        switch[1:] = masks[1:] != masks[:-1]
        defined[1:] = (
            finite_pred[1:, idx]
            & finite_pred[:-1, idx]
            & (counts[1:] >= 2)
            & (counts[:-1] >= 2)
        )
    return switch, defined


def _jump_link(
    accel_error: np.ndarray,
    switch: np.ndarray,
    defined: np.ndarray,
    indices: Sequence[int],
    threshold_mm: float | None,
) -> dict[str, Any]:
    idx = np.asarray(list(indices), dtype=int)
    mask = defined & np.isfinite(accel_error[:, idx])
    switches = switch[mask]
    values = accel_error[:, idx][mask]
    if threshold_mm is None or values.size == 0:
        return {
            "big_jump_threshold_mm": threshold_mm,
            "switch_joint_frames": int(switches.sum()),
            "no_switch_joint_frames": int((~switches).sum()),
            "p_big_jump_given_switch": None,
            "p_big_jump_given_no_switch": None,
        }
    big = values > threshold_mm
    switch_count = int(switches.sum())
    no_switch_count = int((~switches).sum())
    return {
        "big_jump_threshold_mm": float(threshold_mm),
        "switch_joint_frames": switch_count,
        "no_switch_joint_frames": no_switch_count,
        "p_big_jump_given_switch": (
            float(big[switches].mean()) if switch_count else None
        ),
        "p_big_jump_given_no_switch": (
            float(big[~switches].mean()) if no_switch_count else None
        ),
    }


def _camera_exclusion(
    data: Mapping[str, Any],
    frame_count: int | None = None,
    camera_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    cameras = [str(x) for x in np.asarray(data["camera_names"]).reshape(-1)]
    if camera_names is not None and len(camera_names) == len(cameras):
        cameras = [str(name) for name in camera_names]
    else:
        cameras = [_normalise_name(name) for name in cameras]
    inlier = np.asarray(data["triangulation_inlier_mask"], dtype=bool)
    reason = np.asarray(data["triangulation_camera_rejection_reason"]).astype(str)
    if frame_count is not None:
        inlier = inlier[:frame_count]
        reason = reason[:frame_count]
    observed = np.ones_like(inlier, dtype=bool)
    for value in NOT_OBSERVED_REASONS:
        observed &= reason != value
    excluded = observed & ~inlier
    total = int(observed.sum())
    result: dict[str, Any] = {}
    for camera, name in enumerate(cameras):
        camera_observed = int(observed[:, :, camera].sum())
        camera_excluded = int(excluded[:, :, camera].sum())
        result[name] = {
            "observed_joint_frames": camera_observed,
            "excluded_joint_frames": camera_excluded,
            "excluded_fraction_of_observations": _rate(camera_excluded, camera_observed),
            "observed_fraction_of_all_joint_frames": _rate(
                camera_observed, int(observed.shape[0] * observed.shape[1])
            ),
        }
    result["_all"] = {
        "observed_joint_frames": total,
        "excluded_joint_frames": int(excluded.sum()),
        "excluded_fraction_of_observations": _rate(int(excluded.sum()), total),
    }
    return result


def _group_of(joint: int) -> str:
    if joint < 17:
        return "body"
    if joint < 23:
        return "feet"
    if joint < 91:
        return "face"
    if joint < 112:
        return "left_hand"
    return "right_hand"


def _worst_joints(
    error: np.ndarray,
    covered: np.ndarray,
    gt_valid: np.ndarray,
    minimum: int = 50,
    minimum_coverage: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint in range(133):
        mask = covered[:, joint]
        count = int(mask.sum())
        if count < minimum:
            continue
        gt_count = int(gt_valid[:, joint].sum())
        coverage = _rate(count, gt_count)
        if coverage is None or coverage < minimum_coverage:
            continue
        values = error[:, joint][mask]
        rows.append(
            {
                "joint": joint,
                "name": JOINT_NAMES[joint],
                "group": _group_of(joint),
                "fingertip": joint in FINGERTIPS,
                "covered_points": count,
                "coverage": coverage,
                "median_mm": float(np.median(values)),
                "p95_mm": float(np.percentile(values, 95)),
            }
        )
    rows.sort(key=lambda item: item["median_mm"], reverse=True)
    return rows[:5]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def evaluate(
    skeleton3d_npz: str | Path | Mapping[str, Any],
    clip_cfg: Mapping[str, Any],
    *,
    variant: str | None = None,
    align: str = "none",
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Evaluate one skeleton3d artifact against the clip's 3D ground truth."""
    if isinstance(skeleton3d_npz, Mapping):
        data = skeleton3d_npz
        skel_label = "<in-memory>"
    else:
        path = _resolve_path(skeleton3d_npz)
        data = np.load(path, allow_pickle=True)
        skel_label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    pred = np.asarray(data["skeleton3d"], dtype=np.float64)
    if pred.ndim != 3 or pred.shape[1:] != (133, 3):
        raise ValueError(f"skeleton3d must have shape (frames, 133, 3), got {pred.shape}")
    if max_frames is not None:
        pred = pred[: int(max_frames)]
    frames = _global_frames(data, clip_cfg)[: len(pred)]
    gt, gt_valid = _load_gt(clip_cfg, frames)
    if len(gt) != len(pred):
        raise ValueError("ground-truth frame count does not match prediction frame count")

    finite_pred = np.isfinite(pred).all(axis=-1)
    alignment_info: dict[str, Any] = {"mode": align}
    if align == "rigid":
        rotation, translation, rms = _rigid_alignment(pred, gt, gt_valid)
        pred = pred @ rotation.T + translation
        alignment_info.update(
            {
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
                "body_rms_mm": rms,
                "note": "one global rigid transform (no scale), fitted once per clip",
            }
        )
    elif align != "none":
        raise ValueError(f"unsupported alignment mode {align!r}")

    covered = gt_valid & np.isfinite(pred).all(axis=-1)
    error = np.linalg.norm(pred - gt, axis=-1) * 1000.0

    pelvis_pred = 0.5 * (pred[:, 11] + pred[:, 12])
    pelvis_gt = 0.5 * (gt[:, 11] + gt[:, 12])
    pelvis_ok = (
        np.isfinite(pelvis_pred).all(axis=1)
        & np.isfinite(pelvis_gt).all(axis=1)
        & gt_valid[:, 11]
        & gt_valid[:, 12]
    )
    shifted_pred = pred - np.where(pelvis_ok[:, None, None], pelvis_pred[:, None, :], 0.0)
    shifted_gt = gt - np.where(pelvis_ok[:, None, None], pelvis_gt[:, None, :], 0.0)
    error_pelvis = np.linalg.norm(shifted_pred - shifted_gt, axis=-1) * 1000.0
    error_pelvis[~pelvis_ok] = np.nan

    triple_valid = gt_valid[2:] & gt_valid[1:-1] & gt_valid[:-2] & finite_pred[2:] & finite_pred[1:-1] & finite_pred[:-2]
    accel_valid = np.zeros((len(pred), 133), dtype=bool)
    accel_valid[2:] = triple_valid
    pred_accel = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    gt_accel = gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
    accel_error = np.full((len(pred), 133), np.nan, dtype=np.float64)
    accel_error[2:] = np.linalg.norm(pred_accel - gt_accel, axis=-1) * 1000.0

    speed_valid = accel_valid
    gt_speed = np.full((len(pred), 133), np.nan, dtype=np.float64)
    pred_speed = np.full((len(pred), 133), np.nan, dtype=np.float64)
    gt_speed_values = np.linalg.norm(gt[2:] - gt[:-2], axis=-1) * 500.0
    pred_speed_values = np.linalg.norm(pred[2:] - pred[:-2], axis=-1) * 500.0
    gt_speed_values[~triple_valid] = np.nan
    pred_speed_values[~triple_valid] = np.nan
    gt_speed[2:] = gt_speed_values
    pred_speed[2:] = pred_speed_values

    bones = _bone_summary(pred)
    group_blocks: dict[str, Any] = {}
    for name, indices in GROUPS.items():
        group_blocks[name] = _group_block(
            indices,
            error,
            covered,
            gt_valid,
            error_pelvis,
            pelvis_ok,
            accel_error,
            accel_valid,
            gt_speed,
            pred_speed,
            bones if name == "body" else None,
        )
    pooled = _group_block(
        ALL_JOINTS,
        error,
        covered,
        gt_valid,
        error_pelvis,
        pelvis_ok,
        accel_error,
        accel_valid,
        gt_speed,
        pred_speed,
        bones,
    )

    switch_all, defined_all = _switch_arrays(
        np.asarray(data["triangulation_inlier_mask"], dtype=bool)[: len(pred)],
        finite_pred,
        ALL_JOINTS,
    )
    clip_threshold = (
        float(np.nanpercentile(accel_error, 95)) if np.isfinite(accel_error).any() else None
    )
    switch_analysis: dict[str, Any] = {
        "definition": (
            "switch = inlier camera set at t differs from t-1 where both frames have "
            ">=2 inliers and a finite 3D point; big jump = accel error > clip p95"
        ),
        "switch_rate": _rate(int(switch_all[defined_all].sum()), int(defined_all.sum())),
        "switch_joint_frames": int(switch_all[defined_all].sum()),
        "defined_joint_frames": int(defined_all.sum()),
        "big_jump_threshold_mm": clip_threshold,
        "jump_switch_link": _jump_link(
            accel_error, switch_all, defined_all, ALL_JOINTS, clip_threshold
        ),
        "groups": {},
    }
    for name, indices in GROUPS.items():
        switch, defined = _switch_arrays(
            np.asarray(data["triangulation_inlier_mask"], dtype=bool)[: len(pred)],
            finite_pred,
            indices,
        )
        switch_analysis["groups"][name] = {
            "switch_rate": _rate(int(switch[defined].sum()), int(defined.sum())),
            "switch_joint_frames": int(switch[defined].sum()),
            "defined_joint_frames": int(defined.sum()),
            "jump_switch_link": _jump_link(
                accel_error, switch, defined, indices, clip_threshold
            ),
        }

    warnings: list[str] = []
    if not bool(gt_valid[:, FEET].any()):
        warnings.append("feet 17:23 have no GT in this dataset (reported as gt_available=false)")
    dataset = str(clip_cfg.get("dataset", "")).lower()
    if dataset == "aistpp":
        warnings.append("AIST++ GT is body-only (COCO17); face and hands have no GT")
    warnings.extend(str(note) for note in clip_cfg.get("notes", []))

    units = str(_scalar(data["units"])) if "units" in data else "unknown"
    scale_factor = float(_scalar(data["scale_factor"])) if "scale_factor" in data else None
    frame = str(_scalar(data["skeleton3d_frame"])) if "skeleton3d_frame" in data else "unknown"
    floor_identity = None
    if "floor_transform" in data:
        try:
            floor = json.loads(str(data["floor_transform"]))
            floor_identity = bool(
                np.allclose(np.asarray(floor.get("rotation", np.eye(3))), np.eye(3))
                and np.allclose(np.asarray(floor.get("translation", np.zeros(3))), 0.0)
            )
        except (TypeError, json.JSONDecodeError):
            floor_identity = None
    if scale_factor is not None and not np.isclose(scale_factor, 1.0):
        warnings.append(f"scale_factor is {scale_factor}, not 1.0")
    if units not in ("meters", "m", "unknown"):
        warnings.append(f"unexpected skeleton units {units!r}")
    if floor_identity is False:
        warnings.append("floor transform is not identity; evaluation did not invert it")

    result = {
        "clip": str(clip_cfg.get("clip_id", "")),
        "role": str(clip_cfg.get("role", "")),
        "dataset": dataset,
        "sequence": str(clip_cfg.get("sequence", "")),
        "variant": variant,
        "skel": skel_label,
        "frames": int(len(pred)),
        "global_frames": {
            "first": int(frames[0]) if len(frames) else None,
            "last": int(frames[-1]) if len(frames) else None,
            "count": int(len(frames)),
        },
        "units": units,
        "scale_factor": scale_factor,
        "skeleton3d_frame": frame,
        "floor_transform_identity": floor_identity,
        "alignment": alignment_info,
        "groups": group_blocks,
        "pooled": pooled,
        "switch_analysis": switch_analysis,
        "per_camera_exclusion": _camera_exclusion(
            data, len(pred), clip_cfg.get("cameras")
        ),
        "worst_joints": _worst_joints(error, covered, gt_valid),
        "warnings": warnings,
    }
    return result


def _summary_lines(result: Mapping[str, Any]) -> list[str]:
    lines = [
        f"{result['clip']} {result.get('variant') or ''} frames={result['frames']} "
        f"({result['global_frames']['first']}..{result['global_frames']['last']})",
    ]
    for name in ("body", "face", "left_hand", "right_hand", "fingertips", "feet", "pooled"):
        block = result["groups"].get(name) if name != "pooled" else result["pooled"]
        if not block or not block.get("gt_available"):
            continue
        stats = block["mpjpe_mm"]
        if stats["median"] is None:
            lines.append(f"  {name:10s} cov={block['coverage']:.3f} no covered points")
            continue
        lines.append(
            f"  {name:10s} cov={block['coverage']:.3f} "
            f"med={stats['median']:.1f} mean={stats['mean']:.1f} p95={stats['p95']:.1f} mm "
            f"jit={block['jitter_mm_per_frame2']['median']:.3f}"
        )
    switch = result["switch_analysis"]
    link = switch["jump_switch_link"]
    p_switch = link["p_big_jump_given_switch"]
    p_no = link["p_big_jump_given_no_switch"]
    lines.append(
        "  switch_rate={:.4f} P(big|switch)={} P(big|no)={}".format(
            switch["switch_rate"] if switch["switch_rate"] is not None else float("nan"),
            "NA" if p_switch is None else f"{p_switch:.3f}",
            "NA" if p_no is None else f"{p_no:.3f}",
        )
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, help="clip id from evaluation/studio/clips.json")
    parser.add_argument("--skel", required=True, help="skeleton3d.npz path")
    parser.add_argument("--out", required=True, help="output metrics JSON path")
    parser.add_argument("--variant", default=None, help="variant label (NEW/OLD)")
    parser.add_argument("--align", choices=["none", "rigid"], default="none")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--clips", default=None, help="override clips JSON path")
    args = parser.parse_args(argv)

    clip_cfg = load_clip_config(args.clip, args.clips)
    result = evaluate(
        args.skel,
        clip_cfg,
        variant=args.variant,
        align=args.align,
        max_frames=args.max_frames,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    for line in _summary_lines(result):
        print(line)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
