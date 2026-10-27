"""Multi-camera frame-by-frame video pose runner.

Reads N synchronized video feeds, runs the PoseEstimator on each frame for each camera,
draws an annotated grid overlay, and writes a unified `(num_frames, num_cameras, 133, 2)`
tensor of 2D keypoints to disk.

Usage:
    from aimocap.pose.multi_video import run_multi_video
    run_multi_video(["cam1.mp4", "cam2.mp4"], "out.npz", grid_dst="grid.mp4")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import cv2
import numpy as np

from aimocap.video import _pick_subject
from aimocap.pose.keypoints import BODY_17
from aimocap.pipeline.sync_contract import validate_sync_contract

if TYPE_CHECKING:
    from aimocap.pose.infer import PoseEstimator


def _create_grid(frames: list[np.ndarray]) -> np.ndarray:
    """Arranges a list of frames into a roughly square grid."""
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=frames[0].dtype)
    
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        
        # Resize frame if it doesn't match the grid cell size (shouldn't happen in a proper multi-cam setup, but just in case)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
            
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
        
        # Add camera index label
        cv2.putText(
            grid, f"Cam {i}", (c * w + 10, r * h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        
    return grid


def _select_single_subject(poses, threshold: float):
    """Return one unambiguous performer or a status explaining why not."""
    if not poses:
        return None, "none"
    visible = [
        pose for pose in poses
        if int((pose.scores[np.asarray(BODY_17)] >= threshold).sum()) >= 4
    ]
    if len(visible) > 1:
        return None, "ambiguous"
    if not visible:
        return None, "none"
    return visible[0], "selected"


def smooth_boxes(
    raw: list[list[float]] | np.ndarray,
    aspect: float,
    pad: float = 1.25,
    win: int = 7,
    extra: float = 1.05,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Build finite uniform crops from a temporally smoothed box sequence."""
    from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

    raw_array = np.asarray(raw, dtype=np.float64).copy()
    if raw_array.size == 0:
        return []
    raw_array = raw_array.reshape((-1, 4))
    if aspect <= 0.0 or pad <= 0.0 or win < 0 or extra <= 0.0:
        raise ValueError("aspect, pad, and extra must be positive; win must be non-negative")
    valid = np.isfinite(raw_array).all(axis=1)
    indices = np.arange(len(raw_array), dtype=np.float64)
    if valid.any():
        for column in range(4):
            raw_array[~valid, column] = np.interp(
                indices[~valid], indices[valid], raw_array[valid, column]
            )
    else:
        raw_array.fill(0.0)

    centers_x = (raw_array[:, 0] + raw_array[:, 2]) / 2.0
    centers_y = (raw_array[:, 1] + raw_array[:, 3]) / 2.0
    heights = np.maximum(
        raw_array[:, 3] - raw_array[:, 1],
        (raw_array[:, 2] - raw_array[:, 0]) / aspect,
    ) * pad
    window = 2 * win + 1
    centers_x = uniform_filter1d(
        median_filter(centers_x, size=window, mode="nearest"), size=5, mode="nearest"
    )
    centers_y = uniform_filter1d(
        median_filter(centers_y, size=window, mode="nearest"), size=5, mode="nearest"
    )
    # Preserve the experiment's rolling-max safety envelope, without the
    # later averaging step that could make a crop smaller than a nearby box.
    heights = maximum_filter1d(heights, size=window, mode="nearest") * extra
    widths = heights * aspect
    return [
        (
            "uniform",
            (
                float(centers_x[i] - widths[i] / 2.0),
                float(centers_y[i] - heights[i] / 2.0),
                float(centers_x[i] + widths[i] / 2.0),
                float(centers_y[i] + heights[i] / 2.0),
            ),
        )
        for i in range(len(raw_array))
    ]



def _resolve_sync_cameras(
    sources: list[str | Path], sync_data: Mapping
) -> tuple[list[str], int, np.ndarray]:
    """Resolve every input source to one manifest camera and its offset."""
    cameras = sync_data.get("cameras", {})
    source_paths = sync_data.get("source_paths", {})
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError("synchronization manifest has no camera entries")
    reference_id = str(sync_data.get("reference_id", ""))
    if reference_id not in cameras:
        raise ValueError("synchronization manifest reference camera is missing")

    resolved: list[str] = []
    offsets: list[float] = []
    for index, source in enumerate(sources):
        path = Path(source)
        candidates: list[str] = []
        for key in (str(path), path.name, path.stem, f"cam_{index:02d}"):
            if key in cameras and key not in candidates:
                candidates.append(key)
        for camera_id, source_path in source_paths.items():
            if Path(str(source_path)).name == path.name and camera_id in cameras:
                if camera_id not in candidates:
                    candidates.append(camera_id)
        if len(candidates) != 1:
            raise ValueError(
                f"synchronization manifest does not map every input camera uniquely: "
                f"{path} -> {candidates or 'no match'}"
            )
        camera_id = candidates[0]
        item = cameras[camera_id]
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid synchronization entry for camera {camera_id}")
        offset = item.get("offset_sec")
        if offset is None:
            offset = float(item.get("offset_ms", 0.0)) / 1000.0
        offset = float(offset)
        if not np.isfinite(offset):
            raise ValueError(f"invalid synchronization offset for camera {camera_id}")
        resolved.append(camera_id)
        offsets.append(offset)

    if len(set(resolved)) != len(resolved):
        raise ValueError("synchronization manifest maps multiple inputs to one camera")
    reference_index = resolved.index(reference_id) if reference_id in resolved else -1
    if reference_index < 0:
        raise ValueError("synchronization manifest does not map the reference camera input")
    return resolved, reference_index, np.asarray(offsets, dtype=np.float64)


def run_multi_video(
    sources: list[str | Path],
    npz_dst: str | Path,
    grid_dst: str | Path | None = None,
    *,
    estimator: PoseEstimator | None = None,
    threshold: float = 0.3,
    max_frames: int | None = None,
    progress_every: int = 10,
    sync_manifest: Mapping | str | Path | None = None,
    require_sync: bool = False,
    single_subject: bool = True,
    tracker: str = "temporal",
    crop_mode: str = "detector",
    simcc_decode: str = "argmax",
    part_refine: str = "off",
    inference_engine: str = "ort",
) -> None:
    """Run pose estimation over multiple synchronized videos synchronously.
    
    Parameters
    ----------
    sources     : list of input video paths.
    npz_dst     : output path for the extracted 2D keypoint tensor (.npz).
    grid_dst    : optional output path for the annotated grid video.
    threshold   : confidence threshold for subject picking.
    max_frames  : stop after this many frames (None = whole video).
    single_subject : when True, keep exactly one persistent performer identity.
    tracker     : single-subject selection policy: "temporal" (default; a
                  temporally consistent per-camera tracker that follows one
                  identity through gaps and refuses to steal a stronger
                  newcomer) or "strict" (legacy per-frame gate that raises on
                  any frame with two visible people).
    """

    if not sources:
        raise ValueError("No sources provided.")

    if tracker not in {"temporal", "strict"}:
        raise ValueError(f"tracker must be 'temporal' or 'strict', got {tracker!r}")
    if crop_mode not in {"detector", "smoothbox"}:
        raise ValueError(f"crop_mode must be 'detector' or 'smoothbox', got {crop_mode!r}")
    if simcc_decode not in {"argmax", "subpixel"}:
        raise ValueError(
            f"simcc_decode must be 'argmax' or 'subpixel', got {simcc_decode!r}"
        )
    if part_refine not in {"off", "hands", "face", "hands_face"}:
        raise ValueError(
            f"part_refine must be 'off', 'hands', 'face', or 'hands_face', got {part_refine!r}"
        )
    if inference_engine not in {"ort", "cv2dnn"}:
        raise ValueError(f"inference_engine must be 'ort' or 'cv2dnn', got {inference_engine!r}")
    _trackers = None
    if single_subject and tracker == "temporal":
        from aimocap.pose.tracker import SingleSubjectTracker

        _trackers = [SingleSubjectTracker(threshold=threshold) for _ in sources]

    npz_dst = Path(npz_dst)
    npz_dst.parent.mkdir(parents=True, exist_ok=True)
    
    if grid_dst:
        from aimocap.viz.overlay import draw_poses
        grid_dst = Path(grid_dst)
        grid_dst.parent.mkdir(parents=True, exist_ok=True)
        
    if estimator is None:
        from aimocap.pose.infer import PoseEstimator as _PoseEstimator
        if simcc_decode == "argmax":
            est = _PoseEstimator()
        else:
            est = _PoseEstimator(simcc_decode=simcc_decode)
    else:
        est = estimator
        if simcc_decode != "argmax" and hasattr(est, "simcc_decode"):
            est.simcc_decode = simcc_decode
    
    caps = []
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Video not found: {src_path}")
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {src_path}")
        caps.append(cap)

    writer = None

    try:
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        sync_data = None
        if sync_manifest is not None:
            if isinstance(sync_manifest, (str, Path)):
                sync_data = json.loads(Path(sync_manifest).read_text(encoding="utf-8"))
            else:
                sync_data = dict(sync_manifest)
        if require_sync:
            if sync_data is None:
                raise ValueError("synchronization manifest is required; run the sync command first")
            status = str(sync_data.get("status", ""))
            if not status.startswith("VERIFIED"):
                raise ValueError("synchronization manifest is not marked VERIFIED")

        if sync_data is not None:
            sync_camera_ids, reference_index, manifest_offsets = _resolve_sync_cameras(
                sources, sync_data
            )
            sync_offsets = manifest_offsets - manifest_offsets[reference_index]
            # Dataset frame-index sync (Panoptic): each camera's local frame that
            # corresponds to Panoptic global frame G is (G - global_start_frame).
            # When the manifest carries per-camera global starts plus an explicit
            # window-start anchor, we seek by *frame index* instead of by the
            # seconds-offset formula (which only models a shared wall-clock audio
            # timeline and is the wrong mapping for genlocked, index-keyed cameras).
            _manifest_cameras = sync_data.get("cameras", {})
            global_start_by_cam = {
                str(cid): int(m.get("global_start_frame", 0))
                for cid, m in _manifest_cameras.items()
                if isinstance(m, Mapping) and "global_start_frame" in m
            }
            window_start_global = sync_data.get("window_start_global_frame")
            use_frame_index_sync = (
                bool(global_start_by_cam) and window_start_global is not None
            )
            # The NPZ ``sync_status`` is the canonical gate signal consumed by the
            # triangulation contract; keep it "VERIFIED".  The specific mechanism
            # (audio clap vs. dataset frame-index sync) is recorded separately in
            # ``sync_mechanism`` for provenance without changing the gate value.
            sync_status = "VERIFIED"
            sync_mechanism = str(sync_data.get("status", "VERIFIED"))
        else:
            sync_camera_ids = [Path(src).name for src in sources]
            reference_index = 0
            sync_offsets = np.zeros(len(sources), dtype=np.float64)
            sync_status = "UNVERIFIED_EQUAL_FRAME_INDEX"
            sync_mechanism = "UNVERIFIED_EQUAL_FRAME_INDEX"
            global_start_by_cam = {}
            window_start_global = None
            use_frame_index_sync = False

        if sync_data is not None:
            if use_frame_index_sync:
                # Panoptic frame-index sync: land the reference on the requested
                # global window start (its local index = global - global_start).
                # Other cameras are sought per-iteration by global index below.
                ref_gstart = global_start_by_cam.get(sync_camera_ids[reference_index], 0)
                ref_start_local = int(window_start_global) - ref_gstart
                if ref_start_local < 0:
                    raise ValueError(
                        f"window start global {window_start_global} precedes reference "
                        f"camera {sync_camera_ids[reference_index]} global start {ref_gstart}"
                    )
                caps[reference_index].set(cv2.CAP_PROP_POS_FRAMES, ref_start_local)
            else:
                # A positive relative offset means that camera has no frame for
                # the earliest part of the reference timeline. Start after the
                # largest unavailable head interval instead of clamping a
                # negative seek to zero and labelling the mismatch as aligned.
                start_time_s = max(0.0, float(np.max(sync_offsets)))
                if start_time_s > 0.0:
                    start_frame = int(np.ceil(start_time_s * float(fps)))
                    caps[reference_index].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        smooth_frames = []
        refine_frames = [] if part_refine != "off" else None
        smooth_raw_boxes = [[] for _ in sources]
        all_keypoints = []  # will hold list of shape (num_cameras, 133, 2)
        all_scores = []     # will hold list of shape (num_cameras, 133)
        timestamps = []
        source_frame_indices = []
        source_timestamps = []
        requested_timestamps = []
        subject_statuses = []
        image_sizes = None
        
        frame_idx = 0
        read_limit = max_frames
        if crop_mode == "smoothbox" and max_frames is not None:
            # Smooth-box crops use a centred temporal window.  Read a bounded
            # look-ahead context so truncating output does not change crops at
            # the requested end (and therefore does not change default output).
            read_limit = max_frames + 15
        while True:
            if read_limit is not None and frame_idx >= read_limit:
                break

            frames_ok = []
            frames = []
            frame_indices = []
            frame_times = []
            frame_requested_times = []
            if sync_data is None:
                for cap in caps:
                    ok, f = cap.read()
                    frames_ok.append(ok)
                    frames.append(f)
                    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    if not np.isfinite(ts) or ts < 0:
                        ts = max(0.0, idx / float(fps))
                    frame_indices.append(max(0, idx))
                    frame_times.append(ts)
                    frame_requested_times.append(ts)
            elif use_frame_index_sync:
                # Per-iteration global-index seeking.  The reference advances by
                # one decoded frame; the others are sought to the matching local
                # index (global - global_start).  All timestamps are expressed in
                # global-frame seconds so the sync contract's +/-half-frame budget
                # is meaningful on the global timeline.
                target_global = int(window_start_global) + frame_idx
                frames_ok = [False] * len(caps)
                frames = [None] * len(caps)
                frame_indices = [None] * len(caps)
                frame_times = [None] * len(caps)
                frame_requested_times = [None] * len(caps)
                for ci, cap in enumerate(caps):
                    cid = sync_camera_ids[ci]
                    gstart = global_start_by_cam.get(cid, 0)
                    local_idx = target_global - gstart
                    if local_idx < 0:
                        continue  # camera has no frame at this global instant
                    if ci == reference_index:
                        ok, f = cap.read()  # reference is already positioned
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                        ok, f = cap.read()
                    frames_ok[ci] = ok
                    frames[ci] = f
                    actual_local = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    # Source/requested both expressed in global-frame seconds.
                    req_global_s = target_global / float(fps)
                    src_global_s = (actual_local + gstart) / float(fps)
                    frame_indices[ci] = max(0, actual_local)
                    frame_times[ci] = src_global_s
                    frame_requested_times[ci] = req_global_s
            else:
                frames_ok = [False] * len(caps)
                frames = [None] * len(caps)
                frame_indices = [None] * len(caps)
                frame_times = [None] * len(caps)
                frame_requested_times = [None] * len(caps)
                ok, f = caps[reference_index].read()
                frames_ok[reference_index] = ok
                frames[reference_index] = f
                idx = int(caps[reference_index].get(cv2.CAP_PROP_POS_FRAMES)) - 1
                ref_actual_ts = float(caps[reference_index].get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not np.isfinite(ref_actual_ts) or ref_actual_ts < 0:
                    ref_actual_ts = max(0.0, idx / float(fps))
                ref_requested_ts = max(0.0, idx / float(fps))
                frame_indices[reference_index] = max(0, idx)
                frame_times[reference_index] = ref_actual_ts
                frame_requested_times[reference_index] = ref_requested_ts
                for ci, cap in enumerate(caps):
                    if ci == reference_index:
                        continue
                    # The sync engine defines a negative offset as "other
                    # camera's clap is later".  Its source timestamp must
                    # therefore be advanced by ``-offset`` to align content.
                    target_ts = max(0.0, ref_requested_ts - sync_offsets[ci])
                    cap.set(cv2.CAP_PROP_POS_MSEC, target_ts * 1000.0)
                    ok, f = cap.read()
                    frames_ok[ci] = ok
                    frames[ci] = f
                    source_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    actual_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    if not np.isfinite(actual_ts) or actual_ts < 0:
                        actual_ts = target_ts
                    frame_indices[ci] = max(0, source_idx)
                    frame_times[ci] = actual_ts
                    frame_requested_times[ci] = target_ts
                
            if not all(frames_ok):
                # Stop if ANY camera runs out of frames
                break
                
            if frame_idx == 0:
                image_sizes = [(f.shape[1], f.shape[0]) for f in frames]

            if crop_mode == "smoothbox":
                smooth_frames.append([f.copy() for f in frames])
                if refine_frames is not None:
                    refine_frames.append(smooth_frames[-1])
                frame_subject_statuses = []
                for ci, f in enumerate(frames):
                    poses = est.estimate(f, pick="all")
                    if single_subject:
                        if _trackers is not None:
                            subject, subject_status = _trackers[ci].select(
                                poses, f.shape if f is not None else None
                            )
                        else:
                            subject, subject_status = _select_single_subject(poses, threshold)
                            if subject_status == "ambiguous":
                                raise ValueError(
                                    f"ambiguous performer selection at output frame {frame_idx}; "
                                    "use a single-performer capture or explicitly disable the gate"
                                )
                    else:
                        subject = _pick_subject(poses, threshold) if poses else None
                        subject_status = "selected" if subject is not None else "none"
                    frame_subject_statuses.append(subject_status)
                    if subject is None:
                        smooth_raw_boxes[ci].append([np.nan] * 4)
                    else:
                        smooth_raw_boxes[ci].append(np.asarray(subject.bbox, dtype=np.float64))
                reference_timestamp = frame_times[reference_index]
                timestamps.append(reference_timestamp)
                source_frame_indices.append(frame_indices)
                source_timestamps.append(frame_times)
                requested_timestamps.append(frame_requested_times)
                subject_statuses.append(frame_subject_statuses)
                frame_idx += 1
                continue

            t0 = time.perf_counter()
            if refine_frames is not None:
                refine_frames.append([f.copy() for f in frames])
            
            cam_keypoints = []
            cam_scores = []
            annotated_frames = []
            frame_subject_statuses = []
            
            for ci, f in enumerate(frames):
                poses = est.estimate(f, pick="all")
                if single_subject:
                    if _trackers is not None:
                        # Temporally consistent identity: one lock per camera,
                        # follows through gaps, never steals a stronger newcomer.
                        subject, subject_status = _trackers[ci].select(
                            poses, f.shape if f is not None else None
                        )
                    else:
                        subject, subject_status = _select_single_subject(poses, threshold)
                        if subject_status == "ambiguous":
                            raise ValueError(
                                f"ambiguous performer selection at output frame {frame_idx}; "
                                "use a single-performer capture or explicitly disable the gate"
                            )
                else:
                    subject = _pick_subject(poses, threshold) if poses else None
                    subject_status = "selected" if subject is not None else "none"
                frame_subject_statuses.append(subject_status)
                
                if subject is not None:
                    cam_keypoints.append(subject.keypoints.copy())
                    cam_scores.append(subject.scores.copy())
                    to_draw = [subject]
                else:
                    # No person detected in this view
                    cam_keypoints.append(np.full((133, 2), np.nan, dtype=np.float32))
                    cam_scores.append(np.zeros(133, dtype=np.float32))
                    to_draw = []
                    
                if grid_dst:
                    annotated = draw_poses(f, to_draw, threshold=threshold) if to_draw else f.copy()
                    annotated_frames.append(annotated)
            
            all_keypoints.append(cam_keypoints)
            all_scores.append(cam_scores)
            reference_timestamp = frame_times[reference_index]
            timestamps.append(reference_timestamp)
            source_frame_indices.append(frame_indices)
            source_timestamps.append(frame_times)
            requested_timestamps.append(frame_requested_times)
            subject_statuses.append(frame_subject_statuses)
            
            if grid_dst:
                grid = _create_grid(annotated_frames)
                if writer is None:
                    h, w = grid.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if grid_dst.suffix.lower() == ".mp4" else "MJPG"))
                    writer = cv2.VideoWriter(str(grid_dst), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise OSError(f"Could not open VideoWriter for {grid_dst}")
                writer.write(grid)
                
            dt = (time.perf_counter() - t0) * 1000.0
            
            if progress_every and (frame_idx % progress_every == 0):
                det_count = sum(1 for s in cam_scores if s.mean() > 0)
                print(f"  multi-frame {frame_idx:>4} | {det_count}/{len(caps)} cameras detected | {dt:.1f}ms")
                
            frame_idx += 1

        if crop_mode == "smoothbox":
            aspect = est.input_w / float(est.input_h)
            smooth_crops = [
                smooth_boxes(boxes, aspect, pad=1.25, win=7, extra=1.05)
                for boxes in smooth_raw_boxes
            ]
            all_keypoints = []
            all_scores = []
            output_frames = frame_idx
            output_count = (
                min(max_frames, output_frames)
                if max_frames is not None
                else output_frames
            )
            smooth_frames = smooth_frames[:output_count]
            if refine_frames is not None:
                refine_frames = refine_frames[:output_count]
            timestamps = timestamps[:output_count]
            source_frame_indices = source_frame_indices[:output_count]
            source_timestamps = source_timestamps[:output_count]
            requested_timestamps = requested_timestamps[:output_count]
            subject_statuses = subject_statuses[:output_count]
            for output_idx, frames in enumerate(smooth_frames):
                t0 = time.perf_counter()
                cam_keypoints = []
                cam_scores = []
                annotated_frames = []
                for ci, f in enumerate(frames):
                    raw_box = np.asarray(smooth_raw_boxes[ci][output_idx], dtype=np.float64)
                    if np.isfinite(raw_box).all():
                        _, crop_xyxy = smooth_crops[ci][output_idx]
                        pose = est.estimate_with_crop(f, crop_xyxy)
                        # Keep the selected detector box as the pose identity bbox.
                        pose.bbox = raw_box.astype(np.float32)
                        cam_keypoints.append(pose.keypoints.copy())
                        cam_scores.append(pose.scores.copy())
                        to_draw = [pose]
                    else:
                        cam_keypoints.append(
                            np.full((133, 2), np.nan, dtype=np.float32)
                        )
                        cam_scores.append(np.zeros(133, dtype=np.float32))
                        to_draw = []
                    if grid_dst:
                        annotated = (
                            draw_poses(f, to_draw, threshold=threshold)
                            if to_draw
                            else f.copy()
                        )
                        annotated_frames.append(annotated)
                all_keypoints.append(cam_keypoints)
                all_scores.append(cam_scores)
                if grid_dst:
                    grid = _create_grid(annotated_frames)
                    if writer is None:
                        h, w = grid.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(
                            *("mp4v" if grid_dst.suffix.lower() == ".mp4" else "MJPG")
                        )
                        writer = cv2.VideoWriter(str(grid_dst), fourcc, fps, (w, h))
                        if not writer.isOpened():
                            raise OSError(f"Could not open VideoWriter for {grid_dst}")
                    writer.write(grid)
                dt = (time.perf_counter() - t0) * 1000.0
                if progress_every and (output_idx % progress_every == 0):
                    det_count = sum(1 for s in cam_scores if s.mean() > 0)
                    print(
                        f"  multi-frame {output_idx:>4} | {det_count}/{len(caps)} "
                        f"cameras detected | {dt:.1f}ms"
                    )
            frame_idx = output_count

        if part_refine != "off":
            # Refinement is deliberately a second pass.  Body inference and its
            # calibrated scores are complete before any crop is constructed.
            from aimocap.pose.part_refine import (
                HANDS,
                FaceRefiner,
                HandRefiner,
                estimate_hand_sizes,
                hand_agreement_distances,
                hand_crop_boxes,
                head_crop_boxes,
            )
            from aimocap.pose.score_calibration import (
                calibrate_hand_agreement_scores,
                calibrate_refined_part_scores,
                load_part_calibration,
            )

            refine_started = time.perf_counter()
            kpt_arr = np.asarray(all_keypoints, dtype=np.float32)
            score_arr = np.asarray(all_scores, dtype=np.float32)
            keypoints_body_raw = kpt_arr.copy()
            hand_source = np.zeros(kpt_arr.shape[:-1], dtype=np.uint8)
            hand_scores_raw = np.full(score_arr.shape, np.nan, dtype=np.float32)
            face_source = (
                np.zeros(kpt_arr.shape[:-1], dtype=np.uint8)
                if "face" in part_refine
                else None
            )
            face_scores_raw = (
                np.full(score_arr.shape, np.nan, dtype=np.float32)
                if "face" in part_refine
                else None
            )
            providers = getattr(est, "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
            hand_refiner = HandRefiner(
                providers=providers, inference_engine=inference_engine
            ) if "hands" in part_refine else None
            face_refiner = None
            if "face" in part_refine:
                reuse = est if getattr(est, "model_family", None) == "rtmw" else None
                face_refiner = FaceRefiner(
                    providers=providers,
                    estimator=reuse,
                    inference_engine=inference_engine,
                )

            body_by_camera = np.transpose(keypoints_body_raw, (1, 0, 2, 3))
            scores_by_camera = np.transpose(score_arr, (1, 0, 2))
            hand_crops = [hand_crop_boxes(body_by_camera[c], scores_by_camera[c])
                          for c in range(len(caps))] if hand_refiner else None
            face_crops = [head_crop_boxes(body_by_camera[c], scores_by_camera[c])
                          for c in range(len(caps))] if face_refiner else None
            for frame_number, frames_for_cameras in enumerate(refine_frames or []):
                for camera_number, frame in enumerate(frames_for_cameras):
                    body = kpt_arr[frame_number, camera_number]
                    body_scores = score_arr[frame_number, camera_number]
                    if not np.isfinite(body).any():
                        continue
                    if hand_refiner is not None:
                        for hand in ("left", "right"):
                            hand_index = 0 if hand == "left" else 1
                            points, _merged_scores, source = hand_refiner.refine(
                                frame, body, body_scores, hand,
                                tuple(hand_crops[camera_number][frame_number, hand_index]),
                            )
                            raw_scores = hand_refiner.last_raw_scores
                            start = 91 if hand == "left" else 112
                            kpt_arr[frame_number, camera_number, start:start + 21] = points
                            hand_source[frame_number, camera_number, start:start + 21] = source
                            hand_scores_raw[frame_number, camera_number, start:start + 21] = raw_scores
                    if face_refiner is not None:
                        points, raw_scores = face_refiner.refine(
                            frame, tuple(face_crops[camera_number][frame_number])
                        )
                        kpt_arr[frame_number, camera_number, 23:91] = points
                        face_scores_raw[frame_number, camera_number, 23:91] = raw_scores
                        face_source[frame_number, camera_number, 23:91] = (
                            np.isfinite(points).all(axis=1).astype(np.uint8)
                        )
            # The refined points now carry model-native scores.  Face points
            # use the S2 isotonic map; hand points use cross-model agreement
            # with the RTMW body pass.  Points that fell back to RTMW keep the
            # body-calibrated score.  Raw keys stay untouched.
            part_table = load_part_calibration()
            body_calibrated_scores = np.asarray(score_arr, dtype=np.float32).copy()
            score_arr = np.asarray(
                calibrate_refined_part_scores(
                    score_arr,
                    face_scores_raw=face_scores_raw,
                    face_source=face_source,
                    table=part_table,
                ),
                dtype=np.float32,
            )
            hand_agreement_d = None
            hand_size_px = None
            if hand_refiner is not None:
                hand_config = part_table["hand_agreement"]["hand_size"]
                hand_size_px = np.full(
                    (keypoints_body_raw.shape[0], keypoints_body_raw.shape[1], len(HANDS)),
                    np.nan,
                    dtype=np.float64,
                )
                hand_agreement_d = np.full(kpt_arr.shape[:3], np.nan, dtype=np.float64)
                for camera_number in range(keypoints_body_raw.shape[1]):
                    sizes, _hand_size_fallback = estimate_hand_sizes(
                        keypoints_body_raw[:, camera_number],
                        body_calibrated_scores[:, camera_number],
                        hand_config,
                    )
                    hand_size_px[:, camera_number] = sizes
                    hand_agreement_d[:, camera_number] = hand_agreement_distances(
                        kpt_arr[:, camera_number],
                        keypoints_body_raw[:, camera_number],
                        hand_source[:, camera_number],
                        sizes,
                    )
                score_arr = np.asarray(
                    calibrate_hand_agreement_scores(
                        score_arr,
                        hand_agreement_d,
                        hand_source,
                        table=part_table,
                    ),
                    dtype=np.float32,
                )
            part_calibration_sha256 = str(part_table["_sha256"])
            overhead = (time.perf_counter() - refine_started) * 1000.0
            print(
                f"  part refinement {part_refine}: {overhead / max(frame_idx * len(caps), 1):.2f}ms/frame "
                f"(hand providers={getattr(hand_refiner, 'active_providers', []) if hand_refiner else []}; "
                f"face providers={getattr(face_refiner, 'active_providers', []) if face_refiner else []})"
            )

    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
            
    # A manifest is not enough by itself: verify the timestamps actually
    # returned by decoding before labelling the artifact as VERIFIED.
    sync_validation = None
    if sync_data is not None:
        sync_validation = validate_sync_contract(
            sync_status,
            np.asarray(source_timestamps, dtype=np.float64),
            np.asarray(requested_timestamps, dtype=np.float64),
            fps=float(fps),
        )

    # Save to disk
    if frame_idx == 0 or image_sizes is None:
        raise RuntimeError("no synchronized frames were decoded")
    if part_refine == "off":
        kpt_arr = np.array(all_keypoints, dtype=np.float32)  # (F, C, 133, 2)
        score_arr = np.array(all_scores, dtype=np.float32)   # (F, C, 133)
    else:
        # The opt-in second pass has already produced these arrays and the
        # untouched baseline copy above.
        kpt_arr = np.asarray(kpt_arr, dtype=np.float32)
        score_arr = np.asarray(score_arr, dtype=np.float32)
    img_sizes_arr = np.array(image_sizes, dtype=np.int32)
    camera_names_arr = np.array([Path(src).name for src in sources], dtype=str)
    
    npz_fields = {
        "keypoints": kpt_arr,
        "scores": score_arr,
        "image_sizes": img_sizes_arr,
        "camera_names": camera_names_arr,
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "fps": np.asarray(float(fps)),
        "source_frame_indices": np.asarray(source_frame_indices, dtype=np.int64),
        "source_timestamps": np.asarray(source_timestamps, dtype=np.float64),
        "requested_timestamps": np.asarray(requested_timestamps, dtype=np.float64),
        "subject_status": np.asarray(subject_statuses, dtype=str),
        "sync_offsets": sync_offsets,
        "sync_status": np.asarray(sync_status),
        "sync_mechanism": np.asarray(sync_mechanism),
        "sync_camera_ids": np.asarray(sync_camera_ids, dtype=str),
        "sync_reference_camera": np.asarray(sync_camera_ids[reference_index]),
        "sync_validation": np.asarray(json.dumps(sync_validation or {}, sort_keys=True)),
        "subject_selection": np.asarray(tracker if single_subject else "largest"),
        "coordinate_space": np.asarray("image_pixels"),
        "units": np.asarray("pixels"),
        "joint_convention": np.asarray("coco-wholebody"),
        "inference_engine": np.asarray(inference_engine),
        "cv2_version": np.asarray(cv2.__version__),
        "sync_manifest": np.asarray(json.dumps(sync_data or {}, sort_keys=True)),
    }
    if crop_mode != "detector":
        npz_fields["crop_mode"] = np.asarray(crop_mode)
    if simcc_decode != "argmax":
        npz_fields["simcc_decode"] = np.asarray(simcc_decode)
    if part_refine != "off":
        npz_fields.update({
            "keypoints_body_raw": np.asarray(keypoints_body_raw, dtype=np.float32),
            "hand_source": np.asarray(hand_source, dtype=np.uint8),
            "hand_scores_raw": np.asarray(hand_scores_raw, dtype=np.float32),
            "part_refine": np.asarray(part_refine),
            "part_score_calibration_sha256": np.asarray(part_calibration_sha256),
            "hand_model_sha256": np.asarray(
                getattr(hand_refiner, "sha256", "") if "hands" in part_refine else ""
            ),
        })
        if hand_refiner is not None:
            npz_fields.update({
                "hand_agreement_d": np.asarray(hand_agreement_d, dtype=np.float32),
                "hand_size_px": np.asarray(hand_size_px, dtype=np.float32),
            })
        if "face" in part_refine:
            npz_fields.update({
                "face_scores_raw": np.asarray(face_scores_raw, dtype=np.float32),
                "face_source": np.asarray(face_source, dtype=np.uint8),
            })
    np.savez(npz_dst, **npz_fields)
    
    print(f"\nDone: {frame_idx} frames across {len(caps)} cameras.")
    print(f"Saved tensors to {npz_dst} -> keypoints {kpt_arr.shape}, scores {score_arr.shape}, image_sizes {img_sizes_arr.shape}, sync={sync_status}")
    if grid_dst:
        print(f"Saved grid video to {grid_dst}")

__all__ = ["run_multi_video", "smooth_boxes"]
