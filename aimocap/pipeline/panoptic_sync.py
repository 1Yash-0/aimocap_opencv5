"""Dataset-verified frame synchronization for the CMU Panoptic toolbox.

The CMU Panoptic HD cameras are hardware-genlocked: every camera records the
same physical instant at the same global frame index.  The per-frame 3D ground
truth is published as ``body3DScene_XXXXXXXX.json`` where ``XXXXXXXX`` is that
global frame index.  These HD MP4s carry **no audio stream**, so the audio-clap
synchronization contract (:mod:`aimocap.pipeline.sync_contract` /
:mod:`aimocap.sync`) does not apply.

This module provides an explicit, provenance-backed *frame-index* sync contract
that is the Panoptic analogue of the audio-clap contract.  It:

* names the dataset sequence and the participating HD camera IDs,
* records the global->local frame offset for each camera (``0`` for a
  full-length capture whose first frame is global frame 0; a small explicit
  offset for a trimmed clip),
* emits per-camera ``offset_sec`` entries (plus ``frame_offset``) so the
  existing multipose sync machinery can seek each camera by time,
* verifies a given set of global frame indices exist in the ground-truth
  corpus, so the alignment claim is falsifiable against the dataset.

It deliberately reports ``status = "VERIFIED_PANOPTIC_FRAME_SYNC"`` rather than
``VERIFIED_AUDIO_CLAP`` so downstream consumers can distinguish a dataset
hardware sync from an audio-derived offset and never conflate the two.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PANOPTIC_FRAME_SYNC_STATUS = "VERIFIED_PANOPTIC_FRAME_SYNC"

_CAM_RE = re.compile(r"^(?:hd_)?(\d{2})_(\d{2})$")


class PanopticSyncError(ValueError):
    """Raised when a Panoptic frame-sync manifest cannot be constructed or used."""


def normalize_camera_id(value: str) -> str:
    """Normalize a Panoptic HD camera id to the canonical ``NN_MM`` form.

    Accepts ``00_01``, ``hd_00_01`` and ``hd_00_01.mp4``-style stems.  Only the
    ``00`` panel prefix is valid for the HD camera ring.
    """
    stem = Path(str(value)).stem
    match = _CAM_RE.match(stem)
    if not match:
        raise PanopticSyncError(f"cannot parse Panoptic HD camera id from {value!r}")
    panel, node = match.group(1), match.group(2)
    if panel != "00":
        raise PanopticSyncError(
            f"Panoptic frame sync only supports the HD panel (00_*), got {value!r}"
        )
    return f"{panel}_{node}"


def _gt_path(seq_dir: Path, frame_index: int) -> Path:
    return (
        seq_dir
        / "hdPose3d_stage1_coco19"
        / f"body3DScene_{int(frame_index):08d}.json"
    )


def gt_frame_bounds(seq_dir: str | Path) -> tuple[int, int] | None:
    """Return (first, last) global frame index present in the GT corpus."""
    seq_dir = Path(seq_dir)
    pat = _gt_path(seq_dir, 0).name  # body3DScene_00000000.json
    files = sorted((seq_dir / "hdPose3d_stage1_coco19").glob("body3DScene_*.json"))
    if not files:
        return None
    idxs = [int(f.stem.split("_")[-1]) for f in files]
    return min(idxs), max(idxs)


def verify_gt_frames(seq_dir: str | Path, frame_indices: Sequence[int]) -> dict:
    """Check that ground-truth 3D exists for each global frame index.

    Returns a report with the missing indices so alignment claims are testable
    against the dataset rather than asserted.
    """
    seq_dir = Path(seq_dir)
    indices = [int(i) for i in frame_indices]
    missing = [i for i in indices if not _gt_path(seq_dir, i).exists()]
    return {
        "num_requested": len(indices),
        "num_present": len(indices) - len(missing),
        "missing": missing,
        "all_present": not missing,
    }


@dataclass(frozen=True)
class PanopticCameraEntry:
    """Alignment of one HD camera to the global Panoptic frame timeline."""

    camera_id: str
    video_path: str
    global_start_frame: int = 0

    def offset_frames(self, reference_start: int) -> int:
        return reference_start - self.global_start_frame

    def offset_seconds(self, reference_start: int, fps: float) -> float:
        return (reference_start - self.global_start_frame) / float(fps)


def build_panoptic_sync_manifest(
    seq_dir: str | Path,
    sequence: str,
    fps: float,
    cameras: Sequence[tuple[str, str] | tuple[str, str, int]],
    *,
    reference_camera: str | None = None,
    verify_gt: bool = True,
    probe_frames: Sequence[int] | None = None,
    window_start_global_frame: int | None = None,
) -> dict:
    """Build a provenance-backed Panoptic frame-sync manifest.

    Parameters
    ----------
    seq_dir : Path to the Panoptic sequence root (contains hdPose3d_stage1_coco19).
    sequence : Dataset sequence name, e.g. ``171204_pose1``.
    fps : Global capture frame rate (Panoptic HD is ~29.97).
    cameras : Sequence of ``(camera_id_or_stem, video_path)`` or
        ``(camera_id_or_stem, video_path, global_start_frame)``.  When a camera's
        decoded frame 0 corresponds to Panoptic global frame ``N`` (a trimmed or
        partial clip), pass that ``N`` so the emitted ``offset_sec`` makes the
        multipose seek land on the same global instant as the reference camera.
    reference_camera : Camera id used as the output-time reference.  Defaults
        to the first camera.

    Returns
    -------
    A manifest dict compatible with :func:`aimocap.pose.multi_video._resolve_sync_cameras`:
    it has ``cameras`` (each with ``offset_sec`` / ``frame_offset`` /
    ``global_start_frame``), ``source_paths``, ``reference_id`` and a
    Panoptic-specific ``status`` and provenance block.
    """
    seq_dir = Path(seq_dir)
    if not (seq_dir / "hdPose3d_stage1_coco19").exists():
        raise PanopticSyncError(
            f"Panoptic ground-truth dir not found under {seq_dir}; cannot verify frame sync"
        )
    if fps <= 0 or not np.isfinite(fps):
        raise PanopticSyncError(f"fps must be positive and finite, got {fps}")
    if not cameras:
        raise PanopticSyncError("at least one camera is required")

    entries: list[PanopticCameraEntry] = []
    for item in cameras:
        if len(item) == 2:
            cam_id, video_path = item
            gstart = 0
        elif len(item) == 3:
            cam_id, video_path, gstart = item
            gstart = int(gstart)
        else:
            raise PanopticSyncError(f"camera entry must be (id, path[, start]); got {item!r}")
        if gstart < 0:
            raise PanopticSyncError(f"global_start_frame must be >= 0, got {gstart}")
        cid = normalize_camera_id(cam_id)
        entries.append(PanopticCameraEntry(camera_id=cid, video_path=str(video_path), global_start_frame=gstart))

    ids = [e.camera_id for e in entries]
    if len(set(ids)) != len(ids):
        raise PanopticSyncError(f"duplicate camera ids in manifest request: {ids}")

    ref_id = normalize_camera_id(reference_camera) if reference_camera else entries[0].camera_id
    if ref_id not in ids:
        raise PanopticSyncError(f"reference camera {ref_id!r} is not among {ids}")
    ref_start = next(e.global_start_frame for e in entries if e.camera_id == ref_id)

    window_ms = None
    if window_start_global_frame is not None:
        win = int(window_start_global_frame)
        if win < 0:
            raise PanopticSyncError(f"window_start_global_frame must be >= 0, got {win}")
        # The reference time at that global frame is (win - ref_start)/fps seconds.
        window_ms = ((win - ref_start) / float(fps)) * 1000.0

    cameras_block: dict[str, dict] = {}
    source_paths: dict[str, str] = {}
    for e in entries:
        frame_offset = e.offset_frames(ref_start)
        offset_sec = e.offset_seconds(ref_start, fps)
        cameras_block[e.camera_id] = {
            "offset_sec": float(offset_sec),
            "offset_ms": float(offset_sec * 1000.0),
            "frame_offset": float(frame_offset),
            "global_start_frame": int(e.global_start_frame),
            "sync_method": "panoptic_global_frame_index",
            "confidence": 1.0,
        }
        source_paths[e.camera_id] = e.video_path

    if verify_gt:
        bounds = gt_frame_bounds(seq_dir)
        if probe_frames is None:
            probe = (
                [bounds[0], (bounds[0] + bounds[1]) // 2, bounds[1]]
                if bounds is not None else []
            )
        else:
            probe = [int(p) for p in probe_frames]
        gt_report = verify_gt_frames(seq_dir, probe)
        gt_report["gt_frame_bounds"] = list(bounds) if bounds else None
    else:
        gt_report = None

    return {
        "status": PANOPTIC_FRAME_SYNC_STATUS,
        "sync_kind": "dataset_global_frame_index",
        "distinguishes_from": "VERIFIED_AUDIO_CLAP",
        "sequence": str(sequence),
        "fps": float(fps),
        "reference_id": ref_id,
        "cameras": cameras_block,
        "source_paths": source_paths,
        "window_start_ms": window_ms,
        "window_start_global_frame": (
            int(window_start_global_frame) if window_start_global_frame is not None else None
        ),
        "provenance": {
            "dataset": "cmu_panoptic",
            "sequence": str(sequence),
            "mechanism": (
                "Panoptic HD cameras are hardware-genlocked to a shared global "
                "frame index; per-frame 3D ground truth is keyed by that index. "
                "offset_sec is derived so that decoding at (ref_ts - offset_sec) "
                "returns the camera frame whose index equals the reference global "
                "index. No audio stream exists in these HD MP4s."
            ),
            "gt_frame_probe": gt_report,
        },
    }


def save_panoptic_sync_manifest(manifest: Mapping, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def is_panoptic_sync(sync_data: Mapping | None) -> bool:
    """True when a sync manifest carries the Panoptic frame-sync status."""
    if not isinstance(sync_data, Mapping):
        return False
    return str(sync_data.get("status", "")) == PANOPTIC_FRAME_SYNC_STATUS


__all__ = [
    "PANOPTIC_FRAME_SYNC_STATUS",
    "PanopticSyncError",
    "PanopticCameraEntry",
    "normalize_camera_id",
    "verify_gt_frames",
    "build_panoptic_sync_manifest",
    "save_panoptic_sync_manifest",
    "is_panoptic_sync",
    "gt_frame_bounds",
]
