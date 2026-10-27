"""Multi-camera sync engine: video files → per-camera time offset table.

Pipeline per camera: extract audio (16kHz mono) → detect clap → FFT
cross-correlate each camera's clap window against a reference camera. The
reference is whatever camera has the strongest clap onset; its offset is 0.

Output is a SyncTable: {camera_id: {offset_ms, frame_offset, confidence,
clap_time_s}}. Callers (triangulation stage) use frame_offset to align frames
across cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from aimocap.sync.audio import AudioTrack, DEFAULT_SAMPLE_RATE, extract_audio
from aimocap.sync.detect import (
    cross_correlate_offset, detect_clap, offset_samples_to_ms,
)


@dataclass(slots=True)
class CameraSync:
    camera_id: str
    offset_ms: float        # relative to reference camera (positive = later)
    frame_offset: float     # offset_ms * fps / 1000
    clap_time_s: float      # when the clap occurs in THIS camera's timeline
    confidence: float       # xcorr peak sharpness (>= ~3 is trustworthy)


@dataclass(slots=True)
class SyncTable:
    reference_id: str
    sample_rate: int
    fps: float
    cameras: dict[str, CameraSync]

    def to_dict(self) -> dict:
        return {
            "reference_id": self.reference_id,
            "sample_rate": self.sample_rate,
            "fps": self.fps,
            "cameras": {k: asdict(v) for k, v in self.cameras.items()},
        }


def synchronize(
    sources: Sequence[str | Path],
    camera_ids: Sequence[str] | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fps: float = 30.0,
    window_s: float = 0.5,
) -> SyncTable:
    """Synchronize N video files by their clap transients.

    Parameters
    ----------
    sources : paths to the per-camera video files.
    camera_ids : optional names; defaults to the file stem.
    fps : video frame rate, used to convert ms → frame offsets.
    """
    if len(sources) < 2:
        raise ValueError("need at least 2 cameras to synchronize")
    if camera_ids is None:
        camera_ids = [Path(s).stem for s in sources]
    if len(camera_ids) != len(sources):
        raise ValueError("camera_ids length must match sources length")

    # 1. Extract audio for every camera.
    tracks: dict[str, AudioTrack] = {}
    for cid, src in zip(camera_ids, sources):
        tracks[cid] = extract_audio(src, sample_rate=sample_rate)

    # 2. Detect clap per track; reject cameras with no detectable clap.
    claps: dict[str, float] = {}
    for cid, tr in tracks.items():
        t = detect_clap(tr)
        if t < 0:
            raise RuntimeError(f"no clap detected in camera {cid}")
        claps[cid] = t

    # 3. Reference = strongest onset (largest clap-window energy).
    def _clap_strength(tr: AudioTrack, t: float) -> float:
        c = int(t * tr.sample_rate)
        w = tr.samples[max(0, c - 200): c + 200]
        return float(np.sqrt(np.mean(w ** 2))) if w.size else 0.0

    ref_id = max(camera_ids, key=lambda c: _clap_strength(tracks[c], claps[c]))

    # 4. Cross-correlate every other camera against the reference.
    cameras: dict[str, CameraSync] = {ref_id: CameraSync(
        camera_id=ref_id, offset_ms=0.0, frame_offset=0.0,
        clap_time_s=claps[ref_id], confidence=float("inf"),
    )}
    for cid in camera_ids:
        if cid == ref_id:
            continue
        # Full-track FFT xcorr — the clap is the dominant transient and the
        # xcorr finds its lag directly. Do NOT window (windowing was a previous
        # design that pre-aligned the crops and cancelled the offset).
        offset_samp, conf = cross_correlate_offset(
            tracks[ref_id], tracks[cid],
        )
        offset_ms = offset_samples_to_ms(offset_samp, sample_rate)
        cameras[cid] = CameraSync(
            camera_id=cid,
            offset_ms=offset_ms,
            frame_offset=offset_ms * fps / 1000.0,
            clap_time_s=claps[cid],
            confidence=conf,
        )

    return SyncTable(
        reference_id=ref_id,
        sample_rate=sample_rate,
        fps=fps,
        cameras=cameras,
    )


__all__ = ["CameraSync", "SyncTable", "synchronize"]
