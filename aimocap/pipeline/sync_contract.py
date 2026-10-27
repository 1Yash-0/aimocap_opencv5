"""Validation helpers for time-aligned multi-camera pose artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

# Statuses that satisfy the "verified time-aligned capture" contract.  The
# audio-clap contract applies to ad-hoc multi-phone captures; the Panoptic
# contract applies to the dataset's hardware-genlocked, GT-indexed HD cameras.
VERIFIED_STATUSES = frozenset(
    {"VERIFIED", "VERIFIED_AUDIO_CLAP", "VERIFIED_PANOPTIC_FRAME_SYNC"}
)


class SyncContractError(ValueError):
    """Raised when pose timestamps cannot support calibrated triangulation."""


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(value)


def validate_sync_contract(
    sync_status: Any,
    source_timestamps: np.ndarray | None,
    requested_timestamps: np.ndarray | None,
    *,
    fps: float,
    max_error_s: float | None = None,
) -> dict[str, Any]:
    """Validate decoded timestamps against the requested synchronized timeline.

    Both timestamp arrays use ``(frames, cameras)`` order.  The default error
    budget is half a frame, which allows normal frame quantization while
    rejecting a seek/decode result that landed on a materially different time.
    """
    if fps <= 0 or not np.isfinite(fps):
        raise SyncContractError("fps must be a positive finite value")
    status = _scalar_text(sync_status)
    if status not in VERIFIED_STATUSES:
        raise SyncContractError(
            f"sync status must be one of {sorted(VERIFIED_STATUSES)}, got {status!r}"
        )
    if source_timestamps is None or requested_timestamps is None:
        raise SyncContractError(
            "both source_timestamps and requested_timestamps are required"
        )

    source = np.asarray(source_timestamps, dtype=np.float64)
    requested = np.asarray(requested_timestamps, dtype=np.float64)
    if source.ndim != 2 or requested.ndim != 2:
        raise SyncContractError("timestamp arrays must have shape (frames, cameras)")
    if source.shape != requested.shape:
        raise SyncContractError(
            f"timestamp arrays must have the same shape, got {source.shape} and {requested.shape}"
        )
    if source.size == 0:
        raise SyncContractError("timestamp arrays must not be empty")
    if not np.isfinite(source).all() or not np.isfinite(requested).all():
        raise SyncContractError("timestamp arrays must contain only finite values")
    if (source < 0).any() or (requested < 0).any():
        raise SyncContractError("timestamp arrays must contain non-negative values")

    budget = 0.5 / float(fps) if max_error_s is None else float(max_error_s)
    if budget < 0 or not np.isfinite(budget):
        raise SyncContractError("max_error_s must be a finite non-negative value")
    error = np.abs(source - requested)
    max_error = float(error.max())
    if max_error > budget:
        raise SyncContractError(
            f"timestamp error {max_error:.6f}s exceeds allowed {budget:.6f}s"
        )

    return {
        "status": "VERIFIED",
        "fps": float(fps),
        "max_error_s": budget,
        "max_abs_error_s": max_error,
        "mean_abs_error_s": float(error.mean()),
        "per_camera_max_abs_error_s": error.max(axis=0).tolist(),
        "num_samples": int(error.size),
    }


__all__ = ["SyncContractError", "validate_sync_contract"]
