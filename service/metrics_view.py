"""Turn a run's ``metrics.json`` into rows, cards and overview figures."""

from __future__ import annotations

import math
from typing import Any

SYSTEMS = (
    ("ours", "AIMoCap", None),
    ("opencap_hrnet_5cam", "OpenCap", 5),
    ("opencap_hrnet_3cam", "OpenCap", 3),
)
FPS = 60.0  # frame rate of the window and of the shown videos


def _get(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def fmt(value: Any, digits: int = 1) -> str:
    """Format a number for display; missing values become 'n/a'."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def accuracy_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Joint error and smoothness per system (AIMoCap and the OpenCap baselines)."""
    scores = metrics.get("scores") or {}
    rows = []
    ours_cams = len(_get(metrics, "config", "cams") or [])
    for key, label, cams in SYSTEMS:
        entry = scores.get(key)
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "system": label,
                "cameras": str(ours_cams if cams is None else cams),
                "median": fmt(_get(entry, "lag0_procrustes", "median_mm")),
                "p95": fmt(_get(entry, "lag0_procrustes", "p95_mm")),
                "accel": fmt(_get(entry, "smoothness", "acceleration_error_median_mm_per_frame2"), 2),
                "jitter": fmt(_get(entry, "smoothness", "jitter_slowest20_median_mm_per_frame2"), 2),
                "speed": fmt(_get(entry, "smoothness", "peak_speed_ratio_p95"), 2),
            }
        )
    return rows


def calibration_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    cams = _get(metrics, "calibration", "cameras") or {}
    return [
        {
            "camera": cam,
            "rotation": fmt(info.get("dR_deg"), 2),
            "position": fmt(info.get("centre_diff_mm"), 1),
            "rms": fmt(info.get("reproj_rms_px"), 2),
        }
        for cam, info in sorted(cams.items())
        if isinstance(info, dict)
    ]


def sync_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    cams = _get(metrics, "sync_vs_truth", "per_camera") or {}
    unresolved = set(_get(metrics, "sync_vs_truth", "unresolved_cameras") or [])
    rows = []
    for cam, info in sorted(cams.items()):
        if not isinstance(info, dict):
            continue
        rows.append(
            {
                "camera": cam,
                "estimated": fmt(info.get("estimated"), 0),
                "truth": "unresolved" if cam in unresolved else fmt(info.get("truth"), 0),
                "error": "n/a" if cam in unresolved else fmt(info.get("abs_error_frames"), 0),
            }
        )
    return rows


def headline(metrics: dict[str, Any]) -> dict[str, str]:
    """The few numbers a result card shows, as display strings."""
    return {
        "median": fmt(_get(metrics, "scores", "ours", "lag0_procrustes", "median_mm")),
        "opencap_median": fmt(_get(metrics, "scores", "opencap_hrnet_5cam", "lag0_procrustes", "median_mm")),
        "accel": fmt(_get(metrics, "scores", "ours", "smoothness", "acceleration_error_median_mm_per_frame2"), 2),
        "opencap_accel": fmt(
            _get(metrics, "scores", "opencap_hrnet_5cam", "smoothness", "acceleration_error_median_mm_per_frame2"), 2
        ),
        "calib_deg": fmt(_get(metrics, "calibration", "max_dR_deg"), 2),
        "calib_mm": fmt(_get(metrics, "calibration", "max_centre_diff_mm"), 1),
        "sync_frames": fmt(_get(metrics, "sync_vs_truth", "max_abs_error_frames"), 0),
        "frames": fmt(_get(metrics, "window", "frames"), 0),
    }


def session(metrics: dict[str, Any]) -> dict[str, str] | None:
    """Phones, frame rate and window length of one result, or None if the metrics lack them."""
    frames = _get(metrics, "window", "frames")
    cams = _get(metrics, "config", "cams")
    if not frames or not cams:
        return None
    fps = float(_get(metrics, "config", "fps") or FPS)
    return {
        "cameras": str(len(cams)),
        "fps": fmt(fps, 0),
        "frames": str(int(frames)),
        "seconds": fmt(float(frames) / fps, 1),
    }


def _values(metrics_list: list[dict[str, Any]], *keys: str) -> list[float]:
    found = []
    for metrics in metrics_list:
        value = _get(metrics, *keys)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            found.append(float(value))
    return found


def _span(values: list[float], digits: int) -> str | None:
    """Range of values across trials, such as ``41-51``; a single value when they agree."""
    if not values:
        return None
    low, high = fmt(min(values), digits), fmt(max(values), digits)
    return low if low == high else f"{low}\u2013{high}"


def overview(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Ranges and maxima across the processed trials, for the overview page."""
    calib_deg = _values(metrics_list, "calibration", "max_dR_deg")
    calib_mm = _values(metrics_list, "calibration", "max_centre_diff_mm")
    def is_walking(metrics: dict[str, Any]) -> bool:
        trial = metrics.get("trial")
        return isinstance(trial, str) and trial.lower().startswith("walking")

    nonwalking_metrics = [metrics for metrics in metrics_list if not is_walking(metrics)]
    sync = _values(nonwalking_metrics, "sync_vs_truth", "max_abs_error_frames")
    acceptance = next(
        (_get(m, "calibration", "acceptance") for m in metrics_list if _get(m, "calibration", "acceptance")), None
    ) or {}
    unresolved = [cam for m in metrics_list for cam in (_get(m, "sync_vs_truth", "unresolved_cameras") or [])]
    return {
        "trials": len(metrics_list),
        "median": _span(_values(metrics_list, "scores", "ours", "lag0_procrustes", "median_mm"), 0),
        "opencap_median": _span(
            _values(metrics_list, "scores", "opencap_hrnet_5cam", "lag0_procrustes", "median_mm"), 0
        ),
        "accel": _span(
            _values(metrics_list, "scores", "ours", "smoothness", "acceleration_error_median_mm_per_frame2"), 2
        ),
        "opencap_accel": _span(
            _values(metrics_list, "scores", "opencap_hrnet_5cam", "smoothness", "acceleration_error_median_mm_per_frame2"),
            2,
        ),
        "calib_deg": fmt(max(calib_deg), 2) if calib_deg else None,
        "calib_mm": fmt(max(calib_mm), 1) if calib_mm else None,
        "calib_limit_deg": fmt(acceptance.get("max_dR_deg"), 2) if acceptance.get("max_dR_deg") is not None else None,
        "calib_limit_mm": fmt(acceptance.get("max_centre_diff_mm"), 0)
        if acceptance.get("max_centre_diff_mm") is not None
        else None,
        "sync_frames": fmt(max(sync), 0) if sync else None,
        "sync_ms": fmt(max(sync) * 1000.0 / FPS, 0) if sync else None,
        "walking_trials": sum(1 for metrics in metrics_list if is_walking(metrics)),
        "unresolved": len(unresolved),
        "unresolved_trials": sum(1 for m in metrics_list if _get(m, "sync_vs_truth", "unresolved_cameras")),
    }


def stage_rows(metrics: dict[str, Any]) -> list[dict[str, str]]:
    stages = metrics.get("stage_seconds") or {}
    return [{"stage": name, "seconds": fmt(sec, 1)} for name, sec in stages.items()]
