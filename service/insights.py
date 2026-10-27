"""Movement analysis shown on result pages: knee flexion, left/right symmetry, charts.

Knee flexion is 180 degrees minus the hip-knee-ankle angle, so a straight leg reads 0.
It is computed for the AIMoCap skeleton and for the marker motion capture of the same
trial, so a coach or clinician can see both the movement and how closely the system
tracks it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.opencap.mocap import JOINT_NAMES, interp_rows, load_trc, mocap_joints
from evaluation.opencap.scoring import skeleton_mm

IDX = {name: i for i, name in enumerate(JOINT_NAMES)}
VIDEO_FPS = 60.0

# Chart geometry in SVG user units. The page CSS enlarges the tick text on phones.
CHART_W, CHART_H = 640, 300
PAD_L, PAD_R, PAD_T, PAD_B = 64, 16, 16, 40
DEG_STEP = 30.0
TIME_STEP_S = 2
MAX_POINTS = 600


def knee_flexion(points: np.ndarray, side: str) -> np.ndarray:
    """Knee flexion in degrees for (frames, 12, 3) joints; NaN where a joint is missing."""
    hip = points[:, IDX[f"{side}_hip"]]
    knee = points[:, IDX[f"{side}_knee"]]
    ankle = points[:, IDX[f"{side}_ankle"]]
    thigh = hip - knee
    shank = ankle - knee
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.sum(thigh * shank, axis=1) / (np.linalg.norm(thigh, axis=1) * np.linalg.norm(shank, axis=1))
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
    return 180.0 - np.degrees(angle)


@dataclass(frozen=True)
class KneeAnalysis:
    seconds: np.ndarray
    ours: dict[str, np.ndarray]
    mocap: dict[str, np.ndarray] | None


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _peak(values: np.ndarray) -> float:
    """98th percentile of the flexion: the peak, without one bad frame setting it."""
    finite = _finite(values)
    return float(np.percentile(finite, 98)) if finite.size else float("nan")


def _mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(finite.mean()) if finite.size else float("nan")


def _text(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


@functools.lru_cache(maxsize=16)
def _analyse(skeleton_path: str, skeleton_mtime: float, mocap_path: str | None) -> KneeAnalysis | None:
    try:
        ours = skeleton_mm(skeleton_path)
    except (OSError, KeyError, ValueError):
        return None
    seconds = np.arange(len(ours)) / VIDEO_FPS
    ours_angles = {side: knee_flexion(ours, side) for side in ("L", "R")}
    mocap_angles = None
    if mocap_path and Path(mocap_path).is_file():
        trc = load_trc(mocap_path)
        joints = mocap_joints(trc)
        # Window frame i is at mocap time i / 60 s, the same alignment as the accuracy scoring.
        rows = seconds * float(trc.rate)
        rows = rows[rows <= len(joints) - 1]
        resampled = interp_rows(joints, rows)
        mocap_angles = {side: knee_flexion(resampled, side) for side in ("L", "R")}
    return KneeAnalysis(seconds=seconds, ours=ours_angles, mocap=mocap_angles)


def analyse(skeleton: Path, mocap: Path | None) -> KneeAnalysis | None:
    """Knee analysis of a 3D skeleton file, compared with marker mocap when a file is given."""
    if not skeleton.is_file():
        return None
    return _analyse(str(skeleton), skeleton.stat().st_mtime, str(mocap) if mocap else None)


def summary_rows(result: KneeAnalysis) -> list[dict[str, str]]:
    """Peak flexion per side, mean asymmetry and peak gap for AIMoCap and motion capture."""

    def row(label: str, angles: dict[str, np.ndarray]) -> dict[str, str]:
        n = min(len(angles["L"]), len(angles["R"]))
        diff = np.abs(angles["L"][:n] - angles["R"][:n])
        return {
            "source": label,
            "left": _text(_peak(angles["L"])),
            "right": _text(_peak(angles["R"])),
            "asymmetry": _text(_mean(diff), 1),
            "peak_difference": _text(abs(_peak(angles["L"]) - _peak(angles["R"])), 1),
        }

    rows = [row("AIMoCap (phones)", result.ours)]
    if result.mocap is not None:
        rows.append(row("Marker motion capture (lab)", result.mocap))
    return rows


def tracking_error_deg(result: KneeAnalysis) -> str | None:
    """Median absolute knee-angle difference between AIMoCap and motion capture, both knees."""
    if result.mocap is None:
        return None
    errors = []
    for side in ("L", "R"):
        n = min(len(result.ours[side]), len(result.mocap[side]))
        errors.append(np.abs(result.ours[side][:n] - result.mocap[side][:n]))
    values = _finite(np.concatenate(errors))
    return f"{np.median(values):.1f}" if values.size else None


def chart_svg(result: KneeAnalysis) -> str:
    """Inline SVG line chart of knee flexion over time (no scripts, CSP-safe).

    Solid lines are AIMoCap, dashed lines are marker motion capture (when present).
    """
    plot_w = CHART_W - PAD_L - PAD_R
    plot_h = CHART_H - PAD_T - PAD_B
    series = [("ours-l", result.ours["L"]), ("ours-r", result.ours["R"])]
    if result.mocap is not None:
        series += [("mocap-l", result.mocap["L"]), ("mocap-r", result.mocap["R"])]
    peak = max((float(_finite(y).max()) for _, y in series if _finite(y).size), default=0.0)
    top = max(DEG_STEP, float(np.ceil(peak / DEG_STEP) * DEG_STEP))
    duration = float(result.seconds[-1]) if len(result.seconds) > 1 else 1.0
    base = PAD_T + plot_h

    def x_at(seconds: float) -> float:
        return PAD_L + plot_w * (seconds / duration)

    def y_at(degrees: float) -> float:
        return PAD_T + plot_h * (1.0 - degrees / top)

    def path(y: np.ndarray) -> str:
        step = max(1, len(y) // MAX_POINTS)
        parts: list[str] = []
        pen = "M"
        for i in range(0, len(y), step):
            if not np.isfinite(y[i]):
                pen = "M"  # a gap in the data starts a new sub-path
                continue
            parts.append(f"{pen}{x_at(result.seconds[i]):.1f},{y_at(y[i]):.1f}")
            pen = "L"
        return " ".join(parts)

    out = [
        f'<svg class="chart" viewBox="0 0 {CHART_W} {CHART_H}" role="img" '
        'aria-label="Knee flexion over time for the left and right knee, AIMoCap and marker motion capture">'
    ]
    for deg in np.arange(0.0, top + 0.5, DEG_STEP):
        y = y_at(deg)
        out.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{CHART_W - PAD_R}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 10}" y="{y + 5:.1f}" text-anchor="end">{deg:.0f}&#176;</text>')
    out.append(f'<line class="axis" x1="{PAD_L}" y1="{base:.1f}" x2="{CHART_W - PAD_R}" y2="{base:.1f}"/>')
    for sec in range(0, int(duration) + 1, TIME_STEP_S):
        minor = " minor" if (sec // TIME_STEP_S) % 2 else ""
        out.append(f'<text class="tick{minor}" x="{x_at(sec):.1f}" y="{base + 26:.1f}" text-anchor="middle">{sec} s</text>')
    for css, y in series:
        out.append(f'<path class="line {css}" d="{path(y)}"/>')
    out.append("</svg>")
    return "".join(out)
