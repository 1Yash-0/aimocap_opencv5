"""Visual and numeric diagnostics for triangulated 3D NPZ files.

Usage:
    python -m aimocap.diagnostics.triangulation outputs/test_bone_constrained.npz -o outputs/diagnostics/triangulation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aimocap.pose.keypoints import KEYPOINT_NAMES_133, SKELETON_133


def _finite_percentile(x: np.ndarray, q: float) -> float | None:
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q))


def summarize_npz(npz_path: str | Path) -> dict:
    data = np.load(npz_path)
    pts = data["skeleton3d"]
    finite_joint = np.isfinite(pts).all(axis=-1)
    summary: dict[str, object] = {
        "path": str(npz_path),
        "num_frames": int(pts.shape[0]),
        "num_keypoints": int(pts.shape[1]),
        "finite_point_fraction": float(np.mean(finite_joint)),
    }
    if "confidence" in data:
        conf = data["confidence"]
        summary["confidence_median"] = _finite_percentile(conf, 50)
        summary["confidence_p10"] = _finite_percentile(conf, 10)
    if "triangulation_reprojection_error_px" in data:
        err = data["triangulation_reprojection_error_px"]
        summary["reprojection_error_px_median"] = _finite_percentile(err, 50)
        summary["reprojection_error_px_p90"] = _finite_percentile(err, 90)
        summary["reprojection_error_px_p95"] = _finite_percentile(err, 95)
    if "triangulation_num_inliers" in data:
        n = data["triangulation_num_inliers"]
        summary["mean_inliers_when_present"] = float(np.mean(n[n > 0])) if np.any(n > 0) else 0.0
        summary["fraction_points_with_2plus_inliers"] = float(np.mean(n >= 2))
    return summary


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    finite = pts[np.isfinite(pts).all(axis=-1)]
    if finite.size == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)


def render_contact_sheet(npz_path: str | Path, out_png: str | Path, frames: list[int] | None = None) -> None:
    data = np.load(npz_path)
    pts = data["skeleton3d"]
    F = pts.shape[0]
    if frames is None:
        frames = np.linspace(0, max(0, F - 1), num=min(12, F), dtype=int).tolist()
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    all_body = pts[:, :23, :]
    for i, f in enumerate(frames):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        p = pts[int(f)]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=4, c="black", alpha=0.45)
        for a, b in SKELETON_133:
            if a >= p.shape[0] or b >= p.shape[0]:
                continue
            if np.isfinite(p[[a, b]]).all():
                ax.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], [p[a, 1], p[b, 1]], c="#377eb8", lw=1)
        ax.set_title(f"frame {int(f)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        _set_axes_equal(ax, all_body)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_reprojection_heatmap(npz_path: str | Path, out_png: str | Path) -> bool:
    data = np.load(npz_path)
    if "triangulation_reprojection_error_px" not in data:
        return False
    err = data["triangulation_reprojection_error_px"]
    per_joint = np.nanmedian(err, axis=(0, 2))
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(np.arange(len(per_joint)), per_joint)
    ax.set_title("Median triangulation reprojection error by keypoint")
    ax.set_xlabel("keypoint index")
    ax.set_ylabel("px")
    for idx in range(min(23, len(KEYPOINT_NAMES_133))):
        ax.text(idx, per_joint[idx] if np.isfinite(per_joint[idx]) else 0.0, KEYPOINT_NAMES_133[idx], rotation=90, fontsize=6)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Triangulated skeleton NPZ containing skeleton3d.")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--frames", default=None, help="Comma-separated frame indices for the contact sheet.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")] if args.frames else None
    summary = summarize_npz(args.npz)
    (out_dir / "triangulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_contact_sheet(args.npz, out_dir / "triangulation_contact_sheet.png", frames=frames)
    render_reprojection_heatmap(args.npz, out_dir / "triangulation_reprojection_by_joint.png")
    print(json.dumps(summary, indent=2))
    print(f"Wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
