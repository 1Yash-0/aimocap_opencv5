"""2D pose confidence/outlier diagnostics for multi-camera NPZ files.

Usage:
    python -m aimocap.diagnostics.pose2d outputs/panoptic_multipose.npz -o outputs/diagnostics/pose2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aimocap.pose.keypoints import KEYPOINT_NAMES_133


def summarize_pose2d(npz_path: str | Path, jump_px: float = 120.0) -> dict:
    data = np.load(npz_path)
    keypoints = data["keypoints"]
    scores = data["scores"]
    valid = np.isfinite(keypoints).all(axis=-1) & np.isfinite(scores)
    jumps = np.linalg.norm(np.diff(keypoints, axis=0), axis=-1)
    jump_mask = np.isfinite(jumps) & (jumps > jump_px)
    return {
        "path": str(npz_path),
        "num_frames": int(keypoints.shape[0]),
        "num_cameras": int(keypoints.shape[1]),
        "num_keypoints": int(keypoints.shape[2]),
        "valid_observation_fraction": float(np.mean(valid)),
        "score_median": float(np.nanmedian(scores)),
        "score_p10": float(np.nanpercentile(scores, 10)),
        "large_jump_px": float(jump_px),
        "large_jump_fraction": float(np.mean(jump_mask)),
        "large_jump_count": int(np.count_nonzero(jump_mask)),
    }


def render_score_heatmap(npz_path: str | Path, out_png: str | Path) -> None:
    data = np.load(npz_path)
    scores = data["scores"]
    per_cam_joint = np.nanmedian(scores, axis=0)
    fig, ax = plt.subplots(figsize=(18, max(3, 1.2 * scores.shape[1])))
    im = ax.imshow(per_cam_joint, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Median 2D keypoint confidence by camera/keypoint")
    ax.set_xlabel("keypoint index")
    ax.set_ylabel("camera")
    ax.set_yticks(np.arange(scores.shape[1]))
    if "camera_names" in data:
        ax.set_yticklabels([str(x) for x in data["camera_names"]])
    for idx in range(min(23, len(KEYPOINT_NAMES_133))):
        ax.text(idx, -0.55, KEYPOINT_NAMES_133[idx], rotation=90, fontsize=6, va="bottom")
    fig.colorbar(im, ax=ax, label="confidence")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_jump_heatmap(npz_path: str | Path, out_png: str | Path, jump_px: float = 120.0) -> None:
    data = np.load(npz_path)
    keypoints = data["keypoints"]
    jumps = np.linalg.norm(np.diff(keypoints, axis=0), axis=-1)
    per_cam_joint = np.nanpercentile(jumps, 95, axis=0)
    fig, ax = plt.subplots(figsize=(18, max(3, 1.2 * keypoints.shape[1])))
    im = ax.imshow(per_cam_joint, aspect="auto", cmap="magma")
    ax.set_title(f"p95 frame-to-frame 2D jump by camera/keypoint (red flag > {jump_px:.0f}px)")
    ax.set_xlabel("keypoint index")
    ax.set_ylabel("camera")
    ax.set_yticks(np.arange(keypoints.shape[1]))
    if "camera_names" in data:
        ax.set_yticklabels([str(x) for x in data["camera_names"]])
    fig.colorbar(im, ax=ax, label="px/frame")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="multi-camera 2D pose NPZ.")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--jump-px", type=float, default=120.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_pose2d(args.npz, jump_px=args.jump_px)
    (out_dir / "pose2d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_score_heatmap(args.npz, out_dir / "pose2d_confidence_heatmap.png")
    render_jump_heatmap(args.npz, out_dir / "pose2d_jump_heatmap.png", jump_px=args.jump_px)
    print(json.dumps(summary, indent=2))
    print(f"Wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
