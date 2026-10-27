"""Foot/leg orientation diagnostics from triangulated COCO-WholeBody points.

Usage:
    python -m aimocap.diagnostics.leg_orientation outputs/test_bone_constrained.npz -o outputs/diagnostics/legs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aimocap.pose.keypoints import (
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_KNEE,
    LEFT_SMALL_TOE,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_KNEE,
    RIGHT_SMALL_TOE,
)


FOOT_SIDES = {
    "left": {
        "ankle": LEFT_ANKLE,
        "knee": LEFT_KNEE,
        "big": LEFT_BIG_TOE,
        "small": LEFT_SMALL_TOE,
        "heel": LEFT_HEEL,
    },
    "right": {
        "ankle": RIGHT_ANKLE,
        "knee": RIGHT_KNEE,
        "big": RIGHT_BIG_TOE,
        "small": RIGHT_SMALL_TOE,
        "heel": RIGHT_HEEL,
    },
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9 or not np.isfinite(n):
        return np.full_like(v, np.nan, dtype=np.float64)
    return v / n


def foot_frame(points: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    spec = FOOT_SIDES[side]
    need = [spec["ankle"], spec["big"], spec["small"], spec["heel"]]
    if not np.isfinite(points[need]).all():
        return None
    big = points[spec["big"]]
    small = points[spec["small"]]
    heel = points[spec["heel"]]
    origin = points[spec["ankle"]]
    toe_center = 0.5 * (big + small)
    forward = _unit(toe_center - heel)
    lateral = _unit(big - small)
    up = _unit(np.cross(lateral, forward))
    if not np.isfinite(np.r_[forward, lateral, up]).all():
        return None
    return origin, forward, lateral, up


def summarize_leg_orientation(npz_path: str | Path) -> dict:
    data = np.load(npz_path)
    pts = data["skeleton3d"]
    out: dict[str, object] = {"path": str(npz_path), "num_frames": int(pts.shape[0])}
    for side in FOOT_SIDES:
        valid = []
        yaw = []
        for f in range(pts.shape[0]):
            fr = foot_frame(pts[f], side)
            valid.append(fr is not None)
            if fr is not None:
                _, forward, _, _ = fr
                # Internal space is X/Z ground plane with Y up.
                yaw.append(float(np.degrees(np.arctan2(forward[0], forward[2]))))
        yaw_arr = np.asarray(yaw, dtype=np.float64)
        out[f"{side}_valid_foot_frame_fraction"] = float(np.mean(valid))
        out[f"{side}_foot_yaw_median_deg"] = float(np.nanmedian(yaw_arr)) if yaw_arr.size else None
        out[f"{side}_foot_yaw_p95_abs_delta_deg"] = (
            float(np.nanpercentile(np.abs(np.diff(yaw_arr)), 95)) if yaw_arr.size > 1 else None
        )
    return out


def render_leg_axes(npz_path: str | Path, out_png: str | Path, frames: list[int] | None = None) -> None:
    data = np.load(npz_path)
    pts = data["skeleton3d"]
    F = pts.shape[0]
    if frames is None:
        frames = np.linspace(0, max(0, F - 1), num=min(12, F), dtype=int).tolist()
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
    for i, f in enumerate(frames):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        p = pts[int(f)]
        for side, color in (("left", "#377eb8"), ("right", "#e41a1c")):
            spec = FOOT_SIDES[side]
            chain = [spec["knee"], spec["ankle"], spec["heel"], spec["big"], spec["small"], spec["heel"]]
            for a, b in zip(chain[:-1], chain[1:]):
                if np.isfinite(p[[a, b]]).all():
                    ax.plot([p[a, 0], p[b, 0]], [p[a, 2], p[b, 2]], [p[a, 1], p[b, 1]], c=color, lw=2)
            fr = foot_frame(p, side)
            if fr is not None:
                origin, forward, lateral, up = fr
                for vec, c, label in ((forward, "green", "fwd"), (lateral, "orange", "lat"), (up, "purple", "up")):
                    ax.quiver(origin[0], origin[2], origin[1], vec[0], vec[2], vec[1], length=0.18, color=c)
                ax.text(origin[0], origin[2], origin[1], side[0].upper(), color=color)
        finite = p[:23][np.isfinite(p[:23]).all(axis=-1)]
        if finite.size:
            mins = finite.min(axis=0)
            maxs = finite.max(axis=0)
            ctr = (mins + maxs) / 2.0
            rad = max(float(np.max(maxs - mins) / 2.0), 0.5)
            ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
            ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
            ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_title(f"frame {int(f)}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Triangulated skeleton NPZ containing skeleton3d.")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--frames", default=None, help="Comma-separated frame indices.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")] if args.frames else None
    summary = summarize_leg_orientation(args.npz)
    (out_dir / "leg_orientation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_leg_axes(args.npz, out_dir / "leg_orientation_axes.png", frames=frames)
    print(json.dumps(summary, indent=2))
    print(f"Wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
