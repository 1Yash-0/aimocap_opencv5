"""Encode a PNG frame sequence to a browser-friendly H.264 MP4.

Usage::

    python scripts/encode_frames.py <frames_dir> <out.mp4> [--fps 30]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    files = sorted(Path(args.frames).glob("frame_*.png"))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")
    listing = Path(args.frames) / "frames.txt"
    listing.write_text("".join(f"file '{f.name}'\nduration {1 / args.fps:.6f}\n" for f in files), encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    "-r", f"{args.fps:g}", args.out], check=True)
    print(f"wrote {args.out} ({len(files)} frames)")


if __name__ == "__main__":
    main()
