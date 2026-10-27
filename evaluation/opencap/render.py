"""Videos for the evaluation report.

* ``render_overlay_grid``: five (or three) camera views of the synchronised
  window with the 2D keypoints drawn on the original, uncropped frames.
* ``render_skeleton_vs_mocap``: the triangulated skeleton (green) and the marker
  mocap (red), rigidly aligned frame by frame, in a frontal and a sagittal view,
  with the per-frame lag-zero error printed on the frame.

Both videos are H.264 (libx264, yuv420p, faststart) so they play in browsers.
The encoder is the ffmpeg binary bundled with ``imageio-ffmpeg``; frames are
piped to it as raw BGR.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from evaluation.opencap.mocap import JOINT_NAMES, interp_rows, kabsch, load_trc, mocap_joints
from evaluation.opencap.scoring import skeleton_mm

KEYPOINT_MIN_SCORE = 0.3
BODY_JOINTS = 23
EDGES_2D: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22),
)
EDGES_3D: tuple[tuple[str, str], ...] = (
    ("L_hip", "R_hip"), ("L_hip", "L_knee"), ("L_knee", "L_ankle"),
    ("R_hip", "R_knee"), ("R_knee", "R_ankle"),
    ("L_hip", "L_sh"), ("R_hip", "R_sh"), ("L_sh", "R_sh"),
    ("L_sh", "L_elb"), ("L_elb", "L_wr"),
    ("R_sh", "R_elb"), ("R_elb", "R_wr"),
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class H264Writer:
    """Pipes raw BGR frames to ffmpeg and encodes them as H.264 (yuv420p, faststart)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p needs even frame dimensions")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        command = [
            get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame of the configured size."""
        if frame.shape[1::-1] != self.size or frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8 of size {self.size}, got {frame.shape}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        """Finish encoding; raise if ffmpeg failed."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} (exit {code}): {stderr.strip()}")

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._proc.kill()
            self._proc.wait()


def _draw_2d(image: np.ndarray, kp: np.ndarray, sc: np.ndarray, colour: tuple[int, int, int] = (60, 220, 60)) -> None:
    for a, b in EDGES_2D:
        if sc[a] >= KEYPOINT_MIN_SCORE and sc[b] >= KEYPOINT_MIN_SCORE:
            cv2.line(
                image,
                tuple(np.round(kp[a]).astype(int)),
                tuple(np.round(kp[b]).astype(int)),
                colour, 2, cv2.LINE_AA,
            )
    for j in range(BODY_JOINTS):
        if sc[j] >= KEYPOINT_MIN_SCORE and np.isfinite(kp[j]).all():
            cv2.circle(image, tuple(np.round(kp[j]).astype(int)), 3, (40, 40, 240), -1, cv2.LINE_AA)


def render_overlay_grid(
    trial_dir: str | Path,
    window_npz: str | Path,
    out_mp4: str | Path,
    fps: float = 60.0,
) -> dict[str, Any]:
    """Grid of the synchronised camera views with the 2D keypoints on the source frames."""
    data = np.load(window_npz, allow_pickle=True)
    kp = np.asarray(data["keypoints"], dtype=float)
    sc = np.asarray(data["scores"], dtype=float)
    indices = np.asarray(data["source_frame_indices"], dtype=np.int64)
    names = [str(n) for n in data["camera_names"]]
    offsets = np.asarray(data["sync_offsets"]).astype(int) if "sync_offsets" in data.files else None
    frames, cams = kp.shape[0], kp.shape[1]

    thumb_w, thumb_h = 288, 512
    grid_w, grid_h = thumb_w * 3, thumb_h * 2
    caps = []
    for name in names:
        cap = cv2.VideoCapture(str(Path(trial_dir) / "videos" / f"{name}.avi"))
        if not cap.isOpened():
            raise OSError(f"cannot open video for {name}")
        caps.append(cap)
    # Skip to the first window frame of each camera; reads are then sequential.
    for c, cap in enumerate(caps):
        for _ in range(int(indices[0, c])):
            cap.grab()

    with H264Writer(out_mp4, grid_w, grid_h, fps) as writer:
        for i in range(frames):
            canvas = np.full((grid_h, grid_w, 3), 24, dtype=np.uint8)
            for c, cap in enumerate(caps):
                ok, frame = cap.read()
                if not ok:
                    raise OSError(f"video ended early for {names[c]} at window frame {i}")
                _draw_2d(frame, kp[i, c], sc[i, c])
                view = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                cv2.putText(view, f"{names[c]}  source f{int(indices[i, c])}", (8, 22), _FONT, 0.55,
                            (0, 255, 255), 1, cv2.LINE_AA)
                row, col = divmod(c, 3)
                canvas[row * thumb_h:(row + 1) * thumb_h, col * thumb_w:(col + 1) * thumb_w] = view
            panel = canvas[thumb_h:, 2 * thumb_w:]
            panel[:] = (30, 30, 30)
            cv2.putText(panel, f"window frame {i}", (12, 34), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "2D keypoints on original", (12, 66), _FONT, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(panel, "720 x 1280 source pixels", (12, 92), _FONT, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
            if offsets is not None:
                for j, name in enumerate(names):
                    cv2.putText(panel, f"{name} offset {int(offsets[j]):+d}", (12, 130 + 24 * j), _FONT, 0.45,
                                (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(canvas)
    for cap in caps:
        cap.release()
    return {"path": str(out_mp4), "frames": frames, "cameras": names, "fps": fps}


def _body_axes(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean anatomical axes of the mocap: up, lateral (right to left), anterior, and the hip origin."""
    idx = {k: i for i, k in enumerate(JOINT_NAMES)}
    mid_hip = 0.5 * (M[:, idx["L_hip"]] + M[:, idx["R_hip"]])
    mid_sh = 0.5 * (M[:, idx["L_sh"]] + M[:, idx["R_sh"]])
    up = (mid_sh - mid_hip).mean(axis=0)
    up /= np.linalg.norm(up)
    lat = (M[:, idx["L_hip"]] - M[:, idx["R_hip"]]).mean(axis=0)
    lat -= up * np.dot(lat, up)
    lat /= np.linalg.norm(lat)
    ant = np.cross(up, lat)
    ant /= np.linalg.norm(ant)
    return up, lat, ant, mid_hip.mean(axis=0)


def render_skeleton_vs_mocap(
    skeleton_npz: str | Path,
    mocap_trc: str | Path,
    out_mp4: str | Path,
    *,
    lag: int = 0,
) -> dict[str, Any]:
    """Skeleton (green) and mocap (red) in two views; every second window frame at 30 fps (real time)."""
    ours = skeleton_mm(skeleton_npz)
    trc = load_trc(mocap_trc)
    mocap = mocap_joints(trc)
    step = float(trc.rate) / 60.0
    up, lat, ant, origin = _body_axes(mocap)
    scale = 0.30
    width, height = 1280, 720
    index = {k: i for i, k in enumerate(JOINT_NAMES)}
    panels = (("front", 320, lat), ("side", 960, ant))
    frames = len(ours)
    with H264Writer(out_mp4, width, height, 30.0) as writer:
        for k in range(frames):
            a = ours[k]
            target = interp_rows(mocap, np.asarray([(k + lag) * step]))[0]
            ok = np.isfinite(a).all(axis=1) & np.isfinite(target).all(axis=1)
            error = float("nan")
            aligned = a
            if ok.sum() >= 6:
                R, t = kabsch(a[ok], target[ok])
                aligned = (R @ a.T).T + t
                error = float(np.sqrt(np.mean(np.linalg.norm(aligned[ok] - target[ok], axis=1) ** 2)))
            canvas = np.full((height, width, 3), 18, dtype=np.uint8)
            for name, cx, axis in panels:
                cv2.putText(canvas, f"{name} view", (cx - 300, 34), _FONT, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
                for points, colour in ((target, (60, 60, 240)), (aligned, (60, 220, 60))):
                    rel = points - origin
                    # Image rows grow downwards, so the vertical body axis is negated.
                    q = np.stack([rel @ axis, -(rel @ up)], axis=1) * scale
                    q[:, 0] += cx
                    q[:, 1] += 330
                    for u, v in EDGES_3D:
                        iu, iv = index[u], index[v]
                        if np.isfinite(q[iu]).all() and np.isfinite(q[iv]).all():
                            cv2.line(canvas, tuple(np.round(q[iu]).astype(int)), tuple(np.round(q[iv]).astype(int)),
                                     colour, 4, cv2.LINE_AA)
                    for j in range(len(q)):
                        if np.isfinite(q[j]).all():
                            cv2.circle(canvas, tuple(np.round(q[j]).astype(int)), 5, colour, -1, cv2.LINE_AA)
            label = f"frame {k}   lag {lag:+d} f   RMS {error:5.1f} mm" if np.isfinite(error) else f"frame {k}"
            cv2.putText(canvas, label, (20, height - 24), _FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "AIMoCap", (1110, 60), _FONT, 0.6, (60, 220, 60), 1, cv2.LINE_AA)
            cv2.putText(canvas, "motion capture", (1110, 86), _FONT, 0.6, (60, 60, 240), 1, cv2.LINE_AA)
            if k % 2 == 0:
                writer.write(canvas)
    return {"path": str(out_mp4), "frames": frames, "lag_frames": lag}
