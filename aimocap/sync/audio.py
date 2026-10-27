"""Audio extraction: video file → mono PCM waveform at a fixed sample rate.

Uses the ffmpeg binary bundled by imageio-ffmpeg (no system ffmpeg needed).
Runs ffmpeg as a subprocess to decode the audio track and pipe raw PCM to
stdout, which we read into a numpy array. This avoids per-container Python
parser dependencies and handles anything ffmpeg can decode (mp4/mov/webm/...).

16 kHz mono is the working rate for the sync stage: enough time resolution for
sub-millisecond alignment (1 sample = 62.5us), small enough that FFT
cross-correlation of multi-second tracks is fast.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    # imageio-ffmpeg not installed; callers will get a clear error on use.
    _FFMPEG = None


# Default working sample rate for the sync stage (Hz).
DEFAULT_SAMPLE_RATE = 16000


@dataclass(slots=True)
class AudioTrack:
    """A mono PCM audio track extracted from a video.

    samples : (N,) float32 in [-1, 1], mono.
    sample_rate : Hz.
    duration_s : N / sample_rate.
    source : path to the video the track came from.
    """

    samples: np.ndarray
    sample_rate: int
    source: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def time_axis(self) -> np.ndarray:
        return np.arange(len(self.samples)) / self.sample_rate


def _ensure_ffmpeg() -> str:
    if _FFMPEG is None:
        raise RuntimeError(
            "imageio-ffmpeg is required for audio extraction. "
            "Install it: pip install imageio-ffmpeg"
        )
    return _FFMPEG


def extract_audio(
    video_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> AudioTrack:
    """Extract the audio track from a video as mono PCM float32.

    Parameters
    ----------
    sample_rate : resample to this rate (Hz).
    start_s, duration_s : optional window. Useful for large files where only a
        region around the clap is needed.

    Raises FileNotFoundError if the video is missing, and RuntimeError with the
    ffmpeg stderr if decoding fails (so the caller sees the real reason).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    ffmpeg = _ensure_ffmpeg()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", str(int(sample_rate)),
        "-f", "f32le",               # raw 32-bit float little-endian PCM
    ]
    if start_s is not None:
        cmd[4:4] = ["-ss", str(start_s)]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += ["pipe:1"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"ffmpeg failed (code {proc.returncode}) on {video_path}:\n{err}"
        )

    raw = proc.stdout
    if not raw:
        # No audio track in the container. Return an empty track rather than
        # crash — the sync engine will treat this camera as unusable.
        return AudioTrack(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=int(sample_rate),
            source=str(video_path),
        )

    samples = np.frombuffer(raw, dtype=np.float32).copy()
    # Guard against pathological DC spikes from integer-overflow on decode.
    if samples.size and not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return AudioTrack(
        samples=samples,
        sample_rate=int(sample_rate),
        source=str(video_path),
    )


__all__ = ["AudioTrack", "extract_audio", "DEFAULT_SAMPLE_RATE"]
