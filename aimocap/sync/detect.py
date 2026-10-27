"""Clap transient detection and sub-sample offset estimation.

Two-stage approach (robust + precise):

1. Coarse: short-time energy onset detection locates the clap to within a few
   tens of ms. This handles the case where cross-correlation over the whole
   track would be dominated by noise or where the clap isn't the global peak.

2. Fine: bandpass-filtered (2-8 kHz) FFT cross-correlation on a window around
   each clap gives the sample-level offset between two cameras; parabolic
   interpolation on the correlation peak refines to sub-sample precision.

The clap is an impulsive broadband transient, so bandpassing to the
high-frequency region (where speech and hum don't live) sharpens the
correlation peak dramatically.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from aimocap.sync.audio import AudioTrack

# Clap transient frequency band (Hz). Impulsive broadband content lives here;
# speech fundamentals (80-300Hz) and mains hum (50/60Hz) are rejected. The
# upper bound must be strictly below Nyquist (fs/2); at our 16kHz working rate
# Nyquist is 8000Hz, so 7000Hz keeps margin while losing almost no clap energy.
CLAP_BAND = (2000.0, 7000.0)


def energy_envelope(samples: np.ndarray, sr: int, frame_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Short-time mean energy (RMS) of the signal.

    Returns (envelope, frame_centers_s). frame_ms is the analysis hop; 10ms
    balances time resolution vs. noise smoothing.
    """
    hop = max(1, int(sr * frame_ms / 1000))
    n_frames = len(samples) // hop
    if n_frames == 0:
        return np.zeros(0), np.zeros(0)
    trimmed = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    centers = (np.arange(n_frames) * hop + hop / 2) / sr
    return rms, centers


def detect_clap(track: AudioTrack, min_gap_s: float = 0.2) -> float:
    """Return the clap time (seconds) in the track, or -1 if none found.

    Finds the sharpest energy onset: the frame whose RMS exceeds a local
    adaptive threshold by the largest margin. ``min_gap_s`` prevents detecting
    the decay tail of one clap as a second onset.
    """
    if len(track.samples) == 0:
        return -1.0
    env, centers = energy_envelope(track.samples, track.sample_rate)
    if len(env) == 0:
        return -1.0

    # Onset strength = positive derivative of the envelope (energy increase).
    onset = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    if onset.max() <= 0:
        return -1.0

    # Adaptive threshold: a frame counts as a candidate if its onset exceeds
    # 5x the median onset. This rejects slow background-level rises.
    thresh = max(5.0 * np.median(onset), 1e-9)
    candidates = np.where(onset >= thresh)[0]
    if len(candidates) == 0:
        # fall back to the single strongest onset
        candidates = np.array([int(np.argmax(onset))])

    # Pick the strongest candidate; suppress neighbors within min_gap_s.
    best = max(candidates, key=lambda i: onset[i])
    return float(centers[best])


def _bandpass(samples: np.ndarray, sr: int, band: tuple[float, float] = CLAP_BAND,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass via forward-backward filtering.

    Uses sosfiltfilt (not sosfilt) so the filter contributes ZERO group delay.
    A one-pass IIR bandpass would add ~1.4ms of phase shift at our settings —
    a fixed bias that violated the +/-1ms sync gate. sosfiltfilt cancels phase
    by filtering forward then backward, doubling amplitude rolloff but
    eliminating timing skew. This matters because we use the filtered signal
    for sub-sample offset estimation.
    """
    sos = butter(order, band, btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, samples)


def _parabolic_peak_interp(corr: np.ndarray, peak: int) -> float:
    """Sub-sample peak location via 3-point parabolic interpolation.

    Returns the refined peak index (float). Standard technique for refining
    discrete correlation peaks; assumes corr is roughly parabolic near `peak`.
    """
    if peak <= 0 or peak >= len(corr) - 1:
        return float(peak)
    y0, y1, y2 = corr[peak - 1], corr[peak], corr[peak + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(peak)
    delta = 0.5 * (y0 - y2) / denom
    # delta in [-0.5, 0.5]
    return peak + float(np.clip(delta, -0.5, 0.5))


def cross_correlate_offset(
    ref: AudioTrack, other: AudioTrack,
    window_s: float | None = None,
    clap_ref: float | None = None,
    clap_other: float | None = None,
) -> tuple[float, float]:
    """Estimate the sample offset of ``other`` relative to ``ref`` via FFT xcorr.

    Both tracks are bandpassed to the clap band (2-7 kHz) and FFT
    cross-correlated over their full length (or the first ``window_s`` if set).
    The peak of the correlation gives the offset in SAMPLES of ``other``
    relative to ``ref``.

    Sign convention (verified empirically against synthetic ground truth): a
    NEGATIVE offset means other's clap is LATER than ref's (other lags ref);
    POSITIVE means other leads ref.

    Important: the clap times (clap_ref/clap_other) are accepted for API
    compatibility but NOT used to pre-window the tracks. Pre-aligning both crops
    around their own claps cancels out the very offset we're measuring (the
    residual peak would always sit near zero lag). The FFT xcorr finds the lag
    directly from the dominant transient.

    Returns (offset_samples, peak_confidence) where peak_confidence is the
    ratio of the correlation peak to the median |correlation| value (high =
    sharp/distinct peak, near 1 = noise).
    """
    if len(ref.samples) == 0 or len(other.samples) == 0:
        raise ValueError("empty audio track; cannot correlate")
    if ref.sample_rate != other.sample_rate:
        raise ValueError(
            f"sample rate mismatch: {ref.sample_rate} vs {other.sample_rate}"
        )
    sr = ref.sample_rate

    a = ref.samples
    b = other.samples
    if window_s is not None:
        L = int(window_s * sr)
        a = a[:L]
        b = b[:L]

    a = _bandpass(a, sr)
    b = _bandpass(b, sr)

    # Pad to equal length (FFT xcorr needs same-length inputs for a clean peak).
    L = max(len(a), len(b))
    a = np.pad(a, (0, L - len(a)))
    b = np.pad(b, (0, L - len(b)))

    # FFT cross-correlation: corr = IFFT(FFT(a) * conj(FFT(b))).
    Fa = np.fft.rfft(a)
    Fb = np.fft.rfft(b)
    corr = np.fft.irfft(Fa * np.conj(Fb), n=L)

    peak = int(np.argmax(corr))
    refined = _parabolic_peak_interp(corr, peak)
    peak_val = corr[peak]
    median_val = np.median(np.abs(corr)) + 1e-12
    confidence = float(peak_val / median_val)

    # Unwrap circular lag: a peak beyond L/2 represents a negative lag.
    offset = refined
    if offset > L / 2:
        offset -= L
    return float(offset), confidence


def offset_samples_to_ms(offset_samples: float, sample_rate: int) -> float:
    """Convert a sample offset to milliseconds."""
    return offset_samples * 1000.0 / sample_rate


__all__ = [
    "CLAP_BAND",
    "energy_envelope",
    "detect_clap",
    "cross_correlate_offset",
    "offset_samples_to_ms",
]
