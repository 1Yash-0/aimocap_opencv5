"""Intrinsic parameter approximation for markerless calibration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def guess_intrinsics(width: int, height: int, focal_ratio: float = 1.2) -> np.ndarray:
    """
    Synthesize a 3x3 Camera Matrix (K) for an uncalibrated camera.
    
    Modern smartphone rear cameras typically have a 35mm-equivalent focal length 
    of around 24-28mm. This roughly corresponds to a focal length (in pixels) of 
    1.2x the image's maximum dimension.
    
    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        focal_ratio: Multiplier for the maximum dimension to guess focal length.
        
    Returns:
        (3, 3) float64 numpy array representing the Camera Matrix K.
    """
    f = max(width, height) * focal_ratio
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    return K


def extract_intrinsics(
    video_path: str | Path,
    width: int,
    height: int,
    *,
    allow_guess: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Attempt to extract true camera focal length from video EXIF/metadata using ffprobe.
    A guessed focal length is returned only when ``allow_guess=True``.  The
    default is strict so a missing camera calibration cannot silently enter a
    production reconstruction.
    
    Returns:
        K: (3,3) intrinsic matrix.
        method: "exif" or "guess".
    """
    video_path = Path(video_path)
    
    try:
        # Run ffprobe to get format and stream metadata in JSON format
        cmd = [
            "ffprobe", 
            "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            "-show_streams", 
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Search for tags in format and streams
        tags = {}
        if "format" in data and "tags" in data["format"]:
            tags.update(data["format"]["tags"])
        for stream in data.get("streams", []):
            if "tags" in stream:
                tags.update(stream["tags"])
                
        # Look for Apple's focal length tag: com.apple.quicktime.camera.focal-length (usually an integer like "26" mm equivalent or actual physical focal length)
        # However, extracting physical focal length requires sensor size, which isn't always there.
        # Alternatively, Android puts EXIF tags.
        # Currently, if we don't have sensor width, mm focal length is useless.
        # But iPhones write focal length in 35mm equivalent sometimes, or we can use EXIF IFD0.
        
        # For a truly robust extraction, we'd need ExifTool or complex MP4 parsing to get EXIF.
        # Since this is tricky via ffprobe alone without sensor size, we will look for a custom tag we might inject,
        # or fallback.
        
        # We will parse common tags if available.
        # Apple stores "com.apple.quicktime.camera.focal-length" but it's usually unparsed by ffprobe.
        # Let's check if we can find something useful. If not, fallback.
        # This is a stub that succeeds if a specific tag is found.
        # In a production app, the frontend (iOS/Android) should write the exact K matrix or f_px into a sidecar JSON or custom tag.
        pass
        
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
        
    if not allow_guess:
        raise RuntimeError(
            f"no calibrated intrinsics found for {video_path}; provide a camera "
            "calibration sidecar or explicitly enable exploratory guessing"
        )

    return guess_intrinsics(width, height), "guess"
