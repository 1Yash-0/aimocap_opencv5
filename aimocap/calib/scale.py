"""Anthropometric scaling for metric recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


class ScaleContractError(ValueError):
    """Raised when calibration scale metadata is missing, ambiguous, or violated."""
    pass


@dataclass(frozen=True)
class ScalePolicy:
    action: str          # "PRESERVE_METRIC" | "ESTIMATE_ANTHROPOMETRIC"
    scale_source: str    # "checkerboard", "charuco", "panoptic_studio", "anthropometric", etc.
    is_metric: bool      # True if metric ground truth preserved
    scale_factor: float  # 1.0 when preserved, or 0.0 placeholder before estimation
    reason: str          # Explanatory diagnosis


def resolve_scale_policy(
    metadata: Dict[str, Any],
    *,
    allow_exploratory: bool = False,
) -> ScalePolicy:
    """Resolve whether metric scale is preserved from calibration or estimated.

    Parameters
    ----------
    metadata : dict
        Calibration metadata mapping (from CalibrationBundle or calibration JSON).
    allow_exploratory : bool
        If True, permits anthropometric scaling when calibration is explicitly exploratory.

    Returns
    -------
    ScalePolicy
        The resolved scale action and metadata.

    Raises
    ------
    ScaleContractError
        If metadata is missing, invalid, contradictory, or exploratory without permission.
    """
    if not isinstance(metadata, dict):
        raise ScaleContractError("Calibration metadata must be a dictionary")

    # Canonicalize string values
    accuracy_class = str(metadata.get("accuracy_class") or "unknown").strip().lower()
    calibration_source = str(metadata.get("calibration_source") or "unknown").strip().lower()
    scale_source = str(metadata.get("scale_source") or "unknown").strip().lower()
    units = str(metadata.get("units") or metadata.get("source_units") or "unknown").strip().lower()
    is_panoptic = bool(metadata.get("is_panoptic") or "calibDataSource" in metadata)

    # Valid metric units
    metric_units = {"m", "meter", "meters", "cm", "centimeter", "centimeters", "mm", "millimeter", "millimeters"}
    has_metric_units = units in metric_units

    # 1. Check for contradictory metadata: production accuracy but arbitrary / unknown units
    if accuracy_class == "production" and not has_metric_units:
        raise ScaleContractError(
            f"Production calibration contract violation: accuracy_class is 'production' "
            f"but units are non-metric or undefined: {units!r}."
        )

    # 2. Check for authoritative metric sources
    authoritative_sources = {
        "checkerboard", "charuco", "physical_target", "known_baseline",
        "cmu_panoptic", "panoptic", "studio", "calibrated_studio", "panoptic_studio",
    }
    is_authoritative_source = (
        calibration_source in authoritative_sources
        or scale_source in authoritative_sources
        or is_panoptic
    )

    if (accuracy_class == "production" and has_metric_units) or (is_authoritative_source and has_metric_units):
        source = (
            scale_source if scale_source in authoritative_sources
            else (calibration_source if calibration_source in authoritative_sources
                  else ("panoptic_studio" if is_panoptic else "checkerboard"))
        )
        return ScalePolicy(
            action="PRESERVE_METRIC",
            scale_source=source,
            is_metric=True,
            scale_factor=1.0,
            reason=(
                f"Authoritative metric calibration preserved ({source}, units={units}); "
                "anthropometric scaling is prohibited."
            ),
        )

    # 3. Check for exploratory unscaled calibration
    is_exploratory = (
        accuracy_class == "exploratory"
        or calibration_source.startswith("markerless_pose")
        or scale_source in ("unscaled", "arbitrary", "estimated")
        or units in ("arbitrary",)
    )

    if is_exploratory:
        if not allow_exploratory:
            raise ScaleContractError(
                f"Calibration is exploratory / unscaled (accuracy_class={accuracy_class!r}, "
                f"calibration_source={calibration_source!r}, units={units!r}). "
                "In production mode, triangulation requires authoritative metric calibration. "
                "Pass --allow-exploratory-calibration to explicitly permit anthropometric scaling."
            )
        return ScalePolicy(
            action="ESTIMATE_ANTHROPOMETRIC",
            scale_source="anthropometric",
            is_metric=False,
            scale_factor=0.0,
            reason="Exploratory calibration: anthropometric scaling permitted via explicit override.",
        )

    # 4. Missing, ambiguous, or invalid scale metadata
    raise ScaleContractError(
        f"Missing or invalid scale metadata in calibration: accuracy_class={metadata.get('accuracy_class')!r}, "
        f"units={metadata.get('units')!r}, calibration_source={metadata.get('calibration_source')!r}. "
        "Production triangulation requires verified metric units ('meters' or 'cm') and a known metric scale source. "
        "For exploratory runs, explicitly declare accuracy_class='exploratory' and pass --allow-exploratory-calibration."
    )


# Approximate adult human bone lengths in meters.
# Using COCO-WholeBody keypoint indices (0-16 for body).
# 5: L-shoulder, 6: R-shoulder
# 11: L-hip, 12: R-hip
# 13: L-knee, 14: R-knee
# 15: L-ankle, 16: R-ankle
_BONE_PRIORS = [
    ((5, 6), 0.36),   # Shoulder width
    ((11, 12), 0.28), # Hip width
    ((11, 13), 0.40), # Left femur
    ((12, 14), 0.40), # Right femur
    ((13, 15), 0.40), # Left tibia
    ((14, 16), 0.40), # Right tibia
    ((5, 7), 0.30),   # Left upper arm
    ((6, 8), 0.30),   # Right upper arm
    ((7, 9), 0.25),   # Left lower arm
    ((8, 10), 0.25),  # Right lower arm
]


def apply_metric_scale(
    skeleton3d: np.ndarray,
    extrinsics: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], float]:
    """
    Scale the triangulated 3D skeleton and camera translations to metric units (meters)
    using average human anthropometric bone lengths.
    
    Args:
        skeleton3d: (num_frames, 133, 3) float64 array of 3D keypoints.
        extrinsics: List of N (R, t) camera poses.
        
    Returns:
        scaled_skeleton3d: (num_frames, 133, 3) scaled to meters.
        scaled_extrinsics: List of N (R, t) where t is scaled to meters.
        scale_factor: The multiplier applied to the scene.
    """
    if len(skeleton3d) == 0:
        return skeleton3d, extrinsics, 1.0
        
    estimated_scales = []
    
    for bone_idx, true_length in _BONE_PRIORS:
        idx_a, idx_b = bone_idx
        
        # Calculate lengths across all frames
        pts_a = skeleton3d[:, idx_a, :]
        pts_b = skeleton3d[:, idx_b, :]
        
        # Ignore frames where either point is NaN
        valid = ~np.isnan(pts_a[:, 0]) & ~np.isnan(pts_b[:, 0])
        
        if not np.any(valid):
            continue
            
        diff = pts_a[valid] - pts_b[valid]
        lengths = np.linalg.norm(diff, axis=1)
        
        # Median length across all valid frames to ignore outliers
        median_length = np.median(lengths)
        
        if median_length > 1e-5:
            # scale_factor = true_length / measured_length
            estimated_scales.append(true_length / median_length)
            
    if not estimated_scales:
        # Fallback if no bones were visible
        return skeleton3d, extrinsics, 1.0
        
    # Final robust scale factor
    scale_factor = np.median(estimated_scales)
    
    # Apply to skeleton
    scaled_skeleton3d = skeleton3d * scale_factor
    
    # Apply to camera translations (R remains unchanged)
    scaled_extrinsics = [
        (R, t * scale_factor) for R, t in extrinsics
    ]
    
    return scaled_skeleton3d, scaled_extrinsics, float(scale_factor)
