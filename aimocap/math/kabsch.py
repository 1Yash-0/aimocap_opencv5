"""Kabsch algorithm for robust head orientation from multiple face keypoints.

The head rotation is computed by fitting a rigid rotation to the observed
3D positions of face keypoints (nose, eyes, ears) against a canonical
template.  This is far more robust than the nose-to-neck direction vector,
which flips when the nose keypoint crosses the neck midline.

The Kabsch algorithm finds the optimal rotation R minimizing:
    ||R @ P - Q||²
where P is the observed point cloud and Q is the canonical template.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ── Canonical head template ──────────────────────────────────────────────
# Defines the anatomical layout of 5 COCO face keypoints relative to the
# shoulder midpoint (which is what measured["neck"] returns in the pipeline)
# in Manny's Z-up coordinate system (X=lateral, Y=forward, Z=up).
#
# IMPORTANT: The "neck" in our pipeline is the shoulder midpoint (COCO
# joints 5+6 averaged), NOT the anatomical neck joint.  The shoulder
# midpoint is ~10cm BELOW the actual neck.  So the nose, which is ~8cm
# above the actual neck, ends up ~-2cm below the shoulder midpoint when
# the person faces forward (+Y) at rest.
#
# This template was computed from the most upright frame (frame 390) of
# the Panoptic sequence, where the person stands facing roughly forward.
# The observed face geometry was rotated so the nose points +Y (Manny
# forward) and left/right are +X/-X respectively.
#
# Indices: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
# "Left" is +X in Manny (person's left when facing +Y).
CANONICAL_HEAD_TEMPLATE = np.array([
    [ 0.0,  10.0, -2.0],   # nose: 10cm forward, 2cm BELOW shoulder mid
    [ 2.1,   8.0,  0.0],   # left_eye: 2.1cm left, 8cm forward, at shoulder height
    [-2.1,   8.0,  0.0],   # right_eye: 2.1cm right
    [ 9.1,   2.0, -3.4],   # left_ear: 9.1cm left, 2cm forward, 3.4cm below
    [-9.1,   2.0, -3.4],   # right_ear: 9.1cm right
], dtype=np.float64)

# Minimum number of non-collinear points needed for a valid 3D rotation.
MIN_POINTS_FOR_KABSCH = 3


def kabsch_rotation(
    observed: np.ndarray,
    template: np.ndarray,
    weights: np.ndarray | None = None,
) -> Rotation:
    """Compute the optimal rotation aligning ``observed`` to ``template``.

    Uses the Kabsch algorithm (SVD-based) to find the rotation R that
    minimizes ``||R @ P - Q||²`` where P=observed, Q=template.

    Args:
        observed:  ``(N, 3)`` array of observed 3D points (e.g. nose, eyes,
                   ears triangulated from multi-view cameras).
        template:  ``(N, 3)`` array of canonical positions.
        weights:   Optional ``(N,)`` weights for weighted Kabsch.  Points
                   with higher confidence get more influence.

    Returns:
        ``scipy.spatial.transform.Rotation`` mapping observed→template.

    Raises:
        ValueError: if fewer than 3 non-collinear points are provided.
    """
    if observed.shape[0] < MIN_POINTS_FOR_KABSCH:
        raise ValueError(
            f"Kabsch needs ≥{MIN_POINTS_FOR_KABSCH} points, got {observed.shape[0]}")

    # Center both point clouds to their (weighted) centroids.
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        centroid_p = (w[:, None] * observed).sum(axis=0)
        centroid_q = (w[:, None] * template).sum(axis=0)
    else:
        centroid_p = observed.mean(axis=0)
        centroid_q = template.mean(axis=0)

    P = observed - centroid_p
    Q = template - centroid_q

    # Cross-covariance matrix.
    if weights is not None:
        H = (w[:, None] * P).T @ Q
    else:
        H = P.T @ Q

    # SVD: H = U S Vt
    U, S, Vt = np.linalg.svd(H)

    # Rotation: R = V @ U.T (maps observed→template)
    R = Vt.T @ U.T

    # Ensure proper rotation (det=+1), not reflection (det=-1).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return Rotation.from_matrix(R)


def compute_head_rotation_kabsch(
    keypoints: dict[str, np.ndarray],
    neck_pos: np.ndarray,
    confidence: dict[str, float] | None = None,
) -> Rotation | None:
    """Compute head rotation from multiple face keypoints via Kabsch.

    Args:
        keypoints:  dict with keys 'nose', 'left_eye', 'right_eye',
                    'left_ear', 'right_ear' (at least 3 must be present).
                    Values are 3D positions in world space.
        neck_pos:   3D position of the neck (shoulder midpoint).
        confidence: Optional dict mapping keypoint name→confidence [0,1].

    Returns:
        ``Rotation`` representing the head's global orientation, or
        ``None`` if insufficient valid keypoints.
    """
    FACE_KEYS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    valid_keys = []
    observed_pts = []
    template_pts = []
    weights = []

    for i, key in enumerate(FACE_KEYS):
        if key not in keypoints:
            continue
        pt = np.asarray(keypoints[key], dtype=np.float64)
        if not np.all(np.isfinite(pt)):
            continue
        conf = 1.0
        if confidence and key in confidence:
            conf = max(0.01, float(confidence[key]))
        valid_keys.append(key)
        observed_pts.append(pt - neck_pos)  # relative to neck
        template_pts.append(CANONICAL_HEAD_TEMPLATE[i])
        weights.append(conf)

    if len(valid_keys) < MIN_POINTS_FOR_KABSCH:
        return None

    observed = np.array(observed_pts)
    template = np.array(template_pts)
    w = np.array(weights)

    try:
        # kabsch_rotation(P, Q) returns R such that R @ P ≈ Q.
        # We want the head's global orientation = the rotation that maps
        # the canonical template (rest world frame) to the observed points
        # (current world frame).  So P=template, Q=observed.
        return kabsch_rotation(template, observed, w)
    except (ValueError, np.linalg.LinAlgError):
        return None
