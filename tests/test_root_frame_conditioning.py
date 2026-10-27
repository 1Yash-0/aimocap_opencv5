"""Test _orthonormal_frame conditioning on near-parallel inputs.

When spine_dir and the lateral line (hip or shoulder) are nearly parallel,
Gram-Schmidt (x = lateral - (lateral·z)z) strips almost all of the lateral
vector, leaving a tiny noise-dominated residual as the x-axis. This produces
a garbage frame that misorients the shoulders by ~90°.

The fix: when the lateral component perpendicular to z is too small (below a
threshold), fall back to a stable world axis (the one least aligned with z),
same as the existing degenerate-spine fallback. This ensures a well-conditioned
orthonormal frame even when the actor's torso is nearly horizontal.
"""
import numpy as np
import pytest

from aimocap.retarget.root_frame import _orthonormal_frame


def test_orthonormal_frame_near_parallel_lateral():
    """When spine_dir and lateral are nearly anti-parallel (3.8°, as in a
    horizontal torso), the x-axis must point roughly along the lateral direction,
    not be dominated by Y/Z noise. Gram-Schmidt strips the lateral's dominant
    component, leaving a tiny noise-dominated residual. The fix: detect when
    the perpendicular residual is too small relative to the lateral and fall
    back to a stable world axis."""
    # Realistic case from the original synthetic data: spine ~+X, lateral ~-X,
    # 3.8° apart (near anti-parallel).
    spine_dir = np.array([0.999, 0.020, 0.021])
    lateral = np.array([-0.999, 0.026, 0.027])

    F = _orthonormal_frame(spine_dir, lateral)
    x, y, z = F[:, 0], F[:, 1], F[:, 2]

    # Must be orthonormal
    assert np.allclose(F.T @ F, np.eye(3), atol=1e-6), "Frame not orthonormal"

    # z must be the spine direction (normalized)
    assert np.allclose(z, spine_dir / np.linalg.norm(spine_dir), atol=1e-6), (
        f"z-axis should be spine_dir, got {z}"
    )

    # x must be a STABLE direction (not noise). When the lateral is nearly
    # parallel to z, there's no good perpendicular from the lateral — the
    # correct behavior is a reproducible world-axis fallback, not a
    # noise-dominated Gram-Schmidt residual. The bug: x ≈ [0, 0.69, 0.72]
    # (noise). The fix: x ≈ [0, 1, 0] (stable Y-axis fallback, least aligned
    # with z≈+X). Test: x should not have a large Z component (the noise
    # direction) and should be a clean world axis (one component ≈ ±1).
    assert abs(x[2]) < 0.1, (
        f"x-axis has a large Z component ({x[2]:.4f}), indicating noise. "
        f"x={x}"
    )
    # x should be a clean world axis (one dominant component)
    assert max(abs(x[0]), abs(x[1])) > 0.99, (
        f"x-axis is not a clean axis fallback: {x}"
    )


def test_orthonormal_frame_perpendicular_inputs():
    """Sanity check: the well-conditioned case (90°) must still work."""
    spine_dir = np.array([0.0, 0.0, 1.0])
    lateral = np.array([1.0, 0.0, 0.0])

    F = _orthonormal_frame(spine_dir, lateral)
    x, y, z = F[:, 0], F[:, 1], F[:, 2]

    assert np.allclose(F.T @ F, np.eye(3), atol=1e-6)
    assert np.allclose(z, [0, 0, 1], atol=1e-6)
    assert np.allclose(x, [1, 0, 0], atol=1e-6)
    assert np.allclose(y, [0, 1, 0], atol=1e-6)
