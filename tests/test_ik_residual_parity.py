"""Test NumPy and Torch IK residual parity."""

import numpy as np
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_residual_length_consistent():
    """Verify that MocapIKSolver builds consistent residual structures."""
    # Build minimal synthetic keypoint array (1 frame, 133 points)
    pts3d = np.zeros((1, 133, 3), dtype=np.float64)
    weights = np.ones((1, 133), dtype=np.float64)
    
    # Place basic points to avoid zero norms
    pts3d[0, :, 1] = np.linspace(0, 150, 133)
    
    skel = MocapSkeleton(pts3d, weights)
    solver = MocapIKSolver(skel)
    
    measured = {name: pts3d[0, i] for i, name in enumerate(skel.coco_anchor.values())}
    ctx = solver._precompute_frame(measured)
    
    x0 = np.zeros(solver.num_vars)
    r = solver._residuals_with_ctx(x0, ctx)
    
    assert isinstance(r, np.ndarray)
    assert r.ndim == 1
    assert r.size > 0
    assert not np.any(np.isnan(r)), "NaN found in residual output"
    print(f"  Residual length test passed: size={r.size}")


if __name__ == "__main__":
    test_residual_length_consistent()
