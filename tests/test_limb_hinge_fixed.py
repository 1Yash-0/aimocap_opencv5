"""Test limb hinge constraints - simplified version."""
import numpy as np
import pytest

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_skeleton import MocapSkeleton
from aimocap.retarget.mocap_ik import MocapIKSolver


from tests._assets import requires_manny_fbx
def _solver(limb_hinge='off'):
    """Create a solver for testing."""
    fbx = Skeleton("Manny.FBX")
    rest, _ = fbx.get_forward_kinematics()
    ni = fbx.name_to_idx
    pts = np.zeros((2, 133, 3), dtype=float)
    mapping = {
        0: "head", 5: "upperarm_l", 6: "upperarm_r",
        7: "lowerarm_l", 8: "lowerarm_r", 9: "hand_l", 10: "hand_r",
        11: "thigh_l", 12: "thigh_r", 13: "calf_l", 14: "calf_r",
        15: "foot_l", 16: "foot_r", 17: "ball_l", 20: "ball_r",
    }
    for ci, name in mapping.items():
        pts[:, ci] = rest[ni[name]]
    pts[:, 11] = rest[ni["thigh_l"]]
    pts[:, 12] = rest[ni["thigh_r"]]
    pts[:, 5] = rest[ni["upperarm_l"]]
    pts[:, 6] = rest[ni["upperarm_r"]]
    skel = MocapSkeleton(pts, fbx_skel=fbx)
    return MocapIKSolver(skel, limb_hinge=limb_hinge)


@requires_manny_fbx
def test_limb_hinge_parameter():
    """Test that limb_hinge parameter is accepted and validated."""
    solver = _solver(limb_hinge='elbow_knee')
    assert solver is not None
    
    # Test that limb_hinge='off' works
    solver_off = _solver(limb_hinge='off')
    assert solver_off is not None
    
    # Test invalid value
    with pytest.raises(ValueError, match='limb_hinge'):
        _solver(limb_hinge='invalid')
    
    print("✓ Test (a): limb_hinge parameter validation works")


@requires_manny_fbx
def test_hinge_axes_exist():
    """Test that hinge axes can be retrieved."""
    solver = _solver(limb_hinge='elbow_knee')
    hinge_axes = solver._limb_hinge_axes()
    
    # Should contain elbow and knee joints
    assert 'knee_l' in hinge_axes
    assert 'knee_r' in hinge_axes
    assert 'elbow_l' in hinge_axes
    assert 'elbow_r' in hinge_axes
    
    # Each should return (axis, sign_corrected)
    for joint in ['knee_l', 'knee_r', 'elbow_l', 'elbow_r']:
        axis, sign_corrected = hinge_axes[joint]
        assert isinstance(axis, np.ndarray)
        assert axis.shape == (3,)
        assert np.linalg.norm(axis) == pytest.approx(1.0, rel=1e-6)
        assert isinstance(sign_corrected, (bool, np.bool_))
    
    print("✓ Hinge axes exist and have correct format")


@requires_manny_fbx
def test_context_with_hinge():
    """Test that hinge_joints are added to context when enabled."""
    # With hinge enabled
    solver_enabled = _solver(limb_hinge='elbow_knee')
    
    # Create measured data
    measured = {}
    for name in solver_enabled.skel.coco_anchor.values():
        measured[name] = np.zeros(3)
    ctx_enabled = solver_enabled._precompute_frame(measured)
    
    assert 'hinge_joints' in ctx_enabled
    assert len(ctx_enabled['hinge_joints']) > 0
    
    # With hinge disabled
    solver_disabled = _solver(limb_hinge='off')
    ctx_disabled = solver_disabled._precompute_frame(measured)
    
    # Should be empty list when 'off'
    assert 'hinge_joints' in ctx_disabled
    assert ctx_disabled['hinge_joints'] == []
    
    print("✓ Context includes hinge_joints when enabled")


@requires_manny_fbx
def test_residuals_finite():
    """Test that residuals are finite with hinge constraints."""
    solver = _solver(limb_hinge='elbow_knee')
    
    # Create measured data
    measured = {}
    for name in solver.skel.coco_anchor.values():
        measured[name] = np.zeros(3)
    
    # Create a simple state
    x = np.zeros(3 + solver.skel.num_joints * 3)
    ctx = solver._precompute_frame(measured)
    
    # Compute residuals
    residuals = solver._residuals_with_ctx(x, ctx)
    
    assert np.all(np.isfinite(residuals))
    assert len(residuals) > 0
    
    print("✓ Residuals are finite with hinge constraints")


if __name__ == "__main__":
    test_limb_hinge_parameter()
    test_hinge_axes_exist()
    test_context_with_hinge()
    test_residuals_finite()
    print("\n✅ All simplified limb hinge tests passed!")