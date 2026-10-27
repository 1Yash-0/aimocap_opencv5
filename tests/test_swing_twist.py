import numpy as np
from scipy.spatial.transform import Rotation
from aimocap.retarget.swing_twist import constrained_rotation


def test_direction_matched_exactly():
    rest = np.array([0.0, 0.0, 1.0])
    desired = np.array([1.0, 0.0, 0.0])
    R = constrained_rotation(rest, desired, roll_child_rest=None, roll_child_desired=None)
    mapped = R.apply(rest)
    assert np.allclose(mapped, desired, atol=1e-9)


def test_roll_pinned_by_child_direction():
    # rest bone points +Z; roll child points +X off the bone origin
    rest = np.array([0.0, 0.0, 1.0])
    roll_rest = np.array([1.0, 0.0, 0.0])
    # desired bone still +Z (no swing) but child should now point +Y -> 90deg roll
    desired = np.array([0.0, 0.0, 1.0])
    roll_des = np.array([0.0, 1.0, 0.0])
    R = constrained_rotation(rest, desired, roll_rest, roll_des)
    assert np.allclose(R.apply(rest), desired, atol=1e-9)
    # child maps to within roll-plane; its projection on plane perp to desired == roll_des projected
    child_mapped = R.apply(roll_rest)
    # reject component along desired, compare direction
    child_perp = child_mapped - np.dot(child_mapped, desired) * desired
    target_perp = roll_des - np.dot(roll_des, desired) * desired
    assert np.allclose(child_perp / np.linalg.norm(child_perp),
                       target_perp / np.linalg.norm(target_perp), atol=1e-9)


def test_identity_when_already_aligned():
    rest = np.array([0.0, 1.0, 0.0])
    R = constrained_rotation(rest, rest, None, None)
    assert np.allclose(R.as_matrix(), np.eye(3), atol=1e-9)


def test_roll_does_not_break_direction():
    """Adding a roll child must not change where the bone points."""
    rest = np.array([0.0, 0.0, 1.0])
    desired = np.array([0.5, 0.5, 1.0])  # diagonal
    desired = desired / np.linalg.norm(desired)
    roll_rest = np.array([1.0, 0.0, 0.0])
    roll_des = np.array([0.0, 1.0, 0.0])
    R = constrained_rotation(rest, desired, roll_rest, roll_des)
    assert np.allclose(R.apply(rest), desired, atol=1e-9)
