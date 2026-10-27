import numpy as np
import pytest
from aimocap.math.kinematics import COCO_18_PARENTS, REST_DIRECTIONS, forward_kinematics
from aimocap.math.filter import filter_params_one_euro_quaternion
from scipy.spatial.transform import Rotation

def test_topology_has_pelvis_root():
    assert COCO_18_PARENTS[17] == -1
    assert COCO_18_PARENTS[11] == 17
    assert COCO_18_PARENTS[12] == 17
    
def test_fk_at_zero_rotation_matches_tpose():
    root_pos = np.zeros(3)
    joint_rots = np.zeros((18, 3))
    bone_lengths = np.ones(18)
    
    pts3d = forward_kinematics(root_pos, joint_rots, bone_lengths)
    
    # Left hip is 11, should be at (0.707, -0.707, 0)
    assert np.allclose(pts3d[11], [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0])
    # Right hip is 12, should be at (-0.707, -0.707, 0)
    assert np.allclose(pts3d[12], [-np.sqrt(2)/2, -np.sqrt(2)/2, 0.0])
    
def test_quaternion_filter_no_wraparound():
    angles = np.linspace(0, 2*np.pi, 30)
    params_seq = np.zeros((30, 57))
    for i, a in enumerate(angles):
        rotvec = np.array([0, 0, a])
        r = Rotation.from_rotvec(rotvec)
        params_seq[i, 3:6] = r.as_rotvec()
        
    filtered = filter_params_one_euro_quaternion(params_seq, fps=30.0)
    assert not np.any(np.isnan(filtered))
