import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Parent array for the 18 COCO joints (17 original + 1 virtual pelvis root).
# We use a virtual Pelvis (17) as the root.
COCO_18_PARENTS = np.array([
    5,   # 0: nose -> L_shoulder
    0,   # 1: left_eye -> nose
    0,   # 2: right_eye -> nose
    1,   # 3: left_ear -> left_eye
    2,   # 4: right_ear -> right_eye
    17,  # 5: left_shoulder -> pelvis
    17,  # 6: right_shoulder -> pelvis
    5,   # 7: left_elbow -> left_shoulder
    6,   # 8: right_elbow -> right_shoulder
    7,   # 9: left_wrist -> left_elbow
    8,   # 10: right_wrist -> right_elbow
    17,  # 11: left_hip -> pelvis
    17,  # 12: right_hip -> pelvis
    11,  # 13: left_knee -> left_hip
    12,  # 14: right_knee -> right_hip
    13,  # 15: left_ankle -> left_knee
    14,  # 16: right_ankle -> right_knee
    -1,  # 17: pelvis -> ROOT
])

# Canonical T-pose bone directions in the parent's local frame.
# This ensures that zero rotation corresponds to a natural T-pose.
REST_DIRECTIONS = np.array([
    [0.0, 1.0, 0.0],   # 0: nose (up)
    [1.0, 1.0, 0.0],   # 1: left_eye (up and left)
    [-1.0, 1.0, 0.0],  # 2: right_eye (up and right)
    [1.0, 0.0, 0.0],   # 3: left_ear (left)
    [-1.0, 0.0, 0.0],  # 4: right_ear (right)
    [1.0, 2.0, 0.0],   # 5: left_shoulder (up and left from pelvis)
    [-1.0, 2.0, 0.0],  # 6: right_shoulder (up and right from pelvis)
    [1.0, 0.0, 0.0],   # 7: left_elbow (out to left from shoulder)
    [-1.0, 0.0, 0.0],  # 8: right_elbow (out to right from shoulder)
    [1.0, 0.0, 0.0],   # 9: left_wrist (out to left)
    [-1.0, 0.0, 0.0],  # 10: right_wrist (out to right)
    [1.0, -1.0, 0.0],  # 11: left_hip (down and left from pelvis)
    [-1.0, -1.0, 0.0], # 12: right_hip (down and right from pelvis)
    [0.0, -1.0, 0.0],  # 13: left_knee (down from hip)
    [0.0, -1.0, 0.0],  # 14: right_knee (down from hip)
    [0.0, -1.0, 0.0],  # 15: left_ankle (down from knee)
    [0.0, -1.0, 0.0],  # 16: right_ankle (down from knee)
    [0.0, 0.0, 0.0],   # 17: pelvis (root, no bone direction)
], dtype=np.float64)

# Normalize just in case
for i in range(17):
    norm = np.linalg.norm(REST_DIRECTIONS[i])
    if norm > 1e-6:
        REST_DIRECTIONS[i] /= norm

JOINT_TYPES = {
    'hinge':   [7, 8, 13, 14],          # elbows, knees
    'ball':    [5, 6, 11, 12],          # shoulders, hips
    'saddle':  [9, 10, 15, 16],         # wrists, ankles (approximation)
    'fixed':   [0, 1, 2, 3, 4, 17],     # head chain + root (no limits, just FK)
}

# Cluster definitions for non-body points.
# Format: "NAME": (list_of_indices, anchor_body_index)
CLUSTER_DEFS = {
    "FACE": (list(range(23, 91)), 0),       # Anchor to Nose
    "LEFT_HAND": (list(range(91, 112)), 9),   # Anchor to Left Wrist
    "RIGHT_HAND": (list(range(112, 133)), 10),# Anchor to Right Wrist
    "LEFT_FOOT": ([17, 18, 19], 15),          # Anchor to Left Ankle
    "RIGHT_FOOT": ([20, 21, 22], 16),         # Anchor to Right Ankle
}


def compute_median_bone_lengths(
    pts3d: np.ndarray,
    *,
    min_valid_samples: int = 3,
) -> np.ndarray:
    """
    Compute median bone lengths across a sequence to find the rigid structure.
    
    Args:
        pts3d: (num_frames, 17, 3) triangulated 3D points
        
    Returns:
        bone_lengths: (18,) array where bone_lengths[i] is the length of the bone
                      from PARENTS[i] to i. (Index 17 will be 0).
    """
    num_frames, num_kpts, _ = pts3d.shape
    lengths = np.zeros((num_frames, 18))
    
    # Estimate pelvis position as midpoint of hips
    pelvis_pts = (pts3d[:, 11, :] + pts3d[:, 12, :]) / 2.0
    
    for i in range(17):
        p = COCO_18_PARENTS[i]
        if p == -1:
            continue
        
        # Distance between joint i and its parent p
        p_pts = pelvis_pts if p == 17 else pts3d[:, p, :]
        diff = pts3d[:, i, :] - p_pts
        dist = np.linalg.norm(diff, axis=1)
        lengths[:, i] = dist
        
    finite_counts = np.sum(np.isfinite(lengths), axis=0)
    bad = [
        i for i in range(17)
        if finite_counts[i] < min_valid_samples
        or not np.isfinite(np.nanmedian(lengths[:, i]))
        or np.nanmedian(lengths[:, i]) <= 1e-9
    ]
    if bad:
        raise ValueError(
            "insufficient finite geometry for skeleton fitting: "
            + ", ".join(
                f"bone={i} finite_samples={int(finite_counts[i])}"
                for i in bad
            )
        )
    median_lengths = np.nanmedian(lengths, axis=0)
    median_lengths[17] = 0.0  # Root has no bone length
    return median_lengths

def forward_kinematics(root_pos: np.ndarray, joint_rotations: np.ndarray, bone_lengths: np.ndarray) -> np.ndarray:
    """
    Compute 3D joint positions from root position and joint rotations.
    
    Args:
        root_pos: (3,) root position
        joint_rotations: (18, 3) rotation vectors (axis-angle) for each joint
        bone_lengths: (18,) bone lengths
        
    Returns:
        pts3d: (18, 3) 3D positions
    """
    pts3d = np.zeros((18, 3))
    
    # Store global rotations to apply to children
    global_rotvecs = np.zeros((18, 3))
    
    # Root (Pelvis)
    pts3d[17] = root_pos
    global_rotvecs[17] = joint_rotations[17]
    
    # Traverse in topological order
    topo_order = [17, 11, 12, 13, 15, 14, 16, 5, 6, 8, 10, 7, 9, 0, 1, 3, 2, 4]
    
    for i in topo_order:
        p = COCO_18_PARENTS[i]
        if p == -1:
            continue
            
        # Global rotation of this joint is parent's global rotation * local rotation
        R_parent = Rotation.from_rotvec(global_rotvecs[p])
        R_local = Rotation.from_rotvec(joint_rotations[i])
        R_global = R_parent * R_local
        global_rotvecs[i] = R_global.as_rotvec()
        
        local_bone_vec = REST_DIRECTIONS[i] * bone_lengths[i]
        pts3d[i] = pts3d[p] + R_global.apply(local_bone_vec)
        
    return pts3d

def _ik_residual(params: np.ndarray, target_pts3d: np.ndarray, bone_lengths: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    root_pos = params[:3]
    joint_rotations = params[3:3+18*3].reshape((18, 3))
    
    pred_pts3d = forward_kinematics(root_pos, joint_rotations, bone_lengths)
    
    diff = pred_pts3d[:17][valid_mask[:17]] - target_pts3d[:17][valid_mask[:17]]
    return diff.flatten()

def estimate_initial_params(target_pts3d: np.ndarray) -> np.ndarray:
    """
    Analytically estimate joint rotations by aligning bones to the target vectors.
    """
    params = np.zeros(3 + 18*3)
    if (
        not np.isfinite(target_pts3d[11]).all()
        or not np.isfinite(target_pts3d[12]).all()
    ):
            return params
        
    # Pelvis is midpoint of hips
    pelvis_pos = (target_pts3d[11] + target_pts3d[12]) / 2.0
    params[:3] = pelvis_pos
    
    # Root orientation: align with hips and shoulders
    target_vecs = []
    rest_vecs = []
    for c in [11, 12, 5, 6]:
        v = target_pts3d[c] - pelvis_pos
        if np.isfinite(v).all() and np.linalg.norm(v) > 1e-8:
            target_vecs.append(v / np.linalg.norm(v))
            rest_vecs.append(REST_DIRECTIONS[c] / np.linalg.norm(REST_DIRECTIONS[c]))
            
    if len(target_vecs) >= 2:
        R_pelvis, _ = Rotation.align_vectors(target_vecs, rest_vecs)
    else:
        R_pelvis = Rotation.identity()
        
    global_rots = {17: R_pelvis}
    params[3 + 17*3 : 3 + 17*3 + 3] = R_pelvis.as_rotvec()
    
    topo_order = [11, 12, 13, 15, 14, 16, 5, 6, 8, 10, 7, 9, 0, 1, 3, 2, 4]
    
    for i in topo_order:
        p = COCO_18_PARENTS[i]
        if p == -1:
            continue
            
        p_pos = pelvis_pos if p == 17 else target_pts3d[p]
        target_vec = target_pts3d[i] - p_pos
        
        if not np.isfinite(target_vec).all() or np.linalg.norm(target_vec) < 1e-6:
            global_rots[i] = global_rots[p]
            params[3 + i*3 : 3 + i*3 + 3] = np.zeros(3)
            continue
            
        target_vec_norm = target_vec / np.linalg.norm(target_vec)
        rest_vec = REST_DIRECTIONS[i]
        rest_vec_norm = rest_vec / np.linalg.norm(rest_vec)
        
        v1 = rest_vec_norm
        v2 = target_vec_norm
        axis = np.cross(v1, v2)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(v1, v2)
        
        if sin_angle > 1e-6:
            axis = axis / sin_angle
            angle = np.arctan2(sin_angle, cos_angle)
            R_global = Rotation.from_rotvec(axis * angle)
        elif cos_angle < 0:
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(v1[0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            axis = np.cross(v1, ortho)
            axis = axis / np.linalg.norm(axis)
            R_global = Rotation.from_rotvec(axis * np.pi)
        else:
            R_global = Rotation.identity()
            
        global_rots[i] = R_global
        
        R_local = global_rots[p].inv() * R_global
        params[3 + i*3 : 3 + i*3 + 3] = R_local.as_rotvec()
        
    return params


def _ik_residual(params: np.ndarray, target_pts3d: np.ndarray, bone_lengths: np.ndarray, valid_mask: np.ndarray, prev_params: np.ndarray, penalty_weight_multiplier: float) -> np.ndarray:
    root_pos = params[:3]
    joint_rotations = params[3:3+18*3].reshape((18, 3))
    
    pred_pts3d = forward_kinematics(root_pos, joint_rotations, bone_lengths)
    
    diff = pred_pts3d[:17][valid_mask[:17]] - target_pts3d[:17][valid_mask[:17]]
    res_reproj = diff.flatten()
    
    res_limits = []
    sqrt_weight = 6.8 * np.sqrt(penalty_weight_multiplier)
    
    for i in range(18):
        rotvec = joint_rotations[i]
        angle = np.linalg.norm(rotvec)
        
        if i in JOINT_TYPES['hinge']:
            if angle > 1e-6:
                R_local = Rotation.from_rotvec(rotvec)
                rest = REST_DIRECTIONS[i]
                new_dir = R_local.apply(rest)
                
                # Penalize hyperextension based on Z coordinate (assuming Z forward is hyperextension)
                if new_dir[2] > 0.0:
                    res_limits.append(sqrt_weight * 5.0 * new_dir[2])
                else:
                    res_limits.append(0.0)
            else:
                res_limits.append(0.0)
                
            if angle > 2.6:
                res_limits.append(sqrt_weight * (angle - 2.6))
            else:
                res_limits.append(0.0)
                
        elif i in JOINT_TYPES['ball']:
            if angle > 1e-6:
                R_local = Rotation.from_rotvec(rotvec)
                rest = REST_DIRECTIONS[i]
                new_dir = R_local.apply(rest)
                cos_theta = np.clip(np.dot(rest, new_dir), -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
                limit = 1.65 if i in [5, 6] else 2.09
                if theta > limit:
                    res_limits.append(sqrt_weight * (theta - limit))
                else:
                    res_limits.append(0.0)
            else:
                res_limits.append(0.0)
            
        elif i in JOINT_TYPES['saddle']:
            if angle > 1.5:
                res_limits.append(sqrt_weight * (angle - 1.5))
            else:
                res_limits.append(0.0)
                
    res_limits = np.array(res_limits)
    
    if prev_params is not None:
        temporal_diff = params[3:] - prev_params[3:]
        res_temporal = temporal_diff * 0.3
    else:
        res_temporal = np.zeros(18 * 3)
        
    return np.concatenate([res_reproj, res_limits, res_temporal])

def fit_skeleton_sequence(pts3d_seq: np.ndarray, bone_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the skeleton to an entire sequence using Annealing.
    
    Returns:
        fk_pts3d_seq: (num_frames, 18, 3)
        params_seq: (num_frames, 57)
    """
    if pts3d_seq.ndim != 3 or pts3d_seq.shape[1:] != (17, 3):
        raise ValueError(f"expected skeleton sequence shape (F,17,3), got {pts3d_seq.shape}")
    if not np.isfinite(bone_lengths[:17]).all() or np.any(bone_lengths[:17] <= 0.0):
        raise ValueError("skeleton fitting requires finite positive bone lengths")
    if not np.any(np.sum(np.isfinite(pts3d_seq).all(axis=2), axis=1) >= 2):
        raise ValueError("skeleton fitting requires at least one frame with finite geometry")

    num_frames = pts3d_seq.shape[0]
    fk_pts3d_seq = np.zeros((num_frames, 18, 3))
    params_seq = np.zeros((num_frames, 57))
    
    last_params = None
    
    print("Fitting rigid skeleton to sequence (Inverse Kinematics with Annealing)...")
    for f in range(num_frames):
        target = pts3d_seq[f]
        valid_mask = np.isfinite(target).all(axis=1)
        
        if np.sum(valid_mask[:17]) < 2:
            if last_params is not None:
                params_seq[f] = last_params
                root_pos = last_params[:3]
                joint_rots = last_params[3:].reshape((18, 3))
                fk_pts3d_seq[f] = forward_kinematics(root_pos, joint_rots, bone_lengths)
            continue
            
        if last_params is None:
            params = estimate_initial_params(target)
        else:
            params = last_params.copy()

        if not np.isfinite(params).all():
            raise ValueError(f"non-finite initial IK parameters at frame {f}")

        for multiplier in [0.1, 0.3, 0.6, 1.0]:
            initial_residual = _ik_residual(
                params, target, bone_lengths, valid_mask, last_params, multiplier
            )
            if not np.isfinite(initial_residual).all():
                raise ValueError(f"non-finite initial IK residual at frame {f}")
            res = least_squares(
                _ik_residual,
                params,
                args=(target, bone_lengths, valid_mask, last_params, multiplier),
                method='trf',
                max_nfev=20
            )
            if not np.isfinite(res.x).all() or not np.isfinite(res.fun).all():
                raise ValueError(f"non-finite IK result at frame {f}")
            params = res.x
            
        root_pos = params[:3]
        joint_rots = params[3:].reshape((18, 3))
        
        fk_pts3d_seq[f] = forward_kinematics(root_pos, joint_rots, bone_lengths)
        params_seq[f] = params
        last_params = params
        
        if f % 50 == 0 and f > 0:
            print(f"  Processed {f}/{num_frames} frames")
            
    return fk_pts3d_seq, params_seq

def forward_kinematics_sequence(params_seq: np.ndarray, bone_lengths: np.ndarray) -> np.ndarray:
    """
    Apply FK to an entire sequence of parameters.
    
    Args:
        params_seq: (num_frames, 57)
        bone_lengths: (18,)
        
    Returns:
        pts3d_seq: (num_frames, 18, 3)
    """
    num_frames = params_seq.shape[0]
    pts3d_seq = np.zeros((num_frames, 18, 3))
    
    for f in range(num_frames):
        root_pos = params_seq[f, :3]
        joint_rots = params_seq[f, 3:].reshape((18, 3))
        pts3d_seq[f] = forward_kinematics(root_pos, joint_rots, bone_lengths)
        
    return pts3d_seq
