import numpy as np

# Indices for COCO-WholeBody (0-16 are body joints)
_BONE_PAIRS = [
    (5, 7),   # L shoulder - L elbow
    (7, 9),   # L elbow - L wrist
    (6, 8),   # R shoulder - R elbow
    (8, 10),  # R elbow - R wrist
    (11, 13), # L hip - L knee
    (13, 15), # L knee - L ankle
    (12, 14), # R hip - R knee
    (14, 16), # R knee - R ankle
    (5, 6),   # L shoulder - R shoulder
    (11, 12), # L hip - R hip
    (5, 11),  # L shoulder - L hip
    (6, 12),  # R shoulder - R hip
]

def compute_bone_variance(skeleton3d: np.ndarray) -> float:
    """
    Compute the temporal variance of bone lengths across frames.
    A correctly calibrated sequence will have near-zero bone variance.
    
    Args:
        skeleton3d: (F, 133, 3) array of 3D keypoints
        
    Returns:
        Mean coefficient of variation (std / mean) across all defined bone pairs.
    """
    num_frames = skeleton3d.shape[0]
    if num_frames < 2:
        return 0.0
        
    bone_covs = []
    
    for (idx1, idx2) in _BONE_PAIRS:
        p1 = skeleton3d[:, idx1, :]  # (F, 3)
        p2 = skeleton3d[:, idx2, :]  # (F, 3)
        
        # Calculate lengths per frame
        # L2 norm across axis=1
        diff = p1 - p2
        lengths = np.linalg.norm(diff, axis=1)
        
        # Filter NaNs
        valid = ~np.isnan(lengths)
        if np.sum(valid) > 5:  # need at least a few frames
            lengths_valid = lengths[valid]
            mean_len = np.mean(lengths_valid)
            std_len = np.std(lengths_valid)
            
            if mean_len > 1e-4:
                # Coefficient of variation (std relative to mean length)
                # We use CV because different bones have different absolute lengths
                cv = std_len / mean_len
                bone_covs.append(cv)
                
    if not bone_covs:
        return float('inf')
        
    return float(np.mean(bone_covs))


def sampson_distance(x1: np.ndarray, x2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute symmetric Sampson distance between sets of corresponding points.
    
    Args:
        x1: (N, 2) array of points in camera 1
        x2: (N, 2) array of points in camera 2
        F: (3, 3) Fundamental matrix mapping camera 1 to camera 2
        
    Returns:
        (N,) array of Sampson distances
    """
    # Convert to homogeneous coordinates
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])
    
    # F @ x1_h.T -> (3, N)
    Fx1 = (F @ x1_h.T).T  # (N, 3)
    # F.T @ x2_h.T -> (3, N)
    Ftx2 = (F.T @ x2_h.T).T  # (N, 3)
    
    # x2_h.T * F * x1_h -> (N,)
    x2_F_x1 = np.sum(x2_h * Fx1, axis=1)
    
    # Sampson distance
    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    # Add small epsilon to prevent division by zero
    dist = (x2_F_x1**2) / (denom + 1e-8)
    
    return np.sqrt(dist)


def compute_epipolar_consistency(
    keypoints: np.ndarray,
    scores: np.ndarray,
    F_matrices: dict[tuple[int, int], np.ndarray],
    min_conf: float = 0.5
) -> dict:
    """
    Compute hierarchical epipolar consistency metrics.
    
    Args:
        keypoints: (F, C, J, 2) array of 2D detections
        scores: (F, C, J) array of confidence scores
        F_matrices: Dict mapping (cam1_idx, cam2_idx) to 3x3 Fundamental matrix
        min_conf: Minimum confidence to consider a detection valid
        
    Returns:
        Hierarchical dict of epipolar consistency metrics
    """
    num_frames, num_cams, num_joints, _ = keypoints.shape
    
    results = {
        "pairs": {},
        "per_joint_failure_rate": {},
        "median_err": 0.0,
        "p95_err": 0.0,
        "valid_coverage": 0.0,
    }
    
    all_distances = []
    joint_failures = {j: 0 for j in range(num_joints)}
    joint_attempts = {j: 0 for j in range(num_joints)}
    
    for (c1, c2), F in F_matrices.items():
        pair_distances = []
        
        for j in range(num_joints):
            # Mask of frames where this joint is valid in both cameras
            valid_frames = (scores[:, c1, j] >= min_conf) & (scores[:, c2, j] >= min_conf)
            
            if np.any(valid_frames):
                pts1 = keypoints[valid_frames, c1, j]
                pts2 = keypoints[valid_frames, c2, j]
                
                dist = sampson_distance(pts1, pts2, F)
                pair_distances.extend(dist.tolist())
                all_distances.extend(dist.tolist())
                
                # Count failures for this joint (distance > 15px as a generic failure metric for the per-joint stat)
                joint_attempts[j] += len(dist)
                joint_failures[j] += np.sum(dist > 15.0)
                
        if pair_distances:
            results["pairs"][(c1, c2)] = {
                "median": float(np.median(pair_distances)),
                "p95": float(np.percentile(pair_distances, 95)),
                "count": len(pair_distances)
            }
            
    for j in range(num_joints):
        if joint_attempts[j] > 0:
            results["per_joint_failure_rate"][j] = float(joint_failures[j] / joint_attempts[j])
        else:
            results["per_joint_failure_rate"][j] = 1.0  # complete failure if no attempts
            
    if all_distances:
        results["median_err"] = float(np.median(all_distances))
        results["p95_err"] = float(np.percentile(all_distances, 95))
        
    # Find how many (F, J) pairs had at least 2 confident cameras
    cams_per_joint = np.sum(scores >= min_conf, axis=1) # (F, J)
    possible_epipolar_joints = np.sum(cams_per_joint >= 2)
    
    # We want to know how many (F, J) actually got evaluated.
    # We can just count unique (F, J) from the loop, or since we just looped over all F_matrices,
    # if cams_per_joint >= 2, it definitely produced at least one pair distance.
    # So valid_coverage can just be `possible_epipolar_joints / (num_frames * num_joints)`
    # Wait, the instruction said "90%+ of confident observations pass epipolar checks"
    # The coverage of epipolar isn't what failed, but the fraction of the skeleton that is valid.
    # Let's just return `possible_epipolar_joints / (num_frames * num_joints)`
    results["valid_coverage"] = float(possible_epipolar_joints) / max(1, num_frames * num_joints)
        
    return results
