"""Extrinsic parameter solver using markerless human keypoints."""

from __future__ import annotations

import cv2
import numpy as np

from aimocap.math.triangulate import triangulate_n_views
from aimocap.pose.keypoints import BODY_17, FEET_6

# Body keypoints only (indices 0-16). Hands and face are too coplanar/noisy for robust E-matrix estimation.
_VALID_KPT_INDICES = np.array(BODY_17, dtype=np.int32)
# Foot keypoints for floor detection. (15, 16 are ankles, 17-22 are toes)
_FOOT_KPT_INDICES = np.array([15, 16] + list(FEET_6), dtype=np.int32)


def calibrate_pair(
    pts0: np.ndarray,
    pts1: np.ndarray,
    scores0: np.ndarray,
    scores1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    min_conf: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute relative rotation and translation between two cameras from 2D keypoints.
    
    Args:
        pts0: (F, 133, 2) keypoints from Camera 0.
        pts1: (F, 133, 2) keypoints from Camera 1.
        scores0: (F, 133) confidence scores from Camera 0.
        scores1: (F, 133) confidence scores from Camera 1.
        K0: (3, 3) Intrinsic matrix of Camera 0.
        K1: (3, 3) Intrinsic matrix of Camera 1.
        min_conf: Minimum confidence threshold.
        
    Returns:
        R: (3, 3) Rotation matrix from Cam0 to Cam1.
        t: (3, 1) Translation vector from Cam0 to Cam1 (unit scale).
        inlier_mask: (N,) boolean mask of which correspondences were used.
    """
    # Extract only valid body keypoints that meet the confidence threshold in BOTH cameras
    # Shape of scores: (F, 133). We want an (F, 17) mask.
    mask = (scores0[:, _VALID_KPT_INDICES] >= min_conf) & (scores1[:, _VALID_KPT_INDICES] >= min_conf)
    
    # Flatten the Fx17 points into a list of matched pairs
    # pts0[:, _VALID_KPT_INDICES] is (F, 17, 2). Masking gives (N, 2).
    matched_pts0 = pts0[:, _VALID_KPT_INDICES][mask]
    matched_pts1 = pts1[:, _VALID_KPT_INDICES][mask]
    
    if len(matched_pts0) < 15:
        raise ValueError(f"Not enough high-confidence body keypoint correspondences ({len(matched_pts0)} < 15) to calibrate pair.")
        
    # Convert to normalized camera coordinates
    # K inverse @ [u, v, 1]
    K0_inv = np.linalg.inv(K0)
    K1_inv = np.linalg.inv(K1)
    
    pts0_hom = np.hstack((matched_pts0, np.ones((len(matched_pts0), 1))))
    pts1_hom = np.hstack((matched_pts1, np.ones((len(matched_pts1), 1))))
    
    norm_pts0 = (K0_inv @ pts0_hom.T).T[:, :2]
    norm_pts1 = (K1_inv @ pts1_hom.T).T[:, :2]
    
    # Compute Essential Matrix using normalized points
    # focal=1.0 and pp=(0,0) since points are already normalized
    E, inliers = cv2.findEssentialMat(
        norm_pts0, norm_pts1, 
        focal=1.0, pp=(0, 0), 
        method=cv2.RANSAC, prob=0.999, threshold=0.005  # threshold in normalized coords
    )
    
    print(f"[DEBUG] calibrate_pair: fed {len(norm_pts0)} matched pts, got {np.sum(inliers)} inliers for E-mat")

    if E is None or E.shape != (3, 3):
        raise RuntimeError("Failed to compute a valid Essential Matrix.")
        
    # Decompose E to get R and t
    # recoverPose returns the number of inliers passing the cheirality check (must be in front of both cameras)
    # We MUST pass the inliers mask returned by findEssentialMat, otherwise it evaluates ALL points (including outliers)
    num_inliers, R, t, mask_pose = cv2.recoverPose(E, norm_pts0, norm_pts1, mask=inliers.copy())
    print(f"[DEBUG] calibrate_pair: recoverPose kept {num_inliers} points after cheirality check")
    
    # R is the rotation from Cam0 to Cam1. t is translation from Cam0 to Cam1.
    inlier_mask = (inliers.ravel() == 1) & (mask_pose.ravel() == 255)
    
    return R, t, inlier_mask


def calibrate_all(
    keypoints: np.ndarray,
    scores: np.ndarray,
    K_list: list[np.ndarray],
    min_conf: float = 0.65,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Solve the full N-camera system extrinsics using Cam0-Cam1 as the anchor baseline.
    
    Args:
        keypoints: (F, C, 133, 2)
        scores: (F, C, 133)
        K_list: list of N (3, 3) intrinsic matrices.
        
    Returns:
        List of N tuples (R, t) mapping world (Cam0) to each camera.
        Cam0 is defined as R=I, t=0.
    """
    num_frames, num_cameras, num_kpts, _ = keypoints.shape
    if num_cameras < 2:
        raise ValueError("At least 2 cameras are required for calibration.")
        
    # 1. Base case: Cam 0 is world origin
    extrinsics = []
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros((3, 1), dtype=np.float64)
    extrinsics.append((R0, t0))
    
    # 2. Cam 1 relative to Cam 0 (sets the absolute scale for the system to 1.0 unit)
    R1, t1, mask_01 = calibrate_pair(
        keypoints[:, 0], keypoints[:, 1],
        scores[:, 0], scores[:, 1],
        K_list[0], K_list[1],
        min_conf=min_conf
    )
    extrinsics.append((R1, t1))
    
    if num_cameras == 2:
        return extrinsics
        
    # 3. Triangulate anchor 3D points using Cam0 and Cam1
    # We will triangulate ALL body keypoints across ALL frames that meet the confidence threshold in C0 and C1
    P0 = K_list[0] @ np.hstack((R0, t0))
    P1 = K_list[1] @ np.hstack((R1, t1))
    
    anchor_pts3d = []
    anchor_indices = []  # To keep track of which (frame, keypoint_idx) it corresponds to
    
    for f in range(num_frames):
        for k_idx in _VALID_KPT_INDICES:
            if scores[f, 0, k_idx] >= min_conf and scores[f, 1, k_idx] >= min_conf:
                pts2d = np.array([keypoints[f, 0, k_idx], keypoints[f, 1, k_idx]])
                pt3d = triangulate_n_views(pts2d, [P0, P1])
                anchor_pts3d.append(pt3d)
                anchor_indices.append((f, k_idx))
                
    if len(anchor_pts3d) < 30:
        raise ValueError(f"Only {len(anchor_pts3d)} valid 3D points reconstructed from C0-C1. Need more walking data.")
        
    anchor_pts3d = np.array(anchor_pts3d, dtype=np.float64)
    
    # 4. Solve PnP for all remaining cameras
    for c in range(2, num_cameras):
        # Find 2D points in camera c that correspond to our 3D anchor points
        obj_pts = []
        img_pts = []
        
        for i, (f, k_idx) in enumerate(anchor_indices):
            if scores[f, c, k_idx] >= min_conf:
                obj_pts.append(anchor_pts3d[i])
                img_pts.append(keypoints[f, c, k_idx])
                
        if len(obj_pts) < 15:
            raise ValueError(f"Camera {c} only has {len(obj_pts)} confident matches with the C0-C1 anchor cloud.")
            
        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)
        
        # We pass empty distCoeffs because we assume ideal pinhole (distortion corrected or ignored)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K_list[c], distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0
        )
        
        if not success:
            raise RuntimeError(f"solvePnPRansac failed for camera {c}.")
            
        Rc, _ = cv2.Rodrigues(rvec)
        extrinsics.append((Rc, tvec))
        
    return extrinsics


def align_to_floor(
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    K_list: list[np.ndarray],
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_conf: float = 0.65,
) -> list[tuple[np.ndarray, np.ndarray]]:
    num_frames, num_cameras, _, _ = keypoints.shape
    P_list = [K_list[i] @ np.hstack((extrinsics[i][0], extrinsics[i][1])) for i in range(num_cameras)]

    foot_pts3d = []
    frame_floor_levels = []
    spine_vecs = []
    
    for f in range(num_frames):
        # 1. Triangulate feet for floor translation
        frame_foot_points = []
        for k_idx in _FOOT_KPT_INDICES:
            visible_cams = []
            valid_pts2d = []
            for c in range(num_cameras):
                if scores[f, c, k_idx] >= min_conf:
                    visible_cams.append(c)
                    valid_pts2d.append(keypoints[f, c, k_idx])
            if len(visible_cams) >= 2:
                P_sub = [P_list[c] for c in visible_cams]
                point = triangulate_n_views(np.array(valid_pts2d), P_sub)
                foot_pts3d.append(point)
                if np.isfinite(point).all():
                    frame_foot_points.append(point)
        if frame_foot_points:
            # Estimate one floor level per frame, then use the median across
            # time.  This is less sensitive to a single noisy toe than taking
            # the global mean of the upper quantile and keeps translation
            # consistent across the sequence.
            frame_floor_levels.append(float(np.percentile(np.asarray(frame_foot_points)[:, 1], 80)))
                
        # 2. Triangulate hips and shoulders for 'up' rotation
        pts3d = {}
        for k_idx in [5, 6, 11, 12]:
            visible_cams = []
            valid_pts2d = []
            for c in range(num_cameras):
                if scores[f, c, k_idx] >= min_conf:
                    visible_cams.append(c)
                    valid_pts2d.append(keypoints[f, c, k_idx])
            if len(visible_cams) >= 2:
                P_sub = [P_list[c] for c in visible_cams]
                pts3d[k_idx] = triangulate_n_views(np.array(valid_pts2d), P_sub)
                
        if 5 in pts3d and 6 in pts3d and 11 in pts3d and 12 in pts3d:
            mid_shoulder = (pts3d[5] + pts3d[6]) / 2.0
            mid_hip = (pts3d[11] + pts3d[12]) / 2.0
            vec = mid_shoulder - mid_hip
            norm = np.linalg.norm(vec)
            if norm > 1e-4:
                spine_vecs.append(vec / norm)
                
    if len(foot_pts3d) < 5:
        print("Warning: Not enough foot keypoints to detect floor plane. Assuming origin is (0,0,0).")
        centroid = np.zeros(3, dtype=np.float32)
    else:
        foot_pts3d = np.array(foot_pts3d)
        y_coords = foot_pts3d[:, 1]
        floor_y = float(np.median(frame_floor_levels)) if frame_floor_levels else float(np.percentile(y_coords, 80))
        y_tolerance = max(1e-3, 0.1 * float(np.std(y_coords)))
        floor_pts = foot_pts3d[np.abs(y_coords - floor_y) <= y_tolerance]
        if len(floor_pts) < 3:
            floor_pts = foot_pts3d[y_coords >= np.percentile(y_coords, 70)]
        centroid = np.median(floor_pts, axis=0)
        centroid[1] = floor_y
        
    if len(spine_vecs) < 5:
        print("Warning: Not enough spine keypoints to detect upright vector. Falling back to default.")
        normal = np.array([0, -1, 0], dtype=np.float32)
    else:
        avg_spine = np.mean(spine_vecs, axis=0)
        avg_spine /= np.linalg.norm(avg_spine)
        normal = avg_spine
        
    target_normal = np.array([0, -1, 0], dtype=np.float64)
    v = np.cross(normal, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(normal, target_normal)
    
    if s < 1e-6:
        R_align = np.eye(3)
    else:
        vX = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_align = np.eye(3) + vX + (vX @ vX) * ((1 - c) / (s ** 2))
        
    R_align_T = R_align.T
    
    aligned_extrinsics = []
    for R, t in extrinsics:
        R_new = R @ R_align_T
        t_new = R @ centroid.reshape(3, 1) + t
        aligned_extrinsics.append((R_new, t_new))
        
    return aligned_extrinsics
