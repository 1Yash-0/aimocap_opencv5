"""Test the M2B Markerless Auto-Calibration math using synthetic data."""

from __future__ import annotations

import cv2
import numpy as np

from aimocap.calib.extrinsics import calibrate_all, align_to_floor
from aimocap.math.triangulate import project_points
from aimocap.pose.keypoints import BODY_17, FEET_6

def test_auto_calibration():
    np.random.seed(42)
    # 1. Define synthetic ground truth
    # Camera 0: Origin, looking slightly down at the scene
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    
    # Camera 1: Rotated 45 degrees around Y, translated
    theta1 = np.deg2rad(45)
    R1 = np.array([
        [np.cos(theta1), 0, np.sin(theta1)],
        [0, 1, 0],
        [-np.sin(theta1), 0, np.cos(theta1)]
    ])
    t1 = np.array([[-1.0], [0.0], [1.0]])
    
    # Camera 2: Rotated -45 degrees around Y
    theta2 = np.deg2rad(-45)
    R2 = np.array([
        [np.cos(theta2), 0, np.sin(theta2)],
        [0, 1, 0],
        [-np.sin(theta2), 0, np.cos(theta2)]
    ])
    t2 = np.array([[1.0], [0.0], [1.0]])
    
    gt_cameras = [(R0, t0), (R1, t1), (R2, t2)]
    
    # Intrinsics
    K = np.array([
        [1500, 0, 960],
        [0, 1500, 540],
        [0, 0, 1]
    ], dtype=np.float64)
    K_list = [K, K, K]
    
    # 2. Generate synthetic walking data (3D)
    num_frames = 100
    num_kpts = 133
    num_cameras = 3
    
    # We create a random cloud of points that loosely resembles human movement
    # We will ensure the foot points lie roughly on Y=1 (since OpenCV is Y-down, Y=1 is "floor" below origin Y=0)
    # Actually, let's just make Y=2 the floor.
    
    # To pass `calibrate_all`, we only need points in `BODY_17`. 
    # To pass `align_to_floor`, we need points in `_FOOT_KPT_INDICES` (15, 16, 17-22).
    
    pts3d_all = np.zeros((num_frames, num_kpts, 3))
    
    # Center of walking area is (0, 0, 5) relative to Cam0
    for f in range(num_frames):
        # General body keypoints
        for k in BODY_17:
            pts3d_all[f, k] = [np.random.randn(), 1.0, 5 + np.random.randn()]
            
        # Shoulders (Y = 0.5, higher up)
        for k in [5, 6]:
            pts3d_all[f, k] = [np.random.randn(), 0.5, 5 + np.random.randn()]
            
        # Hips (Y = 1.5, lower down)
        for k in [11, 12]:
            pts3d_all[f, k] = [np.random.randn(), 1.5, 5 + np.random.randn()]
            
        # Feet (Y = 2.0 exactly)
        for k in [15, 16] + list(FEET_6):
            pts3d_all[f, k] = [np.random.randn(), 2.0, 5 + np.random.randn()]
            
    # 3. Project to 2D
    keypoints = np.zeros((num_frames, num_cameras, num_kpts, 2))
    scores = np.ones((num_frames, num_cameras, num_kpts)) # perfect confidence
    
    for c in range(num_cameras):
        P = K @ np.hstack(gt_cameras[c])
        for f in range(num_frames):
            pts2d = project_points(pts3d_all[f], P)
            keypoints[f, c] = pts2d
            
    # Add tiny noise to avoid perfect degeneracies in RANSAC
    keypoints += np.random.normal(0, 0.1, keypoints.shape)
    
    # 4. Calibrate
    extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=0.5)
    
    assert len(extrinsics) == 3
    
    # 5. Check relative rotation (Scale is arbitrary, but rotations should match exactly)
    # Relative rotation from Cam0 to Cam1 is R1 @ R0.T
    gt_R01 = R1 @ R0.T
    est_R01 = extrinsics[1][0] @ extrinsics[0][0].T
    
    # Check rotation difference
    diff_R = gt_R01 @ est_R01.T
    angle = np.arccos(np.clip((np.trace(diff_R) - 1) / 2, -1.0, 1.0))
    angle_deg = np.rad2deg(angle)
    
    assert angle_deg < 5.0, f"Rotation error too high: {angle_deg} degrees"
    
    # 6. Align to floor
    # Before alignment, Y was down, floor was at Y=2.
    # After alignment, the floor should be at Y=0 and normal should be -Y (up).
    aligned = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=0.5)
    
    # We will test if the triangulated foot points in the new world actually lie at Y=0
    # Re-triangulate using the new cameras
    P_aligned = [K_list[c] @ np.hstack(aligned[c]) for c in range(num_cameras)]
    
    # Test frame 0, foot 15
    test_pts2d = np.array([keypoints[0, c, 15] for c in range(num_cameras)])
    from aimocap.math.triangulate import triangulate_n_views
    pt3d_aligned = triangulate_n_views(test_pts2d, P_aligned)
    
    # The new Y coordinate should be very close to 0 (since it's on the floor)
    assert abs(pt3d_aligned[1]) < 0.1, f"Floor alignment failed, foot Y = {pt3d_aligned[1]}"

if __name__ == "__main__":
    test_auto_calibration()
    print("All auto calibration tests passed!")
