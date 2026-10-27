"""Test the M3 3D Triangulation engine using synthetic data."""

from __future__ import annotations

import numpy as np

from aimocap.triangulate.engine import triangulate_sequence, triangulate_sequence_with_diagnostics
from aimocap.math.triangulate import project_points


def test_triangulate_engine():
    # 1. Define synthetic ground truth
    num_frames = 10
    num_kpts = 133
    num_cameras = 3
    
    # We will create true 3D points in the Y-up internal coordinate space.
    gt_pts3d = np.random.randn(num_frames, num_kpts, 3) * 2.0
    
    # Define 3 cameras in OpenCV space (Y-down).
    # Since triangulate_sequence takes OpenCV extrinsics, we will construct P accordingly.
    # Cam 0
    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    
    # Cam 1
    R1 = np.eye(3)
    t1 = np.array([[-1.0], [0.0], [1.0]])
    
    # Cam 2
    R2 = np.eye(3)
    t2 = np.array([[1.0], [0.0], [1.0]])
    
    extrinsics = [(R0, t0), (R1, t1), (R2, t2)]
    
    # Intrinsics
    K = np.array([
        [1500, 0, 960],
        [0, 1500, 540],
        [0, 0, 1]
    ], dtype=np.float64)
    K_list = [K, K, K]
    
    # 2. Generate synthetic 2D observations
    # Before projecting, we must convert our Y-up ground truth into Y-down OpenCV space
    gt_pts3d_cv = np.copy(gt_pts3d)
    gt_pts3d_cv[..., 1] = -gt_pts3d_cv[..., 1]
    gt_pts3d_cv[..., 2] = -gt_pts3d_cv[..., 2]
    
    keypoints = np.zeros((num_frames, num_cameras, num_kpts, 2))
    scores = np.ones((num_frames, num_cameras, num_kpts)) # perfect confidence
    
    for c in range(num_cameras):
        P = K_list[c] @ np.hstack(extrinsics[c])
        for f in range(num_frames):
            pts2d = project_points(gt_pts3d_cv[f], P)
            keypoints[f, c] = pts2d
            
    # 3. Add artificial occlusion to test robustness
    # Make keypoint 0 invisible in camera 0 on frame 0
    scores[0, 0, 0] = 0.0
    
    # Make keypoint 1 invisible in camera 0 AND camera 1 on frame 0 (should yield NaN)
    scores[0, 0, 1] = 0.0
    scores[0, 1, 1] = 0.0
    
    # 4. Triangulate
    pred_pts3d = triangulate_sequence(keypoints, scores, K_list, extrinsics, min_conf=0.5, min_aspect_ratio=0.0)
    
    # 5. Verify
    # The output should be in Y-up internal space, matching `gt_pts3d`.
    assert pred_pts3d.shape == (num_frames, num_kpts, 3)
    
    # Keypoint 0 on frame 0 should still be reconstructed accurately from the other 2 cameras
    assert not np.isnan(pred_pts3d[0, 0]).any()
    dist = np.linalg.norm(pred_pts3d[0, 0] - gt_pts3d[0, 0])
    assert dist < 1e-5, f"Reconstruction failed for partially occluded point. Error: {dist}"
    
    # Keypoint 1 on frame 0 should be NaN because it has < 2 cameras
    assert np.isnan(pred_pts3d[0, 1]).all(), "Point with < 2 cameras should be NaN."
    
    # Frame 1 should be perfectly reconstructed for all points
    valid_mask = ~np.isnan(pred_pts3d[1])
    assert valid_mask.all()
    
    max_error = np.max(np.linalg.norm(pred_pts3d[1] - gt_pts3d[1], axis=-1))
    assert max_error < 1e-5, f"Reconstruction error too high: {max_error}"

if __name__ == "__main__":
    test_triangulate_engine()
    print("All triangulate engine tests passed!")


def test_triangulate_engine_rejects_bad_third_view():
    """A gross outlier view should be marked outlier and not drag the 3D point."""
    gt_internal = np.array([[[0.15, 0.25, -3.0]]], dtype=np.float64)
    gt_cv = np.copy(gt_internal)
    gt_cv[..., 1] = -gt_cv[..., 1]
    gt_cv[..., 2] = -gt_cv[..., 2]

    K = np.array([[1000, 0, 500], [0, 1000, 400], [0, 0, 1]], dtype=np.float64)
    extrinsics = [
        (np.eye(3), np.zeros((3, 1))),
        (np.eye(3), np.array([[-0.6], [0.0], [0.0]])),
        (np.eye(3), np.array([[0.6], [0.0], [0.0]])),
    ]
    K_list = [K, K, K]
    keypoints = np.zeros((1, 3, 1, 2), dtype=np.float64)
    for c, ext in enumerate(extrinsics):
        P = K @ np.hstack(ext)
        keypoints[0, c] = project_points(gt_cv[0], P)
    keypoints[0, 2, 0] += np.array([500.0, -350.0])
    scores = np.ones((1, 3, 1), dtype=np.float64)

    diag = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        K_list,
        extrinsics,
        min_conf=0.5,
        reproj_threshold_px=10.0,
        min_aspect_ratio=0.0,
    )

    err = np.linalg.norm(diag.points3d[0, 0] - gt_internal[0, 0])
    assert err < 1e-3
    assert diag.num_inliers[0, 0] == 2
    assert not diag.inlier_mask[0, 0, 2]
    assert diag.reprojection_error_px[0, 0, 2] > 100.0
