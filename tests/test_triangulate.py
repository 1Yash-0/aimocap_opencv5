import pytest
import numpy as np
from pathlib import Path

from aimocap.data.panoptic import load_calibration, load_3d_body_pose
from aimocap.math.triangulate import triangulate_n_views, project_points

PANOPTIC_DATA_DIR = Path("data/panoptic/171204_pose1")

@pytest.mark.skipif(not PANOPTIC_DATA_DIR.exists(), reason="Panoptic dataset not downloaded")
def test_triangulation_accuracy_against_gt():
    """
    Test the DLT triangulation using Panoptic ground truth.
    1. Load Panoptic Calibration.
    2. Load Panoptic 3D Body Pose (Frame 0).
    3. Take a 3D joint, project it into 5 random cameras to get 'perfect' 2D points.
    4. Pass those 2D points into triangulate_n_views().
    5. Assert the output 3D point matches the original GT 3D joint within 1mm.
    """
    calib_json = PANOPTIC_DATA_DIR / "calibration_171204_pose1.json"
    pose_dir = PANOPTIC_DATA_DIR / "hdPose3d_stage1_coco19"
    pose_files = list(pose_dir.glob("body3DScene_*.json"))
    assert len(pose_files) > 0, "No pose files found"
    pose_file = pose_files[0]
    
    assert calib_json.exists(), "Calibration missing"
    
    cameras = load_calibration(calib_json)
    bodies = load_3d_body_pose(pose_file)
    
    # Get the 3D position of the first joint (Nose) of the first person
    gt_point_cv = bodies[0].joints19[0, :3] # (3,)
    
    # Pick 5 random HD cameras
    cam_ids = [k for k in cameras.keys() if k.startswith("00_")]
    selected_cams = cam_ids[:5]
    
    pts2d = []
    proj_matrices = []
    
    # Project the GT point into each camera
    for cam_id in selected_cams:
        cam = cameras[cam_id]
        # P = K[R|t]
        P = cam.K @ cam.extrinsics
        
        # Project
        pt2d = project_points(np.array([gt_point_cv]), P)[0]
        
        pts2d.append(pt2d)
        proj_matrices.append(P)
        
    pts2d = np.array(pts2d)
    
    # Triangulate
    triangulated_pt = triangulate_n_views(pts2d, proj_matrices)
    
    # Assert they are extremely close (sub-millimeter since inputs are perfect projections)
    # The math should be perfect. Let's say < 1e-4 meters (0.1mm)
    error = np.linalg.norm(gt_point_cv - triangulated_pt)
    assert error < 1e-4, f"Triangulation error {error} is too high"
