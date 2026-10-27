import os
from pathlib import Path
import pytest
import numpy as np

from aimocap.data.panoptic import (
    load_calibration,
    load_3d_body_pose,
    get_hd_video_paths,
)
from aimocap.math.coords import opencv_to_internal, internal_to_opencv

# We assume that the user has run:
# python scripts/fetch_panoptic.py --num-hd 3
# and the data resides in `data/panoptic/171204_pose1`.
PANOPTIC_DATA_DIR = Path("data/panoptic/171204_pose1")

@pytest.mark.skipif(not PANOPTIC_DATA_DIR.exists(), reason="Panoptic dataset not downloaded")
def test_load_calibration():
    calib_json = PANOPTIC_DATA_DIR / "calibration_171204_pose1.json"
    assert calib_json.exists(), "Calibration JSON is missing"
    
    cameras = load_calibration(calib_json)
    assert len(cameras) > 0, "No cameras loaded"
    
    # Check structure of the first camera
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    assert cam.name == cam_id
    assert cam.K.shape == (3, 3)
    assert cam.R.shape == (3, 3)
    assert cam.t.shape == (3, 1)
    assert cam.extrinsics.shape == (3, 4)
    assert cam.dist_coef.shape[0] >= 5

@pytest.mark.skipif(not PANOPTIC_DATA_DIR.exists(), reason="Panoptic dataset not downloaded")
def test_load_3d_body_pose():
    # The tar file extracts to hdPose3d_stage1_coco19
    pose_dir = PANOPTIC_DATA_DIR / "hdPose3d_stage1_coco19"
    if not pose_dir.exists():
        pytest.skip("3D pose directory not extracted yet")
        
    pose_files = list(pose_dir.glob("body3DScene_*.json"))
    assert len(pose_files) > 0, "No 3D pose files found"
    
    # Check the first frame
    pose_file = pose_files[0]
    bodies = load_3d_body_pose(pose_file)
    
    # Sequence 171204_pose1 has at least 1 person
    assert len(bodies) > 0, "No bodies loaded in frame 0"
    body = bodies[0]
    assert body.joints19.shape == (19, 4), "COCO-19 should have 19 joints of (x,y,z,conf)"

@pytest.mark.skipif(not PANOPTIC_DATA_DIR.exists(), reason="Panoptic dataset not downloaded")
def test_get_hd_video_paths():
    paths = get_hd_video_paths(PANOPTIC_DATA_DIR)
    assert len(paths) >= 3, "Expected at least 3 HD videos downloaded"
    assert paths[0].exists()

def test_coordinate_conversion():
    # Test opencv to internal and back
    # OpenCV: X=1, Y=2 (down), Z=3 (forward)
    pt_cv = np.array([1.0, 2.0, 3.0])
    pt_int = opencv_to_internal(pt_cv)
    
    # Internal: X=1, Y=-2 (up), Z=-3 (back)
    np.testing.assert_array_equal(pt_int, np.array([1.0, -2.0, -3.0]))
    
    # Back to OpenCV
    pt_cv2 = internal_to_opencv(pt_int)
    np.testing.assert_array_equal(pt_cv, pt_cv2)
