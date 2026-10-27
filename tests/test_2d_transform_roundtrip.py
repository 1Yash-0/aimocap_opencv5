import numpy as np
from aimocap.pose.transforms import get_crop_affine_transform, apply_affine_transform

def test_transform_roundtrip():
    # Frame size
    frame_w, frame_h = 1920, 1080
    
    # Network input size
    input_w, input_h = 288, 384
    
    # Some bounding box
    bbox = [100.0, 50.0, 400.0, 900.0]
    
    M, M_inv, crop = get_crop_affine_transform(bbox, input_w, input_h, frame_w, frame_h)
    
    # Check that M_inv @ M is identity
    I = M_inv @ M
    assert np.allclose(I, np.eye(3), atol=1e-5), f"Inverse matrix check failed:\n{I}"
    
    # Generate some synthetic 2D points inside the bounding box
    pts_original = np.array([
        [150.0, 200.0],
        [300.0, 500.0],
        [200.0, 800.0],
        [250.0, 450.0]
    ])
    
    # Map to network crop
    pts_network = apply_affine_transform(pts_original, M)
    
    # Map back to full frame
    pts_restored = apply_affine_transform(pts_network, M_inv)
    
    # Assert sub-pixel reconstruction precision
    max_error = np.max(np.abs(pts_original - pts_restored))
    assert max_error < 1e-3, f"Roundtrip error too high: {max_error}"

def test_clipped_bbox_transform():
    frame_w, frame_h = 640, 480
    input_w, input_h = 288, 384
    
    # Bbox partially out of bounds (should get clipped)
    bbox = [-100.0, -50.0, 300.0, 500.0]
    
    M, M_inv, crop = get_crop_affine_transform(bbox, input_w, input_h, frame_w, frame_h)
    
    pts_original = np.array([
        [50.0, 100.0],
        [200.0, 400.0]
    ])
    
    pts_network = apply_affine_transform(pts_original, M)
    pts_restored = apply_affine_transform(pts_network, M_inv)
    
    assert np.allclose(pts_original, pts_restored, atol=1e-3)
