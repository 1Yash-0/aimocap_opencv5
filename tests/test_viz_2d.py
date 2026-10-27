import os
import pytest
from pathlib import Path
from aimocap.video import run_video

try:
    from cigpose import load_pose_model  # noqa: F401  (what PoseEstimator needs)
except ImportError:
    load_pose_model = None

def test_viz_2d_overlay():
    if load_pose_model is None:
        pytest.skip("cigpose not installed")
    src = Path("data/panoptic/171204_pose1/hdVideos/hd_00_00.mp4")
    if not src.exists():
        pytest.skip(f"Test video not found: {src}")
    
    out_dir = Path("outputs/visual_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "test_2d_overlay.mp4"
    
    # Run pose estimation for 30 frames
    stats = run_video(
        src=src,
        dst=dst,
        max_frames=30,
        threshold=0.3
    )
    
    assert dst.exists()
    assert len(stats) == 30
