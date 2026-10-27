import numpy as np
import pytest
from pathlib import Path
from aimocap.viz.plot3d import plot_scene

def test_viz_3d_skeleton():
    npz_path = Path("outputs/test_bone_constrained.npz")
    if not npz_path.exists():
        pytest.skip(f"Test data not found: {npz_path}")
        
    data = np.load(npz_path)
    skeleton3d = data["skeleton3d"]
    
    # Take a 30-frame slice
    skeleton3d = skeleton3d[0:30]
    
    out_dir = Path("outputs/visual_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "test_3d_skeleton.gif"
    
    # We can pass empty extrinsics if they aren't available
    extrinsics = []
    
    plot_scene(
        extrinsics=extrinsics,
        skeleton3d=skeleton3d,
        output_path=dst,
        animate=True
    )
    
    assert dst.exists()
