"""Panoptic Dataset loader API.

Parses calibration data and 3D body pose ground truth from CMU Panoptic dataset.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np

@dataclass
class CameraCalibration:
    """Calibration parameters for a single camera."""
    name: str
    type: str  # e.g., 'hd', 'vga'
    resolution: tuple[int, int]
    K: np.ndarray  # (3, 3) Intrinsic matrix
    dist_coef: np.ndarray  # (5,) Distortion coefficients
    R: np.ndarray  # (3, 3) Rotation matrix
    t: np.ndarray  # (3, 1) Translation vector
    
    @property
    def extrinsics(self) -> np.ndarray:
        """Get the 3x4 extrinsic matrix [R | t]."""
        return np.hstack((self.R, self.t))

def load_calibration(json_path: str | Path) -> Dict[str, CameraCalibration]:
    """Load Panoptic camera calibration data from JSON.
    
    Args:
        json_path: Path to calibration_{seq}.json
        
    Returns:
        Dictionary mapping camera name (e.g. '00_00') to CameraCalibration.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    cameras = {}
    for cam in data.get('cameras', []):
        name = cam['name']
        cameras[name] = CameraCalibration(
            name=name,
            type=cam.get('type', ''),
            resolution=tuple(cam.get('resolution', [0, 0])),
            K=np.array(cam['K'], dtype=np.float32),
            dist_coef=np.array(cam.get('distCoef', []), dtype=np.float32),
            R=np.array(cam['R'], dtype=np.float32),
            t=np.array(cam['t'], dtype=np.float32)
        )
    return cameras

@dataclass
class BodyPose3D:
    """3D body pose for a single person in a frame."""
    id: int
    joints19: np.ndarray  # (19, 4) array of (x, y, z, confidence)

def load_3d_body_pose(json_path: str | Path) -> List[BodyPose3D]:
    """Load Panoptic 3D body pose (COCO-19 format) for a single frame.
    
    Args:
        json_path: Path to body3DScene_{frame:08d}.json
        
    Returns:
        List of BodyPose3D objects (one per person detected).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    bodies = []
    for b in data.get('bodies', []):
        joints = np.array(b['joints19'], dtype=np.float32).reshape(-1, 4)
        bodies.append(BodyPose3D(id=b.get('id', -1), joints19=joints))
    return bodies

def get_hd_video_paths(seq_dir: str | Path) -> List[Path]:
    """Get all sorted HD video paths for a downloaded Panoptic sequence."""
    seq_dir = Path(seq_dir)
    hd_dir = seq_dir / 'hdVideos'
    if not hd_dir.exists():
        return []
    
    paths = list(hd_dir.glob('hd_*.mp4'))
    paths.sort()
    return paths
