import numpy as np
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResidualStats:
    camera_id: str
    mean_vector: list[float]  # [x, y]
    cov_matrix: list[list[float]] # 2x2
    median_magnitude: float
    p95_magnitude: float
    
def compute_residual_vectors(
    pts2d: np.ndarray,      # (F, J, C, 2)
    pts3d: np.ndarray,      # (F, J, 3) in camera space (before R/t)? Wait, Triangulation engine takes P_list
    P_list: list[np.ndarray], # num_cameras length list of 3x4 projection matrices
    valid_mask: np.ndarray  # (F, J, C)
) -> np.ndarray:
    """
    Compute 2D reprojection residual vectors (obs - proj)
    
    Args:
        pts2d: Observed 2D keypoints (F, J, C, 2)
        pts3d: Triangulated 3D points (F, J, 3) in the same world space as P_list expects
        P_list: Projection matrices [K @ [R|t]]
        valid_mask: Mask of valid observations
        
    Returns:
        (F, J, C, 2) array of residual vectors. Invalid entries are NaN.
    """
    F_frames, J_joints, C_cams, _ = pts2d.shape
    residuals = np.full((F_frames, J_joints, C_cams, 2), np.nan, dtype=np.float32)
    
    # We need to project pts3d into each camera
    # pts3d is (F, J, 3)
    pts3d_h = np.concatenate([pts3d, np.ones((F_frames, J_joints, 1))], axis=-1) # (F, J, 4)
    
    for c in range(C_cams):
        P = P_list[c]
        # proj_h: (F, J, 3) = (F, J, 4) @ (4, 3)
        proj_h = pts3d_h @ P.T
        
        # normalize
        z = proj_h[..., 2:3]
        # avoid division by zero
        z[z == 0] = 1e-8
        proj_2d = proj_h[..., :2] / z
        
        # calculate vectors
        diff = pts2d[:, :, c, :] - proj_2d
        
        # only keep valid
        mask = valid_mask[:, :, c]
        residuals[mask, c, :] = diff[mask]
        
    return residuals

def analyze_residuals(residuals: np.ndarray, camera_names: list[str]) -> dict[str, ResidualStats]:
    """
    Compute vector statistics of residuals per camera.
    """
    C_cams = residuals.shape[2]
    stats = {}
    
    for c in range(C_cams):
        cam_res = residuals[:, :, c, :] # (F, J, 2)
        valid_res = cam_res[~np.isnan(cam_res).any(axis=-1)] # (N, 2)
        
        if len(valid_res) == 0:
            stats[camera_names[c]] = ResidualStats(
                camera_id=camera_names[c],
                mean_vector=[0.0, 0.0],
                cov_matrix=[[0.0, 0.0], [0.0, 0.0]],
                median_magnitude=0.0,
                p95_magnitude=0.0
            )
            continue
            
        mean_vec = np.mean(valid_res, axis=0)
        cov = np.cov(valid_res, rowvar=False) if len(valid_res) > 1 else np.zeros((2, 2))
        
        mags = np.linalg.norm(valid_res, axis=1)
        med_mag = float(np.median(mags))
        p95_mag = float(np.percentile(mags, 95))
        
        stats[camera_names[c]] = ResidualStats(
            camera_id=camera_names[c],
            mean_vector=mean_vec.tolist(),
            cov_matrix=cov.tolist(),
            median_magnitude=med_mag,
            p95_magnitude=p95_mag
        )
        
    return stats

def save_residual_diagnostics(stats: dict[str, ResidualStats], filepath: Path | str):
    data = {cam: vars(stat) for cam, stat in stats.items()}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
