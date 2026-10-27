import numpy as np
from scipy.optimize import least_squares

def triangulate_n_views(pts2d: np.ndarray, proj_matrices: list[np.ndarray]) -> np.ndarray:
    """
    Triangulate a 3D point from N views using DLT.
    
    Args:
        pts2d: (N, 2) array of 2D coordinates (u, v) for the same point in N views.
               If a view is missing the point, it should NOT be included in these arrays.
        proj_matrices: list of N (3, 4) projection matrices P = K[R|t].
        
    Returns:
        (3,) array of the triangulated 3D point in the coordinate system of the projection matrices (OpenCV).
    """
    N = len(proj_matrices)
    if N < 2:
        raise ValueError("At least 2 views are required for triangulation.")
    
    A = np.zeros((2 * N, 4))
    
    for i in range(N):
        u, v = pts2d[i]
        P = proj_matrices[i]
        
        # u * P[2,:] - P[0,:]
        A[2*i]   = u * P[2, :] - P[0, :]
        # v * P[2,:] - P[1,:]
        A[2*i+1] = v * P[2, :] - P[1, :]
        
    # SVD
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    
    # Normalize homogeneous coordinates
    X = X / X[3]
    return X[:3]

def triangulate_weighted_dlt(pts2d: np.ndarray, proj_matrices: list[np.ndarray], confidences: np.ndarray) -> np.ndarray:
    """
    Triangulate a 3D point using Confidence-Weighted DLT.
    """
    N = len(proj_matrices)
    if N < 2:
        raise ValueError("At least 2 views are required for triangulation.")
    
    A = np.zeros((2 * N, 4))
    for i in range(N):
        u, v = pts2d[i]
        P = proj_matrices[i]
        c = confidences[i]
        A[2*i]   = c * (u * P[2, :] - P[0, :])
        A[2*i+1] = c * (v * P[2, :] - P[1, :])
        
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    return X[:3] / X[3]

def _reprojection_residuals(x3d, pts2d, proj_matrices, confidences):
    X = np.append(x3d, 1.0)
    residuals = []
    for i in range(len(proj_matrices)):
        P = proj_matrices[i]
        proj = P @ X
        proj = proj[:2] / proj[2]
        # Weight residual by confidence
        r = confidences[i] * (proj - pts2d[i])
        residuals.extend(r)
    return np.array(residuals)

def triangulate_robust(
    pts2d: np.ndarray,
    proj_matrices: list[np.ndarray],
    confidences: np.ndarray,
    f_scale: float = 10.0,
    loss: str = "huber",
) -> np.ndarray:
    """
    Triangulate a 3D point robustly using Weighted DLT initialization followed by
    nonlinear refinement with a Huber loss to reject outliers.
    """
    if len(proj_matrices) < 2:
        raise ValueError("At least 2 views are required.")
        
    if loss not in {"linear", "huber", "soft_l1", "cauchy", "arctan"}:
        raise ValueError(f"unsupported robust loss: {loss}")
    # 1. Init with weighted DLT
    x0 = triangulate_weighted_dlt(pts2d, proj_matrices, confidences)
    
    # 2. Non-linear refinement with Huber loss
    res = least_squares(
        _reprojection_residuals, 
        x0, 
        args=(pts2d, proj_matrices, confidences),
        method='trf',
        loss=loss,
        f_scale=f_scale  # Huber threshold in weighted pixels
    )
    return res.x

def project_points(pts3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Project 3D points into a camera using its projection matrix P.
    
    Args:
        pts3d: (N, 3) array of 3D points.
        P: (3, 4) projection matrix K[R|t].
        
    Returns:
        (N, 2) array of 2D image coordinates.
    """
    # Make homogeneous (N, 4)
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    
    # Project (3, 4) @ (4, N) = (3, N)
    pts2d_h = (P @ pts3d_h.T).T
    
    # Normalize by Z
    pts2d = pts2d_h[:, :2] / pts2d_h[:, 2:3]
    return pts2d
