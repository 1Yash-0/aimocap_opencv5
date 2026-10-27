import numpy as np
import itertools

def compute_fundamental_matrix(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray, 
                               K2: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute the Fundamental matrix mapping points from camera 1 to camera 2.
    """
    # P = K[R|t]
    P1 = K1 @ np.hstack((R1, t1))
    P2 = K2 @ np.hstack((R2, t2))
    
    # Camera center of 1
    C1 = -R1.T @ t1
    C1_hom = np.vstack((C1, [[1]]))
    
    # Epipole in cam 2
    e2 = P2 @ C1_hom
    e2_x = np.array([
        [0, -e2[2, 0], e2[1, 0]],
        [e2[2, 0], 0, -e2[0, 0]],
        [-e2[1, 0], e2[0, 0], 0]
    ])
    
    # P1 pseudo-inverse
    P1_pinv = np.linalg.pinv(P1)
    
    F = e2_x @ P2 @ P1_pinv
    # Normalize
    F = F / np.linalg.norm(F)
    return F

def get_fundamental_matrices(K_list: list[np.ndarray], extrinsics: list[tuple[np.ndarray, np.ndarray]]) -> dict[tuple[int, int], np.ndarray]:
    """
    Generate all pairwise fundamental matrices for a set of cameras.
    """
    num_cameras = len(K_list)
    F_matrices = {}
    for c1, c2 in itertools.combinations(range(num_cameras), 2):
        K1 = K_list[c1]
        R1, t1 = extrinsics[c1]
        K2 = K_list[c2]
        R2, t2 = extrinsics[c2]
        
        F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)
        F_matrices[(c1, c2)] = F
    
    return F_matrices
