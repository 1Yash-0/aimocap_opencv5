import numpy as np
from aimocap.calib.extrinsics import calibrate_all, align_to_floor
from aimocap.triangulate.engine import triangulate_sequence
from aimocap.math.metrics import compute_bone_variance

def focal_grid_search(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_sizes: list[tuple[int, int]],
    min_conf: float = 0.5,
    min_focal_ratio: float = 0.6,
    max_focal_ratio: float = 1.5,
    steps: int = 10,
    stride: int = 5,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], float]:
    """
    Search for the optimal focal length by minimizing bone-length variance.
    
    Args:
        keypoints: (F, C, 133, 2)
        scores: (F, C, 133)
        image_sizes: List of (width, height) for each camera
        min_conf: Minimum confidence for keypoints
        min_focal_ratio: Minimum ratio of max(w,h) to search
        max_focal_ratio: Maximum ratio of max(w,h) to search
        steps: Number of focal lengths to try
        stride: Frame stride to speed up search (evaluate every Nth frame)
        
    Returns:
        best_K_list, best_extrinsics, best_focal_ratio
    """
    num_frames = keypoints.shape[0]
    
    # Subsample frames for speed
    if stride > 1:
        search_keypoints = keypoints[::stride]
        search_scores = scores[::stride]
    else:
        search_keypoints = keypoints
        search_scores = scores

    focal_ratios = np.linspace(min_focal_ratio, max_focal_ratio, steps)
    
    best_variance = float('inf')
    best_K_list = None
    best_extrinsics = None
    best_ratio = None
    
    print(f"Starting focal length grid search ({steps} steps from {min_focal_ratio:.2f}x to {max_focal_ratio:.2f}x)...")
    
    for ratio in focal_ratios:
        # 1. Build K matrices for this candidate ratio
        K_list = []
        for w, h in image_sizes:
            # Similar to guess_intrinsics but with our specific ratio
            f = max(w, h) * ratio
            cx, cy = w / 2.0, h / 2.0
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1]
            ], dtype=np.float32)
            K_list.append(K)
            
        try:
            # 2. Calibrate extrinsics
            extrinsics = calibrate_all(search_keypoints, search_scores, K_list, min_conf=min_conf)
            
            # 3. Align to floor
            extrinsics = align_to_floor(extrinsics, K_list, search_keypoints, search_scores, min_conf=min_conf)
            
            # 4. Triangulate
            skeleton3d = triangulate_sequence(search_keypoints, search_scores, K_list, extrinsics, min_conf=min_conf)
            
            # 5. Score
            variance = compute_bone_variance(skeleton3d)
            print(f"  Ratio {ratio:.3f}x -> Bone Variance: {variance:.5f}")
            
            if variance < best_variance:
                best_variance = variance
                best_ratio = ratio
                
        except Exception as e:
            # Calibration might fail for some extreme focal lengths
            print(f"  Ratio {ratio:.3f}x -> Failed: {e}")
            
    if best_ratio is None:
        raise RuntimeError("Focal grid search failed to find any valid calibration.")
        
    print(f"Grid search complete. Best focal ratio: {best_ratio:.3f}x (variance: {best_variance:.5f})")
    
    # Rerun full calibration on all frames with the best K
    K_list = []
    for w, h in image_sizes:
        f = max(w, h) * best_ratio
        cx, cy = w / 2.0, h / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1]
        ], dtype=np.float32)
        K_list.append(K)
        
    print("Re-calibrating full sequence with best focal length...")
    extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=min_conf)
    extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=min_conf)
    
    return K_list, extrinsics, best_ratio
