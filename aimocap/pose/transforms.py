import numpy as np

def get_crop_affine_transform(
    bbox: list[float] | np.ndarray, 
    input_w: int, 
    input_h: int, 
    frame_w: int, 
    frame_h: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Computes the 3x3 homogeneous affine transformation matrix M mapping full-frame
    coordinates to network-crop coordinates, and its exact inverse M_inv.
    
    This matches the `cigpose.preprocess_person` bounding box expansion and clipping logic.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = x2 - x1, y2 - y1

    aspect = input_w / float(input_h)
    if bw / max(bh, 1.0) > aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
        
    bw *= 1.25
    bh *= 1.25

    sx1 = int(max(0, cx - bw / 2.0))
    sy1 = int(max(0, cy - bh / 2.0))
    sx2 = int(min(frame_w, cx + bw / 2.0))
    sy2 = int(min(frame_h, cy + bh / 2.0))

    if sx2 <= sx1 or sy2 <= sy1:
        # Fallback if crop is completely invalid
        sx1, sy1, sx2, sy2 = 0, 0, frame_w, frame_h

    crop_w = sx2 - sx1
    crop_h = sy2 - sy1

    S_x = input_w / float(crop_w)
    S_y = input_h / float(crop_h)

    # M maps from full frame to network crop
    M = np.array([
        [S_x, 0.0, -sx1 * S_x],
        [0.0, S_y, -sy1 * S_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # M_inv maps from network crop to full frame
    M_inv = np.array([
        [1.0 / S_x, 0.0, sx1],
        [0.0, 1.0 / S_y, sy1],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return M, M_inv, (sx1, sy1, sx2, sy2)


def apply_affine_transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 homogeneous affine transformation matrix M to an array of 2D points.
    
    Args:
        pts: (N, 2) array of 2D coordinates.
        M: (3, 3) affine transformation matrix.
        
    Returns:
        (N, 2) array of transformed 2D coordinates.
    """
    N = pts.shape[0]
    pts_homo = np.concatenate([pts, np.ones((N, 1), dtype=pts.dtype)], axis=-1)
    # (N, 3) @ (3, 3).T -> (N, 3)
    transformed = pts_homo @ M.T
    return transformed[:, :2]

