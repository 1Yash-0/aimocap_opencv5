"""Checkerboard calibration recovers a known board pose from a synthetic image."""

from __future__ import annotations

import numpy as np
import cv2

from evaluation.opencap.calibration import (
    board_object_points,
    detect_board,
    estimate_board_pose,
    frame_candidates,
    rotation_angle_deg,
)

WIDTH, HEIGHT = 1280, 720
SQUARE_MM = 60.0
COLS, ROWS = 11, 8  # inner corners


def _render_board(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Render a 12 x 9 square board (60 mm squares) into a grayscale image by homography."""
    tex_square = 40
    cols, rows = COLS + 1, ROWS + 1
    texture = np.zeros((rows * tex_square, cols * tex_square), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            value = 0 if (r + c) % 2 == 0 else 255
            texture[r * tex_square:(r + 1) * tex_square, c * tex_square:(c + 1) * tex_square] = value
    # Board plane in millimetres, outer square edge one square beyond the inner corners.
    x0, y0 = -SQUARE_MM, -SQUARE_MM
    x1, y1 = cols * SQUARE_MM + x0, rows * SQUARE_MM + y0
    board_corners = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(board_corners, rvec, tvec, K, np.zeros(5))
    h, w = texture.shape
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, projected.reshape(4, 2).astype(np.float32))
    canvas = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, H, (WIDTH, HEIGHT), borderValue=128)
    mask = cv2.warpPerspective(np.ones_like(texture), H, (WIDTH, HEIGHT))
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def test_board_geometry_and_frame_order() -> None:
    obj = board_object_points()
    assert obj.shape == (COLS * ROWS, 3)
    assert np.allclose(obj[:, 2], 0.0)
    assert np.isclose(obj[1, 0] - obj[0, 0], SQUARE_MM)
    assert np.isclose(obj[COLS, 1] - obj[0, 1], SQUARE_MM)
    frames = frame_candidates(n_frames=200, fps=60.0)
    assert frames[0] == 200 - 18  # vidLength minus 0.3 s
    assert all(0 <= f <= 199 for f in frames)


def test_known_pose_is_recovered_within_half_a_degree() -> None:
    K = np.array([[900.0, 0.0, WIDTH / 2], [0.0, 900.0, HEIGHT / 2], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    rvec_true = np.array([0.10, -0.12, 0.04])
    tvec_true = np.array([[-180.0], [-120.0], [1500.0]])
    R_true, _ = cv2.Rodrigues(rvec_true)

    image = _render_board(K, rvec_true, tvec_true)
    corners, method = detect_board(image)
    assert corners is not None, "board not detected in the synthetic image"
    assert corners.shape == (COLS * ROWS, 2)
    assert method.startswith("classic")

    R, t, rms = estimate_board_pose(corners, K, dist)
    assert rotation_angle_deg(R, R_true) < 0.5
    assert np.linalg.norm(t - tvec_true) < 5.0  # millimetres
    assert rms < 1.0  # pixels
