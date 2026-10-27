"""Rigorous tests for evidence-preserving robust triangulation.

Cameras are placed around the subject with proper OpenCV world-to-camera
(P = K [R | -R C]) so triangulation has real geometric baselines.  Each test
simulates detection noise or one broken view and asserts the solver recovers
the correct 3D point and reports the right provenance status.
"""

import numpy as np
import cv2
import pytest

from aimocap.geometry.robust_triangulate import (
    CameraObs, triangulate_joint_robust,
    STATUS_STRONG, STATUS_USABLE, STATUS_WEAK, STATUS_UNRESOLVED,
)


def _project(P, X):
    Xh = np.append(X, 1.0)
    p = P @ Xh
    if p[2] <= 1e-9:
        return None
    return p[:2] / p[2]


def _camera_at(pos, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, -1.0, 0.0]),
              focal=1400.0):
    """Build a right-handed OpenCV world-to-cam camera at `pos` looking at `target`."""
    pos = np.asarray(pos, float)
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    n = np.linalg.norm(right)
    if n < 1e-9:
        up = np.array([0.0, 0.0, -1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    else:
        right /= n
    true_up = np.cross(right, forward)
    R = np.stack([right, -true_up, forward])  # rows: x_c, y_c, z_c
    t = -R @ pos
    K = np.array([[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]])
    P = K @ np.hstack((R, t.reshape(3, 1)))
    return K, R, t, P


def _obs(uv, conf=0.99, noise=0.5):
    return CameraObs(point_uv=np.asarray(uv, float), confidence=conf,
                     noise_px=noise, is_finite=True)


def test_two_camera_perfect_recovers_point():
    c0 = _camera_at([4.0, 0.0, 0.0]); c1 = _camera_at([0.0, 4.0, 0.0])
    P0, P1 = c0[3], c1[3]
    X = np.array([0.2, -0.1, 0.7])
    u0, u1 = _project(P0, X), _project(P1, X)
    res = triangulate_joint_robust([_obs(u0), _obs(u1)], [P0, P1],
                                   [c0[0], c1[0]], max_accept_px=5.0)
    assert res.status in (STATUS_STRONG, STATUS_USABLE)
    assert np.linalg.norm(res.point3d - X) < 1e-3
    assert res.num_inliers == 2


def test_three_camera_outlier_rejected():
    c0 = _camera_at([4.0, 0.0, 0.0]); c1 = _camera_at([0.0, 4.0, 0.0]); c2 = _camera_at([-4.0, 0.0, 4.0])
    Ps = [c0[3], c1[3], c2[3]]; Ks = [c0[0], c1[0], c2[0]]
    X = np.array([0.0, 0.0, 0.5])
    uv = [_project(P, X) for P in Ps]
    obs = [_obs(uv[0]), _obs(uv[1]), _obs(uv[2] + np.array([80.0, 40.0]))]  # cam 2 disagrees
    res = triangulate_joint_robust(obs, Ps, Ks, max_accept_px=5.0)
    assert res.num_inliers == 2
    assert res.inlier_mask[0] and res.inlier_mask[1] and not res.inlier_mask[2]
    assert np.linalg.norm(res.point3d - X) < 1e-2


def test_single_camera_is_unresolved():
    c0 = _camera_at([4.0, 0.0, 0.0])
    X = np.array([0.0, 0.0, 0.5])
    uv = _project(c0[3], X)
    res = triangulate_joint_robust([_obs(uv), None, None], [c0[3], c0[3], c0[3]],
                                   [c0[0], c0[0], c0[0]])
    assert res.status == STATUS_UNRESOLVED
    assert res.num_inliers == 0
    assert not np.isfinite(res.point3d).all()


def test_status_strong_requires_three_inliers():
    c0 = _camera_at([4.0, 0.0, 0.0]); c1 = _camera_at([0.0, 4.0, 0.0]); c2 = _camera_at([-4.0, 0.0, 4.0])
    Ps = [c0[3], c1[3], c2[3]]; Ks = [c0[0], c1[0], c2[0]]
    X = np.array([0.0, 0.0, 0.5])
    uv = [_project(P, X) for P in Ps]
    # all three agree within 0.5px of noise
    obs = [_obs(uv[0] + np.array([0.2, 0.1])),
           _obs(uv[1] + np.array([-0.1, 0.2])),
           _obs(uv[2] + np.array([0.1, -0.2]))]
    res = triangulate_joint_robust(obs, Ps, Ks, max_accept_px=5.0)
    assert res.num_inliers == 3
    assert res.status == STATUS_STRONG
    assert np.linalg.norm(res.point3d - X) < 1e-3


def test_iterative_refinement_handles_noise_mixed_with_outlier():
    c0 = _camera_at([4.0, 0.0, 0.0]); c1 = _camera_at([0.0, 4.0, 0.0]); c2 = _camera_at([-4.0, 0.0, 4.0])
    Ps = [c0[3], c1[3], c2[3]]; Ks = [c0[0], c1[0], c2[0]]
    X = np.array([0.3, -0.2, 0.8])
    uvs = [_project(P, X) for P in Ps]
    obs = [_obs(uvs[0] + np.array([-0.4, 0.3]), noise=1.2),
           _obs(uvs[1] + np.array([0.2, -0.5]), noise=1.2),
           _obs(uvs[2] + np.array([40.0, -30.0]), noise=1.2)]
    res = triangulate_joint_robust(obs, Ps, Ks, max_accept_px=5.0)
    assert res.num_inliers == 2
    assert not res.inlier_mask[2]
    assert np.linalg.norm(res.point3d - X) < 0.02
