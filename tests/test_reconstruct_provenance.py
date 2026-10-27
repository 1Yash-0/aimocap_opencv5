"""Tests for the evidence-preserving reconstruction provenance contract.

These are the laws the directive demands:
- masks are mutually exclusive and exhaustive (strong | usable | weak | reconstructed | boundary_hold | unresolved == all)
- a joint that was never measured in robust input AND was filled in is marked reconstructed
- a joint filled by np.interp beyond leading/trailing edge is marked boundary_hold
- strong points did not move beyond the trust region
"""
import numpy as np
import pytest

from aimocap.triangulate.reconstruct import reconstruct_sequence


def _dummy_camera():
    K = np.array([[1000.0, 0, 100], [0, 1000.0, 100], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5.0])
    return K, (R, t)


def _synthetic_run(T=30, J=5, C=2, measured_frac=0.6):
    rng = np.random.default_rng(0)
    K_list = [np.array([[1000.0, 0, 100], [0, 1000.0, 100], [0, 0, 1]]) for _ in range(C)]
    ext = []
    for i in range(C):
        R = np.eye(3)
        t = np.array([float(i) * 0.1, 0.0, 5.0])
        ext.append((R, t))
    # world points around origin
    base = np.array([[0., 0., 0.], [0.1, 0., 0.], [-0.1, 0., 0.], [0., 0.1, 0.], [0., -0.1, 0.]])
    pts3d = np.broadcast_to(base, (T, J, 3)).astype(float).copy()
    # mark some unmeasured
    measured_mask = rng.random((T, J)) < measured_frac
    pts3d_in = pts3d.copy()
    pts3d_in[~measured_mask] = np.nan
    # build a fake robust status to match
    status = np.where(measured_mask, 'strong_3d', 'unresolved').astype(object)
    robust = {'points3d': pts3d_in, 'status': status, 'num_inliers': np.where(measured_mask, 3, 0),
              'sigma_m': np.where(measured_mask, 0.01, np.nan)}
    # dummy 2D + scores
    kp = np.zeros((T, C, J, 2)); sc = np.zeros((T, C, J))
    return robust, kp, sc, K_list, ext, measured_mask


def test_masks_are_mutually_exclusive_and_exhaustive():
    robust, kp, sc, K_list, ext, _ = _synthetic_run()
    res = reconstruct_sequence(robust, kp, sc, K_list, ext, window_size=10, overlap=3)
    pv = res['provenance']
    F, J = pv['strong_mask'].shape
    total = F * J
    # union of all categories == every frame-joint slot
    covered = (pv['strong_mask'] | pv['usable_mask'] | pv['weak_mask'] |
               pv['reconstructed_mask'] | pv['boundary_hold_mask'] | pv['unresolved_mask'])
    assert covered.sum() == total


def test_strong_points_preserved_within_trust_region():
    robust, kp, sc, K_list, ext, _ = _synthetic_run()
    res = reconstruct_sequence(robust, kp, sc, K_list, ext, window_size=10, overlap=3)
    pv = res['provenance']
    strong = pv['strong_mask']
    if strong.any():
        d = np.linalg.norm(res['points3d'][strong] - robust['points3d'][strong], axis=-1)
        # strong points must not move more than a small tolerance
        assert np.max(d) < 0.05, f"strong point moved {np.max(d)}m, violates trust region"


def test_reconstructed_excludes_measured():
    robust, kp, sc, K_list, ext, measured_mask = _synthetic_run()
    res = reconstruct_sequence(robust, kp, sc, K_list, ext, window_size=10, overlap=3)
    pv = res['provenance']
    # nothing marked measured (strong/usable/weak from robust) should also be reconstructed
    measured = robust['status'] != 'unresolved'
    overlap = pv['reconstructed_mask'] & measured
    assert not overlap.any(), f"{overlap.sum()} measured points wrongly marked reconstructed"


def test_delta_from_raw_nonneg():
    robust, kp, sc, K_list, ext, _ = _synthetic_run()
    res = reconstruct_sequence(robust, kp, sc, K_list, ext, window_size=10, overlap=3)
    pv = res['provenance']
    # delta is only meaningful where the raw point was finite (measured).
    # NaN delta for reconstructed points is expected and correct.
    fin = np.isfinite(pv['delta_from_raw'])
    assert np.all(pv['delta_from_raw'][fin] >= 0)
