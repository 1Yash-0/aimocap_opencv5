"""CPU-only regression tests for the production head face-track gate."""

import numpy as np

from aimocap.retarget import mocap_ik
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.mocap_skeleton import MocapSkeleton


from tests._assets import requires_manny_fbx
def _solver_and_measured():
    pts3d = np.zeros((1, 133, 3), dtype=np.float64)
    pts3d[0, :, 1] = np.linspace(0, 150, 133)
    weights = np.ones((1, 133), dtype=np.float64)
    skel = MocapSkeleton(pts3d, weights)
    solver = MocapIKSolver(skel)

    proxy_of_coco = {}
    for idx, coco_name in skel.coco_anchor.items():
        proxy_of_coco.setdefault(coco_name, idx)
    offset = np.array([0.0, 0.0, 100.0])
    measured = {
        name: np.asarray(skel.rest_t[idx], dtype=float) + offset
        for name, idx in proxy_of_coco.items()
    }
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                 "heel_l", "heel_r", "big_toe_l", "big_toe_r"):
        if name not in measured:
            measured[name] = measured.get("neck", np.zeros(3)).copy()

    center = measured["neck"]
    measured.update({
        "left_ear": center + np.array([9.1, 0.0, 0.0]),
        "right_ear": center + np.array([-9.1, 0.0, 0.0]),
        "left_eye": center + np.array([2.0, 8.0, 1.0]),
        "right_eye": center + np.array([-2.0, 8.0, 1.0]),
        "nose": center + np.array([0.0, 10.0, 0.0]),
    })
    return solver, measured


@requires_manny_fbx
def test_default_face_track_activates_and_builds_head_target():
    solver, measured = _solver_and_measured()

    ctx = solver._precompute_frame(measured)

    assert bool(ctx["head_ori_active"])
    assert ctx["head_target_quat"] is not None


@requires_manny_fbx
def test_neck_follow_disables_head_target(monkeypatch):
    solver, measured = _solver_and_measured()
    monkeypatch.setattr(mocap_ik, "HEAD_ORIENTATION_MODE", "neck_follow")

    ctx = solver._precompute_frame(measured)

    assert not ctx["head_ori_active"]
    assert ctx["head_target_quat"] is None


@requires_manny_fbx
def test_collapsed_ears_refuse_head_target():
    solver, measured = _solver_and_measured()
    center = measured["neck"]
    measured["left_ear"] = center + np.array([2.0, 0.0, 0.0])
    measured["right_ear"] = center - np.array([2.0, 0.0, 0.0])

    ctx = solver._precompute_frame(measured)

    assert bool(ctx["head_ori_active"])
    assert ctx["head_target_quat"] is None
