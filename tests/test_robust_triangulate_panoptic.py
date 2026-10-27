"""Round-trip test on the *real* Panoptic calibration bundle for cam_00_01/02/26.

Project known COCO-19 3D points to camera pixels with the official K/R/t/dist,
then verify triangulate_joint_robust recovers the 3D point.
"""
import numpy as np
import json, os, cv2
import pytest

from aimocap.geometry.robust_triangulate import (
    CameraObs, triangulate_joint_robust,
    STATUS_STRONG, STATUS_USABLE, STATUS_WEAK, STATUS_UNRESOLVED,
)

SEQ = "data/panoptic/171204_pose1"
CAL = os.path.join(SEQ, "calibration_171204_pose1.json")
BODY = os.path.join(SEQ, "hdPose3d_stage1_coco19", "body3DScene_00001500.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CAL) or not os.path.exists(BODY),
    reason="Panoptic dataset not present on this machine",
)

NAMES = ["nose", "l_should", "r_should", "l_hip", "r_hip", "l_knee", "r_knee",
         "l_ank", "r_ank", "l_eye", "r_eye", "l_ear", "r_ear",
         "l_elb", "r_elb", "l_wrist", "r_wrist"]


def _gt(j=5):
    data = json.load(open(BODY))
    bodies = data["bodies"]
    pts = np.array(bodies[0]["joints19"], dtype=float).reshape(-1, 4)
    return pts[:, :3]      # (19,3) cm


def _cam(name):
    cams = json.load(open(CAL))["cameras"]
    for c in cams:
        if c["name"] == name:
            return (np.array(c["K"], float), np.array(c["R"], float),
                    np.array(c["t"], float).reshape(3),
                    np.array(c["distCoef"], float))
    raise KeyError(name)


def _proj(K, R, t, dist, X):
    rv, _ = cv2.Rodrigues(np.asarray(R, float))
    pr, _ = cv2.projectPoints(np.asarray(X, float).reshape(1, 1, 3), rv,
                              np.asarray(t, float).reshape(3), K, dist)
    return pr.reshape(2)


@pytest.mark.parametrize("g_idx", [0, 1, 2, 5, 6, 8, 11, 12])
def test_recovers_point_three_cams_panoptic(g_idx):
    X = _gt()[g_idx]  # (3,) cm
    obs = []
    Ps = []
    Ks = []
    in_frame = []
    for name in ["00_01", "00_02", "00_26"]:
        K, R, t, dist = _cam(name)
        uv = _proj(K, R, t, dist, X)
        if 0 <= uv[0] < 1920 and 0 <= uv[1] < 1080:
            in_frame.append(True)
        else:
            in_frame.append(False)
        obs.append(CameraObs(point_uv=uv, confidence=0.95, noise_px=1.5, is_finite=True,
                             camera_K=K, camera_R=R, camera_t=t, camera_dist=dist))
        Ps.append(K @ np.hstack((R, t.reshape(3, 1))))
        Ks.append(K)
    res = triangulate_joint_robust(obs, Ps, Ks, max_accept_px=10.0)
    # If at least two cameras see this joint in-frame, the solver must recover it.
    if sum(in_frame) >= 2:
        assert res.status in (STATUS_STRONG, STATUS_USABLE), \
            f"3D point not recovered despite {sum(in_frame)} in-frame views; status={res.status}"
        assert np.isfinite(res.point3d).all()
        assert np.linalg.norm(res.point3d - X) < 15.0
    else:
        assert res.status == STATUS_UNRESOLVED
