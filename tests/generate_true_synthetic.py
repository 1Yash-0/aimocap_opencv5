"""Generate synthetic ground-truth 3D keypoints for the IK probe.

The data MUST be generated from the SAME skeleton the IK solver uses (the
proxy MocapSkeleton), not the raw FBX rig. The FBX rig and the proxy have
different topologies — e.g. the FBX parents shoulders under clavicle (a
sibling of neck_01 under spine_05), while the proxy parents shoulders under
neck_01. A neck_01 bend then moves the shoulders in the proxy but NOT in the
FBX, producing non-rigid data that no solver can fit.

This generator builds the proxy skeleton first, then applies per-frame local
rotations through the proxy's own forward kinematics, guaranteeing the output
is a rigid articulated transform of the proxy rest pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mocap_skeleton import MocapSkeleton, COCO
from aimocap.retarget.mocap_ik import MocapIKSolver


def _build_rest_sequence(fbx, F=30):
    """Generate a rest-pose 133-keypoint sequence from the FBX FK with identity
    rotations. At rest (no bending), topology doesn't matter — the joint
    positions are the same regardless of parent-child wiring. This gives
    MocapSkeleton exact, stable bone lengths that don't depend on any
    previously-generated data (no chicken-and-egg with the npz on disk).
    """
    rest_world, _ = fbx.get_forward_kinematics()
    i = fbx.name_to_idx
    # Map FBX joints -> COCO 133, same logic as the original generator.
    pts = np.zeros((F, 133, 3))
    for f in range(F):
        gp_m = rest_world / 100.0
        gp_m_yd = np.zeros_like(gp_m)
        gp_m_yd[..., 0] = gp_m[..., 0]
        gp_m_yd[..., 1] = gp_m[..., 2]
        gp_m_yd[..., 2] = -gp_m[..., 1]
        p = np.zeros((133, 3))
        p[COCO["nose"]] = gp_m_yd[i["head"]]
        p[COCO["shoulder_l"]] = gp_m_yd[i["upperarm_l"]]
        p[COCO["shoulder_r"]] = gp_m_yd[i["upperarm_r"]]
        p[COCO["elbow_l"]] = gp_m_yd[i["lowerarm_l"]]
        p[COCO["elbow_r"]] = gp_m_yd[i["lowerarm_r"]]
        p[COCO["wrist_l"]] = gp_m_yd[i["hand_l"]]
        p[COCO["wrist_r"]] = gp_m_yd[i["hand_r"]]
        p[COCO["pelvis_l"]] = gp_m_yd[i["thigh_l"]]
        p[COCO["pelvis_r"]] = gp_m_yd[i["thigh_r"]]
        p[COCO["knee_l"]] = gp_m_yd[i["calf_l"]]
        p[COCO["knee_r"]] = gp_m_yd[i["calf_r"]]
        p[COCO["ankle_l"]] = gp_m_yd[i["foot_l"]]
        p[COCO["ankle_r"]] = gp_m_yd[i["foot_r"]]
        pts[f] = p
    # Apply the same Y-down->Z-up + m->cm conversion that load_slice uses, so
    # MocapSkeleton sees the data in its expected frame.
    conv = np.zeros_like(pts)
    conv[..., 0] = pts[..., 0] * 100.0
    conv[..., 1] = -pts[..., 2] * 100.0
    conv[..., 2] = pts[..., 1] * 100.0
    return conv


def _generate_fk_sequence(mocap_skel, ik, F, root_t, local_rotvecs, coco_to_proxy):
    """Run the proxy FK per frame and emit the 133-keypoint array (Y-down meters)."""
    K = mocap_skel.num_joints
    skel3d = np.zeros((F, 133, 3))
    for f in range(F):
        x = np.zeros(ik.num_vars)
        x[0:3] = root_t[f]
        x[3:] = local_rotvecs[f].flatten()
        root_t_state, lq = ik._state_to_local_rotations(x)
        gpos, _ = ik.forward_kinematics(root_t_state, lq)  # cm, proxy frame

        pts = np.zeros((133, 3))
        for coco_i, proxy_i in coco_to_proxy.items():
            # Convert proxy cm -> Y-down meters (inverse of load_slice's conversion).
            # load_slice: conv[0]=raw[0]*100, conv[1]=-raw[2]*100, conv[2]=raw[1]*100
            # => raw[0]=conv[0]/100, raw[2]=-conv[1]/100, raw[1]=conv[2]/100
            px, py, pz = gpos[proxy_i]
            pts[coco_i] = [px / 100.0, pz / 100.0, -py / 100.0]
        skel3d[f] = pts
    return skel3d


def _build_motion_params(mocap_skel, F):
    """Build root_t and local_rotvecs for the synthetic motion."""
    K = mocap_skel.num_joints
    idx = mocap_skel.name_to_idx
    root_t = np.zeros((F, 3))
    root_t[:, 0] = np.sin(np.linspace(0, 3, F)) * 20.0
    root_t[:, 1] = np.cos(np.linspace(0, 3, F)) * 10.0

    bend_proxy_names = [
        "spine_01", "spine_02", "spine_03", "neck_01",
        "hip_l", "hip_r", "knee_l", "knee_r",
        "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    ]
    bend_joints = {idx[nm] for nm in bend_proxy_names if nm in idx}

    local_rotvecs = np.zeros((F, K, 3))
    for f in range(F):
        for j in range(K):
            if j in bend_joints:
                angle = np.sin(f / F * np.pi + j) * 0.2
                axis = np.array([1.0, 0.5, 0.2])
                axis /= np.linalg.norm(axis)
                local_rotvecs[f, j] = axis * angle
    return root_t, local_rotvecs


def _build_coco_to_proxy(mocap_skel):
    """Map COCO 133 indices -> proxy joint indices. Handles the naming mismatch
    where coco_anchor uses "hip_l"/"hip_r" but COCO stores "pelvis_l"/"pelvis_r"."""
    coco_name_to_idx = dict(COCO)
    coco_name_to_idx["hip_l"] = COCO["pelvis_l"]
    coco_name_to_idx["hip_r"] = COCO["pelvis_r"]
    coco_to_proxy = {}
    for proxy_i, coco_name in mocap_skel.coco_anchor.items():
        if coco_name in coco_name_to_idx:
            coco_idx = coco_name_to_idx[coco_name]
            coco_to_proxy[coco_idx] = proxy_i
    return coco_to_proxy


def _apply_load_slice_conversion(skel3d_meters):
    """Convert Y-down meters -> Z-up cm (same as ik_probe.load_slice)."""
    pts = skel3d_meters * 100.0
    conv = np.zeros_like(pts)
    conv[..., 0] = pts[..., 0]
    conv[..., 1] = -pts[..., 2]
    conv[..., 2] = pts[..., 1]
    return conv


def generate():
    fbx = Skeleton("Manny.FBX")
    F = 30

    # ---- Iterative generation to break the bone-length chicken-and-egg. ----
    # MocapSkeleton estimates bone lengths from the data via median pairwise
    # distances. But the generator uses the skeleton's FK (which depends on
    # those bone lengths) to produce the data. We iterate: build skeleton ->
    # generate data -> rebuild skeleton from that data -> regenerate, until the
    # bone lengths converge (the test will build its skeleton from the final
    # output, so the generator must produce data that is self-consistent with
    # the skeleton the test will estimate from it).

    # Pass 1: skeleton from rest sequence (FBX FK, identity rotations).
    rest_seq = _build_rest_sequence(fbx, F)
    skel_init = MocapSkeleton(rest_seq, np.ones(rest_seq.shape[:-1]), fbx_skel=fbx)
    ik_init = MocapIKSolver(skel_init)
    rt_init, rv_init = _build_motion_params(skel_init, F)
    ctp_init = _build_coco_to_proxy(skel_init)
    skel3d_init = _generate_fk_sequence(skel_init, ik_init, F, rt_init, rv_init, ctp_init)
    conv = _apply_load_slice_conversion(skel3d_init)

    prev_bl = None
    for iteration in range(10):
        skel = MocapSkeleton(conv, np.ones(conv.shape[:-1]), fbx_skel=fbx)
        ik = MocapIKSolver(skel)
        root_t, local_rotvecs = _build_motion_params(skel, F)
        coco_to_proxy = _build_coco_to_proxy(skel)
        skel3d = _generate_fk_sequence(skel, ik, F, root_t, local_rotvecs, coco_to_proxy)
        conv = _apply_load_slice_conversion(skel3d)

        # Check convergence: compare bone lengths to previous iteration.
        bl = skel.bone_lengths.copy()
        if prev_bl is not None:
            max_diff = np.max(np.abs(bl - prev_bl))
            print(f"  iteration {iteration}: max bone_length delta = {max_diff:.6f} cm")
            if max_diff < 1e-4:
                print(f"  converged at iteration {iteration}")
                break
        prev_bl = bl

    import os
    os.makedirs("outputs", exist_ok=True)
    np.savez("outputs/true_synthetic.npz", skeleton3d=skel3d)
    print("Saved outputs/true_synthetic.npz (proxy-FK generated, iterated, 30 frames)")


if __name__ == "__main__":
    generate()
