"""
IK residual probe — measures how well the solved proxy skeleton fits the input
3D points. This is the gate for the IK/retarget math (export is verified
separately by parity).

Run:
    python -m aimocap.retarget.ik_probe
    python -m pytest tests/test_ik_residual.py

Target bar: mean residual < 0.5 cm on the synthetic slice (perfectly-rigid
input, so a correct IK should fit to sub-cm).
"""
from __future__ import annotations

import numpy as np


PASS_BAR_CM = 0.5


def load_slice(npz_path: str = "outputs/true_synthetic.npz",
               start_sec: float = 0.0, duration_sec: float = 1.0,
               fps: float = 30.0) -> np.ndarray:
    """Load the same slice the pipeline uses, with the engine's Y-up->Z-up + m->cm."""
    data = np.load(npz_path)
    skel3d = data["skeleton3d"] * 100.0
    s = int(start_sec * fps)
    e = s + int(duration_sec * fps)
    pts3d = skel3d[s:e].copy()
    conv = np.zeros_like(pts3d)
    conv[..., 0] = pts3d[..., 0]
    conv[..., 1] = -pts3d[..., 2]
    conv[..., 2] = pts3d[..., 1]
    return conv


def solve_ik(pts3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the IK on the slice. Returns (target_positions (F,K,3),
    solved_positions (F,K,3), joint_names). K = MocapSkeleton.num_joints (now
    rig-derived, not fixed at 15)."""
    from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
    from aimocap.retarget.mocap_ik import MocapIKSolver
    from aimocap.retarget.fbx_rig import Skeleton

    fbx = Skeleton("Manny.FBX")
    mocap_skel = MocapSkeleton(pts3d, np.ones(pts3d.shape[:-1]), fbx_skel=fbx)
    ik = MocapIKSolver(mocap_skel)
    F = pts3d.shape[0]
    K = mocap_skel.num_joints

    # Per-frame measured dicts + solved positions in joint-order.
    targets = np.zeros((F, K, 3))
    solved = np.zeros((F, K, 3))
    prev = None
    for f in range(F):
        if f % 5 == 0:
            print(f"Solving Frame {f}/{F}...")
        measured = extract_mocap_points(pts3d[f])
        spine_names = mocap_skel.topo.spine_chain("pelvis", "neck_01")
        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]
        
        idx_hl = mocap_skel.name_to_idx["hip_l"]
        idx_hr = mocap_skel.name_to_idx["hip_r"]
        idx_sl = mocap_skel.name_to_idx["shoulder_l"]
        idx_sr = mocap_skel.name_to_idx["shoulder_r"]
        
        rest_hip = mocap_skel.rest_t[idx_hr] - mocap_skel.rest_t[idx_hl]
        rest_pelvis_mid = (mocap_skel.rest_t[idx_hl] + mocap_skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (mocap_skel.rest_t[idx_sl] + mocap_skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid
        
        from aimocap.retarget.root_frame import root_rotation
        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)
        
        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = mocap_skel.rest_t[idx_sr] - mocap_skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)
        
        # Pelvis/neck targets = measured midpoints offset to joint positions.
        # See mocap_ik.analytic_init for why the offset is needed.
        rest_pelvis_mid = (mocap_skel.rest_t[idx_hl] + mocap_skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (mocap_skel.rest_t[idx_sl] + mocap_skel.rest_t[idx_sr]) / 2.0
        pelvis_offset = mocap_skel.rest_t[0] - rest_pelvis_mid
        neck_offset = mocap_skel.rest_t[mocap_skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)
        
        from aimocap.retarget.spine_chain import distribute_spine_targets
        rest_positions = np.array([mocap_skel.rest_t[mocap_skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)
        
        tgt_pos = np.zeros((K, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = mocap_skel.name_to_idx[nm]
            if k == 0:
                tgt_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                tgt_pos[i] = tgt_neck
            else:
                tgt_pos[i] = inter[si]; si += 1
            
        for i, nm in mocap_skel.coco_anchor.items():
            if mocap_skel.joint_names[i] not in ["pelvis", "neck_01"]:
                tgt_pos[i] = measured[nm]
                
        targets[f] = tgt_pos

        x = ik.solve_frame(measured, prev_x=prev, temporal_weight=0.05)
        prev = x
        root_t, lq = ik._state_to_local_rotations(x)
        gpos, _ = ik.forward_kinematics(root_t, lq)
        solved[f] = gpos
    return targets, solved, mocap_skel.joint_names


def run_probe(verbose: bool = True) -> dict:
    pts3d = load_slice()
    targets, solved, names = solve_ik(pts3d)
    # Evaluate residual on targeted joints (exclude unanchored helper joints like clavicles)
    active_indices = [j for j, nm in enumerate(names) if nm not in ("clavicle_l", "clavicle_r")]
    res = np.linalg.norm(solved[:, active_indices] - targets[:, active_indices], axis=-1)   # (F, active_K)
    active_names = [names[j] for j in active_indices]
    result = {
        "pass": float(res.mean()) < PASS_BAR_CM,
        "mean_cm": float(res.mean()),
        "max_cm": float(res.max()),
        "per_joint_mean": {active_names[k]: float(res[:, k].mean()) for k in range(len(active_names))},
    }
    if verbose:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"\n=== IK Residual: {status} (bar mean < {PASS_BAR_CM} cm) ===")
        print(f"  mean = {result['mean_cm']:.3f} cm   max = {result['max_cm']:.3f} cm")
        print(f"  per-joint mean (cm):")
        for nm, v in result["per_joint_mean"].items():
            mark = "  ***" if v > PASS_BAR_CM else ""
            print(f"    {nm:12s} {v:7.3f}{mark}")
    return result


if __name__ == "__main__":
    run_probe()
