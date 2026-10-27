import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path

from aimocap.retarget.ik_probe import load_slice
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.spine_chain import distribute_spine_targets

def test_viz_ik_overlay():
    try:
        pts3d = load_slice(start_sec=0.0, duration_sec=1.0)
    except FileNotFoundError:
        pytest.skip("outputs/true_synthetic.npz not found")
        
    fbx = Skeleton("Manny.FBX")
    mocap_skel = MocapSkeleton(pts3d, np.ones(pts3d.shape[:-1]), fbx_skel=fbx)
    ik = MocapIKSolver(mocap_skel)
    
    num_frames = len(pts3d)
    
    solved_pos_seq = []
    target_pos_seq = []
    
    prev = None
    import time
    for f in range(num_frames):
        t0 = time.time()
        measured = extract_mocap_points(pts3d[f])
        x = ik.solve_frame(measured, prev_x=prev, temporal_weight=0.0)
        prev = x
        print(f"Frame {f}/{num_frames} solved in {time.time()-t0:.3f}s", flush=True)
        
        root_t, lq = ik._state_to_local_rotations(x)
        gpos, _ = ik.forward_kinematics(root_t, lq)
        solved_pos_seq.append(gpos)
        
        # Build corresponding targets
        K = mocap_skel.num_joints
        tgt = np.zeros((K, 3))
        tgt[0] = measured["pelvis"]
        spine_names = mocap_skel.topo.spine_chain("pelvis", "neck_01")
        rest_positions = np.array([mocap_skel.rest_t[mocap_skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(measured["pelvis"], measured["neck"], rest_positions)
        si = 0
        for k, nm in enumerate(spine_names):
            i = mocap_skel.name_to_idx[nm]
            if k == 0: tgt[i] = measured["pelvis"]
            elif k == len(spine_names)-1: tgt[i] = measured["neck"]
            else: tgt[i] = inter[si]; si += 1
        for i, nm in mocap_skel.coco_anchor.items():
            tgt[i] = measured[nm]
            
        target_pos_seq.append(tgt)
        
    solved_pos_seq = np.array(solved_pos_seq)
    target_pos_seq = np.array(target_pos_seq)
    
    out_dir = Path("outputs/visual_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "test_ik_overlay.gif"
    
    # Animate
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    center = np.mean(target_pos_seq[0], axis=0)
    radius = 100.0 # cm
    
    def update(frame):
        print(f"Drawing frame {frame}/{num_frames}...", flush=True)
        ax.clear()
        ax.set_xlim([center[0] - radius, center[0] + radius])
        ax.set_ylim([center[1] - radius, center[1] + radius])
        ax.set_zlim([center[2] - radius, center[2] + radius])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Up)')
        
        s_pts = solved_pos_seq[frame]
        t_pts = target_pos_seq[frame]
        
        # Plot skeleton
        for start_idx, end_idx in enumerate(mocap_skel.parents):
            if start_idx == 0: continue
            
            # Proxy skeleton (Red)
            p1 = s_pts[start_idx]
            p2 = s_pts[end_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', linewidth=2)
            
        # Plot target points
        ax.scatter(t_pts[:,0], t_pts[:,1], t_pts[:,2], c='blue', s=30, alpha=0.5, label='Measured Targets')
        
        # Slower rotation: complete half a rotation instead of a full one
        ax.view_init(elev=20., azim=frame * (180.0 / num_frames))
        
    print("Saving animation...", flush=True)
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/10.0)
    anim.save(str(dst), writer='pillow', fps=10)
    plt.close(fig)
    
    assert dst.exists()
