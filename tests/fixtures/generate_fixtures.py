import numpy as np
import pandas as pd
from pathlib import Path

def generate_fixtures():
    print("Generating fixtures...")
    base_dir = Path(__file__).parent.parent.parent
    csv_path = base_dir / "outputs" / "diag_pose1_full" / "keypoint_2d_log.csv"
    fixtures_dir = Path(__file__).parent
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # We'll take frames 282 to 301 (20 frames)
    df_60 = df[(df['frame'] >= 282) & (df['frame'] < 302)]
    
    cameras = ["00_11", "00_12", "00_23"]
    num_frames = 20
    num_cameras = 3
    num_joints = 17
    
    valid_kpts = np.zeros((num_frames, num_cameras, num_joints, 2), dtype=np.float32)
    corrupted_kpts = np.zeros((num_frames, num_cameras, num_joints, 2), dtype=np.float32)
    scores = np.zeros((num_frames, num_cameras, num_joints), dtype=np.float32)
    
    for f_idx, frame_num in enumerate(range(282, 302)):
        f_data = df_60[df_60['frame'] == frame_num]
        for c_idx, cam in enumerate(cameras):
            c_data = f_data[f_data['camera'] == cam]
            for j in range(num_joints):
                j_data = c_data[c_data['joint'] == j]
                if len(j_data) > 0:
                    row = j_data.iloc[0]
                    # Corrupted is px_x, px_y
                    corrupted_kpts[f_idx, c_idx, j, 0] = row['px_x']
                    corrupted_kpts[f_idx, c_idx, j, 1] = row['px_y']
                    
                    # Valid is gt_proj_x, gt_proj_y (plus a tiny noise to avoid being artificially perfect)
                    vx = row['gt_proj_x'] + np.random.normal(0, 0.1)
                    vy = row['gt_proj_y'] + np.random.normal(0, 0.1)
                    valid_kpts[f_idx, c_idx, j, 0] = vx
                    valid_kpts[f_idx, c_idx, j, 1] = vy
                    
                    if vx < 0 or vx > 1920 or vy < 0 or vy > 1080:
                        scores[f_idx, c_idx, j] = 0.0
                    else:
                        scores[f_idx, c_idx, j] = row['conf']
                    
    # Save valid obs
    np.savez(fixtures_dir / "valid_obs_f0_f59.npz", kpts=valid_kpts, scores=scores)
    
    # Save corrupted obs
    np.savez(fixtures_dir / "corrupted_obs.npz", kpts=corrupted_kpts, scores=scores)
    
    # Save synthetic translation (+300px X)
    shifted = valid_kpts.copy()
    shifted[:, :, :, 0] += 300.0
    np.savez(fixtures_dir / "synthetic_translation.npz", kpts=shifted, scores=scores)
    
    # Save synthetic axis swap (X <-> Y)
    swapped = valid_kpts.copy()
    swapped = swapped[:, :, :, ::-1]
    np.savez(fixtures_dir / "synthetic_axis_swap.npz", kpts=swapped, scores=scores)
    
    # Save synthetic time shift (+10 frames for camera 1 only)
    time_shifted = valid_kpts.copy()
    time_shifted[10:, 1, :, :] = valid_kpts[:-10, 1, :, :]
    time_shifted[:10, 1, :, :] = valid_kpts[0, 1, :, :] # fill edge
    np.savez(fixtures_dir / "synthetic_time_shift.npz", kpts=time_shifted, scores=scores)
    
    print(f"Fixtures written to {fixtures_dir}")

if __name__ == "__main__":
    generate_fixtures()
