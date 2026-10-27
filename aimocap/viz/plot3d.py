"""3D plotting and visualization utilities using matplotlib."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from aimocap.pose.keypoints import SKELETON_133


def plot_scene(
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    skeleton3d: np.ndarray,
    output_path: str | Path,
    frame_idx: int = 0,
    animate: bool = False,
) -> None:
    """
    Render a 3D scene containing the cameras and the 3D skeleton.
    
    Args:
        extrinsics: N tuples of (R, t) for the cameras.
        skeleton3d: (num_frames, 133, 3) triangulated keypoints in Y-up internal space.
        output_path: Where to save the output (png or mp4).
        frame_idx: Which frame to render if not animating.
        animate: If True, renders an mp4 spinning around the scene.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the cameras
    # Each camera is represented by its origin and a simple forward vector or pyramid
    # Camera center C = -R^T * t
    cam_centers = []
    for i, (R, t) in enumerate(extrinsics):
        # The provided extrinsics are in OpenCV space (Y-down, Z-forward).
        # We need to visualize them in our Y-up internal space.
        # Let's compute the optical center in OpenCV space first:
        C_cv = -R.T @ t
        # Convert to internal space
        C_internal = np.copy(C_cv).flatten()
        C_internal[1] = -C_internal[1]
        C_internal[2] = -C_internal[2]
        cam_centers.append(C_internal)
        # Note: we don't plot it here yet, we plot it in the animation loop
        
    # 2. Setup bounds
    # Find bounding box of skeleton to set reasonable axes
    valid_skel = skeleton3d[~np.isnan(skeleton3d).any(axis=-1)]
    if len(valid_skel) == 0:
        print("Warning: No valid skeleton points to plot.")
        x_min, x_max = -2, 2
        y_min, y_max = 0, 2
        z_min, z_max = -2, 2
    else:
        min_vals = np.min(valid_skel, axis=0)
        max_vals = np.max(valid_skel, axis=0)
        
        # Make the plot cubic so aspect ratio is 1:1:1
        center = (max_vals + min_vals) / 2
        radius = np.max(max_vals - min_vals) / 2
        radius = max(radius, 1.0) # at least 1 meter radius
        
        x_min, x_max = center[0] - radius, center[0] + radius
        y_min, y_max = center[1] - radius, center[1] + radius
        z_min, z_max = center[2] - radius, center[2] + radius

    def update(frame):
        ax.clear()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([z_min, z_max])
        ax.set_zlim([y_min, y_max])
        
        ax.set_xlabel('X (Right)')
        ax.set_ylabel('Z (Back)')
        ax.set_zlabel('Y (Up)')
        
        # Re-plot cameras (Swap Y and Z to match skeleton mapping)
        for i, C in enumerate(cam_centers):
            ax.scatter(C[0], C[2], C[1], c='red', marker='^', s=100)
            ax.text(C[0], C[2], C[1], f" C{i}", color='red')
            
        # Plot floor grid
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, 5), np.linspace(z_min, z_max, 5))
        yy = np.zeros_like(xx)
        ax.plot_wireframe(xx, zz, yy, color='gray', alpha=0.3)
        
        # Plot skeleton
        pts = skeleton3d[frame]
        for start_idx, end_idx in SKELETON_133:
            p1 = pts[start_idx]
            p2 = pts[end_idx]
            
            if not np.isnan(p1).any() and not np.isnan(p2).any():
                # Matplotlib 3D takes (X, Y, Z), but standard 3D plots usually have Z up.
                # To make our Y-up map nicely to standard matplotlib views:
                # We map our X -> plot X, our Z -> plot Y, our Y -> plot Z.
                # Notice in set_xlim/ylim/zlim above we mapped it this way.
                ax.plot(
                    [p1[0], p2[0]], 
                    [p1[2], p2[2]], 
                    [p1[1], p2[1]], 
                    color='blue', linewidth=2
                )
                
        # If animating, rotate the view slightly
        if animate:
            ax.view_init(elev=20., azim=frame * (360.0 / skeleton3d.shape[0]))
        else:
            ax.view_init(elev=20., azim=-45)
            
    if animate:
        anim = FuncAnimation(fig, update, frames=skeleton3d.shape[0], interval=1000/30.0)
        if str(output_path).endswith('.gif'):
            anim.save(str(output_path), writer='pillow', fps=30)
        else:
            anim.save(str(output_path), writer='ffmpeg', fps=30)
    else:
        update(frame_idx)
        plt.savefig(str(output_path))
        
    plt.close(fig)
