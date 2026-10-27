import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.mapping import FULL_MAPPING


class FbxIKSolver:
    def __init__(self, skel: Skeleton):
        self.skel = skel
        
        # Build mapping arrays
        self.target_indices = []
        self.skel_indices = []
        
        for kpt_idx, node_name in FULL_MAPPING.items():
            if node_name in skel.name_to_idx:
                self.target_indices.append(kpt_idx)
                self.skel_indices.append(skel.name_to_idx[node_name])
                
        self.target_indices = np.array(self.target_indices)
        self.skel_indices = np.array(self.skel_indices)
        
        # Determine active joints (joints that can be optimized)
        # For simplicity, let's just optimize the joints that are directly mapped or their parents.
        # Even simpler: we can optimize all joints that have a path to a mapped joint.
        active_set = set()
        for s_idx in self.skel_indices:
            curr = s_idx
            while curr != -1:
                if curr in active_set:
                    break
                active_set.add(curr)
                
                next_curr = skel.parents[curr]
                if next_curr == curr:
                    break
                curr = next_curr
                
        self.active_joints = sorted(list(active_set))
        self.num_active = len(self.active_joints)
        
        # We parameterize the state as:
        # root_translation (3) + active_joint_rotations (num_active * 3, euler angles)
        self.num_vars = 3 + self.num_active * 3
        
        # Precompute parent matrix for fast FK
        self.parents = np.array(skel.parents)
        self.rest_t = np.array(skel.rest_translations)
        self.rest_r = np.array(skel.rest_rotations)
        
        # Compute levels for vectorized FK
        depths = {0: 0}
        for i in range(1, self.skel.num_joints):
            depths[i] = depths[self.parents[i]] + 1
            
        max_depth = max(depths.values())
        self.levels = []
        for d in range(1, max_depth + 1):
            nodes_at_d = [i for i in range(1, self.skel.num_joints) if depths[i] == d]
            self.levels.append(np.array(nodes_at_d))
            
    def _state_to_local_rotations(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root_t = x[0:3]
        angles = x[3:].reshape(-1, 3)
        
        local_rotations = self.rest_r.copy()
        
        # Apply Euler angles on top of rest rotations
        if self.num_active > 0:
            Rs = Rotation.from_euler('xyz', angles, degrees=False).as_quat()
            
            # Multiply (compose) rest rotation with optimized rotation
            R_rest = Rotation.from_quat(self.rest_r[self.active_joints])
            R_opt = Rotation.from_quat(Rs)
            
            local_rotations[self.active_joints] = (R_rest * R_opt).as_quat()
            
        return root_t, local_rotations
        
    def _forward_kinematics_fast(self, root_t: np.ndarray, local_rotations: np.ndarray) -> np.ndarray:
        """Fast vectorized FK that only computes positions"""
        global_pos = np.zeros((self.skel.num_joints, 3))
        global_rot = np.zeros((self.skel.num_joints, 4))
        
        # Root is assumed to be index 0
        global_pos[0] = self.rest_t[0] + root_t
        global_rot[0] = local_rotations[0]
        
        for level_nodes in self.levels:
            p_nodes = self.parents[level_nodes]
            
            R_parent = Rotation.from_quat(global_rot[p_nodes])
            R_local = Rotation.from_quat(local_rotations[level_nodes])
            
            global_pos[level_nodes] = global_pos[p_nodes] + R_parent.apply(self.rest_t[level_nodes])
            global_rot[level_nodes] = (R_parent * R_local).as_quat()
            
        return global_pos
        
    def _residuals(self, x: np.ndarray, target_pts: np.ndarray, weights: np.ndarray) -> np.ndarray:
        root_t, local_rotations = self._state_to_local_rotations(x)
        
        global_pos = self._forward_kinematics_fast(root_t, local_rotations)
        
        # Extract mapped joints
        pred_pts = global_pos[self.skel_indices]
        obs_pts = target_pts[self.target_indices]
        w = weights[self.target_indices]
        
        res_data = (pred_pts - obs_pts) * w[:, None]
        
        # L2 regularization on joint rotations (keep them close to rest pose)
        # root translation x[0:3] is unpenalized, joint rotations x[3:] are penalized
        reg_weight = 0.5  # Tunable hyperparameter
        res_reg = x[3:] * reg_weight
        
        return np.concatenate([res_data.flatten(), res_reg])
        
    def solve_frame(self, target_pts: np.ndarray, weights: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        if x0 is None:
            x0 = np.zeros(self.num_vars)
            
        # Initial alignment of root translation to pelvis/hip center
        if np.all(x0[0:3] == 0):
            # Try to find a good root translation from target points
            valid_targets = target_pts[self.target_indices]
            valid_w = weights[self.target_indices]
            if np.sum(valid_w > 0.5) > 0:
                mean_target = np.average(valid_targets[valid_w > 0.5], axis=0)
                mean_rest = np.mean(self.rest_t[self.skel_indices], axis=0)
                x0[0:3] = mean_target - mean_rest
                
        initial_cost = np.sum(self._residuals(x0, target_pts, weights)**2)
        res = least_squares(
            self._residuals,
            x0,
            args=(target_pts, weights),
            method='trf',
            max_nfev=200,
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3
        )
        print(f"Solver success: {res.success}, nfev: {res.nfev}, init_cost: {initial_cost:.2f}, final_cost: {res.cost:.2f}")
        return res.x
