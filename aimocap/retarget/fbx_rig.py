import ufbx
import numpy as np
from scipy.spatial.transform import Rotation

class Skeleton:
    def __init__(self, filepath: str):
        self.filepath = filepath
        scene = ufbx.load_file(filepath)
        
        self.node_names: list[str] = []
        self.parents: list[int] = []
        self.rest_translations: list[np.ndarray] = []
        self.rest_rotations: list[np.ndarray] = []
        
        self.name_to_idx: dict[str, int] = {}
        
        self._parse_hierarchy(scene)
        
    def _parse_hierarchy(self, scene):
        # Build hierarchy using scene.nodes directly to avoid ufbx children pointer issues
        
        # We only want bones. In Unreal rigs, bones are typically between 'root' and the fingertips.
        # We'll collect all nodes that don't have 'SKM', 'ik_', 'interaction', or 'center_of_mass'
        
        valid_nodes = []
        for node in scene.nodes:
            if node.is_root: continue
            if any(x in node.name for x in ['SKM', 'ik_', 'interaction', 'center_of_mass']): continue
            valid_nodes.append(node)
            
        # We need to ensure parents are added before children. 
        # Usually FBX nodes are already topologically sorted in scene.nodes.
        
        for node in valid_nodes:
            idx = len(self.node_names)
            self.node_names.append(node.name)
            self.name_to_idx[node.name] = idx
            
        for node in valid_nodes:
            # Find parent index
            parent_idx = -1
            if node.parent and node.parent.name in self.name_to_idx:
                parent_idx = self.name_to_idx[node.parent.name]
                
            self.parents.append(parent_idx)
            
            t = node.local_transform.translation
            r = node.local_transform.rotation
            
            self.rest_translations.append(np.array([t.x, t.y, t.z]))
            self.rest_rotations.append(np.array([r.x, r.y, r.z, r.w]))
            
        self.num_joints = len(self.node_names)

        self._repair_hierarchy_roots()

    def _repair_hierarchy_roots(self):
        """Keep the parsed bone list as a real FK tree.

        The Manny FBX has a top-level bone named ``root`` under a non-bone mesh
        node.  Through ufbx's node view this can occasionally resolve back into
        the bone set as ``root -> thigh_r -> pelvis -> root``.  FK and BVH export
        both require a directed tree, so break that bad parent link at the
        explicit root bone and reject any remaining cycles loudly.
        """
        if "root" in self.name_to_idx:
            self.parents[self.name_to_idx["root"]] = -1

        for i, p in enumerate(self.parents):
            if p == i:
                self.parents[i] = -1

        for i in range(len(self.parents)):
            seen = set()
            cur = i
            while cur != -1:
                if cur in seen:
                    cycle = " -> ".join(self.node_names[j] for j in seen)
                    raise ValueError(f"Cycle in FBX skeleton hierarchy near {self.node_names[i]}: {cycle}")
                seen.add(cur)
                cur = self.parents[cur]
        
    def get_forward_kinematics(
        self,
        local_rotations: np.ndarray = None,
        root_translation: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute global positions and rotations for all joints.
        local_rotations: (N, 4) quaternions (xyzw). If None, uses rest rotations.
        root_translation: optional world-space delta applied to root joints.
        """
        if local_rotations is None:
            local_rotations = np.array(self.rest_rotations)
        if root_translation is None:
            root_translation = np.zeros(3)
            
        global_pos = np.zeros((self.num_joints, 3))
        global_rot = np.zeros((self.num_joints, 4))
        global_rot[:, 3] = 1.0  # Identity quaternions
        
        for i in range(self.num_joints):
            p = self.parents[i]
            
            local_t = self.rest_translations[i]
            local_r = local_rotations[i]
            
            if p == -1:
                global_pos[i] = local_t + root_translation
                global_rot[i] = local_r
            else:
                R_parent = Rotation.from_quat(global_rot[p])
                rotated_t = R_parent.apply(local_t)
                global_pos[i] = global_pos[p] + rotated_t
                
                R_local = Rotation.from_quat(local_r)
                R_global = R_parent * R_local
                global_rot[i] = R_global.as_quat()
                
        return global_pos, global_rot
