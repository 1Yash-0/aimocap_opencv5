import numpy as np
from scipy.spatial.transform import Rotation

from aimocap.retarget.fbx_rig import Skeleton


def _nearest_euler_xyz(
    rotation: Rotation,
    previous: np.ndarray | None,
) -> np.ndarray:
    """Return an equivalent XYZ Euler representation nearest ``previous``.

    BVH stores Euler channels, so each frame must choose a continuous branch
    explicitly.  The quaternion/FBX path remains unchanged.
    """
    base = rotation.as_euler("XYZ", degrees=False)
    if previous is None:
        return base

    previous = np.asarray(previous, dtype=np.float64)
    alternate = np.array(
        [base[0] + np.pi, np.pi - base[1], base[2] + np.pi],
        dtype=np.float64,
    )
    offsets = np.array([-1, 0, 1], dtype=np.float64) * (2.0 * np.pi)
    candidates = [
        branch + np.array([dx, dy, dz], dtype=np.float64)
        for branch in (base, alternate)
        for dx in offsets
        for dy in offsets
        for dz in offsets
    ]
    return min(candidates, key=lambda value: float(np.linalg.norm(value - previous)))


def write_bvh(filepath: str, skel: Skeleton, animation_data: np.ndarray, fps: float = 30.0):
    """
    Export animation to BVH with standard Y-up right-handed convention.

    Internal pipeline uses Z-up (Unreal / Manny convention).
    BVH standard is Y-up.

    The FBX ``root`` helper bone is omitted from the BVH hierarchy to avoid
    a visible bone segment from floor to pelvis ("third leg").  ``pelvis`` is
    written as the BVH ROOT node with 6 channels.

    Coordinate conversion (Z-up → Y-up) is a −90° rotation about X:
        [x, y, z]_zup  →  [x, z, −y]_yup
    All offsets, root translations, and local rotations are transformed.
    """
    # Z-up → Y-up: rotate −90° about the X axis.
    R_conv = Rotation.from_euler("x", -90, degrees=True)
    R_conv_inv = R_conv.inv()

    # Build children list from skeleton parents.
    children: list[list[int]] = [[] for _ in range(skel.num_joints)]
    for i in range(1, skel.num_joints):
        p = skel.parents[i]
        if p != -1 and p != i:
            children[p].append(i)

    root_skel_idx = 0
    pelvis_skel_idx = skel.name_to_idx["pelvis"]

    # Safety: verify no unexpected children of root (other than pelvis).
    root_extra_children = [c for c in children[root_skel_idx] if c != pelvis_skel_idx]
    if root_extra_children:
        names = [skel.node_names[c] for c in root_extra_children]
        raise ValueError(
            f"BVH export expected 'root' to have only 'pelvis' as child, "
            f"but also found: {names}"
        )

    # Hierarchy order: skeleton indices in the order joints appear in BVH.
    hierarchy_order: list[int] = []

    with open(filepath, "w") as f:
        f.write("HIERARCHY\n")

        def write_node(skel_idx: int, indent_level: int, is_root: bool = False) -> None:
            indent = "  " * indent_level
            name = skel.node_names[skel_idx]

            if is_root:
                f.write(f"{indent}ROOT {name}\n")
            else:
                f.write(f"{indent}JOINT {name}\n")

            f.write(f"{indent}{{\n")

            if is_root:
                # BVH ROOT offset is 0; position comes from motion data.
                f.write(f"{indent}  OFFSET 0.000000 0.000000 0.000000\n")
                f.write(
                    f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                    f"Xrotation Yrotation Zrotation\n"
                )
            else:
                # Convert rest translation from Z-up to Y-up.
                offset_yup = R_conv.apply(skel.rest_translations[skel_idx])
                f.write(
                    f"{indent}  OFFSET {offset_yup[0]:.6f} "
                    f"{offset_yup[1]:.6f} {offset_yup[2]:.6f}\n"
                )
                f.write(f"{indent}  CHANNELS 3 Xrotation Yrotation Zrotation\n")

            hierarchy_order.append(skel_idx)

            node_children = children[skel_idx]
            if len(node_children) == 0:
                f.write(f"{indent}  End Site\n")
                f.write(f"{indent}  {{\n")
                f.write(f"{indent}    OFFSET 0.000000 0.000000 0.000000\n")
                f.write(f"{indent}  }}\n")
            else:
                for child_idx in node_children:
                    write_node(child_idx, indent_level + 1)

            f.write(f"{indent}}}\n")

        # Start hierarchy from pelvis (skip the root helper bone).
        write_node(pelvis_skel_idx, 0, is_root=True)

        f.write("MOTION\n")
        f.write(f"Frames: {len(animation_data)}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")

        pelvis_rest_t = skel.rest_translations[pelvis_skel_idx]
        previous_eulers: dict[int, np.ndarray] = {}

        for frame_data in animation_data:
            root_t = frame_data[0:3]
            full_rots_rad = frame_data[3:].reshape(-1, 3)

            # Root bone local rotation (skeleton index 0).
            R_root = Rotation.from_euler("xyz", full_rots_rad[root_skel_idx])

            # Pelvis world position in Z-up:
            #   root is at root_t (rest_translations[0] is zero)
            #   pelvis = root_pos + R_root.apply(pelvis_rest_translation)
            pelvis_pos_zup = root_t + R_root.apply(pelvis_rest_t)

            # Pelvis global rotation in Z-up:
            R_pelvis_local = Rotation.from_euler("xyz", full_rots_rad[pelvis_skel_idx])
            R_pelvis_global_zup = R_root * R_pelvis_local

            # Convert to Y-up.
            pelvis_pos_yup = R_conv.apply(pelvis_pos_zup)
            R_pelvis_yup = R_conv * R_pelvis_global_zup * R_conv_inv

            row: list[float] = []
            # BVH ROOT position (absolute, Y-up).
            row.extend(pelvis_pos_yup)

            # Rotations in hierarchy declaration order.
            for skel_idx in hierarchy_order:
                if skel_idx == pelvis_skel_idx:
                    # Pelvis = combined root + pelvis rotation, converted to Y-up.
                    euler_rad = _nearest_euler_xyz(
                        R_pelvis_yup, previous_eulers.get(skel_idx)
                    )
                else:
                    # All other joints: conjugate local rotation by R_conv.
                    R_local = Rotation.from_euler("xyz", full_rots_rad[skel_idx])
                    R_local_yup = R_conv * R_local * R_conv_inv
                    euler_rad = _nearest_euler_xyz(
                        R_local_yup, previous_eulers.get(skel_idx)
                    )
                previous_eulers[skel_idx] = euler_rad
                euler_deg = np.degrees(euler_rad)
                row.extend(euler_deg)

            f.write(" ".join(f"{val:.4f}" for val in row) + "\n")


def read_bvh(filepath: str):
    """
    Minimal BVH parser for round-trip testing.

    Returns
    -------
    joint_names : list[str]
    parents     : list[int]   (-1 for root)
    offsets     : np.ndarray  (J, 3)
    channel_map : list[tuple[int, list[str]]]
        Per-channel-block: (joint_index, [channel_names])
    frames      : np.ndarray  (F, total_channels)
    fps         : float
    """
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    joint_names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channel_map: list[tuple[int, list[str]]] = []

    stack: list[int] = []  # joint index stack
    i = 0
    in_hierarchy = False
    in_motion = False
    in_end_site = False
    frame_time = 1.0 / 30.0
    num_frames = 0
    motion_lines: list[str] = []

    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1
            continue

        if tok[0] == "HIERARCHY":
            in_hierarchy = True
        elif tok[0] == "End" and len(tok) > 1 and tok[1] == "Site":
            in_end_site = True
        elif tok[0] in ("ROOT", "JOINT"):
            name = tok[1]
            joint_names.append(name)
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
        elif tok[0] == "OFFSET" and not in_motion and not in_end_site:
            offsets[-1] = [float(tok[1]), float(tok[2]), float(tok[3])]
        elif tok[0] == "CHANNELS":
            num_ch = int(tok[1])
            ch_names = tok[2: 2 + num_ch]
            channel_map.append((len(joint_names) - 1, ch_names))
        elif tok[0] == "{":
            if not in_end_site and joint_names:
                stack.append(len(joint_names) - 1)
        elif tok[0] == "}":
            if in_end_site:
                in_end_site = False
            elif stack:
                stack.pop()
        elif tok[0] == "MOTION":
            in_hierarchy = False
            in_motion = True
        elif tok[0] == "Frames:":
            num_frames = int(tok[1])
        elif tok[0] == "Frame" and tok[1] == "Time:":
            frame_time = float(tok[2])
        elif in_motion and len(tok) > 1:
            motion_lines.append(lines[i])

        i += 1

    frames_arr = np.array(
        [[float(v) for v in l.split()] for l in motion_lines if l],
        dtype=np.float64,
    )

    return (
        joint_names,
        parents,
        np.array(offsets, dtype=np.float64),
        channel_map,
        frames_arr,
        1.0 / frame_time,
    )
