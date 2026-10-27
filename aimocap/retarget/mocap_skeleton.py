import numpy as np

from aimocap.data.wholebody_layout import COCO_ANCHORS, N_KEYPOINTS
from aimocap.retarget.rig_topology import RigTopology

# COCO-WholeBody source indices that anchor each body region. These are the
# ONLY fixed indices — they come from the pose model, not the rig, so they are
# general across rigs. The rig-side mapping is derived at runtime.
COCO = COCO_ANCHORS


def extract_mocap_points(pts3d):
    """Extract measured body joints from (..., K, 3). Returns dict name->pos.

    Includes COCO-WholeBody foot keypoints (big toe per side) when available
    (K >= 23) so the IK can constrain foot orientation from measurement, not
    from the pelvis root frame. When K < 23 (old 17-joint data), big_toe keys
    are absent and the IK falls back to R_root synthesis for the toe target.

    Hand MCP keypoints (K >= 133) are available but NOT used for palm frame
    (Gate 2 verdict: UNUSABLE). They are extracted here for potential future use
    but are not wired into the IK constraints.
    """
    out = {}
    out["pelvis"] = (pts3d[..., COCO["pelvis_l"], :] + pts3d[..., COCO["pelvis_r"], :]) / 2.0
    out["neck"] = (pts3d[..., COCO["shoulder_l"], :] + pts3d[..., COCO["shoulder_r"], :]) / 2.0
    out["nose"] = pts3d[..., COCO["nose"], :]
    # Face keypoints for Kabsch head orientation (need ≥5 joints in input).
    has_face = pts3d.shape[-2] > COCO["right_ear"]   # >= 5 joints
    if has_face:
        out["left_eye"] = pts3d[..., COCO["left_eye"], :]
        out["right_eye"] = pts3d[..., COCO["right_eye"], :]
        out["left_ear"] = pts3d[..., COCO["left_ear"], :]
        out["right_ear"] = pts3d[..., COCO["right_ear"], :]
    has_feet = pts3d.shape[-2] > COCO["big_toe_r"]   # >= 21 joints
    for side in ("l", "r"):
        out[f"shoulder_{side}"] = pts3d[..., COCO[f"shoulder_{side}"], :]
        out[f"elbow_{side}"] = pts3d[..., COCO[f"elbow_{side}"], :]
        out[f"wrist_{side}"] = pts3d[..., COCO[f"wrist_{side}"], :]
        out[f"hip_{side}"] = pts3d[..., COCO[f"pelvis_{side}"], :]
        out[f"knee_{side}"] = pts3d[..., COCO[f"knee_{side}"], :]
        out[f"ankle_{side}"] = pts3d[..., COCO[f"ankle_{side}"], :]
        if has_feet:
            out[f"big_toe_{side}"] = pts3d[..., COCO[f"big_toe_{side}"], :]
            # Heel keypoint anchors the dorsiflexion clamp in mocap_ik: the
            # sole runs heel->toe, so the toe's height relative to the heel
            # tells us whether the foot is flat (toe ~ heel height) or
            # dorsiflexed (toe above heel -> "on heels").
            out[f"heel_{side}"] = pts3d[..., COCO[f"heel_{side}"], :]

    # Hand MCP keypoints — available in 133-point data (K >= 133)
    # Gate 2 verdict: UNUSABLE for palm frame. Extracted but not wired to IK.
    has_hands = pts3d.shape[-2] >= N_KEYPOINTS
    if has_hands:
        for side in ("l", "r"):
            out[f"hand_wrist_{side}"] = pts3d[..., COCO[f"hand_wrist_{side}"], :]
            out[f"hand_index_mcp_{side}"] = pts3d[..., COCO[f"hand_index_mcp_{side}"], :]
            out[f"hand_middle_mcp_{side}"] = pts3d[..., COCO[f"hand_middle_mcp_{side}"], :]
            out[f"hand_ring_mcp_{side}"] = pts3d[..., COCO[f"hand_ring_mcp_{side}"], :]
            out[f"hand_pinky_mcp_{side}"] = pts3d[..., COCO[f"hand_pinky_mcp_{side}"], :]

    return out




class MocapSkeleton:
    """General proxy skeleton derived from the target rig at runtime.

    Joint set, hierarchy, and bone lengths all come from the rig via
    RigTopology — nothing is hardcoded to Manny. The spine is expanded to the
    rig's full joint count so the upper body can bend.
    """

    def __init__(self, sequence_pts3d, sequence_weights=None, fbx_skel=None,
                 fbx_rig_path="Manny.FBX", confidence=None):
        if fbx_skel is None:
            from aimocap.retarget.fbx_rig import Skeleton
            fbx_skel = Skeleton(fbx_rig_path)
        self._fbx_skel = fbx_skel
        self.topo = RigTopology(fbx_skel)

        # Discover the rig's spine chain (pelvis -> ... -> neck).
        spine_names = self.topo.spine_chain("pelvis", "neck_01")
        spine_segs = self.topo.segment_lengths(spine_names)

        # Build the proxy joint list + parents + COCO anchors.
        self.joint_names, self.parents, self.coco_anchor, self.fbx_mapping = (
            self._build_joint_order(spine_names)
        )
        self.num_joints = len(self.joint_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.joint_names)}

        # Estimate median bone lengths from the sequence per measured joint.
        measured = extract_mocap_points(sequence_pts3d)
        self.bone_lengths = self._estimate_bone_lengths(measured, spine_segs)

        # Rest offsets + rest positions, scaled to actor bone lengths.
        self.rest_offsets, self.rest_t = self._build_rest()

        # Per-joint confidence from triangulation (0-1).  The IK uses this
        # to downweight noisy joints (e.g. right hip conf=0.08).  Shape
        # is (F, K) where K = number of COCO keypoints in the input.
        # When None, all joints get confidence 1.0 (backward-compatible).
        self.confidence = confidence

    def _build_joint_order(self, spine_names):
        """Construct (joint_names, parents, coco_anchor, fbx_mapping).

        Layout: pelvis, <spine intermediates>, neck, then head/arms/legs each as
        a chain off the appropriate anchor. Parents index into joint_names.
        """
        names: list[str] = []
        parents: list[int] = []
        coco_anchor: dict[int, str] = {}     # proxy_idx -> coco measured name
        fbx_mapping: dict[int, str] = {}     # proxy_idx -> rig bone name

        def add(name: str, parent: int, rig_name: str, coco_name: str | None = None):
            idx = len(names)
            names.append(name)
            parents.append(parent)
            fbx_mapping[idx] = rig_name
            if coco_name is not None:
                coco_anchor[idx] = coco_name
            return idx

        # Spine chain: pelvis(0) ... neck
        spine_idx: dict[str, int] = {}
        for k, sn in enumerate(spine_names):
            parent = -1 if k == 0 else spine_idx[spine_names[k - 1]]
            spine_idx[sn] = add(sn, parent, sn,
                                coco_name=("pelvis" if k == 0 else
                                           ("neck" if k == len(spine_names) - 1 else None)))
        neck_i = spine_idx[spine_names[-1]]

        # Head off neck
        add("head", neck_i, "head", coco_name="nose")

        # Arms: clavicle off neck, shoulder off clavicle, elbow off shoulder,
        # wrist off elbow.  The clavicle is unmapped (no COCO anchor) but lets
        # the IK position the shoulder at the measured location instead of a
        # fixed rest offset from the spine.
        for side, suf in (("l", "_l"), ("r", "_r")):
            cl = add(f"clavicle_{side}", neck_i, f"clavicle{suf}")
            sh = add(f"shoulder_{side}", cl, f"upperarm{suf}", coco_name=f"shoulder_{side}")
            el = add(f"elbow_{side}", sh, f"lowerarm{suf}", coco_name=f"elbow_{side}")
            add(f"wrist_{side}", el, f"hand{suf}", coco_name=f"wrist_{side}")

        # Legs: hip off pelvis, knee off hip, ankle off knee, toe off ankle (per side).
        # toe_* maps to Manny ball_* and is COCO-anchored on the big-toe keypoint
        # (COCO-WholeBody 17/20) so the IK gets a measured foot orientation.
        # When big-toe is NaN (not detected), analytic_init falls back to R_root
        # synthesis (see mocap_ik.py).
        pelvis_i = 0
        for side, suf in (("l", "_l"), ("r", "_r")):
            hip = add(f"hip_{side}", pelvis_i, f"thigh{suf}", coco_name=f"hip_{side}")
            knee = add(f"knee_{side}", hip, f"calf{suf}", coco_name=f"knee_{side}")
            ankle = add(f"ankle_{side}", knee, f"foot{suf}", coco_name=f"ankle_{side}")
            add(f"toe_{side}", ankle, f"ball{suf}", coco_name=f"big_toe_{side}")

        return names, parents, coco_anchor, fbx_mapping

    def _estimate_bone_lengths(self, measured: dict, spine_segs: np.ndarray) -> np.ndarray:
        bl = np.zeros(self.num_joints)
        rest_world, _ = self._fbx_skel.get_forward_kinematics()
        top = self.topo

        def median_dist(a_name: str, b_name: str) -> float:
            if a_name not in measured or b_name not in measured:
                return float("nan")
            a = measured[a_name]; b = measured[b_name]
            d = np.linalg.norm(a - b, axis=-1)
            valid_d = d[np.isfinite(d)]
            if len(valid_d) == 0:
                return float("nan")
            return float(np.median(valid_d))

        # Spine scale: ratio of actor's body width to rig's body width, averaged
        # across hip and shoulder width. Both are directly measured from COCO
        # keypoints and do NOT depend on the proxy skeleton's bone lengths or
        # spine scale, so this is stable (no feedback loop). Using mid-hip-to-
        # mid-shoulder instead created a positive feedback loop: the mid-shoulder
        # position depends on the spine length (shoulders are children of neck_01),
        # so longer spine -> higher shoulders -> larger mid-to-mid -> larger scale
        # -> even longer spine, diverging across iterations.
        idx_hl = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["hip_l"]]]
        idx_hr = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["hip_r"]]]
        idx_sl = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["shoulder_l"]]]
        idx_sr = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["shoulder_r"]]]
        rig_hip_width = np.linalg.norm(rest_world[idx_hr] - rest_world[idx_hl])
        rig_shoulder_width = np.linalg.norm(rest_world[idx_sr] - rest_world[idx_sl])

        meas_hip_width = median_dist("hip_l", "hip_r")
        meas_shoulder_width = median_dist("shoulder_l", "shoulder_r")
        scale_hip = meas_hip_width / rig_hip_width if rig_hip_width > 1e-9 else 1.0
        scale_shoulder = meas_shoulder_width / rig_shoulder_width if rig_shoulder_width > 1e-9 else 1.0
        spine_scale = (scale_hip + scale_shoulder) / 2.0

        for i in range(1, self.num_joints):
            p = self.parents[i]
            child_coco = self.coco_anchor.get(i)
            parent_coco = self.coco_anchor.get(p)
            rig_child = self.fbx_mapping[i]
            rig_parent = self.fbx_mapping[p]
            
            rc = self._fbx_skel.name_to_idx[rig_child]
            rp = self._fbx_skel.name_to_idx[rig_parent]
            rest_seg = np.linalg.norm(rest_world[rc] - rest_world[rp])
            
            if parent_coco == "pelvis" and child_coco in ["hip_l", "hip_r"]:
                idx_hl = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["hip_l"]]]
                idx_hr = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["hip_r"]]]
                rig_pelvis_mid = (rest_world[idx_hl] + rest_world[idx_hr]) / 2.0
                rig_horizontal = np.linalg.norm(rest_world[rc] - rig_pelvis_mid)
                measured_horizontal = median_dist(parent_coco, child_coco)
                bl[i] = rest_seg * (measured_horizontal / rig_horizontal) if rig_horizontal > 1e-5 else rest_seg
            elif parent_coco == "neck" and child_coco in ["shoulder_l", "shoulder_r", "nose"]:
                # neck_01's COCO anchor is "neck" = mid-shoulder, but the neck_01
                # JOINT is offset from mid-shoulder (by ~7 cm in Manny). Using
                # median_dist("neck", child) gives the midpoint-to-child distance,
                # NOT the neck_01-to-child bone length. Scale the FBX rest segment
                # by the ratio of measured-to-rig midpoint distance, same approach
                # as the pelvis children above.
                idx_sl = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["shoulder_l"]]]
                idx_sr = self._fbx_skel.name_to_idx[self.fbx_mapping[self.name_to_idx["shoulder_r"]]]
                rig_neck_mid = (rest_world[idx_sl] + rest_world[idx_sr]) / 2.0
                rig_mid_to_child = np.linalg.norm(rest_world[rc] - rig_neck_mid)
                measured_mid_to_child = median_dist(parent_coco, child_coco)
                bl[i] = rest_seg * (measured_mid_to_child / rig_mid_to_child) if rig_mid_to_child > 1e-5 else rest_seg
            elif child_coco and parent_coco:
                # Direct measured distance for joints with COCO anchors on both ends.
                # This covers shoulder (neck->shoulder), head (neck->nose), arms, legs,
                # and toe (ankle->big_toe). Falls back to rest_seg*spine_scale when
                # the measured distance is NaN (e.g. foot keypoint not detected) or
                # the key is absent (e.g. 17-joint input without foot keypoints).
                if child_coco in measured and parent_coco in measured:
                    measured_bl = median_dist(parent_coco, child_coco)
                    bl[i] = measured_bl if np.isfinite(measured_bl) else rest_seg * spine_scale
                else:
                    bl[i] = rest_seg * spine_scale
            else:
                # Spine intermediates: scale each segment proportionally so the
                # total proxy spine length matches the measured pelvis->neck distance.
                bl[i] = rest_seg * spine_scale
        return bl

    def _build_rest(self):
        rest_world, _ = self._fbx_skel.get_forward_kinematics()
        rest_offsets = np.zeros((self.num_joints, 3))
        rest_t = np.zeros((self.num_joints, 3))
        for i in range(1, self.num_joints):
            p = self.parents[i]
            rc = self._fbx_skel.name_to_idx[self.fbx_mapping[i]]
            rp = self._fbx_skel.name_to_idx[self.fbx_mapping[p]]
            vec = rest_world[rc] - rest_world[rp]
            d = np.linalg.norm(vec) + 1e-8
            rest_offsets[i] = (vec / d) * self.bone_lengths[i]
            rest_t[i] = rest_t[p] + rest_offsets[i]
        return rest_offsets, rest_t
