"""Runtime derivation of joint topology from any Skeleton.

Replaces every hardcoded Manny joint index in the retarget pipeline. The
retarget must work on any humanoid rig a user supplies, so chains, roll
children, and segment lengths are discovered from the rig's own hierarchy.

Pure data over a ``Skeleton`` (fbx_rig). No Blender, no ufbx.
"""
from __future__ import annotations

import numpy as np

from aimocap.retarget.fbx_rig import Skeleton


class RigTopology:
    """Read-only view over a Skeleton that answers topology questions."""

    def __init__(self, skel: Skeleton):
        self.skel = skel
        self.name_to_idx = dict(skel.name_to_idx)
        self._children = self._build_children()
        # Rest-pose world positions are constant for a given rig; computing them
        # requires a full FK walk over every joint, so cache the result instead
        # of recomputing on every roll_child() / segment_lengths() call (those
        # are called per-residual-eval inside the IK solver; uncached this was
        # ~60% of the residual's runtime).
        self._rest_world_pos_cache: np.ndarray | None = None

    def _build_children(self) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {i: [] for i in range(self.skel.num_joints)}
        for i, p in enumerate(self.skel.parents):
            if p != -1 and p != i:
                children[p].append(i)
        return children

    def _chain_between(self, ancestor: str, descendant: str) -> list[str]:
        """Joint names along the unique path ancestor->...->descendant."""
        a = self.name_to_idx[ancestor]
        d = self.name_to_idx[descendant]
        # walk up from descendant to ancestor
        path = [d]
        cur = d
        while cur != a:
            p = self.skel.parents[cur]
            if p == -1 or p == cur:
                raise ValueError(f"{ancestor} is not an ancestor of {descendant}")
            cur = p
            path.append(cur)
        path.reverse()
        return [self.skel.node_names[i] for i in path]

    def spine_chain(self, pelvis: str = "pelvis", neck: str = "neck_01") -> list[str]:
        """All joints from pelvis to the neck (inclusive). General: works on
        any rig where pelvis is an ancestor of the neck."""
        if neck not in self.name_to_idx:
            # fall back: find the lowest 'neck*' or 'spine*' chain end
            neck = next((n for n in self.name_to_idx if n.startswith("neck")), neck)
        return self._chain_between(pelvis, neck)

    def roll_child(self, joint: str) -> str | None:
        """The single child that pins this joint's roll. Rule: the child whose
        outgoing bone is longest (the main continuation of the chain). For a
        thigh, that's the calf; for a calf, the foot; for a spine joint, the
        next spine joint. Returns None if the joint is a leaf."""
        i = self.name_to_idx[joint]
        kids = self._children[i]
        if not kids:
            return None
        # pick child with greatest distance from this joint (rest pose)
        rest_pos = self._rest_world_positions()
        best = max(kids, key=lambda c: np.linalg.norm(rest_pos[c] - rest_pos[i]))
        return self.skel.node_names[best]

    def _rest_world_positions(self) -> np.ndarray:
        if self._rest_world_pos_cache is None:
            pos, _ = self.skel.get_forward_kinematics()
            self._rest_world_pos_cache = pos
        return self._rest_world_pos_cache

    def segment_lengths(self, chain: list[str]) -> np.ndarray:
        """Rest bone lengths along a named chain (len = len(chain)-1)."""
        rest_pos = self._rest_world_positions()
        idx = [self.name_to_idx[n] for n in chain]
        segs = np.array([
            np.linalg.norm(rest_pos[idx[k + 1]] - rest_pos[idx[k]])
            for k in range(len(idx) - 1)
        ])
        return segs
