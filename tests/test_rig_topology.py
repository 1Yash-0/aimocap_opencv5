import numpy as np
from aimocap.retarget.fbx_rig import Skeleton
from aimocap.retarget.rig_topology import RigTopology


from tests._assets import requires_manny_fbx
@requires_manny_fbx
def test_build_from_manny():
    sk = Skeleton("Manny.FBX")
    topo = RigTopology(sk)
    # spine chain pelvis->neck_01 should be discovered, length >= 2 joints
    assert "pelvis" in topo.name_to_idx
    spine = topo.spine_chain("pelvis", "neck_01")
    assert len(spine) >= 3  # pelvis + intermediates + neck_01
    assert spine[0] == "pelvis"
    assert spine[-1] == "neck_01"


@requires_manny_fbx
def test_roll_child_derived():
    sk = Skeleton("Manny.FBX")
    topo = RigTopology(sk)
    # thigh_l's roll child should be calf_l (its single child down the leg)
    rc = topo.roll_child("thigh_l")
    assert rc == "calf_l"


@requires_manny_fbx
def test_segment_lengths_positive():
    sk = Skeleton("Manny.FBX")
    topo = RigTopology(sk)
    chain = topo.spine_chain("pelvis", "neck_01")
    segs = topo.segment_lengths(chain)
    assert len(segs) == len(chain) - 1
    assert np.all(segs > 0)


@requires_manny_fbx
def test_coco_anchor_names_present():
    sk = Skeleton("Manny.FBX")
    topo = RigTopology(sk)
    # the COCO-anchored bones must resolve on any humanoid rig
    for nm in ["pelvis", "upperarm_l", "thigh_l", "calf_l", "foot_l"]:
        assert nm in topo.name_to_idx


@requires_manny_fbx
def test_roll_child_leaf_returns_none():
    sk = Skeleton("Manny.FBX")
    topo = RigTopology(sk)
    # a leaf bone (ball_l) has no child
    assert topo.roll_child("ball_l") is None
