# Mapping from COCO-WholeBody (133 keypoints) to Unreal Engine 5 Manny Skeleton

# Body mapping
BODY_MAPPING = {
    # AI Point Index -> FBX Node Name
    # (Note: UE skeleton joints are the origins of the bone. 
    # e.g. upperarm_l is the shoulder joint, lowerarm_l is the elbow, hand_l is the wrist)
    
    5: 'upperarm_l',
    6: 'upperarm_r',
    7: 'lowerarm_l',
    8: 'lowerarm_r',
    9: 'hand_l',
    10: 'hand_r',
    
    11: 'thigh_l',
    12: 'thigh_r',
    13: 'calf_l',
    14: 'calf_r',
    15: 'foot_l',
    16: 'foot_r',
    
    # We can also map neck/head if needed, but often we just use the spine hierarchy.
    0: 'head'
}

# Left hand mapping (indices 91-111 according to COCO WholeBody)
# Wait, let's verify CIGPose index offsets.
# body: 0-16
# foot: 17-22 (Left 3, Right 3? No, COCO-WholeBody has 6 foot points total. 17-19 L, 20-22 R)
# face: 23-90 (68 points)
# left hand: 91-111 (21 points)
# right hand: 112-132 (21 points)
# Total = 17 + 6 + 68 + 21 + 21 = 133.

LEFT_HAND_MAPPING = {
    # 91 is wrist (already mapped to hand_l)
    
    # Thumb (4 points in COCO, 3 bones in UE)
    # COCO: 92 (CMC), 93 (MCP), 94 (IP), 95 (Tip)
    # UE: thumb_01_l, thumb_02_l, thumb_03_l
    92: 'thumb_01_l',
    93: 'thumb_02_l',
    94: 'thumb_03_l',
    
    # Index
    # COCO: 96 (MCP), 97 (PIP), 98 (DIP), 99 (Tip)
    # UE: index_metacarpal_l (optional), index_01_l, index_02_l, index_03_l
    96: 'index_01_l',
    97: 'index_02_l',
    98: 'index_03_l',
    
    # Middle
    100: 'middle_01_l',
    101: 'middle_02_l',
    102: 'middle_03_l',
    
    # Ring
    104: 'ring_01_l',
    105: 'ring_02_l',
    106: 'ring_03_l',
    
    # Pinky
    108: 'pinky_01_l',
    109: 'pinky_02_l',
    110: 'pinky_03_l',
}

RIGHT_HAND_MAPPING = {
    # 112 is wrist
    # Thumb
    113: 'thumb_01_r',
    114: 'thumb_02_r',
    115: 'thumb_03_r',
    
    # Index
    117: 'index_01_r',
    118: 'index_02_r',
    119: 'index_03_r',
    
    # Middle
    121: 'middle_01_r',
    122: 'middle_02_r',
    123: 'middle_03_r',
    
    # Ring
    125: 'ring_01_r',
    126: 'ring_02_r',
    127: 'ring_03_r',
    
    # Pinky
    129: 'pinky_01_r',
    130: 'pinky_02_r',
    131: 'pinky_03_r',
}

FULL_MAPPING = {**BODY_MAPPING}
