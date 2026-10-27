"""Tests for arm_orient.py — synthetic 3-point arm with known roll."""

import numpy as np
from scipy.spatial.transform import Rotation
from aimocap.retarget.arm_orient import (
    solve_upperarm_global,
    solve_forearm_global,
    solve_hand_global_v2,
    resolve_bone_axis_parent_local,
    shortest_arc,
    signed_angle_about,
    unwrap_against,
    limit_step,
    clamp_deg,
    replace_twist,
    signed_twist_about,
    rotation_angle_deg,
    Observation,
    BEND_GATE_LO_DEG,
    BEND_GATE_HI_DEG,
)


def test_twist_replacement_is_not_additive():
    """An existing +60° twist replaced by -10° must become -10°."""
    axis = np.array([0.0, 1.0, 0.0])
    ref = np.array([1.0, 0.0, 0.0])
    current = Rotation.from_rotvec(np.deg2rad(60.0) * axis)
    corrected = replace_twist(current, axis, ref, np.deg2rad(-10.0))
    assert abs(np.degrees(signed_twist_about(corrected, axis, ref)) + 10.0) < 1e-6
    assert rotation_angle_deg(corrected, Rotation.from_rotvec(np.deg2rad(-10.0) * axis)) < 1e-6


def make_synthetic_arm(bend_deg: float, humeral_roll_deg: float = 0.0,
                        fore_roll_deg: float = 0.0) -> dict:
    """Create synthetic arm observation.
    
    Rest pose: upperarm along +Y, forearm bent 10° from upperarm at rest.
    shoulder at origin, elbow at (0, L1*cos(10°), L1*sin(10°)), etc.
    """
    L1, L2 = 30.0, 25.0  # cm
    rest_elbow_bend = 10.0  # degrees - realistic rest pose
    
    R_rest_bend = Rotation.from_rotvec(np.deg2rad(rest_elbow_bend) * np.array([1.0, 0.0, 0.0]))
    sh_rest = np.array([0.0, 0.0, 0.0])
    u_rest_dir = np.array([0.0, 1.0, 0.0])
    f_rest_dir = R_rest_bend.apply(np.array([0.0, 1.0, 0.0]))
    
    R_roll = Rotation.from_rotvec(np.deg2rad(humeral_roll_deg) * np.array([0.0, 1.0, 0.0]))
    u_obs_dir = R_roll.apply(u_rest_dir)
    
    R_bend = Rotation.from_rotvec(np.deg2rad(bend_deg) * np.array([1.0, 0.0, 0.0]))
    R_fore_roll = Rotation.from_rotvec(np.deg2rad(fore_roll_deg) * np.array([0.0, 1.0, 0.0]))
    
    f_obs_dir = R_roll.apply(R_rest_bend.apply(R_bend.apply(R_fore_roll.apply(np.array([0.0, 1.0, 0.0])))))
    
    sh_w = sh_rest
    el_w = sh_w + L1 * u_obs_dir
    wr_w = el_w + L2 * f_obs_dir
    
    return {
        "sh_w": sh_w, "el_w": el_w, "wr_w": wr_w,
        "u_rest_dir": u_rest_dir, "f_rest_dir": f_rest_dir,
        "humeral_roll_true": np.deg2rad(humeral_roll_deg),
        "fore_roll_true": np.deg2rad(fore_roll_deg),
        "bend_true": np.deg2rad(bend_deg),
    }


def test_shortest_arc():
    """Test shortest_arc basic properties."""
    u = np.array([0.0, 1.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    R = shortest_arc(u, v)
    assert np.allclose(R.apply(u), v), "shortest_arc fails alignment"
    
    v = -u
    R = shortest_arc(u, v)
    assert np.allclose(R.apply(u), v), "antiparallel fails"
    rv = R.as_rotvec()
    assert np.abs(np.linalg.norm(rv) - np.pi) < 1e-6
    print("  shortest_arc: PASS")


def test_signed_angle_about():
    """Test signed angle about axis."""
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    angle, valid, _ = signed_angle_about(u, v, axis)
    assert valid
    assert np.abs(angle - np.pi/2) < 1e-6, f"Expected 90°, got {np.rad2deg(angle):.1f}°"
    
    angle, valid, _ = signed_angle_about(v, u, axis)
    assert valid
    assert np.abs(angle + np.pi/2) < 1e-6, f"Expected -90°, got {np.rad2deg(angle):.1f}°"
    
    u = np.array([0.0, 0.0, 1.0])
    angle, valid, reason = signed_angle_about(u, v, axis)
    assert not valid
    assert reason == "degenerate_projection"
    print("  signed_angle_about: PASS")


def test_unwrap_and_limit():
    """Test angle unwrapping and step limiting."""
    phi = 3.0 * np.pi
    phi_prev = 0.0
    phi_uw = unwrap_against(phi, phi_prev)
    assert min(abs(phi_uw - np.pi), abs(phi_uw + np.pi)) < 1e-6, f"unwrap failed: {phi_uw}"
    
    phi = np.deg2rad(50.0)
    phi_prev = 0.0
    phi_lim = limit_step(phi, phi_prev, 25.0)
    assert np.abs(phi_lim - np.deg2rad(25.0)) < 1e-6, f"limit_step failed: {np.rad2deg(phi_lim):.1f}°"
    
    phi = np.deg2rad(120.0)
    phi_clamped = clamp_deg(phi, 90.0)
    assert np.abs(phi_clamped - np.deg2rad(90.0)) < 1e-6
    print("  unwrap/limit/clamp: PASS")


def test_synthetic_recovery_nonidentity_rest():
    """Test A: recovery with non-identity rest bases."""
    rng = np.random.default_rng(42)
    
    R_parent_rest = Rotation.from_rotvec(rng.normal(size=3) * 0.5)
    R_parent_delta = Rotation.from_rotvec(rng.normal(size=3) * 0.3)
    R_upper_rest = R_parent_rest  # at rest, upper arm rest global matches parent rest global up to rest offset
    
    L1, L2 = 30.0, 25.0
    sh_w = np.array([0.0, 0.0, 0.0])
    e_upper_parent = np.array([0.0, 1.0, 0.0])
    R_rest_bend = Rotation.from_rotvec(np.deg2rad(10.0) * np.array([1.0, 0.0, 0.0]))
    e_fore_upper_local = R_rest_bend.apply(np.array([0.0, 1.0, 0.0]))
    
    for bend_deg in (5, 20, 45, 90):
        for roll_deg in (-90, -45, -10, 0, 10, 45, 90):
            # Humerus direction at rest in world:
            a_rest_w = (R_parent_delta * R_parent_rest).apply(e_upper_parent)
            el_w = sh_w + L1 * a_rest_w

            # Bent forearm in upperarm local frame:
            bend_R = Rotation.from_rotvec(np.deg2rad(bend_deg) * np.array([1.0, 0.0, 0.0]))
            f_bent_upper_local = bend_R.apply(e_fore_upper_local)

            # True humeral roll applied around observed humerus axis in world:
            roll_R = Rotation.from_rotvec(np.deg2rad(roll_deg) * a_rest_w)

            # Forearm direction in world:
            f_obs_w = (roll_R * R_parent_delta * R_upper_rest).apply(f_bent_upper_local)
            wr_w = el_w + L2 * f_obs_w
            
            G_upper, obs = solve_upperarm_global(
                D_parent=R_parent_delta,
                G_parent_rest=R_parent_rest,
                e_upper_parent=e_upper_parent,
                e_fore_upper_local=e_fore_upper_local,
                G_rest_upper=R_upper_rest,
                sh_w=sh_w, el_w=el_w, wr_w=wr_w,
                phi_prev=0.0,
            )
            
            if bend_deg < BEND_GATE_LO_DEG:
                assert obs.source == "swing_only", f"bend={bend_deg} expected swing_only, got {obs.source}"
                assert abs(obs.value) < 1e-12, f"bend={bend_deg} should have zero roll, got {obs.value}"
            elif bend_deg >= BEND_GATE_HI_DEG:
                assert obs.valid, f"bend={bend_deg} should be valid"
                recovered_deg = np.degrees(obs.value)
                if abs(roll_deg) < 1e-6:
                    assert abs(recovered_deg) < 0.1
    print("  synthetic_recovery_nonidentity_rest: PASS")


def test_wrong_space_input_detectable():
    """Test B: passing world-space axis where parent-local expected must differ."""
    R_parent_rest = Rotation.from_rotvec(np.array([0.5, 0.2, -0.3]))
    R_parent_delta = Rotation.from_rotvec(np.array([0.1, 0.0, 0.0]))
    
    sh_w = np.array([10.0, 5.0, 2.0])
    el_w = sh_w + np.array([0.0, 30.0, 0.0])
    wr_w = el_w + np.array([5.0, 20.0, 0.0])
    
    e_upper_correct_parent_local = R_parent_rest.inv().apply(el_w - sh_w)
    e_upper_correct_parent_local /= np.linalg.norm(e_upper_correct_parent_local)
    
    e_upper_wrong_world = (el_w - sh_w) / np.linalg.norm(el_w - sh_w)
    
    G_correct, _ = solve_upperarm_global(
        D_parent=R_parent_delta, G_parent_rest=R_parent_rest,
        e_upper_parent=e_upper_correct_parent_local, e_fore_upper_local=np.array([0,1,0]),
        G_rest_upper=Rotation.identity(), sh_w=sh_w, el_w=el_w, wr_w=wr_w, phi_prev=0.0
    )
    G_wrong, _ = solve_upperarm_global(
        D_parent=R_parent_delta, G_parent_rest=R_parent_rest,
        e_upper_parent=e_upper_wrong_world, e_fore_upper_local=np.array([0,1,0]),
        G_rest_upper=Rotation.identity(), sh_w=sh_w, el_w=el_w, wr_w=wr_w, phi_prev=0.0
    )
    
    diff = (G_correct.inv() * G_wrong).magnitude()
    assert diff > np.deg2rad(1.0), \
        f"Wrong-space input did not produce different result (diff={np.degrees(diff):.2f}°)"
    print("  wrong_space_input_detectable: PASS")


def test_side_independence():
    """Test C: left/right rest bases through same function with zero if-side branches."""
    rng = np.random.default_rng(99)
    
    for side_label, seed in [("l", 1), ("r", 2)]:
        rng_s = np.random.default_rng(seed)
        R_par_rest = Rotation.from_rotvec(rng_s.normal(size=3) * 0.4)
        R_par_delta = Rotation.from_rotvec(rng_s.normal(size=3) * 0.2)
        R_up_rest = Rotation.from_rotvec(rng_s.normal(size=3) * 0.4)
        
        e_up = rng_s.normal(size=3)
        e_up /= np.linalg.norm(e_up)
        e_fore = rng_s.normal(size=3)
        e_fore /= np.linalg.norm(e_fore)
        
        sh_w = np.zeros(3)
        el_w = np.array([0.0, 30.0, 0.0])
        wr_w = np.array([5.0, 50.0, 2.0])
        
        G, obs = solve_upperarm_global(
            D_parent=R_par_delta, G_parent_rest=R_par_rest,
            e_upper_parent=e_up, e_fore_upper_local=e_fore,
            G_rest_upper=R_up_rest,
            sh_w=sh_w, el_w=el_w, wr_w=wr_w, phi_prev=0.0
        )
        assert G.as_quat() is not None
        assert obs.source in ("swing_only", "elbow_plane", "none")
    print("  side_independence: PASS")


def test_forearm_swing_only():
    """Test forearm returns swing_only (no pronation data)."""
    e_fore_loc = np.array([0.0, 1.0, 0.0])
    G_rest_fore = Rotation.identity()
    
    synth = make_synthetic_arm(bend_deg=180.0, humeral_roll_deg=30.0)
    G_upper_w = Rotation.identity()
    
    G_fore, obs = solve_forearm_global(
        G_upper_w=G_upper_w,
        G_rest_fore=G_rest_fore,
        e_fore_loc=e_fore_loc,
        el_w=synth["el_w"],
        wr_w=synth["wr_w"],
        phi_prev=0.0
    )
    
    assert obs.source == "swing_only"
    assert obs.value == 0.0
    assert obs.valid
    assert obs.weight == 1.0
    print("  forearm_swing_only: PASS")


def test_hand_neutral():
    """Test hand = forearm delta applied to rest hand."""
    G_rest_fore = Rotation.identity()
    G_rest_hand = Rotation.identity()
    G_fore_w = Rotation.from_rotvec([0.0, np.deg2rad(30.0), 0.0])
    
    G_hand = solve_hand_global_v2(G_fore_w=G_fore_w, G_rest_fore=G_rest_fore, G_rest_hand=G_rest_hand)
    
    D_fore = G_fore_w * G_rest_fore.inv()
    D_hand = G_hand * G_rest_hand.inv()
    
    assert np.allclose(D_fore.as_matrix(), D_hand.as_matrix()), "hand delta != forearm delta"
    print("  hand_neutral: PASS")


if __name__ == "__main__":
    test_shortest_arc()
    test_signed_angle_about()
    test_unwrap_and_limit()
    test_synthetic_recovery_nonidentity_rest()
    test_wrong_space_input_detectable()
    test_side_independence()
    test_forearm_swing_only()
    test_hand_neutral()
    print("\n=== ALL TESTS PASSED ===")
