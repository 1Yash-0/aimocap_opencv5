# Architecture Decisions

## ADR-001: Product Path to Polished Markerless Mocap

- **Status:** Accepted
- **Date:** 2026-10-01
- **Decision scope:** Production architecture direction for turning the current multi-camera mocap pipeline into a polished game-developer product.
- **Review:** Independent design review (adversarial and second-opinion passes) informed this decision. Review outputs were treated as perspectives, not votes or proof.

### Context

Current output maps overall motion well, but users still see jitter, imperfect motion, and failures at extreme views/poses. The user asked whether to prioritize motion-detection accuracy, then mesh/character-aware retargeting, and possibly marker/suit aids.

Repository inspection shows two related but distinct production stories:

1. The shipped CLI path: `sync`/`panoptic-sync` -> `multipose` -> `calib` -> `triangulate` -> `retarget` -> BVH/NPY/FBX. Retarget uses `MocapIKSolver` through `ik_backend`.
2. The scripts/docs validation path: `ProductionIKSolver`/`TorchSkeleton`, production JSON configuration, Layer-2 filtering, batch/contact-aware objectives, and release metrics. This is not the same objective or call path as the CLI.

Other verified issues include:

- `--filter-f4` invokes a Butterworth filter, not the research `RobustTemporalOptimizerF4`.
- Default retarget consumes raw `skeleton3d`; refined artifacts are used only through specific grounded paths.
- Lower-body target and rotation smoothing are already enabled; more global low-pass filtering risks deleting impacts.
- Extreme-angle failures may be owned by 2D detection, border/out-of-frame observations, camera-subset geometry, IK initialization gates, NaN/hold policy, or a latent `wrist_targets_init` code path.
- Retargeting is skeleton-only; no mesh or primitive body volumes are available for chest/self-contact constraints.
- Research adapters import production experiment code from `scripts/`, creating coupling across the intended research/script boundary.

### Decision

Adopt **detection-first with attribution, then proportion-aware retargeting, with marker/IMU aids kept optional and evidence-gated**.

Do **not** begin with a solver merger, a research F4 promotion, mesh SDF projection, a marker/suit product fork, or another global smoothing layer.

The product path is:

1. **Phase 0 — Attribute and gate.** Instrument the actual CLI-to-FBX path, pin a small failing corpus, tag backend/dtype/fps/units/frame/temporal method, and determine which stage owns jitter and extreme-angle failures.
2. **Phase 1 — Detection and geometry.** Use cheap production levers first: A/B CIGPose vs in-tree RTMPose, border/out-of-frame gating, score thresholds, camera-placement guidance, and camera-subset/ray-angle analysis. Do not change camera policies from documentation by fiat.
3. **Phase 2 — Impact-safe temporal quality.** Avoid new global low-pass filters. Prefer discontinuity reduction, event/contact-aware smoothing, confidence weighting, and IK-state smoothing while preserving impact peaks and foot contact.
4. **Phase 3 — IK hygiene, not forced consolidation.** Declare the CLI solver authority, move CLI-owned retarget weights into explicit policy/configuration, fix or retire the latent `wrist_targets_init` path, and only promote the scripts solver if it wins on CLI-to-FBX animation metrics.
5. **Phase 4 — Character/self-contact constraints.** Add rigged primitive volumes and explicit contact labels in a character-space solve or thin post-pass. Do not project onto render meshes and do not put character mesh geometry into triangulation.
6. **Phase 5 — Optional observation aids.** Passive markers or IMU channels may be considered only as additive observations under the same geometry/IK contracts, and only if Phases 1-3 leave a measured extreme-view/occlusion residual.

### Alternatives considered

- **Consolidate solvers immediately:** rejected for now. The CLI and script solvers have different objectives, precision, temporal behavior, and contact handling; merging could change shipped animation without fixing the reported failures.
- **Replace `--filter-f4` with research `RobustTemporalOptimizerF4`:** rejected. It is a different algorithm with units/fps/offline/body-vs-hand risks and crosses the production/research boundary.
- **Mesh SDF projection:** rejected. It can violate IK constraints, chatter at contact boundaries, and is brittle for skinned/non-manifold game meshes.
- **Marker/suit pro mode:** deferred. It must not fork thresholds or hide unresolved markerless failures; if used, it should add observations to the same contracts.
- **Neural refinement promotion:** rejected for production now. Research results do not yet justify promotion over the classical CLI path, especially under per-sequence adaptation and synthetic-to-real transfer risk.

### Consequences

- The immediate work is measurement, correctness, and ownership attribution rather than a large rewrite.
- Documentation must stop treating the scripts validation path and the shipped CLI path as interchangeable.
- Any temporal or solver change must be judged on animation-level outcomes, not only reconstruction RMSE.
- Character-aware retargeting becomes an explicit product layer after actor-space reconstruction is stable enough to distinguish tracking error from proportion mismatch.
- Marker/IMU support remains a possible future additive channel, not a competing product stack.

### Verification criteria

- A golden CLI-to-FBX fixture records solver backend, dtype, fps, units, frame identity, temporal method, and the exact skeleton artifact consumed by retarget.
- A small corpus reproduces the user-visible complaints with stage attribution: 2D confidence, border proximity, inlier set, ray angle, camera-subset switches, NaN/hold behavior, IK init fallthrough, and filter identity.
- Animation metrics include jerk, impact-peak retention, foot slide, constraint violations, subset-switch counts, init-fallthrough counts, and primitive self-intersection/clearance where applicable.
- Extreme-view/pose changes must improve the failing corpus without regressing normal views.
- Any production solver promotion must beat the current CLI path on the same fixtures and metrics.
- Any character-volume constraint must reduce penetration without violating bone lengths, joint limits, foot contact, or impact preservation.

### Risks

- Attribution may show the main issue is not 2D detection but geometry or IK policy; phases must be re-routed by evidence.
- RTMPose may not outperform CIGPose on the user's cameras; A/B evidence is required.
- Fixing `wrist_targets_init` may be dead-code cleanup or may expose an unfinished wrist-twist feature; reachability must be tested before behavior changes.
- Impact-safe smoothing can still reduce responsiveness if tuned to aggregate error rather than motion events.
- Primitive-volume constraints can create contact chatter unless contact labels and priority order are explicit.
- The large dirty worktree makes broad refactors unsafe; changes should remain small, tested, and reversible.

## ADR-002: Promote face-driven head orientation tracking

- **Status:** Accepted
- **Date:** 2026-10-05
- **Component:** `aimocap/retarget/mocap_ik.py` (`HEAD_ORIENTATION_MODE`)
- **Evidence:** `outputs/vq_investigation/audit_v3_stage_trace.json`, `audit_head_flip/metrics.json`, `facing_decomposition.json`, `facing_corrected.json`, `audit_v5_battery.json`

### Context

The head had been deliberately rigid-followed (`neck_follow`) since the attribution phase, after the solve produced an ~85-95 deg head-facing failure with the raw face signal. Open symptom: frame-local head rotation measured 0.0 deg/frame over the full 450-frame user clip.

### Findings

1. The mode constant, not the data, disabled tracking: every gate condition was healthy on the measured clip (ear distance p50 18.2 cm vs the 5 cm gate, face points 3/3, determinant 1.0, target constructible 450/450; the GT case likewise, 300/300).
2. The long-standing facing metric compared head forward against the RAW vector `nose - ear_mid`, which carries a ~-13.3 deg anatomical elevation bias (the nose sits below the ear line on a level head); it flattered an over-pitched head and penalized a correct level one. Durable lesson: audit evaluation metrics for reference-vector bias before adjudicating experiments against them.
3. Under the corrected reference (the production processed face target), the face-tracking head tracks at 0.008 deg mean (0.002 deg p95) vs the neck-follow head's 6.49 deg mean / 17.95 deg p95; raw-metric reproduction was kept as a sanity anchor (12.267 / 16.360 deg mean for v3/variant).
4. Flip safety battery: neck jitter p95 identical (3.571 deg/frame), zero >60 deg/frame pops, cant IMPROVED (up-tilt mean 10.52 -> 4.55 deg), FK-vs-canonical +0.011 cm, two full pipeline runs bit-identical, head local motion recovered (p50/p95 2.425/5.899 deg/frame vs ~0).

### Decision

`HEAD_ORIENTATION_MODE = "face_track"` is the production default. The head residual pins head global rotation to the frame-based face target, with the existing neck-follow fallback when the gate refuses (fail-closed unchanged). Production output verified bit-identical to the pre-promotion experiment (`mannequin_highmotion_v4_solved.npy` == `audit_head_flip/run1_solved.npy`).

### Verification

- `tests/test_head_face_track_gate.py`: default activates the gate and constructs a target; `neck_follow` disables both; collapsed ears correctly refused.
- Full suite: 314 passed, 4 failed (pre-existing NumPy-2.0 `test_gates.py`), 9 skipped.

## ADR-003: Frontal knee-angle matching in the foot lock (VQ-FIX4)

- **Status:** Accepted
- **Date:** 2026-10-05
- **Component:** `aimocap/retarget/foot_lock.py` (`_solve_locked_legs` pole policy)
- **Evidence:** `outputs/vq_investigation/audit_v3_stage_trace.json`, `audit_knee_tail.json`, `audit_knee_tail_frames.csv`, `audit_knee_policy.json`, `audit_v5_battery.json`

### Context

v3 showed visible knock-knee (contact-frame frontal valgus right p50 10.58 deg, >8 deg on 57.7% of contact frames) while the solved input chain was nearly clean (p50 2.90 deg, 11.4%). A per-frame-source-pole experiment fixed the median but traded the tail into varus and was reverted.

### Findings (stage-isolated mechanism)

1. Stage 3 is innocent (valgus unchanged through leg stabilization); the blow-up is created inside `apply_foot_lock` -> `_solve_locked_legs`, and 96.3% of the >4 deg gain frames sit inside foot-contact segments.
2. Root XY offsets are exonerated (zero-offset variant shows the same blow-up; medial-offset correlation is weakly negative; the SMALLEST-discrepancy frames show the LARGEST deltas).
3. NO hint-pole policy fixes it: median / per-frame-old-axis / per-frame-new-axis / new-axis-clamp all leave right contact p50 in the 7.5-10.6 deg band. Pinning the foot to the plant anchor while the hip sways forces the knee line medial regardless of the pole hint - the lock was moving stance error from the (wandering) foot into the knee.
4. The two-bone IK circle angle is exactly the 1-DOF knob setting the viewer-visible frontal knee angle; choosing it to match the source frontal angle recovers the knee line.

### Decision

The pole hint is no longer a segment-median blend. For each contact frame the pole is selected on a fixed 181-point angular grid as the circle position whose frontal knee angle best matches `v_goal = (1-w)*v_current + w*v_smoothed_source` (window-5 centered median, edge-clamped; w = contact strength), with a 1.0 deg tie-break band toward the previous frame's circle angle (state resets at contact gaps). Degenerate frames (NaN target, collapsed circle, degenerate basis) fall back to the current per-frame pole - fail-closed preserved. Sole flattening, anchors, contacts, and root offsets are untouched.

### Verification

- Simulator reproduces the old median policy bit-exactly (valgus series max diff 1.7e-13) before any production change.
- Production v5 vs v3 on the 450-frame clip: right contact valgus p50 10.584 -> 2.172 deg, >8 deg 57.7% -> 11.1%, >15 deg 17.9% -> 2.6%, varus 0.0% -> 0.0%; left contact >15 deg 7.8% -> 0.0%; tracking residual vs smoothed source right p50 6.735 -> 0.823 deg.
- Invariants: foot/ball drift identical (0.218/0.276, 0.359/0.325 cm), sole tilt unchanged, root offsets/contacts unchanged, FK-vs-canonical 5.255 -> 5.259 cm, zero >60 deg/frame events, two full runs bit-identical, v4->v5 column diff exactly the six leg-bone euler triples.
- Regression guard: `tests/test_foot_lock_frontal_match.py` (synthetic planted-foot clip: valgus bounded post-lock, drift reduced, deterministic, no NaN). 314 passed full suite.

### Rejected alternates (measured this round)

- `per_frame_source` pole (prior round, reverted): >8 deg tail 45.1 -> 37.8%, varus worsened, calf single-frame max rose.
- phi rate-limited variant (12 deg/frame): fewer boundary swings but tracking lag (>8 deg 13.1%, varus 1.7%) - rejected for tie-band-only continuity (v2b).
- Pinned-ankle target (v4): valgus target computed with the segment-start anchor rotated the shin away from source geometry (R >8 deg 15.9%) - rejected by measurement.

## ADR-004: Stabilize arm endpoint targets (VQ-FIX5)

- **Status:** Accepted
- **Date:** 2026-10-06
- **Component:** `aimocap/retarget/temporal.py` (`stabilize_mocap_fit_targets`)
- **Evidence:** `outputs/vq_investigation/audit_arm_hf.json`, `arm_frozen/` (frozen-arm control), `arm_stab/audit_v6_battery.json`, `arm_stab/audit_v6_fastcheck.json`, `arm_stab/audit_v6_frame21.json`, `arm_stab/renders/`

### Context

After v5, the remaining visible defect was hand/arm vibration, worst during held/extended-arm configurations. The user reviewed the rendered MP4, not only metrics.

### Findings (stage-isolated)

1. The earliest high-frequency signal is canonical observation noise: wrist position HF residual p95 ~1.1 cm; MCP-derived palm frame orientation HF p95 22-38 deg/frame (a ~1 cm triangulation error on ~6 cm knuckle baselines explains it). "Static palm orientation is ambiguous" (the known observability limitation) is a separate problem; the visible jitter was positional-track noise, not palm-ambiguity.
2. `stabilize_mocap_fit_targets` filtered ONLY pelvis/hips/knees/ankles - arm tracks entered the IK raw at full weight (code-level fact; legs additionally get Stage-3/4).
3. Frozen-arm control (arm targets frozen to window medians over a 107-frame held window, full pipeline re-run): the solve is essentially quiet with still input (forearm-direction HF 0.141 deg vs 2.167 with real targets; twist floor ~1.2 deg; wrist position HF 0.961 cm). The IK/retarget is exonerated; jitter is input-noise transmission.
4. Fast-motion position fidelity check (canonical wrist speed > 50 cm/s): v6 keeps or improves FK-vs-canonical wrist error (L p95 8.39->8.03 cm, R 8.11->7.77 cm; elbows within +0.14 cm; correlations ~0.9995 unchanged). Earlier ±10% derivative-metric shifts were noise-measurement artifacts.

### Decision

The six arm endpoint tracks (`shoulder_l/r, elbow_l/r, wrist_l/r`) are added to the Stage-2 adaptive stabilizer - the same already-accepted production mechanism the lower body ships with (1.5 Hz zero-phase, adaptive cutoff raised toward 6 Hz above 50 cm/s, deterministic). No new filter class, no global smoothing; the adaptive bypass preserves genuine fast motion. Lower-body cutoff estimation is computed from the lower-body list only, so leg outputs are unchanged by construction (asserted by test).

### Verification (v6 vs v5, 450-frame clip)

- Held-arm (frames 141-247): wrist position HF p95 1.539->0.979 cm (frozen floor 0.961); hand twist HF 1.653->0.400 deg; hand swing HF (delta-rate) 3.409->1.924 deg/frame; direct orientation HF 2.778->1.825 deg; forearm-direction HF 2.167->0.778 deg.
- Fast frames: wrist position p95 error improved both sides; ten fastest frames mixed within the solve-noise band; no systematic lag/undershoot.
- Invariants: right contact valgus p50 2.172->2.061 (better), >8 deg fraction 11.08->9.94%, foot/ball drift within production logs, sole tilt unchanged, FK-vs-canonical 5.259->5.298 cm, zero >60 deg/frame events, head metrics identical, two full runs bit-identical, FBX parity PASS (max 0.001 cm). Full suite: 315 passed, 4 failed (pre-existing NumPy-2.0 `test_gates.py`), 9 skipped.
- Regression guard: `tests/test_retarget.py::test_mocap_fit_target_stabilization_filters_arm_endpoints_and_preserves_fast_motion` (slow-segment HF attenuation, fast-segment preservation, lower-body bit-identity).

### Known residual (documented, pre-existing)

A clip-start warm-start transient exists at the first motion transition (v5: solved wrist 393 cm/s at frame 21 where canonical moves 23 cm/s; v6: the transient lands at frame 22 at 469 cm/s, whip frame accurate at 2.47 cm error; v6 frame 21 is a single off-pose frame at 12.56 cm). Filter probes (cutoff 3 Hz / threshold 25) change nothing there (sub-cm displacement, invariant) - the transient is solve-side, pre-existing in v5, and its landing frame shifted. Warm-start transient handling at motion transitions is a separate defect class, deliberately not stacked onto this change.

### Rejected alternates (measured)

- Higher arm base cutoff (3 Hz) and lower speed threshold (25 cm/s): no effect on the frame-21 transient; held-window attenuation essentially identical - unnecessary complexity, rejected.
- Palm-frame orientation gating: palm ambiguity is a separate limitation and the palm frame is not a production solve target (hand MCPs are extracted but not consumed as IK targets) - no change warranted.



## ADR-005: V8 pipeline profile (opt-in): conditioned targets + rigid head/hands, Stage 3/4 off

- **Status:** Accepted
- **Date:** 2026-10-08
- **Component:** `aimocap/retarget` pipeline profile and GT harness evaluation
- **Evidence:** `outputs/gt_harness/V8_REPORT.md`, `outputs/gt_harness/V8_cmp_*_m2.md`, `outputs/gt_harness/V8FL_REPORT.md`, `outputs/vq_investigation/candidates/mannequin_highmotion_v8b_battery.json`, and the C1B/C2/S4/C3 funnel reports

### Context

The V7 GT harness isolated three defects that were not safely addressed by stacking more of the old stages:

1. Stage 4 fixed-anchor foot locking damaged planted-leg geometry. On the diag clip, legs position p95 changed from about 2.7 cm with the lock disabled to 29.9 cm with the fixed-anchor lock enabled.
2. Stage 3 low-pass stabilization introduced foot drift and did not preserve the relevant foot motion.
3. Head gap bridging could carry a stale pose through missing observations, producing a head gap-bridging error.

The harness metrics were also corrected before acceptance: pops now use the defined event population and tolerance, excursion uses the robust excursion definition, clavicle is report-only where it is unobservable, and contact uses the strict GT contact mask. These changes separate real pipeline regressions from metric artifacts.

### Decision

Add `pipeline_profile="v8"` as an explicit opt-in profile. V8 uses conditioned targets with C1b for ordinary tracks and rigid C3h processing for face and hand segments, then recomputes pelvis and neck midpoints. V8 disables Stage 3 leg stabilization and Stage 4 foot locking by default. The `--pipeline-profile v8` CLI path records its effective options and module hashes.

Keep V7 as the default. V7 defaults and behavior remain unchanged and bit-identical; V8 is not a silent default change. Explicit stage and conditioning overrides remain available for diagnostics and controlled experiments.

### Alternatives considered and rejected

- **C1b alone:** rejected because the fast-head path was over-smoothed; the C1B comparator funnel did not preserve the required fast head behavior across the established clips.
- **Sigma-oracle / H1:** rejected because the oracle-sigma experiment falsified H1 as a general promotion path; the oracle is diagnostic only in `C1_sigma_funnel.md`.
- **C2 outlier cleaning:** rejected because rigid outlier replacement still left severe foot tails and converted many foot frames to NaN (for example, 243 left-foot and 374 right-foot diag frames in the C2 table).
- **Stage-4 replant:** rejected because it did not pass the all-run promotion screen; for example, the diag C1b replant legs p95 was 3.628 cm, but real_v7 C1b was 13.464 cm and the overall acceptance was FAIL.
- **S4-lite plant correction:** rejected because the lighter plant correction still retained unacceptable diag foot/leg residuals (the composition funnel reports `P-S3off+S4` legs p95 30.161 cm and foot direction p95 42.920 degrees).
- **H6/H2/H1 head fusion:** rejected because the head-fusion candidates did not provide a stable cross-clip promotion; H1 and H2 trade error for smoothing and H6 did not beat the retained C3h composition consistently.
- **`fk_offsets` foot landmark model:** rejected. `V8FL_REPORT.md` measured diag balls dynamic p95 worsening from 30.289 to 153.071 degrees and feet jitter p95 from 0.558 to 35.220 degrees; the model is experimental and not part of V8.

### Verification

The accepted V8 evidence is:

- FK canonical position error improved from 5.28 cm to 3.58 cm in the high-motion battery.
- Fast wrist position p95 improved from 8.03 to 7.38 cm (left) and 7.74 to 4.78 cm (right); held left-wrist HF p95 improved from 0.979 to 0.660 cm.
- The real_v7 GT comparator legs position p95 improved from 9.54 to 2.11 cm. The diag_clean comparator was PASS with strict contact semantics; the known FAIL rows are retained below rather than hidden.
- The candidate battery recorded zero whole-clip leg-chain events above 60 degrees per frame.
- The accepted deterministic run recorded SHA prefix `63055cbd...` and was byte-identical to the P1 research run.
- C1B conditioning was deterministic and approximately 18x faster: the speedup report measured a minimum 18.372x speedup (18.372x, 18.534x, 18.376x, 18.696x, and 30.213x across the listed cases).

The comparator harness fixes used for this decision are: corrected pops definition, robust excursion calculation, clavicle report-only treatment, strict contact mask, and pops tolerance. Comparisons were run with the `V8_cmp_*_m2.md` reports and strict contact masking.

### Known residual risks

- Stance sole tilt regressed from V7 p95 17.8 to 26.7 degrees on the left and from 8.6 to 42.0 degrees on the right.
- Right contact valgus p50 moved by about +0.59 degrees.
- Diag head jitter p95 regressed from 6.28 to 8.06 degrees per frame squared.
- Real_v7 right strict contact slide regressed from 0.20 to 0.65 cm.
- Foot orientation roll remains weakly observable from the available landmarks; sole orientation is still a model mismatch.
- The V8 comparator still has documented FAIL rows on selected clips (including diag head jitter and real_v7 right strict slide). These are accepted as known residuals, not as evidence that V8 is ready to become the default.

### Criteria to make V8 the default

Do not change the default profile until the stance foot orientation is fixed such that:

1. Left and right stance sole tilt p95 are within +2 degrees of V7 or closer to GT.
2. Right valgus is within 0.5 degrees of the accepted target.
3. No comparator FAIL rows remain beyond the V7 noise floor, using the corrected metric definitions and strict contact mask.

At that point, rerun the full GT comparator and determinism/parity checks before changing the V7 default.

### Consequences

V8 is the safe experimental composition for conditioned body targets plus rigid face/hand segments, while V7 remains the compatibility baseline. Stage 3 and Stage 4 experiments must be evaluated separately because enabling either one overrides the V8 safety profile. Foot orientation and roll observability remain the next quality work rather than being hidden by another lock or landmark offset model.



### ADR-005 stance-foot addendum (2026-10-08)

The stance-foot investigations do not change the accepted V8 opt-in profile, but they revise the evidence used to judge a possible V8 default promotion. Sources: `outputs/gt_harness/V8SF_REPORT.md`, `V8SF2_REPORT.md`, `V8SF3_REPORT.md`, `V8_valgus.md`, and `V8FL_REPORT.md`.

#### Knee valgus attribution

V8 is closer to true GT than V7 on both synthetic clips when direct per-frame error is measured on the strict right-contact GT mask. On `diag`, V7 versus V8 absolute error is p50/p95 `0.560046/1.891419` degrees versus `0.185175/0.612610` degrees. On `diag_clean`, it is `0.708039/2.080220` versus `0.179677/0.414449` degrees. V8 is also closer on p50 and the `abs > 8` fraction on both clips; the `diag_clean` output-distribution p95 alone favors V7 slightly (`2.073031` versus `2.267030` degrees), which does not overturn the direct-error result. The higher V8 value in the real-clip battery is not an error: those clips have no independent GT and use V7 output as the reference.

#### Stance sole orientation and metric bias

The real-clip battery stance sole-tilt metric uses the old V5/V7 Stage-4 contact intervals, which are known to be over-long, and its `real_v7` GT is the V7 output. It is therefore biased toward V7 and is not a neutral physical verdict. On true GT (`diag`, GT strict stance), the sole p50/p95 values are:

- V7: left `13.47/13.59` degrees, right `18.36/18.36` degrees.
- V8: left `12.62/26.72` degrees, right `21.70/29.39` degrees.
- GT: left `2.89/11.44` degrees, right `2.89/7.90` degrees.

Foot stance orientation is consequently a limitation shared by both pipelines at the median, while V8 is worse in the p95 tail. V7 achieved its battery p95 advantage by sole flattening that damaged the legs: on `diag`, V7 leg position p95 is `19.5 cm` versus `1.3 cm` for V8.

The root cause is the foot keypoint model inside IK, as documented in `foot_ik_debug.md`; post-IK orientation edits cannot reliably reconstruct a missing or biased foot landmark geometry without trading against other constraints.

#### Rejected stance-foot experiments

All three post-IK stance-orientation iterations were experimental and are not wired into any profile:

- `V8SF` / `aimocap/retarget/stance_foot.py`: rejected because it failed the pre-registered synthetic jitter/comparator screen and the real-battery sole screen (left/right p95 `26.714/41.973` degrees versus V7 `17.800/8.613`).
- `V8SF2`: rejected because the landmark-frame and contact refinements still failed synthetic sole-error/comparator checks and the physical/battery screen, including six leg-chain events over 60 degrees and FK canonical mean `5.425 cm` versus the V8 baseline `3.580 cm`.
- `V8SF3`: rejected because the corrected replay still failed the pre-registered synthetic jitter/comparator and physical stance screen, despite passing the leg-chain and FK checks; the underlying landmark observability problem remained.
- `foot_landmark_model=fk_offsets`: rejected because it failed the pre-registered ball-orientation and jitter checks: on `diag`, balls dynamic p95 worsened from `30.289` to `153.071` degrees and feet jitter p95 from `0.558` to `35.220` degrees; real foot-direction checks also failed.

The experimental modules are `plant_correction.py`, `stance_foot.py`, and `foot_landmark_model=fk_offsets`.

#### Revised default-promotion assessment

Evidence for making V8 the default includes the large reduction in synthetic leg error (`19.500` to `1.307 cm` on `diag`, and `18.408` to `1.304 cm` on `diag_clean`), improved high-motion FK canonical mean (`5.280` to `3.580 cm`), improved direct GT knee-valgus error on both synthetic clips, and passing real-clip observable comparator runs against V8's stated target.

Evidence against making V8 the default includes the shared but unresolved stance-orientation limitation, V8's worse true-GT sole p95 on both sides (`26.72/29.39` versus V7 `13.59/18.36`), known V8 synthetic comparator residuals, and the failure of all three post-IK stance-foot iterations plus `fk_offsets` under their pre-registered criteria. The V7 sole p95 advantage is not sufficient counter-evidence because it is obtained by damaging the legs.

The default-promotion decision is pending user confirmation. The remaining research item is an IK-level foot landmark model with gauge fixing on planted-flat frames.


## ADR-006: V8 vibration fix finalist F1 (opt-in)

- Status: Accepted as opt-in finalist; not default
- Date: 2026-10-10
- Component: V8 retarget profile and two-pass arm IK
- Evidence: `outputs/gt_harness/F1_eval.md`, `F1_orientation_parity.json`, `VIB_two_pass_arms.md`, `HEAD_axis.md`, `C11_eval.md`

### Context

V8 is the compatibility default and has the lowest runtime, but the real high-motion clip retains excessive vibration, the legacy head target mixes axes, and low-observability arm roll can follow an incorrect smoothed target. TPA demonstrated a large vibration reduction but costs about 2.2x in its archived solve loop and still has comparator regressions. C9 fixed the head and selected a foot-weight variant, but its hand behavior regressed on the real clip. C10 and C11/C11c did not pass the synthetic and real comparator screens.

### Decision

Record F1 as the recommended opt-in finalist, not as the default. F1 uses the exact flags:

```text
--pipeline-profile v8 --limb-twist-anchor two_pass --two-pass-limbs arms --target-conditioning p2 --head-target-mode fixed
```

F1 uses stride 1 and no redundant clavicle prior. It preserves the V8 default unless explicitly requested by a caller. The fixed head target is opt-in, and the two-pass arm path is limited to arms rather than all limbs.

### Alternatives rejected

- V8 default: retained for compatibility and runtime; not promoted because vibration and head-axis errors remain.
- TPA: rejected as the broad finalist because its archived solve loop is about 2.2x V8 and its comparator has new non-pop/jitter failures.
- C9/C9f2 smoothed arm roll: rejected because C9 follows a wrong low-observability target and reports 14.78 cm all-hand p95 versus 7.25 cm for TPA; C9f2 still has synthetic foot/ball failures.
- C10 stride 3/max_iter 15: rejected after 18/2/20 comparator FAIL rows versus V8/TPA/C9.
- C11/C11c: rejected after synthetic arm/foot jitter and pop failures; C11c has 18/20 synthetic FAIL rows and 2/1 real/fast FAIL rows versus V8.
- Redundant clavicle prior: rejected because the clean HF gate fails even though real p2 HF falls 45.4%/54.5%.
- `fk_offsets`, rig-geometry feet, and stance-foot post-passes: rejected by their documented foot orientation, jitter, leg, or contact gates.
- Ball hinge/lock: rejected after `FEET_discriminate_final.md` showed no effect; ball rotation was already approximately zero and the ankle clamp remained active on 82-89% of frames.
- Ankle twist-limit relaxation to 20/30 degrees: rejected after `F1A_eval.md` showed diagnostic feet dynamic p95 worsening from 33.7 to 48.9/51.1 degrees.

### Verification

F1 production metrics are runtime 433.0088 s, 1.8141x V8, vibration 444.3880 deg/s, zero pops above 60 deg/s, fast wrist p95 6.6041/3.9049 cm, held wrist HF p95 0.5184 cm, FK canonical mean 3.5494 cm, valgus p50 2.7334 deg, and zero leg-chain events above 60 deg/s. F1 passes the strict TPA harness comparator on `diag` and `diag_clean`; its remaining V8 comparison rows are concentrated in foot/ball lag, jitter, and dynamic metrics. The solved NPY and BVH twin runs are byte-identical. NPY/FBX at-key p95/max parity is 0.0003/0.0017 deg.

The fixed head reconstruction independently reproduces clean synthetic GT with approximately 1.0 per-axis range ratios and geodesic p95 about 0.000016 deg. The supplied validation records also report 9 passed IK parity/twist-limit/NaN tests and successful py_compile/CLI checks.

### Risks

- Runtime is approximately 1.8x V8 and may be unacceptable for default use.
- Feet/balls remain underconstrained; no generic foot offset, foot weight, free-ball, or stance flattening experiment has passed all gates.
- Residual clavicle and leg HF remains, and the clean minimal-prior gate fails.
- Synthetic Manny GT and Panoptic body-only GT do not measure real hand/face/hidden-roll generalization.
- Between-key export parity has a 11.6236 deg maximum even though at-key parity is near zero.

### Promotion criteria

Promote F1 beyond opt-in only after: (1) independent TotalCapture validation covers body, feet/contact, hands, face, and hidden-roll cases; (2) foot/contact and clavicle/leg HF gates pass without new comparator regressions; (3) runtime is reduced or explicitly accepted; (4) deterministic NPY/BVH and NPY/FBX parity are rerun; and (5) the corrected strict-contact comparator has no unexplained FAIL rows beyond the V8 noise floor. Until then, V8 remains default and F1 remains explicit opt-in.


## ADR-007: Head target calibration and F2 candidate (opt-in)

- Status: Accepted as opt-in candidate; not default
- Date: 2026-10-11
- Scope: V8 retarget profile head target and two-pass arm candidate

### Decision

Use F2 only when explicitly requested with these flags:

```text
--pipeline-profile v8 --limb-twist-anchor two_pass --two-pass-limbs arms --target-conditioning p2 --head-target-mode calibrated
```

F2 is F1 plus the calibrated head target. It is not the default. The calibrated mode removes the real-face frame-convention offset while retaining the existing opt-in two-pass arm path.

`head_target_mode=fixed` is defective on real data and must not be used for real captures: it uses synthetic `FACE_OFFSETS` as the canonical face frame and produces an approximately 89 degree head-up error in the solved NPY/FBX outputs.

### Rejected alternatives

- Hybrid soft limits: constant-target drift remained and real animation worsened; not promoted.
- Projected LM: real-animation foot dynamic error worsened; not promoted.
- Face position weight: zero-weight output was byte-identical, so it does not explain the head residual.
- Stance-gauged feet: failed the clean/noisy gate and introduced large dynamic error/pops.
- Ankle relax: diagnostic foot dynamic p95 worsened from 33.7 to 48.9/51.1 degrees.

### Consequences and verification

F2 remains slower than V8 but lowers the production vibration score from about 699 to 444 deg/s, is deterministic across twin runs, and has near-zero at-key NPY/FBX parity. Head yaw relative to torso remains attenuated to about 0.78 of the source; face-position fighting was rejected. Feet and solver-generated arm high-frequency motion remain open issues. Re-run the strict comparator, deterministic twin, and export parity checks before any promotion beyond opt-in.


## ADR-008: CMU ground truth is not valid for axial roll/twist evaluation

- **Status:** Accepted (evaluation policy)
- **Date:** 2026-10-14
- **Component:** `scripts/datasets/cmu_bvh_to_harness.py`; CMU scoring (`scripts/cmu_metrics.py`, `scripts/cmu_score_v2.py`)
- **Evidence:** `scripts/hinge_world_axes.py` (GT knee hinge axis ~68-73 deg and elbow ~29 deg from the rig lateral axis, identical on 8 clips); converter code (~180, ~224-234); `outputs/gt_harness/HINGE_AUDIT_v2.md`

### Context
Non-spine rotations are transferred as source deltas relative to the first retained motion frame (the BVH T-pose frame is dropped), applied in Manny bone-local axes. Bone directions survive; twist about the bone does not correspond between rigs.

### Decision
On CMU v2d clips, axial roll, twist, roll on bent frames, knee/elbow axis vs GT, and ankle twist vs GT are UNSCORABLE. Earlier CMU arm-roll conclusions are withdrawn. Positions, bone directions, bend planes from positions, contacts and timing remain usable. Head orientation vs CMU GT must be validated before absolute values are trusted.

### Follow-up
Build a new clip generation with world-space rest-relative transfer from the source T-pose plus per-bone rest-swing alignment; validate that GT knee hinge axes become ~lateral; never overwrite v2d.

### Note on defaults
The code default is `pipeline_profile='v7'` (`aimocap/retarget/engine.py`). ADR-006 text saying 'V8 remains default' conflicts with the code; this needs a user decision and is not changed here.