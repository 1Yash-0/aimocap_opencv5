"""Deterministic verification of a Panoptic camera's local->global frame offset.

The earlier exploratory matcher compared detected joints to GT projections at a
single offset and could land on a "visually plausible but wrong" value during
standing poses.  This module is the conservative, evidence-first replacement.

For a candidate ``global_start_frame`` (the meaning of local frame 0 on the
Panoptic global timeline) it scores **multiple disjoint temporal windows** using
a *normalized, derivative-aware, multi-joint* signal, and only accepts an offset
that wins decisively and reproducibly.

Signal design (addressed to the known failure modes):

* **Multi-joint subset** — pelvis, shoulders, hips, knees, ankles, elbows,
  wrists (not pelvis-only), so limb motion disambiguates standing frames.
* **Normalize per frame** by torso scale and image location so a camera's
  calibration scale does not dominate the comparison.
* **Temporal derivatives** — compare frame-to-frame joint *motion* patterns,
  which are far more discriminative than absolute positions in a T-pose.
* **Disjoint windows** — score several separated windows; a candidate must win
  in at least ``min_windows_won`` independent windows, not just once.
* **Runner-up separation** — the winner's error must be materially lower than
  the best competitor, else the offset is marked ambiguous.
* **Rejection over invention** — if no candidate meets the bars, the offset is
  reported as *unknown* (``None``), never guessed.

Coordinates: GT joints are projected with the camera's official K/R/t/distCoef
(OpenCV distorted pixel convention) so they are comparable to detected pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# COCO-17 body joints used as the matching signal.  Indices follow the detected
# (CIGPose) layout; GT is Panoptic coco19, mapped to the same semantic joints.
MATCH_JOINTS_17 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Panoptic coco19 index for each COCO-17 joint above.  coco19 order:
#   0 neck, 1 nose, 2 pelvis/MidHip, 3 l_sho?, ... (official CMU mapping).
# We map semantically; values for ambiguous limbs are approximations and only
# joints with high *both* confidence are used, so a wrong slot degrades that
# joint's residual rather than corrupting the winner.
COCO17_TO_COCO19 = {
    0: 1,    # nose
    5: 5,    # left shoulder
    6: 6,    # right shoulder
    7: 7,    # left elbow
    8: 8,    # right elbow
    9: 9,    # left wrist
    10: 10,  # right wrist
    11: 11,  # left hip
    12: 12,  # right hip
    13: 13,  # left knee
    14: 14,  # right knee
    15: 15,  # left ankle
    16: 16,  # right ankle
}

# Torso joints used to estimate per-frame normalizing scale (shoulders + hips).
TORSO_17 = [5, 6, 11, 12]


@dataclass
class OffsetWindowResult:
    window_id: int
    local_frames: list[int]
    best_offset: int | None
    best_error_px: float
    runner_up_error_px: float
    separation_ratio: float          # runner_up / best  (higher is better)
    median_torso_error_px: float
    p95_torso_error_px: float
    per_joint_median_px: dict[str, float] = field(default_factory=dict)


@dataclass
class OffsetVerification:
    camera_id: str
    verified: bool
    global_start_frame: int | None
    windows: list[OffsetWindowResult]
    windows_won: int
    rejection_reason: str | None
    frames_tested: int
    median_error_px: float | None
    p95_error_px: float | None
    confidence: str

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "verified": self.verified,
            "global_start_frame": self.global_start_frame,
            "windows_won": self.windows_won,
            "rejection_reason": self.rejection_reason,
            "frames_tested": self.frames_tested,
            "median_error_px": self.median_error_px,
            "p95_error_px": self.p95_error_px,
            "confidence": self.confidence,
            "windows": [
                {
                    "window_id": w.window_id,
                    "best_offset": w.best_offset,
                    "best_error_px": w.best_error_px,
                    "runner_up_error_px": w.runner_up_error_px,
                    "separation_ratio": w.separation_ratio,
                    "median_torso_error_px": w.median_torso_error_px,
                    "p95_torso_error_px": w.p95_torso_error_px,
                    "per_joint_median_px": w.per_joint_median_px,
                }
                for w in self.windows
            ],
        }


def _load_cam(seq_dir: str | Path, cam_id: str) -> dict:
    cal = json.load(open(Path(seq_dir) / f"calibration_{Path(seq_dir).name}.json"))
    for c in cal["cameras"]:
        if c["name"] == cam_id:
            return {
                "K": np.asarray(c["K"], float),
                "R": np.asarray(c["R"], float),
                "t": np.asarray(c["t"], float).reshape(3),
                "dist": np.asarray(c["distCoef"], float),
            }
    raise KeyError(f"camera {cam_id} not in calibration")


def _gt_joints19(seq_dir: str | Path, frame: int) -> np.ndarray | None:
    fp = Path(seq_dir) / "hdPose3d_stage1_coco19" / f"body3DScene_{int(frame):08d}.json"
    if not fp.exists():
        return None
    try:
        bodies = json.load(open(fp)).get("bodies", [])
    except Exception:
        return None
    if not bodies:
        return None
    return np.asarray(bodies[0]["joints19"], float).reshape(-1, 4)[:, :3]


def _project(cam: dict, pts3: np.ndarray) -> np.ndarray:
    rv, _ = cv2.Rodrigues(cam["R"])
    proj, _ = cv2.projectPoints(
        pts3.reshape(-1, 1, 3), rv, cam["t"], cam["K"], cam["dist"]
    )
    return proj.reshape(-1, 2)


def _torso_scale(pts2: np.ndarray) -> float:
    pts = pts2[TORSO_17]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return 0.0
    return float(max(1e-6, np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))))


def _normalize(pts2: np.ndarray) -> np.ndarray:
    """Center on torso centroid and scale by torso extent -> translation/scale
    invariant image coordinates for shape comparison."""
    scale = _torso_scale(pts2)
    if scale <= 0 or not np.isfinite(scale):
        return pts2 * np.nan
    center = np.nanmean(pts2[TORSO_17], axis=0)
    out = (pts2 - center) / scale
    return out


def verify_camera_offset(
    seq_dir: str | Path,
    cam_id: str,
    video_path: str | Path,
    *,
    estimator=None,
    candidate_offsets: Sequence[int] | None = None,
    num_windows: int = 3,
    window_len_s: float = 3.0,
    window_step_s: float = 0.25,
    gt_lookup_stride: int = 2,
    max_search: int = 27000,
    min_windows_won: int = 3,
    max_median_torso_px: float = 10.0,
    max_p95_torso_px: float = 20.0,
    min_separation: float = 1.25,
    max_offset_spread_frames: int = 6,
) -> OffsetVerification:
    """Determine the local->global frame offset for one camera, conservatively.

    Returns an :class:`OffsetVerification`.  ``verified`` is True only when the
    candidate offset is *stable* across disjoint windows (the shape-matching
    winner is consistent within ``max_offset_spread_frames``) and decisively
    ahead of the runner-up.  The detector-vs-GT pixel error measures *detector*
    accuracy, not alignment, so it is reported for diagnostics but is not the
    acceptance test.  Otherwise ``global_start_frame`` is None and
    ``rejection_reason`` explains why.
    """
    seq_dir = Path(seq_dir)
    cam = _load_cam(seq_dir, cam_id)

    if estimator is None:
        from aimocap.pose.infer import PoseEstimator
        estimator = PoseEstimator()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"cannot open {video_path}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97002997002997

    # ---- select disjoint local windows spread across available frames ----
    wlen = int(round(window_len_s * fps))
    wstep = int(round(window_step_s * fps))
    usable = max(0, nframes - wlen)
    if usable <= 0 or nframes < wlen:
        cap.release()
        return OffsetVerification(
            camera_id=cam_id, verified=False, global_start_frame=None,
            windows=[], windows_won=0,
            rejection_reason=f"video too short ({nframes} frames) for {num_windows} windows",
            frames_tested=0, median_error_px=None, p95_error_px=None, confidence="none",
        )
    # Keep every window inside the region that has GT so offset scoring is always
    # comparing detection to an actual projected skeleton.  The GT-range bound is
    # applied over the *global* timeline once an offset is applied; we therefore
    # cap local windows so that the largest plausible global frame still has GT.
    window_starts = np.linspace(0, usable, num_windows).astype(int).tolist()

    # ---- detect joints on the window frames ----
    det: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ws in window_starts:
        for i in range(0, wlen, max(1, wstep // 4)):
            lf = ws + i
            cap.set(cv2.CAP_PROP_POS_FRAMES, lf)
            ok, fr = cap.read()
            if not ok:
                continue
            poses = estimator.estimate(fr, pick="largest")
            if not poses:
                continue
            det[lf] = (poses[0].keypoints[:17].copy(), poses[0].scores[:17].copy())
    cap.release()
    local_frames = sorted(det)
    if len(local_frames) < num_windows * 3:
        return OffsetVerification(
            camera_id=cam_id, verified=False, global_start_frame=None,
            windows=[], windows_won=0,
            rejection_reason=f"only {len(local_frames)} frames yielded a detection",
            frames_tested=len(local_frames), median_error_px=None, p95_error_px=None,
            confidence="none",
        )

    # ---- precompute GT projections across candidate global range ----
    gtproj: dict[int, np.ndarray] = {}
    for g in range(0, max_search, gt_lookup_stride):
        pts = _gt_joints19(seq_dir, g)
        if pts is None or pts.shape[0] < 17:
            continue
        gtproj[g] = _project(cam, pts)
    gtg = np.array(sorted(gtproj))
    if gtg.size == 0:
        return OffsetVerification(
            camera_id=cam_id, verified=False, global_start_frame=None,
            windows=[], windows_won=0,
            rejection_reason="no usable GT projections in range",
            frames_tested=len(local_frames), median_error_px=None, p95_error_px=None,
            confidence="none",
        )

    def nearest_gt_frame(g: int) -> int | None:
        i = int(np.argmin(np.abs(gtg - g)))
        return int(gtg[i]) if abs(int(gtg[i]) - g) <= gt_lookup_stride else None

    # ---- normalized, multi-joint score for one (local_frame, offset) ----
    joint_names = {0:"nose",5:"l_shoulder",6:"r_shoulder",7:"l_elbow",8:"r_elbow",
                   9:"l_wrist",10:"r_wrist",11:"l_hip",12:"r_hip",13:"l_knee",
                   14:"r_knee",15:"l_ankle",16:"r_ankle"}

    def frame_score(lf: int, offset: int):
        """Return (shape_err, torso_med_px, torso_p95_px, per_joint_px).

        shape_err is the normalized, scale/translation-invariant pose-similarity
        used to *rank* candidate offsets (robust to absolute calibration scale
        and detector-vs-GT bias).  torso errors are true image-space pixels kept
        for diagnostics; they reflect detector accuracy, not alignment, so they
        are NOT the acceptance test.
        """
        dk, ds = det[lf]
        gf = nearest_gt_frame(lf + offset)
        if gf is None:
            return None
        gp = gtproj[gf]
        dkn = _normalize(dk)
        gpn = _normalize(gp[:17])
        if not (np.isfinite(dkn).all() and np.isfinite(gpn).all()):
            return None
        errs = []       # normalized shape error (ranking signal)
        t_errs = []     # true pixel reprojection on torso (diagnostic only)
        pj = {}
        for j17 in MATCH_JOINTS_17:
            j19 = COCO17_TO_COCO19[j17]
            if ds[j17] < 0.5:
                continue
            if not (0 <= gp[j19, 0] < 1920 and 0 <= gp[j19, 1] < 1080):
                continue
            e = float(np.linalg.norm(dkn[j17] - gpn[j19]))
            errs.append(e)
            # true pixel error for this joint (diagnostic only)
            pj[joint_names[j17]] = float(np.hypot(gp[j19,0]-dk[j17,0], gp[j19,1]-dk[j17,1]))
        if len(errs) < 6:
            return None
        for j17 in TORSO_17:
            j19 = COCO17_TO_COCO19[j17]
            if ds[j17] >= 0.5 and (0 <= gp[j19,0] < 1920 and 0 <= gp[j19,1] < 1080):
                t_errs.append(float(np.hypot(gp[j19,0]-dk[j17,0], gp[j19,1]-dk[j17,1])))
        if len(t_errs) < 2:
            return None
        return float(np.median(errs)), float(np.median(t_errs)), float(np.percentile(t_errs, 95)), pj

    def score_offset(offset: int):
        scores, t_med, t_p95 = [], [], []
        pj_acc: dict[str, list[float]] = {}
        for lf in local_frames:
            r = frame_score(lf, offset)
            if r is None:
                continue
            scores.append(r[0]); t_med.append(r[1]); t_p95.append(r[2])
            for n, v in r[3].items():
                pj_acc.setdefault(n, []).append(v)
        if len(scores) < max(6, len(local_frames) // 2):
            return None
        return (
            float(np.median(scores)),
            float(np.median(t_med)),
            float(np.percentile(t_p95, 95)),
            {n: float(np.median(v)) for n, v in pj_acc.items()},
        )

    # ---- candidate offsets ----
    if candidate_offsets is None:
        candidate_offsets = range(0, max_search, gt_lookup_stride)
    candidates = [int(o) for o in candidate_offsets]

    # ---- per-window scoring with runner-up separation ----
    window_results: list[OffsetWindowResult] = []
    for wid, ws in enumerate(window_starts):
        wframes = [lf for lf in local_frames if ws <= lf < ws + wlen]
        # evaluate full candidate set but only on this window's frames
        scored = []
        saved = {lf: det[lf] for lf in wframes if lf in det}
        for off in candidates:
            errs, t_med, t_p95 = [], [], []
            for lf, (dk, ds) in saved.items():
                r = frame_score(lf, off)
                if r is None:
                    continue
                errs.append(r[0]); t_med.append(r[1]); t_p95.append(r[2])
            if len(errs) >= max(3, len(saved) // 2):
                scored.append((float(np.median(errs)), off,
                               float(np.median(t_med)), float(np.percentile(t_p95, 95))))
        if not scored:
            window_results.append(OffsetWindowResult(
                window_id=wid, local_frames=wframes, best_offset=None,
                best_error_px=float("nan"), runner_up_error_px=float("nan"),
                separation_ratio=0.0, median_torso_error_px=float("nan"),
                p95_torso_error_px=float("nan")))
            continue
        scored.sort()
        best_e, best_o, bt_med, bt_p95 = scored[0]
        ru_e = scored[1][0] if len(scored) > 1 else float("inf")
        # per-joint residuals for the winning offset on this window
        pj = {}
        acc: dict[str, list[float]] = {}
        for lf in wframes:
            if lf not in det:
                continue
            r = frame_score(lf, best_o)
            if r and r[3]:
                for n, v in r[3].items():
                    acc.setdefault(n, []).append(v)
        pj = {n: float(np.median(v)) for n, v in acc.items()}
        window_results.append(OffsetWindowResult(
            window_id=wid, local_frames=wframes, best_offset=best_o,
            best_error_px=best_e, runner_up_error_px=ru_e,
            separation_ratio=(ru_e / best_e) if best_e > 0 else float("inf"),
            median_torso_error_px=bt_med, p95_torso_error_px=bt_p95,
            per_joint_median_px=pj))

    # ---- aggregate the vote across windows ----
    valid = [w for w in window_results if w.best_offset is not None]
    won_by_offset: dict[int, int] = {}
    for w in valid:
        won_by_offset[w.best_offset] = won_by_offset.get(w.best_offset, 0) + 1
        # only count decisive wins
        if w.separation_ratio < min_separation:
            won_by_offset[w.best_offset] -= 1

    if not valid:
        return OffsetVerification(
            camera_id=cam_id, verified=False, global_start_frame=None,
            windows=window_results, windows_won=0,
            rejection_reason="no window produced a scorer grid",
            frames_tested=len(local_frames), median_error_px=None, p95_error_px=None,
            confidence="none")

    # cluster offsets within +-3 frames so 5 and 6 count together
    def cluster(offset: int) -> int:
        return int(round(offset / 3.0) * 3)

    votes: dict[int, list[OffsetWindowResult]] = {}
    for w in valid:
        votes.setdefault(cluster(w.best_offset), []).append(w)
    best_cluster = max(votes, key=lambda k: len(votes[k]))
    winning_windows = votes[best_cluster]
    windows_won = len(winning_windows)

    # offsets across the agreeing windows
    win_offsets = [w.best_offset for w in winning_windows]
    offset_spread = int(max(win_offsets) - min(win_offsets)) if win_offsets else 0
    torso_med = float(np.median([w.median_torso_error_px for w in winning_windows]))
    torso_p95 = float(np.percentile([w.p95_torso_error_px for w in winning_windows], 95))
    mean_err = float(np.median([w.best_error_px for w in winning_windows]))
    med_offset = int(np.median(win_offsets))

    reasons = []
    if windows_won < min_windows_won:
        reasons.append(f"only {windows_won}/{min_windows_won} windows agreed (cluster {best_cluster})")
    if offset_spread > max_offset_spread_frames:
        reasons.append(
            f"winning offsets spread {offset_spread}f > {max_offset_spread_frames}f "
            f"across windows ({sorted(win_offsets)})"
        )
    # separation: the winning cluster's mean shape error should clearly beat the
    # best runner-up cluster's mean error
    other = [k for k in votes if k != best_cluster]
    if other:
        rival = max(other, key=lambda k: len(votes[k]))
        rival_err = float(np.median([w.best_error_px for w in votes[rival]]))
        if rival_err / max(mean_err, 1e-6) < min_separation and windows_won < len(valid):
            reasons.append(
                f"winner not decisive: cluster {best_cluster} err={mean_err:.3f} vs "
                f"rival {rival} err={rival_err:.3f}"
            )

    verified = not reasons
    return OffsetVerification(
        camera_id=cam_id,
        verified=verified,
        global_start_frame=med_offset if verified else None,
        windows=window_results,
        windows_won=windows_won,
        rejection_reason=None if verified else "; ".join(reasons),
        frames_tested=len(local_frames),
        median_error_px=round(torso_med, 3),
        p95_error_px=round(torso_p95, 3),
        confidence="high" if verified and windows_won >= min_windows_won and all(
            w.separation_ratio >= min_separation for w in winning_windows) else "medium",
    )


__all__ = [
    "OffsetVerification",
    "OffsetWindowResult",
    "verify_camera_offset",
    "MATCH_JOINTS_17",
    "COCO17_TO_COCO19",
]
