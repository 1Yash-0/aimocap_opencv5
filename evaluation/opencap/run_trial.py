"""Run the OpenCap evaluation for one prepared trial.

Stages, each logged to ``events.jsonl``:

``calibrate``   checkerboard extrinsics with the shipped intrinsics
``crop``        person-centred crop videos (rtmlib or motion tracker)
``pose2d``      production ``aimocap multipose`` on the crop videos
``map2d``       keypoints mapped back to source pixels
``sync``        geometric synchronisation (or a previously written sync file)
``truth``       window anchor from the mocap-PnP lag scan (evaluation only)
``window``      synchronised window cut (mocap duration, or ``--window-seconds``)
``triangulate`` production ``aimocap triangulate`` with ``--fps``
``bvh``         BVH export through ``scripts/points_to_rot_bvh.py``
``score``       lag-zero accuracy and smoothness against mocap (when mocap exists)
``render``      2D overlay grid and skeleton-versus-mocap video

Reconstruction stages never read mocap.  Mocap enters only through the
``truth`` anchor (a window placement, as in the evaluation reports) and the
``score`` stage.  A sync file written by a full run can be reused with
``--sync-file``; the window is then cut from that run's offsets and anchor.

Fail-closed: any stage error is logged as ``failed`` and the command exits
with status 1.

CLI
---
python -m evaluation.opencap.run_trial --trial-dir DIR --out OUT \\
    [--cams Cam0,Cam1,Cam2,Cam3,Cam4] [--engine ort|cv2dnn] \\
    [--part-refine off|hands|face|hands_face] [--window-seconds S] [--sync-file FILE]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.opencap.calibration import calibrate_trial
from evaluation.opencap.crops import map_to_source, prepare_crops
from evaluation.opencap.dataset import read_trial
from evaluation.opencap.events import EventLog
from evaluation.opencap.mocap import load_trc
from evaluation.opencap.render import render_overlay_grid, render_skeleton_vs_mocap
from aimocap.triangulate.config import GeometryConfig
from evaluation.opencap.scoring import score_window
from evaluation.opencap.sync import (
    cut_window,
    estimate_sync,
    read_sync_file,
    relative_offsets,
    write_sync_file,
)
from evaluation.opencap.truth import resolve_truth

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINES = ("ort", "cv2dnn")
PART_REFINE_MODES = ("off", "hands", "face", "hands_face")
TRACKERS = ("rtmlib", "motion")
DEFAULT_CAMS = "Cam0,Cam1,Cam2,Cam3,Cam4"
# Use the production smoother setting so reported numbers match what users get.
EVALUATION_TEMPORAL_Q = GeometryConfig.temporal_filter_q
UNBOUNDED = 10**9


@dataclass(frozen=True)
class RunConfig:
    """Validated command-line configuration of one run."""

    trial_dir: Path
    out: Path
    cams: tuple[str, ...]
    engine: str
    part_refine: str
    window_seconds: float | None
    sync_file: Path | None
    tracker: str
    temporal_filter_q: float
    reuse_2d: bool


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description="Run the OpenCap evaluation for one prepared trial.")
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cams", default=DEFAULT_CAMS, help="comma-separated camera names")
    parser.add_argument("--engine", choices=ENGINES, default="ort")
    parser.add_argument("--part-refine", choices=PART_REFINE_MODES, default="hands_face")
    parser.add_argument("--window-seconds", type=float, default=None, help="limit the window length")
    parser.add_argument("--sync-file", type=Path, default=None, help="sync.json from an earlier run")
    parser.add_argument(
        "--tracker", choices=TRACKERS, default="motion",
        help="crop tracker: motion (frame differencing, default) or rtmlib (pose-model boxes)",
    )
    parser.add_argument(
        "--temporal-filter-q", type=float, default=EVALUATION_TEMPORAL_Q,
        help="white-jerk process noise passed to triangulate (default: the evaluation setting)",
    )
    parser.add_argument(
        "--reuse-2d", action="store_true",
        help="reuse crop videos and multipose_crop.npz from an earlier run in the same output folder",
    )
    args = parser.parse_args(argv)
    cams = tuple(c.strip() for c in args.cams.split(",") if c.strip())
    if not cams or len(set(cams)) != len(cams):
        parser.error("--cams must name distinct cameras")
    if args.window_seconds is not None and args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.temporal_filter_q <= 0:
        parser.error("--temporal-filter-q must be positive")
    return RunConfig(
        trial_dir=args.trial_dir.resolve(),
        out=args.out.resolve(),
        cams=cams,
        engine=args.engine,
        part_refine=args.part_refine,
        window_seconds=args.window_seconds,
        sync_file=None if args.sync_file is None else args.sync_file.resolve(),
        tracker=args.tracker,
        temporal_filter_q=float(args.temporal_filter_q),
        reuse_2d=bool(args.reuse_2d),
    )


def _run_cli(arguments: list[str], log_path: Path) -> None:
    """Run ``python -m aimocap.cli`` with the given arguments; raise on a non-zero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "aimocap.cli", *arguments]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        code = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if code != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        raise RuntimeError(f"aimocap {arguments[0]} exited with {code}; log {log_path}\n{tail}")


def _make_tracker(tracker: str, engine: str) -> Any:
    """Pose estimator for the rtmlib crop tracker; ``None`` selects the motion tracker."""
    if tracker == "motion":
        return None
    from aimocap.pose.infer_rtmpose import RTMPoseEstimator

    return RTMPoseEstimator(device="cuda" if engine == "ort" else "cpu")


def _window_frames(trial: dict[str, Any], trial_dir: Path, window_seconds: float | None) -> int | None:
    """Window length: the mocap duration when mocap exists, capped by ``window_seconds``."""
    frames: int | None = None
    if trial.get("has_mocap"):
        trc = load_trc(trial_dir / trial["files"]["mocap"])
        step = float(trc.rate) / float(trial["fps"])
        frames = int((len(trc.data) - 1) / step) + 1
    if window_seconds is not None:
        capped = int(round(window_seconds * float(trial["fps"])))
        frames = capped if frames is None else min(frames, capped)
    return frames


def _stage_durations(events_path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["status"] == "completed":
            durations[record["stage"]] = record["seconds"]
    return durations


def run(config: RunConfig) -> dict[str, Any]:
    """Execute every stage for ``config`` and return the metrics dictionary."""
    trial = read_trial(config.trial_dir)
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "trial.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
    unknown = [c for c in config.cams if c not in trial["cams"]]
    if unknown:
        raise ValueError(f"cameras not in trial: {unknown}")
    fps = float(trial["fps"])
    mocap_path = config.trial_dir / trial["files"]["mocap"] if trial.get("has_mocap") else None
    window_frames = _window_frames(trial, config.trial_dir, config.window_seconds)
    cams = list(config.cams)
    (out / "events.jsonl").unlink(missing_ok=True)
    log = EventLog(out / "events.jsonl")
    metrics: dict[str, Any] = {
        "schema": "opencap_run/1",
        "subject": trial["subject"],
        "trial": trial["trial"],
        "session": trial["session"],
        "config": {
            "cams": cams,
            "engine": config.engine,
            "part_refine": config.part_refine,
            "tracker": config.tracker,
            "window_seconds": config.window_seconds,
            "sync_file": None if config.sync_file is None else str(config.sync_file),
            "temporal_filter_q": config.temporal_filter_q,
            "fps": fps,
            "triangulation_preset": "production",
        },
    }
    try:
        with log.stage("calibrate"):
            metrics["calibration"] = calibrate_trial(
                config.trial_dir, cams, out / "calibration.json", out / "calibration_check.json"
            )

        sync_loaded = read_sync_file(config.sync_file) if config.sync_file else None
        ranges: dict[str, tuple[int, int]] | None = None
        if sync_loaded is not None:
            rel_loaded = relative_offsets(sync_loaded)
            anchor_loaded = int(sync_loaded["anchor"]["cam0_start_frame"])
            if window_frames is not None:
                ranges = {
                    c: (anchor_loaded + rel_loaded[c], anchor_loaded + rel_loaded[c] + window_frames)
                    for c in cams
                }

        with log.stage("crop"):
            crop_json = out / "crops" / "crop_params.json"
            if config.reuse_2d and crop_json.exists() and (out / "multipose_crop.npz").exists():
                crop_params = json.loads(crop_json.read_text(encoding="utf-8"))
                metrics["reused_2d"] = True
            else:
                estimator = _make_tracker(config.tracker, config.engine)
                crop_params = prepare_crops(
                    config.trial_dir, out / "crops", cams, ranges=ranges, estimator=estimator
                )

        with log.stage("pose2d"):
            if not metrics.get("reused_2d"):
                _run_cli(
                    [
                        "multipose",
                        *[crop_params[c]["crop_video"] for c in cams],
                        "-o", str(out / "multipose_crop.npz"),
                        "--inference-engine", config.engine,
                        "--part-refine", config.part_refine,
                        "--progress-every", "100",
                    ],
                    out / "logs" / "multipose.log",
                )

        with log.stage("map2d"):
            map_to_source(out / "multipose_crop.npz", crop_params, out / "multipose.npz", fps=fps)

        with log.stage("sync"):
            if sync_loaded is not None:
                missing = [c for c in cams if c not in sync_loaded["cameras"]]
                if missing:
                    raise ValueError(f"sync file has no offsets for {missing}")
                sync = {
                    "method": "sync_file",
                    "source": str(config.sync_file),
                    "reference": cams[0],
                    "cameras": {c: sync_loaded["cameras"][c] for c in cams},
                    "uses_mocap": False,
                }
            else:
                sync = estimate_sync(out / "multipose.npz", out / "calibration.json")
                sync["cameras"] = {c: sync["cameras"][c] for c in cams}
            rel = relative_offsets(sync)

        truth: dict[str, Any] | None = None
        with log.stage("truth"):
            if sync_loaded is not None:
                anchor = dict(sync_loaded["anchor"])
                anchor["source"] = f"sync_file:{anchor.get('source', 'unknown')}"
            elif mocap_path is not None:
                truth = resolve_truth(config.trial_dir, out / "multipose.npz", out / "calibration.json", mocap_path)
                anchor = truth["anchor"]
            else:
                anchor = {"cam0_start_frame": max(0, -min(rel.values())), "source": "first_common_frame",
                          "resolved": False}
            sync["anchor"] = anchor
            write_sync_file(out / "sync.json", sync)

        if truth is not None:
            per_camera = {}
            errors = []
            for c in cams:
                estimated = rel[c]
                true_offset = truth["cameras"][c].get("true_offset_vs_cam0")
                error = None if true_offset is None else abs(estimated - true_offset)
                per_camera[c] = {"estimated": estimated, "truth": true_offset, "abs_error_frames": error}
                if error is not None:
                    errors.append(error)
            metrics["sync_vs_truth"] = {
                "per_camera": per_camera,
                "unresolved_cameras": [c for c in cams if per_camera[c]["truth"] is None],
                "max_abs_error_frames": max(errors) if errors else None,
                "pass": bool(len(errors) == len(cams) and max(errors) <= 1),
            }
            metrics["truth"] = truth

        with log.stage("window"):
            cut = cut_window(
                out / "multipose.npz",
                out / "window.npz",
                rel,
                int(anchor["cam0_start_frame"]),
                window_frames or UNBOUNDED,
                sync_status="SYNC_FILE" if sync_loaded is not None else "GEOMETRIC_ESTIMATE",
            )
            metrics["window"] = cut

        with log.stage("triangulate"):
            _run_cli(
                [
                    "triangulate", str(out / "window.npz"), str(out / "calibration.json"),
                    "-o", str(out / "skeleton3d.npz"),
                    "--fps", f"{fps:g}",
                    "--triangulation-preset", "production",
                    "--raw", "--allow-unverified-sync",
                    "--temporal-filter-q", f"{config.temporal_filter_q:g}",
                ],
                out / "logs" / "triangulate.log",
            )

        with log.stage("bvh"):
            bvh = out / "motion.bvh"
            bvh.unlink(missing_ok=True)
            command = [sys.executable, str(REPO_ROOT / "scripts" / "points_to_rot_bvh.py"),
                       str(out / "skeleton3d.npz"), str(bvh)]
            result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            (out / "logs").mkdir(parents=True, exist_ok=True)
            (out / "logs" / "bvh.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode != 0 or not bvh.exists():
                raise RuntimeError(f"BVH export failed: {result.stderr.strip()[-400:]}")

        with log.stage("score"):
            if mocap_path is not None:
                opencap = {}
                for count in ("5", "3"):
                    rel_path = trial.get("files", {}).get("hrnet", {}).get(count)
                    if rel_path:
                        opencap[f"opencap_hrnet_{count}cam"] = config.trial_dir / rel_path
                metrics["scores"] = score_window(out / "skeleton3d.npz", mocap_path, opencap)
            else:
                metrics["scores"] = None

        with log.stage("render"):
            metrics["overlay_2d"] = render_overlay_grid(config.trial_dir, out / "window.npz", out / "overlay_2d.mp4", fps=fps)
            if mocap_path is not None:
                metrics["skeleton_vs_mocap"] = render_skeleton_vs_mocap(
                    out / "skeleton3d.npz", mocap_path, out / "skeleton_vs_mocap.mp4", lag=0
                )
    finally:
        log.close()

    metrics["stage_seconds"] = _stage_durations(out / "events.jsonl")
    metrics["total_stage_seconds"] = round(sum(metrics["stage_seconds"].values()), 3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    config = parse_args(argv)
    started = time.perf_counter()
    try:
        metrics = run(config)
    except Exception as exc:  # fail closed: the stage has already been logged as failed
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    scores = metrics.get("scores") or {}
    ours = scores.get("ours", {})
    lag0 = ours.get("lag0_procrustes") or {}
    smooth = ours.get("smoothness") or {}
    cal = metrics.get("calibration", {})
    print(
        f"{metrics['subject']} {metrics['trial']}: "
        f"lag0 median {lag0.get('median_mm', float('nan')):.2f} mm, "
        f"accel {smooth.get('acceleration_error_median_mm_per_frame2', float('nan')):.3f}, "
        f"speed p95 ratio {smooth.get('peak_speed_ratio_p95', float('nan')):.4f}, "
        f"calibration max dR {cal.get('max_dR_deg', float('nan')):.3f} deg / "
        f"centre {cal.get('max_centre_diff_mm', float('nan')):.2f} mm"
    )
    print(f"stages {metrics['total_stage_seconds']:.1f} s of work, wall {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
