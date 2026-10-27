"""aimocap command-line interface.

Entry point: ``aimocap <subcommand> [options]`` (installed via pyproject.toml).

Current subcommands:
    pose <video> [-o out.mp4] [--max-frames N] [--threshold T]
        Run 2D whole-body pose estimation over a video, write annotated output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np


# Importing aimocap top-level ensures CUDA DLLs load before onnxruntime.
import aimocap  # noqa: F401
from aimocap.calib.extrinsics import calibrate_all, align_to_floor
from aimocap.calib.io import save_calibration, load_calibration, load_calibration_bundle
from aimocap.calib.scale import apply_metric_scale
from aimocap.triangulate.service import triangulate_calibrated
from aimocap.triangulate.config import GeometryConfig, PartConfidence
from aimocap.triangulate.confidence3d import FEATURE_NAMES as CONFIDENCE3D_FEATURE_NAMES
from aimocap.triangulate.engine import LR_SWAP_GROUP_NAMES
from aimocap.math.filter import filter_skeleton3d
from aimocap.pipeline.gates import GateFailureReport, check_raw_geometry_gates


def _cli_manifest(inputs: dict[str, str | Path | None], config: dict) -> dict:
    from aimocap.pipeline.provenance import build_manifest, print_startup

    manifest = build_manifest(
        root=Path(__file__).resolve().parents[1],
        command=" ".join(sys.argv),
        inputs=inputs,
        config=config,
    )
    print_startup(manifest)
    return manifest


def _save_cli_sidecar(output: str | Path, manifest: dict) -> None:
    from aimocap.pipeline.provenance import write_sidecar

    destination = write_sidecar(output, manifest)
    print(f"Saved provenance manifest to {destination}")


def _parse_camera_map(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"camera map must use input=calibration, got {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target or source in mapping:
            raise ValueError(f"invalid or duplicate camera map entry: {value!r}")
        mapping[source] = target
    return mapping


def _parse_part_confidence(values: list[str] | None) -> dict[str, PartConfidence]:
    """Parse ``GROUP=MIN_CONF:GAMMA`` overrides for the four score groups."""
    overrides: dict[str, PartConfidence] = {}
    for value in values or []:
        group, separator, numbers = value.partition("=")
        group = group.strip().lower()
        if not separator or group not in {"body", "feet", "face", "hands"}:
            raise ValueError(
                f"part confidence must use body|feet|face|hands=MIN_CONF:GAMMA, got {value!r}"
            )
        fields = numbers.split(":")
        if len(fields) != 2:
            raise ValueError(f"part confidence needs MIN_CONF:GAMMA, got {value!r}")
        field = f"{group}_confidence"
        if field in overrides:
            raise ValueError(f"duplicate part confidence entry: {value!r}")
        overrides[field] = PartConfidence(float(fields[0]), float(fields[1]))
    return overrides


def _resolve_retarget_fps(
    npz_src: Path, pipeline_profile: str, requested_fps: float | None
) -> float:
    """Resolve CLI retarget FPS without changing the legacy v7 default."""
    if requested_fps is not None:
        return float(requested_fps)
    if pipeline_profile != "v8":
        return 30.0

    try:
        with np.load(npz_src, allow_pickle=False) as data:
            if "fps" in data:
                raw_fps = np.asarray(data["fps"])
                if raw_fps.size == 1:
                    fps = float(raw_fps.reshape(-1)[0])
                    if np.isfinite(fps) and fps > 0.0:
                        return fps
            if "timestamps" in data:
                timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
                if timestamps.size >= 2 and np.isfinite(timestamps).all():
                    duration = float(timestamps[-1] - timestamps[0])
                    if duration > 0.0:
                        fps = float(timestamps.size - 1) / duration
                        if np.isfinite(fps) and fps > 0.0:
                            return fps
    except (OSError, ValueError, TypeError):
        pass

    raise ValueError(
        "v8 retarget requires FPS metadata in the input NPZ (a valid 'fps' "
        "key or derivable 'timestamps'); provide --fps explicitly"
    )


def _positive_fps(value: object) -> float | None:
    """Return ``value`` as a positive, finite frame rate, or None if unusable."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 1:
        return None
    fps = float(arr[0])
    return fps if np.isfinite(fps) and fps > 0.0 else None


def _sync_manifest_fps(data, npz_src: Path) -> tuple[float, str] | None:
    """Read ``fps`` from the synchronization manifest recorded in the NPZ provenance."""
    if "provenance_manifest" not in data:
        return None
    try:
        provenance = json.loads(str(data["provenance_manifest"]))
    except (TypeError, ValueError):
        return None
    config = provenance.get("all_config_values") if isinstance(provenance, dict) else None
    sync_ref = config.get("sync_manifest") if isinstance(config, dict) else None
    if not isinstance(sync_ref, str) or not sync_ref:
        return None
    for candidate in (Path(sync_ref), npz_src.parent / sync_ref):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        fps = _positive_fps(payload.get("fps")) if isinstance(payload, dict) else None
        return (fps, f"sync_manifest:{candidate}") if fps is not None else None
    return None


def _resolve_triangulate_fps(
    data, npz_src: Path, requested_fps: float | None
) -> tuple[float, str]:
    """Resolve the capture rate used by triangulation and its temporal smoother.

    Precedence: explicit ``--fps``; the NPZ ``fps`` key (written by ``multipose``
    from the video metadata); a rate derived from ``timestamps`` (or
    ``requested_timestamps``); the ``fps`` of the synchronization manifest named in
    the NPZ provenance. There is no silent default: assuming 30 fps for 60 fps data
    over-smooths the trajectory. Returns ``(fps, source)``.
    """
    if requested_fps is not None:
        fps = _positive_fps(requested_fps)
        if fps is None:
            raise ValueError(f"--fps must be a positive finite frame rate, got {requested_fps!r}")
        return fps, "--fps"
    if "fps" in data:
        fps = _positive_fps(data["fps"])
        if fps is not None:
            return fps, "npz:fps"
    for key in ("timestamps", "requested_timestamps"):
        if key not in data:
            continue
        try:
            stamps = np.asarray(data[key], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if stamps.ndim != 1 or stamps.size < 2 or not np.all(np.isfinite(stamps)):
            continue
        duration = float(stamps[-1] - stamps[0])
        fps = _positive_fps((stamps.size - 1) / duration) if duration > 0.0 else None
        if fps is not None:
            return fps, f"npz:{key}"
    sync_fps = _sync_manifest_fps(data, npz_src)
    if sync_fps is not None:
        return sync_fps
    raise ValueError(
        f"cannot determine the capture frame rate of {npz_src}: the NPZ has no valid 'fps' key, "
        "no usable 'timestamps' and no referenced synchronization manifest; "
        "pass --fps <rate> explicitly (for example --fps 60)"
    )


def _cmd_pose(args: argparse.Namespace) -> int:
    from aimocap.pose.infer import PoseEstimator
    from aimocap.video import run_video, summarize

    src = Path(args.video)
    if not src.exists():
        print(f"error: input video not found: {src}", file=sys.stderr)
        return 2

    if args.output:
        dst = Path(args.output)
    else:
        dst = src.with_name(f"{src.stem}_annotated.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    manifest = _cli_manifest(
        {"input_video": src},
        {
            "threshold": args.threshold,
            "max_frames": args.max_frames,
            "pick": args.pick,
            "progress_every": args.progress_every,
            "pose_model": args.pose_model,
        },
    )

    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Threshold: {args.threshold}  max_frames: {args.max_frames or 'all'}")

    # Warm up the estimator so the first frame isn't paying for CUDA init
    # in the timing.
    kwargs = {}
    if args.pose_model:
        kwargs["pose_model"] = args.pose_model
    est = PoseEstimator(**kwargs)
    print(f"Providers: {est.active_providers}")

    stats = run_video(
        src,
        dst,
        estimator=est,
        threshold=args.threshold,
        max_frames=args.max_frames,
        pick=args.pick,
        progress_every=args.progress_every,
    )

    summary = summarize(stats)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")

    # Stability gate: a good single-person run should have a high detection
    # rate and low body-confidence variance (no flicker). Print a verdict.
    det_rate = summary.get("detection_rate", 0)
    conf_std = summary.get("body_conf_std", 0)
    print("\n=== stability verdict ===")
    print(f"  detection_rate = {det_rate:.2%}  (want > 0.95 for a clear subject)")
    print(f"  body_conf_std  = {conf_std:.3f}     (want < 0.10 for no flicker)")
    _save_cli_sidecar(dst, manifest)
    return 0


def _build_pose_estimator(
    backend: str,
    pose_model: str | None,
    simcc_decode: str = "argmax",
    inference_engine: str = "ort",
    detector_model: str | None = None,
):
    """Construct the multipose estimator for the selected backend.

    Default ('cigpose') behavior is unchanged. 'rtmpose' uses the drop-in
    RTMPoseEstimator (same Pose2D interface, identical 133-keypoint layout).
    """
    if backend == "rtmpose":
        from aimocap.pose.infer_rtmpose import RTMPoseEstimator

        return RTMPoseEstimator()
    from aimocap.pose.infer import PoseEstimator

    kwargs = {}
    if pose_model:
        kwargs["pose_model"] = pose_model
    if detector_model:
        kwargs["detector"] = detector_model
    if simcc_decode != "argmax":
        kwargs["simcc_decode"] = simcc_decode
    if inference_engine != "ort":
        kwargs["inference_engine"] = inference_engine
    if backend == "rtmw":
        kwargs["model_family"] = "rtmw"
    return PoseEstimator(**kwargs)


def _cmd_multipose(args: argparse.Namespace) -> int:
    from aimocap.pose.multi_video import run_multi_video

    sources = [Path(s) for s in args.videos]
    for src in sources:
        if not src.exists():
            print(f"error: input video not found: {src}", file=sys.stderr)
            return 2
            
    npz_dst = Path(args.output)
    grid_dst = Path(args.grid_video) if args.grid_video else None

    print(f"Inputs : {[str(s) for s in sources]}")
    print(f"Output : {npz_dst}")
    if grid_dst:
        print(f"Grid   : {grid_dst}")

    manifest_config = {
        "threshold": args.threshold,
        "max_frames": args.max_frames,
        "progress_every": args.progress_every,
        "pose_model": args.pose_model,
        "sync_manifest": args.sync_manifest,
        "require_sync": args.require_sync,
        "single_subject": not args.allow_multiple_subjects,
        "tracker": args.tracker,
        "part_refine": args.part_refine,
        "inference_engine": getattr(args, "inference_engine", "ort"),
        "cv2_version": __import__("cv2").__version__,
    }
    pose_backend = getattr(args, "pose_backend", "rtmw")
    manifest_config["pose_backend"] = pose_backend
    manifest_config["crop_mode"] = args.crop_mode
    manifest_config["simcc_decode"] = args.simcc_decode
    if args.part_refine != "off":
        from aimocap.pose.part_refine import model_hashes
        from aimocap.pose.score_calibration import load_part_calibration

        manifest_config["part_refine_models"] = model_hashes(args.part_refine)
        _part_cal = load_part_calibration()
        manifest_config["part_score_calibration"] = (
            f"{_part_cal['_file']} sha256={_part_cal['_sha256']}"
        )
    if pose_backend == "rtmw":
        from aimocap.pose.score_calibration import load_calibration

        _cal = load_calibration()
        manifest_config["score_calibration"] = f"{_cal['_file']} sha256={_cal['_sha256']}"
    manifest = _cli_manifest(
        {f"input_video_{i}": src for i, src in enumerate(sources)},
        manifest_config,
    )
        
    inference_engine = getattr(args, "inference_engine", "ort")
    est = _build_pose_estimator(
        getattr(args, "pose_backend", "cigpose"),
        args.pose_model,
        args.simcc_decode,
        inference_engine,
        getattr(args, "detector_model", None),
    )
    print(f"Providers: {est.active_providers}")
    
    run_multi_video(
        sources,
        npz_dst,
        grid_dst=grid_dst,
        estimator=est,
        threshold=args.threshold,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
        sync_manifest=args.sync_manifest,
        require_sync=args.require_sync,
        single_subject=not args.allow_multiple_subjects,
        tracker=args.tracker,
        crop_mode=args.crop_mode,
        simcc_decode=args.simcc_decode,
        part_refine=args.part_refine,
        inference_engine=inference_engine,
    )
    from aimocap.pipeline.provenance import add_npz_manifest
    add_npz_manifest(npz_dst, manifest)
    _save_cli_sidecar(npz_dst, manifest)
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    sources = [Path(s) for s in args.videos]
    for src in sources:
        if not src.exists():
            print(f"error: input video not found: {src}", file=sys.stderr)
            return 2

    manifest = _cli_manifest(
        {f"input_video_{i}": src for i, src in enumerate(sources)},
        {"camera_ids": args.camera_ids, "fps": args.fps},
    )
    from aimocap.sync.engine import synchronize

    camera_ids = args.camera_ids or [src.stem for src in sources]
    try:
        table = synchronize(sources, camera_ids=camera_ids, fps=args.fps)
    except Exception as exc:
        print(f"error: synchronization failed: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = table.to_dict()
    payload["status"] = "VERIFIED_AUDIO_CLAP"
    payload["source_paths"] = {cid: str(src) for cid, src in zip(camera_ids, sources)}
    payload["provenance_manifest"] = manifest
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved synchronization manifest to {out_path}")
    _save_cli_sidecar(out_path, manifest)
    print(f"Reference camera: {table.reference_id}")
    for cid, item in table.cameras.items():
        print(f"  {cid}: offset={item.offset_ms:.2f} ms, frame_offset={item.frame_offset:.3f}, confidence={item.confidence:.2f}")
    return 0


def _cmd_panoptic_sync(args: argparse.Namespace) -> int:
    from aimocap.pipeline.panoptic_sync import (
        build_panoptic_sync_manifest,
        save_panoptic_sync_manifest,
        normalize_camera_id,
        PanopticSyncError,
    )

    seq_dir = Path(args.sequence_dir)
    if not seq_dir.exists():
        print(f"error: Panoptic sequence dir not found: {seq_dir}", file=sys.stderr)
        return 2
    if len(args.videos) < 2:
        print("error: at least two HD videos are required", file=sys.stderr)
        return 2

    global_starts = {}
    for spec in args.global_start or []:
        if "=" not in spec:
            print(f"error: --global-start must be CAM=START_FRAMES, got {spec!r}", file=sys.stderr)
            return 2
        cam_key, start_s = (part.strip() for part in spec.split("=", 1))
        try:
            global_starts[normalize_camera_id(cam_key)] = int(start_s)
        except (PanopticSyncError, ValueError) as exc:
            print(f"error: invalid --global-start entry {spec!r}: {exc}", file=sys.stderr)
            return 2

    cameras = []
    for v in args.videos:
        p = Path(v)
        try:
            cid = normalize_camera_id(p.stem)
        except PanopticSyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if cid in global_starts:
            cameras.append((cid, str(p), global_starts[cid]))
        else:
            cameras.append((cid, str(p)))

    try:
        manifest = build_panoptic_sync_manifest(
            seq_dir,
            sequence=args.sequence,
            fps=args.fps,
            cameras=cameras,
            reference_camera=args.reference,
            verify_gt=not args.no_verify_gt,
            window_start_global_frame=args.window_start_global,
        )
    except PanopticSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    gm = _cli_manifest(
        {"sequence_dir": seq_dir if not getattr(args, "skip_sequence_hash", False) else None,
         **{f"video_{i}": v for i, v in enumerate(args.videos)}},
        {"sequence": args.sequence, "fps": args.fps, "reference": args.reference},
    )
    if getattr(args, "skip_sequence_hash", False):
        # D3: explicit opt-out from hashing the (potentially multi-GB)
        # sequence directory. The hash exists for input-integrity provenance;
        # skipping it is recorded in the manifest (sequence_dir: null).
        print("skipped: --skip-sequence-hash (sequence_dir not hashed; recorded in manifest)")
    manifest["build_provenance_manifest"] = gm
    out = save_panoptic_sync_manifest(manifest, args.output)
    print(f"Saved Panoptic frame-sync manifest to {out}")
    print(f"Status: {manifest['status']}  reference={manifest['reference_id']}")
    probe = manifest["provenance"].get("gt_frame_probe")
    if probe:
        print(
            f"GT probe: {probe['num_present']}/{probe['num_requested']} frames present"
            f" (bounds={probe.get('gt_frame_bounds')})"
            + (f", missing={probe['missing'][:8]}" if probe["missing"] else "")
        )
    for cid, c in manifest["cameras"].items():
        print(f"  {cid}: offset={c['offset_ms']:.3f} ms frame_offset={c['frame_offset']:.3f}")
    _save_cli_sidecar(out, gm)
    return 0


def _cmd_calib_board(args: argparse.Namespace) -> int:
    image_dirs = [Path(value) for value in args.image_dirs]
    for image_dir in image_dirs:
        if not image_dir.exists():
            print(f"error: board image path not found: {image_dir}", file=sys.stderr)
            return 2
    if args.camera_names and len(args.camera_names) != len(image_dirs):
        print("error: --camera-names must contain one name per image directory", file=sys.stderr)
        return 2

    manifest = _cli_manifest(
        {f"board_images_{i}": image_dir for i, image_dir in enumerate(image_dirs)},
        {
            "board_size": [args.board_cols, args.board_rows],
            "square_size": args.square_size,
            "min_frames": args.min_frames,
            "camera_names": args.camera_names,
        },
    )
    from aimocap.calib.board import calibrate_checkerboard

    try:
        result = calibrate_checkerboard(
            image_dirs,
            board_size=(args.board_cols, args.board_rows),
            square_size=args.square_size,
            camera_names=args.camera_names,
            min_frames=args.min_frames,
        )
    except Exception as exc:
        print(f"error: checkerboard calibration failed: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    save_calibration(
        out_path,
        result["K_list"],
        result["extrinsics"],
        image_sizes=result["image_sizes"],
        dist_coeffs=result["dist_coeffs"],
        camera_names=result["camera_names"],
        units="meters",
        calibration_source="checkerboard",
        accuracy_class="production",
        scale_source="checkerboard",
        distortion_treatment="distorted_pixels",
        provenance_manifest=manifest,
    )
    print(f"Saved checkerboard calibration to {out_path}")
    print(f"Per-camera RMS: {result['rms_by_camera']}")
    print(f"Stereo RMS: {result['stereo_rms']}")
    _save_cli_sidecar(out_path, manifest)
    return 0


def _cmd_calib_auto(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
        
    print(f"Loading keypoints from {npz_src}")
    data = np.load(npz_src)
    keypoints = data["keypoints"]  # (F, C, 133, 2)
    scores = data["scores"]        # (F, C, 133)
    
    num_frames, num_cameras, _, _ = keypoints.shape
    print(f"Found {num_frames} frames, {num_cameras} cameras.")
    manifest = _cli_manifest(
        {"input_pose_npz": npz_src, **{f"input_video_{i}": value for i, value in enumerate(args.videos or [])}},
        {"threshold": args.threshold, "videos": args.videos, "allow_uncalibrated": args.allow_uncalibrated},
    )
    
    if "image_sizes" in data:
        img_sizes = data["image_sizes"]
    else:
        img_sizes = np.array([[1920, 1080]] * num_cameras)
        
    from aimocap.calib.intrinsics import guess_intrinsics, extract_intrinsics
    from aimocap.calib.focal_search import focal_grid_search

    K_list = []
    has_guess = False
    camera_names = None
    if "camera_names" in data:
        camera_names = [str(x) for x in data["camera_names"]]
    for i in range(num_cameras):
        w, h = img_sizes[i]
        
        if args.videos and i < len(args.videos):
            try:
                K, method = extract_intrinsics(
                    args.videos[i], w, h, allow_guess=args.allow_uncalibrated
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 3
            print(f"Cam {i}: Intrinsics via {method} -> focal ~ {K[0,0]:.1f}")
            if method == "guess":
                has_guess = True
        else:
            K = guess_intrinsics(w, h)
            print(f"Cam {i}: Intrinsics via guess -> focal ~ {K[0,0]:.1f}")
            has_guess = True
        K_list.append(K)

    if has_guess and not args.allow_uncalibrated:
        print(
            "error: calibration requires real intrinsics. One or more cameras "
            "only have guessed focal length; provide calibration metadata or "
            "rerun with --allow-uncalibrated for exploratory output.",
            file=sys.stderr,
        )
        return 3
        
    if has_guess:
        print("\nSome cameras lack EXIF focal length data. Running focal grid search...")
        try:
            K_list, extrinsics, best_ratio = focal_grid_search(
                keypoints, scores, img_sizes, min_conf=args.threshold, steps=10, stride=5
            )
        except Exception as e:
            print(f"Focal grid search failed: {e}. Falling back to initial guess.")
            print("Solving extrinsics...")
            extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=args.threshold)
            print("Aligning to floor...")
            extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=args.threshold)
    else:
        print("Solving extrinsics...")
        try:
            extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=args.threshold)
            print("Aligning to floor...")
            extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=args.threshold)
        except Exception as e:
            print(f"error during calibration: {e}", file=sys.stderr)
            return 1
        
    out_path = Path(args.output)
    save_calibration(
        out_path,
        K_list,
        extrinsics,
        image_sizes=[(int(s[0]), int(s[1])) for s in img_sizes],
        camera_names=camera_names,
        units="arbitrary",
        calibration_source=("markerless_pose_focal_search" if has_guess else "markerless_pose_extrinsics"),
        accuracy_class="exploratory",
        scale_source="unscaled",
        distortion_treatment="unknown",
        provenance_manifest=manifest,
    )
    print(f"Saved calibration to {out_path}")
    _save_cli_sidecar(out_path, manifest)
    return 0


def _cmd_triangulate(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    calib_src = Path(args.calib)
    
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
    if not calib_src.exists():
        print(f"error: calibration json not found: {calib_src}", file=sys.stderr)
        return 2

    try:
        camera_map = _parse_camera_map(args.camera_map)
        part_overrides = _parse_part_confidence(args.part_confidence)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

        
    print(f"Loading keypoints from {npz_src}")
    data = np.load(npz_src)
    keypoints = data["keypoints"]
    scores = data["scores"]
    try:
        fps, fps_source = _resolve_triangulate_fps(data, npz_src, getattr(args, "fps", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Capture fps: {fps:g} (source: {fps_source})")

    from aimocap.pipeline.sync_contract import SyncContractError, validate_sync_contract

    sync_status = data["sync_status"] if "sync_status" in data else None
    sync_validation = None
    if not args.allow_unverified_sync:
        try:
            sync_validation = validate_sync_contract(
                sync_status,
                data["source_timestamps"] if "source_timestamps" in data else None,
                data["requested_timestamps"] if "requested_timestamps" in data else None,
                fps=fps,
            )
        except (KeyError, TypeError, ValueError, SyncContractError) as exc:
            print(
                f"error: input pose data does not satisfy the synchronization contract ({exc}); "
                "run audio synchronization and multipose with --sync-manifest, or "
                "explicitly pass --allow-unverified-sync for exploratory output.",
                file=sys.stderr,
            )
            return 3
    else:
        print("WARNING: synchronization contract explicitly overridden for exploratory output.")

    if args.filter_2d and not args.allow_unsafe_geometry:
        print(
            "error: --filter-2d interpolates observations and is diagnostic-only; "
            "pair it with --allow-unsafe-geometry for exploratory output.",
            file=sys.stderr,
        )
        return 3

    if args.filter_2d:
        print("Applying Savitzky-Golay filter to 2D keypoints before triangulation...")
        from scipy.signal import savgol_filter
        
        # 1. Interpolate missing points (scores < threshold) to prevent savgol overshoot
        F, C, J, _ = keypoints.shape
        for c in range(C):
            for j in range(J):
                valid = scores[:, c, j] >= args.threshold
                if np.any(valid) and not np.all(valid):
                    # Linearly interpolate X and Y
                    valid_idx = np.where(valid)[0]
                    invalid_idx = np.where(~valid)[0]
                    keypoints[invalid_idx, c, j, 0] = np.interp(invalid_idx, valid_idx, keypoints[valid_idx, c, j, 0])
                    keypoints[invalid_idx, c, j, 1] = np.interp(invalid_idx, valid_idx, keypoints[valid_idx, c, j, 1])
                    
        # 2. Apply filter
        window = 15
        poly = 3
        if keypoints.shape[0] > window:
            keypoints = savgol_filter(keypoints, window_length=window, polyorder=poly, axis=0)
        else:
            print(f"Warning: Sequence length ({keypoints.shape[0]}) too short for 2D filtering (requires > {window}).")
            
    if "camera_names" in data:
        camera_names = [str(x) for x in data["camera_names"]]
        print(f"Using camera mapping: {camera_names}")
    else:
        camera_names = None
        
    print(f"Loading cameras from {calib_src}")
    try:
        bundle = load_calibration_bundle(
            calib_src,
            camera_names=camera_names,
            require_metadata=args.require_calibration_metadata,
            camera_map=camera_map,
            observation_image_sizes=(data["image_sizes"] if "image_sizes" in data else None),
        )
    except (ValueError, KeyError) as exc:
        print(f"error: invalid calibration: {exc}", file=sys.stderr)
        return 3
    accuracy_class = str(bundle.metadata.get("accuracy_class", "unknown"))
    if accuracy_class == "exploratory" and not args.allow_exploratory_calibration:
        print(
            "error: calibration is marked exploratory and cannot feed the "
            "production triangulation gate; pass --allow-exploratory-calibration "
            "only for diagnostic output.",
            file=sys.stderr,
        )
        return 3
    if accuracy_class == "unknown":
        print("WARNING: calibration has no accuracy_class metadata; treating it as legacy input.")
    if bundle.metadata.get("projection_convention") != "world_to_camera":
        print(
            "error: calibration projection_convention must be world_to_camera; "
            "convert the calibration explicitly before triangulation.",
            file=sys.stderr,
        )
        return 3
    if bundle.metadata.get("camera_coordinate_system") != "opencv_right_handed_y_down_z_forward":
        print(
            "error: calibration camera_coordinate_system is not the OpenCV "
            "camera frame expected by triangulation.", file=sys.stderr
        )
        return 3
    K_list, extrinsics = bundle.K_list, bundle.extrinsics
    config_kwargs = dict(
        min_detection_conf=args.threshold,
        max_reprojection_error_px=args.reproj_threshold,
        hard_reprojection_error_px=args.hard_reproj_threshold,
        min_ray_angle_deg=args.min_ray_angle,
        min_depth_m=args.min_depth,
        robust_loss=args.robust_loss,
        f_scale=args.f_scale,
    )
    for value, field in (
        (args.camera_policy, "camera_policy"),
        (args.soft_scale_px, "soft_scale_px"),
        (args.soft_loo_ratio, "soft_loo_ratio"),
        (args.soft_ema_alpha, "soft_ema_alpha"),
        (args.soft_iterations, "soft_iterations"),
        (args.soft_support_min_trust, "soft_support_min_trust"),
        (args.soft_support_gate, "soft_support_gate"),
        (args.soft_temporal_prior, "soft_temporal_prior"),
        (args.soft_prior_w_temporal, "soft_prior_w_temporal"),
        (args.soft_prior_w_anat, "soft_prior_w_anat"),
        (args.soft_prior_velocity, "soft_prior_velocity"),
        (args.soft_prior_downweight, "soft_prior_downweight"),
        (args.soft_prior_lockin_scale, "soft_prior_lockin_scale"),
        (args.soft_prior_lockin_frames, "soft_prior_lockin_frames"),
        (args.temporal_spike_gate, "temporal_spike_gate"),
        (args.temporal_mad_window, "temporal_mad_window"),
        (args.temporal_mad_scale, "temporal_mad_scale"),
        (args.temporal_mad_min_samples, "temporal_mad_min_samples"),
        (args.temporal_maneuver_cap, "temporal_maneuver_cap"),
        (args.lr_swap_ratio, "lr_swap_ratio"),
        (args.temporal_filter, "temporal_filter"),
        (args.temporal_filter_sigma, "temporal_filter_sigma_m"),
        (args.temporal_filter_q, "temporal_filter_q"),
        (args.temporal_filter_maneuver_boost, "temporal_filter_maneuver_boost"),
        (args.temporal_filter_fast_boost, "temporal_filter_fast_boost"),
        (args.temporal_filter_max_gap_s, "temporal_filter_max_gap_s"),
        (args.confidence3d, "confidence3d"),
    ):
        if value is not None:
            config_kwargs[field] = value
    if args.lr_swap_fix:
        config_kwargs["lr_swap_fix"] = True
    try:
        geometry_config = GeometryConfig.triangulation_preset(
            args.triangulation_preset, **config_kwargs
        )
    except ValueError as exc:
        print(f"error: invalid triangulation settings: {exc}", file=sys.stderr)
        return 2
    if args.confidence_preset == "legacy":
        geometry_config = replace(
            geometry_config,
            body_confidence=PartConfidence(0.5, 1.0),
            feet_confidence=PartConfidence(0.5, 1.0),
            face_confidence=PartConfidence(0.5, 1.0),
            hands_confidence=PartConfidence(0.5, 1.0),
        )
    if part_overrides:
        geometry_config = replace(geometry_config, **part_overrides)

    manifest = _cli_manifest(
        {"input_pose_npz": npz_src, "calibration": calib_src},
        {
            "fps": fps,
            "fps_source": fps_source,
            "triangulation_preset": args.triangulation_preset,
            "geometry_config": geometry_config.to_dict(),
            "filter_2d": args.filter_2d,
            "raw": args.raw,
            "legacy_kinematics": getattr(args, "legacy_kinematics", False),
            "floor_transform": getattr(args, "floor_transform", None),
            "save_floor_skeleton": getattr(args, "save_floor_skeleton", False),
            "allow_unverified_sync": args.allow_unverified_sync,
            "allow_exploratory_calibration": args.allow_exploratory_calibration,
            "camera_map": camera_map,
        },
    )

    
    print("Triangulating sequence...")
    tri = triangulate_calibrated(
        keypoints,
        scores,
        bundle,
        geometry_config=geometry_config,
        image_sizes=data["image_sizes"] if "image_sizes" in data else None,
        allow_exploratory=args.allow_exploratory_calibration,
        fps=fps,
    )
    skeleton3d = tri.points3d
    finite_reproj = tri.reprojection_error_px[np.isfinite(tri.reprojection_error_px)]
    if finite_reproj.size:
        # NOTE (WP2.C): this line aggregates over ALL joints. The raw-geometry
        # gate below aggregates over the core-23 body+feet subset only, so the
        # two medians legitimately differ (observed: 4.54 vs 6.64 px).
        print(
            "Triangulation reprojection error (all joints; gate uses core-23 subset): "
            f"median={np.median(finite_reproj):.2f}px "
            f"p90={np.percentile(finite_reproj, 90):.2f}px "
            f"p95={np.percentile(finite_reproj, 95):.2f}px"
        )
    supported = tri.num_inliers[tri.num_inliers > 0]
    mean_inliers = float(np.mean(supported)) if supported.size else 0.0
    print(
        "Triangulation support: "
        f"mean_inliers={mean_inliers:.2f} "
        f"valid_points={np.count_nonzero(tri.num_inliers >= 2)}/{tri.num_inliers.size}"
    )
    if hasattr(tri, 'low_quality_mask') and tri.low_quality_mask is not None:
        num_low_quality = np.sum(tri.low_quality_mask)
        if num_low_quality > 0:
            print(f"WARNING: Detected {num_low_quality} frames with extremely narrow baseline (< 35 deg). These frames may have unstable depth.")

    raw_gate = check_raw_geometry_gates(
        tri,
        str(manifest.get("AIMOCAP_CALIBRATION_HASH", "unknown")),
        str(manifest.get("AIMOCAP_INPUT_HASH", "unknown")),
        camera_policy=geometry_config.camera_policy,
        support_min_trust=geometry_config.soft_support_min_trust,
        hard_inlier_threshold_px=geometry_config.hard_reprojection_error_px,
        soft_fail=True,
    )
    if isinstance(raw_gate, GateFailureReport):
        manifest["raw_geometry_gate"] = {
            "status": "FAIL",
            "failing_gate": raw_gate.failing_gate,
            "failure_mode": raw_gate.failure_mode,
            "coverage_fraction": float(raw_gate.coverage_fraction),
            "camera_policy": geometry_config.camera_policy,
            "support_min_trust": geometry_config.soft_support_min_trust,
            "gates": raw_gate.evidence.gate_results,
        }
        print(raw_gate.summary(), file=sys.stderr)
        if not args.raw and not args.allow_unsafe_geometry:
            print(
                "error: raw geometry gate failed; inspect --raw diagnostics before "
                "fitting, or pass --allow-unsafe-geometry for an explicit override.",
                file=sys.stderr,
            )
            return 4
    else:
        manifest["raw_geometry_gate"] = {
            "status": raw_gate.status,
            "coverage_fraction": float(raw_gate.coverage_fraction),
            "camera_policy": geometry_config.camera_policy,
            "support_min_trust": geometry_config.soft_support_min_trust,
            "gates": raw_gate.gate_results,
        }

    from aimocap.math.coords import (
        FloorTransform,
        CAMERA_WORLD_FRAME,
        FLOOR_WORLD_FRAME,
    )

    floor_transform_arg = getattr(args, "floor_transform", None)
    if floor_transform_arg:
        floor_transform_path = Path(floor_transform_arg)
        if not floor_transform_path.exists():
            print(f"error: floor transform file not found: {floor_transform_path}", file=sys.stderr)
            return 2
        floor_transform = FloorTransform.from_json(floor_transform_path)
    else:
        floor_transform = FloorTransform.identity(
            units=str(bundle.metadata.get("units", "meters")),
            grounding_status="unregistered",
            source="none",
        )

    # Diagnostics shared verbatim by the raw and scaled output artifacts.
    base_diag = {
        "skeleton3d_frame": np.asarray(CAMERA_WORLD_FRAME),
        "floor_transform": np.asarray(json.dumps(floor_transform.to_dict(), sort_keys=True)),
        "floor_transform_grounding_status": np.asarray(floor_transform.grounding_status),
        "triangulation_candidate_skeleton3d": tri.candidate_points3d,
        "confidence": tri.confidence,
        "triangulation_candidate_inlier_mask": tri.candidate_inlier_mask,
        "triangulation_inlier_mask": tri.inlier_mask,
        "triangulation_support_mask": tri.support_mask,
        "triangulation_camera_trust": tri.camera_trust,
        "triangulation_reprojection_error_px": tri.reprojection_error_px,
        "triangulation_candidate_num_inliers": tri.candidate_num_inliers,
        "triangulation_num_inliers": tri.num_inliers,
        "triangulation_ray_angle_deg": tri.ray_angle_deg,
        "triangulation_observed_camera_count": tri.observed_camera_count,
        "triangulation_positive_depth_mask": tri.positive_depth_mask,
        "triangulation_observation_provenance": tri.observation_provenance,
        "triangulation_camera_rejection_reason": tri.camera_rejection_reason,
        "triangulation_lr_swap_mask": tri.lr_swap_mask,
        "triangulation_lr_swap_groups": np.asarray(LR_SWAP_GROUP_NAMES),
        "geometry_config": np.asarray(json.dumps(geometry_config.to_dict(), sort_keys=True)),
        "calibration_metadata": np.asarray(json.dumps(bundle.metadata, sort_keys=True, default=str)),
        "provenance_manifest": np.asarray(json.dumps(manifest, sort_keys=True, default=str)),
    }
    if tri.confidence3d is not None:
        base_diag["confidence3d"] = tri.confidence3d
        base_diag["confidence3d_features"] = tri.confidence3d_features
        base_diag["confidence3d_feature_names"] = np.asarray(CONFIDENCE3D_FEATURE_NAMES)

    if args.raw:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dict = {
            "skeleton3d": tri.points3d,
            "scale_factor": np.asarray(1.0),
            "units": np.asarray(bundle.metadata.get("units", "unknown")),
            "coordinate_space": np.asarray("internal_y_up_right_handed_z_back"),
            "sync_validation": np.asarray(json.dumps(sync_validation or {}, sort_keys=True)),
            **base_diag,
            **{key: data[key] for key in ("timestamps", "source_frame_indices", "source_timestamps", "sync_status") if key in data},
            **{key: data[key] for key in ("requested_timestamps", "fps", "sync_camera_ids", "sync_reference_camera") if key in data},
            **({"camera_names": np.array(camera_names, dtype=str)} if camera_names is not None else {}),
        }
        if getattr(args, "save_floor_skeleton", False) or floor_transform.is_registered:
            raw_dict["skeleton3d_floor"] = floor_transform.transform_points(tri.points3d, current_frame=CAMERA_WORLD_FRAME)
            raw_dict["skeleton3d_floor_frame"] = np.asarray(FLOOR_WORLD_FRAME)
        np.savez(out_path, **raw_dict)
        print(f"Saved raw triangulation to {out_path} with shape {tri.points3d.shape}")
        _save_cli_sidecar(out_path, manifest)
        return 0
    
    from aimocap.calib.scale import resolve_scale_policy, ScaleContractError

    try:
        scale_policy = resolve_scale_policy(
            bundle.metadata,
            allow_exploratory=args.allow_exploratory_calibration,
        )
    except ScaleContractError as exc:
        print(f"error: scale contract violation: {exc}", file=sys.stderr)
        return 3

    if scale_policy.action == "PRESERVE_METRIC":
        print(
            f"Preserving authoritative metric scale from calibration ({scale_policy.scale_source}); "
            "anthropometric scale estimation bypassed."
        )
        scale_factor = 1.0
    elif scale_policy.action == "ESTIMATE_ANTHROPOMETRIC":
        print("Calibration is exploratory/unscaled; applying anthropometric scale estimation...")
        skeleton3d, _, scale_factor = apply_metric_scale(skeleton3d, extrinsics)
        print(f"Computed metric scale factor: {scale_factor:.3f}")
    else:
        print(f"error: unexpected scale policy action {scale_policy.action!r}", file=sys.stderr)
        return 3
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "skeleton3d": skeleton3d,
        "scale_factor": scale_factor,
        "scale_source": np.asarray(scale_policy.scale_source),
        "scale_action": np.asarray(scale_policy.action),
        "units": np.asarray(bundle.metadata.get("units", "m")),
        **base_diag,
    }

    if getattr(args, "save_floor_skeleton", False) or floor_transform.is_registered:
        save_dict["skeleton3d_floor"] = floor_transform.transform_points(skeleton3d, current_frame=CAMERA_WORLD_FRAME)
        save_dict["skeleton3d_floor_frame"] = np.asarray(FLOOR_WORLD_FRAME)

    if getattr(args, "filter_f4", False):
        print("Applying authoritative camera-world F4 temporal filter to canonical skeleton3d...")
        skeleton3d_refined_cam = filter_skeleton3d(skeleton3d)
        save_dict["skeleton3d_refined_camera"] = skeleton3d_refined_cam
        save_dict["skeleton3d_refined_camera_frame"] = np.asarray(CAMERA_WORLD_FRAME)

        if getattr(args, "save_floor_skeleton", False) or floor_transform.is_registered:
            save_dict["skeleton3d_refined_floor"] = floor_transform.transform_points(
                skeleton3d_refined_cam, current_frame=CAMERA_WORLD_FRAME
            )
            save_dict["skeleton3d_refined_floor_frame"] = np.asarray(FLOOR_WORLD_FRAME)
            save_dict["skeleton3d_refined_floor_provenance"] = np.asarray(
                json.dumps(
                    {
                        "parent_artifact": "skeleton3d_refined_camera",
                        "registration_epoch_id": floor_transform.epoch_id,
                        "grounding_status": floor_transform.grounding_status,
                        "validation_status": floor_transform.validation_status,
                        "source_frame": CAMERA_WORLD_FRAME,
                        "target_frame": FLOOR_WORLD_FRAME,
                        "units": str(bundle.metadata.get("units", "m")),
                        "transform_version": floor_transform.transform_version,
                    },
                    sort_keys=True,
                )
            )

    if getattr(args, "legacy_kinematics", False):
        print("Computing legacy 17-joint kinematic stick model via WholeBody133ToManny17Adapter...")
        from aimocap.math.kinematics import compute_median_bone_lengths, fit_skeleton_sequence, forward_kinematics_sequence, CLUSTER_DEFS
        from aimocap.math.filter import filter_params_one_euro_quaternion
        from aimocap.math.clusters import reject_cluster_outliers_anchor_distance, anchor_cluster_bone_constrained, filter_cluster_relative, fill_cluster_gaps
        from aimocap.retarget.adapter import WholeBody133ToManny17Adapter

        adapter = WholeBody133ToManny17Adapter()
        body3d, _ = adapter.adapt(skeleton3d)
        bone_lengths = compute_median_bone_lengths(body3d)
        body3d_rigid, params_seq = fit_skeleton_sequence(body3d, bone_lengths)
        params_seq_filtered = filter_params_one_euro_quaternion(params_seq, fps=fps)
        body3d_final = forward_kinematics_sequence(params_seq_filtered, bone_lengths)
        body3d_final_17 = body3d_final[:, :17, :]
        cleaned_3d = reject_cluster_outliers_anchor_distance(skeleton3d, body3d_final_17, CLUSTER_DEFS)
        anchored_3d = anchor_cluster_bone_constrained(cleaned_3d, body3d_final_17, CLUSTER_DEFS)
        filtered_clusters = filter_cluster_relative(anchored_3d, body3d_final_17, CLUSTER_DEFS, fps=fps)
        final_clusters = fill_cluster_gaps(filtered_clusters, body3d_final_17, CLUSTER_DEFS, max_gap=15)
        legacy_skeleton = np.copy(final_clusters)
        legacy_skeleton[:, :17, :] = body3d_final_17
        save_dict["legacy_kinematic_skeleton3d"] = legacy_skeleton

    if camera_names is not None:
        save_dict["camera_names"] = np.array(camera_names, dtype=str)
    for key in (
        "timestamps", "fps", "source_frame_indices", "source_timestamps",
        "requested_timestamps", "sync_status", "sync_camera_ids",
        "sync_reference_camera",
    ):
        if key in data:
            save_dict[key] = data[key]
        
    np.savez(out_path, **save_dict)
    print(f"Saved 3D sequence to {out_path} with shape {skeleton3d.shape}")
    _save_cli_sidecar(out_path, manifest)
    
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    manifest = _cli_manifest(
        {"input_skeleton_npz": args.npz, "calibration": args.cameras},
        {"skip_frames": args.skip_frames},
    )
    print(f"Loading 3D skeleton from {args.npz}")
    data = np.load(args.npz)
    skeleton3d = data["skeleton3d"]
    
    if args.skip_frames > 0:
        print(f"Skipping first {args.skip_frames} frames...")
        skeleton3d = skeleton3d[args.skip_frames:]
    
    camera_names = None
    if "camera_names" in data:
        camera_names = [str(x) for x in data["camera_names"]]
        print(f"Using camera mapping: {camera_names}")
        
    print(f"Loading cameras from {args.cameras}")
    K_list, extrinsics = load_calibration(args.cameras, camera_names=camera_names)
    
    if "scale_factor" in data:
        scale_factor = float(data["scale_factor"])
        print(f"Cameras scaled by loaded factor: {scale_factor:.3f}")
        extrinsics = [(R, t * scale_factor) for R, t in extrinsics]
    
    out_path = Path(args.output)
    print(f"Rendering to {out_path}...")
    from aimocap.viz.plot3d import plot_scene
    plot_scene(extrinsics, skeleton3d, out_path, animate=True)
    print("Done rendering.")
    _save_cli_sidecar(out_path, manifest)
    return 0


def _cmd_retarget(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    fbx_src = Path(args.fbx)
    out_path = Path(args.output)
    
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
    if not fbx_src.exists():
        print(f"error: FBX rig not found: {fbx_src}", file=sys.stderr)
        return 2

    if args.ik_max_nfev is not None:
        if args.ik_max_nfev <= 0:
            print("error: --ik-max-nfev must be positive", file=sys.stderr)
            return 2
        os.environ["AIMOCAP_IK_MAX_NFEV"] = str(args.ik_max_nfev)

    try:
        effective_fps = _resolve_retarget_fps(
            npz_src, args.pipeline_profile, args.fps
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest = _cli_manifest(
        {"input_skeleton_npz": npz_src, "target_fbx": fbx_src, "pose2d_npz": args.pose2d_npz},
        {
            "fps": effective_fps,
            "arm_orientation_rewrite": args.enable_arm_orientation_rewrite,
            "ik_max_nfev": args.ik_max_nfev,
            "target_conditioning": args.target_conditioning,
            "pipeline_profile": args.pipeline_profile,
            "foot_landmark_model": args.foot_landmark_model,
            "foot_landmark_weight": args.foot_landmark_weight,
            "face_position_weight": args.face_position_weight,
            "head_target_mode": args.head_target_mode,
            "limb_twist_anchor": args.limb_twist_anchor,
            "two_pass_limbs": args.two_pass_limbs,
            "two_pass_pass1_stride": args.two_pass_pass1_stride,
            "two_pass_pass1_max_iter": args.two_pass_pass1_max_iter,
            "smoothed_limbs": args.smoothed_limbs,
            "limb_twist_pin": args.limb_twist_pin,
            "limb_hinge": args.limb_hinge,
            "twist_limit_mode": args.twist_limit_mode,
            "limit_projection": args.limit_projection,
            "soft_limit_joints": args.soft_limit_joints,
            "ankle_twist_limit_deg": args.ankle_twist_limit_deg,
            "redundant_dof_prior": args.redundant_dof_prior,
            "foot_gate_reference": args.foot_gate_reference,
            "leg_stabilization": args.leg_stabilization_override,
            "foot_lock": args.foot_lock_override,
        },
    )
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from aimocap.retarget.engine import retarget_to_fbx  # needs ufbx; only the retarget stage uses it
    retarget_to_fbx(
        str(npz_src),
        str(fbx_src),
        str(out_path),
        fps=effective_fps,
        mocap_target_stabilization=not args.disable_mocap_target_stabilization,
        mocap_target_cutoff_hz=args.mocap_target_cutoff,
        target_conditioning=args.target_conditioning,
        pipeline_profile=args.pipeline_profile,
        foot_landmark_model=args.foot_landmark_model,
        foot_landmark_weight=args.foot_landmark_weight,
        face_position_weight=args.face_position_weight,
        head_target_mode=args.head_target_mode,
        foot_gate_reference=args.foot_gate_reference,
        limb_twist_anchor=args.limb_twist_anchor,
        two_pass_limbs=args.two_pass_limbs,
        two_pass_pass1_stride=args.two_pass_pass1_stride,
        two_pass_pass1_max_iter=args.two_pass_pass1_max_iter,
        smoothed_limbs=args.smoothed_limbs,
        limb_twist_pin=args.limb_twist_pin,
        limb_hinge=args.limb_hinge,
        twist_limit_mode=args.twist_limit_mode,
        limit_projection=args.limit_projection,
        soft_limit_joints=args.soft_limit_joints,
        ankle_twist_limit_deg=args.ankle_twist_limit_deg,
        redundant_dof_prior=args.redundant_dof_prior,
        leg_stabilization=args.leg_stabilization_override,
        leg_stabilize_cutoff_hz=args.leg_stabilize_cutoff,
        leg_max_deg_per_frame=args.leg_max_deg_per_frame,
        foot_lock=args.foot_lock_override,
        foot_lock_debug_dir=args.foot_lock_debug_dir,
        pose2d_npz=args.pose2d_npz,
        arm_orientation_rewrite=args.enable_arm_orientation_rewrite,
        require_grounded=getattr(args, "require_grounded", False),
        profile=getattr(args, "profile", False),
    )
    _save_cli_sidecar(out_path, manifest)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    npy_src = Path(args.npy)
    fbx_src = Path(args.fbx)
    out_path = Path(args.output)
    if not npy_src.exists():
        print(f"error: animation NPY not found: {npy_src}", file=sys.stderr)
        return 2
    if not fbx_src.exists():
        print(f"error: FBX rig not found: {fbx_src}", file=sys.stderr)
        return 2

    manifest = _cli_manifest(
        {"input_animation_npy": npy_src, "target_fbx": fbx_src},
        {"fps": args.fps},
    )
    from aimocap.retarget.fbx_export import write_fbx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fbx(str(npy_src), str(fbx_src), str(out_path), fps=args.fps)
    _save_cli_sidecar(out_path, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aimocap",
        description="Markerless multi-camera motion capture (Python core).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pose", help="Run 2D pose estimation over a video.")
    pp.add_argument("video", help="Path to input video (mp4/avi/mov/gif).")
    pp.add_argument("-o", "--output", help="Output video path (default: <name>_annotated.mp4).")
    pp.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (default: whole video).")
    pp.add_argument("--threshold", type=float, default=0.3,
                    help="Min keypoint confidence to draw (default: 0.3).")
    pp.add_argument("--pick", choices=["largest", "all"], default="largest",
                    help="Person selection each frame (default: largest).")
    pp.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N frames.")
    pp.add_argument("--pose-model", help="Path to CIGPose ONNX model (optional).")
    pp.set_defaults(func=_cmd_pose)

    mp = sub.add_parser("multipose", help="Run 2D pose estimation synchronously over multiple videos.")
    mp.add_argument("videos", nargs="+", help="Paths to input videos.")
    mp.add_argument("-o", "--output", required=True, help="Output .npz file path.")
    mp.add_argument("--grid-video", help="Optional output grid video path.")
    mp.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (default: whole video).")
    mp.add_argument("--threshold", type=float, default=0.3,
                    help="Min keypoint confidence to draw (default: 0.3).")
    mp.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N frames.")
    mp.add_argument("--pose-model", help="Path to the selected pose backend's ONNX model (optional).")
    mp.add_argument("--detector-model", help="Path to the YOLOX detector ONNX model (optional).")
    mp.add_argument("--inference-engine", choices=["ort", "cv2dnn"], default="ort",
                    help="Inference engine (default: ort; cv2dnn is CPU/OpenCL opt-in).")
    mp.add_argument("--pose-backend", choices=["cigpose", "rtmpose", "rtmw"], default="rtmw",
                    help=("2D pose backend (default: rtmw = RTMW-x with calibrated scores; "
                          "legacy: --pose-backend cigpose --crop-mode detector --simcc-decode argmax)."))
    mp.add_argument("--sync-manifest", help="Audio synchronization JSON produced by `aimocap sync`.")
    mp.add_argument("--require-sync", action="store_true",
                    help="Fail unless an audio synchronization manifest is provided.")
    mp.add_argument("--allow-multiple-subjects", action="store_true",
                    help="Disable the default single-performer ambiguity gate.")
    mp.add_argument("--tracker", choices=["temporal", "strict"], default="temporal",
                    help=("Single-subject policy: 'temporal' (default) follows one "
                          "persistent identity per camera through gaps; 'strict' is the "
                          "legacy per-frame gate that raises on any ambiguous frame."))
    mp.add_argument("--crop-mode", choices=["detector", "smoothbox"], default="smoothbox",
                    help="2D crop policy (default: smoothbox; legacy: detector).")
    mp.add_argument("--simcc-decode", choices=["argmax", "subpixel"], default="subpixel",
                    help="SimCC decoder (default: subpixel; legacy: argmax).")
    mp.add_argument("--part-refine", choices=["off", "hands", "face", "hands_face"], default="hands_face",
                    help=("Offline hand/face second-pass refinement (default: hands_face; user decision 2026-10-20 after the "
                          "S3 unseen-clip test: hands -11%% median, -16%% jitter). "
                          "Use --part-refine off for the legacy RTMW-only output."))
    mp.set_defaults(func=_cmd_multipose)

    sp = sub.add_parser("sync", help="Estimate camera time offsets from audio clap transients.")
    sp.add_argument("videos", nargs="+", help="Paths to input videos with audio.")
    sp.add_argument("-o", "--output", required=True, help="Output synchronization manifest JSON path.")
    sp.add_argument("--camera-ids", nargs="*", help="Stable camera IDs (defaults to video stems).")
    sp.add_argument("--fps", type=float, default=30.0, help="Capture FPS used for frame-offset reporting.")
    sp.set_defaults(func=_cmd_sync)

    psync = sub.add_parser(
        "panoptic-sync",
        help=("Emit a dataset-verified frame-sync manifest for CMU Panoptic HD "
              "cameras (genlocked global frame index; these videos have no audio)."),
    )
    psync.add_argument("videos", nargs="+", help="Paths to Panoptic HD mp4 files (hd_NN_MM.mp4).")
    psync.add_argument("-o", "--output", required=True, help="Output sync manifest JSON path.")
    psync.add_argument("--sequence-dir", required=True,
                       help="Panoptic sequence root containing hdPose3d_stage1_coco19/.")
    psync.add_argument("--sequence", default="171204_pose1", help="Sequence name.")
    psync.add_argument("--fps", type=float, default=29.97002997002997,
                       help="Global HD capture rate (default: Panoptic ~29.97).")
    psync.add_argument("--reference", default=None,
                       help="Reference camera id (default: first video).")
    psync.add_argument("--no-verify-gt", action="store_true",
                       help="Skip the ground-truth frame-availability probe.")
    psync.add_argument("--global-start", action="append", default=[], metavar="CAM=START",
                       help=("Global Panoptic frame index at this camera's local frame 0 "
                             "(for trimmed/partial clips). Repeat per camera, e.g. "
                             "--global-start 00_30=23312."))
    psync.add_argument("--window-start-global", type=int, default=None, metavar="FRAME",
                       help=("Global Panoptic frame the output timeline should start at; "
                             "the reference camera is seeked to its corresponding local "
                             "position so frame iteration begins inside a chosen window."))
    psync.add_argument("--skip-sequence-hash", action="store_true",
                       help=("D3: skip hashing the (potentially multi-GB) sequence "
                             "directory for provenance. Default OFF: the hash is the "
                             "input-integrity record; skipping it is recorded in the "
                             "manifest (sequence_dir: null) and weakens provenance."))
    psync.set_defaults(func=_cmd_panoptic_sync)

    cp = sub.add_parser("calib", help="Camera calibration commands.")
    c_sub = cp.add_subparsers(dest="calib_cmd", required=True)
    
    ca = c_sub.add_parser("auto", help="Auto-calibrate from multi_pose2d.npz.")
    ca.add_argument("npz", help="Path to multi_pose2d.npz.")
    ca.add_argument("-o", "--output", required=True, help="Output extrinsics.json path.")
    ca.add_argument("--videos", nargs="*", help="Original videos to extract EXIF focal length.")
    ca.add_argument("--threshold", type=float, default=0.5,
                    help="Min keypoint confidence to use for calibration (default: 0.5).")
    ca.add_argument("--allow-uncalibrated", action="store_true",
                    help="Allow guessed/markerless intrinsics for explicitly exploratory calibration output.")
    ca.set_defaults(func=_cmd_calib_auto)

    cb = c_sub.add_parser("board", help="Calibrate real intrinsics/extrinsics from synchronized checkerboard images.")
    cb.add_argument("image_dirs", nargs="+", help="One image directory (or image file) per camera; matching filenames are synchronized poses.")
    cb.add_argument("-o", "--output", required=True, help="Output calibration JSON path.")
    cb.add_argument("--board-cols", type=int, default=9, help="Checkerboard inner corners across (default: 9).")
    cb.add_argument("--board-rows", type=int, default=6, help="Checkerboard inner corners down (default: 6).")
    cb.add_argument("--square-size", type=float, default=0.025, help="Checkerboard square size in meters (default: 0.025).")
    cb.add_argument("--camera-names", nargs="*", help="Stable camera names, one per image directory.")
    cb.add_argument("--min-frames", type=int, default=8, help="Minimum detected board poses per camera/pair (default: 8).")
    cb.set_defaults(func=_cmd_calib_board)

    tp = sub.add_parser("triangulate", help="Triangulate 2D keypoints into 3D using calibrated cameras.")
    tp.add_argument("npz", help="Path to multi_pose2d.npz.")
    tp.add_argument(
        "--fps", type=float, default=None, metavar="HZ",
        help="Capture frame rate in Hz. Default: resolved from the NPZ 'fps' key, then "
             "its 'timestamps', then the sync manifest referenced by its provenance; "
             "the command fails if none is available.",
    )
    tp.add_argument("calib", help="Path to extrinsics.json.")
    tp.add_argument("-o", "--output", required=True, help="Output skeleton3d.npz path.")
    tp.add_argument(
        "--triangulation-preset", choices=["production", "legacy"], default="production",
        help="Geometry policy bundle: 'production' enables soft camera trust, the "
             "robust RTS smoother and confidence3d (default); 'legacy' reproduces "
             "the P3.1 hard-subset output.",
    )
    tp.add_argument("--threshold", type=float, default=0.5,
                    help="Min keypoint confidence to triangulate (default: 0.5).")
    tp.add_argument("--reproj-threshold", type=float, default=5.0,
                    help="Soft pixel reprojection threshold for triangulation inliers (default: 5).")
    tp.add_argument("--hard-reproj-threshold", type=float, default=12.0,
                    help="Hard pixel reprojection threshold recorded in the geometry policy (default: 12).")
    tp.add_argument("--min-ray-angle", type=float, default=10.0,
                    help="Minimum surviving camera-ray angle in degrees (default: 10).")
    tp.add_argument("--min-depth", type=float, default=0.01,
                    help="Minimum camera-space depth in calibration units (default: 0.01).")
    tp.add_argument("--robust-loss", choices=["huber", "cauchy"], default="huber",
                    help="Robust reprojection loss (default: huber).")
    tp.add_argument("--f-scale", type=float, default=10.0,
                    help="Robust-loss transition scale in pixels (default: 10).")
    tp.add_argument(
        "--confidence-preset", choices=["legacy"], default=None,
        help="Score policy preset: 'legacy' gates every group at 0.5 with gamma 1 (P3.1).",
    )
    tp.add_argument(
        "--part-confidence", action="append", default=[], metavar="GROUP=MIN_CONF:GAMMA",
        help="Per-part override for body|feet|face|hands, e.g. face=0.05:0.5; repeat per group. "
             "Weighted-DLT weight is p**GAMMA.",
    )
    tp.add_argument(
        "--camera-policy", choices=["legacy", "soft"], default=None,
        help="Camera-trust policy: 'legacy' hard subset switching, 'soft' robust "
             "IRLS solve with continuous per-camera trust (default: GeometryConfig).",
    )
    tp.add_argument(
        "--soft-scale-px", type=float, default=None,
        help="Soft-policy robust kernel scale in pixels (default: GeometryConfig.soft_scale_px).",
    )
    tp.add_argument(
        "--soft-loo-ratio", type=float, default=None,
        help="Soft-policy leave-one-out down-weight ratio threshold (default: 2.0).",
    )
    tp.add_argument(
        "--soft-ema-alpha", type=float, default=None,
        help="Soft-policy per-camera trust EMA alpha in [0, 1] (default: 0.5).",
    )
    tp.add_argument(
        "--soft-iterations", type=int, default=None,
        help="Soft-policy IRLS iterations (default: 3).",
    )
    tp.add_argument(
        "--soft-support-min-trust", type=float, default=None,
        help="Soft-policy camera trust that counts as support (default 0.25).",
    )
    tp.add_argument(
        "--soft-support-gate", action=argparse.BooleanOptionalAction, default=None,
        help="F1: emit a point only with >=2 supporting and fitting cameras "
             "(default from the triangulation preset).",
    )
    tp.add_argument(
        "--soft-temporal-prior", action=argparse.BooleanOptionalAction, default=None,
        help="F2: initialise the soft IRLS from the camera set that best matches "
             "the previous-frame 3D point (default from the preset).",
    )
    tp.add_argument(
        "--soft-prior-w-temporal", type=float, default=None,
        help="F2 prior weight per millimetre of distance to the previous point.",
    )
    tp.add_argument(
        "--soft-prior-w-anat", type=float, default=None,
        help="F2 anatomical tie-break weight per unit relative bone deviation.",
    )
    tp.add_argument(
        "--soft-prior-velocity", action=argparse.BooleanOptionalAction, default=None,
        help="F2: use a constant-velocity prediction from the last two frames.",
    )
    tp.add_argument(
        "--soft-prior-downweight", type=float, default=None,
        help="F2 initial weight multiplier for cameras outside the chosen set.",
    )
    tp.add_argument(
        "--soft-prior-lockin-scale", type=float, default=None,
        help="F2 lock-in escape residual threshold in units of soft_scale_px.",
    )
    tp.add_argument(
        "--soft-prior-lockin-frames", type=int, default=None,
        help="F2 consecutive frames above the lock-in threshold before escape.",
    )
    tp.add_argument(
        "--temporal-spike-gate", action=argparse.BooleanOptionalAction, default=None,
        help="F3: MAD-based RTS innovation gate plus capped maneuver boost "
             "(default from the triangulation preset).",
    )
    tp.add_argument(
        "--temporal-mad-window", type=int, default=None,
        help="F3 causal innovation window for the robust MAD.",
    )
    tp.add_argument(
        "--temporal-mad-scale", type=float, default=None,
        help="F3 innovation threshold in robust MAD units.",
    )
    tp.add_argument(
        "--temporal-mad-min-samples", type=int, default=None,
        help="F3 minimum innovation samples before the MAD gate applies.",
    )
    tp.add_argument(
        "--temporal-maneuver-cap", type=float, default=None,
        help="F3 cap on the maneuver-boost excess fraction (1.0 = P3.5).",
    )
    tp.add_argument(
        "--lr-swap-fix", action="store_true",
        help="Repair per-camera left/right label swaps on the arm, leg and hand "
             "groups before triangulation (default: off).",
    )
    tp.add_argument(
        "--lr-swap-ratio", type=float, default=None,
        help="Required robust-cost reduction factor for accepting a swap "
             "(default: GeometryConfig.lr_swap_ratio).",
    )
    tp.add_argument(
        "--temporal-filter", choices=["none", "robust_rts"], default=None,
        help="Offline 3D smoothing after triangulation: 'none' (old output) or "
             "'robust_rts' (zero-phase robust Kalman/RTS); default from the "
             "triangulation preset.",
    )
    tp.add_argument(
        "--temporal-filter-sigma", type=float, default=None,
        help="Base measurement sigma in metres for --temporal-filter "
             "(default: GeometryConfig.temporal_filter_sigma_m).",
    )
    tp.add_argument(
        "--temporal-filter-q", type=float, default=None,
        help="Base white-jerk process noise (m^2/s^5) for --temporal-filter "
             "(default: GeometryConfig.temporal_filter_q).",
    )
    tp.add_argument(
        "--temporal-filter-maneuver-boost", type=float, default=None,
        help="Maximum innovation-driven process-noise boost for --temporal-filter.",
    )
    tp.add_argument(
        "--temporal-filter-fast-boost", type=float, default=None,
        help="Speed-adaptive process-noise boost for fast joints (>= 1).",
    )
    tp.add_argument(
        "--temporal-filter-max-gap-s", type=float, default=None,
        help="Longest interior gap (seconds) bridged by --temporal-filter.",
    )
    tp.add_argument(
        "--confidence3d", action=argparse.BooleanOptionalAction, default=None,
        help="Emit per-point 3D confidence (npz key 'confidence3d') from "
             "runtime observability signals. Requires camera_policy soft; "
             "default from the triangulation preset (on for production).",
    )
    tp.add_argument("--allow-unverified-sync", action="store_true",
                    help="Allow pose NPZ data without verified audio synchronization.")
    tp.add_argument("--require-calibration-metadata", action="store_true",
                    help="Require explicit projection, coordinate-system, and units metadata.")
    tp.add_argument("--allow-exploratory-calibration", action="store_true",
                    help="Allow calibration JSON explicitly marked exploratory.")
    tp.add_argument("--raw", action="store_true",
                    help="Write geometry-only 3D output; skip scale policy and IK preparation.")
    tp.add_argument("--legacy-kinematics", action="store_true",
                    help="Also compute legacy 17-joint kinematic stick model and save it under legacy_kinematic_skeleton3d.")
    tp.add_argument("--allow-unsafe-geometry", action="store_true",
                    help="Allow skeleton fitting after a failed raw-geometry gate; use only for diagnostics.")
    tp.add_argument("--filter-2d", action="store_true",
                    help="Apply Savitzky-Golay temporal filter to 2D keypoints prior to 3D lifting.")
    tp.add_argument(
        "--camera-map", action="append", default=[], metavar="INPUT=CALIBRATION",
        help="Explicit one-to-one camera mapping; repeat once per camera.",
    )
    tp.add_argument(
        "--floor-transform", default=None,
        help="Path to floor transform JSON mapping camera-world to floor-world.",
    )
    tp.add_argument(
        "--save-floor-skeleton", action="store_true",
        help="Explicitly derive and save skeleton3d_floor in floor-world frame.",
    )
    tp.add_argument(
        "--filter-f4", action="store_true",
        help="Apply authoritative camera-world F4 temporal filter and save refined camera and floor artifacts.",
    )
    tp.set_defaults(func=_cmd_triangulate)

    vp = sub.add_parser("viz", help="Visualize triangulated 3D skeleton.")
    vp.add_argument("npz", help="Path to skeleton3d.npz.")
    vp.add_argument("--cameras", required=True, help="Path to extrinsics.json.")
    vp.add_argument("-o", "--output", required=True, help="Output gif/mp4 path.")
    vp.add_argument("--skip-frames", type=int, default=0, help="Skip first N frames.")
    vp.set_defaults(func=_cmd_viz)

    rp = sub.add_parser("retarget", help="Retarget triangulated points to an FBX skeleton.")
    rp.add_argument("npz", help="Path to skeleton3d.npz.")
    rp.add_argument("fbx", help="Path to target FBX rig (e.g. Manny.FBX).")
    rp.add_argument("-o", "--output", required=True, help="Output .bvh path.")
    rp.add_argument(
        "--fps", type=float, default=None,
        help="Output animation FPS; v7 defaults to 30, while v8 reads NPZ metadata when omitted.",
    )
    rp.add_argument(
        "--require-grounded", action="store_true",
        help="Require verified physical floor registration and use derived floor-world skeleton.",
    )
    rp.add_argument(
        "--ik-max-nfev",
        type=int,
        default=None,
        help=(
            "Optional maximum nonlinear IK evaluations per frame. "
            "Use a positive value to bound runtime; the default keeps the backend default."
        ),
    )
    rp.add_argument(
        "--target-conditioning",
        choices=["robust", "legacy", "p1", "p2"],
        default=None,
        help="Solver target conditioning mode; pipeline profile v7 defaults to legacy, v8 to p1.",
    )
    rp.add_argument(
        "--pipeline-profile",
        choices=["v7", "v8"],
        default="v7",
        help="Opt-in production preset: v7 preserves legacy defaults; v8 selects P1 and disables stages 3/4.",
    )
    rp.add_argument(
        "--foot-landmark-model", choices=["proxy", "fk_offsets"], default="proxy",
        help="Foot landmark residual model (default: proxy).",
    )
    rp.add_argument(
        "--foot-landmark-weight", type=float, default=1.0,
        help="Scale only big-toe/small-toe/heel residual weights (default: 1.0).",
    )
    rp.add_argument(
        "--head-target-mode", choices=["legacy", "fixed", "calibrated"], default="legacy",
        help="Head face-frame rest convention (default: legacy).",
    )
    rp.add_argument(
        "--face-position-weight", type=float, default=1.0,
        help="Scale the nose-to-head face position residual (default: 1.0).",
    )
    rp.add_argument(
        "--foot-gate-reference", choices=["solved", "observed"], default="solved",
        help="Foot gate reference mode: 'solved' uses previous solved state (default), 'observed' uses only observations.",
    )
    rp.add_argument(
        "--limb-twist-anchor",
        choices=["init", "continuous", "continuous_resid", "smoothed", "two_pass"],
        default=None,
        help="Limb twist pin anchor; omitted resolves to the historical init anchor.",
    )
    rp.add_argument(
        "--two-pass-limbs",
        choices=["all", "arms"],
        default="all",
        help="For limb_twist_anchor=two_pass, filter all limb pins or arms only (default: all).",
    )
    rp.add_argument(
        "--two-pass-pass1-stride", type=int, default=1,
        help="Solve every k-th frame in two-pass pass 1; default 1 preserves the historical path.",
    )
    rp.add_argument(
        "--two-pass-pass1-max-iter", type=int, default=None,
        help="Optional pass-1 IK iteration cap for two-pass mode; default keeps the backend default.",
    )
    rp.add_argument(
        "--smoothed-limbs",
        choices=["all", "arms"],
        default="all",
        help="For limb_twist_anchor=smoothed, filter all limb pins or arms only (default: all).",
    )
    rp.add_argument(
        "--limb-twist-pin",
        choices=["always", "fade", "off"],
        default="always",
        help="Limb twist pin weighting: always (default), fade near straight limbs, or off.",
    )
    rp.add_argument(
        "--limb-hinge",
        choices=["off", "elbow_knee"],
        default="off",
        help="Limb hinge constraints: off (default) or elbow_knee for hinge residuals.",
    )
    rp.add_argument(
        "--twist-limit-mode",
        choices=["clamp", "soft", "hybrid", "off"],
        default="clamp",
        help="Twist-limit enforcement: clamp (historical hard clamp), soft (soft in-cost barrier), hybrid, or off.",
    )
    rp.add_argument(
        "--limit-projection",
        choices=["post", "projected"],
        default="post",
        help=(
            "Twist-limit projection timing: post accepts an unconstrained LM "
            "step then clamps (historical default); projected clamps trials "
            "before acceptance and evaluates cost on the feasible state."
        ),
    )
    rp.add_argument(
        "--soft-limit-joints",
        default=None,
        help="Comma-separated joints using the hybrid in-cost soft barrier (default: clavicle_l,clavicle_r,ankle_l,ankle_r).",
    )
    rp.add_argument(
        "--ankle-twist-limit-deg", type=float, default=None,
        help="Override only ankle_l/ankle_r hard twist clamp in degrees; omitted preserves the 10-degree default.",
    )
    rp.add_argument(
        "--redundant-dof-prior",
        choices=["none", "minimal"],
        default="none",
        help=(
            "Opt-in null-space prior for redundant clavicle rotations: "
            "none preserves legacy behavior; minimal adds temporal and weak "
            "analytic-init regularization."
        ),
    )
    rp.add_argument(
        "--disable-mocap-target-stabilization",
        action="store_true",
        help="Disable lower-body target smoothing before proxy IK.",
    )
    rp.add_argument(
        "--mocap-target-cutoff",
        type=float,
        default=1.5,
        help="Lower-body proxy target smoothing cutoff in Hz (default: 1.5).",
    )
    leg_group = rp.add_mutually_exclusive_group()
    leg_group.add_argument(
        "--disable-leg-stabilization",
        dest="leg_stabilization_override",
        action="store_false",
        help="Explicitly disable target-space lower-body rotation stabilization.",
    )
    leg_group.add_argument(
        "--enable-leg-stabilization",
        dest="leg_stabilization_override",
        action="store_true",
        help="Explicitly enable target-space lower-body rotation stabilization (including in v8).",
    )
    rp.set_defaults(leg_stabilization_override=None)
    rp.add_argument(
        "--leg-stabilize-cutoff",
        type=float,
        default=1.5,
        help="Lower-body zero-phase rotation smoothing cutoff in Hz (default: 1.5).",
    )
    rp.add_argument(
        "--leg-max-deg-per-frame",
        type=float,
        default=30.0,
        help="Clamp lower-body one-frame rotation jumps after smoothing (default: 30).",
    )
    foot_group = rp.add_mutually_exclusive_group()
    foot_group.add_argument(
        "--disable-foot-lock",
        dest="foot_lock_override",
        action="store_false",
        help="Explicitly disable target-space IK foot locking.",
    )
    foot_group.add_argument(
        "--enable-foot-lock",
        dest="foot_lock_override",
        action="store_true",
        help="Explicitly enable target-space IK foot locking (including in v8).",
    )
    rp.set_defaults(foot_lock_override=None)
    rp.add_argument(
        "--foot-lock-debug-dir",
        default=None,
        help="Optional directory for foot-lock contact and metric diagnostics.",
    )
    rp.add_argument(
        "--pose2d-npz",
        default=None,
        help="Optional multi-camera 2D pose NPZ for image-space foot contact detection.",
    )
    rp.add_argument(
        "--profile",
        action="store_true",
        help=(
            "WP-A instrumentation: time each retarget stage and write "
            "<output>.profile.json. Default off (zero overhead)."
        ),
    )
    rp.add_argument(
        "--enable-arm-orientation-rewrite",
        action="store_true",
        help=(
            "Enable the experimental post-IK arm rewrite. Disabled by default "
            "because the production solver must remain the single arm authority."
        ),
    )
    rp.set_defaults(func=_cmd_retarget)

    ep = sub.add_parser("export", help="Bake a solved animation NPY onto the FBX rig via Blender.")
    ep.add_argument("npy", help="Solved animation NPY path.")
    ep.add_argument("fbx", help="Source FBX rig path.")
    ep.add_argument("-o", "--output", required=True, help="Output animated FBX path.")
    ep.add_argument("--fps", type=float, default=30.0, help="Animation FPS (default: 30).")
    ep.set_defaults(func=_cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
