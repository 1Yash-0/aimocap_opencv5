"""Aggregate phase-3 baseline ``eval.json`` files into ``metrics.json``.

Reads ``<out-root>/run_status.json`` and each run's ``eval.json`` produced by
``scripts/phase3_run_baseline.py``, adds the triangulation gate outcome, and
prints a compact table. ``metrics.json`` keeps the full per-group blocks so the
report and later phases can reuse them without re-running the harness.

CLI:
    python -m evaluation.studio.metrics [--out-root DIR]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "phase3" / "p30_baseline_20261021"
GROUPS = ("body", "feet", "face", "left_hand", "right_hand", "fingertips")


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _gate_info(run: dict[str, Any]) -> dict[str, Any]:
    stages = {stage["name"]: stage for stage in run.get("stages", [])}
    tri = stages.get("triangulate", {})
    retry = stages.get("triangulate_unsafe")
    message = tri.get("gate_stderr_tail") or tri.get("stderr_tail") or ""
    message_lines = [line.strip() for line in message.splitlines() if line.strip()][:4]
    return {
        "raw_gate_passed": tri.get("status") in {"ok", "skipped_existing"},
        "raw_gate_returncode": tri.get("returncode"),
        "raw_gate_message": message_lines,
        "unsafe_override": bool(tri.get("gate_override")) or retry is not None,
        "unsafe_retry_status": retry.get("status") if retry else None,
    }


def _compact(block: dict[str, Any]) -> dict[str, Any]:
    stats = block.get("mpjpe_mm") or {}
    jitter = block.get("jitter_mm_per_frame2") or {}
    fast = block.get("fast_frame_jitter_mm_per_frame2") or {}
    bones = block.get("bone_length_std_mm") or {}
    return {
        "gt_available": block.get("gt_available"),
        "gt_valid_points": block.get("gt_valid_points"),
        "covered_points": block.get("covered_points"),
        "coverage": block.get("coverage"),
        "mpjpe_mm": stats,
        "pelvis_aligned_median_mm": block.get("pelvis_aligned_median_mm"),
        "jitter_mm_per_frame2": jitter,
        "fast_frame_jitter_mm_per_frame2": fast,
        "fast_frame_speed_ratio_median": block.get("fast_frame_speed_ratio_median"),
        "bone_length_std_median_mm": bones.get("median_std_mm"),
        "bone_length_std_bones": bones.get("bones"),
    }


def build_metrics(out_root: Path) -> dict[str, Any]:
    status_path = out_root / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    runs: dict[str, Any] = {}
    for run in status.get("runs", []):
        key = f"{run['clip']}:{run['variant']}"
        run_dir = out_root / run["clip"] / run["variant"]
        eval_path = run_dir / "eval.json"
        entry: dict[str, Any] = {
            "clip": run["clip"],
            "variant": run["variant"],
            "status": run.get("status"),
            "run_dir": _rel(run_dir),
            "gate": _gate_info(run),
            "stages": [
                {
                    "name": stage["name"],
                    "status": stage.get("status"),
                    "returncode": stage.get("returncode"),
                    "seconds": stage.get("seconds"),
                }
                for stage in run.get("stages", [])
            ],
        }
        if eval_path.exists():
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            entry.update(
                {
                    "eval_path": _rel(eval_path),
                    "skel": evaluation.get("skel"),
                    "frames": evaluation.get("frames"),
                    "global_frames": evaluation.get("global_frames"),
                    "units": evaluation.get("units"),
                    "scale_factor": evaluation.get("scale_factor"),
                    "skeleton3d_frame": evaluation.get("skeleton3d_frame"),
                    "floor_transform_identity": evaluation.get("floor_transform_identity"),
                    "alignment": evaluation.get("alignment"),
                    "groups": {
                        name: _compact(evaluation["groups"][name])
                        for name in GROUPS
                        if name in evaluation.get("groups", {})
                    },
                    "pooled": _compact(evaluation["pooled"]),
                    "switch_analysis": evaluation.get("switch_analysis"),
                    "per_camera_exclusion": evaluation.get("per_camera_exclusion"),
                    "worst_joints": evaluation.get("worst_joints"),
                    "warnings": evaluation.get("warnings"),
                }
            )
        runs[key] = entry

    comparisons: dict[str, Any] = {}
    for clip in ("clip1", "clip2"):
        new = runs.get(f"{clip}:NEW", {})
        old = runs.get(f"{clip}:OLD", {})
        if "groups" not in new or "groups" not in old:
            continue
        block: dict[str, Any] = {}
        for name in GROUPS + ("pooled",):
            a = new.get("pooled") if name == "pooled" else new["groups"].get(name)
            b = old.get("pooled") if name == "pooled" else old["groups"].get(name)
            if not a or not b:
                continue
            row: dict[str, Any] = {}
            for field, key in (
                ("mpjpe_median_mm", "mpjpe_mm"),
                ("jitter_median_mm_per_frame2", "jitter_mm_per_frame2"),
            ):
                new_value = (a.get(key) or {}).get("median")
                old_value = (b.get(key) or {}).get("median")
                row[field] = {
                    "NEW": new_value,
                    "OLD": old_value,
                    "delta_old_minus_new": (
                        float(old_value) - float(new_value)
                        if new_value is not None and old_value is not None
                        else None
                    ),
                }
            for field in ("coverage", "pelvis_aligned_median_mm", "fast_frame_speed_ratio_median"):
                new_value = a.get(field)
                old_value = b.get(field)
                row[field] = {
                    "NEW": new_value,
                    "OLD": old_value,
                    "delta_old_minus_new": (
                        float(old_value) - float(new_value)
                        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float))
                        else None
                    ),
                }
            block[name] = row
        comparisons[clip] = block

    return {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "out_root": _rel(out_root),
        "baseline_complete": bool(status.get("complete")),
        "baseline_status": status.get("status"),
        "evaluation_frame": (
            "skeleton3d from default triangulate (camera_world, identity floor, "
            "scale_factor 1.0, metres); GT converted cm -> m with (x, -y, -z); "
            "no per-frame alignment"
        ),
        "runs": runs,
        "comparisons_old_minus_new": comparisons,
        "notes": [
            "Panoptic has no foot GT; feet gt_available=false.",
            "AIST++ GT is body-only (COCO17); feet/face/hands gt_available=false.",
            "AIST++ world is Y-up while aimocap's fixed conversion assumes an "
            "OpenCV Y-down world; prediction and GT share the same flipped frame, "
            "so MPJPE is unaffected, but the exported frame label is nominal.",
            "Gate failure is recorded per run; an unsafe override means the raw "
            "geometry gate did not pass for that clip.",
        ],
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def summary_lines(metrics: dict[str, Any]) -> list[str]:
    header = (
        "run           body_med body_cov face_med lh_med rh_med tip_med "
        "pooled_med switch  gate"
    )
    lines = [header]
    for key, run in metrics["runs"].items():
        groups = run.get("groups", {})
        pooled = run.get("pooled", {})
        switch = (run.get("switch_analysis") or {}).get("switch_rate")
        gate = "unsafe" if run.get("gate", {}).get("unsafe_override") else "ok"
        lines.append(
            "{:<13} {:>8} {:>8} {:>8} {:>7} {:>6} {:>7} {:>10} {:>6}  {}".format(
                key,
                _fmt((groups.get("body", {}).get("mpjpe_mm") or {}).get("median")),
                _fmt(groups.get("body", {}).get("coverage"), 3),
                _fmt((groups.get("face", {}).get("mpjpe_mm") or {}).get("median")),
                _fmt((groups.get("left_hand", {}).get("mpjpe_mm") or {}).get("median")),
                _fmt((groups.get("right_hand", {}).get("mpjpe_mm") or {}).get("median")),
                _fmt((groups.get("fingertips", {}).get("mpjpe_mm") or {}).get("median")),
                _fmt((pooled.get("mpjpe_mm") or {}).get("median")),
                _fmt(switch, 4),
                gate,
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out_root = Path(args.out_root)
    metrics = build_metrics(out_root)
    out_path = out_root / "metrics.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="ascii")
    os.replace(tmp, out_path)
    for line in summary_lines(metrics):
        print(line)
    print(f"wrote {_rel(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
