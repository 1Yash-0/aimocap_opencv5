"""Overview metrics aggregation, including trial-specific sync references."""

from __future__ import annotations

from service.metrics_view import overview


def _trial(name: str | None, frames: int) -> dict:
    metrics = {"sync_vs_truth": {"max_abs_error_frames": frames, "unresolved_cameras": []}}
    if name is not None:
        metrics["trial"] = name
    return metrics


def test_overview_excludes_walking_sync_references_but_keeps_other_trial_stats():
    stats = overview([_trial("squats2", 2), _trial("STS1", 1), _trial("walking1", 86)])

    assert stats["trials"] == 3
    assert stats["sync_frames"] == "2"
    assert stats["sync_ms"] == "33"
    assert stats["walking_trials"] == 1


def test_walking_only_sync_is_not_reported_and_unknown_trial_names_remain_included():
    walking = overview([_trial("walking1", 86)])
    assert walking["sync_frames"] is None
    assert walking["sync_ms"] is None
    assert walking["walking_trials"] == 1

    unknown = overview([_trial(None, 8)])
    assert unknown["sync_frames"] == "8"
    assert unknown["walking_trials"] == 0
