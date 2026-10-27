"""Tests for the web service (auth, pages, catalogue, live runs, file serving)."""

from __future__ import annotations

import base64
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import service.app as service_app  # noqa: E402
from service.app import create_app  # noqa: E402
from service.catalog import Catalog  # noqa: E402
from service.config import Settings  # noqa: E402
from service.runs import RunManager  # noqa: E402

PASSWORD = "test-password"
REPO = Path(__file__).resolve().parent.parent
REAL_DATA = REPO / "data" / "opencap_trials"
REAL_SHOWCASE = REPO / "showcase" / "subject4_squats1"
CAMS = ["Cam0", "Cam1", "Cam2"]
PAGES = ("/", "/results", "/results/subject4_squats1", "/live", "/about", "/runs")

METRICS = {
    "config": {"cams": ["Cam0", "Cam1", "Cam2"]},
    "calibration": {"cameras": {"Cam0": {"dR_deg": 0.05, "centre_diff_mm": 3.0, "reproj_rms_px": 0.1}},
                    "max_dR_deg": 0.05, "max_centre_diff_mm": 3.0},
    "sync_vs_truth": {"per_camera": {"Cam1": {"estimated": 4, "truth": 3, "abs_error_frames": 1}},
                      "unresolved_cameras": [], "max_abs_error_frames": 1},
    "scores": {
        "ours": {"lag0_procrustes": {"median_mm": 41.4, "p95_mm": 54.5},
                 "smoothness": {"acceleration_error_median_mm_per_frame2": 0.98,
                                "jitter_slowest20_median_mm_per_frame2": 0.44, "peak_speed_ratio_p95": 1.04}},
        "opencap_hrnet_5cam": {"lag0_procrustes": {"median_mm": 40.4, "p95_mm": 42.7}},
    },
    "window": {"frames": 120},
    "stage_seconds": {"calibrate": 0.7, "pose2d": 150.0},
}

FAKE_RUNNER = r"""
import json, sys, time, pathlib
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
with (out / "events.jsonl").open("w") as f:
    for stage in ("calibrate", "pose2d"):
        f.write(json.dumps({"ts": "", "stage": stage, "status": "started", "seconds": None, "message": ""}) + "\n")
        f.flush(); time.sleep(0.2)
        f.write(json.dumps({"ts": "", "stage": stage, "status": "completed", "seconds": 0.2, "message": ""}) + "\n")
(out / "metrics.json").write_text(sys.argv[2])
(out / "motion.bvh").write_text("HIERARCHY\n")
"""


def _auth(password: str = PASSWORD) -> dict[str, str]:
    token = base64.b64encode(f"judge:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fake_command(python, trial, preset, out_dir):
    return [python, "-c", FAKE_RUNNER, str(out_dir), json.dumps(METRICS)]


def _build(tmp_path: Path, data: Path, showcase: Path, hourly: float = 0.1):
    settings = Settings(
        data_dir=data, runs_dir=tmp_path / "runs", showcase_dir=showcase, log_dir=tmp_path / "logs",
        repo_root=tmp_path, password=PASSWORD, dev_mode=False, s3_bucket=None, cloudwatch=False,
        hourly_usd=hourly, run_timeout_s=30, python=sys.executable,
    )
    manager = RunManager(settings.runs_dir, tmp_path, sys.executable, 30, command_builder=fake_command)
    return TestClient(create_app(settings, manager)), manager


@pytest.fixture()
def env(tmp_path: Path):
    data = tmp_path / "data"
    showcase = tmp_path / "showcase"
    trial_json = {"subject": "subject4", "trial": "squats1", "cams": CAMS}
    for root in (data, showcase):
        (root / "subject4_squats1").mkdir(parents=True)
        (root / "subject4_squats1" / "trial.json").write_text(json.dumps(trial_json))
    (showcase / "subject4_squats1" / "metrics.json").write_text(json.dumps(METRICS))
    (showcase / "subject4_squats1" / "sync.json").write_text("{}")
    (tmp_path / "secret.txt").write_text("secret")
    client, manager = _build(tmp_path, data, showcase)
    return client, manager, tmp_path


def test_requires_auth(env):
    client, _, _ = env
    assert client.get("/").status_code == 401
    assert client.get("/", headers=_auth("wrong")).status_code == 401
    assert "WWW-Authenticate" in client.get("/").headers


def test_healthz_is_public(env):
    client, _, _ = env
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_security_headers(env):
    client, _, _ = env
    response = client.get("/", headers=_auth())
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_overview_has_problem_and_comparison(env):
    client, _, _ = env
    page = client.get("/", headers=_auth())
    assert page.status_code == 200
    assert 'href="/" aria-current="page">Overview</a>' in page.text
    assert "Motion capture from the phones you already own" in page.text
    assert 'href="/results">See the results</a>' in page.text
    assert 'href="/live">Run it live</a>' in page.text
    assert "The problem" in page.text
    assert "Compared with the alternatives" in page.text
    assert "<table" in page.text and "Single-camera AI apps" in page.text
    assert "41 mm median" in page.text  # the median of the one showcase trial
    assert 'class="strip"' in page.text and "Largest calibration error" in page.text
    assert "Uhlrich et al., PLOS Computational Biology, 2023" in page.text
    assert 'href="/results/subject4_squats1#use"' in page.text
    assert 'href="/results/subject4_squats1#movement"' in page.text
    assert 'href="/results/subject4_squats1#accuracy"' in page.text


def test_overview_keeps_measured_cpu_scope_and_architecture_image(env):
    client, _, _ = env
    page = client.get("/", headers=_auth())
    assert page.status_code == 200
    assert "Measured on a development laptop CPU" in page.text
    assert "378" in page.text and "520" in page.text
    how = page.text.index('aria-labelledby="how"')
    image = re.search(r'<img\b(?=[^>]*src="/static/architecture\.png")[^>]*>', page.text[how:])
    assert image and 'loading="lazy"' in image.group(0)
    response = client.get("/static/architecture.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_overview_sync_headline_uses_nonwalking_trials(env):
    client, _, tmp_path = env
    data_trial = tmp_path / "data" / "subject4_walking1"
    showcase_trial = tmp_path / "showcase" / "subject4_walking1"
    for folder in (data_trial, showcase_trial):
        folder.mkdir(parents=True)
    (data_trial / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": "walking1", "cams": CAMS}))
    walking_metrics = json.loads(json.dumps(METRICS))
    walking_metrics["trial"] = "walking1"
    walking_metrics["sync_vs_truth"].update(max_abs_error_frames=86, unresolved_cameras=["Cam1"])
    (showcase_trial / "metrics.json").write_text(json.dumps(walking_metrics))

    metrics_path = tmp_path / "showcase" / "subject4_squats1" / "metrics.json"
    squat_metrics = json.loads(metrics_path.read_text())
    squat_metrics["trial"] = "squats1"
    squat_metrics["sync_vs_truth"].update(max_abs_error_frames=2, unresolved_cameras=[])
    metrics_path.write_text(json.dumps(squat_metrics))

    page = client.get("/", headers=_auth())

    assert page.status_code == 200
    assert "Sync: squats and sit-to-stand" in page.text
    assert re.search(r'class="value">\s*2\s+frames\s*</dd>', page.text)
    assert "walking" in page.text.lower()
    assert "1 camera in 1 trial has an unresolved mocap-derived synchronization reference" in page.text
    assert "the pipeline still estimated camera offsets" in page.text
    assert not re.search(r'class="value">\s*86\s+frames\s*</dd>', page.text)


def test_walking_movement_uses_peak_difference_without_squat_advice(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    showcase = tmp_path / "showcase"
    for name, trial in (("subject4_walking1", "walking1"), ("subject4_squats1", "squats1")):
        data_trial, showcase_trial = data / name, showcase / name
        data_trial.mkdir(parents=True)
        showcase_trial.mkdir(parents=True)
        (data_trial / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": trial, "cams": CAMS}))
        metrics = json.loads(json.dumps(METRICS))
        metrics["trial"] = trial
        (showcase_trial / "metrics.json").write_text(json.dumps(metrics))
        (showcase_trial / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": trial, "cams": CAMS}))
        (showcase_trial / "sync.json").write_text("{}")

    monkeypatch.setattr(service_app, "knee_context", lambda *_: {
        "knee_svg": "<svg></svg>",
        "knee_rows": [{"source": "AIMoCap", "left": "90", "right": "90", "asymmetry": "90.0", "peak_difference": "0.0"}],
        "knee_error": None,
    })
    client, _ = _build(tmp_path, data, showcase)

    walking = client.get("/results/subject4_walking1", headers=_auth())
    assert walking.status_code == 200
    assert "Peak left-right difference" in walking.text
    assert "0.0" in walking.text
    assert "Mean left-right difference" not in walking.text
    assert "favouring one leg" not in walking.text

    squats = client.get("/results/subject4_squats1", headers=_auth())
    assert squats.status_code == 200
    assert "Mean left-right difference" in squats.text
    assert "favouring one leg" in squats.text


def test_results_lists_the_trial_and_hides_empty_groups(env):
    client, _, _ = env
    page = client.get("/results", headers=_auth())
    assert page.status_code == 200
    assert "Subject 4, squats" in page.text
    assert 'href="/results/subject4_squats1"' in page.text
    assert "Walking" not in page.text and "Drop jump" not in page.text
    assert 'href="/results" aria-current="page">Results</a>' in page.text


def test_trial_page_redirect_and_unknown_names(env):
    client, _, _ = env
    page = client.get("/results/subject4_squats1", headers=_auth())
    assert page.status_code == 200
    assert "41.4" in page.text and "OpenCap" in page.text
    assert 'id="use"' in page.text and 'id="downloads"' in page.text
    assert "Movement analysis" not in page.text  # this fixture has no skeleton file
    redirect = client.get("/showcase/subject4_squats1", headers=_auth(), follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "/results/subject4_squats1"
    assert client.get("/results/unknown", headers=_auth()).status_code == 404
    assert client.get("/showcase/unknown", headers=_auth(), follow_redirects=False).status_code == 404


@pytest.mark.parametrize("comparison_videos", [False, True], ids=["character-only", "with-comparison-videos"])
def test_character_preview_appears_only_when_the_video_exists(env, tmp_path: Path, comparison_videos: bool):
    client, _, _ = env
    showcase = tmp_path / "showcase" / "subject4_squats1"
    if comparison_videos:
        for name in ("overlay_2d.mp4", "skeleton_vs_mocap.mp4"):
            (showcase / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")

    page = client.get("/results/subject4_squats1", headers=_auth())
    assert page.status_code == 200
    assert "Character preview" not in page.text
    assert ('href="#videos"' in page.text) is comparison_videos
    assert ('id="videos"' in page.text) is comparison_videos
    if comparison_videos:
        initial_sources = re.findall(r'<video\b[^>]*\bsrc="([^"]+)"', page.text)
        assert initial_sources == [
            "/files/showcase/subject4_squats1/overlay_2d.mp4",
            "/files/showcase/subject4_squats1/skeleton_vs_mocap.mp4",
        ]

    (showcase / "character.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    page = client.get("/results/subject4_squats1", headers=_auth())
    assert page.status_code == 200
    assert "Character preview" in page.text
    overlay_url = "/files/showcase/subject4_squats1/overlay_2d.mp4"
    skeleton_url = "/files/showcase/subject4_squats1/skeleton_vs_mocap.mp4"
    character_url = "/files/showcase/subject4_squats1/character.mp4"
    character_src = f'src="{character_url}"'
    assert page.text.count(character_src) == 1
    video_sources = re.findall(r'<video\b[^>]*\bsrc="([^"]+)"', page.text)
    if comparison_videos:
        assert video_sources == [overlay_url, skeleton_url, character_url]
        assert page.text.index('class="video-toolbar"') < page.text.index('class="videos has-comparison"')
        video_results = re.search(r'<div class="video-results"[^>]*>(.*?)</div>', page.text, flags=re.DOTALL)
        assert video_results
        assert video_results.group(1).index(skeleton_url) < video_results.group(1).index(character_url)
        assert overlay_url not in video_results.group(1)
        assert page.text.index(overlay_url) < page.text.index('class="video-results"')
        assert page.text.count("/static/videos.js") == 1
        toolbar = re.search(r'<div class="video-toolbar"[^>]*data-video-controls[^>]*>(.*?)</div>', page.text, flags=re.DOTALL)
        assert toolbar
        play_button = re.search(r'<button\b[^>]*data-play-all[^>]*>', toolbar.group(1))
        assert play_button and re.search(r"\shidden(?:\s|>|=)", play_button.group(0))
        assert 'role="status"' in toolbar.group(1)
        assert 'aria-live="polite"' in toolbar.group(1)
    else:
        assert video_sources == [character_url]
    videos_section = re.search(r'<section id="videos".*?</section>', page.text, flags=re.DOTALL)
    assert videos_section and character_src in videos_section.group(0)
    assert page.text.index(character_src) < page.text.index('<section id="use"')
    assert 'href="#videos"' in page.text
    assert "The captured motion applied to Manny, rendered in Blender." in page.text

    video = client.get("/files/showcase/subject4_squats1/character.mp4", headers=_auth())
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"


def test_video_playback_controller_is_served(env):
    client, _, _ = env
    response = client.get("/static/videos.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_live_page_lists_runnable_trials_and_cost(env):
    client, _, _ = env
    page = client.get("/live", headers=_auth())
    assert page.status_code == 200
    assert "Try it live" in page.text
    assert "Run live" in page.text and "Subject 4, squats" in page.text
    assert "estimate based on the on-demand EC2 list price" in page.text
    assert 'href="/runs"' in page.text
    assert "6 to 7 minutes" in page.text
    assert "body keypoints" in page.text
    assert "hands and face" in page.text


def test_cost_estimate_is_hidden_without_an_hourly_rate(tmp_path: Path):
    data = tmp_path / "data"
    showcase = tmp_path / "showcase"
    (data / "subject4_squats1").mkdir(parents=True)
    (data / "subject4_squats1" / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": "squats1"}))
    showcase.mkdir()
    client, _ = _build(tmp_path, data, showcase, hourly=0.0)
    page = client.get("/live", headers=_auth())
    assert page.status_code == 200
    assert "estimate based on the on-demand EC2 list price" not in page.text


def test_runs_keeps_failed_history_without_empty_cost_column(tmp_path: Path):
    data = tmp_path / "data"
    showcase = tmp_path / "showcase"
    data.mkdir()
    showcase.mkdir()
    client, manager = _build(tmp_path, data, showcase, hourly=0.0)
    run_dir = manager.runs_dir / "20260101-000000-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "id": run_dir.name, "trial": "subject4_walking1", "preset": "quick", "status": "failed"
    }))

    page = client.get("/runs", headers=_auth())
    assert page.status_code == 200
    assert "subject4_walking1" in page.text and "failed" in page.text
    assert "Estimated cost" not in page.text


def test_pages_have_no_github_links_and_no_inline_script_or_style(env):
    client, _, _ = env
    for url in PAGES:
        page = client.get(url, headers=_auth())
        assert page.status_code == 200, url
        assert "github" not in page.text.lower(), url
        assert re.search(r"<script(?![^>]*\bsrc=)", page.text) is None, url
        assert ' style="' not in page.text, url


def test_live_run_completes(env):
    client, manager, _ = env
    response = client.post("/runs", data={"trial": "subject4_squats1", "preset": "quick"},
                           headers=_auth(), follow_redirects=False)
    assert response.status_code == 303
    run_id = response.headers["location"].rsplit("/", 1)[-1]

    second = client.post("/runs", data={"trial": "subject4_squats1", "preset": "quick"}, headers=_auth())
    assert second.status_code == 409

    deadline = time.time() + 20
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run_id}", headers=_auth()).json()
        if status["run"]["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert status is not None and status["run"]["status"] == "completed"
    assert [s["stage"] for s in status["stages"]] == ["calibrate", "pose2d"]

    page = client.get(f"/runs/{run_id}", headers=_auth())
    assert "motion.bvh" in page.text
    assert "pose2d" in page.text  # stages are rendered server-side once the run has finished
    assert 'id="downloads"' in page.text  # the result sections follow a completed run
    assert "github" not in page.text.lower()
    assert client.get(f"/files/runs/{run_id}/motion.bvh", headers=_auth()).status_code == 200
    assert run_id in client.get("/runs", headers=_auth()).text


def test_rejects_unknown_trial_and_preset(env):
    client, _, _ = env
    assert client.post("/runs", data={"trial": "../etc", "preset": "quick"}, headers=_auth()).status_code == 404
    assert client.post("/runs", data={"trial": "subject4_squats1", "preset": "full"}, headers=_auth()).status_code == 400


def test_file_serving_is_confined(env):
    client, _, _ = env
    for url in (
        "/files/showcase/subject4_squats1/..%2f..%2fsecret.txt",
        "/files/showcase/subject4_squats1/secret.txt",
        "/files/showcase/..%2fsubject4_squats1/metrics.json",
        "/files/other/subject4_squats1/metrics.json",
    ):
        assert client.get(url, headers=_auth()).status_code == 404, url
    assert client.get("/files/showcase/subject4_squats1/metrics.json", headers=_auth()).status_code == 200


def test_interrupted_runs_are_marked_failed(env, tmp_path):
    _, manager, _ = env
    run_dir = manager.runs_dir / "20260101-000000-abcd"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"id": run_dir.name, "trial": "t", "preset": "quick",
                                                  "status": "running"}))
    restarted = RunManager(manager.runs_dir, tmp_path, sys.executable, 30)
    assert restarted.get(run_dir.name).status == "failed"


def test_mocap_falls_back_to_showcase_metadata(tmp_path: Path):
    trial_dir = tmp_path / "data" / "subject4_squats1"
    (trial_dir / "mocap").mkdir(parents=True)
    (trial_dir / "mocap" / "squats1.trc").write_text("trc")
    (trial_dir / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": "squats1", "cams": CAMS}))
    showcase = tmp_path / "showcase" / "subject4_squats1"
    showcase.mkdir(parents=True)
    (showcase / "metrics.json").write_text("{}")
    (showcase / "trial.json").write_text(json.dumps({"files": {"mocap": "mocap/squats1.trc"}}))
    trial = Catalog(tmp_path / "data", tmp_path / "showcase").get("subject4_squats1")
    assert trial.mocap_file() == (trial_dir / "mocap" / "squats1.trc").resolve()


def test_mocap_outside_the_trial_folder_is_refused(tmp_path: Path):
    trial_dir = tmp_path / "data" / "subject4_squats1"
    trial_dir.mkdir(parents=True)
    (tmp_path / "data" / "outside.trc").write_text("trc")
    (trial_dir / "trial.json").write_text(json.dumps({"subject": "subject4", "trial": "squats1",
                                                     "files": {"mocap": "../outside.trc"}}))
    trial = Catalog(tmp_path / "data", tmp_path / "showcase").get("subject4_squats1")
    assert trial.mocap_file() is None


REAL_READY = (
    (REAL_SHOWCASE / "metrics.json").is_file()
    and (REAL_SHOWCASE / "skeleton3d.npz").is_file()
    and (REAL_DATA / "subject4_squats1" / "mocap" / "squats1.trc").is_file()
)


@pytest.mark.skipif(not REAL_READY, reason="no local showcase result and marker data available")
def test_real_metrics_render_with_movement_chart(tmp_path: Path):
    target = tmp_path / "showcase" / "subject4_squats1"
    target.mkdir(parents=True)
    for name in ("metrics.json", "calibration.json", "calibration_check.json", "sync.json", "skeleton3d.npz"):
        if (REAL_SHOWCASE / name).is_file():
            shutil.copy(REAL_SHOWCASE / name, target / name)
    # Minimal trial metadata. Its files.mocap is resolved inside the real data folder, where the marker file lives.
    (target / "trial.json").write_text(json.dumps({
        "subject": "subject4", "trial": "squats1", "cams": ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"],
        "files": {"mocap": "mocap/squats1.trc"},
    }))
    client, _ = _build(tmp_path, REAL_DATA, tmp_path / "showcase")
    page = client.get("/results/subject4_squats1", headers=_auth())
    assert page.status_code == 200
    assert "Cam4" in page.text
    assert (target / "skeleton3d.npz").is_file()
    assert "<svg" in page.text
    assert 'class="line mocap-l"' in page.text  # marker motion capture is drawn
    assert re.search(r'class="line ours-l" d="M[-\d.]+,[-\d.]+', page.text)
    assert "to within" in page.text
    assert "github" not in page.text.lower()
