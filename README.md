<div align="center">

```text
    _    ___ __  __  ___   ____    _    ____
   / \  |_ _|  \/  |/ _ \ / ___|  / \  |  _ \
  / _ \  | || |\/| | | | | |     / _ \ | |_) |
 / ___ \ | || |  | | |_| | |___ / ___ \|  __/
/_/   \_\___|_|  |_|\___/ \____/_/   \_\_|

     MULTI-CAMERA MARKERLESS MOTION CAPTURE
```

**Turn ordinary camera videos into usable 3D motion.**

[Live demo](https://d23lyy85o8o5w8.cloudfront.net/) · [Results](https://d23lyy85o8o5w8.cloudfront.net/results) · [Technical report](docs/technical_report.md) · [Architecture](docs/architecture.md)

</div>

AIMoCap combines video from several ordinary cameras into a calibrated 3D body skeleton, then exports motion as BVH for animation workflows. Its pipeline handles camera calibration, timing alignment, pose detection, 3D reconstruction, smoothing, and export. OpenCV 5 powers the computer-vision pipeline, including neural-network inference through `cv2.dnn`.

## At a glance

- **Markerless capture:** no suit or tracking markers are needed.
- **Multiple views:** combine three or more camera recordings, including recordings that did not start together.
- **Whole-body pose:** estimate up to 133 body, hand, foot, and face keypoints.
- **Animation output:** export a motion skeleton as BVH, with optional character retargeting.
- **CPU deployment:** run the demonstrated workflow on CPU hardware; a hosted demo is available.

## Evaluated results

Evaluated on the public OpenCap LabValidation dataset using five-camera trials (720×1280 portrait video, 60 fps). Position error is measured at zero time shift using AIMoCap's calibration and synchronization. OpenCap results are included as a reference on the same trials. The interactive site includes videos, metrics, calibration details, and downloads for each trial.

| Trial | AIMoCap median / p95 position error (mm) | OpenCap median / p95 (mm) |
|---|---:|---:|
| Subject 4 · squats (held out) | **41.4 / 54.5** | 40.4 / 42.7 |
| Subject 2 · squats | 50.9 / 57.5 | 46.1 / 48.5 |
| Subject 2 · walking | 46.4 / 50.2 | 44.7 / 51.6 |
| Subject 4 · sit-to-stand | 41.6 / 47.6 | 41.3 / 43.4 |

For the held-out subject 4 squat trial, calibration differed from the solved reference cameras by at most **0.09° rotation and 7.2 mm position**. Timing agreed within **one 60-fps frame** across five cameras. On that trial, acceleration error was **0.98 mm/frame²**, still-frame jitter was **0.44 mm/frame²**, and fast-motion speed ratio was **1.04** (1.0 means matching speed).

Walking is periodic, so its motion-capture timing reference can be cycle-ambiguous. Its timing comparison should be read with that context. See the [technical report](docs/technical_report.md) for metric definitions, trial details, and interpretation notes.

## OpenCV 5 performance check

On the development laptop CPU, RTMW-x inference using OpenCV 5's DNN engine took **378 ms per person crop**, compared with **520 ms** using ONNX Runtime in the same check. Across 60 frames from two pre-cropped views, the largest difference in BODY keypoint coordinates was **0.005 pixels**. Checkerboard detection succeeded on **90 of 90 calibration clips**. These are measured checks on the stated setup, not a hardware-wide performance guarantee.

## How it works

```text
Camera videos
     ↓
Checkerboard calibration → camera positions and lens geometry
     ↓
2D whole-body pose → keypoints in each camera view
     ↓
Timing alignment → corresponding moments across cameras
     ↓
3D triangulation + smoothing → one stable motion track
     ↓
BVH export → optional character retargeting
```

| Stage | What happens |
|---|---|
| Calibration | Detect checkerboard corners and estimate each camera's position and orientation. |
| 2D pose | Detect the person and estimate whole-body keypoints in each image. |
| Synchronization | Estimate frame offsets from visual consistency across camera views. |
| 3D reconstruction | Combine camera observations to estimate joint positions in 3D. |
| Smoothing and export | Reduce measurement noise while preserving movement, then write a BVH skeleton. |

## Try it

### Hosted demo

Visit the [AIMoCap demo](https://d23lyy85o8o5w8.cloudfront.net/). The **Results** pages show completed multi-camera trials and let you compare the source videos, reconstructed skeleton, and character preview. The **Run it live** page submits a short CPU-based cloud run; processing takes several minutes.

### Local setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps rtmlib==0.0.16 tqdm==4.70.1
```

The default requirements pin OpenCV 5 and the CPU inference runtime. Model weights are downloaded separately:

```bash
python scripts/fetch_models.py
```

### Run an evaluation trial

Prepare the OpenCap trial archive first, then run:

```bash
python -m evaluation.opencap.dataset --zip LabValidation_withVideos.zip --subject subject4 --trial squats1 --out data/opencap_trials/subject4_squats1
python -m evaluation.opencap.run_trial --trial-dir data/opencap_trials/subject4_squats1 --out runs/subject4_squats1 --engine cv2dnn
```

The run produces calibration and synchronization outputs, 2D keypoints, a 3D skeleton, BVH, comparison videos, and `metrics.json`. Individual pipeline stages are also available through `aimocap <stage> -h`.

### Run the web service locally

```bash
# Windows PowerShell
$env:AIMOCAP_DEV = "1"
python -m service --port 8000
```

## Next improvements

- Improve timing-reference checks for repeated, periodic movements such as walking.
- Continue refining character retargeting for foot contact and fine motion.
- Expand evaluation across more people, movements, and camera arrangements.
- Reduce CPU processing time for the hosted live workflow.

These are planned areas of improvement; the evaluated results above describe the current measured trials.

## Project structure

| Path | Contents |
|---|---|
| `aimocap/` | Pose, calibration, synchronization, triangulation, retargeting, and export package |
| `evaluation/opencap/` | OpenCap LabValidation trial preparation, evaluation, scoring, and rendering |
| `evaluation/studio/` | Evaluation tools for CMU Panoptic and AIST++ studio clips |
| `service/` | FastAPI service used by the hosted demo |
| `deploy/aws/` | AWS infrastructure and deployment scripts |
| `configs/` | Pipeline and quality-contract configuration |
| `tests/` | Unit and integration tests |
| `docs/` | Technical report, architecture, and capture guidance |

## Data and licensing

AIMoCap original code is available for competition evaluation under the [evaluation-only license](LICENSE); organizers and authorized judges/reviewers may evaluate it, and other use requires separate written permission. Third-party code, models, data, and assets retain their own terms, summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
