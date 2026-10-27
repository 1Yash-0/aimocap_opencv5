# AIMoCap: technical report

Markerless multi-camera motion capture with OpenCV 5, served from AWS.

## 1. Problem

Character animation, sports analysis and physiotherapy all need accurate human motion in
3D. The standard tool, optical marker capture, needs a calibrated studio with eight or
more infrared cameras, reflective markers on the performer and an operator; laboratory
setups cost more than USD 150,000 (Uhlrich et al., 2023). Single-camera AI pose
estimators are cheap but guess depth and drift. A few phones already provide enough views
to recover 3D by geometry, but real recordings bring three practical problems that decide
whether the result is usable: the cameras are not calibrated, they are not synchronized,
and per-frame network output jitters.

AIMoCap is a pipeline that handles all three and produces animation-ready motion: a
calibrated 3D skeleton with 133 whole-body keypoints (body, feet, hands, face) and a BVH
file.

## 2. Users

- Indie game developers and animators who need motion clips without a capture studio.
- Coaches and physiotherapists who want quantitative movement data from phones.
- Researchers who need a transparent, testable multi-view reconstruction baseline.

## 3. System overview

```
videos -> calibrate -> 2D pose -> synchronize -> triangulate + smooth -> BVH
```

Diagrams and the mapping of every stage to its module and OpenCV calls are in
[architecture.md](architecture.md).

### 3.1 Calibration (OpenCV 5)

Each camera films a checkerboard (11 x 8 inner corners, 60 mm squares in the OpenCap
data). The board is detected with `cv2.findChessboardCorners` on a 2x upscaled frame,
refined with `cv2.cornerSubPix` and, as a fallback, `cv2.findChessboardCornersSB` (whose
corner order is normalised). The camera pose is solved with `cv2.solvePnPGeneric` using
the IPPE planar solver, with the phone's factory intrinsics. One frame near the end of
the board clip is used, as in OpenCap's own procedure.

### 3.2 2D pose (OpenCV 5 DNN)

- Person detection: YOLOX-Nano.
- Whole-body pose: RTMW-x (384 x 288, SimCC head), 133 COCO-WholeBody keypoints, decoded
  with sub-pixel parabola refinement.
- Hands: RTMPose-m hand model on a hand crop, falling back to RTMW when they disagree;
  the agreement becomes the hand confidence.
- Face: RTMW re-run on a head crop.
- Scores are calibrated to probabilities (isotonic per joint group).
- Crops are temporally smoothed (centred median of the box centre, rolling maximum of the
  box size) so the network sees a stable view.

All three networks run on the OpenCV 5 DNN engine (`cv2.dnn.readNetFromONNX`,
`DNN_BACKEND_OPENCV`) or, optionally, ONNX Runtime. On the same 60 frames x 2 cameras the
two engines agree to 0.005 px maximum (median 0.00006 px). On a laptop CPU the OpenCV
engine is faster than ONNX Runtime CPU (RTMW-x 378 ms vs 520 ms per crop; 0.39 s vs
0.45 s per camera frame end to end); ONNX Runtime on a CUDA GPU remains the fastest
option (0.06 s per frame).

Phone video brings small, portrait subjects. The OpenCap runner therefore tracks the
person with frame differencing (`morphologyEx`, `connectedComponentsWithStats`), writes a
person-centred crop video and maps keypoints back to source pixels exactly.

### 3.3 Synchronization

Phones start recording at different times. AIMoCap estimates each camera's frame offset
from the 2D keypoints alone: an epipolar (Sampson) search initialises pairwise offsets,
and a joint multi-camera refinement minimises triangulation reprojection error. Only
frames where both cameras see the whole person (keypoint score at least 0.5, box inside
the image) count, with a minimum overlap of 30 frames; a parabola gives sub-frame
precision. No audio, clapper or hardware sync is needed.

### 3.4 Triangulation and smoothing

- Soft per-camera trust (iteratively reweighted, leave-one-out) instead of hard inlier
  switching, which removed the frame-to-frame camera flips that caused left-right jitter.
- A bidirectional constant-acceleration Kalman filter with Rauch-Tung-Striebel smoothing
  in 3D, with trust-weighted measurement noise, Huber weights and a speed-dependent
  process-noise boost so that fast motion is not flattened. The process noise is a
  continuous-time quantity, so the same setting works at 30 and 60 fps.
- A per-point 3D confidence, stored for downstream consumers.

### 3.5 Export

The 3D skeleton is written as BVH. The repository also contains an IK retargeting stage
for FBX rigs (UE5 Manny) and a Blender baking script; that stage was not re-evaluated for
this report (see limitations).

## 4. AWS deployment

The pipeline runs on CPU, which makes a small instance sufficient. The reference
deployment (`deploy/aws/deploy.py`, no manual console steps):

- **EC2** `m7i-flex.large` (2 vCPU, 8 GiB, Ubuntu 24.04) runs nginx, the FastAPI service
  and the pipeline with the OpenCV 5 DNN engine.
- **CloudFront** terminates HTTPS; the instance's security group admits port 80 only from
  the CloudFront origin-facing prefix list. There is no SSH port; administration uses
  **SSM Run Command**.
- **S3** holds the deployment bundle, model weights, prepared trials, precomputed results
  and live run outputs (a lifecycle rule deletes run outputs after 7 days).
- **SSM Parameter Store** holds the web password as a SecureString.
- **CloudWatch** collects service, nginx and bootstrap logs, and custom metrics per run
  (`RunSeconds`, `StageSeconds` by stage, `RunFailed`).
- **IAM**: the instance role can read and write only the deployment bucket, read one
  parameter path, write one log group and one metric namespace. IMDSv2 is required.

The web service shows precomputed full runs and lets a visitor start a live "Quick" run
(three cameras, body keypoints, two seconds of video) on the instance CPU. Each run
streams its stage events, then shows the overlay video, the 3D comparison against
motion capture, the metrics and a BVH download. Deployment, update, status, stop/start
and teardown are single commands; the whole stack is tagged and can be removed with
`deploy.py down`.

## 5. Evaluation

### 5.1 Data

- **OpenCap LabValidation** (Uhlrich et al., 2023; Apache-2.0). Five iPhones (60 fps,
  720 x 1280 portrait), per-camera checkerboard clips, raw unsynchronized videos, and
  synchronous marker-based motion capture (100 Hz). Participants consented to public
  release of their videos, so this dataset is used for every visual in the demo.
- **CMU Panoptic** and **AIST++** (studio, hardware-synchronized). Research-only; used for
  development numbers only. No frames are shown or redistributed.

Rules followed throughout: parameters are chosen on one clip and frozen before held-out
clips are scored; gates are written down before a run; no smoothing that reduces the
speed of fast motion is accepted.

### 5.2 Metrics

- Joint position error: 12 joints (hips, knees, ankles, shoulders, elbows, wrists) built
  from markers, per-frame rigid (Kabsch) alignment, median and 95th percentile, at **zero
  time shift**.
- Acceleration error: median |a_pred - a_mocap| in mm/frame^2 after resampling mocap to
  60 fps; jitter: the same quantity on the 20 % slowest frames.
- Fast-motion speed ratio: 95th-percentile joint speed, prediction over mocap (1.0 means
  fast motion is reproduced at full speed).
- Calibration: rotation angle (deg) and camera-centre distance (mm) against OpenCap's own
  solved cameras.
- Synchronization: absolute frame error against offsets derived from motion capture
  (mocap joints projected into each camera and aligned to that camera's 2D keypoints).

### 5.3 Results on OpenCap

Subject 4, squats (held out; all parameters fixed on other trials):

| metric | AIMoCap, 5 cam | OpenCap, 5 cam | OpenCap, 3 cam |
|---|---:|---:|---:|
| joint error, median (mm) | 41.4 | 40.4 | 39.2 |
| joint error, p95 (mm) | 54.5 | 42.7 | 43.3 |
| acceleration error (mm/frame^2) | 0.98 | 1.46 | 1.74 |
| jitter, slowest 20 % (mm/frame^2) | 0.44 | 1.24 | 1.42 |
| fast-motion speed ratio | 1.04 | 1.02 | 1.05 |

Calibration on the same trial: largest error over five cameras 0.09 deg and 7.2 mm.
Synchronization: estimated offsets within 1 frame of the mocap-derived truth on all five
cameras.

Subject 2, squats (same frozen settings):

| metric | AIMoCap, 5 cam | OpenCap, 5 cam | OpenCap, 3 cam |
|---|---:|---:|---:|
| joint error, median (mm) | 50.9 | 46.1 | 45.6 |
| joint error, p95 (mm) | 57.5 | 48.5 | 49.7 |
| acceleration error (mm/frame^2) | 1.72 | 1.99 | 2.44 |
| jitter, slowest 20 % (mm/frame^2) | 1.07 | 2.09 | 2.43 |
| fast-motion speed ratio | 0.96 | 0.97 | 0.99 |

Calibration: largest error 0.14 deg and 12.6 mm. Synchronization: within 2 frames on the
cameras with resolved truth; the mocap-derived truth for one camera was ambiguous and is
reported as unresolved.

Subject 2, walking (same frozen settings; 60 fps):

| metric | AIMoCap, 5 cam | OpenCap, 5 cam | OpenCap, 3 cam |
|---|---:|---:|---:|
| joint error, median (mm) | 46.4 | 44.7 | 45.3 |
| joint error, p95 (mm) | 50.2 | 51.6 | 53.0 |
| acceleration error (mm/frame^2) | 1.08 | 1.07 | 1.07 |
| jitter, slowest 20 % (mm/frame^2) | 0.70 | 1.44 | 1.56 |
| fast-motion speed ratio | 1.02 | 1.01 | 1.01 |

Joint-error scores use 95 AIMoCap frames and 158 frames for each OpenCap reference; smoothness
scores for all three sources use 95 aligned frames. Calibration:
largest error 0.19 deg and 15.4 mm. The automatic per-camera truth scan labels Cam1 and Cam3
resolved, with discrepancies of 86 and 84 frames. Walking is periodic and these mocap-derived
references are unreliable here; those discrepancies are not validated synchronization errors.
Cam4's reference is unresolved. The pipeline estimated offsets for all five cameras.

Subject 4, sit-to-stand (same frozen settings; 60 fps; joint-error scores use 839 AIMoCap frames):

| metric | AIMoCap, 5 cam | OpenCap, 5 cam | OpenCap, 3 cam |
|---|---:|---:|---:|
| joint error, median (mm) | 41.6 | 41.3 | 43.0 |
| joint error, p95 (mm) | 47.6 | 43.4 | 45.9 |
| acceleration error (mm/frame^2) | 1.24 | 1.37 | 1.72 |
| jitter, slowest 20 % (mm/frame^2) | 0.37 | 1.03 | 1.29 |
| fast-motion speed ratio | 0.99 | 0.96 | 0.98 |

Calibration: largest error 0.09 deg and 7.2 mm. Synchronization: Cam0, Cam3 and Cam4 are
within 1 frame of their mocap-derived references. Cam1 and Cam2 have unresolved references;
the pipeline estimated their offsets, but these cameras cannot be scored against truth.
Joint-error scores use 1,399 frames for each OpenCap reference; smoothness scores for all
three sources use 839 aligned frames.

Across both squat trials AIMoCap is 1-5 mm behind OpenCap on median error and worse on the
95th percentile, but markedly smoother: acceleration error 14-33 % lower and still-frame
jitter 49-65 % lower than OpenCap's five-camera result. It also provides hand and face
keypoints, which OpenCap does not track.

### 5.4 Results on studio data (development)

Panoptic and AIST++ held-out clips, current production settings:

| clip | median (mm) | p95 (mm) |
|---|---:|---:|
| Panoptic 171026_pose3 | 20.6 | 57.6 |
| AIST++ break dance (30 fps) | 23.7 | 63.9 |

Against the previous triangulation stage, body jitter fell by 48-57 % and fast-frame
jitter by 32-48 % with unchanged median error and 100 % coverage. The 2D stage (RTMW-x,
stable crops, sub-pixel decoding) reduced 2D body jitter by 35-40 %; hand refinement
reduced finger error by 11 % and finger jitter by 16 %; face refinement reduced face error
by 15 % and face jitter by 37 %.

### 5.5 Findings during evaluation

- **Dataset sync offsets are approximate.** The OpenCap clips labelled "synced with mocap"
  are offset per camera by up to several frames, so they cannot serve as sync ground
  truth. Projecting mocap into each camera gave reliable per-camera truth; scored against
  it, our estimator was within one frame where the clip offsets were within one frame for
  only 31 % of camera-trial pairs.
- **A frame-rate bug.** Triangulation assumed 30 fps when the input lacked a frame-rate
  field, which mis-scaled the smoother for 60 fps phone video. The stage now reads the
  rate from the input, its timestamps or the sync manifest and refuses to guess.
- **Smoother setting.** A tighter process-noise setting (q = 1000 instead of 30000) was
  chosen after it passed pre-registered gates on the studio test clips (median, p95, lag,
  fast-motion speed) and reduced jitter there by 10-31 %.

### 5.6 Failure cases

- **Periodic motion can alias synchronization.** On walking, a shift of one full gait
  cycle fits almost as well as the true offset. Where the mocap-derived truth itself was
  ambiguous we report the camera as unresolved instead of scoring it.
- **Synchronization error of 2 frames** on one camera of subject 2 squats (the camera with
  the largest offset, 46 frames).
- **Tail errors.** The 95th-percentile joint error is higher than OpenCap's, driven by
  occasional wrong camera pairs on wrists and hands and by hip-centre definition
  differences between COCO keypoints and markers.
- **Person detection on portrait phone video** needed the crop tracker; the stock detector
  path misses small, tall subjects.

## 6. Limitations

- Accuracy is evaluated on body joints; hands and face have no marker ground truth in
  OpenCap. Facial expressions are out of scope for full-body camera distances.
- The OpenCap motions are slow to moderate (walking, squats). Fast-motion claims rest on
  the studio datasets.
- Factory intrinsics are used; intrinsic calibration from the board is not evaluated.
- The IK retargeting stage (FBX rigs) has known issues (limb roll, foot orientation in
  turns) and is not part of the evaluated pipeline.
- The cloud demo runs on two vCPUs; a live run is limited to a short clip.

## 7. Responsible use

- The public demo uses only OpenCap LabValidation videos, released with participant
  consent under Apache-2.0. The deployment does not accept uploads of personal video.
- Research-only datasets (Panoptic, AIST++) contribute numbers only; no frames are shown.
- The system estimates skeleton keypoints; it does not identify people and stores no
  biometric templates.
- Access to the demo is password protected and served over HTTPS; run outputs expire
  after 7 days.
- Models: RTMW and RTMPose (OpenMMLab, Apache-2.0 code; weights trained on public
  datasets with their own terms), YOLOX (Apache-2.0). Excluded on licence grounds:
  non-commercial models such as OpenPose, HaMeR, WiLoR, Sapiens and SMPL/MANO.

## 8. Reproducibility

- Pinned dependencies: `requirements.txt` (CPU) and `requirements-gpu.txt`.
- Models: `python scripts/fetch_models.py`.
- One trial end to end: `python -m evaluation.opencap.run_trial --trial-dir ... --out ...`;
  every run writes its configuration, per-stage timings and metrics to `metrics.json`.
- Tests: `python -m pytest`.
- Deployment: `python deploy/aws/deploy.py up`, documented in `deploy/aws/README.md`.

## References

- S. D. Uhlrich et al. OpenCap: Human movement dynamics from smartphone videos.
  PLOS Computational Biology 19(10), 2023.
- T. Jiang et al. RTMW: Real-time multi-person 2D and 3D whole-body pose estimation.
  arXiv:2407.08634, 2024.
- H. Joo et al. Panoptic Studio: a massively multiview system for social interaction
  capture. TPAMI, 2017.
- R. Li et al. AI Choreographer: music conditioned 3D dance generation with AIST++.
  ICCV, 2021.
