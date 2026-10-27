# Third-party notices

This repository contains third-party components and materials with separate
rights. Their terms remain in effect independently of the AIMoCap license; this
notice does not relicense them. See [LICENSE](LICENSE) for the scope of current
AIMoCap terms. Some fitted calibration artifacts identified below are included
in this repository; model weights and dataset videos are not tracked here.
Showcase assets may also be served by the deployed site.

## Runtime libraries

- OpenCV (opencv-python): Apache License 2.0; see [OpenCV license information](https://opencv.org/license/).
- ONNX Runtime (onnxruntime, onnxruntime-gpu; optional, `cpu` or `gpu` extra): MIT License.
- rtmlib (optional, used by the default `--pose-backend rtmw`): Apache License 2.0 for the library code; see [rtmlib's licensing discussion](https://github.com/Tau-J/rtmlib/issues/65). This does not clear pretrained weights or their training data.
- Other Python dependencies (NumPy, SciPy, Pillow, Matplotlib, imageio-ffmpeg, and the optional service packages): see each package's metadata.
- The imageio-ffmpeg Python wrapper is BSD-2-Clause. The locally selected FFmpeg executable reports `--enable-gpl --enable-version3` and includes libx264/libx265. The wrapper license does not cover the executable; binary redistribution requires separate GPL compliance review. See [FFmpeg legal information](https://ffmpeg.org/legal.html).

## Pose models

- RTMPose / RTMW (OpenMMLab MMPose): [MMPose source code](https://github.com/open-mmlab/mmpose#license) is Apache License 2.0. The exact default RTMW cocktail14 384x288 and RTMPose-m hand5 256x256 pretrained weights are downloaded at run time and are not included here. The commercial permissions for these exact weights and their training datasets have not been established; treat their commercial use as **not cleared** pending written rights-holder terms or replacement with demonstrably cleared models. See [rtmlib's licensing discussion](https://github.com/Tau-J/rtmlib/issues/65).
- CIGPose wrapper code (optional, `--pose-backend cigpose`, from [cigpose-onnx](https://github.com/namas191297/cigpose-onnx)): Apache License 2.0. Optional pose weights retain separate model and training-data terms; the wrapper license does not clear those weights.
- YOLOX person detector (yolox_nano.onnx used by the CIGPose path): Apache License 2.0, as identified by the [upstream detector project](https://github.com/namas191297/cigpose-onnx#person-detector).
- MediaPipe Face Landmarker (Google; optional webcam demo only): Apache License 2.0.
- COCO annotations and images have their own source-specific terms; COCO images retain their original Flickr licenses. Do not infer terms for COCO-WholeBody from generic COCO terms.
- [COCO-WholeBody terms](https://github.com/jin-s13/COCO-WholeBody#terms-of-use) restrict use to research/noncommercial purposes. Although that page labels a link “CC BY 4.0,” it resolves to [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). These restrictions are distinct from permissions for generic COCO annotations.

## Evaluation datasets and included fitted artifacts

- CMU Panoptic Studio dataset: [official terms](https://domedb.perception.cs.cmu.edu/) restrict use to research, exclude commercial use, and prohibit redistribution without permission. Panoptic-derived `aimocap/pose/calibration/part_refine_score_calibration.json` and `aimocap/triangulate/confidence3d.json` are included; their provenance is documented in the files. Their legal status as copyrightable derivatives and applicable distribution permissions were not determined by the audit. Review rights or replace/re-fit using cleared data before commercial reuse.
- AIST++ (Google LLC): [annotations](https://google.github.io/aistplusplus_dataset/factsfigures.html) are CC BY 4.0 and require attribution. The included `aimocap/pose/calibration/rtmw_x_score_calibration.json` identifies fitting on AIST++ observations. Original [AIST Dance DB footage terms](https://aistdancedb.ongaaccel.jp/terms_of_use/) limit use to academic research; commercial or non-research use requires prior written consent, and unauthorized redistribution is prohibited. Annotation licensing does not grant rights to the footage.
- OpenCap LabValidation (Uhlrich et al., 2023): a dataset loader exists at `evaluation/opencap/dataset.py`. The official [SimTK project](https://simtk.org/projects/opencap) packages LabValidation_withVideos and related data with an Apache-2.0 Use Agreement (Stanford University, 2022); participant consent and scope restrictions still apply. Do not include withheld participant material. The deployed site serves video excerpts, so “not tracked in Git” does not mean no video material is served. The [study](https://doi.org/10.1371/journal.pcbi.1011462) documents consent for publicly released identifiable videos.
- Manny character previews: the preview rig was exported from Unreal Engine's template/sample. Epic's [Unreal Engine EULA](https://www.unrealengine.com/eula/unreal) governs applicable Samples/Templates and rendered output; it is separate from AIMoCap's license. No FBX is tracked. Do not treat the rig as MIT-licensed AIMoCap material.

## Scope and limits

The licenses above apply to their respective components, data, and models. This
notice is an evidence-based summary, not a legal opinion or commercial clearance
certificate. In particular, downloadable model weights do not by themselves
establish unrestricted rights, and this repository's license cannot override
third-party terms. No patent search or freedom-to-operate review was performed.
