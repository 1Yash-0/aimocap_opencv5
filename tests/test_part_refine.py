from __future__ import annotations

import numpy as np
import pytest

from aimocap import cli
from aimocap.pose import multi_video
from aimocap.pose.infer import Pose2D
from aimocap.pose.part_refine import (
    estimate_hand_sizes,
    hand_agreement_distances,
    hand_crop_boxes,
    hand_size_primary,
    h4_switch,
    head_crop_boxes,
)
from aimocap.pose.score_calibration import (
    calibrate_hand_agreement_scores,
    hand_agreement_probability,
    load_part_calibration,
)


def _synthetic_body(frames: int = 11):
    kp = np.full((frames, 133, 2), np.nan, dtype=np.float32)
    sc = np.zeros((frames, 133), dtype=np.float32)
    for i in range(frames):
        # Stable body/face anchors and deliberately varying hand landmarks.
        kp[i, :5] = np.asarray([[100, 80], [95, 75], [105, 75], [85, 80], [115, 80]], dtype=np.float32)
        sc[i, :5] = 1.0
        kp[i, 7] = [80 + i, 120]
        kp[i, 9] = [60 + i, 150]
        kp[i, 8] = [120 + i, 120]
        kp[i, 10] = [140 + i, 150]
        sc[i, [7, 8, 9, 10]] = 1.0
        for start, cx in ((91, 55 + i), (112, 145 + i)):
            for j in range(21):
                kp[i, start + j] = [cx + (j % 5), 155 + (j // 5)]
            sc[i, start:start + 21] = 1.0
        kp[i, 23:91] = np.asarray([[100 + (j % 8), 90 + (j // 8)] for j in range(68)], dtype=np.float32)
        sc[i, 23:91] = 1.0
    return kp, sc


def test_crop_geometry_matches_accepted_experiment_functions():
    # Keep this comparison against the frozen experiment implementation: the
    # production port must not silently change the H4/F2 crop protocol.
    try:
        from scripts.phase1b_hand_ab3 import crop_geometry as hand_geometry
        from scripts.phase1b_face_ab2 import crop_geometry as face_geometry
    except ImportError:
        pytest.skip("frozen experiment scripts scripts/phase1b_*.py are not part of this repository")

    kp, sc = _synthetic_body()
    centers, sides, _ = hand_geometry(kp, sc, "left", 1.6)
    expected_hand = np.asarray(
        [[c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2] for c, s in zip(centers, sides)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(hand_crop_boxes(kp, sc, "left"), expected_hand, atol=1e-3)

    centers, sides = face_geometry(kp, sc)
    expected_face = np.asarray(
        [[c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2] for c, s in zip(centers, sides)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(head_crop_boxes(kp, sc), expected_face, atol=1e-3)
    assert hand_crop_boxes(kp, sc).shape == (len(kp), 2, 4)


def test_h4_switch_score_and_distance_gate():
    raw = np.zeros((21, 2), dtype=np.float32)
    raw_scores = np.full(21, 0.8, dtype=np.float32)
    candidate = np.ones((21, 2), dtype=np.float32) * 10.0
    candidate_scores = np.full(21, 0.2, dtype=np.float32)
    candidate_scores[1] = 0.05
    candidate[2] = [100.0, 100.0]
    merged, _, source = h4_switch(raw, raw_scores, candidate, candidate_scores, (0, 0, 40, 40))
    assert np.allclose(merged[0], [10, 10])
    assert np.allclose(merged[1], [0, 0])
    assert np.allclose(merged[2], [0, 0])
    assert source[0] == 1 and source[1] == 0 and source[2] == 0


def test_hand_agreement_probability_is_monotone_decreasing():
    table = load_part_calibration()
    d = np.concatenate([np.linspace(0.0, 3.0, 3000), np.asarray([10.0, 100.0])])
    probability = hand_agreement_probability(d, table)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.all(np.diff(probability) <= 1e-12)
    assert probability[0] > probability[-1]


def test_hand_agreement_source_zero_keeps_body_score():
    table = load_part_calibration()
    scores = np.full((2, 133), 0.37, dtype=np.float64)
    distances = np.full((2, 133), np.nan, dtype=np.float64)
    source = np.zeros((2, 133), dtype=np.uint8)
    source[0, 91:112] = 1
    distances[0, 91:112] = 0.01
    distances[0, 112:133] = 0.40
    out = calibrate_hand_agreement_scores(scores, distances, source, table=table)
    assert np.all(out[1] == 0.37)  # no refined points in this row
    assert np.all(out[0, 112:133] == 0.37)  # hand_source == 0 keeps the body score
    assert out[0, 95] != 0.37  # hand_source == 1 is replaced by f(d)
    assert np.all(out[:, :23] == 0.37)  # body indices untouched


def test_hand_agreement_is_nan_safe():
    table = load_part_calibration()
    scores = np.full((2, 133), 0.5, dtype=np.float64)
    scores[0, 95] = np.nan
    distances = np.full((2, 133), np.nan, dtype=np.float64)
    source = np.zeros((2, 133), dtype=np.uint8)
    source[:, 91:112] = 1
    distances[1, 95] = 0.02
    out = calibrate_hand_agreement_scores(scores, distances, source, table=table)
    assert np.isnan(out[0, 95])  # NaN body score and NaN d stay NaN
    assert np.all(out[0, 96:112] == 0.5)  # NaN d keeps the body score
    assert out[1, 95] != 0.5  # finite d is mapped


def test_hand_size_primary_uses_confident_wrist_and_middle_tip():
    table = load_part_calibration()
    config = table["hand_agreement"]["hand_size"]
    frames = 21
    body = np.full((frames, 133, 2), np.nan, dtype=np.float64)
    scores = np.zeros((frames, 133), dtype=np.float64)
    body[:, 91] = [100.0, 100.0]
    body[:, 103] = [100.0, 200.0]
    scores[:, [91, 103]] = 1.0
    sizes, fallback = estimate_hand_sizes(body, scores, config)
    assert sizes.shape == (frames, 2)
    np.testing.assert_allclose(sizes[:, 0], 100.0, rtol=1e-12)
    assert not fallback[:, 0].any()
    # The right hand has no usable points, so it uses the sequence-median fill.
    assert np.all(np.isfinite(sizes[:, 1]))


def test_hand_size_fallback_uses_forearm_and_k():
    table = load_part_calibration()
    config = table["hand_agreement"]["hand_size"]
    frames = 21
    body = np.full((frames, 133, 2), np.nan, dtype=np.float64)
    scores = np.zeros((frames, 133), dtype=np.float64)
    body[:, 7] = [200.0, 200.0]
    body[:, 9] = [200.0, 300.0]
    body[:, 8] = [600.0, 200.0]
    body[:, 10] = [600.0, 300.0]
    scores[:, [7, 9, 8, 10]] = 1.0
    sizes, fallback = estimate_hand_sizes(body, scores, config)
    expected = 100.0 * float(config["forearm_scale_k"])
    np.testing.assert_allclose(sizes, expected, rtol=1e-12)
    assert fallback.all()


def test_hand_agreement_distance_marks_source_and_non_hand_points():
    final = np.full((2, 133, 2), np.nan, dtype=np.float64)
    body = np.full((2, 133, 2), np.nan, dtype=np.float64)
    source = np.zeros((2, 133), dtype=np.uint8)
    sizes = np.full((2, 2), 10.0, dtype=np.float64)
    body[0, 91] = [0.0, 0.0]
    final[0, 91] = [0.0, 5.0]
    source[0, 91] = 1
    body[0, 92] = [0.0, 0.0]
    final[0, 92] = [0.0, 20.0]
    source[0, 92] = 1
    final[0, 112] = [100.0, 100.0]
    source[0, 112] = 0
    source[1, 93] = 1
    distances = hand_agreement_distances(final, body, source, sizes)
    assert distances[0, 91] == 0.5
    assert distances[0, 92] == 2.0
    assert distances[0, 112] == 0.0  # hand_source == 0 is zero by definition
    assert np.isnan(distances[0, 0])
    assert np.isnan(distances[1, 93])  # source == 1 but no finite geometry


def test_cli_part_refine_default_is_face_and_legacy_off_parses():
    # S3 clip2 unseen gate: face passed, hands missed the 20% finger-median
    # threshold, so the CLI default is face while the library stays opt-in.
    parser = cli.build_parser()
    default = parser.parse_args(["multipose", "a.mp4", "-o", "b.npz"])
    assert default.part_refine == "hands_face"
    assert parser.parse_args(["multipose", "a.mp4", "-o", "b.npz", "--part-refine", "off"]).part_refine == "off"
    assert parser.parse_args(["multipose", "a.mp4", "-o", "b.npz", "--part-refine", "hands_face"]).part_refine == "hands_face"


def test_library_part_refine_default_stays_off():
    import inspect

    signature = inspect.signature(multi_video.run_multi_video)
    assert signature.parameters["part_refine"].default == "off"


def test_default_off_npz_has_no_refinement_keys(tmp_path, monkeypatch):
    class FakeCapture:
        def __init__(self, _path):
            self.i = 0
            self.frames = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == multi_video.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == multi_video.cv2.CAP_PROP_POS_FRAMES:
                return self.i
            if prop == multi_video.cv2.CAP_PROP_POS_MSEC:
                return (self.i * 1000.0 / 30.0)
            return 0.0

        def read(self):
            if self.i >= len(self.frames):
                return False, None
            frame = self.frames[self.i]
            self.i += 1
            return True, frame

        def set(self, _prop, value):
            self.i = int(value)

        def release(self):
            pass

    class FakeEstimator:
        input_w, input_h = 288, 384
        active_providers = ["CPUExecutionProvider"]
        simcc_decode = "argmax"

        def estimate(self, frame, pick="all"):
            del frame, pick
            points = np.arange(266, dtype=np.float32).reshape(133, 2)
            return [Pose2D(points, np.ones(133, dtype=np.float32), np.asarray([0, 0, 48, 32], dtype=np.float32))]

    source = tmp_path / "cam.mp4"
    source.write_bytes(b"placeholder")
    output = tmp_path / "default.npz"
    monkeypatch.setattr(multi_video.cv2, "VideoCapture", FakeCapture)
    multi_video.run_multi_video(
        [source], output, estimator=FakeEstimator(), single_subject=False,
        crop_mode="detector", simcc_decode="argmax", part_refine="off",
    )
    with np.load(output, allow_pickle=False) as data:
        assert "keypoints_body_raw" not in data
        assert "hand_source" not in data
        assert "hand_scores_raw" not in data
        assert "face_scores_raw" not in data
        assert "face_source" not in data
        assert "part_refine" not in data
        assert "part_score_calibration_sha256" not in data
        assert "hand_agreement_d" not in data
        assert "hand_size_px" not in data
        assert np.array_equal(data["keypoints"][0, 0], np.arange(266, dtype=np.float32).reshape(133, 2))
        assert np.array_equal(data["scores"][0, 0], np.ones(133, dtype=np.float32))



def test_refiner_normalises_center_side_crop_regression():
    from aimocap.pose.part_refine import HandRefiner, _normalise_crop

    assert _normalise_crop((np.asarray([20.0, 30.0]), 40.0)) == (0.0, 10.0, 40.0, 50.0)
    refiner = object.__new__(HandRefiner)
    refiner.threshold = 0.1
    seen = []

    def fake_predict(_frame, crop):
        seen.append(crop)
        return np.ones((21, 2), dtype=np.float32), np.ones(21, dtype=np.float32)

    refiner._predict = fake_predict
    body = np.zeros((133, 2), dtype=np.float32)
    scores = np.ones(133, dtype=np.float32)
    refiner.refine(np.zeros((32, 32, 3), dtype=np.uint8), body, scores,
                   "left", (np.asarray([20.0, 30.0]), 40.0))
    assert seen == [(0.0, 10.0, 40.0, 50.0)]


@pytest.mark.parametrize("mode", ["hands_face", "face", "hands"])
def test_part_refinement_end_to_end_uses_per_part_crops(tmp_path, monkeypatch, mode):
    """Exercise crop production and the enabled refiners' consumers."""
    from aimocap.pose import part_refine
    from aimocap.pose.part_refine import FaceRefiner, HandRefiner

    class FakeCapture:
        def __init__(self, _path):
            self.i = 0
            self.frames = [np.full((48, 64, 3), 20 + i, dtype=np.uint8) for i in range(3)]

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == multi_video.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == multi_video.cv2.CAP_PROP_POS_FRAMES:
                return self.i
            if prop == multi_video.cv2.CAP_PROP_POS_MSEC:
                return self.i * 1000.0 / 30.0
            return 0.0

        def read(self):
            if self.i >= len(self.frames):
                return False, None
            frame = self.frames[self.i]
            self.i += 1
            return True, frame

        def set(self, _prop, value):
            self.i = int(value)

        def release(self):
            pass

    class FakeEstimator:
        input_w, input_h = 32, 32
        active_providers = ["CPUExecutionProvider"]
        simcc_decode = "argmax"

        def estimate(self, frame, pick="all"):
            del frame, pick
            points = np.full((133, 2), 16.0, dtype=np.float32)
            scores = np.ones(133, dtype=np.float32)
            return [Pose2D(points, scores, np.asarray([8, 8, 24, 24], dtype=np.float32))]

    class FakeSession:
        def __init__(self, count):
            self.count = count

        def run(self, _outputs, _feeds):
            values = np.zeros((1, self.count, 32), dtype=np.float32)
            return [values, values.copy()]

    hand = object.__new__(HandRefiner)
    hand.width = hand.height = 32
    hand.input_name = "input"
    hand.x_index = hand.y_index = 0
    hand.split_ratio = 1.0
    hand.session = FakeSession(21)
    hand.mean = np.zeros(3, dtype=np.float32)
    hand.std = np.ones(3, dtype=np.float32)
    hand.threshold = 0.1
    hand.active_providers = ["CPUExecutionProvider"]
    hand.sha256 = "mock-hand"
    hand.last_raw_scores = np.full(21, np.nan, dtype=np.float32)

    face_estimator = type(
        "FakeFaceEstimator",
        (),
        {
            "input_w": 32,
            "input_h": 32,
            "_mean": np.zeros(3, dtype=np.float32),
            "_std": np.ones(3, dtype=np.float32),
            "split_ratio": 1.0,
            "_run_pose": lambda self, _tensor: (
                np.zeros((1, 133, 32), dtype=np.float32),
                np.zeros((1, 133, 32), dtype=np.float32),
            ),
        },
    )()
    face = object.__new__(FaceRefiner)
    face.estimator = face_estimator
    face.active_providers = ["CPUExecutionProvider"]
    face.sha256 = "mock-face"

    monkeypatch.setattr(multi_video.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(part_refine, "HandRefiner", lambda **_kwargs: hand)
    monkeypatch.setattr(part_refine, "FaceRefiner", lambda **_kwargs: face)
    source = tmp_path / "cam.mp4"
    source.write_bytes(b"placeholder")
    output = tmp_path / "refined.npz"
    multi_video.run_multi_video(
        [source], output, estimator=FakeEstimator(), single_subject=False,
        part_refine=mode, progress_every=0,
    )

    with np.load(output, allow_pickle=False) as data:
        assert data["keypoints"].shape == (3, 1, 133, 2)
        assert data["scores"].shape == (3, 1, 133)
        assert data["keypoints_body_raw"].shape == (3, 1, 133, 2)
        assert data["hand_source"].shape == (3, 1, 133)
        assert data["hand_scores_raw"].shape == (3, 1, 133)
        assert data["part_refine"].item() == mode
        from aimocap.pose.score_calibration import load_part_calibration

        assert data["part_score_calibration_sha256"].item() == load_part_calibration()["_sha256"]
        refined = [data["scores"][..., 91:133][data["hand_source"][..., 91:133] == 1]]
        if "face_source" in data:
            refined.append(data["scores"][..., 23:91][data["face_source"][..., 23:91] == 1])
        refined = np.concatenate(refined)
        if "face_source" not in data:
            # This fixture's uniform SIMCC scores stay below the H4 gate, so no
            # hand point is refined; the off/refined hand path is covered by
            # the calibration unit tests above.
            assert refined.size == 0
        else:
            assert refined.size > 0
            assert np.isfinite(refined).all()
            assert ((refined >= 0.0) & (refined <= 1.0)).all()
        if mode == "hands":
            assert "face_scores_raw" not in data
            assert "face_source" not in data
        else:
            assert data["face_scores_raw"].shape == (3, 1, 133)
            assert data["face_scores_raw"].dtype == np.float32
            assert data["face_source"].shape == (3, 1, 133)
            assert data["face_source"].dtype == np.uint8
            assert np.isnan(data["face_scores_raw"][..., :23]).all()
            assert np.isnan(data["face_scores_raw"][..., 91:]).all()
            assert np.isfinite(data["face_scores_raw"][..., 23:91]).all()
            assert np.array_equal(
                data["face_source"][..., 23:91],
                np.ones((3, 1, 68), dtype=np.uint8),
            )
        if "hands" in mode:
            assert data["hand_agreement_d"].shape == (3, 1, 133)
            assert data["hand_agreement_d"].dtype == np.float32
            assert np.isnan(data["hand_agreement_d"][..., :91]).all()
            assert data["hand_size_px"].shape == (3, 1, 2)
            assert data["hand_size_px"].dtype == np.float32
            assert np.isfinite(data["hand_size_px"]).all()
            source_hand = data["hand_source"][..., 91:133]
            assert np.all(data["hand_agreement_d"][..., 91:133][source_hand == 0] == 0)
            assert np.allclose(data["scores"][..., 91:133][source_hand == 0], 1.0)
            # Face diagnostics must not leak into the hand diagnostics.
            assert np.isnan(data["hand_scores_raw"][..., 23:91]).all()


def test_part_calibration_is_monotonic_probability_and_source_masked():
    from aimocap.pose.score_calibration import (
        calibrate_refined_part_scores,
        load_part_calibration,
    )

    table = load_part_calibration()
    grid = np.linspace(-2.0, 8.0, 1500)
    for group in table["groups"]:
        columns = np.asarray(
            [int(k) for k, g in table["part_joint_index_to_group"].items() if g == group]
        )
        raw = np.zeros((len(grid), 133), dtype=np.float64)
        raw[:, columns] = grid[:, None]
        source = np.zeros((len(grid), 133), dtype=np.uint8)
        source[:, columns] = 1
        body = np.full((len(grid), 133), 0.25, dtype=np.float64)
        probability = calibrate_refined_part_scores(
            body,
            hand_scores_raw=raw if group.startswith("hand_") else None,
            hand_source=source if group.startswith("hand_") else None,
            face_scores_raw=raw if group.startswith("face_") else None,
            face_source=source if group.startswith("face_") else None,
        )
        values = probability[:, columns]
        assert np.isfinite(values).all()
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert np.all(np.diff(values, axis=0) >= -1e-12)
        untouched = np.ones(133, dtype=bool)
        untouched[columns] = False
        assert np.all(probability[:, untouched] == 0.25)


def test_part_calibration_is_nan_safe_and_keeps_unrefined_scores():
    from aimocap.pose.score_calibration import (
        calibrate_refined_part_scores,
        load_part_calibration,
    )

    table = load_part_calibration()
    columns = np.asarray(
        [int(k) for k, g in table["part_joint_index_to_group"].items() if g == "hand_fingertips"]
    )
    raw = np.zeros((2, 133), dtype=np.float64)
    raw[0, columns] = np.nan
    raw[1, columns] = 2.0
    source = np.zeros((2, 133), dtype=np.uint8)
    source[:, columns] = 1
    source[1, columns[-1]] = 0
    body = np.full((2, 133), 0.6, dtype=np.float64)
    body[0, columns] = np.nan
    body[1, columns[-1]] = 0.35
    out = calibrate_refined_part_scores(
        body, hand_scores_raw=raw, hand_source=source
    )
    assert out.shape == body.shape
    assert np.isnan(out[0, columns]).all()  # NaN body and NaN raw stay NaN
    assert out[1, columns[-1]] == 0.35  # source == 0 keeps the body score
    assert out[1, columns[0]] != body[1, columns[0]]  # refined point is calibrated
    assert 0.0 <= out[1, columns[0]] <= 1.0
    untouched = np.ones(133, dtype=bool)
    untouched[columns] = False
    assert np.all(out[:, untouched] == 0.6)

def test_part_calibration_matches_experiment_predictions():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import phase1b_score_calibration as experiment
    except ImportError:
        pytest.skip("frozen experiment script scripts/phase1b_score_calibration.py is not part of this repository")

    from aimocap.pose.score_calibration import (
        calibrate_refined_part_scores,
        load_part_calibration,
    )

    table = load_part_calibration()
    rng = np.random.default_rng(2)
    raw = np.zeros((500, 133), dtype=np.float64)
    source = np.zeros((500, 133), dtype=np.uint8)
    for index in table["part_joint_index_to_group"]:
        raw[:, int(index)] = rng.uniform(0.0, 8.0, 500)
        source[:, int(index)] = 1
    out = calibrate_refined_part_scores(
        np.zeros((500, 133)),
        hand_scores_raw=raw,
        hand_source=source,
        face_scores_raw=raw,
        face_source=source,
    )
    for index, group in table["part_joint_index_to_group"].items():
        expected = experiment.predict(raw[:, int(index)], table["groups"][group])
        assert np.allclose(out[:, int(index)], expected, atol=1e-12)
