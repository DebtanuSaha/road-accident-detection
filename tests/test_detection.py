"""
tests/test_detection.py

Phase 3 tests for YOLO object detection.

These tests use the real, pretrained YOLO model (downloaded once via
Ultralytics into models/yolo/ — see README for details) rather than
mocks, so they validate actual detection behaviour end-to-end:
confidence filtering, target-class filtering, structured output, and
bounding-box drawing. A small bundled test image with real people/bus
is used so detections are deterministic-enough to assert on.

Marked to skip cleanly (not fail) if the model weights aren't
available and can't be fetched in the current environment (e.g. fully
offline CI), since Phase 3's dataclasses/logic can still be tested
independently of model availability.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings, settings as default_settings  # noqa: E402
from engine.detection import BoundingBox, Detection, YOLODetector, YOLODetectorError  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Pure dataclass tests — no model required
# ---------------------------------------------------------------------------

def test_bounding_box_geometry():
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.center == (30.0, 50.0)
    assert box.area == 40.0 * 60.0
    assert box.as_tuple() == (10.0, 20.0, 50.0, 80.0)
    assert box.as_int_tuple() == (10, 20, 50, 80)


def test_detection_to_dict_matches_conceptual_schema():
    det = Detection(
        class_name="car", class_id=2, confidence=0.9137,
        bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
    )
    d = det.to_dict()
    assert set(d.keys()) == {"class", "confidence", "bbox"}
    assert d["class"] == "car"
    assert d["confidence"] == 0.9137
    assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_detect_before_load_raises():
    detector = YOLODetector(default_settings)
    with pytest.raises(YOLODetectorError):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detect_rejects_empty_frame_after_load(yolo_detector):
    with pytest.raises(ValueError):
        yolo_detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))


def test_load_invalid_model_path_raises():
    cfg = Settings(YOLO_MODEL_PATH="/definitely/not/a/real/model/xyz123.pt")
    detector = YOLODetector(cfg)
    with pytest.raises(YOLODetectorError):
        detector.load()


def test_draw_detections_does_not_mutate_original(yolo_detector):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_detections = [
        Detection(class_name="car", class_id=2, confidence=0.9, bbox=BoundingBox(10, 10, 50, 50)),
    ]
    original_copy = frame.copy()
    annotated = YOLODetector.draw_detections(frame, fake_detections)
    assert np.array_equal(frame, original_copy), "draw_detections must not mutate the input frame"
    assert not np.array_equal(annotated, frame), "annotated frame should differ (box was drawn)"
    assert annotated.shape == frame.shape


# ---------------------------------------------------------------------------
# Fixture: a real pretrained detector, loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def yolo_detector():
    detector = YOLODetector(default_settings)
    try:
        detector.load()
    except YOLODetectorError as exc:
        pytest.skip(f"YOLO model unavailable in this environment: {exc}")
    return detector


@pytest.fixture(scope="session")
def sample_frame():
    """A real photo with a bus and pedestrians (checked into tests/fixtures/)."""
    path = FIXTURES_DIR / "sample_scene.jpg"
    if not path.is_file():
        pytest.skip("tests/fixtures/sample_scene.jpg not present")
    frame = cv2.imread(str(path))
    if frame is None:
        pytest.skip("tests/fixtures/sample_scene.jpg could not be decoded")
    return frame


# ---------------------------------------------------------------------------
# Real-model tests
# ---------------------------------------------------------------------------

def test_model_loads_and_reports_target_classes(yolo_detector):
    assert yolo_detector.is_loaded is True


def test_detect_on_blank_frame_returns_no_detections(yolo_detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = yolo_detector.detect(blank)
    assert detections == []


def test_detect_on_real_scene_finds_expected_classes(yolo_detector, sample_frame):
    detections = yolo_detector.detect(sample_frame)
    assert len(detections) > 0

    found_classes = {d.class_name for d in detections}
    # The fixture image contains a bus and several pedestrians.
    assert "bus" in found_classes or "person" in found_classes

    for det in detections:
        assert det.class_name in default_settings.YOLO_TARGET_CLASSES
        assert 0.0 <= det.confidence <= 1.0
        assert det.bbox.width > 0
        assert det.bbox.height > 0
        # bbox must stay within frame bounds
        h, w = sample_frame.shape[:2]
        assert 0 <= det.bbox.x1 < det.bbox.x2 <= w
        assert 0 <= det.bbox.y1 < det.bbox.y2 <= h


def test_detect_respects_confidence_threshold(sample_frame):
    if sample_frame is None:
        pytest.skip("no sample frame")
    strict_cfg = Settings(YOLO_CONFIDENCE_THRESHOLD=0.999)
    strict_detector = YOLODetector(strict_cfg)
    try:
        strict_detector.load()
    except YOLODetectorError as exc:
        pytest.skip(f"YOLO model unavailable: {exc}")

    detections = strict_detector.detect(sample_frame)
    assert detections == [], "an unreasonably high confidence threshold should filter out all detections"


def test_detect_filters_to_target_classes_only(sample_frame):
    if sample_frame is None:
        pytest.skip("no sample frame")
    # Restrict target classes to something not present in the fixture image.
    narrow_cfg = Settings(YOLO_TARGET_CLASSES=["bicycle"])
    narrow_detector = YOLODetector(narrow_cfg)
    try:
        narrow_detector.load()
    except YOLODetectorError as exc:
        pytest.skip(f"YOLO model unavailable: {exc}")

    detections = narrow_detector.detect(sample_frame)
    for det in detections:
        assert det.class_name == "bicycle"


def test_draw_detections_on_real_scene_returns_same_shape(yolo_detector, sample_frame):
    detections = yolo_detector.detect(sample_frame)
    annotated = YOLODetector.draw_detections(sample_frame, detections)
    assert annotated.shape == sample_frame.shape
