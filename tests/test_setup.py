"""
tests/test_setup.py

Phase 1 tests: verifies the configuration layer, directory setup, and
threshold config — independent of any CV/backend code (none exists
yet as of Phase 1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings, Settings  # noqa: E402
from config.threshold_loader import load_thresholds  # noqa: E402


def test_settings_instantiates():
    assert isinstance(settings, Settings)
    assert settings.APP_NAME == "road-accident-detection"


def test_yolo_target_classes_are_road_users():
    expected = {"car", "motorcycle", "bus", "truck", "bicycle", "person"}
    assert expected.issubset(set(settings.YOLO_TARGET_CLASSES))


def test_required_directories_exist():
    required = [
        settings.LOG_DIR,
        settings.DATA_DIR,
        settings.INPUT_VIDEO_DIR,
        settings.INPUT_STREAM_DIR,
        settings.OUTPUT_ANNOTATED_DIR,
        settings.OUTPUT_PROCESSED_DIR,
        settings.EVIDENCE_IMAGE_DIR,
        settings.EVIDENCE_VIDEO_DIR,
        settings.DATABASE_DIR,
    ]
    for d in required:
        assert Path(d).is_dir(), f"Missing directory: {d}"


def test_thresholds_yaml_loads_and_has_required_sections():
    data = load_thresholds()
    for section in ("collision", "motion", "accident", "evidence"):
        assert section in data


def test_thresholds_values_are_sane():
    data = load_thresholds()
    assert 0.0 < data["accident"]["possible_incident_score"] < data["accident"]["accident_score"] <= 1.0
    assert data["accident"]["temporal_verification_frames"] > 0


def test_database_url_points_to_sqlite_by_default():
    assert settings.DATABASE_URL.startswith("sqlite:///")
