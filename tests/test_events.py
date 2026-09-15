"""
tests/test_events.py

Phase 8 tests for accident event generation: the AccidentEvent Pydantic
schema and EventBuilder's exactly-once-per-confirmed-pair behavior.
"""

import sys
from collections import deque
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.detection import BoundingBox  # noqa: E402
from engine.tracking import PositionSample, TrackedObject  # noqa: E402
from engine.accident import AccidentAssessment, AccidentEvidence, AccidentState  # noqa: E402
from engine.events import (  # noqa: E402
    AccidentEvent,
    AccidentEventStatus,
    AccidentSeverity,
    EventBuilder,
    Location,
    ObjectInvolved,
)


def make_assessment(track_id_a=12, track_id_b=7, class_a="car", class_b="motorcycle",
                     confirmed=True, probability=0.91, frame_index=100):
    return AccidentAssessment(
        track_id_a=track_id_a,
        track_id_b=track_id_b,
        class_a=class_a,
        class_b=class_b,
        frame_index=frame_index,
        timestamp_ms=frame_index * 100.0,
        evidence=AccidentEvidence(0.9, 0.9, 0.9),
        accident_probability=probability,
        instantaneous_state=AccidentState.ACCIDENT,
        consecutive_qualifying_frames=15,
        confirmed=confirmed,
        confirmed_at_frame=frame_index if confirmed else None,
    )


def make_tracked_object(track_id, class_name, box=(0, 0, 50, 50)):
    bbox = BoundingBox(*box)
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        class_id=2,
        bbox=bbox,
        confidence=0.9,
        first_seen_frame=0,
        last_seen_frame=0,
        first_seen_timestamp_ms=0.0,
        last_seen_timestamp_ms=0.0,
        position_history=deque([PositionSample(0, 0.0, bbox.center)]),
    )


@pytest.fixture
def cfg():
    return Settings(DEFAULT_CAMERA_ID="CAM-042")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_object_involved_serializes_with_class_alias():
    obj = ObjectInvolved(track_id=12, class_name="car")
    dumped = obj.model_dump(by_alias=True)
    assert dumped == {"track_id": 12, "class": "car"}


def test_object_involved_accepts_alias_on_input():
    obj = ObjectInvolved(track_id=12, **{"class": "car"})
    assert obj.class_name == "car"


def test_accident_event_matches_spec_example_shape():
    event = AccidentEvent(
        event_id="ACC-000001",
        accident_probability=0.91,
        objects_involved=[
            ObjectInvolved(track_id=12, class_name="car"),
            ObjectInvolved(track_id=7, class_name="motorcycle"),
        ],
        camera_id="CAM-001",
    )
    d = event.model_dump(mode="json", by_alias=True)
    assert set(d.keys()) == {
        "event_id", "timestamp", "accident_probability", "severity",
        "objects_involved", "location", "camera_id", "evidence_image",
        "evidence_video", "status",
    }
    assert d["severity"] == "unknown"
    assert d["status"] == "detected"
    assert d["location"] is None
    assert d["evidence_image"] is None
    assert d["evidence_video"] is None
    assert d["objects_involved"][0] == {"track_id": 12, "class": "car"}


def test_accident_probability_must_be_in_unit_range():
    with pytest.raises(ValidationError):
        AccidentEvent(
            event_id="ACC-000001", accident_probability=1.5,
            objects_involved=[], camera_id="CAM-001",
        )


def test_severity_defaults_to_unknown():
    event = AccidentEvent(
        event_id="ACC-000001", accident_probability=0.9,
        objects_involved=[], camera_id="CAM-001",
    )
    assert event.severity == AccidentSeverity.UNKNOWN


def test_status_defaults_to_detected():
    event = AccidentEvent(
        event_id="ACC-000001", accident_probability=0.9,
        objects_involved=[], camera_id="CAM-001",
    )
    assert event.status == AccidentEventStatus.DETECTED


def test_location_shape_when_provided():
    loc = Location(latitude=22.5, longitude=88.3, accuracy_m=8.5)
    event = AccidentEvent(
        event_id="ACC-000001", accident_probability=0.9,
        objects_involved=[], camera_id="CAM-001", location=loc,
    )
    assert event.location.latitude == 22.5
    assert event.location.accuracy_m == 8.5


# ---------------------------------------------------------------------------
# EventBuilder
# ---------------------------------------------------------------------------

def test_build_if_new_returns_none_when_not_confirmed(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(confirmed=False)
    assert builder.build_if_new(assessment, []) is None


def test_build_if_new_creates_event_on_first_confirmation(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(confirmed=True, probability=0.91)
    event = builder.build_if_new(assessment, [])
    assert event is not None
    assert event.event_id == "ACC-000001"
    assert event.accident_probability == pytest.approx(0.91)
    assert event.camera_id == "CAM-042"


def test_build_if_new_does_not_duplicate_same_pair(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(confirmed=True)
    first = builder.build_if_new(assessment, [])
    second = builder.build_if_new(assessment, [])
    assert first is not None
    assert second is None


def test_build_if_new_is_pair_order_independent(cfg):
    builder = EventBuilder(cfg)
    a = make_assessment(track_id_a=7, track_id_b=12, confirmed=True)
    b = make_assessment(track_id_a=12, track_id_b=7, confirmed=True)
    first = builder.build_if_new(a, [])
    second = builder.build_if_new(b, [])
    assert first is not None
    assert second is None  # same physical pair, regardless of a/b order


def test_build_if_new_generates_sequential_event_ids(cfg):
    builder = EventBuilder(cfg)
    e1 = builder.build_if_new(make_assessment(track_id_a=1, track_id_b=2, confirmed=True), [])
    e2 = builder.build_if_new(make_assessment(track_id_a=3, track_id_b=4, confirmed=True), [])
    assert e1.event_id == "ACC-000001"
    assert e2.event_id == "ACC-000002"


def test_build_uses_live_class_name_from_tracked_objects_when_available(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(track_id_a=12, track_id_b=7, class_a="car", class_b="motorcycle", confirmed=True)
    # Tracker's current class_name should win over the (possibly stale) assessment fallback.
    tracked = [
        make_tracked_object(12, "truck"),
        make_tracked_object(7, "bicycle"),
    ]
    event = builder.build_if_new(assessment, tracked)
    classes = {o.track_id: o.class_name for o in event.objects_involved}
    assert classes[12] == "truck"
    assert classes[7] == "bicycle"


def test_build_falls_back_to_assessment_class_when_object_missing(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(track_id_a=12, track_id_b=7, class_a="car", class_b="motorcycle", confirmed=True)
    event = builder.build_if_new(assessment, [])  # no tracked objects supplied
    classes = {o.track_id: o.class_name for o in event.objects_involved}
    assert classes[12] == "car"
    assert classes[7] == "motorcycle"


def test_reset_allows_rebuilding_for_same_pair(cfg):
    builder = EventBuilder(cfg)
    assessment = make_assessment(confirmed=True)
    first = builder.build_if_new(assessment, [])
    builder.reset()
    second = builder.build_if_new(assessment, [])
    assert first is not None
    assert second is not None
    assert first.event_id != second.event_id  # sequence is not reset, only de-dup state


def test_two_independent_pairs_both_produce_events(cfg):
    builder = EventBuilder(cfg)
    e1 = builder.build_if_new(make_assessment(track_id_a=1, track_id_b=2, confirmed=True), [])
    e2 = builder.build_if_new(make_assessment(track_id_a=3, track_id_b=4, confirmed=True), [])
    assert e1 is not None and e2 is not None
    assert e1.event_id != e2.event_id
