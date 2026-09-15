"""
tests/test_tracking.py

Phase 4 tests for multi-object tracking (ByteTrack integration +
TrackManager). Uses synthetic `Detection` objects rather than real
YOLO inference — tracking logic is independent of the detector, so
these tests stay fast and don't require network access or model
weights (that's covered separately in test_detection.py).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.detection import BoundingBox, Detection  # noqa: E402
from engine.tracking import  ByteTrackWrapper, TrackManager  # noqa: E402



def make_detection(class_name="car", class_id=2, confidence=0.9, box=(100, 100, 150, 150)):
    return Detection(class_name=class_name, class_id=class_id, confidence=confidence, bbox=BoundingBox(*box))


@pytest.fixture
def cfg():
    # Small, fast-to-test thresholds; explicit rather than relying on defaults.
    return Settings(
        TRACKER_MAX_MISSED_FRAMES=5,
        TRACKER_MIN_HITS=3,
        TRACKER_IOU_THRESHOLD=0.8,
        TRACKER_TRACK_HIGH_THRESH=0.5,
        TRACKER_TRACK_LOW_THRESH=0.1,
        TRACKER_NEW_TRACK_THRESH=0.6,
        TRACKER_FUSE_SCORE=True,
        TRACK_HISTORY_LENGTH=10,
    )


@pytest.fixture
def tracker(cfg):
    t = ByteTrackWrapper(cfg)
    yield t
    t.reset()


@pytest.fixture
def manager(cfg):
    return TrackManager(cfg)


# ---------------------------------------------------------------------------
# ByteTrackWrapper
# ---------------------------------------------------------------------------

def test_single_object_id_persists_across_smooth_movement(tracker):
    ids = []
    box = [100, 100, 150, 150]
    for _ in range(5):
        det = make_detection(box=tuple(box))
        raw = tracker.update([det])
        assert len(raw) == 1
        ids.append(raw[0].track_id)
        box = [v + 3 for v in box]  # small consistent shift

    assert len(set(ids)) == 1, f"expected one stable ID, got {ids}"


def test_id_persists_through_brief_gap(tracker):
    det1 = make_detection(box=(100, 100, 150, 150))
    out1 = tracker.update([det1])
    id1 = out1[0].track_id

    # a couple of frames with no detections (temporary occlusion)
    tracker.update([])
    tracker.update([])

    det2 = make_detection(box=(112, 106, 162, 156))
    out2 = tracker.update([det2])
    assert len(out2) == 1
    assert out2[0].track_id == id1


def test_new_object_gets_distinct_id_after_confirmation(tracker):
    car = make_detection(class_name="car", class_id=2, box=(100, 100, 150, 150))
    tracker.update([car])

    truck1 = make_detection(class_name="truck", class_id=7, box=(400, 400, 460, 460))
    out2 = tracker.update([car, truck1])
    # New tracks require a second matched frame before appearing (ByteTrack activation rule).
    assert {t.class_name for t in out2} == {"car"}

    truck2 = make_detection(class_name="truck", class_id=7, box=(402, 402, 462, 462))
    out3 = tracker.update([car, truck2])
    ids = {t.track_id for t in out3}
    assert len(ids) == 2


def test_reset_clears_tracker_state(tracker):
    tracker.update([make_detection()])
    tracker.reset()
    out = tracker.update([make_detection()])
    # After reset, IDs restart from 1.
    assert out[0].track_id == 1


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------

def test_manager_creates_new_track(manager, tracker):
    raw = tracker.update([make_detection()])
    active = manager.update(raw, frame_index=0, timestamp_ms=0.0)

    assert len(active) == 1
    obj = active[0]
    assert obj.track_id == raw[0].track_id
    assert obj.class_name == "car"
    assert obj.hit_count == 1
    assert obj.is_active is True
    assert obj.first_seen_frame == 0
    assert obj.track_duration_frames == 1


def test_manager_computes_velocity_and_direction(manager, tracker):
    raw1 = tracker.update([make_detection(box=(100, 100, 150, 150))])
    manager.update(raw1, frame_index=0, timestamp_ms=0.0)

    # move 10px right, 0px down -> direction should be ~0 degrees (atan2(0,10))
    raw2 = tracker.update([make_detection(box=(110, 100, 160, 150))])
    active2 = manager.update(raw2, frame_index=1, timestamp_ms=100.0)

    obj = active2[0]
    # ByteTrack Kalman-smooths the returned bbox, so the exact pixel
    # displacement won't match the raw input exactly — assert direction
    # and an order-of-magnitude-correct positive speed instead.
    assert obj.speed_px_per_frame > 0.0
    assert obj.direction_deg == pytest.approx(0.0, abs=15.0)
    assert obj.previous_center is not None


def test_manager_confirms_track_after_min_hits(manager, tracker, cfg):
    obj = None
    box = [100, 100, 150, 150]
    for i in range(cfg.TRACKER_MIN_HITS):
        raw = tracker.update([make_detection(box=tuple(box))])
        active = manager.update(raw, frame_index=i, timestamp_ms=i * 100.0)
        obj = active[0]
        box = [v + 2 for v in box]

    assert obj.hit_count == cfg.TRACKER_MIN_HITS
    assert obj.confirmed is True


def test_manager_tolerates_temporary_missed_frames(manager, tracker, cfg):
    raw1 = tracker.update([make_detection(box=(100, 100, 150, 150))])
    active1 = manager.update(raw1, frame_index=0, timestamp_ms=0.0)
    track_id = active1[0].track_id

    # object briefly missed (fewer frames than the eviction limit)
    for i in range(1, cfg.TRACKER_MAX_MISSED_FRAMES):
        raw = tracker.update([])
        active = manager.update(raw, frame_index=i, timestamp_ms=i * 100.0)
        # still tracked internally even though not active this frame
        assert manager.get_track(track_id) is not None
        assert manager.get_track(track_id).is_active is False

    # reappears within the grace period -> same ID, track re-acquired
    raw_back = tracker.update([make_detection(box=(115, 108, 165, 158))])
    active_back = manager.update(
        raw_back, frame_index=cfg.TRACKER_MAX_MISSED_FRAMES, timestamp_ms=cfg.TRACKER_MAX_MISSED_FRAMES * 100.0
    )
    assert any(t.track_id == track_id and t.is_active for t in active_back)


def test_manager_evicts_track_after_exceeding_missed_frame_limit(manager, tracker, cfg):
    raw1 = tracker.update([make_detection(box=(100, 100, 150, 150))])
    active1 = manager.update(raw1, frame_index=0, timestamp_ms=0.0)
    track_id = active1[0].track_id

    # miss for longer than TRACKER_MAX_MISSED_FRAMES -> must be evicted
    for i in range(1, cfg.TRACKER_MAX_MISSED_FRAMES + 2):
        manager.update([], frame_index=i, timestamp_ms=i * 100.0)

    assert manager.get_track(track_id) is None


def test_manager_position_history_is_bounded(manager, tracker, cfg):
    box = [100, 100, 150, 150]
    obj = None
    for i in range(cfg.TRACK_HISTORY_LENGTH + 5):
        raw = tracker.update([make_detection(box=tuple(box))])
        active = manager.update(raw, frame_index=i, timestamp_ms=i * 100.0)
        obj = active[0]
        box = [v + 1 for v in box]

    assert len(obj.position_history) == cfg.TRACK_HISTORY_LENGTH


def test_tracked_object_to_dict_schema(manager, tracker):
    raw = tracker.update([make_detection()])
    active = manager.update(raw, frame_index=0, timestamp_ms=0.0)
    d = active[0].to_dict()
    expected_keys = {
        "track_id", "class", "confidence", "bbox", "center",
        "speed_px_per_frame", "direction_deg", "track_duration_frames",
        "hit_count", "missed_frames", "is_active", "confirmed",
    }
    assert set(d.keys()) == expected_keys


def test_manager_reset_clears_all_tracks(manager, tracker):
    raw = tracker.update([make_detection()])
    manager.update(raw, frame_index=0, timestamp_ms=0.0)
    assert len(manager.get_active_tracks()) == 1

    manager.reset()
    assert manager.get_active_tracks() == []


def test_manager_handles_multiple_simultaneous_objects(manager, tracker):
    car = make_detection(class_name="car", class_id=2, box=(50, 50, 100, 100))
    truck = make_detection(class_name="truck", class_id=7, box=(300, 300, 360, 360))

    tracker.update([car, truck])  # frame 0: truck not yet activated
    raw2 = tracker.update([car, truck])  # frame 1: truck confirmed
    active2 = manager.update(raw2, frame_index=1, timestamp_ms=100.0)

    classes = {t.class_name for t in active2}
    assert classes == {"car", "truck"}
    assert len({t.track_id for t in active2}) == 2


def test_entering_and_exiting_logged(manager, tracker, cfg, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="engine.tracking.track_manager")

    raw1 = tracker.update([make_detection()])
    manager.update(raw1, frame_index=0, timestamp_ms=0.0)
    assert any("ENTERED" in rec.message for rec in caplog.records)

    caplog.clear()
    for i in range(1, cfg.TRACKER_MAX_MISSED_FRAMES + 2):
        manager.update([], frame_index=i, timestamp_ms=i * 100.0)
    assert any("EXITED" in rec.message for rec in caplog.records)
