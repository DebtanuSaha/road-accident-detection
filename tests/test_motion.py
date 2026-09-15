"""
tests/test_motion.py

Phase 5 tests for movement and trajectory analysis. Uses synthetic
`PositionSample`/`TrackedObject` data directly — this module operates
purely on tracking history, so no detector/tracker needs to run for
most tests; a couple of integration tests chain the real
ByteTrackWrapper + TrackManager to confirm the full pipeline connects.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.detection import BoundingBox, Detection  # noqa: E402
from engine.tracking import ByteTrackWrapper, PositionSample, TrackedObject, TrackManager  # noqa: E402
from engine.motion import MotionAnalyzer  # noqa: E402
from engine.motion.trajectory import (  # noqa: E402
    angular_difference_deg,
    compute_direction_deg,
    compute_displacement,
    euclidean_distance,
)
from engine.motion.velocity import (  # noqa: E402
    compute_speed_change,
    compute_speed_px_per_frame,
    compute_speed_px_per_sec,
)


def sample(frame_index, timestamp_ms, x, y):
    return PositionSample(frame_index=frame_index, timestamp_ms=timestamp_ms, center=(x, y))


def make_track(history_points, cfg=None):
    """Build a minimal TrackedObject with a given position_history for direct MotionAnalyzer testing."""
    history = [sample(*p) for p in history_points]
    last = history[-1]
    return TrackedObject(
        track_id=1,
        class_name="car",
        class_id=2,
        bbox=BoundingBox(last.center[0] - 25, last.center[1] - 25, last.center[0] + 25, last.center[1] + 25),
        confidence=0.9,
        first_seen_frame=history[0].frame_index,
        last_seen_frame=last.frame_index,
        first_seen_timestamp_ms=history[0].timestamp_ms,
        last_seen_timestamp_ms=last.timestamp_ms,
        position_history=__import__("collections").deque(history),
    )


@pytest.fixture
def cfg():
    return Settings(TRACK_HISTORY_LENGTH=30)


@pytest.fixture
def analyzer(cfg):
    return MotionAnalyzer(cfg)


# ---------------------------------------------------------------------------
# trajectory.py — pure geometry
# ---------------------------------------------------------------------------

def test_euclidean_distance():
    assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_compute_displacement():
    a = sample(0, 0.0, 0, 0)
    b = sample(1, 100.0, 3, 4)
    assert compute_displacement(a, b) == pytest.approx(5.0)


def test_compute_direction_deg_right():
    a = sample(0, 0.0, 0, 0)
    b = sample(1, 100.0, 10, 0)
    assert compute_direction_deg(a, b) == pytest.approx(0.0)


def test_compute_direction_deg_down():
    a = sample(0, 0.0, 0, 0)
    b = sample(1, 100.0, 0, 10)
    assert compute_direction_deg(a, b) == pytest.approx(90.0)


def test_compute_direction_deg_none_for_zero_displacement():
    a = sample(0, 0.0, 5, 5)
    b = sample(1, 100.0, 5, 5)
    assert compute_direction_deg(a, b) is None


def test_angular_difference_wraparound():
    # 350 degrees vs 10 degrees should be a 20-degree difference, not 340.
    assert angular_difference_deg(350.0, 10.0) == pytest.approx(20.0)


def test_angular_difference_none_when_either_missing():
    assert angular_difference_deg(None, 10.0) is None
    assert angular_difference_deg(10.0, None) is None


# ---------------------------------------------------------------------------
# velocity.py — pixel-space speed
# ---------------------------------------------------------------------------

def test_speed_px_per_frame_normalizes_by_frame_gap():
    a = sample(0, 0.0, 0, 0)
    b = sample(2, 200.0, 20, 0)  # 2 frames elapsed, 20px moved
    assert compute_speed_px_per_frame(a, b) == pytest.approx(10.0)


def test_speed_px_per_frame_zero_when_no_frame_progress():
    a = sample(5, 0.0, 0, 0)
    b = sample(5, 0.0, 20, 0)
    assert compute_speed_px_per_frame(a, b) == 0.0


def test_speed_px_per_sec_uses_timestamps():
    a = sample(0, 0.0, 0, 0)
    b = sample(1, 500.0, 10, 0)  # 0.5 sec elapsed, 10px moved
    assert compute_speed_px_per_sec(a, b) == pytest.approx(20.0)


def test_speed_px_per_sec_none_when_timestamps_unreliable():
    a = sample(0, 0.0, 0, 0)
    b = sample(1, 0.0, 10, 0)  # both timestamps 0 (unknown FPS source)
    assert compute_speed_px_per_sec(a, b) is None


def test_speed_change_none_without_previous():
    assert compute_speed_change(None, 5.0) is None


def test_speed_change_sign():
    assert compute_speed_change(10.0, 4.0) == pytest.approx(-6.0)  # decelerating
    assert compute_speed_change(4.0, 10.0) == pytest.approx(6.0)   # accelerating


# ---------------------------------------------------------------------------
# motion_analyzer.py — combined, threshold-driven profile
# ---------------------------------------------------------------------------

def test_analyze_returns_none_with_insufficient_history(analyzer):
    track = make_track([(0, 0.0, 100, 100)])  # only one sample
    assert analyzer.analyze(track) is None


def test_analyze_basic_movement(analyzer):
    track = make_track([(0, 0.0, 100, 100), (1, 100.0, 110, 100)])
    state = analyzer.analyze(track)
    assert state is not None
    assert state.displacement_px == pytest.approx(10.0)
    assert state.direction_deg == pytest.approx(0.0)
    assert state.speed_px_per_frame == pytest.approx(10.0)
    assert state.is_stationary is False
    assert state.previous_speed_px_per_frame is None  # only 2 samples so far


def test_analyze_detects_sudden_deceleration(analyzer):
    track = make_track([
        (0, 0.0, 100, 100),
        (1, 100.0, 120, 100),  # speed 20
        (2, 200.0, 126, 100),  # speed 6 -> 70% drop, exceeds 0.5 ratio default
    ])
    state = analyzer.analyze(track)
    assert state.previous_speed_px_per_frame == pytest.approx(20.0)
    assert state.speed_change_px_per_frame == pytest.approx(-14.0)
    assert state.is_sudden_deceleration is True


def test_analyze_detects_sudden_stop(analyzer):
    track = make_track([
        (0, 0.0, 100, 100),
        (1, 100.0, 120, 100),  # moving at speed 20
        (2, 200.0, 120, 100),  # now stationary
    ])
    state = analyzer.analyze(track)
    assert state.is_stationary is True
    assert state.is_sudden_stop is True


def test_analyze_does_not_flag_sudden_stop_if_already_stationary(analyzer):
    track = make_track([
        (0, 0.0, 100, 100),
        (1, 100.0, 100, 100),  # already stationary
        (2, 200.0, 100, 100),  # still stationary
    ])
    state = analyzer.analyze(track)
    assert state.is_stationary is True
    assert state.is_sudden_stop is False  # wasn't moving before, so this isn't a "sudden" stop


def test_analyze_detects_sudden_direction_change(analyzer):
    track = make_track([
        (0, 0.0, 100, 100),
        (1, 100.0, 120, 100),  # moving right (0 deg)
        (2, 200.0, 140, 100),  # still right (0 deg) -> direction_change 0
        (3, 300.0, 140, 120),  # now moving down (90 deg) -> 90 deg change
    ])
    state = analyzer.analyze(track)
    assert state.direction_change_deg == pytest.approx(90.0)
    assert state.is_sudden_direction_change is True


def test_analyze_no_direction_change_flag_for_straight_line(analyzer):
    track = make_track([
        (0, 0.0, 100, 100),
        (1, 100.0, 110, 100),
        (2, 200.0, 120, 100),
    ])
    state = analyzer.analyze(track)
    assert state.direction_change_deg == pytest.approx(0.0)
    assert state.is_sudden_direction_change is False


def test_analyze_many_skips_tracks_with_insufficient_history(analyzer):
    t1 = make_track([(0, 0.0, 100, 100), (1, 100.0, 110, 100)])
    t1.track_id = 1
    t2 = make_track([(0, 0.0, 200, 200)])
    t2.track_id = 2
    results = analyzer.analyze_many([t1, t2])
    assert set(results.keys()) == {1}


def test_analyzer_window_is_clamped_to_history_length():
    cfg = Settings(TRACK_HISTORY_LENGTH=5)
    analyzer = MotionAnalyzer(cfg)
    assert analyzer._window <= 5


def test_to_dict_schema(analyzer):
    track = make_track([(0, 0.0, 100, 100), (1, 100.0, 110, 100)])
    state = analyzer.analyze(track)
    d = state.to_dict()
    expected_keys = {
        "track_id", "frame_index", "center", "displacement_px", "direction_deg",
        "speed_px_per_frame", "speed_px_per_sec", "speed_change_px_per_frame",
        "direction_change_deg", "is_stationary", "is_sudden_deceleration",
        "is_sudden_direction_change", "is_sudden_stop", "sample_count",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Integration: real ByteTrackWrapper + TrackManager feeding MotionAnalyzer
# ---------------------------------------------------------------------------

def test_end_to_end_with_real_tracker(cfg):
    tracker = ByteTrackWrapper(cfg)
    manager = TrackManager(cfg)
    analyzer = MotionAnalyzer(cfg)

    boxes = [(100, 100, 150, 150), (110, 100, 160, 150), (120, 100, 170, 150)]
    last_state = None
    for i, box in enumerate(boxes):
        det = Detection(class_name="car", class_id=2, confidence=0.9, bbox=BoundingBox(*box))
        raw = tracker.update([det])
        active = manager.update(raw, frame_index=i, timestamp_ms=i * 100.0)
        for obj in active:
            state = analyzer.analyze(obj)
            if state is not None:
                last_state = state

    assert last_state is not None
    assert last_state.speed_px_per_frame > 0
    assert last_state.direction_deg == pytest.approx(0.0, abs=5.0)
