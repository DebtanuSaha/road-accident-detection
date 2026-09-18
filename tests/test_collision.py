"""
tests/test_collision.py

Phase 6 tests for collision detection: spatial checks
(collision_detector.py) and composite, multi-signal scoring
(collision_scorer.py).

Most tests build `TrackedObject` / `MotionState` instances directly so
assertions can be exact (no Kalman-filter smoothing noise from a real
tracker). One integration test chains the real ByteTrackWrapper +
TrackManager + MotionAnalyzer to confirm the full pipeline connects
correctly end-to-end.
"""

import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.detection import BoundingBox, Detection  # noqa: E402
from engine.tracking import ByteTrackWrapper, PositionSample, TrackedObject, TrackManager  # noqa: E402
from engine.motion import MotionAnalyzer, MotionState  # noqa: E402
from engine.collision import CollisionDetector, CollisionScorer, compute_center_distance, compute_iou  # noqa: E402


def make_tracked_object(track_id, box, class_name="car"):
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
        is_active=True,
        position_history=deque([PositionSample(0, 0.0, bbox.center)]),
    )


def make_motion_state(
    track_id,
    direction_change_deg=None,
    speed_change_px_per_frame=None,
    previous_speed_px_per_frame=None,
    is_stationary=False,
    speed_px_per_frame=0.0,
):
    return MotionState(
        track_id=track_id,
        frame_index=0,
        center=(0.0, 0.0),
        displacement_px=0.0,
        direction_deg=0.0,
        speed_px_per_frame=speed_px_per_frame,
        speed_px_per_sec=None,
        previous_speed_px_per_frame=previous_speed_px_per_frame,
        speed_change_px_per_frame=speed_change_px_per_frame,
        direction_change_deg=direction_change_deg,
        is_stationary=is_stationary,
        is_sudden_deceleration=False,
        is_sudden_direction_change=False,
        is_sudden_stop=False,
        sample_count=3,
    )


@pytest.fixture
def cfg():
    return Settings()


# ---------------------------------------------------------------------------
# collision_detector.py — pure geometry
# ---------------------------------------------------------------------------

def test_compute_iou_identical_boxes():
    box = BoundingBox(0, 0, 100, 100)
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_compute_iou_no_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(100, 100, 110, 110)
    assert compute_iou(a, b) == 0.0


def test_compute_iou_partial_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    # intersection = 5x5=25, union = 100+100-25=175
    assert compute_iou(a, b) == pytest.approx(25 / 175)


def test_compute_center_distance():
    a = BoundingBox(0, 0, 10, 10)   # center (5,5)
    b = BoundingBox(30, 40, 40, 50)  # center (35,45)
    assert compute_center_distance(a, b) == pytest.approx(50.0)


def test_detector_finds_overlapping_pair(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    interactions = detector.find_interactions([a, b])
    assert len(interactions) == 1
    assert interactions[0].is_overlapping is True
    assert interactions[0].is_interacting is True


def test_detector_finds_close_but_non_overlapping_pair(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 20, 20))
    b = make_tracked_object(2, (25, 0, 45, 20))  # near, not overlapping
    interactions = detector.find_interactions([a, b])
    assert len(interactions) == 1
    assert interactions[0].is_close is True
    assert interactions[0].is_overlapping is False


def test_detector_ignores_distant_pair(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 20, 20))
    b = make_tracked_object(2, (500, 500, 520, 520))
    assert detector.find_interactions([a, b]) == []


def test_detector_ignores_inactive_objects(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (10, 10, 110, 110))
    b.is_active = False
    assert detector.find_interactions([a, b]) == []


def test_detector_handles_three_objects_pairwise(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))  # overlaps a
    c = make_tracked_object(3, (1000, 1000, 1020, 1020))  # isolated
    interactions = detector.find_interactions([a, b, c])
    pairs = {(i.track_id_a, i.track_id_b) for i in interactions}
    assert pairs == {(1, 2)}


def test_spatial_interaction_to_dict_schema(cfg):
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    d = detector.find_interactions([a, b])[0].to_dict()
    assert set(d.keys()) == {
        "pair", "iou", "center_distance_px", "is_overlapping", "is_sustained_contact",
        "is_close", "spatial_score",
    }


# ---------------------------------------------------------------------------
# collision_scorer.py — composite scoring
# ---------------------------------------------------------------------------

def test_scorer_pure_overlap_no_motion_signal_gives_moderate_score(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scores = scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)

    assert len(scores) == 1
    s = scores[0]
    # No motion states supplied -> trajectory/motion/stopping components are all 0;
    # only the spatial_interaction weight (0.35 by default) contributes.
    assert s.trajectory_change_score == 0.0
    assert s.motion_change_score == 0.0
    assert s.post_interaction_stopping_score == 0.0
    assert s.composite_score == pytest.approx(cfg_weight(cfg, "spatial_interaction") * s.spatial_interaction_score)


def cfg_weight(cfg, name):
    from config.threshold_loader import load_thresholds
    return load_thresholds()["collision"]["weights"][name]


def test_scorer_combines_all_four_signals_for_high_score(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (20, 20, 120, 120))  # strong overlap

    motion_states = {
        1: make_motion_state(1, direction_change_deg=90.0, speed_change_px_per_frame=-10.0,
                              previous_speed_px_per_frame=10.0, is_stationary=True),
        2: make_motion_state(2, direction_change_deg=90.0, speed_change_px_per_frame=-10.0,
                              previous_speed_px_per_frame=10.0, is_stationary=True),
    }
    scores = scorer.update([a, b], motion_states, frame_index=0, timestamp_ms=0.0)

    assert len(scores) == 1
    s = scores[0]
    assert s.trajectory_change_score == pytest.approx(1.0)
    assert s.motion_change_score == pytest.approx(1.0)
    assert s.post_interaction_stopping_score == pytest.approx(1.0)
    # All four signals maxed -> composite should be high (close to sum of weights).
    assert s.composite_score > 0.8


def test_scorer_ignores_non_interacting_pairs(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 20, 20))
    b = make_tracked_object(2, (1000, 1000, 1020, 1020))
    scores = scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)
    assert scores == []


def test_scorer_keeps_scoring_pair_within_post_interaction_window(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)

    # Objects separate (no longer overlapping/close) on the next frame.
    a2 = make_tracked_object(1, (0, 0, 20, 20))
    b2 = make_tracked_object(2, (1000, 1000, 1020, 1020))
    scores = scorer.update([a2, b2], {}, frame_index=1, timestamp_ms=100.0)

    assert len(scores) == 1
    assert scores[0].is_active_spatial_contact is False
    assert scores[0].in_post_interaction_window is True
    assert scores[0].spatial_interaction_score > 0.0  # decayed credit from peak, not zero


def test_scorer_evicts_pair_after_window_expires(cfg):
    small_window_cfg = Settings()
    scorer = CollisionScorer(small_window_cfg)
    # Manually shrink the window for a fast test.
    scorer._post_interaction_window_frames = 2

    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)

    a2 = make_tracked_object(1, (0, 0, 20, 20))
    b2 = make_tracked_object(2, (1000, 1000, 1020, 1020))
    # Still within window (frames 1, 2)
    assert len(scorer.update([a2, b2], {}, frame_index=1, timestamp_ms=100.0)) == 1
    assert len(scorer.update([a2, b2], {}, frame_index=2, timestamp_ms=200.0)) == 1
    # Past the window now.
    assert len(scorer.update([a2, b2], {}, frame_index=3, timestamp_ms=300.0)) == 0


def test_scorer_reset_clears_pair_windows(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)
    assert len(scorer._pair_windows) == 1

    scorer.reset()
    assert len(scorer._pair_windows) == 0


def test_scorer_skips_pair_when_one_object_no_longer_tracked(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)

    # Object 2 vanished entirely (not even passed in as inactive).
    a2 = make_tracked_object(1, (0, 0, 20, 20))
    scores = scorer.update([a2], {}, frame_index=1, timestamp_ms=100.0)
    assert scores == []


def test_collision_score_to_dict_schema(cfg):
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    scores = scorer.update([a, b], {}, frame_index=0, timestamp_ms=0.0)
    d = scores[0].to_dict()
    expected_keys = {
        "pair", "classes", "frame_index", "spatial_interaction_score",
        "trajectory_change_score", "motion_change_score", "post_interaction_stopping_score",
        "composite_score", "is_active_spatial_contact", "is_overlapping",
        "is_sustained_contact", "overlap_streak_frames", "closing_speed_px_per_frame",
        "peak_closing_speed_px_per_frame", "in_post_interaction_window",
    }
    assert set(d.keys()) == expected_keys


def test_overlap_streak_increments_across_consecutive_frames(cfg):
    scorer = CollisionScorer(cfg)
    last = None
    for i in range(4):
        a = make_tracked_object(1, (0, 0, 100, 100))
        b = make_tracked_object(2, (30, 30, 130, 130))  # sustained real overlap
        scores = scorer.update([a, b], {}, frame_index=i, timestamp_ms=i * 100.0)
        last = scores[0]
    assert last.overlap_streak_frames == 4


def test_overlap_streak_resets_when_overlap_stops(cfg):
    scorer = CollisionScorer(cfg)
    for i in range(3):
        scorer.update(
            [make_tracked_object(1, (0, 0, 100, 100)), make_tracked_object(2, (30, 30, 130, 130))],
            {}, frame_index=i, timestamp_ms=i * 100.0,
        )
    # Now close but NOT overlapping -> streak must reset to 0.
    scores = scorer.update(
        [make_tracked_object(1, (0, 0, 20, 20)), make_tracked_object(2, (25, 0, 45, 20))],
        {}, frame_index=3, timestamp_ms=300.0,
    )
    assert scores[0].is_overlapping is False
    assert scores[0].overlap_streak_frames == 0


def test_closing_speed_is_positive_when_pair_approaches(cfg):
    scorer = CollisionScorer(cfg)
    # Frame 0: 35px apart (within min_center_distance_px=40, so registered as
    # "close"). Then closing 20px per frame. The reported value is
    # EXPONENTIALLY SMOOTHED (see _CLOSING_SPEED_EMA_ALPHA) rather than a raw
    # 2-frame derivative, so it rises toward 20 over several frames rather
    # than jumping straight to it — assert it's clearly positive and trending
    # up, not an exact single-frame value.
    a0 = make_tracked_object(1, (0, 0, 20, 20))     # center (10,10)
    b0 = make_tracked_object(2, (35, 0, 55, 20))    # center (45,10) -> distance 35
    scorer.update([a0, b0], {}, frame_index=0, timestamp_ms=0.0)

    a1 = make_tracked_object(1, (0, 0, 20, 20))
    b1 = make_tracked_object(2, (15, 0, 35, 20))    # center (25,10) -> distance 15
    first = scorer.update([a1, b1], {}, frame_index=1, timestamp_ms=100.0)[0]

    assert first.closing_speed_px_per_frame > 0.0
    assert first.closing_speed_px_per_frame < 20.0  # smoothed, hasn't reached the raw rate yet


def test_closing_speed_is_near_zero_for_parallel_travel(cfg):
    """
    Regression test for the real-footage false positive: two vehicles
    traveling in parallel lanes at a constant distance (never actually
    overlapping) should show ~zero closing speed every frame, however
    long they stay close together.
    """
    scorer = CollisionScorer(cfg)
    distance_px = 35  # within min_center_distance_px (40) -> "close" every frame
    last_score = None
    for i in range(10):
        a = make_tracked_object(1, (i * 10, 0, i * 10 + 20, 20))
        b = make_tracked_object(2, (i * 10 + distance_px, 0, i * 10 + distance_px + 20, 20))
        scores = scorer.update([a, b], {}, frame_index=i, timestamp_ms=i * 100.0)
        if scores:
            last_score = scores[0]

    assert last_score is not None
    assert last_score.closing_speed_px_per_frame == pytest.approx(0.0, abs=0.5)
    assert last_score.is_overlapping is False


# ---------------------------------------------------------------------------
# Integration: real tracker + motion analyzer feeding the collision scorer
# ---------------------------------------------------------------------------

def test_end_to_end_converging_objects_produce_rising_collision_score(cfg):
    tracker = ByteTrackWrapper(cfg)
    manager = TrackManager(cfg)
    analyzer = MotionAnalyzer(cfg)
    scorer = CollisionScorer(cfg)

    # Two cars converge, meet, then both stop.
    car_a_boxes = [(50, 100, 100, 150), (70, 100, 120, 150), (90, 100, 140, 150),
                   (105, 100, 155, 150), (105, 100, 155, 150), (105, 100, 155, 150)]
    car_b_boxes = [(250, 100, 300, 150), (225, 100, 275, 150), (195, 100, 245, 150),
                   (150, 100, 200, 150), (150, 100, 200, 150), (150, 100, 200, 150)]

    composite_scores = []
    for i in range(len(car_a_boxes)):
        det_a = Detection(class_name="car", class_id=2, confidence=0.9, bbox=BoundingBox(*car_a_boxes[i]))
        det_b = Detection(class_name="car", class_id=2, confidence=0.9, bbox=BoundingBox(*car_b_boxes[i]))
        raw = tracker.update([det_a, det_b])
        active = manager.update(raw, frame_index=i, timestamp_ms=i * 100.0)
        motion_states = analyzer.analyze_many(active)
        scores = scorer.update(active, motion_states, frame_index=i, timestamp_ms=i * 100.0)
        for s in scores:
            composite_scores.append(s.composite_score)

    assert len(composite_scores) > 0, "expected at least one collision score once the cars meet"
    assert max(composite_scores) > 0.15, "composite score should rise as the cars collide and stop"


def test_hysteresis_keeps_settled_pair_interacting(cfg):
    """
    Regression test for the false NEGATIVE found by real-footage
    debugging: after two objects genuinely collide, they settle just
    outside the strict entry thresholds (measured at IoU ~0.13 /
    distance ~52px on real detections). Without asymmetric entry/exit
    thresholds the pair stops being scored entirely at that point,
    cutting off its confirmation streak partway.
    """
    detector = CollisionDetector(cfg)
    # Frame 0: strong overlap -> pair enters the interacting set.
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))
    assert len(detector.find_interactions([a, b])) == 1

    # Frame 1: settled apart — below the strict entry IoU (0.15) and
    # beyond the strict entry distance (40px), but still within the
    # looser sustain bounds. Must REMAIN an interaction.
    a2 = make_tracked_object(1, (0, 0, 100, 100))
    b2 = make_tracked_object(2, (55, 55, 155, 155))
    interactions = detector.find_interactions([a2, b2])
    assert len(interactions) == 1
    assert interactions[0].is_sustained_contact is True


def test_fresh_pair_is_not_marked_sustained_contact(cfg):
    """A pair that merely drifts near each other (never genuinely entered
    contact) must not be credited with sustained-contact status."""
    detector = CollisionDetector(cfg)
    a = make_tracked_object(1, (0, 0, 20, 20))
    b = make_tracked_object(2, (25, 0, 45, 20))  # close, not overlapping
    interactions = detector.find_interactions([a, b])
    assert len(interactions) == 1
    assert interactions[0].is_sustained_contact is False


def test_pair_is_dropped_once_fully_separated(cfg):
    """Hysteresis must not keep a pair alive forever — once it's beyond
    even the loose sustain bounds, the interaction ends."""
    detector = CollisionDetector(cfg)
    detector.find_interactions([
        make_tracked_object(1, (0, 0, 100, 100)),
        make_tracked_object(2, (30, 30, 130, 130)),
    ])
    far = detector.find_interactions([
        make_tracked_object(1, (0, 0, 100, 100)),
        make_tracked_object(2, (900, 900, 1000, 1000)),
    ])
    assert far == []


# ---------------------------------------------------------------------------
# Mutual corroboration (real-footage regression: a vehicle turning near an
# undisturbed neighbor was being flagged as a collision)
# ---------------------------------------------------------------------------

def test_mutual_corroboration_blend_one_sided_is_discounted():
    from engine.collision.collision_scorer import _mutual_corroboration_blend
    assert _mutual_corroboration_blend(1.0, 0.0) == pytest.approx(0.5)


def test_mutual_corroboration_blend_fully_mutual_is_unchanged():
    from engine.collision.collision_scorer import _mutual_corroboration_blend
    assert _mutual_corroboration_blend(1.0, 1.0) == pytest.approx(1.0)


def test_mutual_corroboration_blend_partial():
    from engine.collision.collision_scorer import _mutual_corroboration_blend
    assert _mutual_corroboration_blend(1.0, 0.5) == pytest.approx(0.75)


def test_mutual_corroboration_blend_both_zero():
    from engine.collision.collision_scorer import _mutual_corroboration_blend
    assert _mutual_corroboration_blend(0.0, 0.0) == 0.0


def test_trajectory_change_score_discounts_one_sided_turn(cfg):
    """
    Regression test for the exact real-footage false positive: one
    object turning sharply (large direction_change_deg) right next to a
    completely undisturbed neighbor must score well below what a
    genuinely mutual disturbance would.
    """
    scorer = CollisionScorer(cfg)
    turning = make_motion_state(1, direction_change_deg=180.0)  # maximal turn ratio
    undisturbed = make_motion_state(2, direction_change_deg=0.0)
    one_sided = scorer._trajectory_change_score(turning, undisturbed)

    both_turning = make_motion_state(2, direction_change_deg=180.0)
    mutual = scorer._trajectory_change_score(turning, both_turning)

    assert one_sided < mutual
    assert one_sided == pytest.approx(0.5)  # fully one-sided -> discounted to 50%
    assert mutual == pytest.approx(1.0)     # fully mutual -> full strength


def test_motion_change_score_discounts_one_sided_deceleration(cfg):
    scorer = CollisionScorer(cfg)
    decelerating = make_motion_state(1, speed_change_px_per_frame=-20.0, previous_speed_px_per_frame=20.0)
    steady = make_motion_state(2, speed_change_px_per_frame=0.0, previous_speed_px_per_frame=20.0)
    one_sided = scorer._motion_change_score(decelerating, steady)
    assert one_sided == pytest.approx(0.5)


def make_motion_state(track_id, direction_change_deg=None, speed_change_px_per_frame=None,
                       previous_speed_px_per_frame=None, is_stationary=False):
    from engine.motion import MotionState
    return MotionState(
        track_id=track_id, frame_index=0, center=(0.0, 0.0), displacement_px=0.0,
        direction_deg=0.0, speed_px_per_frame=0.0, speed_px_per_sec=None,
        previous_speed_px_per_frame=previous_speed_px_per_frame,
        speed_change_px_per_frame=speed_change_px_per_frame,
        direction_change_deg=direction_change_deg, is_stationary=is_stationary,
        is_sudden_deceleration=False, is_sudden_direction_change=False,
        is_sudden_stop=False, sample_count=3,
    )


def test_end_to_end_one_sided_turn_near_undisturbed_neighbor_scores_lower_than_mutual(cfg):
    """
    Full CollisionScorer integration version of the same regression:
    feeds real MotionStates through update() for a pair where only one
    object is disturbed, and confirms the resulting composite score is
    meaningfully lower than the same scenario with both objects disturbed.
    """
    scorer = CollisionScorer(cfg)
    a = make_tracked_object(1, (0, 0, 100, 100))
    b = make_tracked_object(2, (30, 30, 130, 130))

    one_sided_states = {
        1: make_motion_state(1, direction_change_deg=180.0, speed_change_px_per_frame=-15.0, previous_speed_px_per_frame=15.0),
        2: make_motion_state(2, direction_change_deg=0.0, speed_change_px_per_frame=0.0, previous_speed_px_per_frame=15.0),
    }
    mutual_states = {
        1: one_sided_states[1],
        2: make_motion_state(2, direction_change_deg=180.0, speed_change_px_per_frame=-15.0, previous_speed_px_per_frame=15.0),
    }

    one_sided_score = scorer.update([a, b], one_sided_states, frame_index=0, timestamp_ms=0.0)[0]
    scorer.reset()
    mutual_score = scorer.update([a, b], mutual_states, frame_index=0, timestamp_ms=0.0)[0]

    assert one_sided_score.trajectory_change_score < mutual_score.trajectory_change_score
    assert one_sided_score.motion_change_score < mutual_score.motion_change_score
    assert one_sided_score.composite_score < mutual_score.composite_score
