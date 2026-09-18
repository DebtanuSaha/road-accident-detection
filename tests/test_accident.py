"""
tests/test_accident.py

Phase 7 tests for accident detection and temporal verification:
- accident_scorer.py: stateless evidence combination + classification
- temporal_verifier.py: the persistence FSM
- accident_detector.py: the orchestrator tying both together

Uses directly-constructed `CollisionScore` instances so assertions are
exact — Phase 6 already has its own dedicated tests for how those
scores get produced from real detections/tracking.
"""

import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from config.threshold_loader import load_thresholds  # noqa: E402
from engine.detection import BoundingBox  # noqa: E402
from engine.tracking import PositionSample, TrackedObject  # noqa: E402
from engine.collision import CollisionScore  # noqa: E402
from engine.accident import (  # noqa: E402
    AccidentAssessment,
    AccidentDetector,
    AccidentEvidence,
    AccidentScorer,
    AccidentState,
    PairVerificationState,
    TemporalVerifier,
)


def make_collision_score(
    spatial=0.0, motion=0.0, trajectory=0.0, stopping=0.0, frame_index=0, pair=(1, 2),
    is_overlapping=True, closing_speed=0.0, overlap_streak=99, is_sustained_contact=False,
    peak_closing_speed=None,
):
    """
    `overlap_streak` defaults high and `peak_closing_speed` defaults to
    exactly the configured closing-speed threshold (a "just barely converged" case)
    so the active-contact floor applies by default in tests that are
    exercising something other than the convergence-history gate itself;
    pass `peak_closing_speed=0.0` to test a pair that has NEVER genuinely
    converged (e.g. pure camera-perspective overlap between two
    independently-driving vehicles), or a large value to test a strongly-
    converged (severe) collision.
    """
    if peak_closing_speed is None:
        peak_closing_speed = load_thresholds()["collision"]["closing_speed_threshold_px_per_frame"]

    return CollisionScore(
        track_id_a=pair[0],
        track_id_b=pair[1],
        class_a="car",
        class_b="car",
        frame_index=frame_index,
        timestamp_ms=frame_index * 100.0,
        spatial_interaction_score=spatial,
        trajectory_change_score=trajectory,
        motion_change_score=motion,
        post_interaction_stopping_score=stopping,
        composite_score=0.0,  # irrelevant to Phase 7, which recomputes its own weighting
        is_active_spatial_contact=True,
        is_overlapping=is_overlapping,
        is_sustained_contact=is_sustained_contact,
        overlap_streak_frames=overlap_streak,
        closing_speed_px_per_frame=closing_speed,
        peak_closing_speed_px_per_frame=peak_closing_speed,
        in_post_interaction_window=True,
    )


@pytest.fixture
def cfg():
    return Settings()


def make_tracked_object(track_id, class_name="car", box=(0, 0, 50, 50)):
    bbox = BoundingBox(*box)
    return TrackedObject(
        track_id=track_id, class_name=class_name, class_id=2, bbox=bbox, confidence=0.9,
        first_seen_frame=0, last_seen_frame=0, first_seen_timestamp_ms=0.0, last_seen_timestamp_ms=0.0,
        position_history=deque([PositionSample(0, 0.0, bbox.center)]),
    )


@pytest.fixture
def accident_cfg():
    return load_thresholds()["accident"]


# ---------------------------------------------------------------------------
# accident_scorer.py
# ---------------------------------------------------------------------------

def test_compute_evidence_derives_from_collision_score(cfg):
    scorer = AccidentScorer(cfg)
    cs = make_collision_score(spatial=0.8, motion=0.6, trajectory=0.4, stopping=1.0)
    evidence = scorer.compute_evidence(cs)
    # spatial (0.8) already exceeds active_contact_floor, so the floor has no effect here.
    assert evidence.collision_evidence == pytest.approx(0.8)
    assert evidence.motion_evidence == pytest.approx(max(0.6, 1.0))
    assert evidence.trajectory_evidence == pytest.approx(0.4)


def test_compute_evidence_applies_active_contact_floor(cfg, accident_cfg):
    scorer = AccidentScorer(cfg)
    # Low spatial score, but actively in contact and genuinely converged
    # (default peak_closing_speed) -> floored up to at least the base floor.
    cs = make_collision_score(spatial=0.05, motion=0.0, trajectory=0.0, stopping=0.0)
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence >= accident_cfg["active_contact_floor"]


def test_compute_evidence_floor_scales_with_convergence_strength(cfg, accident_cfg):
    """
    A marginal convergence (barely past the gate) earns roughly the base
    floor; a severe, strongly-converged collision earns meaningfully more
    — genuine convergence strength should not be capped at the same flat
    value as a borderline case. Found necessary by real-footage testing:
    a flat floor left a genuine collision's peak score just under
    accident_score, never confirming.
    """
    scorer = AccidentScorer(cfg)
    marginal = make_collision_score(
        spatial=0.05,
        motion=0.0,
        trajectory=0.0,
        stopping=0.0,
        peak_closing_speed=load_thresholds()["collision"]["closing_speed_threshold_px_per_frame"],
    )
    severe = make_collision_score(spatial=0.05, motion=0.0, trajectory=0.0, stopping=0.0, peak_closing_speed=999.0)

    marginal_evidence = scorer.compute_evidence(marginal).collision_evidence
    severe_evidence = scorer.compute_evidence(severe).collision_evidence

    assert marginal_evidence == pytest.approx(accident_cfg["active_contact_floor"], abs=0.2)
    assert severe_evidence > marginal_evidence
    assert severe_evidence == pytest.approx(1.0)  # fully saturated for an overwhelming convergence reading


def test_compute_evidence_no_floor_when_not_in_active_contact(cfg):
    scorer = AccidentScorer(cfg)
    cs = CollisionScore(
        track_id_a=1, track_id_b=2, class_a="car", class_b="car",
        frame_index=0, timestamp_ms=0.0,
        spatial_interaction_score=0.05, trajectory_change_score=0.0,
        motion_change_score=0.0, post_interaction_stopping_score=0.0,
        composite_score=0.0, is_active_spatial_contact=False, is_overlapping=False, is_sustained_contact=False,
        overlap_streak_frames=0, closing_speed_px_per_frame=0.0, peak_closing_speed_px_per_frame=0.0,
        in_post_interaction_window=True,
    )
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence == pytest.approx(0.05)  # no floor applied


def test_compute_evidence_no_floor_when_close_but_not_overlapping(cfg):
    """
    Regression test for the real-footage false positive: two vehicles
    merely CLOSE (adjacent lanes) but not actually overlapping, with no
    meaningful closing speed, must NOT get the active_contact_floor —
    that was earlier applied to any "active contact" (closeness OR
    overlap), which flagged ordinary parallel traffic as ACCIDENT
    CONFIRMED.
    """
    scorer = AccidentScorer(cfg)
    cs = CollisionScore(
        track_id_a=1, track_id_b=2, class_a="car", class_b="car",
        frame_index=0, timestamp_ms=0.0,
        spatial_interaction_score=0.15, trajectory_change_score=0.0,
        motion_change_score=0.0, post_interaction_stopping_score=0.0,
        composite_score=0.0, is_active_spatial_contact=True, is_overlapping=False, is_sustained_contact=False,
        overlap_streak_frames=0, closing_speed_px_per_frame=0.0, peak_closing_speed_px_per_frame=0.0,  # never converged
        in_post_interaction_window=True,
    )
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence == pytest.approx(0.15)  # unfloored — raw spatial score only


def test_compute_evidence_closing_speed_gives_partial_boost_without_overlap(cfg, accident_cfg):
    """A pair rapidly approaching (but not yet touching) should get SOME
    boost — real pre-impact evidence — but less than the full contact floor."""
    scorer = AccidentScorer(cfg)
    cs = CollisionScore(
        track_id_a=1, track_id_b=2, class_a="car", class_b="car",
        frame_index=0, timestamp_ms=0.0,
        spatial_interaction_score=0.1, trajectory_change_score=0.0,
        motion_change_score=0.0, post_interaction_stopping_score=0.0,
        composite_score=0.0, is_active_spatial_contact=True, is_overlapping=False, is_sustained_contact=False,
        overlap_streak_frames=0, closing_speed_px_per_frame=100.0,  # far above threshold -> full ratio (capped at 1.0)
        peak_closing_speed_px_per_frame=100.0, in_post_interaction_window=True,
    )
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence == pytest.approx(accident_cfg["active_contact_floor"])


def test_compute_evidence_overlap_without_prior_convergence_does_not_floor(cfg, accident_cfg):
    """
    Regression test for the SECOND real-footage false positive: two
    vehicles whose boxes are heavily overlapped purely by camera
    perspective (adjacent/converging-looking lanes), where the pair has
    NEVER shown genuine closing speed at any point, must NOT get the
    active-contact floor — even with a long, sustained overlap streak.
    A real collision is always preceded by some period of actual
    approach; a perspective artifact never converges at all.
    """
    scorer = AccidentScorer(cfg)
    cs = CollisionScore(
        track_id_a=1, track_id_b=2, class_a="car", class_b="car",
        frame_index=0, timestamp_ms=0.0,
        spatial_interaction_score=0.3, trajectory_change_score=0.0,
        motion_change_score=0.0, post_interaction_stopping_score=0.0,
        composite_score=0.0, is_active_spatial_contact=True, is_overlapping=True, is_sustained_contact=False,
        overlap_streak_frames=99, closing_speed_px_per_frame=0.0,
        peak_closing_speed_px_per_frame=0.0,  # never converged, ever
        in_post_interaction_window=True,
    )
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence == pytest.approx(0.3)  # unfloored — raw spatial score only


def test_compute_evidence_overlap_with_prior_convergence_still_floors(cfg, accident_cfg):
    """The same sustained overlap DOES floor once the pair has genuinely
    converged at some point in its history — a real collision."""
    scorer = AccidentScorer(cfg)
    cs = CollisionScore(
        track_id_a=1, track_id_b=2, class_a="car", class_b="car",
        frame_index=0, timestamp_ms=0.0,
        spatial_interaction_score=0.05, trajectory_change_score=0.0,
        motion_change_score=0.0, post_interaction_stopping_score=0.0,
        composite_score=0.0, is_active_spatial_contact=True, is_overlapping=True, is_sustained_contact=False,
        overlap_streak_frames=99, closing_speed_px_per_frame=0.0,
        peak_closing_speed_px_per_frame=999.0,  # converged strongly at some point before settling
        in_post_interaction_window=True,
    )
    evidence = scorer.compute_evidence(cs)
    assert evidence.collision_evidence >= accident_cfg["active_contact_floor"]


def test_compute_score_uses_configured_weights(cfg, accident_cfg):
    scorer = AccidentScorer(cfg)
    evidence = AccidentEvidence(collision_evidence=1.0, motion_evidence=1.0, trajectory_evidence=1.0)
    score = scorer.compute_score(evidence)
    assert score == pytest.approx(sum(accident_cfg["evidence_weights"].values()))


def test_compute_score_is_clipped_to_unit_range(cfg):
    scorer = AccidentScorer(cfg)
    evidence = AccidentEvidence(collision_evidence=1.0, motion_evidence=1.0, trajectory_evidence=1.0)
    score = scorer.compute_score(evidence)
    assert 0.0 <= score <= 1.0


def test_classify_normal(cfg, accident_cfg):
    scorer = AccidentScorer(cfg)
    below = accident_cfg["possible_incident_score"] - 0.01
    assert scorer.classify(below) == AccidentState.NORMAL


def test_classify_possible_incident(cfg, accident_cfg):
    scorer = AccidentScorer(cfg)
    mid = (accident_cfg["possible_incident_score"] + accident_cfg["accident_score"]) / 2
    assert scorer.classify(mid) == AccidentState.POSSIBLE_INCIDENT


def test_classify_accident_tier(cfg, accident_cfg):
    scorer = AccidentScorer(cfg)
    high = accident_cfg["accident_score"] + 0.01
    assert scorer.classify(min(high, 1.0)) == AccidentState.ACCIDENT


def test_evidence_to_dict_schema():
    evidence = AccidentEvidence(0.1, 0.2, 0.3)
    assert set(evidence.to_dict().keys()) == {
        "collision_evidence", "motion_evidence", "trajectory_evidence", "stillness_evidence",
    }


# ---------------------------------------------------------------------------
# temporal_verifier.py — the persistence FSM
# ---------------------------------------------------------------------------

def test_verifier_single_high_frame_does_not_confirm(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    key = (1, 2)
    state = verifier.update(key, accident_cfg["accident_score"] + 0.05, frame_index=0)
    assert state.confirmed is False
    assert state.consecutive_qualifying_frames == 1
    assert state.state == AccidentState.POSSIBLE_INCIDENT  # opened, not confirmed


def test_verifier_streak_resets_on_drop(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    key = (1, 2)
    verifier.update(key, accident_cfg["accident_score"], frame_index=0)
    verifier.update(key, accident_cfg["accident_score"], frame_index=1)
    state = verifier.update(key, 0.0, frame_index=2)  # drop below threshold
    assert state.consecutive_qualifying_frames == 0
    assert state.state == AccidentState.NORMAL
    assert state.confirmed is False


def test_verifier_confirms_after_required_streak(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    key = (1, 2)
    n = accident_cfg["temporal_verification_frames"]
    state = None
    for i in range(n):
        state = verifier.update(key, accident_cfg["accident_score"], frame_index=i)
    assert state.confirmed is True
    assert state.consecutive_qualifying_frames == n
    assert state.confirmed_at_frame == n - 1
    assert state.state == AccidentState.ACCIDENT


def test_verifier_requires_streak_to_reach_accident_level_not_just_possible_incident(cfg, accident_cfg):
    """A long streak that never exceeds possible_incident_score (stays in that tier) should not confirm."""
    verifier = TemporalVerifier(cfg)
    key = (1, 2)
    n = accident_cfg["temporal_verification_frames"]
    mid_score = (accident_cfg["possible_incident_score"] + accident_cfg["accident_score"]) / 2
    state = None
    for i in range(n + 5):
        state = verifier.update(key, mid_score, frame_index=i)
    assert state.confirmed is False
    assert state.consecutive_qualifying_frames == n + 5  # streak kept counting...
    assert state.state == AccidentState.POSSIBLE_INCIDENT  # ...but never escalated


def test_verifier_confirmed_state_is_sticky(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    key = (1, 2)
    n = accident_cfg["temporal_verification_frames"]
    for i in range(n):
        verifier.update(key, accident_cfg["accident_score"], frame_index=i)

    # Score collapses afterward — confirmation should persist (sticky).
    state = verifier.update(key, 0.0, frame_index=n)
    assert state.confirmed is True


def test_verifier_pairs_are_independent(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    verifier.update((1, 2), accident_cfg["accident_score"], frame_index=0)
    verifier.update((3, 4), 0.0, frame_index=0)
    assert verifier.get_state((1, 2)).consecutive_qualifying_frames == 1
    assert verifier.get_state((3, 4)).consecutive_qualifying_frames == 0


def test_verifier_prune_removes_inactive_pairs(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    verifier.update((1, 2), accident_cfg["accident_score"], frame_index=0)
    verifier.update((3, 4), accident_cfg["accident_score"], frame_index=0)
    verifier.prune(active_pair_keys={(1, 2)})
    assert verifier.get_state((1, 2)) is not None
    assert verifier.get_state((3, 4)) is None


def test_verifier_reset_clears_all_pairs(cfg, accident_cfg):
    verifier = TemporalVerifier(cfg)
    verifier.update((1, 2), accident_cfg["accident_score"], frame_index=0)
    verifier.reset()
    assert verifier.get_state((1, 2)) is None


def test_pair_verification_state_to_dict_schema():
    state = PairVerificationState()
    d = state.to_dict()
    assert set(d.keys()) == {
        "state", "consecutive_qualifying_frames", "confirmed", "confirmed_at_frame", "last_score",
    }


# ---------------------------------------------------------------------------
# accident_detector.py — orchestrator
# ---------------------------------------------------------------------------

def test_detector_single_spike_frame_never_confirms(cfg, accident_cfg):
    detector = AccidentDetector(cfg)
    results = []
    for i in range(10):
        cs = make_collision_score(spatial=0.9, motion=0.9, trajectory=0.9, stopping=0.9, frame_index=i) \
            if i == 4 else make_collision_score(spatial=0.05, motion=0.0, trajectory=0.0, stopping=0.0, frame_index=i)
        out = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        results.append(out[0])

    assert not any(a.confirmed for a in results)
    spike_frame = results[4]
    assert spike_frame.instantaneous_state == AccidentState.ACCIDENT
    assert spike_frame.confirmed is False


def test_detector_confirms_sustained_interaction(cfg, accident_cfg):
    detector = AccidentDetector(cfg)
    n = accident_cfg["temporal_verification_frames"]
    last = None
    for i in range(n):
        cs = make_collision_score(spatial=0.9, motion=0.9, trajectory=0.9, stopping=0.9, frame_index=i)
        last = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)[0]

    assert last.confirmed is True
    assert last.accident_probability > accident_cfg["accident_score"]


def test_detector_handles_multiple_independent_pairs(cfg, accident_cfg):
    detector = AccidentDetector(cfg)
    cs_ab = make_collision_score(spatial=0.9, motion=0.9, trajectory=0.9, stopping=0.9, frame_index=0, pair=(1, 2))
    cs_cd = make_collision_score(spatial=0.0, motion=0.0, trajectory=0.0, stopping=0.0, frame_index=0, pair=(3, 4))
    results = detector.update([cs_ab, cs_cd], frame_index=0, timestamp_ms=0.0)

    by_pair = {(r.track_id_a, r.track_id_b): r for r in results}
    assert by_pair[(1, 2)].consecutive_qualifying_frames == 1
    assert by_pair[(3, 4)].consecutive_qualifying_frames == 0


def test_detector_reset_clears_verifier_state(cfg, accident_cfg):
    detector = AccidentDetector(cfg)
    detector.update(
        [make_collision_score(spatial=0.9, motion=0.9, trajectory=0.9, stopping=0.9, frame_index=0)],
        frame_index=0, timestamp_ms=0.0,
    )
    detector.reset()
    assert detector._verifier.get_state((1, 2)) is None


def test_accident_assessment_to_dict_schema(cfg):
    detector = AccidentDetector(cfg)
    cs = make_collision_score(spatial=0.5, motion=0.5, trajectory=0.5, stopping=0.5, frame_index=0)
    result = detector.update([cs], frame_index=0, timestamp_ms=0.0)[0]
    d = result.to_dict()
    expected_keys = {
        "pair", "classes", "frame_index", "evidence", "accident_probability",
        "instantaneous_state", "consecutive_qualifying_frames", "confirmed", "confirmed_at_frame",
        "is_single_object",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Real-footage regressions (see README for the source screenshots)
# ---------------------------------------------------------------------------

def test_sustained_parallel_traffic_never_confirms(cfg, accident_cfg):
    """
    Regression test for the false positive seen in real CCTV footage:
    two vehicles traveling in adjacent lanes at a constant safe
    distance — never overlapping, never closing the gap — must NEVER
    reach ACCIDENT CONFIRMED, no matter how long they stay near each
    other. Ordinary lane-keeping steering produces small, occasional
    trajectory/motion blips, which is exactly the scenario that used
    to accumulate a qualifying streak under the old (closeness-only)
    active_contact_floor logic.
    """
    detector = AccidentDetector(cfg)
    n = accident_cfg["temporal_verification_frames"] * 3  # well beyond the confirmation window
    any_confirmed = False
    for i in range(n):
        # Close (within min_center_distance_px) but never overlapping, never
        # approaching — plus small, non-sudden motion/trajectory noise typical
        # of ordinary driving (kept below the "sudden" thresholds).
        cs = make_collision_score(
            spatial=0.3, motion=0.1, trajectory=0.1, stopping=0.0,
            frame_index=i, is_overlapping=False, closing_speed=0.0,
        )
        results = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        any_confirmed = any_confirmed or any(r.confirmed for r in results)

    assert any_confirmed is False


def test_single_object_loss_of_control_can_confirm_without_a_partner(cfg, accident_cfg):
    """
    Regression test for the false negative seen in real footage: a
    motorcycle suddenly losing control (extreme deceleration +
    direction change) AND then remaining down must be detectable via
    the single-object pathway, with no second object involved.
    """
    detector = AccidentDetector(cfg)
    from engine.motion import MotionState

    n = accident_cfg["temporal_verification_frames"]
    # Went down and STAYED down — is_stationary carries the persistent evidence.
    downed_state = MotionState(
        track_id=99, frame_index=0, center=(0.0, 0.0), displacement_px=0.5,
        direction_deg=90.0, speed_px_per_frame=0.5, speed_px_per_sec=None,
        previous_speed_px_per_frame=20.0, speed_change_px_per_frame=-18.0,
        direction_change_deg=170.0, is_stationary=True,
        is_sudden_deceleration=True, is_sudden_direction_change=True,
        is_sudden_stop=True, sample_count=5,
    )
    tracked = [make_tracked_object(99, "motorcycle")]

    confirmed_at = None
    for i in range(n):
        results = detector.update(
            [], frame_index=i, timestamp_ms=i * 100.0,
            motion_states={99: downed_state}, tracked_objects=tracked,
        )
        single_object_results = [r for r in results if r.is_single_object]
        assert len(single_object_results) == 1
        if single_object_results[0].confirmed:
            confirmed_at = i
            break

    assert confirmed_at is not None, "a sustained single-object anomaly should eventually confirm"


def test_single_object_hard_braking_that_keeps_moving_never_confirms(cfg, accident_cfg):
    """
    Regression test for the false positives seen in real footage (a
    screenful of "solo p=0.xx" labels on ordinary traffic): a vehicle
    braking hard and steering sharply but CONTINUING TO MOVE must never
    confirm. The stillness component is what separates "braked hard"
    from "actually went down and stayed there".
    """
    detector = AccidentDetector(cfg)
    from engine.motion import MotionState

    braking_state = MotionState(
        track_id=42, frame_index=0, center=(0.0, 0.0), displacement_px=8.0,
        direction_deg=90.0, speed_px_per_frame=8.0, speed_px_per_sec=None,
        previous_speed_px_per_frame=20.0, speed_change_px_per_frame=-18.0,
        direction_change_deg=170.0, is_stationary=False,  # still moving
        is_sudden_deceleration=True, is_sudden_direction_change=True,
        is_sudden_stop=False, sample_count=5,
    )
    tracked = [make_tracked_object(42, "car")]

    any_confirmed = False
    for i in range(accident_cfg["temporal_verification_frames"] * 3):
        results = detector.update(
            [], frame_index=i, timestamp_ms=i * 100.0,
            motion_states={42: braking_state}, tracked_objects=tracked,
        )
        any_confirmed = any_confirmed or any(r.confirmed for r in results)

    assert any_confirmed is False


def test_single_object_pathway_ignored_when_no_motion_states_passed(cfg):
    """Existing call sites that don't pass motion_states/tracked_objects
    must behave exactly as before (no single-object evaluation at all)."""
    detector = AccidentDetector(cfg)
    cs = make_collision_score(spatial=0.9, motion=0.9, trajectory=0.9, stopping=0.9)
    results = detector.update([cs], frame_index=0, timestamp_ms=0.0)
    assert all(not r.is_single_object for r in results)
    assert len(results) == 1  # only the pairwise result, nothing else


def test_one_sided_turn_near_undisturbed_neighbor_never_confirms(cfg, accident_cfg):
    """
    Direct regression test for the real-footage false positive: a
    vehicle turning sharply right next to a completely undisturbed
    neighbor (axis-aligned bbox overlap inflated by the turning
    vehicle's rotation, sustained the whole time) must never confirm —
    this exact combination of inputs would have reached
    ACCIDENT CONFIRMED under the pre-fix max()-based evidence blending.
    """
    detector = AccidentDetector(cfg)
    from engine.collision.collision_scorer import _mutual_corroboration_blend as blend

    traj = blend(1.0, 0.0)    # only one object turning
    motion = blend(0.6, 0.0)  # only one object decelerating

    any_confirmed = False
    last = None
    for i in range(60):
        cs = make_collision_score(
            spatial=0.85, motion=motion, trajectory=traj, stopping=0.0,
            frame_index=i, overlap_streak=99,
        )
        results = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        any_confirmed = any_confirmed or results[0].confirmed
        last = results[0]

    assert any_confirmed is False
    assert last.accident_probability < accident_cfg["accident_score"]


def test_mutual_disturbance_still_confirms(cfg):
    """The mutual-corroboration fix must not break genuine collisions
    where BOTH participants are disturbed."""
    detector = AccidentDetector(cfg)
    from engine.collision.collision_scorer import _mutual_corroboration_blend as blend

    traj = blend(0.9, 0.9)
    motion = blend(0.9, 0.85)

    any_confirmed = False
    for i in range(60):
        cs = make_collision_score(
            spatial=0.9, motion=motion, trajectory=traj, stopping=0.0,
            frame_index=i, overlap_streak=99,
        )
        results = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        any_confirmed = any_confirmed or results[0].confirmed

    assert any_confirmed is True


def test_pure_perspective_overlap_never_confirms_even_with_mutual_motion_noise(cfg, accident_cfg):
    """
    Regression test for the SECOND real-footage false positive: two
    vehicles heavily overlapped by camera perspective the whole time
    (never genuinely converging), each independently showing moderate,
    MUTUAL motion/heading noise (ordinary, uncorrelated driving —
    already passes the mutual-corroboration check from the first fix),
    must still never confirm. Only the convergence-history gate catches
    this: the pair never showed real closing speed, so the active-
    contact floor is never granted no matter how sustained or mutual
    the surrounding evidence looks.
    """
    detector = AccidentDetector(cfg)
    any_confirmed = False
    last = None
    for i in range(60):
        cs = make_collision_score(
            spatial=0.3, motion=1.0, trajectory=1.0, stopping=0.0,
            frame_index=i, overlap_streak=99, is_overlapping=True,
            peak_closing_speed=0.0,  # never converged
        )
        results = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        any_confirmed = any_confirmed or results[0].confirmed
        last = results[0]

    assert any_confirmed is False
    assert last.accident_probability < accident_cfg["accident_score"]


def test_same_scenario_with_genuine_convergence_does_confirm(cfg):
    """Control: identical motion/trajectory evidence DOES confirm once the
    pair has shown genuine convergence at some point — proves the gate
    discriminates on convergence history, not on suppressing everything."""
    detector = AccidentDetector(cfg)
    any_confirmed = False
    for i in range(60):
        cs = make_collision_score(
            spatial=0.3, motion=1.0, trajectory=1.0, stopping=0.0,
            frame_index=i, overlap_streak=99, is_overlapping=True,
            peak_closing_speed=999.0,  # genuinely converged before settling
        )
        results = detector.update([cs], frame_index=i, timestamp_ms=i * 100.0)
        any_confirmed = any_confirmed or results[0].confirmed

    assert any_confirmed is True
