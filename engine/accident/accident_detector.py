"""
engine/accident/accident_detector.py

Phase 7 — Accident Detection: top-level orchestrator.

Thin facade wiring `AccidentScorer` (per-frame evidence + score) and
`TemporalVerifier` (persistence FSM) together into one call per frame,
so the rest of the pipeline (scripts, and eventually the FastAPI
backend) has a single entry point rather than needing to know about
both sub-modules.

Handles TWO independent evaluation pathways, both routed through the
same `TemporalVerifier` FSM (so both get the identical no-single-frame-
trigger persistence protection):

  1. PAIRWISE — from Phase 6's `CollisionScore`s, as originally
     designed: two tracked objects interacting.
  2. SINGLE-OBJECT anomaly — added after real-footage testing showed a
     real gap: a vehicle (e.g. a motorcycle) losing control and
     falling, with NO second object involved at all, is invisible to
     the pairwise pathway by construction. Uses a self-paired key
     `(track_id, track_id)` to reuse the exact same FSM code rather
     than duplicating persistence logic. Only evaluated for objects
     whose current `MotionState` already shows a sudden-deceleration
     or sudden-direction-change flag — calm objects are skipped
     entirely, which has no effect on confirmation behavior (a skipped
     object's streak would reset to 0 anyway) but avoids needless
     per-frame work across every idle object in a busy scene.

Produces `AccidentAssessment` per currently-scored pair (or single
object): the instantaneous evidence/score/tier for this frame, PLUS
the temporal verification state (consecutive streak, and whether this
pair/object has been CONFIRMED as an accident). Only `confirmed=True`
represents an actual decision this system stands behind — everything
else is provisional, single-frame information kept for visibility/
debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from engine.accident.accident_scorer import AccidentEvidence, AccidentScorer, AccidentState
from engine.accident.temporal_verifier import PairKey, TemporalVerifier
from engine.collision import CollisionScore
from engine.motion import MotionState
from engine.tracking import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


def _pair_key(id_a: int, id_b: int) -> PairKey:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


@dataclass(frozen=True)
class AccidentAssessment:
    """
    Full per-frame, per-pair accident assessment: this frame's evidence
    and score, plus where that pair currently stands in temporal
    verification.
    """

    track_id_a: int
    track_id_b: int
    class_a: str
    class_b: str
    frame_index: int
    timestamp_ms: float

    evidence: AccidentEvidence
    accident_probability: float  # this frame's instantaneous score, 0-1
    instantaneous_state: AccidentState  # this frame's tier alone — NOT a confirmation

    consecutive_qualifying_frames: int
    confirmed: bool  # the only field that represents an actual decision
    confirmed_at_frame: Optional[int]
    is_single_object: bool = False  # True for the loss-of-control pathway (track_id_a == track_id_b)

    def to_dict(self) -> dict:
        return {
            "pair": [self.track_id_a, self.track_id_b],
            "classes": [self.class_a, self.class_b],
            "frame_index": self.frame_index,
            "evidence": self.evidence.to_dict(),
            "accident_probability": round(self.accident_probability, 4),
            "instantaneous_state": self.instantaneous_state.value,
            "consecutive_qualifying_frames": self.consecutive_qualifying_frames,
            "confirmed": self.confirmed,
            "confirmed_at_frame": self.confirmed_at_frame,
            "is_single_object": self.is_single_object,
        }


class AccidentDetector:
    """
    Usage (call once per frame, after Phase 6's CollisionScorer):
        detector = AccidentDetector(settings)
        assessments = detector.update(collision_scores, frame_index, timestamp_ms)
        confirmed = [a for a in assessments if a.confirmed]
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._scorer = AccidentScorer(cfg)
        self._verifier = TemporalVerifier(cfg)
        self._newly_confirmed_pairs: set = set()

    def update(
        self,
        collision_scores: Sequence[CollisionScore],
        frame_index: int,
        timestamp_ms: float,
        motion_states: Optional[Dict[int, MotionState]] = None,
        tracked_objects: Sequence[TrackedObject] = (),
    ) -> List[AccidentAssessment]:
        """
        `motion_states` and `tracked_objects` are optional — omitting
        them (existing call sites) simply skips the single-object
        anomaly pathway and behaves exactly as before; pass both to
        enable it.
        """
        assessments: List[AccidentAssessment] = []
        active_pair_keys = set()

        for cs in collision_scores:
            key = _pair_key(cs.track_id_a, cs.track_id_b)
            active_pair_keys.add(key)

            evidence = self._scorer.compute_evidence(cs)
            score = self._scorer.compute_score(evidence)
            instantaneous_state = self._scorer.classify(score)

            was_confirmed_before = (
                self._verifier.get_state(key).confirmed if self._verifier.get_state(key) else False
            )
            pair_state = self._verifier.update(key, score, frame_index)

            if pair_state.confirmed and not was_confirmed_before:
                logger.info(
                    "AccidentDetector: pair=%s classes=(%s,%s) CONFIRMED at frame=%d "
                    "(probability=%.3f, streak=%d frames)",
                    key, cs.class_a, cs.class_b, frame_index, score, pair_state.consecutive_qualifying_frames,
                )

            assessments.append(
                AccidentAssessment(
                    track_id_a=cs.track_id_a,
                    track_id_b=cs.track_id_b,
                    class_a=cs.class_a,
                    class_b=cs.class_b,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    evidence=evidence,
                    accident_probability=score,
                    instantaneous_state=instantaneous_state,
                    consecutive_qualifying_frames=pair_state.consecutive_qualifying_frames,
                    confirmed=pair_state.confirmed,
                    confirmed_at_frame=pair_state.confirmed_at_frame,
                    is_single_object=False,
                )
            )

        if motion_states:
            for obj in tracked_objects:
                if not obj.is_active:
                    continue
                state = motion_states.get(obj.track_id)
                if state is None or not (state.is_sudden_deceleration or state.is_sudden_direction_change):
                    continue  # calm object — skipping has no effect on FSM outcome, see docstring

                key = _pair_key(obj.track_id, obj.track_id)  # self-paired key
                active_pair_keys.add(key)

                evidence = self._scorer.compute_single_object_evidence(state)
                score = self._scorer.compute_single_object_score(evidence)
                instantaneous_state = self._scorer.classify(score)

                was_confirmed_before = (
                    self._verifier.get_state(key).confirmed if self._verifier.get_state(key) else False
                )
                pair_state = self._verifier.update(key, score, frame_index)

                if pair_state.confirmed and not was_confirmed_before:
                    logger.info(
                        "AccidentDetector: SINGLE-OBJECT track_id=%d class=%s CONFIRMED at frame=%d "
                        "(probability=%.3f, streak=%d frames) — loss of control, no interacting partner",
                        obj.track_id, obj.class_name, frame_index, score, pair_state.consecutive_qualifying_frames,
                    )

                assessments.append(
                    AccidentAssessment(
                        track_id_a=obj.track_id,
                        track_id_b=obj.track_id,
                        class_a=obj.class_name,
                        class_b=obj.class_name,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        evidence=evidence,
                        accident_probability=score,
                        instantaneous_state=instantaneous_state,
                        consecutive_qualifying_frames=pair_state.consecutive_qualifying_frames,
                        confirmed=pair_state.confirmed,
                        confirmed_at_frame=pair_state.confirmed_at_frame,
                        is_single_object=True,
                    )
                )

        self._verifier.prune(active_pair_keys)
        return assessments

    def reset(self) -> None:
        self._verifier.reset()
