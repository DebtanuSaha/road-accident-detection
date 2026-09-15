"""
engine/accident/accident_scorer.py

Phase 7 — Accident Detection: per-frame evidence combination and
threshold classification. Stateless — has no memory of past frames;
`temporal_verifier.py` is what adds persistence on top of this.

PAIRWISE evidence combines three types, each derived from Phase 6's
`CollisionScore`:

  - collision evidence  <- spatial_interaction_score, floored to at
                            least `accident.active_contact_floor` ONLY
                            while the pair has a TRUE bounding-box
                            overlap this frame (`is_overlapping`), not
                            merely pixel closeness. If not overlapping
                            but the pair's gap is shrinking faster than
                            `collision.closing_speed_threshold_px_per_frame`,
                            a smaller boost is applied instead — real
                            pre-impact evidence, but weaker than actual
                            contact. Two vehicles at constant safe
                            distance (e.g. parallel travel in adjacent
                            lanes) get NEITHER boost, since they are
                            neither touching nor closing the gap.
                            (Both refinements were added after testing
                            against real footage showed ordinary
                            adjacent-lane traffic being flagged
                            ACCIDENT CONFIRMED purely from sustained
                            proximity with zero real approach — see
                            the git history / README for the specific
                            false-positive case this fixes.)
  - motion evidence     <- max(motion_change_score, post_interaction_stopping_score) —
                            a deceleration SPIKE at impact and SUSTAINED
                            stillness afterward are different time
                            windows of the same evidence; averaging
                            them (an earlier version of this code did)
                            let a genuine sustained collision lose its
                            qualifying streak within ~3 frames.
  - trajectory evidence <- trajectory_change_score

SINGLE-OBJECT evidence (`compute_single_object_evidence`) is a
separate, additive pathway for a scenario the pairwise formula above
structurally cannot see at all: one object (e.g. a motorcycle) showing
extreme, simultaneous sudden-deceleration + sudden-direction-change
with NO interacting partner — a loss-of-control / fall, not a
collision between two tracked objects. This has no "collision"
evidence component (there is no second object to be in contact with),
so its two weights (`accident.single_object_evidence_weights`) sum to
1.0 on their own. This is a genuine scope EXTENSION beyond the
original Phase 6/7 spec, added after real-footage testing showed
exactly this scenario going completely undetected. It is a heuristic,
not a guaranteed fix: if tracking itself breaks during a violent fall
(occlusion, motion blur erasing the object briefly), there is no
motion history left to evaluate, and no scoring change can recover
that — a hardware/camera-angle problem, not a software one.

`classify()` maps a bare score to a 3-tier `AccidentState` using
`accident.possible_incident_score` / `accident.accident_score` from
thresholds.yaml. IMPORTANT: this classification is purely a per-frame
threshold lookup — an `AccidentState.ACCIDENT` result from `classify()`
means "this frame's score is in the accident-level range", NOT "an
accident has been confirmed". Confirmation requires persistence across
multiple frames, which only `TemporalVerifier` (and the `confirmed`
flag it produces) can say has happened. Never treat a single-frame
`classify()` result as a confirmed accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from engine.collision import CollisionScore
from engine.motion import MotionState
from utils.logger import get_logger

logger = get_logger(__name__)


class AccidentState(str, Enum):
    """Three-tier classification of a (single-frame or confirmed) accident score."""

    NORMAL = "NORMAL"
    POSSIBLE_INCIDENT = "POSSIBLE_INCIDENT"
    ACCIDENT = "ACCIDENT"


@dataclass(frozen=True)
class AccidentEvidence:
    """The weighted evidence components behind one instantaneous accident score."""

    collision_evidence: float
    motion_evidence: float
    trajectory_evidence: float

    def to_dict(self) -> dict:
        return {
            "collision_evidence": round(self.collision_evidence, 4),
            "motion_evidence": round(self.motion_evidence, 4),
            "trajectory_evidence": round(self.trajectory_evidence, 4),
        }


class AccidentScorer:
    """
    Stateless per-frame accident evidence combiner and classifier.

    Usage (pairwise):
        scorer = AccidentScorer(settings)
        evidence = scorer.compute_evidence(collision_score)
        score = scorer.compute_score(evidence)
        state = scorer.classify(score)   # per-frame tier only, NOT a confirmation

    Usage (single-object anomaly, e.g. a vehicle losing control with no
    interacting partner):
        evidence = scorer.compute_single_object_evidence(motion_state)
        score = scorer.compute_single_object_score(evidence)
        state = scorer.classify(score)
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        thresholds = load_thresholds()
        accident_cfg = thresholds["accident"]
        collision_cfg = thresholds["collision"]

        self._possible_incident_score: float = accident_cfg["possible_incident_score"]
        self._accident_score: float = accident_cfg["accident_score"]
        self._weights = accident_cfg["evidence_weights"]
        self._active_contact_floor: float = accident_cfg["active_contact_floor"]
        self._single_object_weights = accident_cfg["single_object_evidence_weights"]
        self._closing_speed_threshold: float = collision_cfg["closing_speed_threshold_px_per_frame"]

        logger.info(
            "AccidentScorer initialized: possible_incident_score=%.2f accident_score=%.2f "
            "weights=%s active_contact_floor=%.2f closing_speed_threshold=%.2f "
            "single_object_weights=%s",
            self._possible_incident_score, self._accident_score, self._weights,
            self._active_contact_floor, self._closing_speed_threshold, self._single_object_weights,
        )

    # -- pairwise -----------------------------------------------------------------------

    def compute_evidence(self, collision_score: CollisionScore) -> AccidentEvidence:
        """Derive the three pairwise evidence components from a Phase 6 CollisionScore."""
        collision_evidence = collision_score.spatial_interaction_score

        if collision_score.is_active_spatial_contact and collision_score.is_overlapping:
            # Real, current bounding-box contact — the strongest, most direct signal.
            collision_evidence = max(collision_evidence, self._active_contact_floor)
        elif collision_score.closing_speed_px_per_frame >= self._closing_speed_threshold:
            # Not touching yet, but the gap is shrinking meaningfully faster
            # than the "just driving near each other" baseline — a weaker,
            # pre-impact signal. Capped below the full contact floor.
            ratio = min(
                1.0,
                collision_score.closing_speed_px_per_frame / (self._closing_speed_threshold * 2.0),
            )
            collision_evidence = max(collision_evidence, ratio * self._active_contact_floor)
        # else: merely close with no real overlap and no meaningful closing
        # speed (e.g. two vehicles traveling in parallel at a constant safe
        # distance) — no boost. collision_evidence stays at its raw,
        # typically-low spatial_interaction_score.

        motion_evidence = max(
            collision_score.motion_change_score, collision_score.post_interaction_stopping_score
        )
        return AccidentEvidence(
            collision_evidence=collision_evidence,
            motion_evidence=motion_evidence,
            trajectory_evidence=collision_score.trajectory_change_score,
        )

    def compute_score(self, evidence: AccidentEvidence) -> float:
        """Weighted sum of the three pairwise evidence components, clipped to [0, 1]."""
        score = (
            self._weights.get("collision", 0.0) * evidence.collision_evidence
            + self._weights.get("motion", 0.0) * evidence.motion_evidence
            + self._weights.get("trajectory", 0.0) * evidence.trajectory_evidence
        )
        return max(0.0, min(1.0, score))

    # -- single-object anomaly -----------------------------------------------------------

    def compute_single_object_evidence(self, motion_state: Optional[MotionState]) -> AccidentEvidence:
        """
        Evidence for a single tracked object showing anomalous, erratic
        motion with no interacting partner — see module docstring.
        `collision_evidence` is always 0 here: there is no second object.
        """
        if motion_state is None:
            return AccidentEvidence(collision_evidence=0.0, motion_evidence=0.0, trajectory_evidence=0.0)

        motion_evidence = 0.0
        if motion_state.previous_speed_px_per_frame:
            drop_ratio = abs(motion_state.speed_change_px_per_frame or 0.0) / motion_state.previous_speed_px_per_frame
            motion_evidence = min(1.0, drop_ratio)

        trajectory_evidence = 0.0
        if motion_state.direction_change_deg is not None:
            # Reuse the same sudden-direction-change threshold Phase 5 uses,
            # sourced from settings via the motion state's own flag for the
            # boolean case, and a direct ratio otherwise for a smoother score.
            trajectory_evidence = 1.0 if motion_state.is_sudden_direction_change else min(
                1.0, motion_state.direction_change_deg / 180.0
            )

        return AccidentEvidence(
            collision_evidence=0.0,
            motion_evidence=motion_evidence,
            trajectory_evidence=trajectory_evidence,
        )

    def compute_single_object_score(self, evidence: AccidentEvidence) -> float:
        """Weighted sum of the two single-object evidence components, clipped to [0, 1]."""
        score = (
            self._single_object_weights.get("motion", 0.0) * evidence.motion_evidence
            + self._single_object_weights.get("trajectory", 0.0) * evidence.trajectory_evidence
        )
        return max(0.0, min(1.0, score))

    # -- shared -------------------------------------------------------------------------

    def classify(self, score: float) -> AccidentState:
        """
        Pure threshold lookup for a single score. Does NOT consider
        persistence — see the module docstring's warning about not
        treating this as a confirmation.
        """
        if score >= self._accident_score:
            return AccidentState.ACCIDENT
        if score >= self._possible_incident_score:
            return AccidentState.POSSIBLE_INCIDENT
        return AccidentState.NORMAL

    @property
    def possible_incident_score(self) -> float:
        return self._possible_incident_score

    @property
    def accident_score(self) -> float:
        return self._accident_score
