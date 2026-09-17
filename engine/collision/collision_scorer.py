"""
engine/collision/collision_scorer.py

Phase 6 — Collision Detection: composite, multi-signal scoring.

Combines this frame's spatial interactions (collision_detector.py)
with each involved object's current motion state (Phase 5,
engine.motion.MotionState) into one composite collision score per
interacting pair, using the configurable weights in
config/thresholds.yaml (collision.weights):

    Collision Score =
        spatial_interaction
        + trajectory_change
        + motion_change
        + post_interaction_stopping

Also implements the two indicators that are inherently multi-frame:
  6. Abnormal movement immediately after interaction
  7. Multiple objects becoming stationary after interaction
via a short "active window" kept per pair: spatial contact opens the
window, and the window's signals keep being evaluated for a few more
frames afterward even once the boxes have separated again — a real
collision's aftermath (vehicles stopping, swerving) naturally lags the
contact moment by a frame or two.

Also tracks two additional per-pair signals consumed by Phase 7's
AccidentScorer to distinguish genuine contact from camera-perspective
artifacts (added after real-footage testing kept showing ordinary
adjacent-lane traffic flagged as colliding):
  - `overlap_streak_frames` — how many CONSECUTIVE frames real bbox
    overlap has been observed. A one-frame overlap from near/far-lane
    perspective compression looks very different from a streak of 3+.
  - `closing_speed_px_per_frame` — EXPONENTIALLY SMOOTHED rate the
    pair's center distance is shrinking, not a raw 2-frame derivative
    (which is noisy enough at typical CCTV tracking precision to
    spuriously cross a low threshold from jitter alone).

MUTUAL CORROBORATION (added after real-footage testing showed a car
mid-turn near an undisturbed, normally-moving neighboring vehicle
flagged as ACCIDENT CONFIRMED): `trajectory_change` and `motion_change`
now require disturbance from BOTH objects in the pair, not just one —
see `_mutual_corroboration_blend()` for the exact formula and
reasoning. One vehicle's ordinary turn or brake near a completely
undisturbed neighbor no longer single-handedly maxes out a pair's
collision evidence.

This module produces a SCORE, not a decision. "Do not classify an
accident from a single frame" is honored two ways here: (a) the
composite score only gets high when several independent signals agree
in the same frame, and (b) whether a score — however high — represents
a CONFIRMED accident is Phase 7's job (temporal verification across
many frames), not this module's. No "accident" state is produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from engine.collision.collision_detector import CollisionDetector, SpatialInteraction
from engine.motion import MotionState
from engine.tracking import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)

PairKey = Tuple[int, int]

# Smoothing factor for the closing-speed EMA (0 < alpha <= 1). Lower =
# smoother/less reactive to a single noisy frame, higher = tracks raw
# per-frame changes more closely. Not exposed via thresholds.yaml —
# an internal noise-rejection detail rather than a scoring policy knob.
_CLOSING_SPEED_EMA_ALPHA = 0.4


def _pair_key(id_a: int, id_b: int) -> PairKey:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def _mutual_corroboration_blend(value_a: float, value_b: float) -> float:
    """
    Blend two per-object evidence values (each already in [0,1]) into one
    pairwise score, discounting evidence that comes from only ONE side of
    the pair.

    ARCHITECTURAL FIX (found via real-footage debugging): the previous
    design took max(value_a, value_b) — meaning ONE vehicle's ordinary,
    unilateral maneuver (a normal turn, a normal lane change, normal
    braking) could single-handedly max out a pair's "collision evidence",
    even while its neighbor showed ZERO disturbance. A real collision
    physically disturbs both participants to some degree (an impact
    transfers momentum, or the struck vehicle visibly reacts) — a vehicle
    turning near an undisturbed, normally-moving neighbor is an ordinary
    traffic event, not two bodies interacting.

    This blend keeps max(a,b) as the ceiling (so a genuinely severe
    one-sided reading still contributes something — real crashes are
    often asymmetric, e.g. one car brakes hard while the other barely
    reacts) but scales it down toward 50% when the OTHER side shows
    little to no corroborating disturbance, and keeps it at 100% when
    both sides show comparable disturbance:

        blended = max(a,b) * (0.5 + 0.5 * min(a,b)/max(a,b))

    e.g. a=1.0, b=0.0 (fully one-sided)  -> 0.50
         a=1.0, b=0.5 (partly mutual)    -> 0.75
         a=1.0, b=1.0 (fully mutual)     -> 1.00
    """
    hi = max(value_a, value_b)
    if hi <= 0.0:
        return 0.0
    lo = min(value_a, value_b)
    mutual_ratio = lo / hi
    return hi * (0.5 + 0.5 * mutual_ratio)


@dataclass
class _PairWindow:
    """Internal bookkeeping: how recently (and how strongly) a pair has interacted."""

    last_interaction_frame: int
    peak_spatial_score: float
    last_distance_px: Optional[float] = None
    last_distance_frame: Optional[int] = None
    consecutive_overlap_frames: int = 0
    smoothed_closing_speed: float = 0.0


@dataclass(frozen=True)
class CollisionScore:
    """Composite, multi-signal collision score for one pair of objects, this frame."""

    track_id_a: int
    track_id_b: int
    class_a: str
    class_b: str
    frame_index: int
    timestamp_ms: float

    spatial_interaction_score: float
    trajectory_change_score: float
    motion_change_score: float
    post_interaction_stopping_score: float
    composite_score: float

    is_active_spatial_contact: bool  # boxes overlapping/close THIS exact frame
    is_overlapping: bool  # TRUE bounding-box overlap this frame (strict entry threshold)
    is_sustained_contact: bool  # pair remains in contact via the looser SUSTAIN thresholds (collision aftermath)
    overlap_streak_frames: int  # consecutive frames of real overlap (carried through the grace window)
    closing_speed_px_per_frame: float  # SMOOTHED rate the pair's center distance is SHRINKING; 0 if not fresh this frame
    in_post_interaction_window: bool  # still within the grace window after a past contact

    def to_dict(self) -> dict:
        return {
            "pair": [self.track_id_a, self.track_id_b],
            "classes": [self.class_a, self.class_b],
            "frame_index": self.frame_index,
            "spatial_interaction_score": round(self.spatial_interaction_score, 4),
            "trajectory_change_score": round(self.trajectory_change_score, 4),
            "motion_change_score": round(self.motion_change_score, 4),
            "post_interaction_stopping_score": round(self.post_interaction_stopping_score, 4),
            "composite_score": round(self.composite_score, 4),
            "is_active_spatial_contact": self.is_active_spatial_contact,
            "is_overlapping": self.is_overlapping,
            "is_sustained_contact": self.is_sustained_contact,
            "overlap_streak_frames": self.overlap_streak_frames,
            "closing_speed_px_per_frame": round(self.closing_speed_px_per_frame, 3),
            "in_post_interaction_window": self.in_post_interaction_window,
        }


class CollisionScorer:
    """
    Stateful (per object-pair) multi-signal collision scorer.

    Usage (call once per frame, after tracking + motion analysis):
        scorer = CollisionScorer(settings)
        scores = scorer.update(tracked_objects, motion_states, frame_index, timestamp_ms)
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._detector = CollisionDetector(cfg)

        collision_cfg = load_thresholds()["collision"]
        self._weights: Dict[str, float] = collision_cfg["weights"]
        self._post_interaction_window_frames: int = collision_cfg["post_interaction_window_frames"]

        motion_cfg = load_thresholds()["motion"]
        self._sudden_direction_change_deg: float = motion_cfg["sudden_direction_change_deg"]

        self._pair_windows: Dict[PairKey, _PairWindow] = {}

        logger.info(
            "CollisionScorer initialized: weights=%s post_interaction_window_frames=%d",
            self._weights, self._post_interaction_window_frames,
        )

    def update(
        self,
        tracked_objects: Sequence[TrackedObject],
        motion_states: Dict[int, MotionState],
        frame_index: int,
        timestamp_ms: float,
    ) -> List[CollisionScore]:
        objects_by_id = {t.track_id: t for t in tracked_objects}
        interactions = self._detector.find_interactions(tracked_objects)
        interactions_by_pair: Dict[PairKey, SpatialInteraction] = {
            _pair_key(i.track_id_a, i.track_id_b): i for i in interactions
        }

        # 1) Register/refresh this frame's spatial contacts into the per-pair
        #    window: peak spatial score, EMA-smoothed closing speed, and the
        #    consecutive-overlap streak (reset to 0 the instant a frame with
        #    fresh geometry shows no real overlap; carried forward unchanged
        #    for frames without fresh geometry — see class docstring).
        for key, interaction in interactions_by_pair.items():
            existing = self._pair_windows.get(key)
            peak = interaction.spatial_score if existing is None else max(existing.peak_spatial_score, interaction.spatial_score)

            closing_speed_raw = 0.0
            if existing is not None and existing.last_distance_px is not None and existing.last_distance_frame is not None:
                frame_delta = frame_index - existing.last_distance_frame
                if frame_delta > 0:
                    # Positive = distance shrinking (approaching each other);
                    # negative = separating.
                    closing_speed_raw = (existing.last_distance_px - interaction.center_distance_px) / frame_delta

            prev_smoothed = existing.smoothed_closing_speed if existing is not None else 0.0
            smoothed_closing_speed = (
                _CLOSING_SPEED_EMA_ALPHA * closing_speed_raw + (1 - _CLOSING_SPEED_EMA_ALPHA) * prev_smoothed
            )

            if interaction.is_overlapping:
                prev_streak = existing.consecutive_overlap_frames if existing is not None else 0
                contiguous = (
                    existing is not None
                    and existing.last_interaction_frame == frame_index - 1
                    and prev_streak > 0
                )
                overlap_streak = prev_streak + 1 if contiguous else 1
            else:
                overlap_streak = 0

            self._pair_windows[key] = _PairWindow(
                last_interaction_frame=frame_index,
                peak_spatial_score=peak,
                last_distance_px=interaction.center_distance_px,
                last_distance_frame=frame_index,
                consecutive_overlap_frames=overlap_streak,
                smoothed_closing_speed=smoothed_closing_speed,
            )

        # 2) Every pair still within its post-interaction window gets scored
        #    this frame (includes pairs with fresh contact this frame).
        pairs_to_score = {
            key: w for key, w in self._pair_windows.items()
            if frame_index - w.last_interaction_frame <= self._post_interaction_window_frames
        }

        # 3) Evict pairs that have aged out of the window entirely.
        for key in list(self._pair_windows.keys()):
            if key not in pairs_to_score:
                del self._pair_windows[key]

        scores: List[CollisionScore] = []
        for key, window in pairs_to_score.items():
            id_a, id_b = key
            obj_a = objects_by_id.get(id_a)
            obj_b = objects_by_id.get(id_b)
            if obj_a is None or obj_b is None:
                continue  # one side of the pair is no longer tracked at all

            interaction = interactions_by_pair.get(key)
            if interaction is not None:
                effective_spatial_score = interaction.spatial_score
                closing_speed = window.smoothed_closing_speed
            else:
                # Still within the grace window but not in contact this
                # exact frame — credit half the peak so the composite
                # doesn't collapse to near-zero the instant boxes
                # separate slightly. This is what lets indicators #6/#7
                # (post-interaction behaviour) actually register. Closing
                # speed isn't meaningful without fresh geometry this frame.
                effective_spatial_score = window.peak_spatial_score * 0.5
                closing_speed = 0.0

            state_a = motion_states.get(id_a)
            state_b = motion_states.get(id_b)

            trajectory_change_score = self._trajectory_change_score(state_a, state_b)
            motion_change_score = self._motion_change_score(state_a, state_b)
            post_interaction_stopping_score = self._post_interaction_stopping_score(state_a, state_b)

            composite = (
                self._weights.get("spatial_interaction", 0.0) * effective_spatial_score
                + self._weights.get("trajectory_change", 0.0) * trajectory_change_score
                + self._weights.get("motion_change", 0.0) * motion_change_score
                + self._weights.get("post_interaction_stopping", 0.0) * post_interaction_stopping_score
            )

            score = CollisionScore(
                track_id_a=id_a,
                track_id_b=id_b,
                class_a=obj_a.class_name,
                class_b=obj_b.class_name,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                spatial_interaction_score=effective_spatial_score,
                trajectory_change_score=trajectory_change_score,
                motion_change_score=motion_change_score,
                post_interaction_stopping_score=post_interaction_stopping_score,
                composite_score=composite,
                is_active_spatial_contact=interaction is not None,
                is_overlapping=(interaction.is_overlapping if interaction is not None else False),
                is_sustained_contact=(interaction.is_sustained_contact if interaction is not None else False),
                overlap_streak_frames=window.consecutive_overlap_frames,
                closing_speed_px_per_frame=closing_speed,
                in_post_interaction_window=True,
            )
            scores.append(score)

            if composite >= 0.5:  # informational only; Phase 7 owns real decision thresholds
                logger.info(
                    "Elevated collision score: pair=(%d,%d) classes=(%s,%s) frame=%d composite=%.3f",
                    id_a, id_b, obj_a.class_name, obj_b.class_name, frame_index, composite,
                )

        return scores

    def reset(self) -> None:
        """Clear all per-pair interaction windows (e.g. between independent video runs)."""
        self._pair_windows.clear()

    # -- per-signal component scores, each in [0, 1] -----------------------------------

    def _trajectory_change_score(self, a: Optional[MotionState], b: Optional[MotionState]) -> float:
        """
        Combined heading-change evidence for the pair, requiring MUTUAL
        (not just one-sided) disturbance — see `_mutual_corroboration_blend`
        docstring for why this matters.
        """
        ratio_a = self._single_direction_ratio(a)
        ratio_b = self._single_direction_ratio(b)
        return _mutual_corroboration_blend(ratio_a, ratio_b)

    def _single_direction_ratio(self, state: Optional[MotionState]) -> float:
        if state is None or state.direction_change_deg is None:
            return 0.0
        if self._sudden_direction_change_deg <= 0:
            return 0.0
        return min(1.0, state.direction_change_deg / self._sudden_direction_change_deg)

    def _motion_change_score(self, a: Optional[MotionState], b: Optional[MotionState]) -> float:
        """
        Combined deceleration/acceleration evidence for the pair, requiring
        MUTUAL (not just one-sided) disturbance.
        """
        ratio_a = self._single_motion_ratio(a)
        ratio_b = self._single_motion_ratio(b)
        return _mutual_corroboration_blend(ratio_a, ratio_b)

    def _single_motion_ratio(self, state: Optional[MotionState]) -> float:
        if state is None or state.speed_change_px_per_frame is None:
            return 0.0
        if not state.previous_speed_px_per_frame:
            return 0.0
        return min(1.0, abs(state.speed_change_px_per_frame) / state.previous_speed_px_per_frame)

    def _post_interaction_stopping_score(self, a: Optional[MotionState], b: Optional[MotionState]) -> float:
        """Fraction of the pair that is now stationary (0.0, 0.5, or 1.0)."""
        stationary_count = 0
        evaluated = 0
        for state in (a, b):
            if state is None:
                continue
            evaluated += 1
            if state.is_stationary:
                stationary_count += 1
        if evaluated == 0:
            return 0.0
        return stationary_count / evaluated
