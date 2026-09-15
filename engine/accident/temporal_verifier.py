"""
engine/accident/temporal_verifier.py

Phase 7 — Accident Detection: temporal verification (persistence).

Implements the FSM the project spec describes:

    Potential Accident
        v
    Observe following frames
        v
    Condition persists?
        v (yes)
    Accident Confirmed

Per pair of tracked objects, tracks a CONSECUTIVE streak of frames
whose instantaneous accident score is at or above
`accident.possible_incident_score`. A pair only becomes CONFIRMED once
that streak reaches `accident.temporal_verification_frames` AND the
streak actually reached accident-level severity at some point (not
just possible-incident level) — this is what makes "avoid triggering
an alert from one abnormal frame" literally true: a single high frame
has a streak of 1, which can never reach a multi-frame requirement.

Stateful and keyed per object pair; state is NOT shared across pairs
and does not know anything about collision/motion computation itself
(that's `accident_scorer.py`) — this module only tracks WHETHER a
given per-frame score persisted for a given pair, independent of how
that score was computed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from engine.accident.accident_scorer import AccidentScorer, AccidentState
from utils.logger import get_logger

logger = get_logger(__name__)

PairKey = Tuple[int, int]


@dataclass
class PairVerificationState:
    """Per-pair temporal verification bookkeeping."""

    state: AccidentState = AccidentState.NORMAL
    consecutive_qualifying_frames: int = 0
    reached_accident_level: bool = False  # did the streak ever hit accident_score, not just possible_incident?
    confirmed: bool = False
    confirmed_at_frame: Optional[int] = None
    last_score: float = 0.0
    last_updated_frame: int = -1
    score_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "consecutive_qualifying_frames": self.consecutive_qualifying_frames,
            "confirmed": self.confirmed,
            "confirmed_at_frame": self.confirmed_at_frame,
            "last_score": round(self.last_score, 4),
        }


class TemporalVerifier:
    """
    Stateful, per-pair persistence tracker.

    Usage (call once per frame, per currently-scored pair):
        verifier = TemporalVerifier(settings)
        state = verifier.update(pair_key, instantaneous_score, frame_index)
        verifier.prune(current_frame_pair_keys)   # drop pairs no longer being scored
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        accident_cfg = load_thresholds()["accident"]
        self._possible_incident_score: float = accident_cfg["possible_incident_score"]
        self._accident_score: float = accident_cfg["accident_score"]
        self._temporal_verification_frames: int = accident_cfg["temporal_verification_frames"]

        self._pairs: Dict[PairKey, PairVerificationState] = {}

        logger.info(
            "TemporalVerifier initialized: temporal_verification_frames=%d "
            "possible_incident_score=%.2f accident_score=%.2f",
            self._temporal_verification_frames, self._possible_incident_score, self._accident_score,
        )

    def update(self, pair_key: PairKey, instantaneous_score: float, frame_index: int) -> PairVerificationState:
        pair_state = self._pairs.setdefault(pair_key, PairVerificationState())

        pair_state.last_score = instantaneous_score
        pair_state.last_updated_frame = frame_index
        pair_state.score_history.append(instantaneous_score)

        is_qualifying = instantaneous_score >= self._possible_incident_score

        if pair_state.confirmed:
            # Sticky once confirmed — this interaction is a done deal for
            # reporting purposes; Phase 8 owns what happens to it next.
            return pair_state

        if is_qualifying:
            pair_state.consecutive_qualifying_frames += 1
            if instantaneous_score >= self._accident_score:
                pair_state.reached_accident_level = True

            if pair_state.state == AccidentState.NORMAL:
                pair_state.state = AccidentState.POSSIBLE_INCIDENT
                logger.info(
                    "Potential accident opened: pair=%s frame=%d score=%.3f",
                    pair_key, frame_index, instantaneous_score,
                )

            if (
                pair_state.consecutive_qualifying_frames >= self._temporal_verification_frames
                and pair_state.reached_accident_level
            ):
                pair_state.state = AccidentState.ACCIDENT
                pair_state.confirmed = True
                pair_state.confirmed_at_frame = frame_index
                logger.info(
                    "ACCIDENT CONFIRMED: pair=%s frame=%d streak=%d frames, last_score=%.3f",
                    pair_key, frame_index, pair_state.consecutive_qualifying_frames, instantaneous_score,
                )
        else:
            if pair_state.consecutive_qualifying_frames > 0:
                logger.debug(
                    "Potential accident streak broken: pair=%s frame=%d (was %d frames)",
                    pair_key, frame_index, pair_state.consecutive_qualifying_frames,
                )
            pair_state.consecutive_qualifying_frames = 0
            pair_state.reached_accident_level = False
            pair_state.state = AccidentState.NORMAL

        return pair_state

    def get_state(self, pair_key: PairKey) -> Optional[PairVerificationState]:
        return self._pairs.get(pair_key)

    def prune(self, active_pair_keys: Set[PairKey]) -> None:
        """
        Drop verification state for pairs no longer being scored by
        CollisionScorer (e.g. its own post-interaction window expired).
        Confirmed pairs are pruned too once inactive — Phase 8 is
        expected to have already consumed a confirmed accident by then.
        """
        stale = [key for key in self._pairs if key not in active_pair_keys]
        for key in stale:
            del self._pairs[key]

    def reset(self) -> None:
        self._pairs.clear()
