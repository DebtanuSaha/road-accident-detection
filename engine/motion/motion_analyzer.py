"""
engine/motion/motion_analyzer.py

Phase 5 — Movement and Trajectory Analysis: per-object motion profile.

Combines `trajectory.py` (geometry) and `velocity.py` (pixel-space
speed) with the configurable thresholds in `config/thresholds.yaml`
(`motion:` section) to produce a `MotionState` per tracked object:
displacement, direction, relative speed, change in speed
(acceleration proxy), direction change, and sudden-stop /
sudden-deceleration / sudden-direction-change flags.

Reads only from `TrackedObject.position_history` (built by
`engine.tracking.TrackManager`, Phase 4) — this module adds no new
state to the track itself and has no awareness of collision/accident
logic (that's Phase 6/7), keeping the "Detection → Tracking → Motion
→ Collision → Accident" stages cleanly separated per the project
architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from engine.motion.trajectory import angular_difference_deg, compute_direction_deg, compute_displacement
from engine.motion.velocity import compute_speed_change, compute_speed_px_per_frame, compute_speed_px_per_sec
from engine.tracking import PositionSample, TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MotionState:
    """
    A tracked object's motion profile as of its most recent position
    sample. All speed/acceleration fields are PIXEL-SPACE, RELATIVE
    measurements — see engine/motion/velocity.py for why these are
    never real-world km/h.
    """

    track_id: int
    frame_index: int
    center: Tuple[float, float]

    displacement_px: float
    direction_deg: Optional[float]  # heading of the most recent movement segment

    speed_px_per_frame: float  # primary metric; used for all threshold comparisons
    speed_px_per_sec: Optional[float]  # informational only; None if FPS/timestamps unreliable

    previous_speed_px_per_frame: Optional[float]
    speed_change_px_per_frame: Optional[float]  # acceleration/deceleration proxy

    direction_change_deg: Optional[float]  # heading delta vs. the previous segment

    is_stationary: bool
    is_sudden_deceleration: bool
    is_sudden_direction_change: bool
    is_sudden_stop: bool

    sample_count: int  # how many position samples this estimate was derived from

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "center": [round(v, 2) for v in self.center],
            "displacement_px": round(self.displacement_px, 3),
            "direction_deg": round(self.direction_deg, 1) if self.direction_deg is not None else None,
            "speed_px_per_frame": round(self.speed_px_per_frame, 3),
            "speed_px_per_sec": round(self.speed_px_per_sec, 3) if self.speed_px_per_sec is not None else None,
            "speed_change_px_per_frame": (
                round(self.speed_change_px_per_frame, 3) if self.speed_change_px_per_frame is not None else None
            ),
            "direction_change_deg": (
                round(self.direction_change_deg, 1) if self.direction_change_deg is not None else None
            ),
            "is_stationary": self.is_stationary,
            "is_sudden_deceleration": self.is_sudden_deceleration,
            "is_sudden_direction_change": self.is_sudden_direction_change,
            "is_sudden_stop": self.is_sudden_stop,
            "sample_count": self.sample_count,
        }


class MotionAnalyzer:
    """
    Computes a `MotionState` for tracked objects from their stored
    position history.

    Usage:
        analyzer = MotionAnalyzer(settings)
        state = analyzer.analyze(tracked_object)          # single object
        states = analyzer.analyze_many(tracked_objects)    # {track_id: MotionState}
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        motion_cfg = load_thresholds()["motion"]
        self._stationary_speed_px: float = motion_cfg["stationary_speed_px"]
        self._sudden_deceleration_ratio: float = motion_cfg["sudden_deceleration_ratio"]
        self._sudden_direction_change_deg: float = motion_cfg["sudden_direction_change_deg"]
        # Rolling window used for these calculations; clamped to the hard
        # storage cap so it can never exceed what's actually retained.
        self._window = min(motion_cfg["history_length"], cfg.TRACK_HISTORY_LENGTH)
        logger.info(
            "MotionAnalyzer initialized: stationary_speed_px=%.2f sudden_deceleration_ratio=%.2f "
            "sudden_direction_change_deg=%.1f window=%d",
            self._stationary_speed_px, self._sudden_deceleration_ratio,
            self._sudden_direction_change_deg, self._window,
        )

    def analyze(self, track: TrackedObject) -> Optional[MotionState]:
        """
        Compute the current MotionState for one tracked object.

        Returns None if there isn't yet enough position history (fewer
        than 2 samples) to measure any movement at all.
        """
        history = self._recent_window(track.position_history)
        if len(history) < 2:
            return None

        prev, curr = history[-2], history[-1]
        displacement = compute_displacement(prev, curr)
        direction = compute_direction_deg(prev, curr)
        speed_frame = compute_speed_px_per_frame(prev, curr)
        speed_sec = compute_speed_px_per_sec(prev, curr)

        previous_speed_frame: Optional[float] = None
        previous_direction: Optional[float] = None
        if len(history) >= 3:
            prev2, prev1 = history[-3], history[-2]
            previous_speed_frame = compute_speed_px_per_frame(prev2, prev1)
            previous_direction = compute_direction_deg(prev2, prev1)

        speed_change = compute_speed_change(previous_speed_frame, speed_frame)
        direction_change = angular_difference_deg(previous_direction, direction)

        is_stationary = speed_frame <= self._stationary_speed_px
        # Direction is noisy/meaningless at near-zero speed (a couple of
        # pixels of detection/Kalman jitter can swing the heading by 90+
        # degrees for no real reason) — only trust a direction-change flag
        # when the object is actually moving meaningfully in both segments.
        is_sudden_direction_change = (
            direction_change is not None
            and direction_change >= self._sudden_direction_change_deg
            and speed_frame > self._stationary_speed_px
            and (previous_speed_frame or 0.0) > self._stationary_speed_px
        )

        is_sudden_deceleration = False
        if previous_speed_frame is not None and previous_speed_frame > 0 and speed_change is not None:
            drop_ratio = -speed_change / previous_speed_frame  # positive when decelerating
            is_sudden_deceleration = drop_ratio >= self._sudden_deceleration_ratio

        # "Sudden stop": the object was meaningfully moving in the previous
        # segment and has now dropped to (near-)stationary — distinct from
        # simply having been stationary for a while.
        is_sudden_stop = (
            is_stationary
            and previous_speed_frame is not None
            and previous_speed_frame > self._stationary_speed_px
        )

        state = MotionState(
            track_id=track.track_id,
            frame_index=curr.frame_index,
            center=curr.center,
            displacement_px=displacement,
            direction_deg=direction,
            speed_px_per_frame=speed_frame,
            speed_px_per_sec=speed_sec,
            previous_speed_px_per_frame=previous_speed_frame,
            speed_change_px_per_frame=speed_change,
            direction_change_deg=direction_change,
            is_stationary=is_stationary,
            is_sudden_deceleration=is_sudden_deceleration,
            is_sudden_direction_change=is_sudden_direction_change,
            is_sudden_stop=is_sudden_stop,
            sample_count=len(history),
        )

        if is_sudden_stop:
            logger.info("Sudden stop detected: track_id=%d frame=%d", track.track_id, curr.frame_index)
        if is_sudden_direction_change:
            logger.info(
                "Sudden direction change detected: track_id=%d frame=%d delta=%.1f deg",
                track.track_id, curr.frame_index, direction_change,
            )

        return state

    def analyze_many(self, tracks: Sequence[TrackedObject]) -> Dict[int, MotionState]:
        """Analyze a batch of tracked objects; entries with insufficient history are omitted."""
        results: Dict[int, MotionState] = {}
        for track in tracks:
            state = self.analyze(track)
            if state is not None:
                results[track.track_id] = state
        return results

    def _recent_window(self, history) -> List[PositionSample]:
        samples = list(history)
        if self._window > 0:
            samples = samples[-self._window:]
        return samples
