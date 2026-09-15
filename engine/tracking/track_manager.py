"""
engine/tracking/track_manager.py

Phase 4 — Multi-Object Tracking (per-track state management).

Turns the per-frame `RawTrack` list produced by `byte_tracker.py` into
stateful `TrackedObject`s: persistent per-object history (bounded),
frame-to-frame velocity/direction estimates, track duration, and
entering/leaving/temporary-failure lifecycle handling.

Deliberately independent of accident/collision logic (per project
requirement "keep tracking independent from accident logic") — this
module only describes what each tracked object IS DOING, never
whether that constitutes a collision or accident.

IMPORTANT: velocity/direction here are PIXEL-SPACE estimates
(px/frame), not real-world speed. Converting to km/h requires camera
calibration, which is out of scope for this phase — see
`engine/motion/` (Phase 5) for the fuller motion-analysis layer built
on top of this history.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from engine.detection import BoundingBox
from engine.tracking.byte_tracker import RawTrack
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSample:
    """One historical observation of a tracked object's position."""

    frame_index: int
    timestamp_ms: float
    center: Tuple[float, float]


@dataclass
class TrackedObject:
    """
    Full, stateful record of one tracked road user across frames.

    `is_active` is True only for frames where this object was actually
    matched to a detection this frame; when a track is temporarily
    unmatched (occlusion, missed detection) it stays in the manager
    with `is_active=False` and a growing `missed_frames` count, rather
    than being deleted immediately — this is the "temporary detection
    failure" tolerance required by the project spec.
    """

    track_id: int
    class_name: str
    class_id: int

    bbox: BoundingBox
    confidence: float

    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp_ms: float
    last_seen_timestamp_ms: float

    hit_count: int = 1  # number of frames this track has been successfully matched
    missed_frames: int = 0  # consecutive frames since the last match
    is_active: bool = True  # matched to a detection in the current frame?
    confirmed: bool = False  # hit_count has reached TRACKER_MIN_HITS

    previous_center: Optional[Tuple[float, float]] = None
    velocity_px_per_frame: Tuple[float, float] = (0.0, 0.0)  # (vx, vy), pixel space only
    speed_px_per_frame: float = 0.0
    direction_deg: Optional[float] = None  # heading in image-space degrees; None until movement is observed

    position_history: Deque[PositionSample] = field(default_factory=deque)

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center

    @property
    def track_duration_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1

    @property
    def track_duration_seconds(self) -> float:
        return max(0.0, (self.last_seen_timestamp_ms - self.first_seen_timestamp_ms) / 1000.0)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox.as_tuple()],
            "center": [round(v, 2) for v in self.center],
            "speed_px_per_frame": round(self.speed_px_per_frame, 3),
            "direction_deg": round(self.direction_deg, 1) if self.direction_deg is not None else None,
            "track_duration_frames": self.track_duration_frames,
            "hit_count": self.hit_count,
            "missed_frames": self.missed_frames,
            "is_active": self.is_active,
            "confirmed": self.confirmed,
        }


class TrackManager:
    """
    Owns the full set of currently-known tracked objects across frames.

    Call `update(raw_tracks, frame_index, timestamp_ms)` once per
    frame (every frame, including frames with zero raw tracks) to
    advance all state. Returns the list of `TrackedObject`s considered
    part of the scene right now (active this frame, or still within
    their missed-frame grace period).
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._tracks: Dict[int, TrackedObject] = {}
        self._history_length = cfg.TRACK_HISTORY_LENGTH
        self._min_hits = cfg.TRACKER_MIN_HITS
        self._max_missed_frames = cfg.TRACKER_MAX_MISSED_FRAMES

    def update(
        self,
        raw_tracks: List[RawTrack],
        frame_index: int,
        timestamp_ms: float,
    ) -> List[TrackedObject]:
        matched_ids = set()

        for raw in raw_tracks:
            matched_ids.add(raw.track_id)
            if raw.track_id in self._tracks:
                self._update_existing(raw, frame_index, timestamp_ms)
            else:
                self._create_new(raw, frame_index, timestamp_ms)

        self._age_unmatched(matched_ids, frame_index)
        self._evict_expired()

        return list(self._tracks.values())

    def get_active_tracks(self) -> List[TrackedObject]:
        """Tracks matched to a detection in the most recently processed frame."""
        return [t for t in self._tracks.values() if t.is_active]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Drop all tracked state. Does not reset the underlying ByteTrack ID counter."""
        self._tracks.clear()

    # -- internals ------------------------------------------------------------------

    def _create_new(self, raw: RawTrack, frame_index: int, timestamp_ms: float) -> None:
        track = TrackedObject(
            track_id=raw.track_id,
            class_name=raw.class_name,
            class_id=raw.class_id,
            bbox=raw.bbox,
            confidence=raw.confidence,
            first_seen_frame=frame_index,
            last_seen_frame=frame_index,
            first_seen_timestamp_ms=timestamp_ms,
            last_seen_timestamp_ms=timestamp_ms,
            hit_count=1,
            missed_frames=0,
            is_active=True,
            confirmed=self._min_hits <= 1,
        )
        track.position_history.append(PositionSample(frame_index, timestamp_ms, raw.bbox.center))
        self._tracks[raw.track_id] = track
        logger.info(
            "Track ENTERED frame: id=%d class=%s at frame=%d",
            raw.track_id, raw.class_name, frame_index,
        )

    def _update_existing(self, raw: RawTrack, frame_index: int, timestamp_ms: float) -> None:
        track = self._tracks[raw.track_id]

        frames_elapsed = max(1, frame_index - track.last_seen_frame)
        prev_center = track.center
        new_center = raw.bbox.center

        dx = new_center[0] - prev_center[0]
        dy = new_center[1] - prev_center[1]
        vx = dx / frames_elapsed
        vy = dy / frames_elapsed
        speed = math.hypot(vx, vy)

        # Direction is undefined (None) if the object hasn't moved a
        # meaningful amount — avoids noisy headings from near-zero jitter.
        direction_deg: Optional[float] = None
        if speed > 1e-3:
            direction_deg = math.degrees(math.atan2(dy, dx)) % 360.0

        was_missed = track.missed_frames > 0
        track.previous_center = prev_center
        track.bbox = raw.bbox
        track.confidence = raw.confidence
        track.class_name = raw.class_name
        track.class_id = raw.class_id
        track.velocity_px_per_frame = (vx, vy)
        track.speed_px_per_frame = speed
        track.direction_deg = direction_deg
        track.last_seen_frame = frame_index
        track.last_seen_timestamp_ms = timestamp_ms
        track.hit_count += 1
        track.missed_frames = 0
        track.is_active = True
        if not track.confirmed and track.hit_count >= self._min_hits:
            track.confirmed = True
            logger.info("Track CONFIRMED: id=%d class=%s (hit_count=%d)", track.track_id, track.class_name, track.hit_count)

        track.position_history.append(PositionSample(frame_index, timestamp_ms, new_center))
        while len(track.position_history) > self._history_length:
            track.position_history.popleft()

        if was_missed:
            logger.info(
                "Track RE-ACQUIRED after temporary loss: id=%d class=%s",
                track.track_id, track.class_name,
            )

    def _age_unmatched(self, matched_ids: set, frame_index: int) -> None:
        for track_id, track in self._tracks.items():
            if track_id in matched_ids:
                continue
            track.is_active = False
            track.missed_frames += 1
            if track.missed_frames == 1:
                logger.debug("Track temporarily unmatched: id=%d class=%s", track_id, track.class_name)

    def _evict_expired(self) -> None:
        expired = [
            track_id for track_id, track in self._tracks.items()
            if track.missed_frames > self._max_missed_frames
        ]
        for track_id in expired:
            track = self._tracks.pop(track_id)
            logger.info(
                "Track EXITED (expired after %d missed frames): id=%d class=%s duration_frames=%d",
                self._max_missed_frames, track_id, track.class_name, track.track_duration_frames,
            )
