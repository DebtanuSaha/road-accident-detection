"""
engine/evidence/frame_buffer.py

Phase 9 — Evidence Capture: rolling frame buffer.

Keeps the last `evidence.pre_event_seconds` worth of RAW frames (plus
each frame's already-annotated counterpart, if the caller has one) so
that when an accident is confirmed, evidence capture can reach BACK in
time and include footage from before the confirming frame — something
that's otherwise impossible once a frame has already been discarded by
the video-input loop.

Deliberately holds only frame IMAGES here, not detection/tracking
metadata — that heavier, structured information is only needed once,
at the moment of confirmation (see `evidence_recorder.py`), not for
every buffered frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameSnapshot:
    """One frame, captured for possible use as evidence."""

    frame_index: int
    timestamp_ms: float
    frame: np.ndarray  # raw, unannotated — the actual footage
    annotated_frame: Optional[np.ndarray] = None  # optional visualization overlay


class RollingFrameBuffer:
    """
    Fixed-length ring buffer of the most recent frames, sized from
    `evidence.pre_event_seconds` and the video source's actual FPS
    (must be known at construction — pass the source's `get_fps()`,
    falling back to a sane default if the source can't report one).

    Usage:
        buffer = RollingFrameBuffer(fps=source.get_fps() or 25.0, cfg=settings)
        for frame in frames:
            buffer.add(FrameSnapshot(...))
            ...
        # at the moment of accident confirmation:
        pre_event_frames = buffer.snapshot_recent()
    """

    def __init__(self, fps: float, cfg: Settings = default_settings):
        self._cfg = cfg
        evidence_cfg = load_thresholds()["evidence"]
        self._pre_event_seconds: float = evidence_cfg["pre_event_seconds"]
        self._post_event_seconds: float = evidence_cfg["post_event_seconds"]
        self._fps = fps if fps and fps > 0 else 25.0

        max_len = max(1, round(self._pre_event_seconds * self._fps))
        self._buffer: Deque[FrameSnapshot] = deque(maxlen=max_len)

        logger.info(
            "RollingFrameBuffer initialized: fps=%.2f pre_event_seconds=%.1f "
            "(buffer holds up to %d frames), post_event_seconds=%.1f",
            self._fps, self._pre_event_seconds, max_len, self._post_event_seconds,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def pre_event_seconds(self) -> float:
        return self._pre_event_seconds

    @property
    def post_event_seconds(self) -> float:
        return self._post_event_seconds

    @property
    def max_length(self) -> int:
        return self._buffer.maxlen

    def add(self, snapshot: FrameSnapshot) -> None:
        """Append a frame. Oldest frame is dropped automatically once full."""
        self._buffer.append(snapshot)

    def snapshot_recent(self) -> List[FrameSnapshot]:
        """
        A copy of everything currently buffered, oldest first. Safe to
        keep after this call even as the buffer keeps evolving —
        the objects returned are not affected by future `add()` calls
        (only the deque itself is mutated, not the FrameSnapshot
        instances already handed out).
        """
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
