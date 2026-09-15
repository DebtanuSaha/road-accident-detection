"""
engine/tracking/byte_tracker.py

Phase 4 — Multi-Object Tracking (ByteTrack integration).

Wraps the real ByteTrack algorithm (the implementation bundled with
Ultralytics, `ultralytics.trackers.byte_tracker.BYTETracker` — the
same association/Kalman-filter code used by `yolo.track()`) so it can
be fed our own `engine.detection.Detection` objects directly, without
going through `model.track()`. This keeps detection (Phase 3) and
tracking (Phase 4) as separate, independently testable stages, per the
project architecture.

This module's only responsibility is: given one frame's detections,
return each one's persistent track ID. It does NOT maintain multi-
frame history, velocity, or lifecycle bookkeeping — that is
`track_manager.py`'s job. Keeping this split means the ByteTrack
integration can be swapped or upgraded without touching the state
management logic, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from config.settings import Settings
from config.settings import settings as default_settings
from engine.detection import BoundingBox, Detection
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawTrack:
    """
    One tracked object's state for the current frame, as produced by
    ByteTrack. "Raw" because it carries no multi-frame history —
    `track_manager.TrackManager` turns a sequence of these into stateful
    `TrackedObject`s.
    """

    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox


class _DetectionsForTracker:
    """
    Minimal adapter exposing the `.xywh` / `.conf` / `.cls` attributes,
    `len()`, and boolean-mask `__getitem__` that Ultralytics' BYTETracker
    expects from a "Results-like" object — without depending on
    Ultralytics' actual `Results` class (which is built from a live
    model prediction, not from arbitrary detections we already have).
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetectionsForTracker":
        return _DetectionsForTracker(self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrackWrapper:
    """
    Config-driven wrapper around Ultralytics' BYTETracker.

    Usage:
        tracker = ByteTrackWrapper(settings)
        raw_tracks = tracker.update(detections, frame)  # call once per frame, every frame
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=cfg.TRACKER_TRACK_HIGH_THRESH,
            track_low_thresh=cfg.TRACKER_TRACK_LOW_THRESH,
            new_track_thresh=cfg.TRACKER_NEW_TRACK_THRESH,
            track_buffer=cfg.TRACKER_MAX_MISSED_FRAMES,
            match_thresh=cfg.TRACKER_IOU_THRESHOLD,
            fuse_score=cfg.TRACKER_FUSE_SCORE,
        )
        self._tracker = BYTETracker(self._args)
        logger.info(
            "ByteTrackWrapper initialized: track_buffer=%d match_thresh=%.2f "
            "high_thresh=%.2f low_thresh=%.2f new_track_thresh=%.2f fuse_score=%s",
            self._args.track_buffer, self._args.match_thresh,
            self._args.track_high_thresh, self._args.track_low_thresh,
            self._args.new_track_thresh, self._args.fuse_score,
        )

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[RawTrack]:
        """
        Advance the tracker by exactly one frame.

        Must be called once per frame, every frame — including frames
        with zero detections — so ByteTrack's internal Kalman
        prediction and lost-track aging stay correct. Returns only
        tracks matched (or newly created) in this call; tracks that
        temporarily have no matching detection are held internally
        (up to `TRACKER_MAX_MISSED_FRAMES`) and simply do not appear
        in the returned list until they're matched again or expire.
        """
        n = len(detections)
        if n == 0:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            xywh = np.array(
                [
                    [
                        (d.bbox.x1 + d.bbox.x2) / 2.0,
                        (d.bbox.y1 + d.bbox.y2) / 2.0,
                        d.bbox.width,
                        d.bbox.height,
                    ]
                    for d in detections
                ],
                dtype=np.float32,
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            cls = np.array([float(d.class_id) for d in detections], dtype=np.float32)

        adapter = _DetectionsForTracker(xywh=xywh, conf=conf, cls=cls)
        raw_output = self._tracker.update(adapter, img=frame)

        raw_tracks: List[RawTrack] = []
        for row in raw_output:
            x1, y1, x2, y2, track_id, score, cls_id, idx = row
            idx = int(idx)
            # `idx` indexes back into the exact `detections` list we just
            # passed in (guaranteed by BYTETracker: every row returned by
            # _format_output() this call came from this call's input).
            class_name = detections[idx].class_name if 0 <= idx < n else str(int(cls_id))
            raw_tracks.append(
                RawTrack(
                    track_id=int(track_id),
                    class_name=class_name,
                    class_id=int(cls_id),
                    confidence=float(score),
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return raw_tracks

    def reset(self) -> None:
        """
        Clear all tracked/lost/removed tracks and reset the ID counter.

        Call this between independent video sources (e.g. between test
        cases, or when starting a new run against a different camera)
        so track IDs don't carry over from an unrelated video.
        """
        self._tracker.reset()
        logger.info("ByteTrackWrapper reset — all tracks cleared, ID counter reset.")
