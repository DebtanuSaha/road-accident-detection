"""
engine/evidence/evidence_recorder.py

Phase 9 — Evidence Capture.

Once an accident is confirmed (Phase 7/8), automatically saves — per
the project spec's list:
  - Accident frame          (the raw frame at the moment of confirmation)
  - Annotated frame         (that same frame, with boxes/labels drawn)
  - Short video clip        (pre_event_seconds before -> post_event_seconds
                             after, assembled from RAW frames — see note below)
  - Timestamp, camera ID, detection info, tracking info, accident score
    (all captured into one structured `EvidenceRecord`)

DESIGN NOTE — why the video clip uses RAW frames, not annotated ones:
"Annotated frame" is already a separate, single-frame artifact in the
spec's own list. Evidence whose purpose may eventually include human
review, insurance, or legal use is more defensible as close to
unmodified source footage as possible — so the clip is built from the
same raw frames the video source produced, not frames with bounding
boxes/scores burned in. The annotated frame remains available
separately for a quick-glance visual summary.

CLOUD-STORAGE READINESS: every artifact here is written to a stable
local path and recorded on the returned `EvidenceRecord`. Nothing
about this design assumes local disk is the final destination — a
future `CloudUploader` (out of scope for this project per its own
"Future Extensions" list) could read these exact paths, upload them,
and replace them with URLs on the same record without any change to
how evidence is captured here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
from pydantic import BaseModel

from config.settings import Settings
from config.settings import settings as default_settings
from engine.detection import Detection
from engine.evidence.frame_buffer import FrameSnapshot, RollingFrameBuffer
from engine.events import AccidentEvent
from engine.tracking import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceRecord(BaseModel):
    """
    Structured record of everything captured for one confirmed
    accident — the project spec's evidence checklist as one validated
    object.
    """

    event_id: str
    timestamp: datetime
    camera_id: str
    accident_probability: float

    detections: List[dict]  # structured detection info at the confirming frame
    tracked_objects: List[dict]  # structured tracking info at the confirming frame

    accident_frame_path: str
    annotated_frame_path: Optional[str] = None
    evidence_video_path: Optional[str] = None  # filled in once the clip is finalized

    pre_event_frames_captured: int
    post_event_frames_captured: int = 0
    clip_truncated: bool = False  # True if the video ended before post_event_seconds was reached


@dataclass
class _PendingClip:
    """Internal bookkeeping while a clip is still collecting its 'after' frames."""

    pre_frames: List[FrameSnapshot]
    confirming_frame: FrameSnapshot  # the exact frame at which the accident was confirmed
    frames_needed_after: int
    record: EvidenceRecord
    post_frames: List[FrameSnapshot] = field(default_factory=list)


class EvidenceRecorder:
    """
    Usage (call once per frame, after Phase 8's EventBuilder):
        recorder = EvidenceRecorder(settings)

        for each frame:
            ...
            if new_event is not None:
                recorder.start_capture(new_event, buffer, current_snapshot, detections, tracked_objects)
            finalized = recorder.update(current_snapshot)  # feeds any pending clips

        # once the video source is exhausted:
        finalized += recorder.finalize_all()
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._pending: Dict[str, _PendingClip] = {}

    def start_capture(
        self,
        event: AccidentEvent,
        buffer: RollingFrameBuffer,
        current: FrameSnapshot,
        detections: Sequence[Detection],
        tracked_objects: Sequence[TrackedObject],
    ) -> EvidenceRecord:
        """
        Immediately saves the accident frame + annotated frame (if
        available) and begins collecting the post-event portion of the
        video clip. Returns the (not-yet-finalized) `EvidenceRecord` —
        `evidence_video_path` is still `None` at this point.
        """
        accident_frame_path = self._save_image(current.frame, f"{event.event_id}_frame.jpg")

        annotated_frame_path = None
        if current.annotated_frame is not None:
            annotated_frame_path = self._save_image(current.annotated_frame, f"{event.event_id}_annotated.jpg")

        pre_frames = buffer.snapshot_recent()
        frames_needed_after = max(1, round(buffer.post_event_seconds * buffer.fps))

        record = EvidenceRecord(
            event_id=event.event_id,
            timestamp=datetime.now(timezone.utc),
            camera_id=event.camera_id,
            accident_probability=event.accident_probability,
            detections=[d.to_dict() for d in detections],
            tracked_objects=[t.to_dict() for t in tracked_objects],
            accident_frame_path=accident_frame_path,
            annotated_frame_path=annotated_frame_path,
            pre_event_frames_captured=len(pre_frames),
        )

        self._pending[event.event_id] = _PendingClip(
            pre_frames=pre_frames,
            confirming_frame=current,
            frames_needed_after=frames_needed_after,
            record=record,
        )

        logger.info(
            "Evidence capture started: event_id=%s pre_frames=%d (need %d post_frames) "
            "accident_frame=%s annotated_frame=%s",
            event.event_id, len(pre_frames), frames_needed_after,
            accident_frame_path, annotated_frame_path,
        )
        return record

    def update(self, snapshot: FrameSnapshot) -> List[EvidenceRecord]:
        """
        Call once per frame, every frame, regardless of whether a clip
        is pending — feeds the current frame to any clips still
        collecting their post-event window. Returns the list of
        records that were finalized (their video clip fully written)
        during this call; usually empty.
        """
        finalized: List[EvidenceRecord] = []
        for event_id in list(self._pending.keys()):
            pending = self._pending[event_id]
            pending.post_frames.append(snapshot)
            if len(pending.post_frames) >= pending.frames_needed_after:
                finalized.append(self._finalize(event_id, pending, truncated=False))
        return finalized

    def finalize_all(self) -> List[EvidenceRecord]:
        """
        Flush every still-pending clip with whatever post-event frames
        were collected so far — call this once the video source is
        exhausted, so an accident near the end of a video still gets a
        (possibly shorter) clip instead of silently losing it.
        """
        finalized: List[EvidenceRecord] = []
        for event_id in list(self._pending.keys()):
            pending = self._pending[event_id]
            truncated = len(pending.post_frames) < pending.frames_needed_after
            finalized.append(self._finalize(event_id, pending, truncated=truncated))
        return finalized

    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()

    # -- internals ------------------------------------------------------------------

    def _finalize(self, event_id: str, pending: _PendingClip, truncated: bool) -> EvidenceRecord:
        del self._pending[event_id]

        # The clip always includes the exact confirming frame — without
        # this, [pre_frames] + [post_frames] would silently skip the one
        # frame the accident was actually confirmed on.
        all_frames = pending.pre_frames + [pending.confirming_frame] + pending.post_frames
        fps = self._infer_fps(all_frames)
        video_path = self._save_clip(all_frames, f"{event_id}_clip.mp4", fps=fps)

        pending.record.evidence_video_path = video_path
        pending.record.post_event_frames_captured = len(pending.post_frames)
        pending.record.clip_truncated = truncated

        if truncated:
            logger.warning(
                "Evidence clip for %s truncated: video ended before the full "
                "post-event window was captured (%d/%d post-frames).",
                event_id, len(pending.post_frames), pending.frames_needed_after,
            )

        logger.info(
            "Evidence capture finalized: event_id=%s clip=%s total_frames=%d truncated=%s",
            event_id, video_path, len(all_frames), truncated,
        )
        return pending.record

    def _save_image(self, frame, filename: str) -> str:
        path = Path(self._cfg.EVIDENCE_IMAGE_DIR) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return str(path)

    def _save_clip(self, frames: Sequence[FrameSnapshot], filename: str, fps: float) -> str:
        path = Path(self._cfg.EVIDENCE_VIDEO_DIR) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0].frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for snap in frames:
                writer.write(snap.frame)
        finally:
            writer.release()
        return str(path)

    @staticmethod
    def _infer_fps(frames: Sequence[FrameSnapshot], default: float = 25.0) -> float:
        """Estimate a playback FPS for the saved clip from consecutive frame timestamps."""
        if len(frames) < 2:
            return default
        deltas_ms = [
            b.timestamp_ms - a.timestamp_ms
            for a, b in zip(frames, frames[1:])
            if b.timestamp_ms > a.timestamp_ms
        ]
        if not deltas_ms:
            return default
        avg_delta_ms = sum(deltas_ms) / len(deltas_ms)
        return 1000.0 / avg_delta_ms if avg_delta_ms > 0 else default
