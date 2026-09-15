"""
engine/evidence package

Phase 9 — Evidence Capture.

Exposes:
- `frame_buffer.py` — `FrameSnapshot`, `RollingFrameBuffer`: keeps
  recent raw frames so evidence can include footage from BEFORE a
  confirmed accident, not just after.
- `evidence_recorder.py` — `EvidenceRecorder`, `EvidenceRecord`:
  saves the accident frame, annotated frame, and a before/after video
  clip, plus structured detection/tracking/score metadata, once an
  accident is confirmed.
"""

from engine.evidence.evidence_recorder import EvidenceRecord, EvidenceRecorder
from engine.evidence.frame_buffer import FrameSnapshot, RollingFrameBuffer

__all__ = [
    "FrameSnapshot",
    "RollingFrameBuffer",
    "EvidenceRecord",
    "EvidenceRecorder",
]
