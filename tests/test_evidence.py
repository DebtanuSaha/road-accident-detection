"""
tests/test_evidence.py

Phase 9 tests for evidence capture: the rolling frame buffer and the
evidence recorder (accident frame, annotated frame, before/after video
clip, and structured metadata).
"""

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.detection import BoundingBox, Detection  # noqa: E402
from engine.tracking import PositionSample, TrackedObject  # noqa: E402
from engine.events import AccidentEvent, ObjectInvolved  # noqa: E402
from engine.evidence import EvidenceRecorder, FrameSnapshot, RollingFrameBuffer  # noqa: E402


def make_frame(value=0, size=(60, 80)):
    h, w = size
    return np.full((h, w, 3), value % 255, dtype=np.uint8)


def make_event(event_id="ACC-000001", probability=0.8):
    return AccidentEvent(
        event_id=event_id,
        accident_probability=probability,
        objects_involved=[ObjectInvolved(track_id=1, class_name="car")],
        camera_id="CAM-001",
    )


def make_detection():
    return Detection(class_name="car", class_id=2, confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))


def make_tracked_object(frame_index=0):
    bbox = BoundingBox(0, 0, 10, 10)
    return TrackedObject(
        track_id=1, class_name="car", class_id=2, bbox=bbox, confidence=0.9,
        first_seen_frame=0, last_seen_frame=frame_index,
        first_seen_timestamp_ms=0.0, last_seen_timestamp_ms=frame_index * 100.0,
        position_history=deque([PositionSample(frame_index, frame_index * 100.0, bbox.center)]),
    )


@pytest.fixture
def cfg(tmp_path):
    return Settings(
        EVIDENCE_IMAGE_DIR=str(tmp_path / "images"),
        EVIDENCE_VIDEO_DIR=str(tmp_path / "videos"),
    )


# ---------------------------------------------------------------------------
# RollingFrameBuffer
# ---------------------------------------------------------------------------

def test_buffer_length_derived_from_fps_and_pre_event_seconds(cfg):
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    assert buffer.max_length == round(buffer.pre_event_seconds * 10.0)


def test_buffer_falls_back_to_default_fps_when_unreliable(cfg):
    buffer = RollingFrameBuffer(fps=0.0, cfg=cfg)
    assert buffer.fps == 25.0  # sane fallback, not zero/negative


def test_buffer_evicts_oldest_frame_once_full(cfg):
    buffer = RollingFrameBuffer(fps=2.0, cfg=cfg)  # small buffer for a fast test
    for i in range(buffer.max_length + 3):
        buffer.add(FrameSnapshot(frame_index=i, timestamp_ms=i * 100.0, frame=make_frame(i)))
    recent = buffer.snapshot_recent()
    assert len(recent) == buffer.max_length
    assert recent[0].frame_index == 3  # the first 3 were evicted
    assert recent[-1].frame_index == buffer.max_length + 2


def test_buffer_snapshot_is_independent_copy(cfg):
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    buffer.add(FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0)))
    snap1 = buffer.snapshot_recent()
    buffer.add(FrameSnapshot(frame_index=1, timestamp_ms=100.0, frame=make_frame(1)))
    assert len(snap1) == 1  # unaffected by the later add()


def test_buffer_len_reflects_current_contents(cfg):
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    assert len(buffer) == 0
    buffer.add(FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0)))
    assert len(buffer) == 1


# ---------------------------------------------------------------------------
# EvidenceRecorder — start_capture
# ---------------------------------------------------------------------------

def test_start_capture_saves_accident_frame(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(1))

    record = recorder.start_capture(make_event(), buffer, current, [make_detection()], [make_tracked_object()])

    assert Path(record.accident_frame_path).is_file()
    assert record.annotated_frame_path is None  # no annotated frame provided


def test_start_capture_saves_annotated_frame_when_provided(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(1), annotated_frame=make_frame(2))

    record = recorder.start_capture(make_event(), buffer, current, [], [])

    assert record.annotated_frame_path is not None
    assert Path(record.annotated_frame_path).is_file()


def test_start_capture_captures_pre_event_frames_from_buffer(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    for i in range(7):
        buffer.add(FrameSnapshot(frame_index=i, timestamp_ms=i * 100.0, frame=make_frame(i)))
    current = FrameSnapshot(frame_index=7, timestamp_ms=700.0, frame=make_frame(7))

    record = recorder.start_capture(make_event(), buffer, current, [], [])
    assert record.pre_event_frames_captured == 7


def test_start_capture_records_detection_and_tracking_info(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0))

    record = recorder.start_capture(
        make_event(probability=0.77), buffer, current, [make_detection()], [make_tracked_object()],
    )
    assert len(record.detections) == 1
    assert record.detections[0]["class"] == "car"
    assert len(record.tracked_objects) == 1
    assert record.tracked_objects[0]["track_id"] == 1
    assert record.accident_probability == pytest.approx(0.77)
    assert record.camera_id == "CAM-001"


def test_start_capture_video_path_is_none_until_finalized(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0))
    record = recorder.start_capture(make_event(), buffer, current, [], [])
    assert record.evidence_video_path is None


# ---------------------------------------------------------------------------
# EvidenceRecorder — update / finalize
# ---------------------------------------------------------------------------

def test_clip_finalizes_once_enough_post_frames_collected(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)  # post_event_seconds=5 -> needs 50 frames
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0))
    recorder.start_capture(make_event(), buffer, current, [], [])

    finalized = []
    for i in range(1, 50):
        finalized += recorder.update(FrameSnapshot(frame_index=i, timestamp_ms=i * 100.0, frame=make_frame(i)))
    assert finalized == []  # 49 frames, not yet enough

    finalized += recorder.update(FrameSnapshot(frame_index=50, timestamp_ms=5000.0, frame=make_frame(50)))
    assert len(finalized) == 1
    record = finalized[0]
    assert record.clip_truncated is False
    assert record.post_event_frames_captured == 50
    assert Path(record.evidence_video_path).is_file()


def test_finalized_clip_contains_pre_and_post_frames(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=5.0, cfg=cfg)  # smaller buffer, faster test (25 post-frames needed)
    for i in range(10):
        buffer.add(FrameSnapshot(frame_index=i, timestamp_ms=i * 200.0, frame=make_frame(i)))
    current = FrameSnapshot(frame_index=10, timestamp_ms=2000.0, frame=make_frame(10))
    recorder.start_capture(make_event(), buffer, current, [], [])

    finalized = []
    for i in range(11, 36):
        finalized += recorder.update(FrameSnapshot(frame_index=i, timestamp_ms=i * 200.0, frame=make_frame(i)))

    assert len(finalized) == 1
    record = finalized[0]
    cap = cv2.VideoCapture(record.evidence_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # +1 for the confirming frame itself, included alongside pre/post frames.
    assert total_frames == record.pre_event_frames_captured + 1 + record.post_event_frames_captured


def test_finalize_all_flushes_pending_clip_with_truncation_flag(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)  # needs 50 post-frames
    current = FrameSnapshot(frame_index=0, timestamp_ms=0.0, frame=make_frame(0))
    recorder.start_capture(make_event(), buffer, current, [], [])

    for i in range(1, 6):  # only 5 post-frames before "video ends"
        recorder.update(FrameSnapshot(frame_index=i, timestamp_ms=i * 100.0, frame=make_frame(i)))

    finalized = recorder.finalize_all()
    assert len(finalized) == 1
    assert finalized[0].clip_truncated is True
    assert finalized[0].post_event_frames_captured == 5
    assert Path(finalized[0].evidence_video_path).is_file()


def test_finalize_all_with_no_pending_clips_returns_empty(cfg):
    recorder = EvidenceRecorder(cfg)
    assert recorder.finalize_all() == []


def test_multiple_independent_pending_clips(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)

    recorder.start_capture(make_event("ACC-000001"), buffer, FrameSnapshot(0, 0.0, make_frame(0)), [], [])
    recorder.start_capture(make_event("ACC-000002"), buffer, FrameSnapshot(1, 100.0, make_frame(1)), [], [])
    assert recorder.pending_count() == 2

    finalized = recorder.finalize_all()
    ids = {r.event_id for r in finalized}
    assert ids == {"ACC-000001", "ACC-000002"}
    assert recorder.pending_count() == 0


def test_reset_clears_pending_clips(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    recorder.start_capture(make_event(), buffer, FrameSnapshot(0, 0.0, make_frame(0)), [], [])
    assert recorder.pending_count() == 1
    recorder.reset()
    assert recorder.pending_count() == 0


def test_evidence_record_is_json_serializable(cfg):
    recorder = EvidenceRecorder(cfg)
    buffer = RollingFrameBuffer(fps=10.0, cfg=cfg)
    record = recorder.start_capture(
        make_event(), buffer, FrameSnapshot(0, 0.0, make_frame(0)), [make_detection()], [make_tracked_object()],
    )
    dumped = record.model_dump(mode="json")
    assert dumped["event_id"] == "ACC-000001"
    assert isinstance(dumped["detections"], list)
