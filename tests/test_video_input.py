"""
tests/test_video_input.py

Phase 2 tests for the video input layer.

A tiny synthetic video file is generated on the fly (a handful of solid
colour frames written with cv2.VideoWriter) so these tests are fully
self-contained and don't depend on any external sample footage,
webcam, or network stream. Webcam/RTSP sources are covered by
interface/error-path tests only, since no real device/stream is
available in a CI or sandbox environment.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from engine.input import create_video_source  # noqa: E402
from engine.input.video_source import FileVideoSource, VideoSourceError  # noqa: E402
from engine.input.webcam_source import WebcamVideoSource  # noqa: E402
from engine.input.rtsp_source import RTSPVideoSource  # noqa: E402

FRAME_SIZE = (64, 48)  # width, height
NUM_FRAMES = 20
FPS = 10.0


@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory) -> str:
    """Create a short synthetic .mp4 file and return its path."""
    tmp_dir = tmp_path_factory.mktemp("video_input_fixtures")
    path = str(tmp_dir / "synthetic.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)
    for i in range(NUM_FRAMES):
        # Each frame is a solid colour that shifts slightly, so frames are distinguishable.
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), fill_value=(i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# FileVideoSource
# ---------------------------------------------------------------------------

def test_file_video_source_opens_and_reports_metadata(synthetic_video_path):
    source = FileVideoSource(path=synthetic_video_path)
    assert source.open() is True
    assert source.is_opened() is True

    width, height = source.get_frame_size()
    assert (width, height) == FRAME_SIZE
    assert source.get_fps() > 0

    source.release()
    assert source.is_opened() is False


def test_file_video_source_reads_all_frames(synthetic_video_path):
    with FileVideoSource(path=synthetic_video_path) as source:
        read_count = 0
        while True:
            result = source.read()
            if not result.success:
                break
            assert result.frame is not None
            assert result.frame.shape[1::-1] == FRAME_SIZE
            read_count += 1

        assert read_count == NUM_FRAMES


def test_file_video_source_frame_skip(synthetic_video_path):
    # frame_skip=1 -> only every 2nd frame is returned.
    with FileVideoSource(path=synthetic_video_path, frame_skip=1) as source:
        read_count = 0
        while True:
            result = source.read()
            if not result.success:
                break
            read_count += 1

        assert read_count == NUM_FRAMES // 2


def test_file_video_source_missing_file_fails_gracefully():
    source = FileVideoSource(path="/nonexistent/path/video.mp4")
    assert source.open() is False
    assert source.is_opened() is False
    # release() must be safe even though open() never succeeded.
    source.release()


def test_file_video_source_context_manager_raises_on_missing_file():
    with pytest.raises(VideoSourceError):
        with FileVideoSource(path="/nonexistent/path/video.mp4"):
            pass


def test_file_video_source_rejects_negative_frame_skip():
    with pytest.raises(ValueError):
        FileVideoSource(path="irrelevant.mp4", frame_skip=-1)


def test_file_video_source_timestamps_increase(synthetic_video_path):
    with FileVideoSource(path=synthetic_video_path) as source:
        first = source.read()
        second = source.read()
        assert first.success and second.success
        assert second.timestamp_ms > first.timestamp_ms
        assert second.frame_index == first.frame_index + 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_create_video_source_file(synthetic_video_path):
    cfg = Settings(VIDEO_SOURCE_TYPE="file", VIDEO_SOURCE_PATH=synthetic_video_path)
    source = create_video_source(cfg)
    assert isinstance(source, FileVideoSource)


def test_create_video_source_webcam():
    cfg = Settings(VIDEO_SOURCE_TYPE="webcam", WEBCAM_INDEX=0)
    source = create_video_source(cfg)
    assert isinstance(source, WebcamVideoSource)


def test_create_video_source_rtsp():
    cfg = Settings(VIDEO_SOURCE_TYPE="rtsp", VIDEO_SOURCE_PATH="rtsp://example.invalid/stream")
    source = create_video_source(cfg)
    assert isinstance(source, RTSPVideoSource)


def test_create_video_source_rejects_unknown_type():
    cfg = Settings(VIDEO_SOURCE_TYPE="carrier-pigeon")
    with pytest.raises(ValueError):
        create_video_source(cfg)


# ---------------------------------------------------------------------------
# RTSP-specific: credential masking (no live stream needed)
# ---------------------------------------------------------------------------

def test_rtsp_url_masking_hides_credentials():
    masked = RTSPVideoSource._mask_url("rtsp://user:secret@192.0.2.10:554/stream1")
    assert "secret" not in masked
    assert "user" not in masked
    assert "192.0.2.10" in masked


def test_rtsp_url_masking_passthrough_without_credentials():
    url = "rtsp://192.0.2.10:554/stream1"
    assert RTSPVideoSource._mask_url(url) == url
