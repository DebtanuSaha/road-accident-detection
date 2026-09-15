"""
engine/input package

Phase 2 — Video Input Layer.

Exposes the `VideoSource` interface and a `create_video_source()`
factory that builds the correct concrete source (file / webcam / rtsp)
from `config.settings`, so callers never need to import or choose
concrete classes themselves.
"""

from __future__ import annotations

from config.settings import Settings, settings as default_settings
from engine.input.video_source import FrameResult, VideoSource, VideoSourceError, FileVideoSource
from engine.input.webcam_source import WebcamVideoSource
from engine.input.rtsp_source import RTSPVideoSource
from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "FrameResult",
    "VideoSource",
    "VideoSourceError",
    "FileVideoSource",
    "WebcamVideoSource",
    "RTSPVideoSource",
    "create_video_source",
]


def create_video_source(cfg: Settings = default_settings) -> VideoSource:
    """
    Build the `VideoSource` configured by `cfg.VIDEO_SOURCE_TYPE`.

    Supported values: "file", "webcam", "rtsp".
    Raises `ValueError` for an unrecognized source type.
    """
    source_type = cfg.VIDEO_SOURCE_TYPE.lower().strip()

    if source_type == "file":
        return FileVideoSource(path=cfg.VIDEO_SOURCE_PATH, frame_skip=cfg.FRAME_SKIP)

    if source_type == "webcam":
        return WebcamVideoSource(device_index=cfg.WEBCAM_INDEX, frame_skip=cfg.FRAME_SKIP)

    if source_type == "rtsp":
        return RTSPVideoSource(
            url=cfg.VIDEO_SOURCE_PATH,
            frame_skip=cfg.FRAME_SKIP,
            reconnect_attempts=cfg.RTSP_RECONNECT_ATTEMPTS,
            reconnect_delay_sec=cfg.RTSP_RECONNECT_DELAY_SEC,
            read_failure_limit=cfg.RTSP_READ_FAILURE_LIMIT,
        )

    raise ValueError(
        f"Unsupported VIDEO_SOURCE_TYPE: {cfg.VIDEO_SOURCE_TYPE!r} "
        "(expected 'file', 'webcam', or 'rtsp')"
    )
