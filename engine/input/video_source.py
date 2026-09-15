"""
engine/input/video_source.py

Phase 2 — Video Input Layer.

Defines the `VideoSource` abstraction that every frame-producing input
(local file, webcam, RTSP/IP stream) implements, plus the concrete
`FileVideoSource` for local video files.

Design goals (per project spec):
- The rest of the pipeline (detection, tracking, ...) depends only on
  this interface — never on `cv2.VideoCapture` directly — so the
  source can be swapped (file / webcam / RTSP) without touching
  downstream code.
- FPS handling, frame dimensions, stream failure handling, graceful
  shutdown, and optional frame skipping all live here so every
  concrete source gets them for free.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameResult:
    """Result of a single `VideoSource.read()` call."""

    success: bool
    frame: Optional[np.ndarray]
    frame_index: int
    timestamp_ms: float


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or fails unrecoverably."""


class VideoSource(ABC):
    """
    Abstract base class for all frame-producing sources.

    Subclasses implement `open()`, `_read_raw()`, `get_fps()`,
    `get_frame_size()`, `is_opened()`, and `release()`. Frame skipping
    and frame indexing/timestamps are handled once, here, so every
    subclass behaves consistently.
    """

    def __init__(self, frame_skip: int = 0):
        """
        Args:
            frame_skip: number of frames to discard between each frame
                that is actually returned by `read()`. 0 = return every
                frame. Used for performance on slower hardware.
        """
        if frame_skip < 0:
            raise ValueError("frame_skip must be >= 0")
        self._frame_skip = frame_skip
        self._frame_index = -1

    # -- subclasses must implement -------------------------------------------------
    @abstractmethod
    def open(self) -> bool:
        """Open the underlying source. Returns True on success."""

    @abstractmethod
    def _read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read exactly one frame from the underlying source, unmodified by frame_skip."""

    @abstractmethod
    def get_fps(self) -> float:
        """Return the source's reported FPS, or 0.0 if unknown."""

    @abstractmethod
    def get_frame_size(self) -> Tuple[int, int]:
        """Return (width, height) in pixels, or (0, 0) if unknown."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Return whether the source is currently open and readable."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources. Must be safe to call multiple times."""

    # -- shared behaviour ------------------------------------------------------------
    def read(self) -> FrameResult:
        """
        Return the next frame, honoring `frame_skip`.

        Skipped frames are still consumed from the underlying source
        (so playback stays roughly real-time-ordered); only the
        returned frame's index/timestamp are advanced for frames that
        are actually handed back to the caller.
        """
        for _ in range(self._frame_skip):
            ok, _ = self._read_raw()
            if not ok:
                return FrameResult(False, None, self._frame_index, 0.0)
            self._frame_index += 1

        ok, frame = self._read_raw()
        if not ok:
            return FrameResult(False, None, self._frame_index, 0.0)

        self._frame_index += 1
        fps = self.get_fps()
        timestamp_ms = (self._frame_index / fps) * 1000.0 if fps > 0 else 0.0
        return FrameResult(success=True, frame=frame, frame_index=self._frame_index, timestamp_ms=timestamp_ms)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def __enter__(self) -> "VideoSource":
        if not self.open():
            raise VideoSourceError(f"Failed to open video source: {self!r}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class FileVideoSource(VideoSource):
    """Video source backed by a local video file (e.g. .mp4, .avi)."""

    def __init__(self, path: str, frame_skip: int = 0):
        super().__init__(frame_skip=frame_skip)
        self._path = path
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if not self._path:
            logger.error("FileVideoSource: no path provided.")
            return False
        if not Path(self._path).is_file():
            logger.error("FileVideoSource: file does not exist: %s", self._path)
            return False

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error("FileVideoSource: OpenCV failed to open file: %s", self._path)
            self._cap = None
            return False

        w, h = self.get_frame_size()
        logger.info(
            "FileVideoSource opened: %s (fps=%.2f, size=%dx%d)",
            self._path, self.get_fps(), w, h,
        )
        return True

    def _read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            logger.info("FileVideoSource: end of stream reached for %s", self._path)
        return ok, frame

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0

    def get_frame_size(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_total_frames(self) -> int:
        """Total frame count if known (file sources only), else 0."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("FileVideoSource released: %s", self._path)
            self._cap = None

    def __repr__(self) -> str:
        return f"FileVideoSource(path={self._path!r})"
