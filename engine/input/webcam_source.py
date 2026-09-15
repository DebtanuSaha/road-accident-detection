"""
engine/input/webcam_source.py

Phase 2 — Video Input Layer.

Webcam frame source. Implements the `VideoSource` interface so the
rest of the pipeline can treat it identically to a file or RTSP
source.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from engine.input.video_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class WebcamVideoSource(VideoSource):
    """Video source backed by a locally attached webcam."""

    def __init__(self, device_index: int = 0, frame_skip: int = 0):
        super().__init__(frame_skip=frame_skip)
        self._device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            logger.error("WebcamVideoSource: could not open webcam index %d", self._device_index)
            self._cap = None
            return False

        w, h = self.get_frame_size()
        logger.info(
            "WebcamVideoSource opened: index=%d (fps=%.2f, size=%dx%d)",
            self._device_index, self.get_fps(), w, h,
        )
        return True

    def _read_raw(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            logger.warning("WebcamVideoSource: failed to read frame from webcam index %d", self._device_index)
        return ok, frame

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Many webcams report 0 or an unreliable FPS value.
        return float(fps) if fps and fps > 0 else 0.0

    def get_frame_size(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("WebcamVideoSource released: index=%d", self._device_index)
            self._cap = None

    def __repr__(self) -> str:
        return f"WebcamVideoSource(device_index={self._device_index})"
