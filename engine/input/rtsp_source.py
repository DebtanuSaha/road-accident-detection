"""
engine/input/rtsp_source.py

Phase 2 — Video Input Layer.

RTSP / IP-camera (CCTV) frame source. Implements the `VideoSource`
interface. CCTV/network streams are the most failure-prone input type
(dropped connections, transient network errors), so this source adds
automatic reconnect handling on top of the base interface.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2

from config.settings import settings
from engine.input.video_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class RTSPVideoSource(VideoSource):
    """
    Video source backed by an RTSP/IP-camera stream.

    Reconnect behaviour is configurable via settings:
        RTSP_RECONNECT_ATTEMPTS   - reconnect attempts per failure episode
        RTSP_RECONNECT_DELAY_SEC  - delay between reconnect attempts
        RTSP_READ_FAILURE_LIMIT   - consecutive failed reads tolerated
                                    before a reconnect is triggered
    """

    def __init__(
        self,
        url: str,
        frame_skip: int = 0,
        reconnect_attempts: Optional[int] = None,
        reconnect_delay_sec: Optional[float] = None,
        read_failure_limit: Optional[int] = None,
    ):
        super().__init__(frame_skip=frame_skip)
        self._url = url
        self._cap: Optional[cv2.VideoCapture] = None

        self._reconnect_attempts = (
            reconnect_attempts if reconnect_attempts is not None else settings.RTSP_RECONNECT_ATTEMPTS
        )
        self._reconnect_delay_sec = (
            reconnect_delay_sec if reconnect_delay_sec is not None else settings.RTSP_RECONNECT_DELAY_SEC
        )
        self._read_failure_limit = (
            read_failure_limit if read_failure_limit is not None else settings.RTSP_READ_FAILURE_LIMIT
        )
        self._consecutive_failures = 0

    def open(self) -> bool:
        if not self._url:
            logger.error("RTSPVideoSource: no stream URL provided.")
            return False

        self._cap = cv2.VideoCapture(self._url)
        if not self._cap.isOpened():
            logger.error("RTSPVideoSource: could not open stream: %s", self._mask_url(self._url))
            self._cap = None
            return False

        self._consecutive_failures = 0
        w, h = self.get_frame_size()
        logger.info(
            "RTSPVideoSource opened: %s (fps=%.2f, size=%dx%d)",
            self._mask_url(self._url), self.get_fps(), w, h,
        )
        return True

    def _read_raw(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if ok:
            self._consecutive_failures = 0
            return True, frame

        self._consecutive_failures += 1
        logger.warning(
            "RTSPVideoSource: read failure %d/%d for %s",
            self._consecutive_failures, self._read_failure_limit, self._mask_url(self._url),
        )

        if self._consecutive_failures < self._read_failure_limit:
            # Tolerate isolated dropped frames without a full reconnect.
            return False, None

        # Too many consecutive failures — attempt to reconnect.
        logger.warning(
            "RTSPVideoSource: failure limit reached, attempting reconnect to %s",
            self._mask_url(self._url),
        )
        if self._reconnect():
            self._consecutive_failures = 0
            ok, frame = self._cap.read()
            return ok, frame

        logger.error("RTSPVideoSource: reconnect exhausted for %s", self._mask_url(self._url))
        return False, None

    def _reconnect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        for attempt in range(1, self._reconnect_attempts + 1):
            logger.info(
                "RTSPVideoSource: reconnect attempt %d/%d for %s",
                attempt, self._reconnect_attempts, self._mask_url(self._url),
            )
            time.sleep(self._reconnect_delay_sec)
            self._cap = cv2.VideoCapture(self._url)
            if self._cap.isOpened():
                logger.info("RTSPVideoSource: reconnect succeeded for %s", self._mask_url(self._url))
                return True

        self._cap = None
        return False

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

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("RTSPVideoSource released: %s", self._mask_url(self._url))
            self._cap = None

    @staticmethod
    def _mask_url(url: str) -> str:
        """Avoid logging embedded stream credentials (rtsp://user:pass@host/...)."""
        if "@" in url:
            scheme_and_creds, _, rest = url.partition("@")
            scheme = scheme_and_creds.split("://")[0]
            return f"{scheme}://***:***@{rest}"
        return url

    def __repr__(self) -> str:
        return f"RTSPVideoSource(url={self._mask_url(self._url)!r})"
