"""
utils/preview.py

Shared live-preview helper used by every scripts/run_*.py entry point.

Lets a person watch the annotated output while a script is still
running/writing the output video, instead of only being able to open
the finished .mp4 afterward. Purely a debugging/demo convenience —
no engine module depends on this.

IMPORTANT — why this checks for a display BEFORE calling any OpenCV
GUI function, rather than just wrapping those calls in try/except:
`ultralytics` (a hard dependency since Phase 3) itself depends on full
`opencv-python`, not the headless build — so despite this project's
own `requirements.txt` pinning `opencv-python-headless`, a real
install commonly ends up with GUI-capable OpenCV anyway (whichever
package's files land in site-packages last). On Linux, when that
GUI-capable OpenCV's Qt backend cannot connect to a display (a remote
server, CI, this project's own sandbox, SSH without X forwarding), it
does not raise a normal Python exception — it calls the Qt platform
plugin's fatal-error path, which aborts the entire process (SIGABRT),
uncatchable by try/except. This was found by testing `--preview` in
exactly that environment. The only reliable fix is to never make the
call at all when no display is present, checked via the `DISPLAY` /
`WAYLAND_DISPLAY` environment variables before touching `cv2.namedWindow`.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def _has_display() -> bool:
    """
    Best-effort, pre-flight check for "is a GUI display actually
    available", so we can skip calling into OpenCV's GUI backend
    entirely rather than risk a native crash. Windows and macOS don't
    use this env-var mechanism and (for a script someone runs locally)
    essentially always have a display, so only Linux is gated here.
    """
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class LivePreview:
    """
    Usage:
        preview = LivePreview(enabled=args.preview, window_name="My Script", fps=fps)
        for frame in frames:
            ...
            if not preview.show(annotated_frame):
                break  # user pressed 'q' (or the window was closed)
        preview.close()

    Safe to use even when `enabled=False` (every method becomes a
    cheap no-op) so callers don't need to branch on whether preview is
    on — this keeps the calling script's main loop identical either way.
    """

    def __init__(self, enabled: bool, window_name: str, fps: Optional[float] = None):
        self._window_name = window_name
        self._enabled = False
        self._cv2 = None

        if not enabled:
            return

        if not _has_display():
            logger.warning(
                "Live preview requested, but no display was detected (DISPLAY/"
                "WAYLAND_DISPLAY not set) — this is expected on a remote server, "
                "CI, or SSH session without X forwarding. Skipping preview and "
                "continuing with normal processing (the output file is unaffected)."
            )
            return

        try:
            import cv2  # deferred import: never required unless --preview is actually used
        except ImportError:
            logger.warning(
                "Live preview requested but OpenCV isn't importable at all — continuing without it."
            )
            return

        if not hasattr(cv2, "imshow"):
            logger.warning(
                "Live preview requested, but this OpenCV build has no GUI support at all "
                "— continuing without it."
            )
            return

        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:  # covers cv2.error and any other catchable failure mode
            logger.warning(
                "Live preview requested but the window could not be opened (%s) — "
                "continuing without it.",
                exc,
            )
            return

        self._cv2 = cv2
        self._enabled = True
        self._delay_ms = max(1, int(1000 / fps)) if fps and fps > 0 else 1
        logger.info("Live preview window opened: %r (delay=%dms/frame)", window_name, self._delay_ms)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def show(self, frame: np.ndarray) -> bool:
        """
        Display one frame. Returns False if the person pressed 'q' (or
        Esc) and processing should stop early; True otherwise (including
        whenever preview is disabled — there's nothing to stop for).
        """
        if not self._enabled:
            return True

        try:
            self._cv2.imshow(self._window_name, frame)
            key = self._cv2.waitKey(self._delay_ms) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                logger.info("Live preview: quit requested by user.")
                return False
        except Exception as exc:
            logger.warning("Live preview failed mid-run (%s) — disabling for the rest of this run.", exc)
            self._enabled = False
        return True

    def close(self) -> None:
        if not self._enabled:
            return
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass  # never let cleanup failures surface as a script error
