"""
scripts/check_video_input.py

Phase 2 verification script.

Opens the video source configured in `.env` (VIDEO_SOURCE_TYPE /
VIDEO_SOURCE_PATH / WEBCAM_INDEX) via the Phase 2 `create_video_source()`
factory, reads a handful of frames, and reports FPS, frame size, and
read success — without depending on any later-phase code (detection,
tracking, etc. don't exist yet).

Usage:
    python scripts/check_video_input.py [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from engine.input import create_video_source, VideoSourceError  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 video input check")
    parser.add_argument("--max-frames", type=int, default=60, help="Number of frames to read before stopping")
    args = parser.parse_args()

    logger.info("=== Phase 2 video input check ===")
    logger.info("VIDEO_SOURCE_TYPE=%s VIDEO_SOURCE_PATH=%s FRAME_SKIP=%d",
                settings.VIDEO_SOURCE_TYPE, settings.VIDEO_SOURCE_PATH, settings.FRAME_SKIP)

    source = create_video_source(settings)

    try:
        with source:
            width, height = source.get_frame_size()
            logger.info("Opened source: fps=%.2f size=%dx%d", source.get_fps(), width, height)

            read_count = 0
            failed = False
            start = time.time()

            while read_count < args.max_frames:
                result = source.read()
                if not result.success:
                    logger.warning("Read failed at frame_index=%d — stopping.", result.frame_index)
                    failed = True
                    break
                read_count += 1

            elapsed = time.time() - start
            logger.info(
                "Read %d frame(s) in %.2fs (%.1f fps actual read rate).",
                read_count, elapsed, read_count / elapsed if elapsed > 0 else 0.0,
            )

            if read_count == 0:
                logger.error("No frames were read — check VIDEO_SOURCE_PATH/type in .env")
                return 1

    except VideoSourceError as exc:
        logger.error("Failed to open video source: %s", exc)
        return 1

    logger.info("=== Phase 2 video input check PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
