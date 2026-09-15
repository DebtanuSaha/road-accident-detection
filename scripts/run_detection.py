"""
scripts/run_detection.py

Phase 3 entry point: reads frames from the configured video source
(Phase 2), runs YOLO detection on each frame (Phase 3), draws bounding
boxes/labels/confidence, logs structured detection info, and writes
the annotated result to data/output/annotated/.

This script uses only engine.input and engine.detection — no tracking,
motion, or collision logic exists yet (later phases).

Usage:
    python scripts/run_detection.py [--max-frames N] [--output NAME.mp4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from engine.input import create_video_source, VideoSourceError  # noqa: E402
from engine.detection import YOLODetector, YOLODetectorError  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.preview import LivePreview  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 YOLO detection run")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (default: whole source)")
    parser.add_argument("--output", type=str, default="detection_output.mp4", help="Annotated output filename")
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window while processing (requires a non-headless "
             "OpenCV build and a display; see utils/preview.py). Press 'q' to stop early.",
    )
    args = parser.parse_args()

    logger.info("=== Phase 3 detection run ===")
    logger.info(
        "source=%s path=%s model=%s conf=%.2f classes=%s",
        settings.VIDEO_SOURCE_TYPE, settings.VIDEO_SOURCE_PATH,
        settings.YOLO_MODEL_PATH, settings.YOLO_CONFIDENCE_THRESHOLD,
        settings.YOLO_TARGET_CLASSES,
    )

    detector = YOLODetector(settings)
    try:
        detector.load()
    except YOLODetectorError as exc:
        logger.error("Could not load YOLO model: %s", exc)
        return 1

    source = create_video_source(settings)
    try:
        with source:
            width, height = source.get_frame_size()
            fps = source.get_fps() or 25.0

            output_path = str(Path(settings.OUTPUT_ANNOTATED_DIR) / args.output)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.error("Could not open output writer at %s", output_path)
                return 1

            preview = LivePreview(enabled=args.preview, window_name='Phase 3 - Detection', fps=fps)

            frame_count = 0
            detection_count = 0
            start = time.time()

            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break

                result = source.read()
                if not result.success:
                    break

                detections = detector.detect(result.frame)
                detection_count += len(detections)
                if detections:
                    logger.info(
                        "frame=%d detections=%s",
                        result.frame_index, [d.to_dict() for d in detections],
                    )

                annotated = YOLODetector.draw_detections(result.frame, detections)
                writer.write(annotated)
                if not preview.show(annotated):
                    break  # user pressed 'q'
                frame_count += 1

            writer.release()
            preview.close()
            elapsed = time.time() - start

            logger.info(
                "Processed %d frame(s) with %d total detection(s) in %.2fs (%.1f fps).",
                frame_count, detection_count, elapsed, frame_count / elapsed if elapsed > 0 else 0.0,
            )
            logger.info("Annotated video written to: %s", output_path)

            if frame_count == 0:
                logger.error("No frames were processed — check VIDEO_SOURCE_PATH/type in .env")
                return 1

    except VideoSourceError as exc:
        logger.error("Failed to open video source: %s", exc)
        return 1

    logger.info("=== Phase 3 detection run PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
