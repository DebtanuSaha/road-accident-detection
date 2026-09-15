"""
scripts/run_motion_analysis.py

Phase 5 entry point: chains video input (Phase 2) -> detection
(Phase 3) -> tracking (Phase 4) -> motion analysis (Phase 5). Draws
each active tracked object's box/ID plus its current relative speed
and any sudden-stop / sudden-deceleration / sudden-turn flags, and
writes the annotated result to data/output/annotated/.

No collision/accident logic exists yet (later phases).

Usage:
    python scripts/run_motion_analysis.py [--max-frames N] [--output NAME.mp4]
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
from engine.tracking import ByteTrackWrapper, TrackManager  # noqa: E402
from engine.motion import MotionAnalyzer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.preview import LivePreview  # noqa: E402

logger = get_logger(__name__)


def draw_motion(frame, tracked_objects, motion_states) -> "cv2.Mat":
    annotated = frame.copy()
    for obj in tracked_objects:
        if not obj.is_active:
            continue
        x1, y1, x2, y2 = obj.bbox.as_int_tuple()

        state = motion_states.get(obj.track_id)
        flagged = state is not None and (
            state.is_sudden_stop or state.is_sudden_deceleration or state.is_sudden_direction_change
        )
        color = (0, 0, 255) if flagged else ((0, 200, 255) if obj.confirmed else (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"ID {obj.track_id} {obj.class_name}"
        if state is not None:
            label += f" {state.speed_px_per_frame:.1f}px/f"
            tags = []
            if state.is_sudden_stop:
                tags.append("STOP")
            if state.is_sudden_deceleration:
                tags.append("DECEL")
            if state.is_sudden_direction_change:
                tags.append("TURN")
            if tags:
                label += " [" + ",".join(tags) + "]"

        cv2.putText(
            annotated, label, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 detection + tracking + motion analysis run")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=str, default="motion_output.mp4")
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window while processing (requires a non-headless "
             "OpenCV build and a display; see utils/preview.py). Press 'q' to stop early.",
    )
    args = parser.parse_args()

    logger.info("=== Phase 5 motion analysis run ===")

    detector = YOLODetector(settings)
    try:
        detector.load()
    except YOLODetectorError as exc:
        logger.error("Could not load YOLO model: %s", exc)
        return 1

    tracker = ByteTrackWrapper(settings)
    manager = TrackManager(settings)
    analyzer = MotionAnalyzer(settings)

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

            preview = LivePreview(enabled=args.preview, window_name='Phase 5 - Motion Analysis', fps=fps)

            frame_count = 0
            event_count = 0
            start = time.time()

            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break

                result = source.read()
                if not result.success:
                    break

                detections = detector.detect(result.frame)
                raw_tracks = tracker.update(detections, result.frame)
                tracked_objects = manager.update(raw_tracks, result.frame_index, result.timestamp_ms)
                motion_states = analyzer.analyze_many(tracked_objects)

                for state in motion_states.values():
                    if state.is_sudden_stop or state.is_sudden_deceleration or state.is_sudden_direction_change:
                        event_count += 1
                        logger.info("frame=%d motion_event=%s", result.frame_index, state.to_dict())

                annotated = draw_motion(result.frame, tracked_objects, motion_states)
                writer.write(annotated)
                if not preview.show(annotated):
                    break  # user pressed 'q'
                frame_count += 1

            writer.release()
            preview.close()
            elapsed = time.time() - start

            logger.info(
                "Processed %d frame(s), %d motion event(s) flagged, in %.2fs (%.1f fps).",
                frame_count, event_count, elapsed, frame_count / elapsed if elapsed > 0 else 0.0,
            )
            logger.info("Annotated video written to: %s", output_path)

            if frame_count == 0:
                logger.error("No frames were processed — check VIDEO_SOURCE_PATH/type in .env")
                return 1

    except VideoSourceError as exc:
        logger.error("Failed to open video source: %s", exc)
        return 1

    logger.info("=== Phase 5 motion analysis run PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
