"""
scripts/run_collision_detection.py

Phase 6 entry point: chains video input (Phase 2) -> detection
(Phase 3) -> tracking (Phase 4) -> motion analysis (Phase 5) ->
collision scoring (Phase 6). Draws a connecting line between any pair
of objects currently being scored, colored by composite score, and
writes the annotated result to data/output/annotated/.

No accident/temporal-verification logic exists yet (Phase 7) — this
script only proves the composite collision score behaves sensibly on
real detected/tracked objects. A high score here is NOT an accident
declaration.

Usage:
    python scripts/run_collision_detection.py [--max-frames N] [--output NAME.mp4]
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
from engine.collision import CollisionScorer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.preview import LivePreview  # noqa: E402

logger = get_logger(__name__)


def score_color(score: float) -> tuple:
    """Green (low score) -> red (high score), interpolated. Returns a BGR tuple for OpenCV."""
    score = max(0.0, min(1.0, score))
    green = (0, 200, 0)
    red = (0, 0, 255)
    return tuple(int(g + (r - g) * score) for g, r in zip(green, red))


def draw_collisions(frame, tracked_objects, motion_states, collision_scores):
    annotated = frame.copy()
    objects_by_id = {t.track_id: t for t in tracked_objects}

    # Base boxes/IDs (reuse Phase 5 style, simplified).
    for obj in tracked_objects:
        if not obj.is_active:
            continue
        x1, y1, x2, y2 = obj.bbox.as_int_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(
            annotated, f"ID {obj.track_id}", (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    # Collision pairs: connecting line + score label, colored by severity.
    for score in collision_scores:
        obj_a = objects_by_id.get(score.track_id_a)
        obj_b = objects_by_id.get(score.track_id_b)
        if obj_a is None or obj_b is None:
            continue

        color = score_color(score.composite_score)
        ax, ay = (int(v) for v in obj_a.bbox.center)
        bx, by = (int(v) for v in obj_b.bbox.center)
        cv2.line(annotated, (ax, ay), (bx, by), color, 2)

        # Also highlight both boxes involved.
        for obj in (obj_a, obj_b):
            x1, y1, x2, y2 = obj.bbox.as_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        mx, my = (ax + bx) // 2, (ay + by) // 2
        cv2.putText(
            annotated, f"{score.composite_score:.2f}", (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 collision scoring run")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=str, default="collision_output.mp4")
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window while processing (requires a non-headless "
             "OpenCV build and a display; see utils/preview.py). Press 'q' to stop early.",
    )
    args = parser.parse_args()

    logger.info("=== Phase 6 collision detection run ===")

    detector = YOLODetector(settings)
    try:
        detector.load()
    except YOLODetectorError as exc:
        logger.error("Could not load YOLO model: %s", exc)
        return 1

    tracker = ByteTrackWrapper(settings)
    manager = TrackManager(settings)
    analyzer = MotionAnalyzer(settings)
    scorer = CollisionScorer(settings)

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

            preview = LivePreview(enabled=args.preview, window_name='Phase 6 - Collision Detection', fps=fps)

            frame_count = 0
            max_composite_seen = 0.0
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
                collision_scores = scorer.update(
                    tracked_objects, motion_states, result.frame_index, result.timestamp_ms
                )

                for cs in collision_scores:
                    max_composite_seen = max(max_composite_seen, cs.composite_score)
                    logger.info("frame=%d collision_score=%s", result.frame_index, cs.to_dict())

                annotated = draw_collisions(result.frame, tracked_objects, motion_states, collision_scores)
                writer.write(annotated)
                if not preview.show(annotated):
                    break  # user pressed 'q'
                frame_count += 1

            writer.release()
            preview.close()
            elapsed = time.time() - start

            logger.info(
                "Processed %d frame(s), max composite collision score seen: %.3f, in %.2fs (%.1f fps).",
                frame_count, max_composite_seen, elapsed, frame_count / elapsed if elapsed > 0 else 0.0,
            )
            logger.info("Annotated video written to: %s", output_path)

            if frame_count == 0:
                logger.error("No frames were processed — check VIDEO_SOURCE_PATH/type in .env")
                return 1

    except VideoSourceError as exc:
        logger.error("Failed to open video source: %s", exc)
        return 1

    logger.info("=== Phase 6 collision detection run PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
