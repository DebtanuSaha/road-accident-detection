"""
scripts/run_evidence_capture.py

Phase 9 entry point: chains video input (Phase 2) -> detection
(Phase 3) -> tracking (Phase 4) -> motion analysis (Phase 5) ->
collision scoring (Phase 6) -> accident detection + temporal
verification (Phase 7) -> event generation (Phase 8) -> evidence
capture (Phase 9).

The very first time a pair is confirmed, this saves the accident
frame, the annotated frame, structured metadata (detections, tracking,
accident score), and — after continuing to feed subsequent frames —
a video clip covering several seconds before and after the confirmed
accident. Any clip still pending when the video source ends is
flushed (possibly truncated) rather than silently lost.

Usage:
    python scripts/run_evidence_capture.py [--max-frames N] [--output NAME.mp4] [--preview]
"""

from __future__ import annotations

import argparse
import json
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
from engine.accident import AccidentDetector, AccidentState  # noqa: E402
from engine.events import EventBuilder  # noqa: E402
from engine.evidence import EvidenceRecorder, FrameSnapshot, RollingFrameBuffer  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.preview import LivePreview  # noqa: E402

logger = get_logger(__name__)


def score_color(score: float) -> tuple:
    score = max(0.0, min(1.0, score))
    green = (0, 200, 0)
    red = (0, 0, 255)
    return tuple(int(g + (r - g) * score) for g, r in zip(green, red))


def draw_assessments(frame, tracked_objects, assessments):
    annotated = frame.copy()
    objects_by_id = {t.track_id: t for t in tracked_objects}

    for obj in tracked_objects:
        if not obj.is_active:
            continue
        x1, y1, x2, y2 = obj.bbox.as_int_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(
            annotated, f"ID {obj.track_id}", (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

    for a in assessments:
        obj_a = objects_by_id.get(a.track_id_a)
        if obj_a is None:
            continue

        color = (0, 0, 255) if a.confirmed else score_color(a.accident_probability)
        thickness = 3 if a.confirmed else 2

        if a.is_single_object:
            # Loss-of-control pathway: one object, no partner -> draw its
            # box only, no connecting line (there is nothing to connect to).
            # Only draw once CONFIRMED — real-footage testing showed
            # displaying every unconfirmed "solo" candidate (which fires
            # on ordinary braking/turning constantly) buried the display
            # in noise and looked like a wall of false alarms even though
            # none of them had actually been confirmed.
            if not a.confirmed:
                continue
            x1, y1, x2, y2 = obj_a.bbox.as_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                annotated, "LOSS OF CONTROL", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            continue

        obj_b = objects_by_id.get(a.track_id_b)
        if obj_b is None:
            continue

        ax, ay = (int(v) for v in obj_a.bbox.center)
        bx, by = (int(v) for v in obj_b.bbox.center)
        cv2.line(annotated, (ax, ay), (bx, by), color, thickness)
        for obj in (obj_a, obj_b):
            x1, y1, x2, y2 = obj.bbox.as_int_tuple()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        mx, my = (ax + bx) // 2, (ay + by) // 2
        label = "ACCIDENT CONFIRMED" if a.confirmed else f"p={a.accident_probability:.2f}"
        cv2.putText(
            annotated, label, (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6 if a.confirmed else 0.5, color, 2, cv2.LINE_AA,
        )

    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 evidence capture run")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=str, default="evidence_output.mp4")
    parser.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window while processing. Press 'q' to stop early.",
    )
    args = parser.parse_args()

    logger.info("=== Phase 9 evidence capture run ===")

    yolo_detector = YOLODetector(settings)
    try:
        yolo_detector.load()
    except YOLODetectorError as exc:
        logger.error("Could not load YOLO model: %s", exc)
        return 1

    tracker = ByteTrackWrapper(settings)
    manager = TrackManager(settings)
    motion_analyzer = MotionAnalyzer(settings)
    collision_scorer = CollisionScorer(settings)
    accident_detector = AccidentDetector(settings)
    event_builder = EventBuilder(settings)
    evidence_recorder = EvidenceRecorder(settings)

    source = create_video_source(settings)
    try:
        with source:
            width, height = source.get_frame_size()
            fps = source.get_fps() or 25.0
            frame_buffer = RollingFrameBuffer(fps=fps, cfg=settings)

            output_path = str(Path(settings.OUTPUT_ANNOTATED_DIR) / args.output)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.error("Could not open output writer at %s", output_path)
                return 1

            preview = LivePreview(enabled=args.preview, window_name="Phase 9 - Evidence Capture", fps=fps)

            frame_count = 0
            evidence_records = []
            start = time.time()

            while True:
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break

                result = source.read()
                if not result.success:
                    break

                detections = yolo_detector.detect(result.frame)
                raw_tracks = tracker.update(detections, result.frame)
                tracked_objects = manager.update(raw_tracks, result.frame_index, result.timestamp_ms)
                motion_states = motion_analyzer.analyze_many(tracked_objects)
                collision_scores = collision_scorer.update(
                    tracked_objects, motion_states, result.frame_index, result.timestamp_ms
                )
                assessments = accident_detector.update(
                    collision_scores, result.frame_index, result.timestamp_ms,
                    motion_states=motion_states, tracked_objects=tracked_objects,
                )

                annotated = draw_assessments(result.frame, tracked_objects, assessments)

                current_snapshot = FrameSnapshot(
                    frame_index=result.frame_index,
                    timestamp_ms=result.timestamp_ms,
                    frame=result.frame.copy(),
                    annotated_frame=annotated.copy(),
                )

                for a in assessments:
                    if a.instantaneous_state != AccidentState.NORMAL or a.confirmed:
                        logger.info("frame=%d assessment=%s", result.frame_index, a.to_dict())

                    event = event_builder.build_if_new(a, tracked_objects)
                    if event is not None:
                        event_json = event.model_dump(mode="json", by_alias=True)
                        event_path = Path(settings.OUTPUT_PROCESSED_DIR) / f"{event.event_id}.json"
                        event_path.write_text(json.dumps(event_json, indent=2))
                        print(f"\n*** ACCIDENT EVENT CREATED: {event.event_id} *** frame={result.frame_index}\n")

                        involved_detections = [d for d in detections]
                        record = evidence_recorder.start_capture(
                            event, frame_buffer, current_snapshot, involved_detections, tracked_objects,
                        )
                        print(f"Evidence capture started for {event.event_id}: {record.accident_frame_path}")

                # Feed every frame (whether or not any clip is pending) so
                # already-pending clips keep collecting their post-event window.
                finalized = evidence_recorder.update(current_snapshot)
                for record in finalized:
                    evidence_records.append(record)
                    print(
                        f"\n*** EVIDENCE CLIP FINALIZED: {record.event_id} *** "
                        f"{record.evidence_video_path} (truncated={record.clip_truncated})\n"
                    )

                frame_buffer.add(current_snapshot)
                writer.write(annotated)
                if not preview.show(annotated):
                    break  # user pressed 'q'
                frame_count += 1

            # Flush any clips still pending when the video source ran out.
            for record in evidence_recorder.finalize_all():
                evidence_records.append(record)
                print(
                    f"\n*** EVIDENCE CLIP FINALIZED (video ended): {record.event_id} *** "
                    f"{record.evidence_video_path} (truncated={record.clip_truncated})\n"
                )

            writer.release()
            preview.close()
            elapsed = time.time() - start

            logger.info(
                "Processed %d frame(s), %d evidence record(s) finalized, in %.2fs (%.1f fps).",
                frame_count, len(evidence_records), elapsed, frame_count / elapsed if elapsed > 0 else 0.0,
            )
            logger.info("Annotated video written to: %s", output_path)

            if frame_count == 0:
                logger.error("No frames were processed — check VIDEO_SOURCE_PATH/type in .env")
                return 1

    except VideoSourceError as exc:
        logger.error("Failed to open video source: %s", exc)
        return 1

    logger.info("=== Phase 9 evidence capture run PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
