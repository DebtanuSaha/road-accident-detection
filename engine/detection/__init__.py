"""
engine/detection package

Phase 3 — YOLO Object Detection.

Exposes the structured detection types and the `YOLODetector` wrapper
so callers never need to import Ultralytics directly.
"""

from engine.detection.yolo_detector import (
    BoundingBox,
    Detection,
    YOLODetector,
    YOLODetectorError,
)

__all__ = ["BoundingBox", "Detection", "YOLODetector", "YOLODetectorError"]
