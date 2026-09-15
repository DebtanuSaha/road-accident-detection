"""
engine/detection/yolo_detector.py

Phase 3 — YOLO Object Detection.

Wraps a pretrained Ultralytics YOLO model to detect road users (car,
motorcycle, bus, truck, bicycle, person) frame-by-frame.

Responsibilities:
- Run inference on a single BGR frame (as produced by engine.input).
- Filter results by confidence threshold and by the configured
  allow-list of target class names.
- Convert raw Ultralytics results into `Detection` dataclasses so
  every downstream module (tracking, motion, collision — later
  phases) depends only on this module's types, never on Ultralytics
  internals.
- Draw bounding boxes, class labels, and confidence scores onto a
  frame for visualization/evidence purposes.

No custom model is trained here — a pretrained (COCO) model is used,
per the project spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import Settings
from config.settings import settings as default_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


@dataclass(frozen=True)
class Detection:
    """A single structured detection, ready for tracking/motion/collision consumption."""

    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict:
        """Conceptual schema per project spec: {"class", "confidence", "bbox"}."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox.as_tuple()],
        }


class YOLODetectorError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or inference fails unrecoverably."""


# Deterministic per-class colours (BGR) so the same class always draws the same colour.
_CLASS_COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (245, 135, 66),   # blue-ish
    (135, 245, 66),   # green
    (66, 66, 245),    # red
    (66, 209, 245),   # yellow
    (245, 66, 186),   # purple
    (230, 245, 66),   # cyan
]


def _color_for_class(class_name: str) -> Tuple[int, int, int]:
    index = hash(class_name) % len(_CLASS_COLOR_PALETTE)
    return _CLASS_COLOR_PALETTE[index]


class YOLODetector:
    """
    Thin, config-driven wrapper around an Ultralytics YOLO model.

    Usage:
        detector = YOLODetector(settings)
        detector.load()
        detections = detector.detect(frame)
        annotated = YOLODetector.draw_detections(frame, detections)
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._model: Optional[YOLO] = None
        self._target_classes: Set[str] = {c.lower() for c in cfg.YOLO_TARGET_CLASSES}
        self._class_id_to_name: Dict[int, str] = {}

    def load(self) -> None:
        """Load the pretrained model. Raises YOLODetectorError on failure."""
        try:
            logger.info(
                "Loading YOLO model from %s (device=%s)",
                self._cfg.YOLO_MODEL_PATH, self._cfg.YOLO_DEVICE,
            )
            self._model = YOLO(self._cfg.YOLO_MODEL_PATH)
            self._class_id_to_name = {int(k): str(v) for k, v in self._model.names.items()}
        except Exception as exc:  # Ultralytics/torch can raise several exception types
            self._model = None
            raise YOLODetectorError(
                f"Failed to load YOLO model from {self._cfg.YOLO_MODEL_PATH}: {exc}"
            ) from exc

        available = {name.lower() for name in self._class_id_to_name.values()}
        missing = self._target_classes - available
        if missing:
            logger.warning(
                "Configured target classes not found in model vocabulary "
                "and will never be detected: %s",
                sorted(missing),
            )
        matched = sorted(self._target_classes & available)
        logger.info("YOLO model loaded successfully. Watching for classes: %s", matched)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame and return structured,
        confidence- and class-filtered Detection objects.
        """
        if self._model is None:
            raise YOLODetectorError("YOLODetector.detect() called before load().")
        if frame is None or frame.size == 0:
            raise ValueError("detect() received an empty frame.")

        results = self._model.predict(
            source=frame,
            conf=self._cfg.YOLO_CONFIDENCE_THRESHOLD,
            iou=self._cfg.YOLO_IOU_THRESHOLD,
            device=self._cfg.YOLO_DEVICE,
            verbose=False,
        )

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            class_name = self._class_id_to_name.get(class_id, str(class_id)).lower()
            confidence = float(box.conf.item())

            # Redundant with conf= passed to predict(), kept as an explicit,
            # config-driven guard per the "filter detections using confidence
            # thresholds" requirement.
            if confidence < self._cfg.YOLO_CONFIDENCE_THRESHOLD:
                continue
            if class_name not in self._target_classes:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=BoundingBox(x1, y1, x2, y2),
                )
            )

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence scores onto a
        COPY of the given frame. The original frame is never mutated.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.as_int_tuple()
            color = _color_for_class(det.class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y1 = max(y1 - text_h - baseline, 0)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w, y1), color, thickness=-1)
            cv2.putText(
                annotated, label, (x1, y1 - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return annotated
