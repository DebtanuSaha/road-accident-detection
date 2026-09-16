"""
engine/collision/collision_detector.py

Phase 6 — Collision Detection: spatial interaction rules.

Geometric checks between pairs of currently tracked objects for the
CURRENT frame, with short-lived contact memory for hysteresis:
  1. Bounding-box overlap/intersection (IoU)
  2. Very small distance between tracked objects (center distance)

These are two of the project's seven potential collision indicators.
The remaining five (trajectory change, deceleration, stopping,
post-interaction behaviour) come from Phase 5's `MotionState` and are
combined with this module's output in `collision_scorer.py`. Keeping
spatial geometry separate from scoring/state means this module stays
trivially testable and has no memory of past frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Set, Tuple

from config.settings import Settings
from config.settings import settings as default_settings
from config.threshold_loader import load_thresholds
from engine.detection import BoundingBox
from engine.tracking import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-Union of two axis-aligned boxes, in [0, 1]."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = a.area + b.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Euclidean pixel distance between two boxes' centers."""
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass(frozen=True)
class SpatialInteraction:
    """Result of the current-frame spatial checks between one pair of objects."""

    track_id_a: int
    track_id_b: int
    iou: float
    center_distance_px: float
    is_overlapping: bool
    is_close: bool
    is_sustained_contact: bool
    spatial_score: float  # combined [0,1] strength: max(IoU, closeness fraction)

    @property
    def is_interacting(self) -> bool:
        return self.is_overlapping or self.is_close

    def to_dict(self) -> dict:
        return {
            "pair": [self.track_id_a, self.track_id_b],
            "iou": round(self.iou, 4),
            "center_distance_px": round(self.center_distance_px, 2),
            "is_overlapping": self.is_overlapping,
            "is_sustained_contact": self.is_sustained_contact,
            "is_close": self.is_close,
            "spatial_score": round(self.spatial_score, 4),
        }


class CollisionDetector:
    """
    Per-frame spatial interaction checks across all pairs of currently
    active tracked objects, retaining genuinely overlapping pairs for
    hysteresis across the collision aftermath.
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        collision_cfg = load_thresholds()["collision"]
        self._min_center_distance_px: float = collision_cfg["min_center_distance_px"]
        self._bbox_overlap_iou: float = collision_cfg["bbox_overlap_iou"]
        self._sustain_bbox_overlap_iou: float = collision_cfg["sustain_bbox_overlap_iou"]
        self._sustain_min_center_distance_px: float = collision_cfg["sustain_min_center_distance_px"]
        self._contact_pairs: Set[Tuple[int, int]] = set()
        logger.info(
            "CollisionDetector initialized: min_center_distance_px=%.1f bbox_overlap_iou=%.2f",
            self._min_center_distance_px, self._bbox_overlap_iou,
        )

    def find_interactions(self, tracked_objects: Sequence[TrackedObject]) -> List[SpatialInteraction]:
        """
        Check every pair of currently active tracked objects for spatial
        interaction (bbox overlap or very small distance). Only
        interacting pairs are returned — O(n^2) in the number of active
        objects, which is fine at typical road-scene object counts, and
        keeps the output relevant rather than emitting every non-
        interacting pair every frame.
        """
        active = [t for t in tracked_objects if t.is_active]
        active_ids = {t.track_id for t in active}
        self._contact_pairs.intersection_update(
            pair for pair in self._contact_pairs if pair[0] in active_ids and pair[1] in active_ids
        )
        interactions: List[SpatialInteraction] = []

        for a, b in combinations(active, 2):
            pair_key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            iou = compute_iou(a.bbox, b.bbox)
            distance = compute_center_distance(a.bbox, b.bbox)
            is_overlapping = iou >= self._bbox_overlap_iou
            is_close = distance <= self._min_center_distance_px
            is_sustained_contact = pair_key in self._contact_pairs and (
                iou >= self._sustain_bbox_overlap_iou
                or distance <= self._sustain_min_center_distance_px
            )

            if not (is_overlapping or is_close or is_sustained_contact):
                self._contact_pairs.discard(pair_key)
                continue

            if is_overlapping:
                self._contact_pairs.add(pair_key)

            closeness = 0.0
            if self._min_center_distance_px > 0:
                closeness = max(0.0, 1.0 - (distance / self._min_center_distance_px))
            spatial_score = max(iou, closeness)

            interactions.append(
                SpatialInteraction(
                    track_id_a=a.track_id,
                    track_id_b=b.track_id,
                    iou=iou,
                    center_distance_px=distance,
                    is_overlapping=is_overlapping,
                    is_close=is_close,
                    is_sustained_contact=is_sustained_contact,
                    spatial_score=spatial_score,
                )
            )

        return interactions
