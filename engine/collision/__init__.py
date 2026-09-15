"""
engine/collision package

Phase 6 — Collision Detection.

Exposes:
- `collision_detector.py` — stateless per-frame spatial checks
  (`CollisionDetector`, `SpatialInteraction`, `compute_iou`,
  `compute_center_distance`).
- `collision_scorer.py` — stateful, multi-signal composite scoring
  built on top (`CollisionScorer`, `CollisionScore`).
"""

from engine.collision.collision_detector import (
    CollisionDetector,
    SpatialInteraction,
    compute_center_distance,
    compute_iou,
)
from engine.collision.collision_scorer import CollisionScore, CollisionScorer

__all__ = [
    "CollisionDetector",
    "SpatialInteraction",
    "compute_iou",
    "compute_center_distance",
    "CollisionScorer",
    "CollisionScore",
]
