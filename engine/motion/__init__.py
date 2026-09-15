"""
engine/motion package

Phase 5 — Movement and Trajectory Analysis.

Exposes:
- `trajectory.py` — pure geometry (displacement, heading, angular diff).
- `velocity.py` — pixel-space relative speed (never real-world km/h).
- `motion_analyzer.py` — `MotionAnalyzer` / `MotionState`, the combined,
  threshold-driven per-object motion profile used by later phases
  (collision detection, Phase 6).
"""

from engine.motion.motion_analyzer import MotionAnalyzer, MotionState
from engine.motion.trajectory import (
    angular_difference_deg,
    compute_direction_deg,
    compute_displacement,
    euclidean_distance,
)
from engine.motion.velocity import (
    compute_speed_change,
    compute_speed_px_per_frame,
    compute_speed_px_per_sec,
)

__all__ = [
    "MotionAnalyzer",
    "MotionState",
    "euclidean_distance",
    "compute_displacement",
    "compute_direction_deg",
    "angular_difference_deg",
    "compute_speed_px_per_frame",
    "compute_speed_px_per_sec",
    "compute_speed_change",
]
