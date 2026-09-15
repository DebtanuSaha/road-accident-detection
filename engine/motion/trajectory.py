"""
engine/motion/trajectory.py

Phase 5 — Movement and Trajectory Analysis: geometry primitives.

Pure functions operating on `engine.tracking.PositionSample` history
(no thresholds, no classification, no config) — displacement and
heading between two samples, and angular difference between two
headings. `velocity.py` and `motion_analyzer.py` build on these.

Everything here is pixel-space geometry. A "direction" is a heading in
image coordinates (0° = pointing along +x/right, increasing clockwise
since image y grows downward) — it has no relationship to compass
bearing or real-world orientation without camera calibration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from engine.tracking import PositionSample


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Straight-line pixel distance between two (x, y) points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def compute_displacement(prev: PositionSample, curr: PositionSample) -> float:
    """Pixel distance moved between two consecutive position samples."""
    return euclidean_distance(prev.center, curr.center)


def compute_direction_deg(prev: PositionSample, curr: PositionSample) -> Optional[float]:
    """
    Heading of movement from `prev` to `curr`, in degrees [0, 360).

    Returns None if the two samples are at (essentially) the same
    point — direction is undefined for zero displacement, and forcing
    a value there would be noise, not signal.
    """
    dx = curr.center[0] - prev.center[0]
    dy = curr.center[1] - prev.center[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angular_difference_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Smallest absolute angular difference between two headings, in
    degrees [0, 180]. Returns None if either heading is undefined
    (e.g. one of the two segments had no measurable movement).
    """
    if a is None or b is None:
        return None
    diff = abs(a - b) % 360.0
    return 360.0 - diff if diff > 180.0 else diff
