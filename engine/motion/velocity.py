"""
engine/motion/velocity.py

Phase 5 — Movement and Trajectory Analysis: relative speed estimation.

IMPORTANT — pixel velocity vs real-world velocity:
Every value produced here is a PIXEL-SPACE, RELATIVE measurement
(pixels moved per frame, or per second of video time). None of it is,
or should be interpreted as, real-world speed in km/h or m/s. Doing
that conversion correctly requires camera calibration (focal length,
mounting height/angle, or a known real-world reference distance in the
scene) — none of which this system has. Every function and field name
here is deliberately suffixed `_px_per_frame` / `_px_per_sec` rather
than a bare "speed" to keep that distinction visible wherever this
data is used downstream (collision scoring, accident events, API
responses, evidence records).
"""

from __future__ import annotations

from typing import Optional

from engine.motion.trajectory import compute_displacement
from engine.tracking import PositionSample


def compute_speed_px_per_frame(prev: PositionSample, curr: PositionSample) -> float:
    """
    Relative pixel speed, normalized by elapsed frame count between the
    two samples (handles the case where frames were skipped/missed
    between them). This is the primary speed metric used for threshold
    comparisons throughout Phase 5/6, since it doesn't depend on the
    video source reporting a reliable FPS.
    """
    frame_delta = curr.frame_index - prev.frame_index
    if frame_delta <= 0:
        return 0.0
    return compute_displacement(prev, curr) / frame_delta


def compute_speed_px_per_sec(prev: PositionSample, curr: PositionSample) -> Optional[float]:
    """
    Relative pixel speed normalized by elapsed wall-clock video time.

    Returns None when elapsed time can't be trusted (e.g. the video
    source couldn't report FPS, so `timestamp_ms` stayed at 0 for both
    samples) — informational only, never used for threshold decisions,
    since it silently degrades to None on sources without known FPS.
    """
    time_delta_sec = (curr.timestamp_ms - prev.timestamp_ms) / 1000.0
    if time_delta_sec <= 0:
        return None
    return compute_displacement(prev, curr) / time_delta_sec


def compute_speed_change(previous_speed: Optional[float], current_speed: float) -> Optional[float]:
    """
    Change in relative pixel speed between two consecutive movement
    segments (current minus previous). Negative = decelerating,
    positive = accelerating. This is an ACCELERATION PROXY in
    pixel-space only — not a physical acceleration measurement.

    Returns None if there is no previous segment to compare against.
    """
    if previous_speed is None:
        return None
    return current_speed - previous_speed
