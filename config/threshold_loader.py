"""
config/threshold_loader.py

Loads the fine-grained, per-signal thresholds declared in
`config/thresholds.yaml`. Kept separate from `settings.py` so that:

- `settings.py` covers coarse, environment-overridable knobs (paths,
  model paths, top-level thresholds).
- `thresholds.yaml` covers detailed scoring weights that a non-developer
  (or the same developer, mid-tuning) may want to edit directly without
  redeploying environment variables.

Used starting Phase 6 (collision scoring) and Phase 7 (accident
decision engine). Provided in Phase 1 so the config layer is complete
and stable before those modules are built.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from config.settings import BASE_DIR

THRESHOLDS_PATH = BASE_DIR / "config" / "thresholds.yaml"


class ThresholdConfigError(RuntimeError):
    """Raised when thresholds.yaml is missing or malformed."""


@lru_cache(maxsize=1)
def load_thresholds(path: Path = THRESHOLDS_PATH) -> Dict[str, Any]:
    """
    Load and cache the threshold configuration.

    Cached with lru_cache since this file should not change at runtime
    during normal operation; call `load_thresholds.cache_clear()` in
    tests if you need to reload after editing the file on disk.
    """
    if not path.exists():
        raise ThresholdConfigError(f"Threshold config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ThresholdConfigError(f"Threshold config at {path} did not parse to a mapping.")

    required_sections = {"collision", "motion", "accident", "evidence"}
    missing = required_sections - data.keys()
    if missing:
        raise ThresholdConfigError(f"Threshold config missing sections: {sorted(missing)}")

    _validate_cross_section_constraints(data, path)

    return data


def _validate_cross_section_constraints(data: Dict[str, Any], path: Path) -> None:
    """
    Check invariants that span sections, so a mis-tune fails loudly at
    startup rather than silently disabling detection at runtime.
    """
    window = data["collision"].get("post_interaction_window_frames")
    streak = data["accident"].get("temporal_verification_frames")

    if window is not None and streak is not None and window < streak:
        # Found the hard way during real-footage debugging: a pair stops
        # being scored once its post-interaction window expires, so if
        # that window is shorter than the confirmation streak, a real
        # collision's streak gets cut off before it can ever complete.
        raise ThresholdConfigError(
            f"Invalid threshold config at {path}: "
            f"collision.post_interaction_window_frames ({window}) must be >= "
            f"accident.temporal_verification_frames ({streak}), otherwise a pair is "
            "evicted from scoring before its confirmation streak can complete and "
            "real collisions can never be confirmed."
        )
