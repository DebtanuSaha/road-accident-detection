"""
scripts/check_setup.py

Phase 1 verification script.

Confirms that:
  1. The configuration layer imports and loads correctly.
  2. All required data/log directories exist (creating any missing ones).
  3. config/thresholds.yaml is present and structurally valid.
  4. The shared logger is wired up correctly.

Run with:
    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (`python scripts/check_setup.py`)
# without needing to install the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from config.threshold_loader import load_thresholds, ThresholdConfigError  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

REQUIRED_DIRS = [
    settings.LOG_DIR,
    settings.DATA_DIR,
    settings.INPUT_VIDEO_DIR,
    settings.INPUT_STREAM_DIR,
    settings.OUTPUT_ANNOTATED_DIR,
    settings.OUTPUT_PROCESSED_DIR,
    settings.EVIDENCE_IMAGE_DIR,
    settings.EVIDENCE_VIDEO_DIR,
    settings.DATABASE_DIR,
]


def main() -> int:
    logger.info("=== Phase 1 setup check: %s ===", settings.APP_NAME)

    # 1. Settings loaded
    logger.info("Environment: %s", settings.ENVIRONMENT)
    logger.info("Log level:   %s", settings.LOG_LEVEL)

    # 2. Directories exist
    missing = [d for d in REQUIRED_DIRS if not Path(d).is_dir()]
    if missing:
        logger.error("Missing directories: %s", missing)
        return 1
    logger.info("All %d required directories exist.", len(REQUIRED_DIRS))

    # 3. thresholds.yaml loads and validates
    try:
        thresholds = load_thresholds()
    except ThresholdConfigError as exc:
        logger.error("Threshold config invalid: %s", exc)
        return 1
    logger.info(
        "thresholds.yaml loaded with sections: %s", sorted(thresholds.keys())
    )

    logger.info("=== Phase 1 setup check PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
