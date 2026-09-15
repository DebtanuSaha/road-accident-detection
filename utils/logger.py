"""
utils/logger.py

Central logging setup used by every layer of the system (CV engine,
location service, backend, alerts, scripts).

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")

Design:
- One rotating file handler (config.settings.LOG_DIR/LOG_FILE) shared
  by the whole app, plus a console handler.
- Log level is read from config.settings.LOG_LEVEL, so it is
  configurable via environment variables without code changes.
- Idempotent: calling get_logger() multiple times for the same name
  (or re-importing this module) never duplicates handlers.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logging() -> None:
    global _configured
    if _configured:
        return

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_path = os.path.join(settings.LOG_DIR, settings.LOG_FILE)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root_logging()
    return logging.getLogger(name)
