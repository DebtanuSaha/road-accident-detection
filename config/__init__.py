"""
Configuration package.

Exposes the global `settings` object (environment-driven) and the
threshold-loading utility used across the accident-detection pipeline.
"""

from config.settings import settings, Settings  # noqa: F401

__all__ = ["settings", "Settings"]
