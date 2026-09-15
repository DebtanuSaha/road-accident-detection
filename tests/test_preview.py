"""
tests/test_preview.py

Tests for the shared utils/preview.py live-preview helper.

The critical property under test: LivePreview must NEVER crash the
calling script, in any environment — including this project's own
sandbox, which has no display and ships an OpenCV build with GUI
symbols present (via ultralytics's transitive dependency) but unable
to actually open a window. That combination previously caused an
uncatchable native abort(); these tests guard against a regression.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preview import LivePreview, _has_display  # noqa: E402


def test_disabled_preview_is_always_a_safe_no_op():
    preview = LivePreview(enabled=False, window_name="test")
    assert preview.enabled is False
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert preview.show(frame) is True  # never blocks/stops processing
    preview.close()  # must not raise


def test_enabled_preview_without_display_does_not_crash(monkeypatch):
    # Force the "no display" path regardless of the actual sandbox env,
    # so this test is deterministic wherever it runs.
    monkeypatch.setattr("utils.preview._has_display", lambda: False)
    preview = LivePreview(enabled=True, window_name="test")
    assert preview.enabled is False  # gracefully disabled, not crashed

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert preview.show(frame) is True
    preview.close()


def test_has_display_false_on_linux_without_env_vars(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert _has_display() is False


def test_has_display_true_on_linux_with_display_env_var(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    assert _has_display() is True


def test_has_display_true_on_non_linux_platforms(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert _has_display() is True


def test_delay_ms_derived_from_fps(monkeypatch):
    monkeypatch.setattr("utils.preview._has_display", lambda: False)
    # Even though this instance ends up disabled (no display), constructing
    # it with an fps must not raise for any fps value, including None/0.
    LivePreview(enabled=True, window_name="test", fps=None)
    LivePreview(enabled=True, window_name="test", fps=0)
    LivePreview(enabled=True, window_name="test", fps=30.0)
