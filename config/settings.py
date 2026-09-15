"""
config/settings.py

Central, environment-driven configuration for the entire system.

Design goals (per project spec):
- No hardcoded paths, thresholds, model names, API keys, or coordinates.
- A single source of truth (`settings`) that every layer (CV engine,
  location service, FastAPI backend, alert service) imports from.
- Safe to import even before later phases (backend, GPS, alerts) exist —
  fields for those layers are declared now so Phase 1 config is stable
  and does not need to be redesigned later.

All values are overridable via environment variables or a `.env` file
in the project root (see `.env.example`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory containing this `config/` package's parent.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application-wide settings.

    Values are read from (in order of precedence):
      1. Actual environment variables
      2. A `.env` file at the project root
      3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # General / environment
    # ------------------------------------------------------------------
    APP_NAME: str = "road-accident-detection"
    ENVIRONMENT: str = Field(default="development")  # development | testing | production
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default=str(BASE_DIR / "logs"))
    LOG_FILE: str = Field(default="app.log")

    # ------------------------------------------------------------------
    # Paths (all relative to BASE_DIR unless an absolute path is given)
    # ------------------------------------------------------------------
    DATA_DIR: str = Field(default=str(BASE_DIR / "data"))
    INPUT_VIDEO_DIR: str = Field(default=str(BASE_DIR / "data" / "input" / "videos"))
    INPUT_STREAM_DIR: str = Field(default=str(BASE_DIR / "data" / "input" / "streams"))
    OUTPUT_ANNOTATED_DIR: str = Field(default=str(BASE_DIR / "data" / "output" / "annotated"))
    OUTPUT_PROCESSED_DIR: str = Field(default=str(BASE_DIR / "data" / "output" / "processed"))
    EVIDENCE_IMAGE_DIR: str = Field(default=str(BASE_DIR / "data" / "evidence" / "images"))
    EVIDENCE_VIDEO_DIR: str = Field(default=str(BASE_DIR / "data" / "evidence" / "videos"))
    DATABASE_DIR: str = Field(default=str(BASE_DIR / "data" / "database"))

    # ------------------------------------------------------------------
    # YOLO / detection (used starting Phase 3)
    # ------------------------------------------------------------------
    YOLO_MODEL_PATH: str = Field(default=str(BASE_DIR / "models" / "yolo" / "yolov8n.pt"))
    YOLO_DEVICE: str = Field(default="cpu")  # "cpu" | "cuda" | "cuda:0" ...
    YOLO_CONFIDENCE_THRESHOLD: float = Field(default=0.4)
    YOLO_IOU_THRESHOLD: float = Field(default=0.45)
    # Classes of interest, restricted to road users per project spec.
    YOLO_TARGET_CLASSES: List[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck", "bicycle", "person"]
    )

    # ------------------------------------------------------------------
    # Video input (used starting Phase 2)
    # ------------------------------------------------------------------
    VIDEO_SOURCE_TYPE: str = Field(default="file")  # "file" | "webcam" | "rtsp"
    VIDEO_SOURCE_PATH: str = Field(default="")  # file path or RTSP URL
    WEBCAM_INDEX: int = Field(default=0)
    FRAME_SKIP: int = Field(default=0)  # process every (FRAME_SKIP + 1)th frame
    TARGET_FPS: Optional[float] = Field(default=None)  # None = use source FPS

    # RTSP/IP-camera stream failure handling (Phase 2)
    RTSP_RECONNECT_ATTEMPTS: int = Field(default=5)
    RTSP_RECONNECT_DELAY_SEC: float = Field(default=2.0)
    RTSP_READ_FAILURE_LIMIT: int = Field(default=10)  # consecutive failed reads before giving up

    # ------------------------------------------------------------------
    # Tracking (ByteTrack) — used starting Phase 4
    # ------------------------------------------------------------------
    TRACKER_MAX_MISSED_FRAMES: int = Field(default=30)  # maps to ByteTrack's track_buffer
    TRACKER_MIN_HITS: int = Field(default=3)  # consecutive matches before TrackManager marks a track "confirmed"
    # Maps to ByteTrack's match_thresh: the MAXIMUM allowed association cost
    # (cost = 1 - IoU, optionally fused with detection confidence when
    # TRACKER_FUSE_SCORE is True) — NOT a minimum IoU fraction. Higher =
    # more lenient matching. 0.8 is ByteTrack's own published default.
    TRACKER_IOU_THRESHOLD: float = Field(default=0.8)
    TRACKER_TRACK_HIGH_THRESH: float = Field(default=0.5)  # first-stage (high-confidence) match threshold
    TRACKER_TRACK_LOW_THRESH: float = Field(default=0.1)  # second-stage (low-confidence) match threshold
    TRACKER_NEW_TRACK_THRESH: float = Field(default=0.6)  # min confidence to start a brand-new track
    TRACKER_FUSE_SCORE: bool = Field(default=True)  # fuse detection confidence into the association cost
    TRACK_HISTORY_LENGTH: int = Field(default=30)  # max positions/timestamps kept per tracked object

    # ------------------------------------------------------------------
    # Collision / accident thresholds — used starting Phase 6/7
    #
    # NOTE: the fine-grained thresholds for both collision scoring
    # (Phase 6) and accident decision/temporal verification (Phase 7)
    # all live in config/thresholds.yaml (`collision:` and `accident:`
    # sections) rather than here. Phase 1 originally declared
    # placeholder knobs in this Settings class (COLLISION_DISTANCE_PX,
    # ACCIDENT_PROBABILITY_THRESHOLD, TEMPORAL_VERIFICATION_FRAMES)
    # before that pattern was established; they were removed once
    # Phase 6/7 confirmed thresholds.yaml as the single source of
    # truth for this domain, so that editing one place always has an
    # effect rather than silently doing nothing. See thresholds.yaml
    # for every tunable value in this area.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # GPS / location — used starting Phase 10
    # ------------------------------------------------------------------
    DEFAULT_CAMERA_ID: str = Field(default="CAM-001")
    LOCATION_MODE: str = Field(default="fixed")  # "fixed" | "gps"

    # ------------------------------------------------------------------
    # Database — used starting Phase 12
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'database' / 'accidents.db'}")

    # ------------------------------------------------------------------
    # Backend / API — used starting Phase 11
    # ------------------------------------------------------------------
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_V1_PREFIX: str = Field(default="/api/v1")
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------------
    # Alerts — used starting Phase 13. No real credentials belong here;
    # this project never fabricates or hardcodes secrets.
    # ------------------------------------------------------------------
    ALERT_DESTINATIONS: List[str] = Field(default_factory=lambda: ["console"])  # console|email|webhook|mock
    WEBHOOK_ALERT_URL: str = Field(default="")
    SMTP_HOST: str = Field(default="")
    SMTP_PORT: int = Field(default=587)
    SMTP_USERNAME: str = Field(default="")
    SMTP_PASSWORD: str = Field(default="")
    ALERT_EMAIL_FROM: str = Field(default="")
    ALERT_EMAIL_TO: List[str] = Field(default_factory=list)

    @field_validator(
        "YOLO_TARGET_CLASSES",
        "CORS_ALLOW_ORIGINS",
        "ALERT_DESTINATIONS",
        "ALERT_EMAIL_TO",
        mode="before",
    )
    @classmethod
    def _split_comma_separated(cls, value):
        """Allow list-type env vars to be given as comma-separated strings."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def ensure_directories(self) -> None:
        """Create all data/log directories declared above if they don't exist."""
        dirs = [
            self.LOG_DIR,
            self.DATA_DIR,
            self.INPUT_VIDEO_DIR,
            self.INPUT_STREAM_DIR,
            self.OUTPUT_ANNOTATED_DIR,
            self.OUTPUT_PROCESSED_DIR,
            self.EVIDENCE_IMAGE_DIR,
            self.EVIDENCE_VIDEO_DIR,
            self.DATABASE_DIR,
            os.path.dirname(self.YOLO_MODEL_PATH),
        ]
        for d in dirs:
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)


# Singleton settings instance used across the whole application.
settings = Settings()
settings.ensure_directories()
