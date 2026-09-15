"""
engine/events/event_builder.py

Phase 8 — Accident Event Generation.

Once a pair is CONFIRMED by Phase 7's TemporalVerifier, builds one
structured `AccidentEvent` (a Pydantic model, per the project spec's
schema) for it — exactly once per confirmed pair.

ARCHITECTURE NOTE: the canonical `AccidentEvent` Pydantic model lives
here, in the CV engine, NOT in `backend/schemas/accident.py` — even
though the project's own example schema and Phase 11's FastAPI
backend will both want it. This is deliberate: `engine/` must stay
usable completely independently of `backend/` (a hard project
requirement since Phase 1 — "Keep the CV engine usable independently
from FastAPI"), and Phase 8 explicitly runs via `scripts/` alone, with
no backend/database/API involved yet. `backend/schemas/accident.py`
will import and reuse this same model once Phase 11 exists, rather
than redefining it — see that file's own docstring.

Fields deliberately left `None`/`"unknown"` for now, per the project's
own example schema and its "do not fabricate" rule:
  - `severity` — always "unknown"; severity estimation is an explicit
    Future Extension, out of scope for this entire project (per spec).
  - `location` — always None; populated starting Phase 10 (GPS/camera
    location system).
  - `evidence_image` / `evidence_video` — always None; populated
    starting Phase 9 (evidence capture).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings
from config.settings import settings as default_settings
from engine.accident import AccidentAssessment
from engine.tracking import TrackedObject
from utils.logger import get_logger

logger = get_logger(__name__)

PairKey = Tuple[int, int]


class AccidentSeverity(str, Enum):
    """
    Severity classification. Only UNKNOWN is ever assigned by any code
    in this project — severity estimation is an explicit "Future
    Extension — DO NOT IMPLEMENT INITIALLY" item in the project spec.
    The other members exist purely so the schema doesn't need a
    breaking change whenever that future work happens.
    """

    UNKNOWN = "unknown"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class AccidentEventStatus(str, Enum):
    """
    Event lifecycle status. This project only ever sets DETECTED (at
    creation, matching the project spec's own example schema exactly).
    The other members exist for later phases (e.g. Phase 13's alert
    service, or a future human-review workflow) to use — nothing here
    transitions status automatically.
    """

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class ObjectInvolved(BaseModel):
    """One object involved in an accident event, per the project's example schema."""

    model_config = ConfigDict(populate_by_name=True)

    track_id: int
    class_name: str = Field(alias="class")


class Location(BaseModel):
    """
    Matches the location payload shape the project spec itself uses for
    Phase 10 (GPS). Declared now so `AccidentEvent.location`'s shape is
    stable; no code populates an actual `Location` instance until
    Phase 10 exists — until then `AccidentEvent.location` is always None.
    """

    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None


class AccidentEvent(BaseModel):
    """
    Structured accident event, matching the project spec's example
    schema field-for-field:

        {
          "event_id": "ACC-000001",
          "timestamp": "...",
          "accident_probability": 0.91,
          "severity": "unknown",
          "objects_involved": [{"track_id": 12, "class": "car"}, ...],
          "location": null,
          "camera_id": "CAM-001",
          "evidence_image": "...",
          "evidence_video": "...",
          "status": "detected"
        }
    """

    model_config = ConfigDict(populate_by_name=True)

    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accident_probability: float = Field(ge=0.0, le=1.0)
    severity: AccidentSeverity = AccidentSeverity.UNKNOWN
    objects_involved: List[ObjectInvolved]
    location: Optional[Location] = None
    camera_id: str
    evidence_image: Optional[str] = None
    evidence_video: Optional[str] = None
    status: AccidentEventStatus = AccidentEventStatus.DETECTED


def _pair_key(id_a: int, id_b: int) -> PairKey:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


class EventBuilder:
    """
    Builds exactly one `AccidentEvent` per confirmed pair.

    Usage (call once per frame, after Phase 7's AccidentDetector):
        builder = EventBuilder(settings)
        for assessment in assessments:
            event = builder.build_if_new(assessment, tracked_objects)
            if event is not None:
                ...  # a brand-new confirmed accident this frame
    """

    def __init__(self, cfg: Settings = default_settings):
        self._cfg = cfg
        self._next_sequence = 1
        # In-memory only for now — Phase 12 (database layer) will be the
        # real, persistent source of event IDs and de-duplication across
        # process restarts. This set only prevents duplicate events
        # within a single run.
        self._already_built: Set[PairKey] = set()

    def build_if_new(
        self,
        assessment: AccidentAssessment,
        tracked_objects: Sequence[TrackedObject],
    ) -> Optional[AccidentEvent]:
        """
        Returns a new `AccidentEvent` the first time a given pair
        becomes confirmed, and `None` on every subsequent call for that
        same pair (or for any pair that isn't confirmed at all) — so
        callers can invoke this every frame without checking anything
        themselves and still get exactly one event per accident.
        """
        if not assessment.confirmed:
            return None

        key = _pair_key(assessment.track_id_a, assessment.track_id_b)
        if key in self._already_built:
            return None

        self._already_built.add(key)
        event = self.build(assessment, tracked_objects)
        logger.info("Accident event created: %s", event.model_dump(mode="json", by_alias=True))
        return event

    def build(
        self,
        assessment: AccidentAssessment,
        tracked_objects: Sequence[TrackedObject],
    ) -> AccidentEvent:
        """
        Construct an `AccidentEvent` from a (presumably confirmed)
        assessment, without the de-duplication `build_if_new()` does —
        exposed separately so callers/tests can build an event directly
        when they already know it's warranted.
        """
        objects_by_id: Dict[int, TrackedObject] = {t.track_id: t for t in tracked_objects}

        involved: List[ObjectInvolved] = []
        pairs = (
            [(assessment.track_id_a, assessment.class_a)]
            if assessment.track_id_a == assessment.track_id_b
            else [(assessment.track_id_a, assessment.class_a), (assessment.track_id_b, assessment.class_b)]
        )
        for track_id, fallback_class in pairs:
            obj = objects_by_id.get(track_id)
            class_name = obj.class_name if obj is not None else fallback_class
            involved.append(ObjectInvolved(track_id=track_id, class_name=class_name))

        return AccidentEvent(
            event_id=self._next_event_id(),
            accident_probability=assessment.accident_probability,
            objects_involved=involved,
            camera_id=self._cfg.DEFAULT_CAMERA_ID,
        )

    def reset(self) -> None:
        """Clear de-duplication state. Does NOT reset the event ID sequence."""
        self._already_built.clear()

    def _next_event_id(self) -> str:
        event_id = f"ACC-{self._next_sequence:06d}"
        self._next_sequence += 1
        return event_id
