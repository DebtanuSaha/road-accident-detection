"""
engine/events package

Phase 8 — Accident Event Generation.

Exposes the canonical `AccidentEvent` Pydantic model (and its
sub-models `ObjectInvolved`, `Location`, plus the `AccidentSeverity` /
`AccidentEventStatus` enums) and `EventBuilder`, which constructs
exactly one `AccidentEvent` per confirmed pair from Phase 7.
"""

from engine.events.event_builder import (
    AccidentEvent,
    AccidentEventStatus,
    AccidentSeverity,
    EventBuilder,
    Location,
    ObjectInvolved,
)

__all__ = [
    "AccidentEvent",
    "AccidentEventStatus",
    "AccidentSeverity",
    "EventBuilder",
    "Location",
    "ObjectInvolved",
]
