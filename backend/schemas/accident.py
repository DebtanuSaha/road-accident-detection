"""
backend/schemas/accident.py

Phase: Phase 11
Responsibility: Accident-related request/response schemas for the FastAPI backend.

NOT IMPLEMENTED YET — intentionally left as an architectural placeholder.
Per the project's phase-by-phase development rule, this module's real
logic is implemented in Phase 11, not in Phase 1.

IMPORTANT (added during Phase 8): the canonical `AccidentEvent` Pydantic
model already exists — in `engine.events.event_builder` — because the CV
engine must stay usable independently of this backend (see that module's
own docstring for why). Phase 11 should `from engine.events import
AccidentEvent, ObjectInvolved, Location, AccidentSeverity,
AccidentEventStatus` and reuse them directly (or wrap them in
request/response-specific schemas here, e.g. a paginated list response),
rather than redefining an equivalent model in this file. Only
backend/HTTP-specific shapes (pagination envelopes, filter query
params, etc.) belong here.
"""
