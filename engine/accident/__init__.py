"""
engine/accident package

Phase 7 — Accident Detection and Temporal Verification.

Exposes:
- `accident_scorer.py` — stateless per-frame evidence combination and
  3-tier classification (`AccidentScorer`, `AccidentEvidence`,
  `AccidentState`).
- `temporal_verifier.py` — the persistence FSM that turns a stream of
  per-frame scores into a confirmed/not-confirmed decision per pair
  (`TemporalVerifier`, `PairVerificationState`).
- `accident_detector.py` — top-level orchestrator combining both
  (`AccidentDetector`, `AccidentAssessment`).
"""

from engine.accident.accident_detector import AccidentAssessment, AccidentDetector
from engine.accident.accident_scorer import AccidentEvidence, AccidentScorer, AccidentState
from engine.accident.temporal_verifier import PairVerificationState, TemporalVerifier

__all__ = [
    "AccidentDetector",
    "AccidentAssessment",
    "AccidentScorer",
    "AccidentEvidence",
    "AccidentState",
    "TemporalVerifier",
    "PairVerificationState",
]
