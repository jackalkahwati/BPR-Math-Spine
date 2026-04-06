"""Research loop utilities for evidence intake and theory iteration."""

from .audit import AuditEvent, append_audit_event, load_audit_events
from .evidence import (
    EvidenceMeasurement,
    EvidenceRecord,
    PaperCandidate,
    SearchQuery,
)
from .ingest import (
    InMemoryEvidenceSource,
    build_search_queries,
    map_candidate_to_prediction,
)
from .loop import ResearchLoopConfig, ResearchLoopResult, ResearchLoopRunner
from .policy import DEFAULT_AUTONOMY_POLICY, AutonomyPolicy, SandboxTarget

__all__ = [
    "AuditEvent",
    "append_audit_event",
    "load_audit_events",
    "EvidenceMeasurement",
    "EvidenceRecord",
    "PaperCandidate",
    "SearchQuery",
    "InMemoryEvidenceSource",
    "build_search_queries",
    "map_candidate_to_prediction",
    "ResearchLoopConfig",
    "ResearchLoopResult",
    "ResearchLoopRunner",
    "DEFAULT_AUTONOMY_POLICY",
    "AutonomyPolicy",
    "SandboxTarget",
]
