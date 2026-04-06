"""Structured evidence models for the continuous BPR research loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Optional


def utc_now_iso() -> str:
    """Return an ISO timestamp with second-level precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class EvidenceMeasurement:
    """A machine-readable measurement extracted from a paper."""

    relation: str
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    unit: str = ""
    extracted_text: str = ""
    comparison_target: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "unit": self.unit,
            "extracted_text": self.extracted_text,
            "comparison_target": self.comparison_target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceMeasurement":
        return cls(
            relation=payload["relation"],
            value=payload.get("value"),
            uncertainty=payload.get("uncertainty"),
            unit=payload.get("unit", ""),
            extracted_text=payload.get("extracted_text", ""),
            comparison_target=payload.get("comparison_target"),
        )


@dataclass(frozen=True)
class SearchQuery:
    """A search query generated from a measurement target."""

    source: str
    query: str
    prediction_id: str
    measurement_name: str
    priority: int = 0
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "query": self.query,
            "prediction_id": self.prediction_id,
            "measurement_name": self.measurement_name,
            "priority": self.priority,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class PaperCandidate:
    """A candidate paper discovered by an evidence source."""

    source: str
    source_id: str
    title: str
    abstract: str
    url: str
    published: str
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_key(self) -> str:
        """Stable identity for deduplication and file naming."""
        return f"{self.source}:{self.source_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "published": self.published,
            "authors": list(self.authors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperCandidate":
        return cls(
            source=payload["source"],
            source_id=payload["source_id"],
            title=payload["title"],
            abstract=payload.get("abstract", ""),
            url=payload.get("url", ""),
            published=payload.get("published", ""),
            authors=list(payload.get("authors", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence staged for review and later verdict assignment."""

    record_id: str
    prediction_id: str
    measurement_name: str
    candidate: PaperCandidate
    proposed_verdict: str
    measurement: EvidenceMeasurement
    mapping_score: float
    extraction_confidence: float
    review_status: str = "unreviewed"
    notes: list[str] = field(default_factory=list)
    linked_prediction_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    @staticmethod
    def build_record_id(candidate: PaperCandidate, prediction_id: str) -> str:
        """Build a deterministic ID from the candidate/prediction pairing."""
        digest = sha1(
            f"{candidate.stable_key}|{prediction_id}".encode("utf-8")
        ).hexdigest()[:12]
        return f"ev-{digest}"

    @classmethod
    def from_candidate(
        cls,
        candidate: PaperCandidate,
        prediction_id: str,
        measurement_name: str,
        proposed_verdict: str,
        measurement: EvidenceMeasurement,
        mapping_score: float,
        extraction_confidence: float,
        notes: Optional[list[str]] = None,
        linked_prediction_ids: Optional[list[str]] = None,
    ) -> "EvidenceRecord":
        return cls(
            record_id=cls.build_record_id(candidate, prediction_id),
            prediction_id=prediction_id,
            measurement_name=measurement_name,
            candidate=candidate,
            proposed_verdict=proposed_verdict,
            measurement=measurement,
            mapping_score=float(mapping_score),
            extraction_confidence=float(extraction_confidence),
            notes=list(notes or []),
            linked_prediction_ids=list(linked_prediction_ids or [prediction_id]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prediction_id": self.prediction_id,
            "measurement_name": self.measurement_name,
            "candidate": self.candidate.to_dict(),
            "proposed_verdict": self.proposed_verdict,
            "measurement": self.measurement.to_dict(),
            "mapping_score": self.mapping_score,
            "extraction_confidence": self.extraction_confidence,
            "review_status": self.review_status,
            "notes": list(self.notes),
            "linked_prediction_ids": list(self.linked_prediction_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            record_id=payload["record_id"],
            prediction_id=payload["prediction_id"],
            measurement_name=payload["measurement_name"],
            candidate=PaperCandidate.from_dict(payload["candidate"]),
            proposed_verdict=payload["proposed_verdict"],
            measurement=EvidenceMeasurement.from_dict(payload["measurement"]),
            mapping_score=float(payload["mapping_score"]),
            extraction_confidence=float(payload["extraction_confidence"]),
            review_status=payload.get("review_status", "unreviewed"),
            notes=list(payload.get("notes", [])),
            linked_prediction_ids=list(
                payload.get("linked_prediction_ids", [payload["prediction_id"]])
            ),
            created_at=payload.get("created_at", utc_now_iso()),
        )
