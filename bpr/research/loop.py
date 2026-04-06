"""Orchestration for the continuous BPR evidence loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bpr.experimental_data import get_measurement

from .audit import AuditEvent, append_audit_event
from .evidence import EvidenceMeasurement, EvidenceRecord
from .ingest import (
    ArxivEvidenceSource,
    CrossrefEvidenceSource,
    EvidenceSource,
    build_search_queries,
    map_candidate_to_prediction,
)


@dataclass(frozen=True)
class ResearchLoopConfig:
    """Filesystem configuration for evidence staging and reporting."""

    staging_dir: Path = Path("data/evidence/staging")
    audit_log_path: Path = Path("data/research/audit/research_events.jsonl")
    triage_report_path: Path = Path("doc/experiments/evidence_queue.md")


@dataclass(frozen=True)
class ResearchLoopResult:
    """Summary of a single evidence scan."""

    staged_records: int
    queries_run: int
    report_path: Path
    staged_paths: tuple[Path, ...] = ()
    audit_event_ids: tuple[str, ...] = ()


def _infer_verdict(match_score: float) -> str:
    """Convert mapping confidence into a default draft verdict."""
    if match_score >= 0.65:
        return "LIKELY_RELEVANT"
    if match_score >= 0.3:
        return "REVIEW"
    return "UNLINKED"


def _extraction_confidence(match_score: float) -> float:
    """Keep extraction confidence conservative until human review."""
    return min(0.95, round(0.45 + 0.5 * match_score, 2))


class ResearchLoopRunner:
    """Stage evidence candidates and write append-only audit records."""

    def __init__(
        self,
        config: ResearchLoopConfig | None = None,
        sources: Optional[list[EvidenceSource]] = None,
    ) -> None:
        self.config = config or ResearchLoopConfig()
        self.sources = sources or [ArxivEvidenceSource(), CrossrefEvidenceSource()]

    def run_evidence_scan(
        self,
        prediction_ids: list[str] | None = None,
        max_queries_per_prediction: int = 2,
        max_results_per_query: int = 5,
    ) -> ResearchLoopResult:
        """Run discovery, stage evidence JSON, and emit a triage markdown report."""
        queries = build_search_queries(
            prediction_ids=prediction_ids,
            max_queries_per_prediction=max_queries_per_prediction,
        )
        records_by_id: dict[str, EvidenceRecord] = {}
        audit_ids: list[str] = []

        for query in queries:
            for source in self.sources:
                source_query = query
                if source.name != query.source:
                    source_query = type(query)(
                        source=source.name,
                        query=query.query,
                        prediction_id=query.prediction_id,
                        measurement_name=query.measurement_name,
                        priority=query.priority,
                        tags=query.tags,
                    )
                try:
                    candidates = source.fetch_candidates(
                        source_query, max_results=max_results_per_query
                    )
                except Exception as exc:
                    event = AuditEvent(
                        event_type="evidence_source_error",
                        summary=f"{source.name} failed for {query.prediction_id}",
                        prediction_ids=[query.prediction_id],
                        payload={"query": query.query, "error": str(exc)},
                    )
                    append_audit_event(self.config.audit_log_path, event)
                    audit_ids.append(event.event_id)
                    continue

                for candidate in candidates:
                    match = map_candidate_to_prediction(candidate, [query.prediction_id])
                    measurement = get_measurement(match.prediction_id)
                    if measurement is None:
                        continue
                    record = EvidenceRecord.from_candidate(
                        candidate=candidate,
                        prediction_id=match.prediction_id,
                        measurement_name=match.measurement_name,
                        proposed_verdict=_infer_verdict(match.score),
                        measurement=EvidenceMeasurement(
                            relation="qualitative",
                            value=None,
                            uncertainty=None,
                            unit=measurement.unit,
                            extracted_text=self._evidence_excerpt(candidate),
                            comparison_target=(
                                f"{measurement.prediction_id}: {measurement.name}"
                            ),
                        ),
                        mapping_score=match.score,
                        extraction_confidence=_extraction_confidence(match.score),
                        notes=[
                            "Measurement values are not auto-extracted yet; "
                            "this record stages relevance and provenance only.",
                            f"Benchmark target: {measurement.name}",
                            f"Reference source: {measurement.source}",
                            f"Matched terms: {', '.join(match.matched_terms) or 'none'}",
                            f"Source query: {query.query}",
                        ],
                    )
                    existing = records_by_id.get(record.record_id)
                    if existing is None or record.mapping_score > existing.mapping_score:
                        records_by_id[record.record_id] = record
                        self._write_record(record)

        staged_records = list(records_by_id.values())

        report_path = self._write_triage_report(staged_records)
        if staged_records:
            event = AuditEvent(
                event_type="evidence_staged",
                summary=f"Staged {len(staged_records)} evidence candidate(s).",
                prediction_ids=sorted(
                    {record.prediction_id for record in staged_records}
                ),
                payload={
                    "queries_run": len(queries),
                    "records": [record.record_id for record in staged_records],
                    "report_path": str(report_path),
                },
            )
            append_audit_event(self.config.audit_log_path, event)
            audit_ids.append(event.event_id)

        return ResearchLoopResult(
            staged_records=len(staged_records),
            queries_run=len(queries),
            report_path=report_path,
            staged_paths=tuple(
                self.config.staging_dir / f"{record.record_id}.json"
                for record in staged_records
            ),
            audit_event_ids=tuple(audit_ids),
        )

    @staticmethod
    def _evidence_excerpt(record_candidate: object) -> str:
        """Render a concise evidence excerpt from a paper candidate."""
        title = getattr(record_candidate, "title", "").strip()
        abstract = getattr(record_candidate, "abstract", "").strip()
        if abstract:
            return f"{title} -- {abstract[:280]}".strip()
        return title[:280]

    def _write_record(self, record: EvidenceRecord) -> None:
        self.config.staging_dir.mkdir(parents=True, exist_ok=True)
        target = self.config.staging_dir / f"{record.record_id}.json"
        target.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_triage_report(self, records: list[EvidenceRecord]) -> Path:
        self.config.triage_report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Evidence Queue",
            "",
            "> Auto-generated research staging queue for newly discovered evidence.",
            "",
            "| Record | Prediction ID | Verdict Draft | Mapping Score | Source | Title |",
            "|--------|---------------|---------------|---------------|--------|-------|",
        ]
        for record in sorted(records, key=lambda item: item.mapping_score, reverse=True):
            lines.append(
                "| "
                f"{record.record_id} | {record.prediction_id} | "
                f"{record.proposed_verdict} | {record.mapping_score:.2f} | "
                f"{self._escape_markdown_cell(record.candidate.source)} | "
                f"{self._escape_markdown_cell(record.candidate.title)} |"
            )
        if not records:
            lines.append("| — | — | — | — | — | No new evidence staged |")
        self.config.triage_report_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return self.config.triage_report_path

    @staticmethod
    def _escape_markdown_cell(value: str) -> str:
        """Escape table-breaking characters in external metadata."""
        return value.replace("|", "\\|").replace("\n", " ").strip()
