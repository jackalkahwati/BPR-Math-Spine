import json
from pathlib import Path

from bpr.experimental_data import get_measurement
from bpr.research.audit import AuditEvent, append_audit_event, load_audit_events
from bpr.research.evidence import (
    EvidenceMeasurement,
    EvidenceRecord,
    PaperCandidate,
    SearchQuery,
)
from bpr.research.ingest import (
    InMemoryEvidenceSource,
    build_search_queries,
    map_candidate_to_prediction,
)
from bpr.research.loop import ResearchLoopConfig, ResearchLoopRunner
from bpr.research.policy import DEFAULT_AUTONOMY_POLICY


def test_evidence_record_round_trip_preserves_prediction_mapping():
    measurement = EvidenceMeasurement(
        relation="upper_bound",
        value=6.0e-21,
        unit="",
        extracted_text="Fermi-LAT bounds |δc/c| below 6e-21.",
    )
    candidate = PaperCandidate(
        source="arxiv",
        source_id="2308.03031",
        title="Quadratic Lorentz invariance violation constraints from GRB 221009A",
        abstract="We derive the strongest quadratic Lorentz invariance violation limits from GRB timing.",
        url="https://arxiv.org/abs/2308.03031",
        published="2023-08-05",
        authors=["Piran", "Ofengeim"],
        metadata={"doi": ""},
    )
    record = EvidenceRecord.from_candidate(
        candidate=candidate,
        prediction_id="P20.7_LI_delta_c_over_c",
        measurement_name="Speed-of-light deviation below Fermi-LAT bound",
        proposed_verdict="INCONCLUSIVE",
        measurement=measurement,
        mapping_score=0.82,
        extraction_confidence=0.74,
        notes=["Current bound remains above BPR target sensitivity."],
    )

    restored = EvidenceRecord.from_dict(record.to_dict())

    assert restored.prediction_id == "P20.7_LI_delta_c_over_c"
    assert restored.measurement.relation == "upper_bound"
    assert restored.notes == ["Current bound remains above BPR target sensitivity."]


def test_build_search_queries_uses_measurement_names_and_prediction_ids():
    queries = build_search_queries(
        prediction_ids=["P20.7_LI_delta_c_over_c", "P4.9_Tc_MgB2_K"],
        max_queries_per_prediction=2,
    )

    rendered = [query.query for query in queries]
    assert any("P20.7_LI_delta_c_over_c" in query.prediction_id for query in queries)
    assert any("Lorentz" in query for query in rendered)
    assert any("MgB2" in query or "MgB" in query for query in rendered)


def test_build_search_queries_rejects_unknown_prediction_ids():
    try:
        build_search_queries(prediction_ids=["P999.unknown"])
    except ValueError as exc:
        assert "Unknown prediction_id" in str(exc)
    else:
        raise AssertionError("Unknown prediction IDs must raise ValueError")


def test_map_candidate_to_prediction_prefers_best_overlap():
    candidate = PaperCandidate(
        source="crossref",
        source_id="10.1038/35065039",
        title="Superconductivity at 39 K in magnesium diboride",
        abstract="Measurements of the superconducting transition temperature in MgB2 establish Tc near 39 K.",
        url="https://doi.org/10.1038/35065039",
        published="2001-03-01",
        authors=["Nagamatsu"],
    )

    match = map_candidate_to_prediction(candidate)

    assert match.prediction_id == "P4.9_Tc_MgB2_K"
    assert match.score > 0.2


def test_audit_log_is_append_only(tmp_path: Path):
    log_path = tmp_path / "research_events.jsonl"
    append_audit_event(
        log_path,
        AuditEvent(
            event_type="evidence_staged",
            summary="Staged one candidate paper.",
            prediction_ids=["P20.7_LI_delta_c_over_c"],
            payload={"count": 1},
        ),
    )
    append_audit_event(
        log_path,
        AuditEvent(
            event_type="theory_proposal",
            summary="Prepared charged lepton sandbox proposal.",
            prediction_ids=["P18.2_m_muon_MeV"],
            payload={"target": "charged_leptons"},
        ),
    )

    events = load_audit_events(log_path)

    assert [event.event_type for event in events] == [
        "evidence_staged",
        "theory_proposal",
    ]
    assert log_path.read_text().count("\n") == 2


def test_policy_freezes_evaluator_files_and_exposes_primary_sandbox():
    assert DEFAULT_AUTONOMY_POLICY.is_evaluator_path("bpr/experimental_data.py")
    assert DEFAULT_AUTONOMY_POLICY.is_evaluator_path(
        "scripts/benchmark_predictions.py"
    )

    sandbox = DEFAULT_AUTONOMY_POLICY.primary_sandbox()
    assert sandbox.module_path == "bpr/charged_leptons.py"
    assert "P18.2_m_muon_MeV" in sandbox.related_prediction_ids


def test_research_loop_runner_stages_records_and_writes_queue(tmp_path: Path):
    measurement = get_measurement("P20.7_LI_delta_c_over_c")
    candidate = PaperCandidate(
        source="arxiv",
        source_id="2308.03031",
        title="Quadratic Lorentz invariance violation constraints from GRB 221009A",
        abstract="We derive strong Lorentz invariance violation constraints from gamma-ray burst timing.",
        url="https://arxiv.org/abs/2308.03031",
        published="2023-08-05",
        authors=["Piran", "Ofengeim"],
    )
    runner = ResearchLoopRunner(
        ResearchLoopConfig(
            staging_dir=tmp_path / "staging",
            audit_log_path=tmp_path / "audit" / "research_events.jsonl",
            triage_report_path=tmp_path / "reports" / "evidence_queue.md",
        ),
        sources=[InMemoryEvidenceSource([candidate])],
    )

    result = runner.run_evidence_scan(
        prediction_ids=[measurement.prediction_id],
        max_queries_per_prediction=1,
        max_results_per_query=3,
    )

    assert result.staged_records == 1
    record_paths = sorted((tmp_path / "staging").glob("*.json"))
    assert len(record_paths) == 1
    payload = json.loads(record_paths[0].read_text())
    assert payload["prediction_id"] == "P20.7_LI_delta_c_over_c"
    assert payload["measurement"]["relation"] == "qualitative"
    assert payload["measurement"]["value"] is None
    assert (tmp_path / "reports" / "evidence_queue.md").exists()


def test_runner_deduplicates_same_candidate_across_queries(tmp_path: Path):
    candidate = PaperCandidate(
        source="arxiv",
        source_id="2308.03031",
        title="Quadratic Lorentz invariance violation constraints from GRB 221009A",
        abstract="We derive strong Lorentz invariance violation constraints from gamma-ray burst timing.",
        url="https://arxiv.org/abs/2308.03031",
        published="2023-08-05",
        authors=["Piran", "Ofengeim"],
    )
    runner = ResearchLoopRunner(
        ResearchLoopConfig(
            staging_dir=tmp_path / "staging",
            audit_log_path=tmp_path / "audit" / "research_events.jsonl",
            triage_report_path=tmp_path / "reports" / "evidence_queue.md",
        ),
        sources=[InMemoryEvidenceSource([candidate])],
    )

    result = runner.run_evidence_scan(
        prediction_ids=["P20.7_LI_delta_c_over_c"],
        max_queries_per_prediction=2,
        max_results_per_query=3,
    )

    assert result.staged_records == 1
    assert len(list((tmp_path / "staging").glob("*.json"))) == 1
