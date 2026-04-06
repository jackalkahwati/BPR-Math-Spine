# Continuous Evidence Pipeline

This document describes the first autonomous loop in the continuous BPR
research system: evidence discovery and staging.

## Purpose

The evidence loop searches for new experimental papers, maps them to existing
`prediction_id` entries, and stages structured records before any theory edits
are considered.

## Commands

```bash
# Show the frozen evaluator boundary and first sandbox target
python3 scripts/research_loop.py show-policy

# Scan flagship predictions and stage candidate evidence
python3 scripts/research_loop.py evidence-scan

# Limit the scan to one target prediction
python3 scripts/research_loop.py evidence-scan --prediction-id P20.7_LI_delta_c_over_c
```

## Outputs

- Structured evidence records: `data/evidence/staging/*.json`
- Evidence schema: `data/evidence/schema/evidence_record.schema.json`
- Triage markdown report: `doc/experiments/evidence_queue.md`
- Append-only audit trail: `data/research/audit/research_events.jsonl`
- Frozen evaluator boundary: `data/research/evaluator_boundary.json`

## Safety Rules

The evidence loop may discover papers and draft links, but it does not change
the scoring policy for those papers. The benchmark registry, validation gate,
and falsification policy remain frozen during autonomous theory-improvement
cycles.

## First Sandboxed Theory Target

The first autonomous theory-improvement sandbox is
`bpr/charged_leptons.py`, focused on the local discrepancy around
`P18.2_m_muon_MeV` while preserving the wider benchmark surface.
