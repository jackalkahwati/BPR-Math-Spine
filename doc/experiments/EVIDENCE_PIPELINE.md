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
python3 scripts/research_loop.py evidence-scan --prediction-id P4.9_Tc_MgB2_K
```

Note (2026-10): the single-target example above read
`--prediction-id P20.7_LI_delta_c_over_c` until 2026-10, and P20.7 was a
flagship. P20.7 (|δc/c| = e^(−p^(1/3))) is superseded: it is stipulated on
the α-calibrated p, and section 4 of the
[architecture decision](../derivations/architecture_decision_2026-09-26.md)
makes BPR-6D exactly Lorentz-invariant. It is no longer in
`flagship_prediction_ids` in `bpr/research/policy.py`, so the bare
`evidence-scan` skips it. Its registry row and query hints are kept, so an
explicit `--prediction-id P20.7_LI_delta_c_over_c` scan still runs, as a legacy
pipeline example only. Found in the 2026-10 review of unmerged branches; see
`doc/CLOSED_AND_DEPRECATED.md`.

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
