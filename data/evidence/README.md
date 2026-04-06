# Evidence Staging

This directory stores machine-readable evidence discovered by the continuous
research loop before it is promoted into curated summaries.

## Layout

- `schema/evidence_record.schema.json`: canonical schema for one staged record
- `staging/`: JSON evidence candidates awaiting review or promotion

## Workflow

1. Run `python scripts/research_loop.py evidence-scan`.
2. Inspect the staged JSON files under `staging/`.
3. Review the generated triage report at `doc/experiments/evidence_queue.md`.
4. Promote vetted evidence into the curated ledgers and benchmark registry.

The staging layer is intentionally append-friendly and reviewable. It is where
new papers and extracted measurements land before they influence theory-level
claims.
