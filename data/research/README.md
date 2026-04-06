# Continuous Research Artifacts

This directory holds machine-readable metadata for the continuous BPR research
loop.

## Contents

- `evaluator_boundary.json`: frozen evaluator policy and first sandbox target
- `audit/research_events.jsonl`: append-only event log written by the loop

The research loop is deliberately split so evidence discovery and theory
mutation cannot silently rewrite the benchmark policy in the same cycle.
