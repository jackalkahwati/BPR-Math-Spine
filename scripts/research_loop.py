#!/usr/bin/env python3
"""Continuous evidence intake and policy reporting for BPR research."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bpr.research.loop import ResearchLoopConfig, ResearchLoopRunner
from bpr.research.policy import DEFAULT_AUTONOMY_POLICY


def _policy_snapshot() -> dict[str, object]:
    sandbox = DEFAULT_AUTONOMY_POLICY.primary_sandbox()
    return {
        "evaluator_paths": list(DEFAULT_AUTONOMY_POLICY.evaluator_paths),
        "append_only_paths": list(DEFAULT_AUTONOMY_POLICY.append_only_paths),
        "flagship_prediction_ids": list(
            DEFAULT_AUTONOMY_POLICY.flagship_prediction_ids
        ),
        "primary_sandbox": {
            "slug": sandbox.slug,
            "module_path": sandbox.module_path,
            "description": sandbox.description,
            "related_prediction_ids": list(sandbox.related_prediction_ids),
            "rationale": sandbox.rationale,
            "acceptance_commands": list(sandbox.acceptance_commands),
        },
    }


def _resolve_repo_path(value: str) -> Path:
    """Resolve relative outputs against the repository root."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the continuous BPR research loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "evidence-scan", help="Discover and stage candidate evidence."
    )
    scan_parser.add_argument(
        "--prediction-id",
        action="append",
        default=[],
        help="Specific prediction ID to scan. Repeat for multiple IDs.",
    )
    scan_parser.add_argument(
        "--staging-dir",
        default="data/evidence/staging",
        help="Directory for staged evidence JSON files.",
    )
    scan_parser.add_argument(
        "--audit-log",
        default="data/research/audit/research_events.jsonl",
        help="Append-only JSONL audit log.",
    )
    scan_parser.add_argument(
        "--report",
        default="doc/experiments/evidence_queue.md",
        help="Markdown triage report path.",
    )
    scan_parser.add_argument(
        "--max-queries-per-prediction",
        type=int,
        default=2,
        help="Maximum discovery queries to build per prediction.",
    )
    scan_parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=5,
        help="Maximum papers to fetch from each source per query.",
    )

    subparsers.add_parser("show-policy", help="Print the frozen evaluator policy.")

    args = parser.parse_args()

    if args.command == "show-policy":
        print(json.dumps(_policy_snapshot(), indent=2, sort_keys=True))
        return

    if args.max_queries_per_prediction < 1:
        parser.error("--max-queries-per-prediction must be >= 1")
    if args.max_results_per_query < 1 or args.max_results_per_query > 25:
        parser.error("--max-results-per-query must be between 1 and 25")

    config = ResearchLoopConfig(
        staging_dir=_resolve_repo_path(args.staging_dir),
        audit_log_path=_resolve_repo_path(args.audit_log),
        triage_report_path=_resolve_repo_path(args.report),
    )
    runner = ResearchLoopRunner(config)
    try:
        result = runner.run_evidence_scan(
            prediction_ids=args.prediction_id or None,
            max_queries_per_prediction=args.max_queries_per_prediction,
            max_results_per_query=args.max_results_per_query,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "staged_records": result.staged_records,
                "queries_run": result.queries_run,
                "report_path": str(result.report_path),
                "staged_paths": [str(path) for path in result.staged_paths],
                "audit_event_ids": list(result.audit_event_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
