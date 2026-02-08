#!/usr/bin/env python3
"""
Benchmark BPR Predictions Against Experimental Data
=====================================================

Compares every testable BPR prediction to published experimental
measurements (PDG, Planck, CODATA, etc.) and produces a quantitative
scorecard with sigma deviations, relative errors, and pass/fail
classification.

Usage:
    python scripts/benchmark_predictions.py [--output doc/BENCHMARK_SCORECARD.md]
    python scripts/benchmark_predictions.py --csv data/benchmark.csv
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpr.first_principles import SubstrateDerivedTheories
from bpr.experimental_data import (
    MEASUREMENTS,
    Measurement,
    list_measurement_ids,
    summary_counts,
)


# ─── Classification thresholds ───────────────────────────────────────

class Grade:
    MATCH = "MATCH"             # within 2σ or < 5% relative
    CLOSE = "CLOSE"             # within 5σ or < 20% relative
    TENSION = "TENSION"         # within 10x
    FAIL = "FAIL"               # more than 10x off
    BOUND_OK = "BOUND_OK"       # satisfies experimental bound
    BOUND_FAIL = "BOUND_FAIL"   # violates experimental bound
    EXACT_OK = "EXACT_OK"       # matches exact value
    EXACT_FAIL = "EXACT_FAIL"   # fails exact value
    NO_PREDICTION = "NO_PRED"   # prediction key not found


@dataclass
class BenchmarkResult:
    """Result of comparing one BPR prediction to experiment."""
    prediction_id: str
    name: str
    bpr_value: Optional[float]
    exp_value: float
    exp_uncertainty: float
    unit: str
    source: str
    bpr_status: str          # DERIVED / FRAMEWORK / SUSPICIOUS / etc.
    grade: str               # from Grade
    sigma: Optional[float]   # n-sigma deviation (None for bounds)
    rel_deviation: Optional[float]  # |pred - obs| / |obs|
    note: str = ""


def _try_float(val) -> Optional[float]:
    """Attempt to convert a prediction value to float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def grade_result(bpr_val: Optional[float], meas: Measurement) -> BenchmarkResult:
    """Grade a single prediction against its measurement."""
    base = dict(
        prediction_id=meas.prediction_id,
        name=meas.name,
        bpr_value=bpr_val,
        exp_value=meas.value,
        exp_uncertainty=meas.uncertainty,
        unit=meas.unit,
        source=meas.source,
        bpr_status=meas.bpr_status,
    )

    # No numeric prediction available
    if bpr_val is None:
        return BenchmarkResult(**base, grade=Grade.NO_PREDICTION,
                               sigma=None, rel_deviation=None,
                               note="Non-numeric prediction")

    # ── Exact values ─────────────────────────────────────────────
    if meas.is_exact:
        # For integer-valued exact checks (dimensions, etc.)
        if meas.value == int(meas.value) and abs(bpr_val - meas.value) < 0.01:
            return BenchmarkResult(**base, grade=Grade.EXACT_OK,
                                   sigma=0.0, rel_deviation=0.0)
        # For continuous exact values (Tsirelson bound, etc.)
        rel = abs(bpr_val - meas.value) / max(abs(meas.value), 1e-300)
        if rel < 1e-4:
            return BenchmarkResult(**base, grade=Grade.EXACT_OK,
                                   sigma=None, rel_deviation=rel)
        return BenchmarkResult(**base, grade=Grade.EXACT_FAIL,
                               sigma=None, rel_deviation=rel,
                               note=f"Expected exact {meas.value}")

    # ── Bounds ───────────────────────────────────────────────────
    if meas.is_upper_bound:
        ok = bpr_val <= meas.value
        rel = abs(bpr_val - meas.value) / max(abs(meas.value), 1e-300)
        grade = Grade.BOUND_OK if ok else Grade.BOUND_FAIL
        margin = (meas.value - bpr_val) / max(abs(meas.value), 1e-300)
        note = f"margin: {margin:+.1%}" if ok else f"exceeds by {-margin:.1%}"
        return BenchmarkResult(**base, grade=grade,
                               sigma=None, rel_deviation=rel, note=note)

    if meas.is_lower_bound:
        ok = bpr_val >= meas.value
        rel = abs(bpr_val - meas.value) / max(abs(meas.value), 1e-300)
        grade = Grade.BOUND_OK if ok else Grade.BOUND_FAIL
        margin = (bpr_val - meas.value) / max(abs(meas.value), 1e-300)
        note = f"margin: {margin:+.1%}" if ok else f"below by {-margin:.1%}"
        return BenchmarkResult(**base, grade=grade,
                               sigma=None, rel_deviation=rel, note=note)

    # ── Measurements with uncertainty ────────────────────────────
    rel = meas.relative_deviation(bpr_val)
    sigma = meas.sigma_deviation(bpr_val)

    if sigma is not None and sigma <= 2.0:
        grade = Grade.MATCH
    elif sigma is not None and sigma <= 5.0:
        grade = Grade.CLOSE
    elif rel is not None and rel < 0.05:
        grade = Grade.MATCH
    elif rel is not None and rel < 0.20:
        grade = Grade.CLOSE
    elif rel is not None and rel < 10.0:
        grade = Grade.TENSION
    else:
        grade = Grade.FAIL

    return BenchmarkResult(**base, grade=grade, sigma=sigma,
                           rel_deviation=rel)


def run_benchmark(
    p: int = 104729,
    N: int = 10000,
    J_eV: float = 1.0,
    radius: float = 0.01,
    geometry: str = "sphere",
) -> List[BenchmarkResult]:
    """Generate all predictions and benchmark against experiment."""

    # Generate BPR predictions
    sdt = SubstrateDerivedTheories.from_substrate(
        p=p, N=N, J_eV=J_eV, radius=radius, geometry=geometry,
    )
    preds = sdt.predictions()

    results: List[BenchmarkResult] = []
    for mid in sorted(MEASUREMENTS.keys()):
        meas = MEASUREMENTS[mid]
        bpr_raw = preds.get(mid)
        bpr_val = _try_float(bpr_raw)
        results.append(grade_result(bpr_val, meas))

    return results


# ─── Reporting ───────────────────────────────────────────────────────

def grade_symbol(grade: str) -> str:
    """Unicode symbol for display."""
    return {
        Grade.MATCH: "PASS",
        Grade.CLOSE: "CLOSE",
        Grade.TENSION: "TENSION",
        Grade.FAIL: "FAIL",
        Grade.BOUND_OK: "PASS",
        Grade.BOUND_FAIL: "FAIL",
        Grade.EXACT_OK: "PASS",
        Grade.EXACT_FAIL: "FAIL",
        Grade.NO_PREDICTION: "N/A",
    }.get(grade, "?")


def format_value(v: Optional[float], precision: int = 4) -> str:
    """Format a float for display, handling None and large/small values."""
    if v is None:
        return "—"
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1e4 or av < 1e-2:
        return f"{v:.{precision}e}"
    return f"{v:.{precision}f}"


def generate_markdown_scorecard(results: List[BenchmarkResult]) -> str:
    """Generate the BENCHMARK_SCORECARD.md content."""
    lines = []
    lines.append("# BPR Benchmark Scorecard")
    lines.append("")
    lines.append("> **Generated by** `scripts/benchmark_predictions.py`")
    lines.append("> ")
    lines.append("> Compares BPR predictions against published experimental data")
    lines.append("> (PDG 2024, Planck 2018, CODATA 2018, dedicated experiments).")
    lines.append("> ")
    lines.append("> **Grading key:**")
    lines.append("> | Grade | Criterion |")
    lines.append("> |-------|-----------|")
    lines.append("> | PASS | Within 2σ, satisfies bound, or matches exact value |")
    lines.append("> | CLOSE | Within 5σ or < 20% relative deviation |")
    lines.append("> | TENSION | Within 10× but beyond 5σ |")
    lines.append("> | FAIL | More than 10× off or violates bound |")
    lines.append("> | N/A | Non-numeric prediction, cannot compare |")
    lines.append("")

    # ── Summary statistics ────────────────────────────────────────
    total = len(results)
    grade_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for r in results:
        g = grade_symbol(r.grade)
        grade_counts[g] = grade_counts.get(g, 0) + 1
        status_counts[r.bpr_status] = status_counts.get(r.bpr_status, 0) + 1

    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Total benchmarked predictions: {total}**")
    lines.append("")
    lines.append("| Grade | Count | % |")
    lines.append("|-------|-------|---|")
    for g in ["PASS", "CLOSE", "TENSION", "FAIL", "N/A"]:
        c = grade_counts.get(g, 0)
        pct = 100.0 * c / max(total, 1)
        lines.append(f"| {g} | {c} | {pct:.0f}% |")
    lines.append("")

    lines.append("### By BPR derivation status")
    lines.append("")
    lines.append("| Status | Count | Meaning |")
    lines.append("|--------|-------|---------|")
    status_meaning = {
        "DERIVED": "Genuinely predicted from (J, p, N)",
        "FRAMEWORK": "BPR formula, some experimental inputs",
        "SUSPICIOUS": "Likely reverse-engineered from data",
        "CONSISTENT": "Matches, but also predicted by SM/GR",
        "CONJECTURAL": "Not yet testable",
        "OPEN": "BPR does not yet derive this",
    }
    for s in ["DERIVED", "FRAMEWORK", "SUSPICIOUS", "CONSISTENT", "OPEN"]:
        c = status_counts.get(s, 0)
        if c > 0:
            lines.append(f"| {s} | {c} | {status_meaning.get(s, '')} |")
    lines.append("")

    # ── Per-prediction table ──────────────────────────────────────
    # Group results by theory prefix
    theory_names = {
        "P2": "II: Impedance / Dark Sector",
        "P4": "IV: Phase Transitions",
        "P5": "V: Neutrino Physics",
        "P7": "VII: Gravitational Waves",
        "P11": "XI: Cosmology",
        "P12": "XII: QCD & Flavor",
        "P13": "XIII: Emergent Spacetime",
        "P14": "XIV: Topological Matter",
        "P16": "XVI: Quantum Foundations",
        "P17": "XVII: Gauge Unification",
        "P18": "XVIII: Charged Leptons",
        "P19": "XIX: Nuclear Physics",
        "P20": "XX: QG Phenomenology",
    }

    def theory_key(pred_id: str) -> str:
        """Extract theory prefix from prediction ID."""
        parts = pred_id.split(".")
        if len(parts) >= 2:
            base = parts[0]  # e.g. "P5" or "P12"
            return base
        return "P?"

    # Group
    grouped: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        tk = theory_key(r.prediction_id)
        grouped.setdefault(tk, []).append(r)

    lines.append("## Detailed Results")
    lines.append("")

    for tk in sorted(grouped.keys(), key=lambda k: int(k[1:])):
        theory_label = theory_names.get(tk, f"Theory {tk}")
        lines.append(f"### {theory_label}")
        lines.append("")
        lines.append("| ID | Name | BPR | Experiment | σ | Rel. Dev. | Grade | Status | Note |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for r in grouped[tk]:
            bpr_str = format_value(r.bpr_value)
            if r.exp_uncertainty > 0 and not (MEASUREMENTS[r.prediction_id].is_upper_bound
                                               or MEASUREMENTS[r.prediction_id].is_lower_bound):
                exp_str = f"{format_value(r.exp_value)} ± {format_value(r.exp_uncertainty)}"
            elif MEASUREMENTS[r.prediction_id].is_upper_bound:
                exp_str = f"< {format_value(r.exp_value)}"
            elif MEASUREMENTS[r.prediction_id].is_lower_bound:
                exp_str = f"> {format_value(r.exp_value)}"
            else:
                exp_str = format_value(r.exp_value)
            sigma_str = f"{r.sigma:.1f}" if r.sigma is not None else "—"
            rel_str = f"{r.rel_deviation:.1%}" if r.rel_deviation is not None else "—"
            grade_str = grade_symbol(r.grade)

            lines.append(
                f"| {r.prediction_id} | {r.name} | {bpr_str} "
                f"| {exp_str} | {sigma_str} | {rel_str} "
                f"| **{grade_str}** | {r.bpr_status} | {r.note} |"
            )
        lines.append("")

    # ── Honest assessment section ─────────────────────────────────
    lines.append("## Honest Assessment")
    lines.append("")

    # Count pass/fail by status
    derived_results = [r for r in results if r.bpr_status == "DERIVED"]
    framework_results = [r for r in results if r.bpr_status == "FRAMEWORK"]
    suspicious_results = [r for r in results if r.bpr_status == "SUSPICIOUS"]

    derived_pass = sum(1 for r in derived_results
                       if grade_symbol(r.grade) in ("PASS",))
    framework_pass = sum(1 for r in framework_results
                         if grade_symbol(r.grade) in ("PASS", "CLOSE"))
    suspicious_pass = sum(1 for r in suspicious_results
                          if grade_symbol(r.grade) in ("PASS",))

    lines.append("### DERIVED predictions (genuine BPR, no hand-tuning)")
    lines.append("")
    if derived_results:
        lines.append(f"- {derived_pass}/{len(derived_results)} pass benchmarks")
        for r in derived_results:
            status = grade_symbol(r.grade)
            sigma_info = f", {r.sigma:.1f}σ" if r.sigma is not None else ""
            rel_info = f", {r.rel_deviation:.1%} off" if r.rel_deviation is not None else ""
            lines.append(f"  - {r.prediction_id} ({r.name}): **{status}**{sigma_info}{rel_info}")
    lines.append("")

    lines.append("### FRAMEWORK predictions (BPR formula, some exp. input)")
    lines.append("")
    if framework_results:
        lines.append(f"- {framework_pass}/{len(framework_results)} pass or close")
        for r in framework_results:
            status = grade_symbol(r.grade)
            sigma_info = f", {r.sigma:.1f}σ" if r.sigma is not None else ""
            rel_info = f", {r.rel_deviation:.1%} off" if r.rel_deviation is not None else ""
            lines.append(f"  - {r.prediction_id} ({r.name}): **{status}**{sigma_info}{rel_info}")
    lines.append("")

    lines.append("### SUSPICIOUS predictions (likely fitted to data)")
    lines.append("")
    if suspicious_results:
        lines.append(f"- {suspicious_pass}/{len(suspicious_results)} match experiment")
        lines.append("- **These matches do NOT count as evidence for BPR** because the")
        lines.append("  parameters were tuned to reproduce known values.")
        for r in suspicious_results:
            status = grade_symbol(r.grade)
            sigma_info = f", {r.sigma:.1f}σ" if r.sigma is not None else ""
            lines.append(f"  - {r.prediction_id} ({r.name}): **{status}**{sigma_info}")
    lines.append("")

    return "\n".join(lines)


def generate_csv(results: List[BenchmarkResult], path: str):
    """Write benchmark results as CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prediction_id", "name", "bpr_value", "exp_value",
            "exp_uncertainty", "unit", "source", "bpr_status",
            "grade", "sigma", "rel_deviation", "note",
        ])
        for r in results:
            writer.writerow([
                r.prediction_id, r.name,
                r.bpr_value if r.bpr_value is not None else "",
                r.exp_value, r.exp_uncertainty, r.unit, r.source,
                r.bpr_status, r.grade,
                f"{r.sigma:.2f}" if r.sigma is not None else "",
                f"{r.rel_deviation:.6f}" if r.rel_deviation is not None else "",
                r.note,
            ])
    print(f"CSV written to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BPR predictions against experiment")
    parser.add_argument("--output", "-o", default="doc/BENCHMARK_SCORECARD.md",
                        help="Output markdown path")
    parser.add_argument("--csv", default=None,
                        help="Also write CSV results")
    parser.add_argument("--p", type=int, default=104729)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--geometry", default="sphere")
    args = parser.parse_args()

    print("Running BPR prediction benchmark...")
    print("=" * 70)

    results = run_benchmark(
        p=args.p, N=args.N, J_eV=args.J,
        radius=args.radius, geometry=args.geometry,
    )

    # Print summary to console
    total = len(results)
    passes = sum(1 for r in results if grade_symbol(r.grade) == "PASS")
    close = sum(1 for r in results if grade_symbol(r.grade) == "CLOSE")
    tension = sum(1 for r in results if grade_symbol(r.grade) == "TENSION")
    fails = sum(1 for r in results if grade_symbol(r.grade) == "FAIL")

    print(f"\nBenchmark complete: {total} predictions compared")
    print(f"  PASS:    {passes}")
    print(f"  CLOSE:   {close}")
    print(f"  TENSION: {tension}")
    print(f"  FAIL:    {fails}")
    print()

    # Print each result
    for r in results:
        sym = grade_symbol(r.grade)
        sigma_str = f"({r.sigma:.1f}σ)" if r.sigma is not None else ""
        rel_str = f"[{r.rel_deviation:.1%}]" if r.rel_deviation is not None else ""
        print(f"  {sym:8s} {r.prediction_id:<35s} "
              f"BPR={format_value(r.bpr_value):>14s}  "
              f"Exp={format_value(r.exp_value):>14s}  "
              f"{sigma_str:>8s} {rel_str:>8s}  "
              f"[{r.bpr_status}] {r.note}")

    # Write markdown scorecard
    md = generate_markdown_scorecard(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"\nScorecard written to: {args.output}")

    # Write CSV if requested
    if args.csv:
        generate_csv(results, args.csv)


if __name__ == "__main__":
    main()
