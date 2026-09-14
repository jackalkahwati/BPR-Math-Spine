#!/usr/bin/env python3
"""Stdout-only fixed internal-sphere diagnostic; no artifacts or fitted inputs."""
import argparse
import json
from pathlib import Path
import sys

# An absolute script path works even when cwd is empty or outside this repo.
# Disable bytecode before importing the scientific package: the demo writes no files.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.chiral_source_compatibility import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete strict JSON report")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2))
        return
    print("Supplied internal sphere source compatibility")
    print("Status: " + report["status"])
    print("Reason: " + report["reason"])
    print("q=3, R=1; sources one, z, p2; epsilon=0, 0.01, 0.1")
    print("Internal chirality only; no spacetime chirality or Standard Model derivation.")
    print("No full Dirac solver, certified numerical error, or empirical validation.")
    print("Conjugation transports the kernel; scalar addition has first-order shifts.")
    print("The formal second-order zero follows from chirality, not leakage smallness.")
    print("Raw matrices and negative eigenvalues are not repaired or clipped.")
    for case in report["cases"]:
        grid = case["quadrature"]
        print("\n{} at {}x{}: {}".format(case["source_id"], grid["n_polar"],
                                         grid["n_azimuth"], case["status"]))
        print("  Reason: " + case["reason"])
        print("  Raw Gram error: {}".format(case["gram"]["error_from_identity"]))
        print("  G eigenvalues (heuristic): {}".format(case["leakage_spectrum"]["eigenvalues"]))
        print("  Analytic first-order coefficients: {}".format(
            case["analytic"]["first_order_shift_coefficients"]))
        for phase in case["phases"]:
            print("  epsilon={}: {}; leading-order error={}; literal upper inequality={}".format(
                phase["epsilon"], phase["status"], phase["leading_order_error"],
                phase["upper_bound_comparison"]["literal_inequality"]))
            print("    " + phase["reason"])
    print("\nRefinement comparisons")
    for comparison in report["refinement_comparisons"]:
        print("  {}: {}; {}".format(comparison["source_id"], comparison["status"],
                                    comparison["reason"]))
    print("\nSummary: " + json.dumps(report["summary"], allow_nan=False, sort_keys=True))
    print("Use --json for all raw complex matrices, spectra and comparison records.")


if __name__ == "__main__":
    main()
