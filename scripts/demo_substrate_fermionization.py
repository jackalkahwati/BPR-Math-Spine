#!/usr/bin/env python3
"""Conditional bosonic-ring fermionization; stdout only, no particle prediction.

Run from any directory. Add --json for the complete strict structured report.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_fermionization import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional one-dimensional fermionization, not physical fermion emergence")
    print("Model:", report["model_id"])
    print("Finite repulsion adds virtual corrections to the hard-core dictionary.")
    for case in report["cases"]:
        print("\nFROZEN CASE:", json.dumps(case["parameters"], allow_nan=False))
        print("Dimensions:", json.dumps(case["dimensions"], allow_nan=False))
        print("Fermionic boundary twist:", case["boundary_twist"])
        print("Certificate:", json.dumps(case["certificate"], indent=2, allow_nan=False))
        correction = case["correction"]
        print("Virtual correction norm:", correction["operator_norm"])
        print("Correction resolved in total:", correction["total_addition_resolved"])
        for witness in correction["virtual_witnesses"]:
            print(f"  {witness['initial']} -> {witness['final']}: "
                  f"element={witness['matrix_element']:.8g}, "
                  f"coefficient={witness['coefficient_in_C_squared_over_g']:.8g} C²/g")
        neutrality = case["neutrality"]
        print("Comparison sector neutral:", neutrality["comparison_sector_neutral"])
        print("Allowed hard-core sectors:", json.dumps(neutrality["sectors"], allow_nan=False))
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
