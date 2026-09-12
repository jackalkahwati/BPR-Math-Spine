#!/usr/bin/env python3
"""Neutral population stability in the stipulated Bose ring; stdout only.

Run from any directory. --json emits the strict, lossless structured report.
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

from bpr.substrate_vacuum_selection import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional neutral vacuum-sector selection, not physical vacuum emergence")
    print("Model:", report["model_id"])
    print("Unshifted H = g D + V; number conservation supplies no preparation mechanism.")
    print("Only exact rational bounds certify comparisons; numerics are diagnostics.")
    for case in report["cases"]:
        print("\nFROZEN CASE:", json.dumps(case["parameters"], allow_nan=False))
        print("Status:", case["status"])
        print("Selected number sector:", case["selected_sector"])
        print("Surviving candidates:", case["surviving_sectors"])
        print("Unit filling:", case["unit_filling_status"])
        print("Best neutral trial:", json.dumps(case["best_trial"], allow_nan=False))
        print("Tail:", json.dumps(case["tail"], allow_nan=False))
        for sector in case["sectors"]:
            print(f"  N={sector['N']}: {sector['status']}; "
                  f"numerical={sector['numerical']['status']}")
    print("\nDIRECT N>L WITNESS:", json.dumps(report["direct_above_filling_witness"], allow_nan=False))
    print("\nDIFFERENT SHIFTED HAMILTONIAN:")
    for shifted in report["shifted_convention"]:
        print(shifted["status"], "selected sector", shifted["selected_sector"])
        print(shifted["proof"])
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
