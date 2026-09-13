#!/usr/bin/env python3
"""Bounded hard-core band, charge and flux diagnostics; stdout only.

Run from any directory. --json emits the complete strict-JSON report.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_chirality import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional hard-core branches and signed flux flow, not physical chirality")
    print("Model:", report["model_id"])
    for case in report["cases"]:
        p = case["parameters"]
        ground = case["ground_filling"]
        flow = case["flux_crossings"]
        print(f"L={p['L']} N={p['N']} phi={p['phi']:.8g} mu={p['mu']:.8g}: "
              f"E0={ground['ground_energy']:.8g}, cut gap={ground['occupation_gap']}, "
              f"numerical ground degeneracy={ground['ground_degeneracy_numerical']}, "
              f"signed flow={flow['net_signed_flow']}")
    for control in report["finite_g_controls"]:
        print("Finite-g existing negative control:", control["g"])
        for witness in control["virtual_witnesses"]:
            print(f"  {witness['initial']} -> {witness['final']}: "
                  f"{witness['matrix_element']:.8g} "
                  f"({witness['coefficient_in_C_squared_over_g']:.8g} C^2/g)")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
