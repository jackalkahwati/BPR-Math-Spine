#!/usr/bin/env python3
"""Exact extensive stability bounds and bounded heuristic diagnostics; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_extensive_stability import demonstration_report


def _fraction(record):
    return record["numerator"] + "/" + record["denominator"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Volume-uniform extensive stability")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    if report["reason"]:
        print("Reason:", report["reason"])
    print("Scalar controls: L=10,1000,1000000; rho=1/2,1,2; C=1; g=7/10,40")
    for case in report["scalar_cases"]:
        bounds = case["bounds"]
        print(f"L={bounds['L']} N={bounds['N']} g={_fraction(bounds['g'])}: "
              f"{case['status']} ({case['parameter_semantics']})")
        print(f"  Per-site bounds=[{_fraction(bounds['lower_per_site'])}, "
              f"{_fraction(bounds['best_upper_per_site'])}]; "
              f"best trial={bounds['best_trial']}")
        print("  All-population lower bound:", _fraction(bounds["all_population_lower"]))
    print("Dense controls: L=N=3,4,5; C=1; g=0,0.7,40 (binary64 lifted exactly)")
    for case in report["dense_cases"]:
        print(f"L={case['L']} N={case['N']} g={case['g']}: {case['status']}")
        if case["reason"]:
            print(f"  Reason ({case['reason_code']}): {case['reason']}")
        if case["spectrum"] is not None:
            print("  Ground energy:", case["spectrum"]["ground_energy"])
        if case["comparison"] is not None:
            comparison = case["comparison"]
            print(f"  Signed margins (E/C): lower={_fraction(comparison['lower_margin'])}; "
                  f"upper={_fraction(comparison['upper_margin'])}; "
                  f"heuristic proxy={_fraction(comparison['proxy'])}")
    print("Dense status counts:", report["summary"]["dense_status_counts"])
    print("Free case:", report["free_classification"]["scope"])
    exclusion = report["unit_filling_exclusion"]
    print("Unit-filling exclusion:", exclusion["scope"])
    for witness in exclusion["witnesses"]:
        print(f"  {witness['parameter_semantics']}: excluded={witness['excluded']}; "
              f"strict margin={_fraction(witness['strict_margin'])}")
    print(report["scope"])
    print("Numerical screens and comparison allowances are heuristic, not certificates.")


if __name__ == "__main__":
    main()
