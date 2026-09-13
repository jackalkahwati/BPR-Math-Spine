#!/usr/bin/env python3
"""Frozen occupation-orbit gauge encoding diagnostics; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_gauge_encoding import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Fixed occupation-orbit gauge encoding")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    print("Controls: L=N=3,4,5; C=1; g=0,0.7,40; Ct=0,0.01,0.1")
    for case in report["cases"]:
        leakage, target = case["leakage"], case["target"]
        print(f"L={case['L']} N={case['N']} g={case['g']}: {case['status']}")
        if case["reason"]:
            print("  Reason:", case["reason"])
        print(f"  Leakage generator norm observed={leakage['norm']}; "
              f"analytic reference={leakage['analytic_norm']}")
        print(f"  Analytic all-states-leak claim: {leakage['all_states_leak']}")
        print(f"  Target: {target['status']}; {target['reason']}")
        if case["L"] == 5:
            print(f"  Target mismatch observed={target['generator_mismatch_norm']}; "
                  f"analytic reference={target['analytic_generator_mismatch_norm']}")
        for time_record in case["times"]:
            for name in ("compression", "target"):
                result = time_record[name]
                if result["status"] == "not_applicable":
                    continue
                print(f"  Ct={time_record['tau']} {name}: {result['status']}; "
                      f"full-space error={result['full_space_error']}; "
                      f"projected error={result['projected_error']}; "
                      f"leakage amplitude={result['leakage_amplitude']}")
                print(f"    Exact-model envelope=[{result['analytic_lower_bound']}, "
                      f"{result['analytic_upper_bound']}]")
                for key in ("upper_bound_comparison", "lower_bound_comparison"):
                    comparison = result[key]
                    print(f"    {key}: {comparison['status']}; {comparison['reason']}")
                if result["reason"]:
                    print("    Reason:", result["reason"])
    print("Case status counts:", report["summary"]["case_status_counts"])
    print("D5 analytic obstruction reference claims, not numerical successes:",
          report["summary"]["analytic_d5_obstruction_count"])
    print("This is a diagnostic encoding, not local gauge emergence or empirical validation.")
    print("Exact-model bounds exclude numerical error. Regression checks are not certificates.")
    print("Leakage is an amplitude, not a probability. No phase, scale or energy offset is fitted.")


if __name__ == "__main__":
    main()
