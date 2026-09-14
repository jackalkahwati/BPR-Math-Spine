#!/usr/bin/env python3
"""Consistently dressed neutral response; stdout only, run anywhere with --json."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_neutral_effective import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional second-order all-D neutral effective response")
    print("Model:", report["model_id"])
    print("N=L=q is assumed; finite-ring lines do not establish a bound particle.")
    for case in report["cases"]:
        print("\nFROZEN CASE:", json.dumps(case["parameters"], allow_nan=False))
        print("Dimensions:", json.dumps(case["dimensions"], allow_nan=False))
        cert, effective, checks = (case[key] for key in ("remainder_certificate", "effective", "numerical_checks"))
        print("Exact ground / second-order shift:", case["ground_energy"], effective["ground_shift"])
        print("Block identification available:", cert["block_identification_available"])
        print("RH / g RH / Rrho:", cert["RH"], cert["energy_bound"], cert["Rrho"])
        print("Global sorted eigenvalue error:", checks["global_sorted_eigenvalue_max_error"])
        print("Ground-vector / source-error bounds:", cert["ground_vector_error_bound"], cert["source_error_bound"])
        print("Exact / squared-truncated total density weight:",
              case["exact_density_measure"]["total_weight"], effective["full_density_measure"]["total_weight"])
        print("Source orders (quartic term is incomplete):", json.dumps(effective["source_weights"], allow_nan=False))
        print("Exact sum rules:", json.dumps(case["exact_sum_rules"], allow_nan=False))
        print("Effective sum rules:", json.dumps(effective["full_sum_rules"], allow_nan=False))
        print("Dynamics check:", json.dumps(checks["dynamics"], allow_nan=False))
        if cert["reason"]:
            print(cert["reason"])
        print("Partially resummed P1 gaps:", json.dumps(effective["P1_gaps"], allow_nan=False))
        print("P1 grouped weights:", json.dumps(effective["P1_density_measure"]["groups"], allow_nan=False))
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
