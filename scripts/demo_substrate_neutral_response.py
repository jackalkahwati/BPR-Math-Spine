#!/usr/bin/env python3
"""Conditional neutral density response, stdout only; run anywhere with --json."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_neutral_response import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional neutral finite-repulsion excitation and density diagnostic")
    print("Model:", report["model_id"])
    print("N=L=q is assumed, not a derived vacuum or confinement mechanism.")
    for case in report["cases"]:
        print("\nFROZEN CASE:", json.dumps(case["parameters"], allow_nan=False))
        print("Dimensions:", json.dumps(case["dimensions"], allow_nan=False))
        density = case["density_certificate"]
        print("Excited cluster certificate available:", case["excited_certificate"]["available"])
        print("Exact total density weight:", case["exact_density_measure"]["total_weight"])
        print("Leading total density weight:", density["leading_weight"])
        print("Conservative total-weight error bound:", density["total_weight_error_bound"])
        if density["available"] and case["parameters"]["m"] != 0:
            print("Leading signal:", "resolved by this bound" if density["leading_signal_resolved_by_bound"] else "UNRESOLVED by this bound; no parameter tuning")
        elif case["parameters"]["m"] == 0:
            print("Structural zero-density mode control")
        else:
            print("Sufficient certificate unavailable, not disproved.")
        print("Elastic weight numerical residual:", case["numerical_checks"]["elastic_weight"])
        print("Total-weight sum-rule residual:", case["numerical_checks"]["total_weight_sum_rule_residual"])
        print("First-moment sum-rule residual:", case["numerical_checks"]["first_moment_sum_rule_residual"])
        if case["leading_compression_measure"] is None:
            print(case["leading_measure_suppressed_reason"])
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
