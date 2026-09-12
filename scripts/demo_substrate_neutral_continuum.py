#!/usr/bin/env python3
"""Leading neutral compression continuum, stdout only; run anywhere with --json."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_neutral_continuum import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional neutral compression continuum diagnostic")
    print("Model:", report["model_id"])
    print("Exact compressed-model theorem; numerical evaluations exclude certified roundoff.")
    for case in report["cases"]:
        print("\nFROZEN CASE:", json.dumps(case["parameters"], allow_nan=False))
        print("Sequence:", case["limit_sequence"])
        measure = case["compression_measure"]
        audit = case["validity_audit"]
        print("Dimensionless centered support:", measure["dimensionless_support"])
        print("Compression energy range:", measure["energy_range"])
        print("Leading total density weight:", measure["total_weight"])
        print("Population-selection sufficient condition:", audit["population_selection_sufficient"])
        print("Finite-size excitation-separation sufficient condition:", audit["excitation_separation_sufficient"])
        if case["numerical_checks"] is not None:
            checks = case["numerical_checks"]
            print("Numerical mass residual:", checks["mass_residual"])
            print("Maximum line fraction:", checks["max_line_fraction"])
            print("Analytic line-fraction upper bound:", checks["line_fraction_bound"])
            print("Dimensionless Lipschitz-one weak-limit bound:", case["weak_limit_bound"])
        else:
            print(measure["normalization_reason"])
        for key in ("total_weight_reason", "absolute_weight_reason", "energy_reason"):
            if measure[key]:
                print(key + ":", measure[key])
    print("\nFROZEN SCALAR THRESHOLD AUDITS")
    for row in report["frozen_threshold_audits"]:
        print(json.dumps(row["parameters"], allow_nan=False),
              "excitation separation:", row["audit"]["excitation_separation_sufficient"])
    print("\nLIMITS")
    for name, statement in report["limit_statements"].items():
        print(name + ":", statement)
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
