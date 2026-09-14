#!/usr/bin/env python3
"""Frozen complete-ring energy response and partition controls; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_energy_response import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional microscopic energy continuity and response")
    print("Model:", report["model_id"])
    for case in report["cases"]:
        p = case["parameters"]
        print(f"L={p['L']} N={p['N']} C={p['C']} g={p['g']} dimension={p['dimension']}")
        for partition in case["partitions"]:
            d = partition["diagnostics"]
            print(f"  {partition['partition']}: continuity residual={d['continuity_max_residual']:.3g}, "
                  f"range2 witness norm={d['range2_witness_norm']:.6g}")
            for mode in partition["momenta"]:
                ward = max(max(r["ward_first_residual"], r["ward_second_residual"],
                               r["ward_combined_residual"]) for r in mode["responses"])
                fsum = max(r["fsum_residual"] for r in mode["responses"])
                print(f"    m={mode['m']}: Ward residual={ward:.3g}, f-sum residual={fsum:.3g}")
        strict = case["partition_change"]["strict_L3_m1_imaginary_z_difference"]
        if strict is not None:
            print(f"  L3 m1 imaginary-frequency partition difference={strict:.9g}")
        for shift in case["identity_shifts"]:
            print(f"  Identity shift {shift['kappa']}: response residual={shift['response_max_residual']:.3g}")
    print("Numerical policies and limitations")
    for statement in report["numerical_policy"] + report["limitations"]:
        print(statement)


if __name__ == "__main__":
    main()
