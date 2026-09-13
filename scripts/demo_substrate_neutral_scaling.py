#!/usr/bin/env python3
"""Joint neutral scaling bounds, stdout only; use --json for strict JSON."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_neutral_scaling import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional full-model neutral joint scaling")
    print("Model:", report["model_id"])
    print("Theorem status:", report["theorem_status"])
    print("Analytical inequalities exclude roundoff; not numerical enclosures.")
    for case in report["cases"]:
        print("\n" + case["limit_sequence"])
        print("L={L}, m={m}, g={g:g}, t={time:g}".format(**case))
        print("Excitation separation:", case["separation_sufficient"])
        print("Source normalization gate:", case["source_normalization_sufficient"])
        print("Source norm / error / relative error:", case["source_norm"],
              case["source_error_bound"], case["relative_source_error"])
        print("Generator / ground-shift bounds:", case["generator_bound"], case["ground_shift_bound"])
        print("Full/compression bound:", case["full_compression_bound"])
        print("Full/semicircle bound:", case["full_limit_bound"])
        if case["unavailable_reason"]:
            print(case["unavailable_reason"])
    print("\nFIXED CIRCUMFERENCE, m=1, lambda=L^-5")
    for row in report["spatial_scaling"]:
        print("L={L}: support-edge shift={threshold_shift:.9g}, quadratic={quadratic_threshold_shift:.9g}, "
              "physical generator bound={physical_generator_bound:.9g}, "
              "finite minimum offset={finite_compression_edge_offset:.9g}".format(**row))
    print("The continuum support edge is not the finite compression minimum or first bright line.")
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("*", limitation)
    print("Physical predictions:", json.dumps(report["physical_predictions"], allow_nan=False))


if __name__ == "__main__":
    main()
