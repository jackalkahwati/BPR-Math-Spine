#!/usr/bin/env python3
"""Stdout-only fixed joint-source demonstration; run from any working directory."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_joint_source_kernel import demonstration_report


def _title(case):
    return "L={L} g={g} C={C} m={m}".format(**case)


def _text(report):
    print("Joint matter-source kernel and source-contact ambiguity")
    print("Complete finite Bose rings, fixed controls, energy E (not E/C).")
    print("Physical q=(C f_rho,f_h); D=diag(C,1); K_f=D K_q D.")
    print("No numerical error certificate, graviton identification or empirical validation.")
    for case in report["joint_cases"] + [report["dark_control"]]:
        print("\nJoint " + _title(case))
        if case.get("status") == "numerical_unavailable":
            print("  numerical_unavailable: " + case["reason"])
            continue
        print("  ground gap={gap:.12g}, resolution={resolution:.6g}".format(**case))
        print("  uniform controls " + json.dumps(case["uniform_controls"], allow_nan=False))
        for name, partition in case["partitions"].items():
            print("  " + name)
            print("    dimensionless Gram eigenvalues " + str(partition["gram_eigenvalues_dimensionless"]))
            print("    dimensionless static Hessian " + str(partition["static_hessian_dimensionless"]))
            print("    raw closure residual " + str(partition["closure_residual"]))
            free = partition["free_control"]
            if free is not None:
                print("    analytic free rank={analytic_rank}, active source gap={expected_gap:.12g}, energy dark={energy_dark}".format(**free))
            for response in partition["responses"]:
                print("    z=" + str(response["z"]) + " kernel=" + json.dumps(response["kernel_physical"], allow_nan=False))
    for case in report["finite_difference_cases"]:
        print("\nFinite differences " + _title(case))
        if case.get("status") == "numerical_unavailable":
            print("  numerical_unavailable: " + case["reason"])
            continue
        print("  model gap lower bound " + str(case["model_gap_lower_bound"]))
        for name, partition in case["partitions"].items():
            print("  " + name + " contact=" + str(partition["contact_matrix"]))
            for step in partition["steps"]:
                print("    step=" + str(step["step"]) + " raw stencil=" + str(step["stencil"]))
                print("    absolute error=" + str(step["absolute_error"]))
                print("    subtraction proxy=" + str(step["roundoff_proxy"]) + " status=" + str(step["roundoff_status"]))
                for key in ("model_bounds", "heuristic_bounds"):
                    print("    " + key + "=" + json.dumps(step[key], allow_nan=False))
                print("    point statuses=" + str([point["evaluation_status"] for point in step["points"]]))
                print("    isolation statuses=" + str([point["isolation_status"] for point in step["points"]]))
                print("    contact-shifted stencil=" + str(step["contact_shifted_stencil"]))
                for point in step["points"]:
                    if point["evaluation_status"] == "numerical_unavailable":
                        print("    unavailable point " + str(point["f"]) + ": " + point["reason"])
    print("\nLimitations")
    for limitation in report["limitations"]:
        print("  " + limitation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit strict JSON instead of text")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _text(report)


if __name__ == "__main__":
    main()
