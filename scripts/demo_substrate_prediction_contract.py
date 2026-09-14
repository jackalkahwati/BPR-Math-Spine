#!/usr/bin/env python3
"""Stdout-only frozen scaling and synthetic held-out demonstration."""
import argparse
import json
from pathlib import Path
import sys

# Direct invocation works from any working directory; do not create artifacts.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_prediction_contract import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit strict JSON instead of text")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2))
        return
    print("Same-model prediction and identifiability contract")
    print("Independent complete Bose rings; multiplicity-counted ground gaps.")
    print("27 owned cases, with the selected baseline reused for held-out controls.")
    for grid in report["scaling_cases"]:
        print("L=N={L}, g/C={g_over_C:g}".format(**grid))
        for slot, comparison in zip(grid["cases"], grid["comparisons"]):
            print("  scale={:g}: construction={}, comparison={}".format(
                slot["scale"], slot["status"], comparison["status"]))
            if slot["reason"] is not None:
                print("    " + slot["reason"])
            if slot["report"] is not None:
                observables = slot["report"]["observables"]
                for name in ("gap_ratio", "density_moment", "energy_moment"):
                    item = observables[name]
                    print("    {}: {} ({})".format(name, item["value"], item["status"]))
                for eta, item in zip((0.5, 1.0, 2.0), observables["joint_ratios"]):
                    print("    Q eta={:g}: {} ({}) proxy={}".format(
                        eta, item["value"], item["status"], item["proxy"]))
    heldout = report["heldout"]
    print("Synthetic held-out detection: " + heldout["detection_status"])
    print("  " + heldout["detection_reason"])
    if heldout["development"] is not None:
        for name in ("positive", "negative"):
            print("  {} development: {}".format(name, ", ".join(
                item["status"] for item in heldout["development"][name])))
            print("  {} heldout: {}".format(name, heldout["heldout"][name]["status"]))
    print("Missing measurement requirements")
    for item in report["measurement_requirements"]:
        print("  " + item)
    print("Limitations")
    for item in report["limitations"]:
        print("  " + item)
    print("empirical_test_unavailable; numerical_error_certified=false; empirical_validation=false")


if __name__ == "__main__":
    main()
