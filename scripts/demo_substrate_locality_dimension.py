#!/usr/bin/env python3
"""Print the frozen locality/dimension counting controls to stdout only."""

import sys

# Keep the stdout-only demo from writing import bytecode caches.
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path


# Resolve the sibling module without executing the scientific package initializer.
# This also supports absolute script invocation with no PYTHONPATH from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))

from substrate_locality_dimension import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit only the strict JSON report"
    )
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return

    print("Conditional locality and dimension: exact counting diagnostics")
    print("Source: integer lattice balls; target: a simple cycle.")
    print("Growth cases ({})".format(len(report["growth_cases"])))
    for case in report["growth_cases"]:
        dilation = case["dilation"]
        print(
            "  d={} r={} L={} K={}/{} k={} m={}: "
            "source={} target_ball={} capacity={} signed_excess={} {}".format(
                case["dimension"],
                case["radius"],
                case["target_sites"],
                dilation["numerator"],
                dilation["denominator"],
                dilation["effective_integer"],
                case["multiplicity"],
                case["source_count"],
                case["target_ball_count"],
                case["capacity"],
                case["signed_excess"],
                case["status"],
            )
        )

    print("Finite-box controls ({})".format(len(report["finite_box_controls"])))
    for case in report["finite_box_controls"]:
        print(
            "  shape={} center={} r={} count={}".format(
                case["shape"], case["center"], case["radius"], case["count"]
            )
        )
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)


if __name__ == "__main__":
    main()
