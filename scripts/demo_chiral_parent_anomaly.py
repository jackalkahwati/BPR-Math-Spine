#!/usr/bin/env python3
"""Exact supplied six-dimensional Weyl16 local anomaly controls, stdout only.

Run from any directory; --json emits deterministic strict JSON with exact
rational coefficient records. No files, fits, simulations, or sends occur.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.chiral_parent_anomaly import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict exact JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return
    print("Supplied 6D complex Weyl Spin(10) 16: exact local anomaly")
    print("Model:", report["model_id"])
    print("I8(s=+1):", report["symbolic"]["parent"]["expression"])
    print("Formal I6 per signed family (s=+1):", report["symbolic"]["lower"]["expression"])
    print("All six parent contributions (s=+1)")
    for name, polynomial in report["symbolic"]["contributions"].items():
        print("  {}: {}".format(name, polynomial["expression"]))
    print("Irreducible p2 coefficient = -s/90, nonzero for one supplied complex Weyl16")
    print("Fixed charge lattice; Q is charge, m is sphere flux, q=Q*m")
    for case in report["cases"]:
        parameters = case["parameters"]
        modes = case["bookkeeping"]
        print("Q={charge} m={flux} q={q} s={chirality}".format(**parameters))
        print("  Parent:", case["parent"]["expression"])
        print("  Sphere pushforward:", case["pushforward"]["expression"])
        print("  Net left families={}, gauge components={}; not SM matter derivation".format(
            modes["net_left_families"], modes["net_left_gauge_components"],
        ))
        print("  Exact case controls:", "pass" if all(case["controls"].values()) else "FAIL")
    print("Exact global controls")
    for name, passed in report["controls"].items():
        print("  {}: {}".format(name, "pass" if passed else "FAIL"))
    print("Charge/flux cap:", report["resource_policy"]["description"])
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
