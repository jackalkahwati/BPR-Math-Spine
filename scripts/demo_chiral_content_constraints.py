#!/usr/bin/env python3
"""Stdout-only fixed anomaly/reduction demonstration, including from empty cwd."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


# Avoid package-wide optional imports and bytecode output. Resolve by script
# location, not cwd, so the demo does not require installation or local files.
sys.dont_write_bytecode = True
_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "bpr" / "chiral_content_constraints.py")
_SPEC = importlib.util.spec_from_file_location(
    "_chiral_content_constraints_demo", str(_MODULE_PATH))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _vector(values):
    return "(" + ", ".join(str(value) for value in values) + ")"


def _text_report(report):
    print("Fixed-content anomaly constraints and restricted sphere reduction")
    print("Exact Fraction arithmetic; no numerical rank or parameter search.")
    for name in ("without_singlet", "with_singlet"):
        system = report["anomaly_systems"][name]
        print("\nAnomaly system " + name)
        print("  columns = " + _vector(system["columns"]))
        for label, row in zip(system["row_labels"], system["matrix"]):
            print("  " + label + " = " + _vector(row))
        print("  rank = {rank}; nullity = {nullity}".format(**system))
        print("  kernel = " + "; ".join(_vector(vector)
                                         for vector in system["kernel_basis"]))
        print("  rank minor determinant = " + system["rank_minor_determinant"])
    print("\nFixed multiplicity controls")
    for case in report["multiplicity_cases"]:
        print("  " + _vector(case["multiplicities"])
              + " local_free=" + str(case["local_anomaly_free"])
              + " Witten_parity=" + str(case["witten_parity"])
              + " tested_consistent="
              + str(case["fully_consistent_with_tested_constraints"])
              + " family_count=" + str(case["family_count"])
              + " singlet_count=" + str(case["singlet_count"]))
        print("    anomalies = " + json.dumps(case["anomalies"], allow_nan=False))
    print("\nRestricted reduction controls")
    for case in report["reduction_cases"]:
        system = case["system"]
        print("  flux = {flux}; rank = {rank}; nullity = {nullity}".format(**system))
        print("    domain = " + _vector(system["domain_basis"]))
        print("    codomain = " + _vector(system["codomain_basis"]))
        print("    formal integer-image steps = "
              + _vector(system["integer_image_steps"])
              + "; index = " + str(system["integer_image_index"]))
        print("    basis images = " + "; ".join(_vector(vector)
                                                for vector in case["basis_images"]))
        print("    kernel images = " + "; ".join(_vector(vector)
                                                 for vector in case["kernel_images"]))
    print("\nFixed charge=1, chirality=+1 parent")
    for case in report["parent_cases"]:
        print("  flux = " + str(case["flux"])
              + "; parent_nonzero = " + str(case["parent_nonzero"])
              + "; reduced_nonzero = " + str(case["reduced_nonzero"]))
        print("    parent = " + _vector(case["parent_coefficients"]))
        print("    reduced = " + _vector(case["reduced_coefficients"]))
        print("    invisible = " + _vector(case["invisible_coefficients"]))
        print("    invisible pushforward = " + _vector(case["invisible_pushforward"]))
        print("    p2 = " + case["p2_coefficient"]
              + "; split = " + case["split_convention"])
        print("    restricted_reduction_certifies_parent = "
              + str(case["restricted_reduction_certifies_parent"]))
    print("\nScope and limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    print("\nArithmetic domain")
    for key, value in report["arithmetic_domain"].items():
        print("  " + key + " = " + str(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit strict JSON only")
    args = parser.parse_args(argv)
    report = _MODULE.demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _text_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
