#!/usr/bin/env python3
"""Print the Green-Schwarz Dirac-quantization test of the BPR-6D matter sector."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from green_schwarz_quantization import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Parent charge 1: I8 = {}".format(report["integral_polynomial_q1"]))
    print("  Gram requirements {} -> necessary conditions pass: {}".format(
        report["necessary_q1"]["gram"], report["necessary_q1"]["pass"]))
    print("Parent charge 3: I8 = {}".format(report["integral_polynomial_q3"]))
    print("  U-lattice solution: Y_e = {}, Y_g = {}".format(report["hyperbolic_q3"]["Y_e"], report["hyperbolic_q3"]["Y_g"]))
    print("  lattice vectors: {}".format(report["lattice_vectors_q3"]))
    print("  odd lattice I_(1,1): {}".format(report["odd_lattice_q3"]["reason"]))
    print("Scan q <= 12 (parent charge, passes, families at unit flux):")
    print("  " + ", ".join("({parent_charge}, {U}, {families_at_unit_flux})".format(**row) for row in report["minimal_scan"]))
    print("Family number: {}".format(report["family_number"]))
    print("Unit-flux landscape: {}".format(report["unit_flux_landscape"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
