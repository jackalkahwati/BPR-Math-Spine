#!/usr/bin/env python3
"""Print the brane and bulk-vector Yukawa analysis for the BPR-6D flux families."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from yukawa_mechanisms import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("J_z-invariant symmetric Yukawas (one brane): dimension {}".format(report["jz_invariant_dimension"]))
    print("Single-brane spectra (degenerate pair): {}".format([[round(x, 4) for x in s] for s in report["single_brane_spectra"]]))
    print("Zero-mode vanishing orders at the pole: {}".format(report["vanishing_orders"]))
    print("Derivatives needed per allowed entry: {}".format(report["brane_suppression_orders"]))
    print("Two-brane spectra: {}".format([[round(x, 4) for x in s] for s in report["two_brane_spectra"]]))
    print("Bulk Proca (g=2, M=0) lowest level m^2 r^2 = {}; Higgs tuning ~ {:.1e}".format(
        report["proca_level_g2_M0"], report["higgs_mass_tuning"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
