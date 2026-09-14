#!/usr/bin/env python3
"""Print fixed supplied-cubic Bose diagnostics; no scientific input parameters."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from supplied_cubic_bose import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Supplied cubic Bose demonstrator")
    print("Graph: {} sites, {} edges".format(report["graph"]["sites"], report["graph"]["edges"]))
    print("Sector controls: {}".format(len(report["sectors"])))
    for case in report["sectors"]:
        print("  N={} g={} dimension={} Hermiticity residual={} interaction trace={} commutator={}".format(
            case["N"], case["g"], case["dimension"], case["hermiticity_residual"],
            case["interaction_trace"], case["commutator_sd"]))
    print("Fourier orthogonality residual: {}".format(report["fourier"]["orthogonality_residual"]))
    print("Fourier eigenvector residual: {}".format(report["fourier"]["eigenvector_residual"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
