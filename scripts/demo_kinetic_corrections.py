#!/usr/bin/env python3
"""Print the Phase 2b check: can the leading corrections rescue the pinned-brane fermion fit?"""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpr.kinetic_corrections import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("Zero modes orthonormal (max |G - 1|): {:.1e}; tight frame sum u u^dag = (4/3) 1: {}".format(
        report["zero_mode_gram_deviation"], report["tight_frame"]))
    print("A4-averaged kinetic matrix is a multiple of 1 (Schur): {}".format(report["schur"]))
    nat = report["natural_sizes"]
    print("Natural sizes: BKT eps {:.3f}-{:.3f} (rM {:.1f}-{:.1f}); general K and displacement ~ {} (deficit spread)".format(
        nat["bkt_eps_range"][0], nat["bkt_eps_range"][1], nat["rM_range"][0], nat["rM_range"][1],
        nat["tension_estimate"]))
    for kind, settings in report["scans"].items():
        print("{} (bound: {})".format(kind, report["bound_meaning"][kind]))
        for setting, rows in settings.items():
            print("  {:8s} ".format(setting) + "  ".join("{:g}:{:.1f}".format(b, c) for b, c in rows))
    print("At natural size: {}".format(report["chi2_at_natural_size"]))
    print("Smallest scanned bound giving chi^2 <= number of observables: {}".format(report["bound_needed"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
