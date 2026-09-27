#!/usr/bin/env python3
"""Print how the chi vortex condensate pins the four brane positions of BPR-6D-M (Phase 1d), metastably."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from brane_stabilization import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("chi lowest level: {}".format(report["lowest_level"]))
    print("Quartic minimum {:.6f} (spread of local minima {:.1e}); alternatives: {}".format(
        report["quartic_minimum"], report["spread_of_local_minima"], report["alternatives"]))
    print("Zeros form a regular tetrahedron (max deviation of pairwise dot products from -1/3): {:.1e}".format(
        report["tetrahedron_deviation"]))
    print("Hessian on CP^4 (3 rotations + 5 massive): {}".format([round(x, 4) for x in report["quartic_hessian"]]))
    print("Pinning stiffness at a zero: {}".format([round(x, 4) for x in report["pinning_stiffness"]]))
    print("Residual family symmetry: {} (rotation phases {})".format(
        report["residual_family_group"], report["rotation_phases"]))
    print("Pinning barrier (edge midpoint) {:.4f}; stacking pinning energy {:.1e} (metastable)".format(
        report["pinning_barrier"], report["stacking_pinning_energy"]))
    print("Yukawas with branes fixed at the tetrahedron: {}; Phase 1 example reachability cost {:.1e} "
          "(unreachable: fixed positions constrain the Yukawas)".format(
              report["tetrahedral_yukawa"], report["phase1_example_reachability_cost"]))
    print("Type II: {}".format(report["type_II"]))
    print("Scale window for v r: {}".format(report["scale_window"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
