#!/usr/bin/env python3
"""Print Bogoliubov-level common-light-cone diagnostics for two species."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from common_light_cone import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Two-species phonon speeds: c^2 = eig(2 D^1/2 G D^1/2)")
    for case in report["cases"]:
        print("  {}: miscible={} common_cone={} speeds={} (numeric {})".format(
            case["name"], case["miscible"], case["common_cone"],
            case.get("speeds_formula"), case.get("speeds_long_wave_numeric")))
    z2 = report["z2_splitting_example"]
    print("Z2 point, mu_ab={}: speeds {}".format(z2["mu_ab"], z2["speeds"]))
    print("Collider bound on isotropic photon-electron speed difference: ~{}".format(
        report["collider_speed_difference_bound_order"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
