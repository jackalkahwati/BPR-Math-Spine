#!/usr/bin/env python3
"""Print the flux and orbifold analysis of Spin(10) -> Standard Model breaking in BPR-6D."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from gut_breaking import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("16 content: {}".format(report["sm_dimensions"]))
    print("T3R flux example (chirality neutrality): {}".format(report["flux_example_T3R"]))
    print("Flux plane theorem: {}".format(report["flux_plane_theorem"]))
    print("Flux scan |h_i| <= 2: {}".format(report["flux_scan"]))
    print("Orbifold twist classes: {}".format(report["orbifold_classes"]))
    print("Orbifold j=1 family counts: {}".format(report["orbifold_j1"]["counts"]))
    print("Uniform orbifold choices for j <= 10: {}".format(report["orbifold_uniform_j_up_to_10"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
