#!/usr/bin/env python3
"""Print the brane self-tuning test for the BPR-6D flux vacuum."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from cosmological_constant import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Self-tuning test at the flat point: {}".format(report["self_tuning_test"]))
    print("Flat tensions (one per flux quantum): {}".format(report["discrete_flat_tensions"]))
    print("Tuning magnitudes: {}".format(report["tuning_magnitudes"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
