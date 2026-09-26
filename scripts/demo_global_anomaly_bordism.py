#!/usr/bin/env python3
"""Print the Adams E2 computation of Omega_7^Spin(B(Spin(10) x U(1))) and its validation."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from global_anomaly_bordism import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("A(1): {}".format(report["a1"]))
    val = report["validation"]
    print("Validation: ko(point) stems 0-8 {}".format(
        {k: len(v) for k, v in val["ko_point_stems_0_to_8"].items()}))
    print("  Omega_5(BSU(2)) E2 {} (Witten anomaly), Omega_7(BSU(2)) E2 {}, Omega_7(BSU(3)) E2 {}".format(
        val["bsu2_reduced_stem5"], val["bsu2_reduced_stem7"], val["bsu3_reduced_stem7"]))
    for name, summary in report["summands"].items():
        print("{}: stem 7 E2 {}; E2 classes by stem (s<=12) {}".format(
            name, summary["E2_stem_7"], summary["E2_total_by_stem"]))
    print("Verdict (Spin x Spin(10) x U(1)): {}".format(report["verdict"]))
    tw = report["twisted_structure"]
    print("Twisted Spin x_Z2 Spin(10) x U(1): Omega_7 E2 {}, Omega_5(Spin(5)) order <= {}, 16 = (4,4): {}".format(
        tw["groups"]["omega7_spin10_u1_E2"], tw["groups"]["omega5_spin5_order_bound"],
        tw["groups"]["branching"]["is_4_times_4"]))
    print("  anomaly on generator {}: {} -> global anomaly {}".format(
        tw["argument"]["generator"], tw["argument"]["anomaly_on_generator"], tw["global_anomaly"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
