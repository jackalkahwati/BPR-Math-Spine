#!/usr/bin/env python3
"""Print the exact minimal local-anomaly completion of the supplied 6D parent."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# The completion module imports only sympy and the standard library.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from chiral_parent_completion import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Six-dimensional chiral parent: minimal local-anomaly completion")
    print("Parent alone: {} ({})".format(report["parent_only"]["I8"], report["parent_only"]["status"]))
    for label, key in (("Minimal completion (class C, 16 added components)", "minimal_completion"),
                       ("Narrower class C0 minimum (36 added components)", "restricted_class_minimal_completion")):
        case = report[key]
        print("{}: I8 = {}".format(label, case["I8_factored_sympy"]))
        print("  status: {}; pushforward (m=3): {}".format(case["status"], case["pushforward"]))
        for mode in case["zero_modes"]:
            print("  4D zero modes: {} x ({}, F-charge={})".format(
                mode["multiplicity"], mode["representation"], mode["charge"]))
        ledger = case["four_d_ledger"]
        print("  4D ledger: Spin10^3={} Spin10^2F (I6 coeff)={} F^3={} grav F={} doublets={} Z16 odd-center count={}".format(
            ledger["spin10_cubic"], ledger["spin10_squared_u1_I6_coefficient"]["numerator"],
            ledger["u1_cubic"], ledger["gravity_u1"], ledger["su2_doublets"],
            ledger["z16_odd_center_count_mod16"]))
    print("Class C budget scan (|Q|<=4): components -> factorizable solutions")
    print("  " + ", ".join("{}:{}".format(row["extra_components"], row["solutions"])
                          for row in report["class_C_budget_scan_q_max_4"] if row["solutions"]))
    print("Class C0 searches at 36 added components:")
    for search in report["class_C0_searches_budget_36"]:
        print("  |Q|<={}: {} candidates, {} factorizable".format(
            search["q_max"], search["examined"], len(search["solutions"])))
        for solution in search["solutions"]:
            print("    negative singlets by |Q|: {} abelian-anomaly-free massless sector: {}".format(
                solution["negative_singlet_charge_counts"], solution["four_d_abelian_anomaly_free"]))
    print("Standard Model Z16 controls (15 Weyl per generation + n_nu):")
    for key, value in report["standard_model_z16_controls"].items():
        print("  {}: total mod 16 = {}".format(key, value["total_mod16"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
