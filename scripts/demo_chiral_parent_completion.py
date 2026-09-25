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
    minimal = report["minimal_completion"]
    print("Minimal completion I8 = {}".format(minimal["I8_factored_sympy"]))
    print("  status: {}; pushforward (m=3): {}".format(minimal["status"], minimal["pushforward"]))
    for mode in minimal["zero_modes"]:
        print("  4D zero modes: {} x ({}, X={})".format(
            mode["multiplicity"], mode["representation"], mode["charge"]))
    ledger = minimal["four_d_ledger"]
    print("  4D ledger: Spin10^3={} Spin10^2 X={} X^3={} grav X={} doublets={} Z16(Spin10-charged)={}".format(
        ledger["spin10_cubic"], ledger["spin10_squared_u1"]["numerator"], ledger["u1_cubic"],
        ledger["gravity_u1"], ledger["su2_doublets"], ledger["z16_spin10_charged_count_mod16"]))
    print("Exhaustive minimal-class searches (36 added Weyl components):")
    for search in report["searches"]:
        print("  |Q|<={}: {} candidates, {} factorizable".format(
            search["q_max"], search["examined"], len(search["solutions"])))
        for solution in search["solutions"]:
            print("    negative singlets by |Q|: {} abelian-anomaly-free in 4D: {}".format(
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
