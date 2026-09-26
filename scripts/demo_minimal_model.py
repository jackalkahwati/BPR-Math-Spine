#!/usr/bin/env python3
"""Print the minimal BPR-6D model (Phase 1a): field content, coupling rules, breaking pattern, brane Yukawas, and the
failure of brane Higgs copies versus a single bulk Higgs."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from minimal_model import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Fields:")
    for name, f in report["fields"].items():
        print("  {:8s} {:5s} {:7s} F={:>3} weight={} x{}  {}".format(
            name, f["location"], f["so10"], f["F"], f["normal_weight"], f["copies"], f["role"]))
    print("Wanted couplings allowed: {}".format({k: v["allowed"] for k, v in report["wanted_couplings"].items()}))
    print("Dangerous couplings allowed: {}".format({k: v["allowed"] for k, v in report["forbidden_couplings"].items()}))
    print("Unbroken dimensions: {}".format({k: v["unbroken_dimension"] for k, v in report["breaking"].items()}))
    print("Unbroken Cartan directions with 126bar: {}".format(report["breaking"]["with 126bar (SM)"]["cartan_unbroken"]))
    print("c = 2 brane matrices span {} of 6 dimensions (J = 2 only)".format(report["c2_brane_span_dimension"]))
    print("Realizability rank modulo U(3) by number of branes (target 24): {}".format(report["realizability_rank_mod_U3"]))
    print("Three-brane Bargmann relation mismatch: {:.1e}".format(report["bargmann_relation_mismatch"]))
    print("Four-brane realization of a hierarchical pair: {}".format(report["hierarchical_example"]))
    print("Version A (brane copies): tree-level Yukawa rank with one light doublet = {}".format(
        report["version_A_brane_copies_light_yukawa_rank"]))
    print("Version B (bulk Higgs): levels m^2 r^2 = {}; lowest-level checks: {}".format(
        report["bulk_scalar_levels"], report["lowest_level_checks"]))
    print("Version B light combination over random branes: {}".format(report["bulk_higgs_scan"]))
    print("Degenerate light states with positive brane value terms only: {}".format(
        report["positive_value_terms_degenerate_light_states"]))
    print("Family isometry left unbroken: {}".format(report["isometry_stabilizer"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
