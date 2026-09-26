#!/usr/bin/env python3
"""Print the brane and bulk-vector Yukawa analysis for the BPR-6D flux families."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from yukawa_mechanisms import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Brane J_z charge c = s_h + 3: {}".format(report["brane_jz_charge"]))
    print("Single-brane spectrum pattern by c: {}".format(report["single_brane_patterns_by_c"]))
    print("c=0 single-brane spectra (degenerate pair): {}".format(
        [[round(x, 4) for x in s] for s in report["single_brane_spectra_c0"]]))
    print("Zero-mode vanishing orders at the pole: {}".format(report["vanishing_orders"]))
    print("Derivatives needed per allowed entry, by c: {}".format(report["brane_suppression_orders_by_c"]))
    print("Two c=2 branes at separation 0.1 (rank 2): {}".format(
        [[float("{:.3g}".format(x)) for x in s] for s in report["two_c2_branes_gamma_0.1"]]))
    print("Three clustered c=2 branes, [m2/m1/g^2, m3/m1/g^4]: {}".format(report["three_c2_branes_scaling"]))
    print("J_z-broken texture, eps=0.1, median log10 mass ratios: {}".format(
        [round(x, 2) for x in report["fn_texture_eps_0.1_median_log10_ratios"]]))
    print("Bulk Proca (g=2, M=0) lowest level m^2 r^2 = {}; Higgs tuning ~ {:.1e}".format(
        report["proca_level_g2_M0"], report["higgs_mass_tuning"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
