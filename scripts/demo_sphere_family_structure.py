#!/usr/bin/env python3
"""Print the flux-family SU(2) structure and the Yukawa selection rules on S^2 (BPR-6D)."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from sphere_family_structure import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    zm = report["zero_modes_k3"]
    print("Flux 3: {} zero modes, SU(2) spin {}, J_z {}".format(zm["count"], zm["spin"], zm["jz_weights"]))
    print("Zero-mode counts by flux: {}".format(report["zero_mode_counts"]))
    print("KK levels (m^2 r^2): {}".format([(row["l"], round(row["numeric"], 9)) for row in report["kk_spectrum_k3"]]))
    print("6D chirality lemma (same chirality): {}".format(report["chirality_lemma"]))
    for row in report["yukawa_channels_k3"]:
        print("  Higgs spin weight {higgs_spin_weight}: total weight {total_spin_weight}, "
              "allowed isospins {allowed_isospins}".format(**row))
    ex = report["orientation_examples"]
    print("Vev orientations: m=2 {}, m=0 {}; real-vev sum-rule violation {:.1e}".format(
        ex["ferromagnetic_m2"], ex["uniaxial_m0"], ex["real_vev_sum_rule_max_violation"]))
    print("Any spectrum reachable: target {} -> {}".format(ex["hierarchical_target"], ex["hierarchical_reached"]))
    status = report["minimal_yukawa_status"]
    print("Minimal Yukawa status: {} (SO(12) gauge-Higgs: {} families, Higgs level {})".format(
        status["status"], status["so12_gauge_higgs"]["families"], status["so12_gauge_higgs"]["higgs_level"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
