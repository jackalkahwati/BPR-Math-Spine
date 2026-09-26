#!/usr/bin/env python3
"""Print the one-loop scales and kill checks of the minimal BPR-6D model (Phase 1b)."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from model_scales import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print("One-loop coefficients: {}".format(report["beta"]))
    print("MSSM anchor: M_GUT = {:.2e} GeV".format(report["mssm_anchor"]["M_GUT"]))
    for label in ("SM below M_I", "2HDM below M_I"):
        r = report[label]
        print("{}: M_I = {:.2e}, M_GUT = {:.2e} GeV, 1/alpha_G = {:.1f}".format(
            label, r["M_I"], r["M_GUT"], r["alpha_G_inverse"]))
        print("  proton lifetime ~ {:.1e} yr (Super-K bound 2.4e34 passed: {})".format(
            r["proton_lifetime_yr"], r["super_k_ok"]))
        w = r["window_rM3"]
        print("  control window (rM >= 3, M_GUT <= 1/r): g_F in [{:.3f}, {:.3f}], 1/r in [{:.1e}, {:.1e}] GeV".format(
            w["g_F_min"], w["g_F_max"], *w["inverse_radius_range"]))
        print("  seesaw Dirac Yukawa needed at M_R = M_I: {:.1e}".format(r["seesaw_yD_needed_at_M_I"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
