#!/usr/bin/env python3
"""Print the Phase 2 confrontation of the minimal BPR-6D model with fermion data (stored best fits)."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from fermion_fit import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    run = report["gut_scale_inputs"]
    print("GUT-scale inputs (one-loop SM at 2e16 GeV): up {} down {} lepton {}; CKM (s12, s23, s13, delta) {}".format(
        [float("{:.4g}".format(x)) for x in run["up"]], [float("{:.4g}".format(x)) for x in run["down"]],
        [float("{:.4g}".format(x)) for x in run["lepton"]], [float("{:.4g}".format(x)) for x in run["ckm"]]))
    print("Identical branes: {}".format(report["identical_brane_rule"]))
    print("Generic parametrization rank modulo U(3) and arg r: {} (full model: 18)".format(
        report["generic"]["parametrization_rank"]))
    for label in ("generic", "pinned", "pinned_charged_only"):
        r = report[label]
        print("{}: chi^2 = {:.2f}".format(label, r["chi2"]))
        print("  pulls: {}".format(r["pulls"]))
        if "neutrino_sector" in r:
            nu = r["neutrino_sector"]
            print("  neutrinos: masses {} eV, sum {:.4f} eV, m_bb {:.2e} eV, sin(delta_CP) {:.2f}; M_R {} GeV".format(
                [float("{:.3g}".format(x)) for x in nu["light_masses_ev"]], nu["sum_ev"], nu["m_betabeta_ev"],
                nu["sin_delta_cp"], [float("{:.2g}".format(x)) for x in nu["M_R_gev"]]))
    print("Prediction ranges over the generic minima within Delta chi^2 = 4:")
    for key, value in report["prediction_ranges"].items():
        print("  {}: {}".format(key, [float("{:.3g}".format(x)) for x in value]))
    print("Seesaw: {}".format(report["seesaw"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
