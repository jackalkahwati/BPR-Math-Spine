#!/usr/bin/env python3
"""Print BPR-6D's conditional predictions: the Green-Schwarz QCD axion and the family number."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from predictions import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("Stueckelberg charges: {}".format(report["stueckelberg_charges"]))
    print("theta-bar (gauge invariant, a absent): {}".format(report["theta_bar"]))
    print("Gauge-invariant phases: {}".format(report["gauge_invariant_phases"]))
    print("Minimal Higgs: PQWW decay constant {:.1f} GeV (excluded)".format(report["pqww_decay_constant_gev"]))
    print("With a heavy F-charged singlet (f_S = 1e12 GeV, c_S = 6): f = {:.2e} GeV".format(report["dfsz_like_example_gev"]))
    print("Quality: needed action {:.0f}, wrapped-string estimate {}".format(
        report["quality"]["required_action"], [round(x) for x in report["quality"]["wrapped_string_estimate"]]))
    band = report["axion_band"]
    print("Upper band (f_b/3, kappa ~ 1): f {:.1e}-{:.1e} GeV, m_a {:.1e}-{:.1e} eV, {:.0f} Hz-{:.1e} Hz".format(
        band["f_gev"][0], band["f_gev"][1], band["ma_ev"][0], band["ma_ev"][1],
        band["frequency_hz"][0], band["frequency_hz"][1]))
    print("If dark matter: theta_i {:.1e}-{:.1e}; isocurvature {}; superradiance disfavours f in {}".format(
        band["theta_i_for_dm"][0], band["theta_i_for_dm"][1], band["isocurvature_if_dm"],
        band["superradiance_excluded_f"]))
    print("Family number: {}".format(report["family_number"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
