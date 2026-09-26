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
    print("Physical axion QCD component vs f_a/f_b: {}".format(report["physical_axion_qcd_component_vs_fa_over_fb"]))
    band = report["axion_band"]
    print("f_a {:.1e}-{:.1e} GeV, m_a {:.1e}-{:.1e} eV, DM needs theta_i {:.1e}-{:.1e}".format(
        band["fa_gev"][0], band["fa_gev"][1], band["ma_ev"][0], band["ma_ev"][1],
        band["theta_i_for_dm"][0], band["theta_i_for_dm"][1]))
    print("Family number: {}".format(report["family_number"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
