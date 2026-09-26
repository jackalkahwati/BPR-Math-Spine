#!/usr/bin/env python3
"""Print the BPR-6D flux vacuum M4 x S^2: conditions, radion stability and 4D scales."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from six_dim_flux_vacuum import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    cond, vac, rad, rel = (report[key] for key in ("einstein_conditions", "vacuum", "radion", "four_d_relations"))
    print("Einstein equations on M4 x S^2 (must vanish): 4D block {}; sphere block {}".format(
        cond["four_d"], cond["sphere"]))
    print("Vacuum: r = {}, Lambda = {}, B = {}".format(vac["radius"], vac["Lambda"], vac["B"]))
    print("Radion: V'' = {}, K = {}, m^2 r0^2 = {}".format(
        rad["d2V"], rad["kinetic_K"], rad["radion_mass_squared_times_r0_squared"]))
    print("4D: M_Pl^2 = {}, 1/r over M6 = {}, control needs g4 < {}".format(
        rel["M_Pl_squared"], rel["inverse_radius_over_M"], rel["control_bound_on_g4"]))
    print("Green-Schwarz background: {}".format(report["green_schwarz_background"]))
    print("Illustrative scales for flux 3 (assumed g4):")
    for row in report["illustrative_scales_flux_3"]:
        print("  g4={g4}: 1/r={inverse_radius_GeV:.3g} GeV, M6={M6_GeV:.3g} GeV, "
              "(1/r)/M6={inverse_radius_over_M6:.3f}".format(**row))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
