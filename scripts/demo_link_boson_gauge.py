#!/usr/bin/env python3
"""Print link-boson emergent-gauge diagnostics; no fitted scientific inputs."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from link_boson_gauge import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Link-boson emergent U(1) gauge structure (proposed amendment)")
    print("Clusters: name, links, sector dim, ice dim, 4-cycles, bipartite")
    for c in report["clusters"]:
        print("  {} links={} sector={} ice={} squares={} bipartite={}".format(
            c["name"], c["links"], c["sector_dimension"], c["ice_dimension"],
            c["four_cycles"], c["bipartite"]))
    print("Exact diagonalization vs effective theory (U=1):")
    for comp in report["low_energy_comparison"]:
        print("  {}: 2nd-order error ratio {:.2f} (t^3 -> 8), 3rd-order error ratio {:.2f} (t^4 -> 16)".format(
            comp["name"], comp["second_order_error_ratio"], comp["third_order_error_ratio"]))
    print("Gauss-law commutator norms on the open cube at t=0.1: min {:.3f}".format(
        min(report["gauss_commutator_norms_open_cube_t_0_1"])))
    print("RK point: zero-energy states vs flip-connected sectors")
    for rk in report["rk_point"]:
        print("  {}: {} zero modes, {} sectors".format(rk["name"], rk["zero_energy_states"], rk["components"]))
    cubic = report["cubic_lattice_effective_couplings"]
    print("Cubic lattice: K = {} t^2/U + {} t^3/U^2 (plaquette), K6 = {} t^3/U^2 (hexagon)".format(
        cubic["plaquette_K_second_order_over_t2_U"], cubic["plaquette_delta_K_third_order_over_t3_U2"],
        cubic["hexagon_K6_third_order_over_t3_U2"]))
    diamond = report["diamond_lattice_effective_couplings"]
    print("Diamond lattice: no plaquettes; K6 = {} t^3/U^2 (pyrochlore hard-core bosons, V=2U)".format(
        diamond["hexagon_K6_third_order_over_t3_U2"]))
    print("Coulomb (photon) phase established here: {}".format(report["coulomb_phase_established"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
