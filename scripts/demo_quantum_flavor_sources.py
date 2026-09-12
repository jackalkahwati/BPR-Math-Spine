#!/usr/bin/env python3
"""Frozen quantum source diagnostic; stdout only, no fitting or simulations.

This instantaneous internal-fermion model is not a microscopic BPR derivation.
Run using the script path from any directory; add --json for structured output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.quantum_flavor_sources import demonstration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Quantum source-operator diagnostic, not a substrate-derived vacuum")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    print("Frozen parameters:", json.dumps(report["parameters"], allow_nan=False))
    print("\nNUMBER-SECTOR COMPARISON (conservation is not population selection)")
    for ordering, result in report["orderings"].items():
        print("\nOrdering:", ordering)
        print("Counterterm coefficient:", result["counterterm_coefficient"])
        print("Nu Nd  dimension  lowest energy       ground dimension  status")
        for sector in result["sectors"]:
            nu, nd = sector["numbers"]
            print(f"{nu:2} {nd:2}  {sector['dimension']:9}  {sector['energy']: .10f}"
                  f"  {sector['ground_dimension']:16}  {sector['numerical_status']}")
        print("Lowest sectors of this stipulated Hamiltonian:", result["lowest_sectors"])
    print("\nONE PARTICLE PER SECTOR: complete subspace, not a preferred eigenvector")
    for label in ("full_square", "normal_ordered", "eta_zero_control"):
        result = report["one_plus_one"][label]
        print(f"{label}: ground dimension={result['ground_dimension']}, "
              f"status={result['numerical_status']}")
        for channel in result["channels"]:
            print(f"  J={channel['J']}: energy={channel['energy']:.10f}, "
                  f"gap={channel['gap_from_ground']:.10g}")
    print("Chosen ground representatives (not unique vacua):")
    for label, state in report["one_plus_one"]["representatives"].items():
        print(f"  {label}: purity={state['purity']:.6g}")
        for species in ("u", "d"):
            print(f"    rho_{species}={json.dumps(state['rho_' + species])}")
    print("\nFULL FILLING")
    filled = report["filled_sector"]
    print("Full-square energy:", filled["full_square_energy_oracle"])
    print("Constant density per species:", filled["constant_density_per_species"])
    print("Interpretation:", filled["interpretation"])
    print("\nQUANTUM VERSUS CLASSICAL DENSITY VARIANCES (product states only)")
    for key in ("classical_energy", "quantum_energy",
                "weighted_variance_correction", "identity_residual"):
        print(f"{key}: {report['product_variance'][key]:.12g}")
    print("\nMISSING MICROSCOPIC MATCHING INPUTS")
    for assumption in report["assumptions"]:
        print(f"- {assumption['id']}: {assumption['missing_input']}")
    print("\nCONCLUSIONS")
    for conclusion in report["conclusions"]:
        print("-", conclusion)
    print("Physical masses:", report["physical_masses"])
    print("Physical mixing:", report["physical_mixing"])


if __name__ == "__main__":
    main()
