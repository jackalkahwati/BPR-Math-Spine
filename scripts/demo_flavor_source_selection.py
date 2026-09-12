#!/usr/bin/env python3
"""Stdout-only conditional source selection; no fitting, optimization or MC.

Run from any directory with the script path, optionally adding --json.
All inputs are stipulated mean-field choices, not physical calibrations.
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

import numpy as np

from bpr.flavor_source_selection import demonstration


def json_value(value):
    """Encode complex arrays explicitly and leave undefined predictions null."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def print_case(name, case):
    print("\nCase:", name)
    print("Status:", case["status"])
    print("Source coefficients:\n", case["source_coefficients"])
    print("Profile coefficients:\n", case["profile_coefficients"])
    print("Action:", case["action"])
    print("Analytic lower bound:", case["energy_bound"])
    print("Bound residual:", case["bound_residual"])
    print("Residual resolution:", case["bound_residual_resolution"])
    print("Residual interpretation:", case["bound_residual_status"])
    print("Scalar stationarity norm:", case["stationarity_norm"])
    observables = case["observables"]
    for sector in ("up", "down"):
        print(sector, "Y:\n", np.array2string(observables[sector]["matrix"],
                                             precision=8, suppress_small=True))
        print(sector, "singular values:", observables[sector]["singular_values"])
    print("Representative |V|:", observables["abs_mixing"])
    print("Representative CP quartet J:", observables["J"])
    if observables["mixing_undefined_reason"]:
        print("Undefined:", observables["mixing_undefined_reason"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    result = demonstration()
    if args.json:
        print(json.dumps(json_value(result), indent=2, allow_nan=False))
        return
    print("Conditional source-selection theorem, not a BPR-derived vacuum or flavor fit")
    print("Model:", result["model_id"])
    print("Inputs:", json.dumps(json_value(result["inputs"]), allow_nan=False))
    print_case("equal filling (control, not a global minimum)", result["equal_filled"])
    print_case("selected aligned coherent occupations", result["aligned"])
    print("Selection:", result["aligned"]["selection_status"])
    print("Axis profile coefficients (A, B, C):\n", result["aligned"]["axis_profile_ABC"])
    print("\nUncoupled sectors:", result["uncoupled"]["selection_status"])
    print("Predicted mixing:", result["uncoupled"]["mixing"])
    for name, case in result["uncoupled"]["representatives"].items():
        print_case("unselected representative: " + name, case)
    average = result["rotational_average"]
    print("\nROTATIONAL AVERAGE")
    print(average["explanation"])
    print("Averaged orbit energy:", average["averaged_orbit_energy"])
    print("Energy evaluated at averaged density:", average["energy_of_average_density"])
    print("\nASSUMPTION LEDGER")
    print(json.dumps(json_value(result["assumption_ledger"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
