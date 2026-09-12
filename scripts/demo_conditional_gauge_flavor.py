#!/usr/bin/env python3
"""Stdout-only exact gauge and conditional flavor demonstration; no fitting/MC.

From repo root: python3 scripts/demo_conditional_gauge_flavor.py [--json]
From elsewhere, use the absolute script path; no output files are generated.
All displayed inputs are stipulated toy choices, not physical calibrations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bpr.gauge_heat_kernel import (
    isolated_square,
    isolated_square_diagnostics,
    model_spec,
    wilson_comparison,
)


def json_value(value):
    """Explicit complex encoding and exact rational strings; never emit NaN."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def build_report():
    from bpr.chiral_flavor_prototype import demonstration

    n, lam, dt = 5, 1.3, 0.08
    return {
        "status": "Conditional toy calculations, not empirical validation or a TOE",
        "gauge": {
            "specification": model_spec(n, lam),
            "square": isolated_square(n, lam),
            "refinement": isolated_square_diagnostics(n, lam, dt),
            "wilson_comparison": wilson_comparison(n, beta=1.8, lam=lam, dt=1.0),
        },
        "flavor": demonstration(),
        "boundaries": [
            "No production Monte Carlo or benchmark comparison was performed.",
            "Historical Wilson runs are not runs of the new central model.",
            "Flux and scalar sources are inputs; source selection remains open.",
            "No physical mass matching or higher-dimensional consistency is claimed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    result = build_report()
    if args.json:
        print(json.dumps(json_value(result), indent=2, allow_nan=False))
        return
    print(result["status"])
    gauge = result["gauge"]
    print("\nGAUGE:", gauge["specification"]["model_id"])
    print("Inputs: D5, lambda=1.3; refinement dt=0.08, 0.04, 0.02")
    print("Isolated-square Hamiltonian:\n", gauge["square"]["hamiltonian"])
    print("Toy energies:", gauge["square"]["energies"])
    print("Effective-H errors:", gauge["refinement"]["effective_hamiltonian_errors"])
    print("Convergence orders:", gauge["refinement"]["observed_orders"])
    comparison = gauge["wilson_comparison"]
    print("Single-link comparison: beta=1.8, lambda=1.3, dt=1; no calibration")
    print("Irreps:", comparison["irrep_names"])
    print("Central energies:", comparison["heat_energies"])
    print("Wilson energies:", comparison["wilson"]["energies"])
    flavor = result["flavor"]
    print("\nFLAVOR:", flavor["model_id"])
    print("Inputs: R=kappa=mu_squared=1, h0=2, q=3, effective coupling=1")
    print("Forced sources: Ju=z+(3z^2-1)/4; Jd=x+z/3+xy/2")
    for name in ("unforced", "cyclic", "forced"):
        case = flavor[name]
        print("\nCase:", name)
        for sector in ("up", "down"):
            print(sector, "Y:\n", np.array2string(case[sector]["matrix"], precision=7,
                                                 suppress_small=True))
            print(sector, "singular values:", case[sector]["singular_values"])
        print("|V|:", case["abs_mixing"])
        print("Signed J:", case["J"])
        if case["mixing_undefined_reason"]:
            print("Mixing undefined:", case["mixing_undefined_reason"])
        print("Stationarity norms:", case["stationarity_norm_up"],
              case["stationarity_norm_down"])
    print("\nASSUMPTION LEDGER")
    print(json.dumps(json_value(flavor["assumption_ledger"]), indent=2, allow_nan=False))
    print("\nLIMITATIONS")
    for item in result["boundaries"]:
        print("-", item)


if __name__ == "__main__":
    main()
