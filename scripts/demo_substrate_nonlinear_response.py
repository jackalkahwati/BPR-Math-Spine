#!/usr/bin/env python3
"""Frozen DNLS response calculation; stdout only, not a physical flavor prediction.

Run from any directory. Add --json for the complete strict structured report.
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

from bpr.substrate_nonlinear_response import demonstration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Interaction-generated ring response, not a universal flavor source matrix")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    print("Frozen inputs:", json.dumps(report["parameters"], allow_nan=False))
    print("\nLOCAL DENSITY RESPONSE: retained and omitted terms both matter")
    print(json.dumps(report["witness"]["response"], indent=2, allow_nan=False))
    print("\nANALYTIC WITNESS (derivative in g at zero, not a finite-g prediction)")
    print(json.dumps(report["witness"]["analytic_coefficients"], indent=2, allow_nan=False))
    print("The order-t retained and omitted contributions cancel in the total.")
    print("A bound larger than the correction does not certify its finite-g sign.")
    print("\nCONTROLS")
    for name, control in report["controls"].items():
        print(f"{name}: derivative={control['full_derivative']:.8g}, "
              f"status={control['arithmetic']['full']['status']}, "
              f"bound_status={control['bounds']['status']}")
    print("\nLIMITATIONS")
    for limitation in report["limitations"]:
        print("-", limitation)
    print("Physical masses:", report["physical_masses"])
    print("Physical mixing:", report["physical_mixing"])


if __name__ == "__main__":
    main()
