#!/usr/bin/env python3
"""Frozen ring-to-triplet diagnostic; stdout only, no physical flavor prediction.

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

from bpr.substrate_triplet_projection import demonstration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    result = demonstration()
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print("Frozen ring projection: three modes do not establish monopole flavor")
    print("Model:", result["model_id"])
    print("Frozen inputs:", json.dumps(result["parameters"], allow_nan=False))
    print("\nFREE SPECTRAL WINDOW")
    print(json.dumps(result["free_window"], indent=2, allow_nan=False))
    print("\nINTERACTION/GAP DIAGNOSTIC (not an error theorem)")
    print(json.dumps(result["interaction_gap_diagnostic"], indent=2, allow_nan=False))
    print("\nDIRECT LOCAL SOURCE MATCHING")
    source = result["source_matching"]
    print("Ring linear source dimension:", source["direct_source_real_rank"])
    print("Monopole linear source dimension:", source["target_source_real_rank"])
    print("Generated complex matrix-algebra dimension:",
          source["generated_algebra_complex_rank"])
    print("Full direct source matching:", source["direct_matching"])
    print("The rank deficit concerns single local scalar sources, not their products.")
    print("\nPROJECTED ENERGY AND NONLINEAR LEAKAGE")
    for example in result["examples"]:
        print(f"{example['name']}: N={example['norm']:.8g}, energy={example['energy']:.10g}")
        leakage = example["leakage"]
        print(f"  leakage norm={leakage['raw_norm']:.10g}, "
              f"status={leakage['status']}, resolved={leakage['resolved_norm']}")
    print("\nCONSERVATIVE FINITE-TIME CLASSICAL ERROR BOUNDS")
    print(json.dumps(result["finite_time_bounds"], indent=2, allow_nan=False))
    print("\nLIMITATIONS")
    for limitation in result["limitations"]:
        print("-", limitation)
    print("Physical masses:", result["physical_masses"])
    print("Physical mixing:", result["physical_mixing"])


if __name__ == "__main__":
    main()
