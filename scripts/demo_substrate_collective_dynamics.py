#!/usr/bin/env python3
"""Frozen classical modes and quantum finite memory; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_collective_dynamics import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Classical canonical modes and finite quantum projected memory")
    print("Model:", report["model_id"])
    for case in report["classical_cases"]:
        p = case["parameters"]
        print(f"Classical L={p['L']} g={p['g']}: {case['real_oscillator_count']} real oscillators; "
              f"Hessian residual={case['independent_oracle']['hessian_mode_max_residual']:.3g}")
    for case in report["quantum_cases"]:
        p, source = case["parameters"], case["source"]
        print(f"Quantum L={p['L']} g={p['g']} dimension={p['dimension']}: "
              f"s={source['s']:.9g}, a={source['a']:.9g}, beta_squared={source['beta_squared']:.9g}")
        for response in case["responses"]:
            print(f"  z={response['z']}: full resolvent residual={response['full_resolvent_residual']:.3g}; "
                  f"retarded identity residual={response['retarded_identity_residual']:.3g}")
    print("Toy controls: analytic two-transition, exact one-transition memory zero, zero source")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
