#!/usr/bin/env python3
"""Frozen finite-ring currents, Ward checks and external flux; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_current_response import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional number-current response and external-source flux")
    print("Model:", report["model_id"])
    for label, cases in (("Interacting cases", report["cases"]),
                         ("Analytic free controls", report["free_controls"])):
        print(label)
        for case in cases:
            p, c = case["parameters"], case["curvature"]
            print(f"L={p['L']} N={p['N']} g={p['g']}: "
                  f"curvature={c['value']} ({c['status']}), "
                  f"dia={c['diamagnetic']:.9g}, param={c['paramagnetic']:.9g}")
            for mode in case["momenta"]:
                ward = max(max(r["ward_first_residual"], r["ward_second_residual"],
                               r["ward_combined_residual"]) for r in mode["responses"])
                print(f"  m={mode['m']}: Ward residual={ward:.3g}, "
                      f"f-sum residual={mode['fsum']['residual']:.3g}")
            for difference in c["finite_differences"]:
                print(f"  Phi step={difference['step']}: "
                      f"FD={difference['value']} ({difference['status']}), "
                      f"raw diagnostic={difference['raw_value']:.9g}")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
