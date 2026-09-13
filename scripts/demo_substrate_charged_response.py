#!/usr/bin/env python3
"""Complete finite-ring charged response, frozen controls; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_charged_response import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional charged response, not physical particle masses")
    print("Model:", report["model_id"])
    for case in report["cases"]:
        p, t = case["parameters"], case["thresholds"]
        print(f"L={p['L']} N={p['N']} g={p['g']}: "
              f"mu_plus={t['mu_plus']:.9g}, mu_minus={t['mu_minus']:.9g}, "
              f"Delta_c={t['Delta_c']:.9g}, removal cost={t['removal_ground_cost']:.9g}")
        for mode in case["modes"]:
            print(f"  m={mode['m']}: W-={mode['removal']['total_weight']:.9g}, "
                  f"W+={mode['addition']['total_weight']:.9g}, "
                  f"commutator residual={mode['commutator_residual']:.3g}")
    print("Actual H+kappa*N controls")
    for control in report["number_offset_controls"]:
        print(f"L={control['L']} g={control['g']} kappa={control['kappa']}: "
              f"Green translation residual={control['green_translation_max_residual']:.3g}")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
