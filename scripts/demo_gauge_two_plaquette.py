#!/usr/bin/env python3
"""Frozen bounded two-plaquette controls; stdout only, optional strict JSON."""
import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpr.gauge_two_plaquette import demonstration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit strict JSON instead of text")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2))
        return
    print(report["scope"])
    print("Frozen coupling lambda=1.3; transfer steps 0.08, 0.04, 0.02")
    for control in report["controls"]:
        print("D{}: coordinates {}, physical {} (Burnside {})".format(
            control["n"], control["coordinate_dimension"],
            control["physical_dimension"], control["burnside_count"]))
        print("  Ground energy {:.12g}; electric minimum {:.5g}".format(
            control["hamiltonian_energies"][0], control["electric_min_eigenvalue"]))
        for step in control["transfer_diagnostics"]:
            print("  dt={:.2f}: H_eff error {:.8g} <= analytic bound {:.8g}; min(T) {:.8g}".format(
                step["dt"], step["effective_hamiltonian_error"],
                step["analytic_generator_error_bound"], step["transfer_min_eigenvalue"]))
        print("  Error ratios: " + ", ".join("{:.8g}".format(ratio) for ratio in control["error_ratios"]))
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
