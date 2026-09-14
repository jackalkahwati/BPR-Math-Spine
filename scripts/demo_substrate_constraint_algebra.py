#!/usr/bin/env python3
"""Frozen diagonal-constraint and projection-multiplication controls; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_constraint_algebra import demonstration_report


class StdoutParser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        if message:
            sys.stdout.write(message)


def main():
    parser = StdoutParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Diagonal microscopic constraints and projection multiplication")
    print("Model:", report["model_id"])
    print("Complete fixed-number Bose graph (not the hard-core ambient space)")
    for case in report["constraint_cases"]:
        nonconstant = [row for row in case["onsite_projectors"] if row["status"] == "nonconstant"]
        print(f"L={case['L']} N={case['N']} C={case['C']} g={case['g']} "
              f"dimension={case['dimension']} components={case['graph']['component_count']} "
              f"nonconstant onsite projectors={len(nonconstant)}")
    print("Inherited hard-core three-mode windows (no interaction parameter g)")
    for case in report["projection_cases"]:
        defect = max(row["audit"]["defect_operator_norm"] for row in case["density_pairs"])
        residual = max(row["audit"]["corrected_gram_residual"] for row in case["density_pairs"])
        print(f"L={case['L']} N={case['N']} dimension={case['dimension']} "
              f"rank={case['rank']} {case['status']} "
              f"max defect operator2={defect:.6g} corrected Gram Frobenius residual={residual:.3g}")
        witness = case["predetermined_witness"]
        if witness is not None:
            print(f"  Sites0,1 compressed commutator operator2="
                  f"{witness['actual_commutator_operator_norm']:.9g} "
                  f"reviewed target={witness['expected_commutator_operator_norm']:.9g}")
    print("Vacuum controls:", report["vacuum_controls"]["complete_bose"]["structural_status"],
          report["vacuum_controls"]["hard_core"]["status"])
    print("Norms and residuals are binary64 diagnostics, not exactness certificates.")
    for statement in report["limitations"]:
        print(statement)


if __name__ == "__main__":
    main()
