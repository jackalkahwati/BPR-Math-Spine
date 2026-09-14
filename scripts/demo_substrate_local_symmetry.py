#!/usr/bin/env python3
"""Conditional local-symmetry theorem and frozen finite diagnostics; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_local_symmetry import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional bounded local-symmetry theorem")
    print("On the unrestricted uniform Bose ring with C>0, a bounded operator")
    print("on a contiguous interval with at least two exterior sites is scalar")
    print("if it obeys weak conservation. Support checks alone do not test conservation.")
    print("Module:", report["module"])
    print("Complete-sector finite-compression diagnostics, not infinite-Fock proofs")
    for slot in report["cases"]:
        print(f"L={slot['L']} g={slot['g']}: {slot['status']}")
        if slot["report"] is None:
            print("  Reason:", slot["reason"])
            continue
        case = slot["report"]
        print(f"  C={case['C']} cutoff={case['cutoff']} dimension={case['dimension']}")
        for operator in case["operators"]:
            print(f"  {operator['name']}: finite Frobenius norm="
                  f"{operator['finite_frobenius_norm']:.9g}; "
                  f"{operator['conservation_diagnostic']}")
            witness = operator["witness"]
            if witness is not None:
                print(f"    Witness={witness['value']} expected={witness['expected']} "
                      f"allowance={witness['allowance']:.9g}; {witness['status']}")
    print("Interval hypothesis controls (peeling uses the original ring)")
    for interval in report["interval_controls"]:
        print(f"L={interval['L']} start={interval['start']} length={interval['length']}: "
              f"{interval['status']}")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
