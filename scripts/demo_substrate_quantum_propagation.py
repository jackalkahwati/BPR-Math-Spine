#!/usr/bin/env python3
"""Frozen complete-sector quantum density propagation diagnostics; stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_quantum_propagation import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Finite-sector quantum density propagation")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    print("Reason:", report["reason"])
    print("Controls: (L,N)=(3,3),(4,4),(5,5),(8,2); C=1; g=0,0.7,40")
    print("Ct=0,0.001,0.01,0.1; reference site0; each nonzero ring distance")
    print("Complete controls: 12 models, 48 case/time records, 108 observations")
    for case in report["cases"]:
        print(f"L={case['L']} N={case['N']} g={case['g']} "
              f"dimension={case['dimension']}: {case['status']}")
        print("  Reason:", case["reason"])
        print("  Hamiltonian norm bound BN:", case["hamiltonian_norm_bound"])
        print("  Eigensystem:", case["eigensystem"]["status"])
        for time_record in case["times"]:
            print(f"  Ct={time_record['tau']} z={time_record['z']}: "
                  f"{time_record['status']}")
            for observation in time_record["distances"]:
                tail = observation["bound_evaluation"]
                comparison = observation["comparison"]
                print(f"    distance={observation['distance']}: {observation['status']}; "
                      f"commutator norm={observation['commutator_norm']}; "
                      f"envelope={observation['analytic_upper_bound']}")
                print(f"      tail={tail['status']}; terms={tail['terms']}; "
                      f"partial_sum={tail['partial_sum']}; remainder={tail['remainder_bound']}")
                print(f"      excess={observation['signed_excess']}; "
                      f"literal_inequality={observation['literal_inequality']}; "
                      f"comparison={comparison['status']}")
                print("      Reason:", observation["reason"])
                if tail["status"] == "numerical_unavailable":
                    print("      Tail reason:", tail["reason"])
    for key, counts in report["summary"].items():
        print(key + ":", counts)
    print("Exact stipulated-operator envelope; its floating evaluation is heuristic.")
    print("Regression consistency is not a literal-inequality or numerical-error certificate.")
    print("Cap2 is the stipulated envelope cap, not physical commutator saturation.")
    print("Finite fixed-N sectors are not site tensor products. No thermodynamic velocity,")
    print("physical clock or length, Lorentz invariance, or relativistic spacetime is established.")
    print("Empirical status:", report["scope"]["empirical_status"])


if __name__ == "__main__":
    main()
