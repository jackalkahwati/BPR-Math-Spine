#!/usr/bin/env python3
"""Bounded parity-aware quantum projection controls; stdout only.

Run from any directory. --json emits the complete strict-JSON report.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_quantum_matching import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Conditional determinant projection, not a gauge or generation derivation")
    print("Model:", report["model_id"])
    for case in report["cases"]:
        p, candidate, sources, spectrum, symmetry = (
            case[key] for key in ("parameters", "candidate", "sources", "spectral_separation", "symmetries"))
        print(f"L={p['L']} N={p['N']}: candidate={candidate['dimension']}, "
              f"full hard-core={candidate['hard_core_dimension']}, "
              f"H closure={candidate['hamiltonian_closure_norm']:.3g}, "
              f"many-body separation={spectrum['minimum_separation']} ({spectrum['status']})")
        print(f"  real source span={sources['direct_real_span_dimension']}, "
              f"complex associative algebra={sources['unital_complex_associative_algebra_dimension']}, "
              f"H commutant={symmetry['hamiltonian_commutant_complex_dimension']}, "
              f"reflection leakage={symmetry['many_body_reflection_leakage']:.3g}")
    for control in report["intertwiners"]:
        lifts = control["lifts"]
        print(f"L={control['L']} odd vector maximum ranks: "
              f"proper half-turn={lifts['proper_half_turn']['maximum_rank']}, "
              f"alternate lift={lifts['alternate_parity_twisted']['maximum_rank']}; "
              f"bilinear sign identity residual={control['bilinear_conjugation_identity_residual']:.3g}")
    for control in report["common_scale_controls"]:
        print(f"Common scale L={control['L']} N={control['N']} C={control['C']}: "
              f"H scaling residual={control['Hproj_scaling_residual']:.3g}")
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
