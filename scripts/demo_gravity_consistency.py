#!/usr/bin/env python3
"""Frozen gravity consistency controls; stdout only, no empirical target fitting.

Run from any directory. --json emits a complete strict-JSON report.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.gravity_consistency import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Action-consistent gravity normalization and conditional identifiability")
    print("Model:", report["model_id"])
    print("Exact action formulas: reduced M, M^2 R/2 + alpha R^2/2")
    for case in report["exact_action_formulas"]:
        print("  M={M:g} alpha={alpha:g}: mass^2={mass_squared:.9g}, plateau={plateau:.9g}".format(**case))
        print("    V(phi/M=0,.1,1,5):", ", ".join("{:.9g}".format(row["V"]) for row in case["potential"]))
    print("Leading large-Ne slow-roll algebraic controls, not exact cosmologies")
    for case in report["leading_slow_roll"]:
        print("  Ne={Ne:g} alpha={alpha:g}: As={As:.9g}, epsilon={epsilon:.9g}, n_s={n_s:.9g}, r={r:.9g}".format(**case))
    print("Synthetic inverse calibration, not prediction")
    for case in report["inverse_calibrations"]:
        print("  Ne={Ne:g} As={synthetic_As:g}: alpha={calibrated_alpha:.9g}, round-trip residual={round_trip_residual:.3g}".format(**case))
    print("SI energy conversion and Einstein-frame entropy controls")
    for case in report["si_energy_and_entropy_controls"]:
        print("  reduced E={reduced_energy_j:g} J, unreduced E={unreduced_energy_j:.9g} J, G={G_si:.9g} SI".format(**case))
        print("    energy inverse residual={G_energy_round_trip_residual:.3g}, entropy relative residual={area_lP_entropy_relative_residual:.3g}".format(**case))
    print("Conditional induced term only, assumes b=c=0")
    for case in report["conditional_induced_units"]:
        print("  p={p:g} cutoff={supplied_cutoff_energy_j:g} J: a/lP={spacing_planck_ratio:.9g}, residual={spacing_ratio_residual:.3g}".format(**case))
    print("Constant ZG degeneracy: log gradient (1,1), null (1,-1)")
    for case in report["constant_ZG_degeneracy"]:
        print("  factor={factor:g}: Z={new_Z:g}, G={new_G:g}, amplitude residual={amplitude_residual:.3g}".format(**case))
    induced = report["induced_identifiability"]
    print("Induced effective coefficient:", induced["M_eff_squared"])
    print("  Jacobian (b,c,cutoff):", induced["jacobian"])
    print("  Numerical null residuals:", induced["numerical_null_residuals"])
    for case in induced["bare_counterterm_shifts"]:
        print("  b/c shift={shift:g}: coefficient residual={coefficient_residual:.3g}".format(**case))
    for case in induced["cutoff_shifts"]:
        print("  new cutoff={cutoff:g}: coefficient residual={coefficient_residual:.3g}".format(**case))
    print("Actual flat-H identity-shift spectra and dynamics, numerical residuals only")
    for case in report["flat_identity_shifts"]:
        print("  offset={offset:g}, pairwise gap residual={pairwise_gap_residual:.3g}, source commutator norm={source_commutator_norm:.3g}".format(**case))
        for row in case["dynamics"]:
            print("    t={time:g}: conjugation residual={conjugation_residual:.3g}, global-phase operator residual={state_phase_operator_residual:.3g}".format(**row))
    print("Limitations")
    for limitation in report["limitations"]:
        print(limitation)


if __name__ == "__main__":
    main()
