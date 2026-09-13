#!/usr/bin/env python3
"""Print frozen long-wave and uniform-condensate controls; never write files."""
import os

# Set thread defaults before numerical imports; preserve operator overrides.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_longwave_limit import demonstration_report


class _StdoutParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stdout)
        self.exit(2, self.prog + ": error: " + message + "\n")

    def exit(self, status=0, message=None):
        if message:
            print(message, end="", file=sys.stdout)
        raise SystemExit(status)


def _text(report):
    print("Classical long-wave controls and bounded uniform-condensate bridge")
    print("Analytic envelopes are separate from raw floating residuals; no exactness certificate")
    for case in report["acoustic_cases"] + report["dispersive_cases"]:
        print("{scaling} L={L} m={m} g={g} {time_scaling} {source_scaling} "
              "generator_spectral2_error={generator_error_operator_norm!r} "
              "bound={generator_error_bound!r}".format(**case))
        for propagation in case["propagator_cases"]:
            print("  T={T} propagator_spectral2_error={error_operator_norm!r} "
                  "bound={error_bound!r} {bound_status}".format(**propagation))
        for response in case["response_cases"]:
            print("  zeta={zeta} density_absolute_error={absolute_error!r} "
                  "bound={error_bound!r} resolvent_spectral2_error="
                  "{resolvent_error_operator_norm!r} contraction_residual="
                  "{response_contraction_residual!r} {bound_kind}".format(**response))
    for case in report["quantum_cases"]:
        print("quantum L=N={L} g={g} t={t} dimension={dimension} "
              "phase_corrected_euclidean_error={phase_corrected_error!r} "
              "bound={error_bound!r} {bound_status} expected_variance={expected_variance!r} "
              "variance_residual={variance_residual!r} free_structure={free_exact_structure} "
              "eigenpair_frobenius={eigenpair_frobenius_residual!r} "
              "orthogonality_frobenius={orthogonality_frobenius_residual!r}".format(**case))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    print("Numerical domain: " + json.dumps(report["numerical_domain"], allow_nan=False, sort_keys=True))


def main(argv=None):
    parser = _StdoutParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print strict JSON instead of text")
    args = parser.parse_args(argv)
    try:
        report = demonstration_report()
        # Check both modes before emitting a partial numerical report.
        encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    except ValueError as exc:
        if args.json:
            print(json.dumps({"status": "unavailable", "error": str(exc)}, allow_nan=False))
        else:
            print("Long-wave report unavailable: " + str(exc))
        return 1
    if args.json:
        print(encoded)
    else:
        _text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
