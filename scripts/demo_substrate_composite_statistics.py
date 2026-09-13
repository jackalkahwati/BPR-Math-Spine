#!/usr/bin/env python3
"""Print the frozen composite-algebra controls; never write output files."""
import os

# Set thread defaults before any numerical imports; preserve operator overrides.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_composite_statistics import demonstration_report


class _StdoutParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stdout)
        self.exit(2, self.prog + ": error: " + message + "\n")

    def exit(self, status=0, message=None):
        if message:
            print(message, end="", file=sys.stdout)
        raise SystemExit(status)


def _text(report):
    print("Composite grading, canonical algebra and original-tensor support")
    print("Exact int64 binary controls; complete-sector float64 Bose ladders")
    for case in report["binary_cases"] + report["counterfactual_cases"]:
        print("binary L={L} sites={sites} q={modulus} ({neutrality_rule}) "
              "charge={number_charge} grading={grading} CAR={canonical_car} "
              "support={tensor_support} neutral={neutral_compression_status}".format(**case))
    for case in report["pair_cases"]:
        print("pair L={L} S={S} T={T} disjoint={disjoint} sign={exchange_sign} "
              "exchange_exact={annihilator_exchange_exact} "
              "mixed_bracket_zero={mixed_graded_bracket_zero} "
              "overlap_formula_exact={overlap_formula_exact}".format(**case))
    for case in report["bose_cases"]:
        witness = case["double_creation_vacuum"]
        print("Bose L={L} N={N} sites={sites} lowering={lowering_status} "
              "shapes={lowering_shape}/{creation_shape} "
              "raw_product_residuals={annihilation_product_frobenius_residual!r},"
              "{creation_product_frobenius_residual!r}".format(**case)
              + " double_creation_vacuum_residual=" + repr(witness["frobenius_residual"])
              + " norm_squared=" + repr(witness["norm_squared"])
              + " expected=" + str(witness["expected_norm_squared"]))
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
        # Validate strict JSON in both modes, before emitting any partial report.
        encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    except ValueError as exc:
        if args.json:
            print(json.dumps({"status": "unavailable", "error": str(exc)}, allow_nan=False))
        else:
            print("Composite algebra report unavailable: " + str(exc))
        return 1
    if args.json:
        print(encoded)
    else:
        _text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
