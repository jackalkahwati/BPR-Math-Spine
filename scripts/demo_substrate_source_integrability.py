#!/usr/bin/env python3
"""Frozen mixed external-source integrability controls, stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys
import warnings

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
warnings.simplefilter("error")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_source_integrability import demonstration_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON on stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Mixed external-source integrability")
    print("Model:", report["model_id"])
    print("Status:", report["status"])
    print("Reason:", report["reason"])
    print(report["scope"])
    print("Frozen controls:", json.dumps(report["controls"], allow_nan=False))
    for case in report["cases"]:
        print("L={L} N={N} g={g} {partition} {background}: {status}".format(**case))
        print("  Reason:", case["reason"])
        for name, check in case["operator_checks"].items():
            print("  Operator", name, check["status"], "error", check["error"], "reason", check["reason"])
        print("  Ground:", json.dumps(case["ground"], allow_nan=False))
        print("  Hessian:", case["hessian"]["status"], case["hessian"]["reason"])
        for name, check in case["ward_checks"].items():
            print("  Ward", name, check["status"], "error", check["error"], "reason", check["reason"])
        print("  Free oracle:", case["free_oracle"]["status"], case["free_oracle"]["reason"])
        print("  Missing-contact control:", case["missing_contact_control"]["status"],
              case["missing_contact_control"]["reason"])
        for stencil in case["finite_differences"]["stencils"]:
            print("  FD", stencil["name"], stencil["status"], "target", stencil["target"],
                  "reason", stencil["reason"])
            for step in stencil["steps"]:
                print("    step", step["step"], step["status"], "value", step["value"],
                      "proxy", step["combined_proxy"], "drift", step["drift"], "reason", step["reason"])
    print("Summary:", json.dumps(report["summary"], allow_nan=False))
    print("Arithmetic proxies are entry-local diagnostics, not certified errors.")
    print("Endpoint screens do not establish branch isolation or truncation bounds.")
    print("An integrable external-source family is not a dynamical gravitational theory.")


if __name__ == "__main__":
    main()
