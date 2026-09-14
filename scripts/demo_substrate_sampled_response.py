#!/usr/bin/env python3
"""Frozen sampled-response reconstruction; text or strict JSON on stdout only."""
import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpr.substrate_sampled_response import demonstration_report


def _records(records):
    for record in records:
        schedule = record["schedule"]
        print(f"  theta={schedule['theta']} intervals={schedule['intervals']} "
              f"eta={record['eta']} epsilon={record['error_level']}: {record['status']}")
        if record["reason"] is not None:
            print(f"    {record['reason']}: {record['detail']}")
        print("    Diagnostic envelopes:",
              ", ".join(name + "=" + item["status"]
                        for name, item in record["diagnostics"].items()))
        ratio = record["ratio"]
        print(f"    Ratio: {ratio['status']}; value={ratio['value']}; radius={ratio['radius']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="strict JSON to stdout")
    args = parser.parse_args()
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print("Sampled connected joint response")
    print("Module:", report["module"])
    print("Prescribed slots, not successful-computation counts:", report["counts"])
    for case in report["physical_cases"]:
        print(f"L=N={case['L']} C={case['C']} g={case['g']}: {case['status']}")
        if case["reason"] is not None:
            print(f"  {case['reason']}: {case['detail']}")
        for partition in case["partitions"]:
            print(f"Partition {partition['partition']}: {partition['status']}")
            _records(partition["records"])
    synthetic = report["synthetic"]
    print("Reused synthetic reference:", synthetic["reference"])
    print("Synthetic status:", synthetic["status"])
    print(synthetic["perturbation_definition"])
    _records(synthetic["records"])
    for limitation in report["limitations"]:
        print(limitation)
    print("Empirical status:", report["scope"]["empirical_status"])


if __name__ == "__main__":
    main()
