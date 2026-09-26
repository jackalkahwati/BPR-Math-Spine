#!/usr/bin/env python3
"""Print the Morrison-Taylor check of BPR-6D and the 6D (1,0) SO(10) x U(1) analogues."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from string_embedding import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str, allow_nan=False))
        return 0
    print("BPR-6D vs Morrison-Taylor: {}".format(report["bpr6d_mt"]))
    print("With a vector-like charge-1 pair: {}".format(report["vectorlike_extension"]))
    print("SU(2) T=0 validation: {}".format([(r["b"], str(r["genus"]), str(r["fundamentals"])) for r in report["su2_validation"]]))
    print("Swapped chirality labels: {}".format(report["bpr6d_mt_swapped_chirality"]))
    print("SO(10) on T=0: {}; with adjoint hypers: {}".format(report["so10_t0"], report["so10_t0_with_adjoints"]))
    print("Pure U(1) on T=1 (charge 3, then 1): {}".format(report["t1_pure_u1"]))
    for name, ex in report["susy_examples"].items():
        print("SUSY {}: all anomaly equations {}, gcd {}, 16s per unit flux {}".format(
            name, ex["all_ok"], ex["charge_gcd"], ex["net_16_per_unit_flux"]))
    print("Reduction lemma: {}".format(report["reduction_lemma"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
