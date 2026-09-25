#!/usr/bin/env python3
"""Print fixed cubic-lattice condensate diagnostics; no fitted scientific inputs."""

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

# Avoid the eager scientific package initializer, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpr"))
from cubic_condensate_regime import demonstration_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit the strict JSON report")
    parser.add_argument("--full", action="store_true",
                        help="extend the mean-field ladder to N=5 (slower)")
    args = parser.parse_args(argv)
    report = demonstration_report((2, 3, 4, 5) if args.full else (2, 3, 4))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Cubic lattice condensate regime demonstrator")
    print("Sector vacuum (n=3): N, g, gap, min component, max symmetry residual")
    for case in report["sector_vacuum"]:
        print("  N={} g={} gap={:.6f} min={:.3e} sym={:.1e}".format(
            case["N"], case["g"], case["first_gap"], case["min_component"],
            max(case["symmetry_residuals"].values())))
    print("Mean-field ladder, mode (1,0,0): N, lam, gap error, residue vs Bogoliubov, f-sum residual")
    for case in report["mean_field_ladder"]:
        print("  N={} lam={} gap_err={:+.4f} Z/N={:.4f} (Bog {:.4f}) depletion={:.4f}<= {:.3f} fsum={:.1e}".format(
            case["N"], case["lam"], case["gap_error"], case["residue_per_particle"],
            case["bogoliubov_residue"], case["depletion"], case["depletion_bound"],
            case["fsum_residual"]))
    print("Acoustic windows: n, mu, c_s, relative deviation from c_s|q| (lowest mode; worst mode)")
    for case in report["acoustic_windows"]:
        print("  n={} mu={} c_s={:.4f} lowest_dev={:.4f} worst_dev={:.4f}".format(
            case["n"], case["mu"], case["c_s"], case["lowest_mode_relative_deviation"],
            case["max_relative_deviation"]))
    print("Moving condensates: n, m_p, mu, dynamical, energetic, signature, ergoregion")
    for case in report["moving_condensates"]:
        print("  n={} p={} mu={} {} {} {} {}".format(
            case["n"], case["m_p"], case["mu"], case["dynamical_stability"],
            case["energetic_stability"], case["signature"], case["ergoregion"]))
    print("Limitations")
    for limitation in report["limitations"]:
        print("  " + limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
