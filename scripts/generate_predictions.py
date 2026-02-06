#!/usr/bin/env python3
"""
Generate the full table of 40+ falsifiable predictions
=======================================================

Runs the entire BPR-Math-Spine from first principles and outputs
a CSV / printed table of every prediction from the Ten Adjacent
Theories, with derived numerical values.

Usage:
    python scripts/generate_predictions.py [--output data/predictions.csv]
"""

import argparse
import csv
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpr.first_principles import SubstrateDerivedTheories


def main():
    parser = argparse.ArgumentParser(description="Generate BPR predictions table")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: print to stdout)")
    parser.add_argument("--p", type=int, default=104729,
                        help="Substrate prime modulus (default: 104729)")
    parser.add_argument("--N", type=int, default=10000,
                        help="Lattice nodes (default: 10000)")
    parser.add_argument("--J", type=float, default=1.0,
                        help="Coupling strength in eV (default: 1.0)")
    parser.add_argument("--radius", type=float, default=0.01,
                        help="Boundary radius in m (default: 0.01)")
    parser.add_argument("--geometry", default="sphere",
                        choices=["ring", "square", "sphere"],
                        help="Boundary geometry (default: sphere)")
    args = parser.parse_args()

    # ── Build theories from first principles ──
    sdt = SubstrateDerivedTheories.from_substrate(
        p=args.p, N=args.N, J_eV=args.J,
        radius=args.radius, geometry=args.geometry,
    )

    # ── Print summary ──
    print(sdt.summary())
    print()

    # ── Generate predictions ──
    preds = sdt.predictions()

    # Categorise by theory
    theory_map = {
        "P1": "I   Memory",
        "P2": "II  Impedance",
        "P3": "III Decoherence",
        "P4": "IV  Phase Transitions",
        "P5": "V   Neutrino",
        "P6": "VI  Info Geometry",
        "P7": "VII GW Phenomenology",
        "P8": "VIII Complexity",
        "P9": "IX  Bioelectric",
        "P10": "X   Collective",
    }

    rows = []
    for key in sorted(preds.keys()):
        value = preds[key]
        prefix = key.split("_")[0]
        theory = "?"
        for code, name in theory_map.items():
            if prefix.startswith(code):
                theory = name
                break
        rows.append({
            "prediction_id": key,
            "theory": theory,
            "value": str(value),
            "numeric": isinstance(value, (int, float)),
        })

    # ── Print table ──
    print("═" * 90)
    print(f"{'ID':<40} {'Theory':<22} {'Value'}")
    print("─" * 90)
    for r in rows:
        print(f"{r['prediction_id']:<40} {r['theory']:<22} {r['value']}")
    print("═" * 90)
    print(f"\nTotal predictions: {len(rows)}")

    # ── Write CSV ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prediction_id", "theory", "value", "numeric"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {args.output}")


if __name__ == "__main__":
    main()
