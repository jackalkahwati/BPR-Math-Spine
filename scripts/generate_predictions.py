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
    parser.add_argument("--p", type=int, default=104761,
                        help="Substrate prime modulus (default: 104761)")
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
            "ccr_affected": _ccr_affects(key, theory),
            "ccr_role": _ccr_role(key, theory),
        })

    # Append CCR-derived top-level predictions (Postulate 0)
    rows.extend(_ccr_predictions(sdt))

    # ── Print table ──
    print("═" * 100)
    print(f"{'ID':<40} {'Theory':<22} {'CCR':<5} {'Value'}")
    print("─" * 100)
    for r in rows:
        ccr_flag = "Y" if r.get("ccr_affected") else "·"
        print(f"{r['prediction_id']:<40} {r['theory']:<22} {ccr_flag:<5} {r['value']}")
    print("═" * 100)
    print(f"\nTotal predictions: {len(rows)}")
    n_ccr = sum(1 for r in rows if r.get("ccr_affected"))
    print(f"CCR-affected: {n_ccr} / {len(rows)} "
          f"({100.0 * n_ccr / max(len(rows), 1):.1f}%)")

    # ── Write CSV ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "prediction_id", "theory", "value", "numeric",
                    "ccr_affected", "ccr_role",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {args.output}")


# ─────────────────────────────────────────────────────────────────
# CCR (Postulate 0) classification of predictions
# ─────────────────────────────────────────────────────────────────

# Theories whose mode sums or shell structures are restricted by the
# C_n selection rule when CCR is enforced.
_CCR_AFFECTED_PREFIXES = {
    "P5",   # neutrino mass spectrum / PMNS overlaps
    "P7",   # GW quadrupole modes
    "P10",  # Kuramoto cluster modes
}
_CCR_AFFECTED_THEORY_TOKENS = (
    "Lepton", "Nuclear", "Quantum Chemistry",
    "Cosmology", "QCD", "Topological",
)


def _ccr_affects(key: str, theory: str) -> bool:
    """Return True if CCR (C_n + scaling) modifies this prediction."""
    head = key.split("_")[0]            # e.g., "P5.10" or "P10"
    base = head.split(".")[0]            # strip subindex → "P5", "P10"
    if base in _CCR_AFFECTED_PREFIXES:
        return True
    return any(tok in theory for tok in _CCR_AFFECTED_THEORY_TOKENS)


def _ccr_role(key: str, theory: str) -> str:
    """Short tag describing how CCR changes the prediction."""
    base = key.split("_")[0].split(".")[0]
    if not _ccr_affects(key, theory):
        return "ccr-neutral"
    if "Neutrino" in theory or base == "P5":
        return "C_n selection on PMNS overlaps"
    if "Lepton" in theory:
        return "C_n restricts cohomology mode sum"
    if "Nuclear" in theory:
        return "winding shells projected to C_n orbits"
    if "GW" in theory or base == "P7":
        return "quadrupole restricted to m=0,±n"
    if "Collective" in theory or base == "P10":
        return "cluster count locked to n"
    if "Cosmology" in theory:
        return "K=2 finite recursion in inflation"
    if "Topological" in theory:
        return "winding sectors restricted to C_n"
    return "C_n filter applied"


def _ccr_predictions(sdt) -> list:
    """Top-level Postulate-0 predictions derived directly from CCR."""
    delta = sdt.ccr_universal_delta()
    return [
        {
            "prediction_id": "CCR_universal_delta",
            "theory": "0   CCR / Postulate 0",
            "value": f"{delta:.3f}",
            "numeric": True,
            "ccr_affected": True,
            "ccr_role": "δ = 2 Δ_φ universal Casimir exponent",
        },
        {
            "prediction_id": "CCR_n_rotation",
            "theory": "0   CCR / Postulate 0",
            "value": str(sdt.n_rotation),
            "numeric": True,
            "ccr_affected": True,
            "ccr_role": "C_n rotation order (canonical 6)",
        },
        {
            "prediction_id": "CCR_recursion_depth",
            "theory": "0   CCR / Postulate 0",
            "value": str(sdt.ccr_depth),
            "numeric": True,
            "ccr_affected": True,
            "ccr_role": "K finite recursion depth (canonical 2)",
        },
        {
            "prediction_id": "CCR_sigma",
            "theory": "0   CCR / Postulate 0",
            "value": f"{sdt.sigma:.3f}",
            "numeric": True,
            "ccr_affected": True,
            "ccr_role": "σ scaling factor between layers",
        },
        {
            "prediction_id": "CCR_Delta_phi",
            "theory": "0   CCR / Postulate 0",
            "value": f"{sdt.delta_phi:.3f}",
            "numeric": True,
            "ccr_affected": True,
            "ccr_role": "scaling weight Δ_φ = δ/2",
        },
        {
            "prediction_id": "CCR_outer_overlap_bound",
            "theory": "0   CCR / Postulate 0",
            "value": "σ < 2  ⇒ outer rings overlap (Star of David)",
            "numeric": False,
            "ccr_affected": True,
            "ccr_role": "geometric falsifier on σ",
        },
    ]


if __name__ == "__main__":
    main()
