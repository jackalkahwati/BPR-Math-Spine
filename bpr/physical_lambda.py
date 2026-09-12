"""Historical substrate-to-Wilson coupling ansatz, not a derived physical map.

Preserves two numerical routes from frozen inputs:
  A: beta_A = ln(p), assuming a plaquette energy J and T_eff = J/ln(p).
  B: beta_B = 1/(4*pi*alpha), assuming EM normalization also defines the
     discrete-group Wilson coupling. This is not a universal finite-group
     normalization or a derivation of the emergent gauge coupling.
Their geometric mean and spread are historical ansatz summaries, not an
inferred physical coupling or uncertainty interval. Their numerical agreement
largely reflects ln(p) ~= 4*pi for the selected p, not independent evidence.

The retained pseudo-critical beta values are from the character-Wilson MC
model in gauge_phase_mc, not the original noncentral M2 Hamiltonian and not
the NEW central heat-kernel Hamiltonian. No beta/lambda map, monotonicity,
or physical substrate matching has been established. Calibration is not
merely an unknown factor that must exceed the reported margin to matter.

Inserting the ansatz in the historical Wilson scan places it above those
peaks; that arithmetic is retained. It does not locate the substrate phase,
prove topological anyons are physical excitations, or exclude a confining
physical sector. The substrate phase remains UNKNOWN. ln(p)/z remains an
unsupported per-link observation. No benchmark targets are used or opened.
"""
from __future__ import annotations

import numpy as np

from .nonabelian_gauge_sector import ALLOWED_CLASSES

EULER_GAMMA = 0.5772156649015329

# Frozen substrate integers (Postulate 0c selection; bpr/qcd_flavor.py).
P_SUBSTRATE = 104761
Z_COORDINATION = 6

# Historical character-Wilson pseudo-critical couplings (L=4->6); not lambda.
BETA_C_MEASURED = {5: 1.95, 8: 2.10, 9: 2.10, 12: 2.20}
BETA_C_RESOLUTION = 0.15     # scan spacing + L=4->6 drift


def alpha_inverse(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> float:
    """The frozen Postulate 0c formula: 1/alpha = [ln p]^2 + z/2 + gamma - 1/(2 pi)."""
    return float(np.log(p) ** 2 + z / 2.0 + EULER_GAMMA - 1.0 / (2.0 * np.pi))


def beta_thermal(p: int = P_SUBSTRATE) -> float:
    """Route A ansatz: ln(p); assumes plaquette energy J and T_eff = J/ln(p)."""
    return float(np.log(p))


def beta_gauge(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> float:
    """Route B ansatz: 1/(4*pi*alpha); discrete-gauge EM matching is assumed."""
    return alpha_inverse(p, z) / (4.0 * np.pi)


def beta_per_link_observation(p: int = P_SUBSTRATE,
                              z: int = Z_COORDINATION) -> float:
    """The recorded-not-used third value: ln(p)/z. No frozen provenance."""
    return float(np.log(p) / z)


def convergence(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> dict:
    """Historical ansatz agreement, not independent physical matching.

    Legacy beta_phys keys retain arithmetic only; spread is not uncertainty.
    """
    a, b = beta_thermal(p), beta_gauge(p, z)
    return {
        "beta_A_thermal": a,
        "beta_B_gauge": b,
        "ratio_B_over_A": b / a,
        "agree_within_10pct": bool(abs(b / a - 1.0) < 0.10),
        "why_they_agree": "ln p ~= 4 pi for the selected p (11.56 vs 12.57); "
                          "a property of the frozen p, not an identity",
        "beta_phys_central": float(np.sqrt(a * b)),
        "beta_phys_spread": float(abs(a - b) / 2.0),
        "interpretation": "historical coupling ansatz; not physical matching",
        "beta_lambda_mapping": "NOT ESTABLISHED",
    }


def phase_verdict(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> dict:
    """Conditional comparison to historical Wilson peaks, not substrate phase."""
    c = convergence(p, z)
    beta_phys = c["beta_phys_central"]
    ratios = {n: beta_phys / bc for n, bc in BETA_C_MEASURED.items()}
    min_ratio = min(ratios.values())
    return {
        "beta_phys": beta_phys,
        "beta_c_measured": dict(BETA_C_MEASURED),
        "ratio_to_transition": {n: round(r, 2) for n, r in ratios.items()},
        "verdict": "CONDITIONAL Wilson ansatz above historical transition peaks",
        "physical_phase": "UNKNOWN",
        "model_id": "finite-group-character-wilson-v1",
        "beta_lambda_mapping": "NOT ESTABLISHED",
        "margin": f"~{min_ratio:.0f}x above beta_c for every allowed class (ansatz only)",
        "kill_condition": ("requires a derived physical matching and transfer "
                           "calibration; no established monotone beta/lambda "
                           "map or factor-of-margin robustness claim"),
        "status": "PROPOSED v0.1 ansatz — physical interpretation UNESTABLISHED",
        "consequence_if_stands": (
            "only with independent matching to a topological phase would M1 "
            "anyons apply; the confining physical sector is not excluded"),
        "per_link_observation": {
            "beta_C": beta_per_link_observation(p, z),
            "vs_beta_c": {n: round(beta_per_link_observation(p, z) / bc, 2)
                          for n, bc in BETA_C_MEASURED.items()},
            "provenance": "NONE frozen — recorded as a numerical observation "
                          "only (near historical Wilson peaks, not a physical phase)",
        },
    }


def report(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> str:  # pragma: no cover
    c = convergence(p, z)
    v = phase_verdict(p, z)
    lines = [
        "Historical Wilson coupling ansatz (PROPOSED v0.1; not physical matching)",
        "=" * 68,
        f"Route A (thermal, T_eff = J/ln p frozen in rpst): beta = ln p "
        f"= {c['beta_A_thermal']:.3f}",
        f"Route B (gauge, beta = 1/(4 pi alpha), frozen alpha formula):   "
        f"beta = {c['beta_B_gauge']:.3f}",
        f"agreement: {c['ratio_B_over_A']:.3f} "
        f"(caveat: ln p ~= 4 pi for this p — stated, not hidden)",
        "",
        f"ansatz beta ~= {c['beta_phys_central']:.1f}, route spread "
        f"{c['beta_phys_spread']:.1f} (not uncertainty)",
        f"beta_c (historical character-Wilson peaks): {v['beta_c_measured']}",
        f"VERDICT: {v['verdict']} — {v['margin']}",
        f"Physical substrate phase: {v['physical_phase']}",
        f"beta/lambda mapping: {v['beta_lambda_mapping']}",
        f"kill condition: {v['kill_condition']}",
        f"consequence: {v['consequence_if_stands']}",
        "",
        f"recorded observation (no frozen provenance): ln(p)/z = "
        f"{v['per_link_observation']['beta_C']:.2f} ~= beta_c "
        f"(ratios {v['per_link_observation']['vs_beta_c']}) — "
        "near historical Wilson peaks under an unsupported normalization; "
        "not a determination of the substrate phase",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
