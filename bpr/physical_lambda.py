"""The M2 remainder: derive the substrate's physical Wilson coupling (PROPOSED).

After `gauge_phase_mc` located the confinement-deconfinement transition
(beta_c ~= 1.9-2.2 across the allowed classes), exactly one question remained
for M2: WHERE DOES THE SUBSTRATE SIT? That requires a map from frozen
substrate quantities (p, z) to the dimensionless plaquette coupling beta of
the frozen Wilson dynamics. This module derives that map two independent
ways, checks them against each other, and states the verdict with its kill
condition. Status: **PROPOSED (v0.1)** — a derivation from frozen inputs, but
the Euclidean<->Hamiltonian calibration caveat is open, so it is not merged
into the frozen core.

ROUTE A — thermal (uses a relation already frozen elsewhere in the repo)
------------------------------------------------------------------------
`bpr/rpst/boundary_energy.py` froze T_eff = J / ln(p): coarse-graining over p
discrete states equilibrates the boundary at effective temperature J/ln p.
A Euclidean lattice theory at temperature T has plaquette coupling
beta = (energy per plaquette unit)/(k_B T) = J/T_eff:

    beta_A = J / T_eff = ln p = 11.559          (p = 104,761)

No new quantity is introduced: J cancels, and ln p was already the frozen
temperature denominator.

ROUTE B — gauge normalization (uses the frozen alpha formula)
-------------------------------------------------------------
The standard Wilson-coupling convention is beta = 1/g^2 (discrete group,
2-dim faithful irrep). BPR's frozen EM normalization gives the substrate
gauge coupling g^2 = 4 pi alpha with 1/alpha = [ln p]^2 + z/2 + gamma
- 1/(2 pi) (the Postulate 0c formula, `bpr/consistency.py`):

    beta_B = 1/(4 pi alpha) = 10.905            (p = 104,761, z = 6)

CONVERGENCE — AND ITS HONEST CAVEAT
-----------------------------------
beta_B / beta_A = 0.943: the routes agree to 6%. RECORDED PLAINLY: this
agreement holds because ln p ~= 4 pi for the selected prime (11.56 vs 12.57,
8%); it is a property of the frozen p, not an independent identity. Both
routes are nevertheless derivations from frozen inputs with no tuning here.

THE VERDICT (and what would kill it)
------------------------------------
    beta_phys ~= 11.2 +- 0.4  (both routes)
    beta_c    ~= 1.9 - 2.2    (measured, gauge_phase_mc)
    beta_phys / beta_c ~= 5-6

The substrate sits on the **DECONFINED / TOPOLOGICAL side**, a factor ~5 above
the transition — robust to any O(1) convention ambiguity smaller than that
factor. KILL CONDITION: the Euclidean-beta <-> Hamiltonian-lambda map is
monotone but uncalibrated here; an anisotropic-lattice calibration (or exact
duality) shifting the effective coupling by more than ~5x would overturn the
verdict. Nothing smaller can.

CONSEQUENCES IF THE VERDICT STANDS (both branches recorded)
-----------------------------------------------------------
* The physical D_n sector is in its topological phase: the M1 anyon content
  (A2 pseudoscalar charge, E_k doublets, non-Abelian fusion) IS the physical
  excitation content. Consistent with the quantum-double kinematics of M1.
* The confining-phase glueball-analog spectrum (the M4 machinery's target)
  describes a phase the substrate does NOT occupy. Benchmark v3 then tests
  the confining phase of the frozen dynamics as a THEORY exercise, not the
  substrate's physical state — unless this module's verdict falls to its
  kill condition. Path B's particle-sector interpretation must route through
  anyonic (topological) excitations, not confined flux loops.

A THIRD OBSERVATION, WEAKEST PROVENANCE, RECORDED NOT USED
----------------------------------------------------------
The per-link share beta_C = ln(p)/z = 1.93 lands ON the measured beta_c
(0.88-0.99 x beta_c for all four classes). If a per-link normalization were
the right convention, the substrate would sit essentially AT criticality.
No frozen derivation in the repo supports dividing by z here, so this is
logged as a numerical observation only — not the answer. It is falsifiable
the same way (calibration), and its near-coincidence with beta_c across all
four classes is flagged for future work.

BLINDNESS: no glueball target appears here (grep-enforced in tests).
"""
from __future__ import annotations

import numpy as np

from .nonabelian_gauge_sector import ALLOWED_CLASSES

EULER_GAMMA = 0.5772156649015329

# Frozen substrate integers (Postulate 0c selection; bpr/qcd_flavor.py).
P_SUBSTRATE = 104761
Z_COORDINATION = 6

# Measured pseudo-critical couplings (bpr/gauge_phase_mc.py, L=4->6 stable).
BETA_C_MEASURED = {5: 1.95, 8: 2.10, 9: 2.10, 12: 2.20}
BETA_C_RESOLUTION = 0.15     # scan spacing + L=4->6 drift


def alpha_inverse(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> float:
    """The frozen Postulate 0c formula: 1/alpha = [ln p]^2 + z/2 + gamma - 1/(2 pi)."""
    return float(np.log(p) ** 2 + z / 2.0 + EULER_GAMMA - 1.0 / (2.0 * np.pi))


def beta_thermal(p: int = P_SUBSTRATE) -> float:
    """Route A: beta = J/T_eff with T_eff = J/ln p (frozen in rpst/boundary_energy)."""
    return float(np.log(p))


def beta_gauge(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> float:
    """Route B: beta = 1/g^2 with g^2 = 4 pi alpha (standard Wilson convention,
    alpha from the frozen Postulate 0c formula)."""
    return alpha_inverse(p, z) / (4.0 * np.pi)


def beta_per_link_observation(p: int = P_SUBSTRATE,
                              z: int = Z_COORDINATION) -> float:
    """The recorded-not-used third value: ln(p)/z. No frozen provenance."""
    return float(np.log(p) / z)


def convergence(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> dict:
    """The two derived routes and their agreement — with the ln p ~ 4 pi
    coincidence stated rather than hidden."""
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
    }


def phase_verdict(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> dict:
    """Which side of the measured transition the substrate sits on."""
    c = convergence(p, z)
    beta_phys = c["beta_phys_central"]
    ratios = {n: beta_phys / bc for n, bc in BETA_C_MEASURED.items()}
    min_ratio = min(ratios.values())
    return {
        "beta_phys": beta_phys,
        "beta_c_measured": dict(BETA_C_MEASURED),
        "ratio_to_transition": {n: round(r, 2) for n, r in ratios.items()},
        "verdict": "DECONFINED / TOPOLOGICAL",
        "margin": f"~{min_ratio:.0f}x above beta_c for every allowed class",
        "kill_condition": ("Euclidean<->Hamiltonian calibration shifting the "
                           f"effective coupling by more than ~{min_ratio:.0f}x "
                           "(anisotropic-lattice study or exact duality)"),
        "status": "PROPOSED v0.1 — not merged into the frozen core",
        "consequence_if_stands": (
            "physical excitations are the M1 anyons (topological phase); the "
            "confining glueball-analog spectrum belongs to a phase the "
            "substrate does not occupy"),
        "per_link_observation": {
            "beta_C": beta_per_link_observation(p, z),
            "vs_beta_c": {n: round(beta_per_link_observation(p, z) / bc, 2)
                          for n, bc in BETA_C_MEASURED.items()},
            "provenance": "NONE frozen — recorded as a numerical observation "
                          "only (would place the substrate AT criticality)",
        },
    }


def report(p: int = P_SUBSTRATE, z: int = Z_COORDINATION) -> str:  # pragma: no cover
    c = convergence(p, z)
    v = phase_verdict(p, z)
    lines = [
        "Physical Wilson coupling from substrate quantities (PROPOSED v0.1)",
        "=" * 68,
        f"Route A (thermal, T_eff = J/ln p frozen in rpst): beta = ln p "
        f"= {c['beta_A_thermal']:.3f}",
        f"Route B (gauge, beta = 1/(4 pi alpha), frozen alpha formula):   "
        f"beta = {c['beta_B_gauge']:.3f}",
        f"agreement: {c['ratio_B_over_A']:.3f} "
        f"(caveat: ln p ~= 4 pi for this p — stated, not hidden)",
        "",
        f"beta_phys ~= {c['beta_phys_central']:.1f} +- {c['beta_phys_spread']:.1f}",
        f"beta_c (measured): {v['beta_c_measured']}",
        f"VERDICT: {v['verdict']} — {v['margin']}",
        f"kill condition: {v['kill_condition']}",
        f"consequence: {v['consequence_if_stands']}",
        "",
        f"recorded observation (no frozen provenance): ln(p)/z = "
        f"{v['per_link_observation']['beta_C']:.2f} ~= beta_c "
        f"(ratios {v['per_link_observation']['vs_beta_c']}) — "
        "a per-link convention would put the substrate AT criticality; "
        "logged for future work only",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
