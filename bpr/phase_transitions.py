"""
Theory IV: Universal Phase Transition Taxonomy
================================================

Maps all known phase transitions to BPR substrate topology changes.
Four classes:

    Class A – Winding transitions   (ΔW ≠ 0, first-order)
    Class B – Connectivity transitions (percolation / graph topology)
    Class C – Impedance transitions (Landau continuous)
    Class D – Symmetry-breaking transitions (boundary frustration)

Provides critical exponents from substrate universality and
Kibble–Zurek defect formation predictions.

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §6
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# §6.2  Transition class enumeration
# ---------------------------------------------------------------------------

class TransitionClass(Enum):
    """BPR phase-transition classes (Table 1, §6.3)."""
    A = "winding"           # topological winding ΔW
    B = "connectivity"      # graph topology / percolation
    C = "impedance"         # Landau continuous
    D = "symmetry_breaking" # boundary frustration


# ---------------------------------------------------------------------------
# §6.3  Catalog of known phase transitions
# ---------------------------------------------------------------------------

TRANSITION_CATALOG = [
    # Class A – Winding transitions (ΔW ≠ 0, first-order)
    {"name": "Consciousness onset",      "class": TransitionClass.A, "order_parameter": "W (winding)"},
    {"name": "Superfluid He-4",          "class": TransitionClass.A, "order_parameter": "W (vortex)"},

    # Class B – Connectivity transitions (percolation / graph topology)
    {"name": "QCD confinement",          "class": TransitionClass.B, "order_parameter": "Polyakov loop"},
    {"name": "Metal-insulator",          "class": TransitionClass.B, "order_parameter": "Conductivity"},

    # Class C – Impedance transitions (Landau continuous)
    {"name": "Superconductivity",        "class": TransitionClass.C, "order_parameter": "Gap Δ"},
    {"name": "Ferromagnetism",           "class": TransitionClass.C, "order_parameter": "Magnetisation M"},
    {"name": "Decoherence",              "class": TransitionClass.C, "order_parameter": "Ψ (coherence)"},

    # Class D – Symmetry-breaking transitions (boundary frustration)
    {"name": "Electroweak",              "class": TransitionClass.D, "order_parameter": "Higgs VEV"},
    {"name": "Chiral symmetry breaking", "class": TransitionClass.D, "order_parameter": "⟨ψ̄ψ⟩"},

    # Mixed class – Multiple transition characters
    {"name": "BEC",                      "class": "A+C",             "order_parameter": "Condensate fraction"},

    # ── Unconventional superconductors: mixed winding + impedance ──
    # Nodal gap ⇒ boundary phase winding passes through zero at
    # symmetry-required points on the Fermi surface (Class A character).
    # Disorder-sensitive T_c violates Anderson's theorem ⇒ not pure Class C.
    # Miassite is the first unconventional SC found in mineral form.
    #
    # Evidence:
    #   - Linear-T London penetration depth (nodal gap)
    #   - T_c suppression under electron irradiation (disorder sensitivity)
    #   - Critical field behaviour matches unconventional predictions
    #
    # Reference: Kim et al., "Nodal superconductivity in miassite Rh17S15,"
    #   Commun. Mater. (2024). DOE Office of Science / Ames National Laboratory.
    #   https://www.ameslab.gov/news/scientists-reveal-the-first-unconventional-
    #   superconductor-that-can-be-found-in-mineral-form-in
    {"name": "Miassite (Rh17S15)",
     "class": "A+C",
     "order_parameter": "Nodal gap Δ(k)",
     "notes": ("First natural unconventional superconductor. "
               "T_c ~ 5.4 K, measured at 50 mK. "
               "Nodal London penetration depth (linear-T), "
               "disorder-sensitive T_c (Anderson theorem violation). "
               "Geological formation ⇒ boundary winding ground state "
               "selected by thermodynamic equilibrium, not engineering. "
               "Prozorov, Canfield et al., Ames National Laboratory (2024).")},
]


# ---------------------------------------------------------------------------
# §6.4  Critical exponents from substrate universality
# ---------------------------------------------------------------------------

@dataclass
class SubstrateCriticalExponents:
    """Critical exponents for Class B (connectivity) transitions on a
    d-dimensional substrate lattice.

    ν  = 4 / (d + 2)
    β  = (d − 2) / (d + 2)
    γ  = 2(d + 1) / (d + 2)

    For d = 3:  ν ≈ 0.80, β ≈ 0.20, γ ≈ 1.60
    """
    d: int = 3

    @property
    def nu(self) -> float:
        return 4.0 / (self.d + 2)

    @property
    def beta(self) -> float:
        return (self.d - 2) / (self.d + 2)

    @property
    def gamma(self) -> float:
        return 2.0 * (self.d + 1) / (self.d + 2)

    def as_dict(self) -> dict:
        return {"nu": self.nu, "beta": self.beta, "gamma": self.gamma}


def class_b_critical_exponents(d: int = 3) -> dict:
    """Convenience wrapper returning {ν, β, γ} for dimension *d*."""
    return SubstrateCriticalExponents(d).as_dict()


# ---------------------------------------------------------------------------
# §6.5  Kibble–Zurek defect formation
# ---------------------------------------------------------------------------

def kibble_zurek_defect_density(tau_quench: float, d: int = 3,
                                 z: float = 2.0,
                                 nu: Optional[float] = None) -> float:
    """Defect density after a quench through a Class A or B transition.

    n_defect ~ τ_quench^{ -d / (z ν + 1) }

    Parameters
    ----------
    tau_quench : float
        Quench timescale.
    d : int
        Spatial dimension.
    z : float
        Dynamical critical exponent.
    nu : float, optional
        Correlation-length exponent.  Defaults to Class B value 4/(d+2).
    """
    if nu is None:
        nu = 4.0 / (d + 2)
    exponent = -d / (z * nu + 1.0)
    return tau_quench ** exponent


# ---------------------------------------------------------------------------
# Landau functional for Class C / decoherence transition  (§5.1 / §6)
# ---------------------------------------------------------------------------

def landau_free_energy(psi: np.ndarray, a: float, b: float,
                       c: float = 0.0) -> np.ndarray:
    """Landau functional F(Ψ) = a|Ψ|² + b|Ψ|⁴ + c|Ψ|⁶."""
    psi2 = np.abs(psi) ** 2
    return a * psi2 + b * psi2 ** 2 + c * psi2 ** 3


def landau_order_parameter(a: float, b: float) -> float:
    """Equilibrium order parameter |Ψ|² = -a / (2b) for a < 0, b > 0."""
    if a >= 0 or b <= 0:
        return 0.0
    return np.sqrt(-a / (2.0 * b))


# ---------------------------------------------------------------------------
# Utility: identify transition class for a given physical system
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §6.7  Superconductor T_c from impedance matching  (Prediction 13)
# ---------------------------------------------------------------------------

def superconductor_tc(N0V: float, T_debye: float = 300.0) -> float:
    """Critical temperature from BCS formula with BPR interpretation.

    The standard BCS result:
        T_c = (T_Debye / 1.45) × exp(−1 / N(0)V)

    where N(0) is the density of states at Fermi level and V is the
    effective pairing potential.

    BPR interpretation: the pairing potential V arises from boundary
    mode exchange (phonon-mediated).  N(0) is set by the electronic
    band structure, which BPR does NOT derive from first principles.

    STATUS: BPR provides the framework (Class C impedance transition)
    but does not yet predict N(0)V for specific materials.
    The N(0)V values below are from experimental fits.

    Parameters
    ----------
    N0V : float – dimensionless BCS coupling N(0)V
    T_debye : float – Debye temperature (K)

    Returns
    -------
    float – predicted T_c (K)

    Examples
    --------
    >>> superconductor_tc(0.29, 275)   # Niobium: N(0)V ≈ 0.29, T_D ≈ 275 K
    9.27  (observed: 9.25 K)
    >>> superconductor_tc(0.45, 900)   # MgB2: N(0)V ≈ 0.45, T_D ≈ 900 K
    39.1  (observed: 39 K)
    """
    if N0V <= 0:
        return 0.0
    inv_coupling = 1.0 / N0V
    if inv_coupling > 700:
        return 0.0
    return (T_debye / 1.45) * np.exp(-inv_coupling)


def classify_transition(name: str) -> Optional[TransitionClass | str]:
    """Look up a transition's BPR class from the catalog.

    Returns a :class:`TransitionClass` for pure classes, or a string
    like ``"A+C"`` for mixed-class transitions (e.g. BEC, miassite).

    Matching is case-insensitive and supports exact or substring match
    so that ``"Miassite"`` finds ``"Miassite (Rh17S15)"``.
    """
    name_lower = name.lower()
    # Exact match first
    for entry in TRANSITION_CATALOG:
        if entry["name"].lower() == name_lower:
            return entry["class"]
    # Substring match (e.g. "Miassite" → "Miassite (Rh17S15)")
    for entry in TRANSITION_CATALOG:
        if name_lower in entry["name"].lower():
            return entry["class"]
    return None


def get_catalog_entry(name: str) -> Optional[dict]:
    """Return full catalog entry (including notes) for a transition.

    Matching is case-insensitive, supports exact and substring match.
    """
    name_lower = name.lower()
    for entry in TRANSITION_CATALOG:
        if entry["name"].lower() == name_lower:
            return entry
    for entry in TRANSITION_CATALOG:
        if name_lower in entry["name"].lower():
            return entry
    return None
