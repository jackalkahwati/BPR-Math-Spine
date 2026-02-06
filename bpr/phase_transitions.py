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
    {"name": "Consciousness onset",      "class": TransitionClass.A, "order_parameter": "W (winding)"},
    {"name": "Superfluid He-4",          "class": TransitionClass.A, "order_parameter": "W (vortex)"},
    {"name": "QCD confinement",          "class": TransitionClass.B, "order_parameter": "Polyakov loop"},
    {"name": "Metal-insulator",          "class": TransitionClass.B, "order_parameter": "Conductivity"},
    {"name": "Superconductivity",        "class": TransitionClass.C, "order_parameter": "Gap Δ"},
    {"name": "Ferromagnetism",           "class": TransitionClass.C, "order_parameter": "Magnetisation M"},
    {"name": "Decoherence",              "class": TransitionClass.C, "order_parameter": "Ψ (coherence)"},
    {"name": "Electroweak",              "class": TransitionClass.D, "order_parameter": "Higgs VEV"},
    {"name": "Chiral symmetry breaking", "class": TransitionClass.D, "order_parameter": "⟨ψ̄ψ⟩"},
    {"name": "BEC",                      "class": "A+C",             "order_parameter": "Condensate fraction"},
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

def superconductor_tc(Z_material: float, Z0: float = 376.73,
                       T_debye: float = 300.0) -> float:
    """Critical temperature for superconductivity from impedance matching.

    Class C impedance transition: the material becomes superconducting
    when its boundary impedance Z_material matches the substrate vacuum
    impedance Z₀.

    T_c ~ T_Debye × exp(−Z₀ / |Z₀ − Z_material|)

    Materials with Z_material ≈ Z₀ have highest T_c.
    This reproduces the BCS exponential form with a geometric interpretation.

    Parameters
    ----------
    Z_material : float – material boundary impedance (Ω)
    Z0 : float – vacuum impedance (Ω)
    T_debye : float – Debye temperature (K)

    Returns
    -------
    float – predicted T_c (K)
    """
    delta_Z = abs(Z0 - Z_material)
    if delta_Z < 1e-12:
        return T_debye  # Perfect impedance match → T_c = T_Debye
    # Normalised mismatch: ε = |ΔZ| / Z₀ ∈ [0, ∞)
    # T_c = T_Debye × exp(−1/ε)  (BCS-like, geometric interpretation)
    epsilon = delta_Z / Z0
    if epsilon < 1e-12:
        return T_debye
    inv_epsilon = 1.0 / epsilon
    if inv_epsilon > 700:
        return 0.0
    return T_debye * np.exp(-inv_epsilon)


def classify_transition(name: str) -> Optional[TransitionClass]:
    """Look up a transition's BPR class from the catalog."""
    for entry in TRANSITION_CATALOG:
        if entry["name"].lower() == name.lower():
            cls = entry["class"]
            if isinstance(cls, TransitionClass):
                return cls
    return None
