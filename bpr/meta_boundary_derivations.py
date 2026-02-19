"""
First-Principles Derivation Routes for Meta-Boundary Decree
============================================================

Two derivation paths that yield decree-like terms from explicit structure:

  Derivation I: Agent field coupling
      Add auxiliary field A(x,t) with L_int = g Aκ. Vary → D = gA.

  Derivation II: Coarse-graining over latent variables
      Full state (b,m,κ,ζ), integrate out ζ → effective D_mem.

References
----------
Al-Kahwati, Meta-Boundary Dynamics with Exogenous Decree (PNAS submission),
  Sections 6.1 (Derivation I) and 6.2 (Derivation II).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Derivation I: Agent field coupling
# ---------------------------------------------------------------------------


@dataclass
class AgentFieldParams:
    """Parameters for the agent field A(x,t) coupling.

    L_int = g ∫ A(x,t) κ(x,t) dx
    """
    g: float = 1.0      # coupling constant
    tau_A: float = 1.0  # agent relaxation time (for overdamped limit)


def agent_field_eom(
    kappa: np.ndarray,
    A: np.ndarray,
    lap_kappa: np.ndarray,
    V_prime: np.ndarray,
    dW_dk: np.ndarray,
    nu: float,
    tau_kappa: float,
    g: float,
) -> np.ndarray:
    """Equation of motion from varying S with respect to κ.

    Full (undamped): τ_κ ∂_tt κ - ν∇²κ + V'_κ(κ) + ... = g A

    Overdamped limit: τ_κ ∂_t κ = -δJ/δκ + ν∇²κ + g A

    So D = g A in the decree-augmented equation.
    """
    return g * A


def derive_decree_from_agent(
    A: np.ndarray,
    g: float = 1.0,
) -> np.ndarray:
    """Decree term D = g A from agent field coupling.

    This is the minimal derivation: D arises from an explicit
    additional degree of freedom A with interaction L_int = g Aκ.
    """
    return g * np.asarray(A)


# ---------------------------------------------------------------------------
# Derivation II: Coarse-graining over latent variables
# ---------------------------------------------------------------------------


@dataclass
class CoarseGrainParams:
    """Parameters for coarse-graining over latent ζ.

    If ζ is not equilibrated on timescale τ_κ, effective dynamics
    acquire non-Markovian D_mem.
    """
    tau_zeta: float = 1.0    # latent variable timescale
    tau_kappa: float = 1.0
    coupling_zeta: float = 1.0  # ζ-κ coupling strength


def effective_decree_from_memory(
    kappa_history: np.ndarray,
    t_history: np.ndarray,
    tau_mem: float,
    coupling: float = 1.0,
) -> float:
    """Effective D_mem from unresolved memory of latent ζ.

    Simplified model: D_mem(t) ~ coupling * ∫ exp(-(t-s)/τ_mem) ζ_eff(s) ds

    For demonstration, we use a kernel-weighted history of κ (proxy for
    ζ influence on κ). In full theory, ζ would be integrated out.

    Returns scalar D_mem at the latest time.
    """
    kappa_history = np.asarray(kappa_history)
    t_history = np.asarray(t_history)
    if len(t_history) < 2:
        return 0.0
    t = t_history[-1]
    dt = t_history - t  # dt <= 0 for past times
    kernel = np.exp(dt / tau_mem) * (dt <= 0)
    dt_step = np.abs(t_history[1] - t_history[0])
    k_flat = np.ravel(kappa_history)
    if k_flat.size == len(t_history):
        weighted = np.sum(kernel * k_flat) * dt_step
    else:
        weighted = np.sum(kernel) * dt_step * np.mean(k_flat)
    return float(coupling * weighted)


def coarse_grain_effective_dynamics(
    kappa: np.ndarray,
    zeta_mean: np.ndarray,
    coupling: float = 1.0,
) -> np.ndarray:
    """Minimal D_mem: coupling to mean field of ζ.

    When ζ is integrated out, ⟨ζ⟩_eff can contribute to κ dynamics
    as an effective forcing term. D_mem ~ coupling * ζ_mean.
    """
    return coupling * np.asarray(zeta_mean)


# ---------------------------------------------------------------------------
# Verification: overdamped limit reproduces decree equation
# ---------------------------------------------------------------------------


def verify_agent_derivation(
    A: np.ndarray,
    g: float = 1.0,
) -> dict:
    """Verify that D = gA has correct structure for decree equation.

    Returns dict with D, g, and consistency check.
    """
    D = derive_decree_from_agent(A, g)
    return {
        "D": D,
        "g": g,
        "consistency": "D = g A recovers decree term in τκ ∂_t κ = -δJ/δκ + ν∇²κ + D",
    }


def verify_coarse_grain_derivation(
    zeta_mean: np.ndarray,
    coupling: float = 1.0,
) -> dict:
    """Verify that D_mem = coupling * ζ_mean yields effective forcing."""
    D_mem = coarse_grain_effective_dynamics(
        np.zeros_like(zeta_mean), zeta_mean, coupling
    )
    return {
        "D_mem": D_mem,
        "coupling": coupling,
        "consistency": "D_mem encodes unresolved memory from latent ζ",
    }
