"""Coupled vs uncoupled harmonic fields — does coupling produce emergent coherence?

MOTIVATION / HONESTY (read first)
---------------------------------
This tests a GENERAL and well-established question in nonlinear dynamics:
whether a nonlinear coupling term produces qualitatively different behavior
(phase-locking, attractors, robustness) than linear superposition. The question
stands entirely on its own physics; it is NOT motivated by, and provides NO
evidence for, crop circles or any UAP artifact (those are Tier-0 evidence, see
doc/CLOSED_AND_DEPRECATED.md). The prompt that raised it framed the Tidcombe
formation as a hint — that framing is declined; the dynamical question is kept.

Concretely, compare
    linear   :  Φ = Σ_i φ_i                       (independent modes)
    coupled  :  dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)   (Kuramoto coupling)

The Kuramoto model is the canonical, textbook realization of the user's
Φ = Σφ_i + Σ K_ij φ_i φ_j idea (the sin coupling is the leading nonlinear phase
interaction). Its order parameter r ∈ [0,1] measures global phase coherence.

KNOWN RESULT (this is not novel — it is a 1975 textbook phase transition):
  * K = 0        → r ≈ 1/√N  (incoherent; modes drift independently)
  * K > K_c      → r → 1      (coherent; modes phase-lock)
  for a unimodal frequency distribution, K_c = 2/(π g(0)) (Kuramoto 1975).

What this DOES show: coherence, stability under perturbation, and multistability
are emergent properties of the COUPLING, absent in the linear sum. The user's
central intuition is correct.
What this does NOT show: anything specific to BPR, or that any particular K_ij
matrix is physical. Coupling also generically ADDS parameters (N² of them),
which is a cost unless the K_ij are fixed by substrate symmetry — flagged below.
"""
from __future__ import annotations

import numpy as np


def order_parameter(theta: np.ndarray) -> float:
    """Kuramoto global coherence r = |mean(e^{iθ})| ∈ [0,1]."""
    return float(np.abs(np.mean(np.exp(1j * theta))))


def simulate(K: float, omega: np.ndarray, theta0: np.ndarray,
             steps: int = 3000, dt: float = 0.01) -> dict:
    """Integrate the Kuramoto field. K=0 recovers independent (linear) modes.
    Returns the time-averaged coherence over the second half of the run."""
    N = len(omega)
    theta = theta0.copy()
    r_hist = np.empty(steps)
    for t in range(steps):
        # mean-field form: dθ_i = ω_i + K r sin(ψ − θ_i)
        z = np.mean(np.exp(1j * theta))
        r, psi = np.abs(z), np.angle(z)
        theta = theta + dt * (omega + K * r * np.sin(psi - theta))
        r_hist[t] = r
    return {"r_final": float(np.mean(r_hist[steps // 2:])),
            "r_series": r_hist, "theta": theta}


def critical_coupling(sigma: float = 1.0) -> float:
    """Kuramoto K_c = 2/(π g(0)) for a Gaussian frequency spread (std σ),
    g(0) = 1/(σ√(2π)).  ⇒  K_c = σ·√(8/π)."""
    return sigma * np.sqrt(8.0 / np.pi)


def coupling_sweep(N: int = 300, sigma: float = 1.0, seed: int = 0,
                   K_values=None) -> dict:
    """Sweep K and record steady-state coherence r(K). Shows the transition
    from incoherent (linear-like) to coherent (coupled) fields."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma, N)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    if K_values is None:
        K_values = np.linspace(0.0, 4.0, 17)
    scan = [{"K": float(K), "r": simulate(K, omega, theta0)["r_final"]}
            for K in K_values]
    return {"K_c_theory": critical_coupling(sigma), "scan": scan,
            "r_uncoupled": scan[0]["r"], "r_strong": scan[-1]["r"]}


def perturbation_recovery(K: float, N: int = 300, sigma: float = 1.0,
                          kick: float = 2.0, seed: int = 1) -> dict:
    """Robustness test: settle the field, kick every phase by uniform noise,
    then measure whether coherence RECOVERS. A coupled field (K > K_c) has a
    coherent attractor it returns to; a linear field (K=0) has no restoring
    force and does not recover."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, sigma, N)
    theta0 = rng.uniform(-np.pi, np.pi, N)
    settled = simulate(K, omega, theta0, steps=3000)
    r_before = settled["r_final"]
    kicked = settled["theta"] + rng.uniform(-kick, kick, N)
    r_just_after = order_parameter(kicked)
    recovered = simulate(K, omega, kicked, steps=3000)
    return {"K": K, "r_before": r_before, "r_just_after": r_just_after,
            "r_recovered": recovered["r_final"],
            "recovers": recovered["r_final"] > 0.5 * r_before + 0.4}


def report() -> str:
    sweep = coupling_sweep()
    rec_coupled = perturbation_recovery(K=3.0)
    rec_linear = perturbation_recovery(K=0.0)
    lines = [
        "Coupled vs uncoupled harmonic fields — coherence from interaction",
        "================================================================",
        f"Kuramoto K_c (theory, σ=1): {sweep['K_c_theory']:.2f}",
        "",
        "Coherence r vs coupling K (steady state):",
    ]
    for row in sweep["scan"]:
        bar = "#" * int(row["r"] * 40)
        lines.append(f"  K={row['K']:4.2f}  r={row['r']:.3f}  {bar}")
    lines += [
        "",
        f"Uncoupled (K=0):     r = {sweep['r_uncoupled']:.3f}  (incoherent, ~1/√N)",
        f"Strong coupling:     r = {sweep['r_strong']:.3f}  (phase-locked)",
        "",
        "Robustness (kick all phases, measure recovery):",
        f"  coupled K=3: r {rec_coupled['r_before']:.2f} → kick "
        f"{rec_coupled['r_just_after']:.2f} → recovered "
        f"{rec_coupled['r_recovered']:.2f}   recovers={rec_coupled['recovers']}",
        f"  linear  K=0: r {rec_linear['r_before']:.2f} → kick "
        f"{rec_linear['r_just_after']:.2f} → recovered "
        f"{rec_linear['r_recovered']:.2f}   recovers={rec_linear['recovers']}",
        "",
        "CONCLUSION: coherence and robustness are emergent properties of the",
        "COUPLING, absent in the linear sum. The hypothesis is correct — and it",
        "is standard nonlinear dynamics (Kuramoto 1975), not BPR-specific and not",
        "evidence for any artifact. Coupling also adds O(N²) parameters unless the",
        "K_ij are fixed by substrate symmetry: that constraint is the real open",
        "question, not whether coupling 'works'.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
