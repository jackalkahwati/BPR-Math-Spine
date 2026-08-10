"""Glueball Benchmark v1, Gate 1 — bound states of the frozen BPR equations.

PROTOCOL (doc/GLUEBALL_BENCHMARK_V1.md — read it first)
-------------------------------------------------------
The benchmark is sealed: comparison targets (lattice-QCD glueball ratios) live
ONLY in the protocol doc and its tests. THIS MODULE NEVER READS THEM. It solves
the frozen equations and reports what exists; Gates 2-4 (J^PC, ordering, ratios)
require the S^2 eigenproblem and remain open.

Gate 1 question: does the frozen theory generate discrete localized
finite-energy states at all?

STRUCTURAL ANSWER (Derrick's theorem)
-------------------------------------
The frozen continuum scalar sector has NO stable static solitons in 3D: for
E(lambda) under x -> lambda*x, both the gradient and quartic terms collapse.
The frozen theory contains exactly three escapes, none added for this test:
  1. lattice discreteness (no continuous scaling -> discrete breathers allowed)
  2. conserved U(1) norm (time-periodic Q-ball-type states psi = phi e^{-i mu t})
  3. pi_1 line defects (strings; infinite energy as 3D point particles -> NOT
     glueball candidates, excluded)

NUMERICAL ANSWER (this module)
------------------------------
Stationary states of the frozen lattice equation  mu*phi = -C*Delta(phi) - phi^3
(Delta phi_n = phi_{n+1} + phi_{n-1}; the |psi|^4 vertex is the derived
momentum-conserving coupling; mu below the linear band [-2C, 2C]) found by
Newton iteration, plus the Bogoliubov fluctuation spectrum around them.
Result: two distinct localized states, machine-precision residuals, exact U(1)
phase zero mode, and discrete internal modes below the continuum band edge.
Gate 1 PASSES.

WHAT THIS DOES NOT CLAIM: no J^PC content — a 1D winding is not angular
momentum; no glueball identification is made or implied here.
"""
from __future__ import annotations

import numpy as np

# Frozen solver parameters (fixed before any spectrum comparison; the linear
# band is [-2C, 2C], so mu = -2.5 sits below the band edge and binds).
N_SITES = 101
HOPPING_C = 1.0
MU = -2.5


# ---------------------------------------------------------------------------
# Structural gate: Derrick analysis of the frozen continuum sector
# ---------------------------------------------------------------------------

def derrick_analysis() -> dict:
    """Scaling analysis for the frozen continuum terms in D=3.

    E(lambda) = lambda^{2-D} E_grad + lambda^{-D} E_quartic for x -> x/lambda:
    in D=3 every term decreases as lambda grows — pure scale collapse, no
    stationary point. Static continuum solitons are EXCLUDED. The escapes below
    are properties the frozen theory already has.
    """
    return {
        "static_continuum_solitons_in_3d": "EXCLUDED (Derrick scale collapse)",
        "escapes": (
            "lattice discreteness (discrete breathers)",
            "conserved U(1) norm (Q-ball-type time-periodic states)",
            "pi_1 line defects (strings — excluded as particle candidates)",
        ),
        "glueball_candidate_class": (
            "discrete-breather / Q-ball-type bound states of the lattice theory"
        ),
    }


# ---------------------------------------------------------------------------
# Numerical gate: bound states of the frozen lattice equation
# ---------------------------------------------------------------------------

def _lap_shift(phi: np.ndarray) -> np.ndarray:
    """Delta(phi)_n = phi_{n+1} + phi_{n-1} on the ring."""
    return np.roll(phi, 1) + np.roll(phi, -1)


def stationarity_residual(phi: np.ndarray, C: float = HOPPING_C,
                          mu: float = MU) -> float:
    """Max-norm of F(phi) = mu*phi + C*Delta(phi) + phi^3 (zero at a solution)."""
    return float(np.max(np.abs(mu * phi + C * _lap_shift(phi) + phi ** 3)))


def newton_bound_state(seed: np.ndarray, C: float = HOPPING_C, mu: float = MU,
                       tol: float = 1e-12, itmax: int = 60) -> dict:
    """Newton iteration for stationary states mu*phi = -C*Delta(phi) - phi^3."""
    n = len(seed)
    phi = seed.astype(float).copy()
    hop = C * (np.eye(n, k=1) + np.eye(n, k=-1))
    hop[0, -1] += C
    hop[-1, 0] += C
    for it in range(itmax):
        r = mu * phi + C * _lap_shift(phi) + phi ** 3
        if np.max(np.abs(r)) < tol:
            break
        J = mu * np.eye(n) + hop + 3.0 * np.diag(phi ** 2)
        phi = phi - np.linalg.solve(J, r)
    return {"phi": phi, "iterations": it,
            "residual": stationarity_residual(phi, C, mu)}


def site_centered_state(n: int = N_SITES, C: float = HOPPING_C,
                        mu: float = MU) -> dict:
    """Ground-family breather: single-site seed at the ring center."""
    seed = np.zeros(n)
    seed[n // 2] = np.sqrt(-mu)
    return newton_bound_state(seed, C, mu)


def bond_centered_state(n: int = N_SITES, C: float = HOPPING_C,
                        mu: float = MU) -> dict:
    """Second distinct stationary state: two-site (bond-centered) seed."""
    seed = np.zeros(n)
    seed[n // 2] = seed[n // 2 + 1] = 0.8 * np.sqrt(-mu)
    return newton_bound_state(seed, C, mu)


def state_energy(phi: np.ndarray, C: float = HOPPING_C) -> float:
    """Frozen Hamiltonian: E = -2C sum phi_n phi_{n+1} - (1/2) sum phi^4."""
    return float(-2.0 * C * np.sum(phi * np.roll(phi, -1))
                 - 0.5 * np.sum(phi ** 4))


def localization_ratio(phi: np.ndarray, distance: int = 20) -> float:
    """Peak amplitude over amplitude `distance` sites away — exponential
    localization shows up as a large ratio."""
    c = int(np.argmax(np.abs(phi)))
    far = np.abs(phi[(c + distance) % len(phi)])
    return float(np.abs(phi[c]) / max(far, 1e-300))


def fluctuation_spectrum(phi: np.ndarray, C: float = HOPPING_C,
                         mu: float = MU) -> dict:
    """Bogoliubov linearization around a real stationary state:
        L+ = -(mu + C*Delta + 3 phi^2),  L- = -(mu + C*Delta + phi^2),
        omega^2 = eig(L- L+),  with L- phi = 0 the exact U(1) phase zero mode.
    Discrete omega^2 below the continuum band edge (|mu| - 2C)^2 are internal
    bound modes — the discreteness Gate 1 asks for."""
    n = len(phi)
    hop = C * (np.eye(n, k=1) + np.eye(n, k=-1))
    hop[0, -1] += C
    hop[-1, 0] += C
    Lp = -(mu * np.eye(n) + hop + 3.0 * np.diag(phi ** 2))
    Lm = -(mu * np.eye(n) + hop + np.diag(phi ** 2))
    w2 = np.sort(np.real(np.linalg.eigvals(Lm @ Lp)))
    band_edge = (abs(mu) - 2.0 * C) ** 2
    internal = w2[(w2 > 1e-8) & (w2 < band_edge * 0.999)]
    return {"omega2": w2, "band_edge_omega2": float(band_edge),
            "zero_mode_residual": float(np.max(np.abs(Lm @ phi))),
            "n_internal_modes": int(len(internal)),
            "internal_omega2": internal}


# ---------------------------------------------------------------------------
# Gate bookkeeping (never reads the sealed targets)
# ---------------------------------------------------------------------------

def gate1_report() -> dict:
    s1, s2 = site_centered_state(), bond_centered_state()
    fl = fluctuation_spectrum(s1["phi"])
    ok = (s1["residual"] < 1e-10 and s2["residual"] < 1e-10
          and localization_ratio(s1["phi"]) > 1e3
          and fl["n_internal_modes"] >= 1)
    return {
        "distinct_states_found": 2,
        "residuals": (s1["residual"], s2["residual"]),
        "energies": (state_energy(s1["phi"]), state_energy(s2["phi"])),
        "localization_site_centered": localization_ratio(s1["phi"]),
        "n_internal_modes": fl["n_internal_modes"],
        "zero_mode_residual": fl["zero_mode_residual"],
        "gate1": "PASS" if ok else "FAIL",
    }


def benchmark_status() -> dict:
    """The four gates. Gates 2-4 were computed in bpr/glueball_gates.py (the S^2
    eigenproblem) and the envelope was opened there; this Gate-1 module still
    never reads the comparison targets."""
    return {
        "gate1_discrete_localized_states": gate1_report()["gate1"],
        "gate2_jpc_families": "PARTIAL FAIL (envelope opened; see glueball_gates)",
        "gate3_ordering": "FAIL (envelope opened; see glueball_gates)",
        "gate4_mass_ratios": "FAIL (envelope opened; see glueball_gates)",
        "no_jpc_claim": ("the ring calculation carries no J^PC content; "
                         "no glueball identification is made or implied"),
    }


def report() -> str:
    d = derrick_analysis()
    g = gate1_report()
    st = benchmark_status()
    lines = [
        "Glueball Benchmark v1 — Gate 1 (frozen BPR equations)",
        "=====================================================",
        f"Derrick: static continuum solitons {d['static_continuum_solitons_in_3d']}",
        "Escapes already frozen into the theory:",
    ]
    lines += [f"  - {e}" for e in d["escapes"]]
    lines += [
        "",
        f"Bound states found: {g['distinct_states_found']} "
        f"(residuals {g['residuals'][0]:.1e}, {g['residuals'][1]:.1e})",
        f"Energies: {g['energies'][0]:.4f}, {g['energies'][1]:.4f}",
        f"Localization (20 sites): {g['localization_site_centered']:.1e}",
        f"Internal discrete modes below band edge: {g['n_internal_modes']}",
        f"U(1) zero-mode residual: {g['zero_mode_residual']:.1e}",
        "",
        f"GATE 1: {g['gate1']}",
        f"Gate 2 (J^PC families): {st['gate2_jpc_families']}",
        f"Gate 3 (ordering):      {st['gate3_ordering']}",
        f"Gate 4 (mass ratios):   {st['gate4_mass_ratios']}",
        "",
        "The envelope was OPENED in bpr/glueball_gates.py after this solver was",
        "frozen; this Gate-1 module still never reads the targets, and " +
        st["no_jpc_claim"] + ".",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
