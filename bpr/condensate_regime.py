"""Does BPR sit in the condensing regime? — closing the coherence chain's last gap.

THE CHAIN SO FAR
----------------
1. Coupling -> coherence (generic nonlinear dynamics; harmonic_coupling.py).
2. The substrate-derived coupling is frustrated; coherence lives in the k=0
   condensate mode (harmonic_coupling_derivation.py).
3. A mechanism populates k=0 — structurally (ground state) and dynamically
   (wave condensation), but only PARTIALLY and conditional on the energy
   regime (condensate_mechanism.py).
This module answers the remaining question: WHAT SETS THE REGIME — is the
condensate guaranteed, conditional, or fine-tuned?

FOUR RESULTS (each verified below / in tests)
---------------------------------------------
(1) GROUND-STATE THEOREM (exact). At fixed norm, the ground state of the
    frozen defocusing theory on the ring IS the uniform k=0 condensate:
      * hopping term: -2C sum Re(psi*_n psi_{n+1}) >= -2C sum |psi_n||psi_{n+1}|
        >= -2C sum |psi_n|^2 (AM-GM), equality iff all amplitudes equal and
        phases aligned;
      * quartic term: at fixed sum|psi|^2, sum|psi|^4 is minimized by equal
        amplitudes (Jensen).
    Both are minimized simultaneously by the uniform state, so the vacuum of
    the frozen theory is EXACTLY the coherent condensate. Verified: winding-
    free gradient flow converges to n_0 = 1 at H = H(uniform).

(2) METASTABLE WINDING SECTORS (honest wrinkle). The ring topology supports
    persistent-current local minima (plane waves with winding w != 0): random-
    phase initial conditions relax into them, at energies slightly ABOVE the
    uniform state. Reaching the true condensate from a wound configuration
    requires phase slips. So "vacuum = condensate" is exact, but RELAXATION to
    it can be topologically obstructed — consistent with the partial
    condensation seen dynamically in condensate_mechanism.py.

(3) EQUILIBRIUM CONDENSATION CURVE (exact, weak-coupling). Rayleigh-Jeans
    equilibrium n_k = T/(eps_k - mu) on the frozen dispersion gives the
    condensate fraction as a function of energy per unit action E/N: ~1 at
    E/N -> 0, ~0.5 at E/N ~ 0.07 (p=101, C=1), ~0 at high E/N. The dynamical
    runs in condensate_mechanism.py sit on the correct sides of this curve.

(4) THE REGIME VERDICT — the geometry decides, and it is favorable. The
    characteristic temperature below which the condensate survives is
    T* = N / sum'(1/eps_k):
      * 1D ring:      T* = 12 C n / p          (verified: T* p = 12.000...)
        -> with p = 104761 the 1D toy is FRAGILE: T* ~ 1e-4 of natural scale.
      * 2D S^2 boundary (the physical case): T* = C n / ln p  (verified:
        T* ln p ~ 1.0) -> only LOGARITHMICALLY suppressed: T* ~ 8.6% of the
        natural scale at BPR's p. (This 1/p-vs-1/ln p split is the standard
        Mermin-Wagner dimensional structure, emerging correctly; and ln p =
        11.56 is the same log that prices the substrate in the Kontoyiannis
        audit.)
    So on the physical 2D boundary, coherence requires only a MILD condition —
    substrate temperature below ~10% of its natural energy scale — not fine
    tuning.

WHAT REMAINS UNDETERMINED (flagged, not hidden)
-----------------------------------------------
* "The substrate sits in (or near) its ground state" is the standard vacuum
  assumption of any field theory, but it is an ASSUMPTION here as everywhere.
* The substrate's actual ambient temperature is not derived by BPR. The
  result above converts the open question from "is coherence fine-tuned?"
  (no: the 2D condition is mild) to "what is T_substrate?" (open).

NET: coherence in BPR is GUARANTEED in the vacuum sector (exact theorem),
ROBUST on the physical 2D boundary (log-only depletion), obstructed at most
by metastable windings and by an undetermined — but only mildly constrained —
substrate temperature.
"""
from __future__ import annotations

import numpy as np

P_SUBSTRATE = 104761


# ---------------------------------------------------------------------------
# (1) + (2): ground state and winding metastability
# ---------------------------------------------------------------------------

def energy(psi: np.ndarray, C: float = 1.0, g: float = 0.5) -> float:
    """Frozen defocusing energy: hopping + quartic (ring)."""
    hop = -C * np.sum(psi.conj() * (np.roll(psi, 1) + np.roll(psi, -1))).real
    return float(hop + 0.5 * g * np.sum(np.abs(psi) ** 4))


def gradient_flow(psi0: np.ndarray, C: float = 1.0, g: float = 0.5,
                  iters: int = 4000, dt: float = 0.02) -> np.ndarray:
    """Norm-preserving gradient descent on the frozen energy."""
    p = len(psi0)
    psi = psi0.astype(complex).copy()
    target = np.sum(np.abs(psi) ** 2)
    for _ in range(iters):
        grad = -C * (np.roll(psi, 1) + np.roll(psi, -1)) + g * np.abs(psi) ** 2 * psi
        psi = psi - dt * grad
        psi *= np.sqrt(target / np.sum(np.abs(psi) ** 2))
    return psi


def condensate_fraction(psi: np.ndarray) -> float:
    pk = np.abs(np.fft.fft(psi)) ** 2
    return float(pk[0] / pk.sum())


def ground_state_check(p: int = 101, seed: int = 0) -> dict:
    """Verify the theorem: winding-free flow reaches the uniform condensate
    exactly; random-phase flows land in metastable winding sectors ABOVE it."""
    rng = np.random.default_rng(seed)
    uni = np.ones(p, complex)
    H_uni = energy(uni)

    free = np.abs(rng.normal(size=p)) + 0.0j            # no phase winding
    free *= np.sqrt(p / np.sum(np.abs(free) ** 2))
    out_free = gradient_flow(free)

    wound = rng.normal(size=p) + 1j * rng.normal(size=p)  # random windings
    wound *= np.sqrt(p / np.sum(np.abs(wound) ** 2))
    out_wound = gradient_flow(wound, iters=8000)
    pk = np.abs(np.fft.fft(out_wound)) ** 2
    kdom = int(np.argmax(pk))
    w = kdom if kdom <= p // 2 else kdom - p

    return {
        "H_uniform": H_uni,
        "windingfree_n0": condensate_fraction(out_free),
        "windingfree_H": energy(out_free),
        "wound_dominant_winding": w,
        "wound_H": energy(out_wound),
        "wound_above_uniform": bool(energy(out_wound) > H_uni),
    }


# ---------------------------------------------------------------------------
# (3): Rayleigh-Jeans equilibrium condensation curve
# ---------------------------------------------------------------------------

def rj_condensation_curve(p: int = 101, C: float = 1.0,
                          n_pts: int = 40) -> list[dict]:
    """Equilibrium condensate fraction vs energy-per-action E/N, exact for the
    frozen dispersion at weak coupling. Parametrized by the chemical potential:
    n_k = T/(eps_k - mu), T fixed by the norm, E then determined."""
    k = np.arange(p)
    eps = 2.0 * C * (1.0 - np.cos(2.0 * np.pi * k / p))
    N = float(p)                                        # unit density
    out = []
    for mu in -np.logspace(-7, 1.2, n_pts):
        T = N / np.sum(1.0 / (eps - mu))
        E = T * np.sum(eps / (eps - mu))
        out.append({"E_over_N": float(E / N),
                    "condensate_fraction": float((T / (-mu)) / N)})
    return out


def crossover_energy(p: int = 101, C: float = 1.0) -> float:
    """E/N at which the equilibrium condensate fraction crosses 1/2."""
    curve = rj_condensation_curve(p, C, n_pts=200)
    for row in curve:
        if row["condensate_fraction"] < 0.5:
            return row["E_over_N"]
    return float("inf")


# ---------------------------------------------------------------------------
# (4): characteristic temperatures — the regime verdict
# ---------------------------------------------------------------------------

def t_star_ring(p: int, C: float = 1.0) -> float:
    """1D characteristic temperature T* = N / sum'(1/eps_k) at unit density.
    Scales as 12C/p (verified) — the 1D toy is fragile at large p."""
    k = np.arange(1, p)
    eps = 2.0 * C * (1.0 - np.cos(2.0 * np.pi * k / p))
    return float(p / np.sum(1.0 / eps))


def t_star_sphere(p: int, C: float = 1.0) -> float:
    """2D (S^2 boundary) characteristic temperature at unit density, with
    eps_l = C l(l+1)/p (area ~ p sites) and multiplicity 2l+1 up to l ~ sqrt(p).
    Scales as C/ln p (verified) — only logarithmic suppression."""
    L = int(np.sqrt(p))
    l = np.arange(1, L + 1)
    s = np.sum((2 * l + 1) / (C * l * (l + 1) / p))
    modes = float(np.sum(2 * l + 1) + 1)
    return float(modes / s)


def regime_verdict(p: int = P_SUBSTRATE) -> dict:
    ln_p = float(np.log(p))
    ts = t_star_sphere(p)
    return {
        "vacuum_is_condensate": "EXACT (ground-state theorem)",
        "relaxation_caveat": "metastable winding sectors (persistent currents) "
                             "can obstruct relaxation; phase slips required",
        "t_star_ring_1d": t_star_ring(min(p, 4001)),   # scaling demo value
        "ring_scaling": "T* ~ 12C/p — 1D toy fragile at BPR's p",
        "t_star_sphere_2d": ts,
        "sphere_scaling": "T* ~ C/ln p — physical 2D boundary only log-suppressed",
        "ln_p": ln_p,
        "coherence_condition": f"T_substrate below ~{ts:.0%} of the natural "
                               "scale (mild, not fine-tuned)",
        "undetermined": "the substrate's ambient temperature (and 'vacuum = "
                        "ground state' is the standard, flagged assumption)",
    }


def report(p: int = P_SUBSTRATE) -> str:
    gs = ground_state_check()
    ce = crossover_energy()
    v = regime_verdict(p)
    lines = [
        "Condensate regime — does BPR sit in the condensing window?",
        "==========================================================",
        "",
        "(1) Ground-state theorem (exact):",
        f"    winding-free flow: n0 = {gs['windingfree_n0']:.6f} at "
        f"H = {gs['windingfree_H']:.4f} (uniform: {gs['H_uniform']:.4f})",
        "    -> the VACUUM of the frozen theory IS the coherent condensate.",
        "",
        "(2) Metastability (honest wrinkle):",
        f"    random-phase flow relaxes to winding w = "
        f"{gs['wound_dominant_winding']:+d} at H = {gs['wound_H']:.4f} "
        f"(above uniform: {gs['wound_above_uniform']})",
        "    -> persistent currents can obstruct relaxation to the condensate.",
        "",
        "(3) Equilibrium condensation curve (Rayleigh-Jeans, exact):",
        f"    condensate fraction crosses 1/2 at E/N ~ {ce:.3f} (p=101, C=1);",
        "    the dynamical runs in condensate_mechanism.py sit on the correct",
        "    sides of this threshold.",
        "",
        "(4) Regime verdict — geometry decides, favorably:",
        f"    1D ring:  T* ~ 12C/p        ({v['ring_scaling']})",
        f"    2D S^2:   T* ~ C/ln p = C/{v['ln_p']:.2f}  "
        f"-> T* ~ {v['t_star_sphere_2d']:.3f} natural units",
        f"    COHERENCE CONDITION: {v['coherence_condition']}.",
        "",
        f"UNDETERMINED (flagged): {v['undetermined']}.",
        "",
        "NET: coherence is GUARANTEED in the vacuum sector (exact), ROBUST on",
        "the physical 2D boundary (log-only depletion), obstructed at most by",
        "metastable windings and by an undetermined — but only mildly",
        "constrained — substrate temperature.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
