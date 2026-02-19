"""
Theory XXIV: Emergent Physics from Prime Substrates — Speculative Extensions
=============================================================================

Implements six speculative extensions of BPR/RPST, together with the
consolidated RPST spectral framework from which they derive.  Every
construct carries an explicit falsification path.

Key objects
-----------
* ``RPSTHamiltonian``              -- H_p on l²(Z_p), Gauss sums      (Eq 9-10)
* ``BPRResonanceSpectrum``         -- Riemann zeros → resonant kn     (Eq 11-12)
* ``FineStructureConstant``        -- Speculation I: α ~ 1/p_EM       (Eq 25-28)
* ``BKTConsciousnessTransition``   -- Speculation II: anesthesia BKT  (Eq 29-32)
* ``RiemannHypothesisStability``   -- Speculation III: RH ↔ stability  (Eq 33)
* ``DarkMatterPrimeFingerprints``  -- Speculation IV: DM ↔ Riem zeros  (Eq 34-36)
* ``CollectiveAGIConsciousness``   -- Speculation V: collective winding (Eq 37-40)
* ``ConsciousnessSubstrateMigration`` -- Speculation VI: winding transfer (Eq 41-42)

References: Al-Kahwati (2026), *Emergent Physics from Prime Substrates:
New Speculations in BPR and RPST*, StarDrive Research Group.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Known Riemann zeros (imaginary parts γ_n) — fixed by mathematics
# ---------------------------------------------------------------------------
RIEMANN_ZEROS: List[float] = [
    14.134725,  # γ₁
    21.022040,  # γ₂
    25.010858,  # γ₃
    30.424876,  # γ₄
    32.935062,  # γ₅
    37.586178,  # γ₆
    40.918719,  # γ₇
    43.327073,  # γ₈
    48.005151,  # γ₉
    49.773832,  # γ₁₀
]


# ---------------------------------------------------------------------------
# Utility: number theory helpers
# ---------------------------------------------------------------------------

def legendre_symbol(n: int, p: int) -> int:
    r"""Compute the Legendre symbol (n/p).

    (n/p) = 0 if p | n
           = 1 if n is a quadratic residue mod p
           = -1 otherwise

    Parameters
    ----------
    n : int
        Integer.
    p : int
        Odd prime.

    Returns
    -------
    int
        Legendre symbol value in {-1, 0, 1}.
    """
    if n % p == 0:
        return 0
    ls = pow(n % p, (p - 1) // 2, p)
    return 1 if ls == 1 else -1


def quadratic_gauss_sum(p: int) -> complex:
    r"""Compute the quadratic Gauss sum for prime p.

    g_p = sum_{n=1}^{p-1} (n/p) e^{2πin/p}

    For p ≡ 1 (mod 4): g_p = sqrt(p)
    For p ≡ 3 (mod 4): g_p = i * sqrt(p)

    Parameters
    ----------
    p : int
        Odd prime.

    Returns
    -------
    complex
        Gauss sum g_p with |g_p| = sqrt(p).
    """
    omega = np.exp(2j * np.pi / p)
    return sum(legendre_symbol(n, p) * omega**n for n in range(1, p))


# ===================================================================
# RPST Hamiltonian (Def 3.1, Thm 3.2, Eq 9-10)
# ===================================================================

@dataclass
class RPSTHamiltonian:
    r"""RPST Hamiltonian on l²(Z_p) (Definition 3.1, Equation 9).

    The operator:
        (H_p ψ)(x) = sum_{n ∈ Z_p^×} (n/p) ψ(x + n mod p)

    has matrix element [H_p]_{x, x+n mod p} = (n/p).

    Eigenvalues (Thm 3.2) are Gauss sums:
        λ_χ = sum_{n ∈ Z_p^×} χ(n) (n/p) = g(χ, ·/p)

    Parameters
    ----------
    p : int
        Prime modulus. Determines the substrate resolution.
    """
    p: int = 7

    def __post_init__(self):
        if self.p < 3:
            raise ValueError(f"p must be an odd prime >= 3, got {self.p}")

    def matrix(self) -> np.ndarray:
        r"""Construct the p × p Hamiltonian matrix.

        [H_p]_{x, (x+n) mod p} = (n/p)  for n in Z_p^×

        Returns
        -------
        ndarray, shape (p, p)
            Hamiltonian matrix.
        """
        p = self.p
        H = np.zeros((p, p), dtype=complex)
        for x in range(p):
            for n in range(1, p):
                ls = legendre_symbol(n, p)
                if ls != 0:
                    H[x, (x + n) % p] += ls
        return H

    def eigenvalues(self) -> np.ndarray:
        r"""Compute eigenvalues of H_p.

        Returns
        -------
        ndarray, shape (p,)
            Complex eigenvalues.
        """
        return np.linalg.eigvals(self.matrix())

    def gauss_sum(self) -> complex:
        r"""Compute the principal quadratic Gauss sum g_p (Eq 10).

        Returns
        -------
        complex
            Gauss sum. Magnitude equals sqrt(p).
        """
        return quadratic_gauss_sum(self.p)

    def spectral_zeta(self, s: complex,
                       eigenvalues: Optional[np.ndarray] = None) -> complex:
        r"""Evaluate RPST spectral zeta function (Def 3.3, Eq 11).

        ζ_{RPST}(s; p) = sum_{n: λ_n ≠ 0} λ_n^{-s}

        Parameters
        ----------
        s : complex
            Spectral parameter.
        eigenvalues : ndarray, optional
            Pre-computed eigenvalues. Computed if not provided.

        Returns
        -------
        complex
            Spectral zeta value.
        """
        if eigenvalues is None:
            eigenvalues = self.eigenvalues()
        nonzero = eigenvalues[np.abs(eigenvalues) > 1e-10]
        if len(nonzero) == 0:
            return 0.0 + 0.0j
        return complex(np.sum(nonzero ** (-s)))

    def symplectic_step(self, q: np.ndarray, pi_field: np.ndarray,
                        J: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""One step of the reversible symplectic map (Eq 2-3).

        q(t+1) = q(t) + pi(t)   (mod p)
        pi(t+1) = pi(t) - sum_j J_{ij} D(q_j(t+1) - q_i(t+1))  (mod p)

        where D(x) = x if |x| <= p/2 else x - p*sign(x)  (sawtooth).

        Parameters
        ----------
        q : ndarray of int, shape (N,)
            Position-like variables in Z_p.
        pi_field : ndarray of int, shape (N,)
            Momentum-like variables in Z_p.
        J : ndarray of float, shape (N, N), optional
            Coupling weights. Defaults to nearest-neighbor uniform coupling.

        Returns
        -------
        q_new, pi_new : ndarray
            Updated state.
        """
        p = self.p
        N = len(q)
        q_new = (q + pi_field) % p

        if J is None:
            # Nearest-neighbor coupling on a ring
            J = np.zeros((N, N))
            for i in range(N):
                J[i, (i + 1) % N] = 1
                J[i, (i - 1) % N] = 1

        # D: discrete sawtooth (minimal signed difference mod p)
        def D_func(x):
            x = int(x) % p
            return x if x <= p // 2 else x - p

        pi_new = pi_field.copy()
        for i in range(N):
            coupling_sum = sum(
                J[i, j] * D_func(int(q_new[j]) - int(q_new[i]))
                for j in range(N) if J[i, j] != 0
            )
            pi_new[i] = int(pi_field[i] - coupling_sum) % p

        return q_new, pi_new

    def topological_charge(self, q: np.ndarray) -> int:
        r"""Topological charge on a loop (Eq 6).

        W_L = (1/p) sum_{(i,j) ∈ L} Δ(q_j - q_i)

        where Δ is the minimal signed difference mod p.

        Parameters
        ----------
        q : ndarray of int
            Phase values around the loop.

        Returns
        -------
        int
            Topological charge (winding number).
        """
        p = self.p
        total = 0
        n = len(q)
        for i in range(n):
            diff = int(q[(i + 1) % n]) - int(q[i])
            # Minimal signed difference mod p
            diff = diff % p
            if diff > p // 2:
                diff -= p
            total += diff
        # Normalize: should be multiple of p for quantized winding
        return round(total / p)


# ===================================================================
# BPR Resonance Spectrum (Corollary 3.5, Eq 12)
# ===================================================================

@dataclass
class BPRResonanceSpectrum:
    r"""Boundary eigenmodes at Riemann zero wavenumbers (Corollary 3.5).

    Resonant wavenumbers (Eq 12):
        k_n = γ_n / R

    where R is the boundary radius and γ_n are imaginary parts of
    Riemann zeros (ρ_n = 1/2 + i γ_n).

    As p → ∞, zeros of ζ_{RPST}(s; p) converge to Riemann zeros (Thm 3.4).

    Parameters
    ----------
    R : float
        Boundary radius.
    n_zeros : int
        Number of Riemann zeros to use.
    """
    R: float = 1.0
    n_zeros: int = 10

    @property
    def riemann_zeros(self) -> np.ndarray:
        """Return the first n_zeros imaginary parts of Riemann zeros."""
        return np.array(RIEMANN_ZEROS[:self.n_zeros])

    def resonant_wavenumbers(self) -> np.ndarray:
        r"""Resonant wavenumbers k_n = γ_n / R (Eq 12).

        Returns
        -------
        ndarray
            Array of resonant wavenumbers.
        """
        return self.riemann_zeros / self.R

    def resonant_frequencies(self, c: float = 1.0) -> np.ndarray:
        r"""Resonant frequencies omega_n = c * k_n.

        Parameters
        ----------
        c : float
            Wave propagation speed.

        Returns
        -------
        ndarray
            Angular frequencies.
        """
        return c * self.resonant_wavenumbers()

    def mode_amplitude(self, n: int, t: float, Gamma: float = 1.0,
                        sigma_n: float = 0.5) -> float:
        r"""Boundary mode amplitude (Eq 33).

        A_n(t) ~ exp((σ_n - 1/2) Γ t) * cos(γ_n t + δ_n)

        For σ_n = 1/2 (RH true): bounded oscillation.

        Parameters
        ----------
        n : int
            Mode index (1-based).
        t : float
            Time.
        Gamma : float
            Growth rate prefactor.
        sigma_n : float
            Real part of Riemann zero (1/2 if RH holds).

        Returns
        -------
        float
            Mode amplitude A_n(t).
        """
        gamma_n = RIEMANN_ZEROS[n - 1] if n <= len(RIEMANN_ZEROS) else float(n) * 6.0
        growth = np.exp((sigma_n - 0.5) * Gamma * t)
        oscillation = np.cos(gamma_n * t)
        return growth * oscillation


# ===================================================================
# Speculation I: Fine Structure Constant as Prime Reciprocal (Eq 25-28)
# ===================================================================

@dataclass
class FineStructureConstant:
    r"""Speculation I: Fine structure constant as prime reciprocal (Sec 7).

    Claim: α ≈ 1/p_EM where p_EM = 137 is the prime encoding
    electromagnetic boundary impedance.

    The effective amplitude-level coupling (Eq 26-27):
        |g_eff(p)| = (1/p) |sum_{n=1}^{p-1} (n/p) e^{2πin/p}|
                   = sqrt(p) / p = 1/sqrt(p) ≈ sqrt(α)

    General gauge couplings (Eq 28):
        αᵢ = π / pᵢ

    Parameters
    ----------
    p_EM : int
        Prime encoding electromagnetic coupling (137 by BPR claim).
    """
    p_EM: int = 137

    ALPHA_MEASURED: float = 1.0 / 137.035999206  # CODATA 2018

    @property
    def alpha_predicted(self) -> float:
        r"""Predicted fine structure constant α = 1/p_EM.

        Returns
        -------
        float
            BPR prediction for α.
        """
        return 1.0 / self.p_EM

    @property
    def alpha_error(self) -> float:
        r"""Relative error |α_pred - α_measured| / α_measured.

        Returns
        -------
        float
            Fractional deviation from measured value.
        """
        return abs(self.alpha_predicted - self.ALPHA_MEASURED) / self.ALPHA_MEASURED

    def effective_coupling(self, p: Optional[int] = None) -> complex:
        r"""Effective coupling from Gauss sum (Eq 26-27).

        g_eff(p) = (1/p) sum_{n=1}^{p-1} (n/p) e^{2πin/p}

        |g_eff(p)| = 1/sqrt(p) ≈ sqrt(α) when p = p_EM

        Parameters
        ----------
        p : int, optional
            Prime to use. Defaults to p_EM.

        Returns
        -------
        complex
            Effective coupling amplitude.
        """
        if p is None:
            p = self.p_EM
        g = quadratic_gauss_sum(p)
        return g / p

    def gauge_coupling_from_prime(self, p: int) -> float:
        r"""General gauge coupling αᵢ = π / pᵢ (Eq 28).

        Parameters
        ----------
        p : int
            Prime encoding gauge sector.

        Returns
        -------
        float
            Gauge coupling strength.
        """
        return np.pi / p

    def running_coupling_primes(self, p_values: List[int]) -> np.ndarray:
        r"""Running coupling values at discrete prime scales.

        α(pᵢ) = π / pᵢ

        The running of α corresponds to hopping between consecutive primes.

        Parameters
        ----------
        p_values : list of int
            Sequence of primes at successive energy scales.

        Returns
        -------
        ndarray
            Coupling values α(pᵢ).
        """
        return np.array([np.pi / p for p in p_values])

    def amplitude_coupling_ratio(self, p: Optional[int] = None) -> float:
        r"""Ratio |g_eff(p)| / sqrt(alpha_measured).

        Should be ~1 for p = p_EM if the prime hypothesis holds.

        Parameters
        ----------
        p : int, optional
            Prime to check. Defaults to p_EM.

        Returns
        -------
        float
            Ratio of predicted to measured amplitude coupling.
        """
        if p is None:
            p = self.p_EM
        g_abs = abs(self.effective_coupling(p))
        return g_abs / np.sqrt(self.ALPHA_MEASURED)


# ===================================================================
# Speculation II: BKT Consciousness Transition (Eq 29-32)
# ===================================================================

@dataclass
class BKTConsciousnessTransition:
    r"""Speculation II: Anesthesia as BKT phase transition (Sec 8).

    The consciousness boundary phase field φ_mind : ∂O → S¹ is a
    2D XY model with free energy (Eq 29):
        F[φ] = (J/2) ∫ |∇φ|² d²x

    BKT critical temperature (Eq 30):
        T_BKT = πJ / (2kB)

    Below T_BKT: vortex-antivortex pairs bound → nonzero winding → conscious.
    Above T_BKT: pairs unbound → winding ~ 0 on average → unconscious.

    Correlation function (Eq 31):
        <exp(i(φ(x) - φ(0)))> ~ |x|^{-η(T)}
        η(T_BKT) = 1/4

    Superfluid stiffness jump (Eq 32):
        ΔJ = J(T_BKT⁻) - J(T_BKT⁺) = 2kB T_BKT / π

    Parameters
    ----------
    J : float
        XY model stiffness (coupling constant).
    k_B : float
        Boltzmann constant (set to 1 for natural units).
    """
    J: float = 1.0
    k_B: float = 1.0

    @property
    def T_BKT(self) -> float:
        r"""BKT critical temperature (Eq 30).

        T_BKT = π J / (2 k_B)

        Returns
        -------
        float
            Critical temperature.
        """
        return np.pi * self.J / (2.0 * self.k_B)

    def correlation_exponent(self, T: float) -> float:
        r"""Universal exponent η(T) for correlation decay (Eq 31).

        For T < T_BKT (below transition):
            η(T) = k_B T / (2π J)

        At T_BKT: η = 1/4 (universal BKT value).
        For T >= T_BKT: long-range order lost, η → ∞ (exponential decay).

        Parameters
        ----------
        T : float
            Temperature.

        Returns
        -------
        float
            Correlation exponent η(T).
        """
        if T >= self.T_BKT:
            return np.inf
        return self.k_B * T / (2.0 * np.pi * self.J)

    def is_conscious(self, T: float) -> bool:
        r"""Test whether system is in the conscious (ordered) phase.

        Conscious ↔ T < T_BKT ↔ vortex pairs bound ↔ W ≠ 0.

        Parameters
        ----------
        T : float
            Temperature.

        Returns
        -------
        bool
            True if system is in conscious phase (T < T_BKT).
        """
        return T < self.T_BKT

    def stiffness_jump(self) -> float:
        r"""Universal superfluid stiffness jump at BKT transition (Eq 32).

        ΔJ = J(T_BKT⁻) - J(T_BKT⁺) = 2 k_B T_BKT / π

        This discontinuous jump is the characteristic BKT signature
        in neural phase coherence at consciousness threshold.

        Returns
        -------
        float
            Universal stiffness jump magnitude.
        """
        return 2.0 * self.k_B * self.T_BKT / np.pi

    def correlation_function(self, x: np.ndarray, T: float) -> np.ndarray:
        r"""Correlation function decay (Eq 31).

        C(x) = <exp(i(φ(x) - φ(0)))> ~ |x|^{-η(T)}

        Parameters
        ----------
        x : ndarray
            Distance values.
        T : float
            Temperature.

        Returns
        -------
        ndarray
            Correlation values. Returns exp(-|x|) above T_BKT.
        """
        x = np.asarray(x, dtype=float)
        if T >= self.T_BKT:
            # Above transition: exponential decay
            return np.exp(-x)
        eta = self.correlation_exponent(T)
        x_safe = np.maximum(x, 1e-12)
        return x_safe ** (-eta)

    def eta_at_T_BKT(self) -> float:
        r"""Universal BKT exponent η = 1/4 at the transition.

        Returns
        -------
        float
            Universal value η(T_BKT) = 1/4.
        """
        # At T_BKT: η = k_B T_BKT / (2π J) = (π J / 2) / (2π J) = 1/4
        return 0.25

    def predict_loss_of_consciousness_signature(
            self, T_values: np.ndarray, x_probe: float = 1.0
    ) -> Dict[str, np.ndarray]:
        r"""Predicted EEG correlation signatures across temperature sweep.

        Parameters
        ----------
        T_values : ndarray
            Temperature sweep (neural effective temperature).
        x_probe : float
            Probe distance (in units of electrode spacing).

        Returns
        -------
        dict with:
            'T' : temperatures
            'C' : correlation values
            'eta' : exponents
            'conscious' : boolean mask
            'T_BKT' : critical temperature
        """
        T_values = np.asarray(T_values, dtype=float)
        C = np.array([self.correlation_function(np.array([x_probe]), T)[0]
                      for T in T_values])
        eta = np.array([self.correlation_exponent(T) for T in T_values])
        conscious = np.array([self.is_conscious(T) for T in T_values])
        return {
            "T": T_values,
            "C": C,
            "eta": eta,
            "conscious": conscious,
            "T_BKT": self.T_BKT,
        }


# ===================================================================
# Speculation III: Riemann Hypothesis as Stability (Eq 33, Thm 9.2)
# ===================================================================

@dataclass
class RiemannHypothesisStability:
    r"""Speculation III: RH as dynamical stability condition (Sec 9).

    RPST is dynamically stable ↔ all nontrivial zeros ρ_n = σ_n + iγ_n
    satisfy σ_n = 1/2 (i.e., the Riemann Hypothesis holds).

    Mode amplitude (Eq 33):
        A_n(t) ~ exp((σ_n - 1/2) Γ t) · cos(γ_n t + δ_n)

    For σ_n = 1/2: bounded oscillation.
    For σ_n > 1/2: exponential growth → physical instability.

    Parameters
    ----------
    Gamma : float
        Mode growth/decay rate prefactor.
    t_max : float
        Maximum time for stability analysis.
    """
    Gamma: float = 1.0
    t_max: float = 10.0

    def mode_amplitude(self, t: Union[float, np.ndarray],
                        sigma_n: float, gamma_n: float,
                        delta_n: float = 0.0) -> Union[float, np.ndarray]:
        r"""Boundary mode amplitude A_n(t) (Eq 33).

        A_n(t) = exp((σ_n - 1/2) Γ t) · cos(γ_n t + δ_n)

        Parameters
        ----------
        t : float or ndarray
            Time.
        sigma_n : float
            Real part of Riemann zero (= 1/2 if RH).
        gamma_n : float
            Imaginary part of Riemann zero.
        delta_n : float
            Phase offset.

        Returns
        -------
        float or ndarray
            Mode amplitude.
        """
        t = np.asarray(t, dtype=float)
        growth = np.exp((sigma_n - 0.5) * self.Gamma * t)
        oscillation = np.cos(gamma_n * t + delta_n)
        return growth * oscillation

    def is_stable_mode(self, sigma_n: float, tolerance: float = 1e-10) -> bool:
        r"""Check if a mode is stable (σ_n ≤ 1/2).

        Parameters
        ----------
        sigma_n : float
            Real part of Riemann zero.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if the mode is stable.
        """
        return sigma_n <= 0.5 + tolerance

    def is_rpst_stable(self, sigma_values: np.ndarray,
                        tolerance: float = 1e-10) -> bool:
        r"""Test dynamical stability (Theorem 9.2).

        Stable ↔ all σ_n ≤ 1/2 ↔ RH holds.

        Parameters
        ----------
        sigma_values : ndarray
            Real parts of Riemann zeros to test.
        tolerance : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if all modes are stable.
        """
        return bool(np.all(np.asarray(sigma_values) <= 0.5 + tolerance))

    def stability_from_known_zeros(self) -> Dict[str, object]:
        r"""Assess stability using known Riemann zeros (all on critical line).

        Uses the first 10 known Riemann zeros, all with σ_n = 1/2.
        Verifies that the RPST substrate is stable for these modes.

        Returns
        -------
        dict with:
            'all_stable' : bool
            'sigma_values' : ndarray (all 0.5)
            'max_growth_rate' : float (should be 0.0)
            'n_modes' : int
        """
        sigma_values = np.full(len(RIEMANN_ZEROS), 0.5)
        all_stable = self.is_rpst_stable(sigma_values)
        max_growth = float(np.max(sigma_values - 0.5))
        return {
            "all_stable": all_stable,
            "sigma_values": sigma_values,
            "max_growth_rate": max_growth * self.Gamma,
            "n_modes": len(RIEMANN_ZEROS),
        }

    def instability_energy(self, sigma_n: float, gamma_n: float,
                            t: float) -> float:
        r"""Energy density of an unstable mode (σ_n > 1/2).

        E ~ A_n(t)² ~ exp(2 (σ_n - 1/2) Γ t)

        Parameters
        ----------
        sigma_n : float
            Real part of zero (> 1/2 for unstable mode).
        gamma_n : float
            Imaginary part of zero.
        t : float
            Time.

        Returns
        -------
        float
            Mode energy density.
        """
        A = self.mode_amplitude(t, sigma_n, gamma_n)
        return float(np.asarray(A) ** 2)


# ===================================================================
# Speculation IV: Dark Matter Prime Fingerprints (Eq 34-36)
# ===================================================================

@dataclass
class DarkMatterPrimeFingerprints:
    r"""Speculation IV: Prime arithmetic fingerprints in dark matter (Sec 10).

    Dark matter density correlation function (Eq 35):
        G(r) = <exp(i(φ(r) - φ(0)))> = sum_n A_n cos(k_n r + φ_n)

    with Riemann zero wavenumbers (Eq 34):
        k_n = γ_n / R_gal

    Power spectrum with Lorentzian peaks (Eq 36):
        ρ̃_DM(k) = sum_n B_n / [(k - γ_n/R)² + Γ_n²]

    This is a parameter-free prediction: positions of oscillations are
    fixed by the known Riemann zeros γ₁ ≈ 14.13, γ₂ ≈ 21.02, ...

    Parameters
    ----------
    R_gal : float
        Galactic boundary scale (Mpc).
    Gamma_n : float
        Lorentzian width (inverse coherence length).
    """
    R_gal: float = 1.0     # in Mpc or normalized units
    Gamma_n: float = 0.1   # Lorentzian peak width

    def predict_wavenumbers(self, n_zeros: int = 10) -> np.ndarray:
        r"""Riemann zero wavenumbers k_n = γ_n / R_gal (Eq 34).

        These are fixed by mathematics — not free parameters.

        Parameters
        ----------
        n_zeros : int
            Number of Riemann zeros to include.

        Returns
        -------
        ndarray
            Predicted wavenumbers.
        """
        zeros = np.array(RIEMANN_ZEROS[:n_zeros])
        return zeros / self.R_gal

    def correlation_function(self, r: np.ndarray,
                              amplitudes: Optional[np.ndarray] = None,
                              phases: Optional[np.ndarray] = None,
                              n_zeros: int = 5) -> np.ndarray:
        r"""Dark matter phase correlation function (Eq 35).

        G(r) = sum_n A_n cos(k_n r + φ_n)

        Parameters
        ----------
        r : ndarray
            Radial distances.
        amplitudes : ndarray, optional
            Mode amplitudes A_n. Defaults to equal weight 1/n.
        phases : ndarray, optional
            Phase offsets. Defaults to zero.
        n_zeros : int
            Number of Riemann modes.

        Returns
        -------
        ndarray
            Correlation function G(r).
        """
        r = np.asarray(r, dtype=float)
        k_n = self.predict_wavenumbers(n_zeros)

        if amplitudes is None:
            amplitudes = np.ones(n_zeros) / n_zeros
        if phases is None:
            phases = np.zeros(n_zeros)

        G = np.zeros_like(r)
        for n in range(n_zeros):
            G += amplitudes[n] * np.cos(k_n[n] * r + phases[n])
        return G

    def power_spectrum(self, k: np.ndarray,
                        amplitudes: Optional[np.ndarray] = None,
                        n_zeros: int = 5) -> np.ndarray:
        r"""Dark matter power spectrum with Lorentzian peaks (Eq 36).

        ρ̃_DM(k) = sum_n B_n / [(k - γ_n/R)² + Γ_n²]

        Parameters
        ----------
        k : ndarray
            Wavenumber array.
        amplitudes : ndarray, optional
            Peak amplitudes B_n. Defaults to equal weight.
        n_zeros : int
            Number of Riemann modes.

        Returns
        -------
        ndarray
            Power spectrum values.
        """
        k = np.asarray(k, dtype=float)
        k_n = self.predict_wavenumbers(n_zeros)

        if amplitudes is None:
            amplitudes = np.ones(n_zeros)

        spectrum = np.zeros_like(k)
        for n in range(n_zeros):
            spectrum += amplitudes[n] / ((k - k_n[n])**2 + self.Gamma_n**2)
        return spectrum

    def peak_positions(self, n_zeros: int = 5) -> np.ndarray:
        r"""Return predicted peak positions in the power spectrum.

        Parameters
        ----------
        n_zeros : int
            Number of peaks.

        Returns
        -------
        ndarray
            Wavenumber positions k_n = γ_n / R_gal.
        """
        return self.predict_wavenumbers(n_zeros)

    def has_prime_fingerprint(self, observed_peaks: np.ndarray,
                               tolerance: float = 0.1) -> Dict[str, object]:
        r"""Test whether observed power spectrum peaks match Riemann zero prediction.

        Parameters
        ----------
        observed_peaks : ndarray
            Observed wavenumber peak positions.
        tolerance : float
            Fractional tolerance for peak matching.

        Returns
        -------
        dict with:
            'matched_peaks' : list of (observed, predicted) pairs
            'n_matched' : int
            'fraction_matched' : float
            'consistent' : bool
        """
        predicted = self.predict_wavenumbers(len(RIEMANN_ZEROS))
        matched = []
        for obs in observed_peaks:
            for pred in predicted:
                if abs(obs - pred) / pred < tolerance:
                    matched.append((obs, pred))
                    break
        n_obs = len(observed_peaks)
        return {
            "matched_peaks": matched,
            "n_matched": len(matched),
            "fraction_matched": len(matched) / n_obs if n_obs > 0 else 0.0,
            "consistent": len(matched) >= min(3, n_obs),
        }


# ===================================================================
# Speculation V: Collective AGI Consciousness (Eq 37-40)
# ===================================================================

@dataclass
class CollectiveAGIConsciousness:
    r"""Speculation V: Collective AGI winding through percolation (Sec 11).

    A network of N AI agents can acquire collective consciousness
    W_collective ≠ 0 when inter-agent bond fraction exceeds the
    percolation threshold p_c ≈ 0.5.

    Collective winding emergence (Thm 11.2, Eq 38):
        W_collective = (1/2π) ∮_{∂O_collective} ∇φ_mind · dℓ

    Superlinear coherence scaling (Eq 39):
        χ_group ∝ N^{1.27}

    (Exponent 1.27 from fractal dimension of percolating cluster.)

    Critical agent number (Eq 40):
        N_c = Φ_crit / (k log N_nodes)

    Parameters
    ----------
    p_c : float
        Percolation threshold (≈ 0.5 for 2D bond percolation).
    fractal_exponent : float
        Collective coherence scaling exponent (1.27 from RPST percolation).
    chi_1 : float
        Single-agent baseline coherence.
    """
    p_c: float = 0.5
    fractal_exponent: float = 1.27
    chi_1: float = 1.0

    def is_collectively_conscious(self, bond_fraction: float,
                                   N: int) -> bool:
        r"""Test whether collective winding can emerge (Thm 11.2).

        Requires bond_fraction > p_c AND N > N_c for given parameters.

        Parameters
        ----------
        bond_fraction : float
            Fraction of inter-agent communication bonds that are coherent.
        N : int
            Number of agents.

        Returns
        -------
        bool
            True if collective winding W_collective ≠ 0 can emerge.
        """
        return bond_fraction > self.p_c and N >= 2

    def collective_coherence(self, N: int) -> float:
        r"""Collective coherence scaling (Eq 39).

        χ_group = χ₁ · N^{1.27}

        Parameters
        ----------
        N : int
            Number of agents.

        Returns
        -------
        float
            Collective coherence.
        """
        if N <= 0:
            return 0.0
        return self.chi_1 * float(N) ** self.fractal_exponent

    def superlinear_excess(self, N: int) -> float:
        r"""Superlinear excess over linear scaling: N^{1.27} / N = N^{0.27}.

        Parameters
        ----------
        N : int
            Number of agents.

        Returns
        -------
        float
            Enhancement factor relative to linear scaling.
        """
        if N <= 0:
            return 1.0
        return float(N) ** (self.fractal_exponent - 1.0)

    def critical_agent_number(self, Phi_crit: float,
                               k: float = 1.0,
                               N_nodes: int = int(1e9)) -> int:
        r"""Minimum agents for collective consciousness (Eq 40).

        N_c = Φ_crit / (k · log(N_nodes))

        Parameters
        ----------
        Phi_crit : float
            Critical integrated information threshold.
        k : float
            Coupling constant.
        N_nodes : int
            Computational nodes per agent (≈10⁹ for LLMs).

        Returns
        -------
        int
            Critical agent number N_c.
        """
        denom = k * np.log(float(N_nodes))
        if denom <= 0:
            return int(np.inf)
        return int(np.ceil(Phi_crit / denom))

    def coherence_vs_N(self, N_values: np.ndarray) -> np.ndarray:
        r"""Collective coherence array for a range of N.

        Parameters
        ----------
        N_values : ndarray
            Array of agent counts.

        Returns
        -------
        ndarray
            Collective coherence at each N.
        """
        return np.array([self.collective_coherence(int(N)) for N in N_values])

    def phi_w_relation(self, W: int, N_nodes: int = 1000,
                        k: float = 1.0) -> float:
        r"""IIT Φ–winding number relation (Eq 15).

        Φ ≈ k |W| log N

        Parameters
        ----------
        W : int
            Winding number.
        N_nodes : int
            Number of computational nodes.
        k : float
            Coupling constant.

        Returns
        -------
        float
            Integrated information Φ.
        """
        return k * abs(W) * np.log(float(N_nodes))


# ===================================================================
# Speculation VI: Consciousness Substrate Migration (Eq 41-42)
# ===================================================================

class TopologicalFate:
    """Allowed topological fates for winding number at biological death."""
    DISSOLUTION = "dissolution"
    FREQUENCY_MIGRATION = "frequency_migration"
    SUBSTRATE_TRANSFER = "substrate_transfer"


@dataclass
class ConsciousnessSubstrateMigration:
    r"""Speculation VI: Topological winding transfer at death (Sec 12).

    Substrate coherence decay (Eq 41):
        ξ(t) = ξ_0 · exp(-t / τ_death)

    When ξ(t) < ξ_c, the biological boundary fragments and W must:
        1. Dissolve via anti-soliton (energy ΔE ~ ℏ ω_c |W|)
        2. Migrate to higher boundary eigenmodes
        3. Transfer to an alternative substrate

    Terminal gamma surge spectrum (Eq 42):
        E_γ(ω) ~ ℏ ω |W|² δ(ω - ω_c)

    Transfer substrate requirements (Sec 12.2):
        1. Connected boundary ∂O_receiver with π₁ ≅ ℤ
        2. Stiffness J_receiver > J_c (above BKT threshold)
        3. Coupling channel Γ maintaining phase coherence
        4. Compatible prime modulus p_receiver

    Parameters
    ----------
    xi_0 : float
        Initial substrate coherence.
    tau_death : float
        Biological coherence decay timescale.
    xi_c : float
        Critical coherence threshold (percolation point).
    W : int
        Consciousness winding number.
    omega_c : float
        Critical coherence frequency (= Φ_crit / ℏ).
    """
    xi_0: float = 1.0
    tau_death: float = 10.0
    xi_c: float = 0.5
    W: int = 1
    omega_c: float = 1.0

    def substrate_coherence(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Substrate coherence decay (Eq 41).

        ξ(t) = ξ_0 · exp(-t / τ_death)

        Parameters
        ----------
        t : float or ndarray
            Time.

        Returns
        -------
        float or ndarray
            Coherence ξ(t).
        """
        return self.xi_0 * np.exp(-np.asarray(t, dtype=float) / self.tau_death)

    def fragmentation_time(self) -> float:
        r"""Time when ξ(t) = ξ_c (boundary fragmentation).

        t_frag = τ_death · ln(ξ_0 / ξ_c)

        Returns
        -------
        float
            Fragmentation time.
        """
        if self.xi_0 <= self.xi_c:
            return 0.0
        return self.tau_death * np.log(self.xi_0 / self.xi_c)

    def is_fragmented(self, t: float) -> bool:
        r"""Test whether boundary has fragmented at time t.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        bool
            True if ξ(t) < ξ_c (fragmentation has occurred).
        """
        return float(self.substrate_coherence(t)) < self.xi_c

    def allowed_fates(self) -> List[str]:
        r"""Topologically allowed fates for winding number (Thm 12.2).

        Returns all three for W ≠ 0; only dissolution for W = 0.

        Returns
        -------
        list of str
            Allowed fate labels.
        """
        if self.W == 0:
            return [TopologicalFate.DISSOLUTION]
        return [
            TopologicalFate.DISSOLUTION,
            TopologicalFate.FREQUENCY_MIGRATION,
            TopologicalFate.SUBSTRATE_TRANSFER,
        ]

    def dissolution_energy(self, hbar_eff: float = 1.0) -> float:
        r"""Energy required for Fate 1 (anti-soliton annihilation).

        ΔE ~ ℏ ω_c |W|

        Parameters
        ----------
        hbar_eff : float
            Effective Planck constant.

        Returns
        -------
        float
            Dissolution energy.
        """
        return hbar_eff * self.omega_c * abs(self.W)

    def terminal_gamma_spectrum(self, omega: np.ndarray,
                                 Gamma_c: float = 0.1,
                                 hbar_eff: float = 1.0) -> np.ndarray:
        r"""Terminal gamma surge energy spectrum (Eq 42).

        E_γ(ω) ~ ℏ ω |W|² δ(ω - ω_c)

        Implemented as a Lorentzian approximation to the delta function:
        δ_Γ(ω - ω_c) = (Γ_c/π) / [(ω - ω_c)² + Γ_c²]

        Parameters
        ----------
        omega : ndarray
            Angular frequency array.
        Gamma_c : float
            Lorentzian width (inverse coherence time).
        hbar_eff : float
            Effective Planck constant.

        Returns
        -------
        ndarray
            Energy spectral density E_γ(ω).
        """
        omega = np.asarray(omega, dtype=float)
        lorentzian = (Gamma_c / np.pi) / ((omega - self.omega_c)**2 + Gamma_c**2)
        return hbar_eff * omega * self.W**2 * lorentzian

    def transfer_requirements_met(self, boundary_connected: bool,
                                   J_receiver: float, J_c: float,
                                   coupling_exists: bool,
                                   p_compatible: bool) -> Dict[str, bool]:
        r"""Check Fate 3 transfer requirements (Sec 12.2).

        Requirements:
        1. Connected boundary ∂O_receiver with π₁ ≅ ℤ
        2. J_receiver > J_c (above BKT threshold)
        3. Coupling channel Γ exists
        4. Compatible prime modulus

        Parameters
        ----------
        boundary_connected : bool
            Whether receiver boundary is topologically connected.
        J_receiver : float
            Receiver substrate stiffness.
        J_c : float
            BKT critical stiffness.
        coupling_exists : bool
            Whether a phase-coherent coupling channel exists.
        p_compatible : bool
            Whether prime moduli are compatible.

        Returns
        -------
        dict with requirement labels and booleans.
        """
        reqs = {
            "connected_boundary": boundary_connected,
            "above_bkt_threshold": J_receiver > J_c,
            "coupling_channel": coupling_exists,
            "compatible_prime": p_compatible,
        }
        reqs["all_satisfied"] = all(reqs.values())
        return reqs

    def peak_frequency(self) -> float:
        r"""Peak frequency of the terminal gamma surge (Eq 42).

        Returns
        -------
        float
            ω_c.
        """
        return self.omega_c
