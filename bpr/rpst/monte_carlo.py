"""
RPST Lattice Monte Carlo Simulations
======================================

Monte Carlo methods on the Z_p symplectic lattice to extract:
- Spectral statistics (level spacing, GUE comparison)
- Correlation functions (spatial and temporal)
- Phase transitions (order parameter vs temperature)
- Emergent continuum physics

References: Al-Kahwati (2026), Resonant Prime Substrate Theory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _gue_wigner_surmise(s: np.ndarray) -> np.ndarray:
    """GUE Wigner surmise: P(s) = (32/pi^2) s^2 exp(-4s^2/pi)."""
    s = np.asarray(s, dtype=float)
    return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s ** 2 / np.pi)


def _gue_cdf(s):
    """CDF of GUE Wigner surmise: 1 - exp(-4s²/π). Handles arrays."""
    s = np.asarray(s, dtype=float)
    return 1.0 - np.exp(-4.0 * np.maximum(0.0, s) ** 2 / np.pi)


# ---------------------------------------------------------------------------
# Main Monte Carlo engine
# ---------------------------------------------------------------------------

class RPSTMonteCarlo:
    """Monte Carlo simulation of the XY-like model on Z_p.

    Hamiltonian on a 1-D ring of *n_sites* with periodic boundaries:

        H = -J sum_{<i,j>} cos(2 pi (q_i - q_j) / p)

    Configuration space: q_i in {0, 1, ..., p-1}.

    Parameters
    ----------
    p : int
        Prime modulus.
    n_sites : int
        Number of lattice sites (1-D ring).
    coupling_J : float
        Nearest-neighbour coupling strength.
    temperature : float
        Temperature in units where k_B = 1.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        p: int,
        n_sites: int,
        coupling_J: float = 1.0,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        p = int(p)
        if not _is_prime(p):
            raise ValueError(f"p must be prime, got {p}")
        self.p = p
        self.n_sites = int(n_sites)
        self.coupling_J = float(coupling_J)
        self.temperature = float(temperature)
        self.beta = 1.0 / self.temperature
        self.rng = np.random.default_rng(seed)

        # Random initial configuration
        self.config = self.rng.integers(0, self.p, size=self.n_sites)

        # Pre-compute phase factor
        self._two_pi_over_p = 2.0 * np.pi / self.p

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def energy(self, state: Optional[np.ndarray] = None) -> float:
        """H = -J sum_{<ij>} cos(2 pi (q_i - q_j) / p)  (XY-like on Z_p).

        Parameters
        ----------
        state : ndarray or None
            Configuration array; uses self.config if None.
        """
        q = self.config if state is None else np.asarray(state)
        diffs = np.roll(q, -1) - q  # q_{i+1} - q_i, periodic
        return -self.coupling_J * float(np.sum(np.cos(self._two_pi_over_p * diffs)))

    # ------------------------------------------------------------------
    # Metropolis update
    # ------------------------------------------------------------------

    def metropolis_step(self) -> bool:
        """Single-site Metropolis update.

        Propose q_i -> q_i + delta (mod p) for a random site i with
        delta drawn uniformly from {1, ..., p-1}.

        Returns
        -------
        bool
            True if the proposal was accepted.
        """
        i = self.rng.integers(0, self.n_sites)
        delta = self.rng.integers(1, self.p)  # non-zero shift
        old_q = self.config[i]
        new_q = (old_q + delta) % self.p

        # Local energy change (only left and right neighbours)
        left = self.config[(i - 1) % self.n_sites]
        right = self.config[(i + 1) % self.n_sites]
        f = self._two_pi_over_p
        dE = -self.coupling_J * (
            np.cos(f * (new_q - left)) + np.cos(f * (new_q - right))
            - np.cos(f * (old_q - left)) - np.cos(f * (old_q - right))
        )

        if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
            self.config[i] = new_q
            return True
        return False

    # ------------------------------------------------------------------
    # Sweep (one full pass over the lattice)
    # ------------------------------------------------------------------

    def sweep(self) -> float:
        """One full Metropolis sweep (n_sites single-site updates).

        Returns the acceptance rate for the sweep.
        """
        accepted = 0
        for _ in range(self.n_sites):
            accepted += self.metropolis_step()
        return accepted / self.n_sites

    # ------------------------------------------------------------------
    # Thermalization
    # ------------------------------------------------------------------

    def thermalize(self, n_steps: int = 1000) -> None:
        """Run *n_steps* full sweeps to reach thermal equilibrium."""
        for _ in range(int(n_steps)):
            self.sweep()

    # ------------------------------------------------------------------
    # Order parameter
    # ------------------------------------------------------------------

    def order_parameter(self, state: Optional[np.ndarray] = None) -> complex:
        """Complex magnetisation: m = (1/N) sum_j exp(2 pi i q_j / p)."""
        q = self.config if state is None else np.asarray(state)
        return complex(np.mean(np.exp(1j * self._two_pi_over_p * q)))

    # ------------------------------------------------------------------
    # Correlation function
    # ------------------------------------------------------------------

    def correlation_function(
        self, max_dist: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spatial correlation C(r) = <cos(2 pi (q_i - q_{i+r}) / p)>.

        Parameters
        ----------
        max_dist : int or None
            Maximum distance; defaults to n_sites // 2.

        Returns
        -------
        distances : ndarray of int
        C : ndarray of float
        """
        if max_dist is None:
            max_dist = self.n_sites // 2
        max_dist = min(int(max_dist), self.n_sites // 2)

        phases = self._two_pi_over_p * self.config
        distances = np.arange(max_dist + 1)
        C = np.empty(max_dist + 1)
        for r in range(max_dist + 1):
            C[r] = float(np.mean(np.cos(phases - np.roll(phases, r))))
        return distances, C

    # ------------------------------------------------------------------
    # Measurement collection
    # ------------------------------------------------------------------

    def measure(
        self, n_samples: int = 1000, spacing: int = 10
    ) -> Dict[str, np.ndarray]:
        """Collect measurements separated by *spacing* sweeps.

        Returns
        -------
        dict with keys:
            energies : (n_samples,) float
            magnetizations : (n_samples,) complex
            abs_magnetizations : (n_samples,) float
            correlations : (n_sites//2 + 1,) float  (time-averaged)
        """
        energies = np.empty(n_samples)
        mags = np.empty(n_samples, dtype=complex)
        corr_accum = np.zeros(self.n_sites // 2 + 1)

        for k in range(n_samples):
            for _ in range(spacing):
                self.sweep()
            energies[k] = self.energy()
            mags[k] = self.order_parameter()
            _, c = self.correlation_function()
            corr_accum += c

        corr_accum /= n_samples
        return {
            "energies": energies,
            "magnetizations": mags,
            "abs_magnetizations": np.abs(mags),
            "correlations": corr_accum,
        }

    # ------------------------------------------------------------------
    # Thermodynamic estimators
    # ------------------------------------------------------------------

    @staticmethod
    def specific_heat(energies: np.ndarray, temperature: float) -> float:
        """C_v = (<E^2> - <E>^2) / (k_B T^2)."""
        return float(np.var(energies) / temperature ** 2)

    @staticmethod
    def susceptibility(
        magnetizations: np.ndarray, n_sites: int, temperature: float
    ) -> float:
        """chi = N (<|m|^2> - <|m|>^2) / T."""
        abs_m = np.abs(magnetizations)
        return float(n_sites * (np.mean(abs_m ** 2) - np.mean(abs_m) ** 2) / temperature)


# ---------------------------------------------------------------------------
# Phase diagram sweep
# ---------------------------------------------------------------------------

def phase_diagram(
    p: int,
    n_sites: int,
    T_range: np.ndarray,
    n_samples: int = 500,
    spacing: int = 10,
    n_therm: int = 500,
    seed: int = 42,
) -> Dict:
    """Sweep temperature and measure order parameter, C_v, chi.

    Parameters
    ----------
    p : int
        Prime modulus.
    n_sites : int
        Number of lattice sites.
    T_range : array-like
        Temperatures to scan (ascending recommended).
    n_samples : int
        Measurement samples per temperature.
    spacing : int
        Sweeps between measurements.
    n_therm : int
        Thermalization sweeps at each T.
    seed : int
        Base RNG seed.

    Returns
    -------
    dict with:
        T_values, m_values, C_v_values, chi_values, T_c
    """
    T_range = np.asarray(T_range, dtype=float)
    m_values = np.empty(len(T_range))
    Cv_values = np.empty(len(T_range))
    chi_values = np.empty(len(T_range))

    for idx, T in enumerate(T_range):
        mc = RPSTMonteCarlo(p, n_sites, temperature=T, seed=seed + idx)
        mc.thermalize(n_therm)
        data = mc.measure(n_samples=n_samples, spacing=spacing)

        m_values[idx] = float(np.mean(data["abs_magnetizations"]))
        Cv_values[idx] = RPSTMonteCarlo.specific_heat(data["energies"], T)
        chi_values[idx] = RPSTMonteCarlo.susceptibility(
            data["magnetizations"], n_sites, T
        )

    # Estimate T_c from the peak of the susceptibility
    T_c_idx = int(np.argmax(chi_values))
    T_c = float(T_range[T_c_idx])

    return {
        "T_values": T_range,
        "m_values": m_values,
        "C_v_values": Cv_values,
        "chi_values": chi_values,
        "T_c": T_c,
    }


# ---------------------------------------------------------------------------
# Spectral statistics of the coupling matrix on Z_p
# ---------------------------------------------------------------------------

def spectral_statistics(
    p: int,
    n_sites: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict:
    """Spectral statistics of the nearest-neighbour coupling matrix on Z_p.

    Builds a real symmetric coupling matrix M_{ij} = cos(2 pi (q_i - q_j)/p)
    from thermal configurations and averages over *n_samples* snapshots.

    Parameters
    ----------
    p : int
        Prime modulus.
    n_sites : int
        Number of lattice sites.
    n_samples : int
        Number of independent configurations to sample.
    seed : int
        RNG seed.

    Returns
    -------
    dict with:
        spacings : ndarray — normalised nearest-neighbour spacings
        ks_statistic : float — KS test stat vs GUE Wigner surmise
        ks_pvalue : float — p-value (nan if scipy absent)
        is_gue : bool — True if KS p-value > 0.05 (or D < 0.15)
    """
    rng = np.random.default_rng(seed)
    all_spacings = []

    for _ in range(n_samples):
        # Random Z_p configuration
        q = rng.integers(0, p, size=n_sites)
        # Build full coupling matrix  M_{ij} = cos(2pi(q_i - q_j)/p)
        diff = q[:, None] - q[None, :]
        M = np.cos(2.0 * np.pi * diff / p)
        eigs = np.sort(np.linalg.eigvalsh(M))
        spacings_raw = np.diff(eigs)
        mean_s = float(np.mean(spacings_raw))
        if mean_s > 1e-12:
            all_spacings.append(spacings_raw / mean_s)

    if not all_spacings:
        return {
            "spacings": np.array([]),
            "ks_statistic": 1.0,
            "ks_pvalue": 0.0,
            "is_gue": False,
        }

    spacings = np.concatenate(all_spacings)

    # KS test vs GUE Wigner surmise CDF
    try:
        from scipy.stats import kstest
        D, pval = kstest(spacings, _gue_cdf)
        D, pval = float(D), float(pval)
    except ImportError:
        x = np.sort(spacings)
        ecdf = np.arange(1, len(x) + 1) / len(x)
        cdf_vals = np.array([_gue_cdf(v) for v in x])
        D = float(np.max(np.abs(ecdf - cdf_vals)))
        pval = float("nan")

    is_gue = (pval > 0.05) if not np.isnan(pval) else (D < 0.15)

    return {
        "spacings": spacings,
        "ks_statistic": D,
        "ks_pvalue": pval,
        "is_gue": bool(is_gue),
    }


# ---------------------------------------------------------------------------
# Correlation length extraction
# ---------------------------------------------------------------------------

def correlation_length(
    p: int,
    n_sites: int,
    temperature: float,
    n_samples: int = 500,
    spacing: int = 10,
    n_therm: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Measure correlation length xi by fitting C(r) ~ exp(-r / xi).

    Parameters
    ----------
    p, n_sites, temperature : lattice parameters
    n_samples, spacing, n_therm : MC parameters
    seed : RNG seed

    Returns
    -------
    xi : float
        Correlation length (in lattice units).
    r_squared : float
        R^2 of the exponential fit (1.0 = perfect).
    """
    mc = RPSTMonteCarlo(p, n_sites, temperature=temperature, seed=seed)
    mc.thermalize(n_therm)
    data = mc.measure(n_samples=n_samples, spacing=spacing)
    corr = data["correlations"]

    # Fit ln(C(r)) = -r/xi + const for r >= 1 where C(r) > 0
    distances = np.arange(len(corr))
    mask = (distances >= 1) & (corr > 1e-10)
    if np.sum(mask) < 2:
        return 0.0, 0.0

    r_fit = distances[mask].astype(float)
    log_c = np.log(corr[mask])

    # Linear least squares: log_c = a * r + b  =>  xi = -1/a
    A = np.vstack([r_fit, np.ones_like(r_fit)]).T
    result = np.linalg.lstsq(A, log_c, rcond=None)
    slope, intercept = result[0]

    if slope >= 0:
        # Correlation is not decaying — effectively infinite
        return float("inf"), 0.0

    xi = -1.0 / slope

    # R^2
    predicted = slope * r_fit + intercept
    ss_res = float(np.sum((log_c - predicted) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(xi), float(r_squared)


# ---------------------------------------------------------------------------
# Emergent dispersion relation
# ---------------------------------------------------------------------------

def emergent_dispersion_relation(
    p: int,
    n_sites: int,
    temperature: float,
    n_samples: int = 500,
    spacing: int = 10,
    n_therm: int = 500,
    seed: int = 42,
) -> Dict:
    """Extract the emergent dispersion relation from equilibrium fluctuations.

    FFT thermal configurations to obtain the power spectrum P(k),
    then infer omega(k) proportional to 1/P(k) (equipartition).

    In the continuum limit the effective dispersion should approach
    omega = c_eff |k| for small k.

    Parameters
    ----------
    p, n_sites, temperature : lattice parameters
    n_samples, spacing, n_therm : MC parameters
    seed : RNG seed

    Returns
    -------
    dict with:
        k_values : ndarray — wave-numbers (positive only)
        omega_values : ndarray — inferred omega(k)
        c_eff : float — effective speed of sound (slope at small k)
        power_spectrum : ndarray — averaged |phi_k|^2
    """
    mc = RPSTMonteCarlo(p, n_sites, temperature=temperature, seed=seed)
    mc.thermalize(n_therm)

    N = n_sites
    # Accumulate power spectrum
    power = np.zeros(N)

    for _ in range(n_samples):
        for _ in range(spacing):
            mc.sweep()
        # Map config to continuous phase: phi_j = 2 pi q_j / p
        phi = mc._two_pi_over_p * mc.config
        phi_k = np.fft.fft(phi) / np.sqrt(N)
        power += np.abs(phi_k) ** 2

    power /= n_samples

    # Keep positive frequencies only (excluding k=0)
    freqs = np.fft.fftfreq(N) * (2.0 * np.pi)  # angular wave-number
    pos = (freqs > 0) & (np.arange(N) <= N // 2)
    k_values = freqs[pos]
    P_k = power[pos]

    # Equipartition: <|phi_k|^2> = T / omega(k)^2  =>  omega(k) = sqrt(T / P(k))
    with np.errstate(divide="ignore", invalid="ignore"):
        omega_values = np.where(P_k > 1e-30, np.sqrt(temperature / P_k), 0.0)

    # Fit c_eff from smallest k modes: omega ~ c_eff |k|
    n_fit = max(3, len(k_values) // 10)
    k_small = k_values[:n_fit]
    omega_small = omega_values[:n_fit]
    if len(k_small) >= 2 and np.any(k_small > 0):
        # Weighted least squares through origin: omega = c * k
        c_eff = float(np.sum(k_small * omega_small) / np.sum(k_small ** 2))
    else:
        c_eff = 0.0

    return {
        "k_values": k_values,
        "omega_values": omega_values,
        "c_eff": c_eff,
        "power_spectrum": P_k,
    }
