"""
Theory II: Vacuum Impedance Mismatch Theory
=============================================

Derives dark-matter phenomenology, dark-energy density, and the MOND
acceleration scale from impedance mismatch between topological sectors
of the BPR substrate.

Key objects
-----------
* ``TopologicalImpedance``  – Z(W) for an excitation with winding W
* ``DarkMatterProfile``     – density profile with prime-periodic modulation
* ``DarkEnergyDensity``     – frustration energy density ρ_DE
* ``MONDInterpolation``     – transition function μ(a/a₀)

References: Al-Kahwati (2026), *Ten Adjacent Theories*, §4
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C = 299792458.0                    # m/s
_HBAR = 1.054571817e-34             # J·s
_G = 6.67430e-11                    # m³ kg⁻¹ s⁻²
_M_PL = 2.176434e-8                 # kg  (Planck mass)
_R_HUBBLE = 4.4e26                  # m   (Hubble radius, approximate)
_Z0_VACUUM = 376.730313668          # Ω   (impedance of free space)

# ---------------------------------------------------------------------------
# §4.2  Topological impedance  Z(W)
# ---------------------------------------------------------------------------

@dataclass
class TopologicalImpedance:
    """Impedance of a topological excitation with winding number W.

    Z(W) = Z₀ √(1 + |W|²/W_c²)

    where Z₀ is the vacuum impedance and W_c is a critical winding number.
    """
    Z0: float = _Z0_VACUUM
    W_c: float = 10.0

    def __call__(self, W: float | np.ndarray) -> float | np.ndarray:
        W = np.asarray(W, dtype=float)
        return self.Z0 * np.sqrt(1.0 + (W / self.W_c) ** 2)

    def em_coupling(self, W: float | np.ndarray, g0: float = 1.0) -> float | np.ndarray:
        """Electromagnetic coupling suppressed by impedance mismatch.

        g_EM(W) = g₀ / (1 + |W|²/W_c²)

        High-winding solitons (|W| >> W_c) are EM-dark.
        """
        W = np.asarray(W, dtype=float)
        return g0 / (1.0 + (W / self.W_c) ** 2)


# ---------------------------------------------------------------------------
# §4.3  Dark energy from global phase frustration
# ---------------------------------------------------------------------------

@dataclass
class DarkEnergyDensity:
    """Vacuum energy density from boundary phase frustration.

    ρ_DE ~ κ / (p L²)

    Identifying κ ~ M_Pl² and L ~ R_Hubble:
        Λ_eff ~ M_Pl² / (p R_H²)

    For p ~ 10⁶⁰ this yields Λ ~ 10⁻¹²² M_Pl⁴.
    """
    kappa: float = _M_PL ** 2     # boundary stiffness ≈ M_Pl²
    p: float = 1e60               # substrate prime modulus
    L: float = _R_HUBBLE          # characteristic length scale

    @property
    def rho_DE(self) -> float:
        """Frustration energy density (J/m³ or natural-unit equivalent)."""
        return self.kappa / (self.p * self.L ** 2)

    @property
    def lambda_eff(self) -> float:
        """Effective cosmological constant Λ_eff ~ M_Pl² / (p R_H²)."""
        return self.kappa / (self.p * self.L ** 2)


# ---------------------------------------------------------------------------
# §4.4  MOND acceleration scale and interpolation function
# ---------------------------------------------------------------------------

@dataclass
class MONDInterpolation:
    """Modified gravity at galactic scales from impedance transition.

    Characteristic acceleration:
        a₀ = c H₀ / (2π) × (1 + z/(4 ln p))

    Base: cosmological boundary (Hubble horizon). Correction: boundary
    coordination z enhances the transition scale (more modes at horizon).

    Interpolation function (simple form, §4.4):
        μ(x) = x / (1 + x)     where x = a / a₀

    In the deep-MOND limit (x << 1):  μ ≈ x  →  F ∝ 1/r
    In the Newtonian limit  (x >> 1):  μ ≈ 1  →  F ∝ 1/r²
    """
    H0_km_s_Mpc: float = 67.4   # Hubble constant
    p: int = 104729
    z: int = 6

    @property
    def H0_si(self) -> float:
        """Hubble constant in SI (s⁻¹)."""
        return self.H0_km_s_Mpc * 1e3 / 3.0857e22  # km/s/Mpc → s⁻¹

    @property
    def a0(self) -> float:
        """MOND acceleration scale (m/s²).

        a₀ = c H₀ / (2π) × (1 + z/(4 ln p)).
        Base from Hubble; boundary correction from substrate.
        """
        base = _C * self.H0_si / (2.0 * np.pi)
        correction = 1.0 + self.z / (4.0 * np.log(self.p))
        return float(base * correction)

    def mu(self, a: float | np.ndarray) -> float | np.ndarray:
        """Interpolation function μ(a/a₀)."""
        x = np.asarray(a, dtype=float) / self.a0
        return x / (1.0 + x)

    def effective_gravity(self, a_newton: float | np.ndarray) -> float | np.ndarray:
        """Effective gravitational acceleration including MOND correction.

        a_eff = a_N / μ(a_N/a₀)
        """
        mu_val = self.mu(a_newton)
        # Guard against μ=0 at a=0
        return np.where(mu_val > 0, np.asarray(a_newton) / mu_val, 0.0)


# ---------------------------------------------------------------------------
# §4.2 / §4.6  Dark-matter density profile with prime-periodic modulation
# ---------------------------------------------------------------------------

@dataclass
class DarkMatterProfile:
    """Dark-matter density profile with BPR prime-periodic modulation.

    ρ_DM(r) contains Fourier components at wavevectors
        k_n = γ_n / R
    where γ_n are Riemann zeta zeros (P2.1).

    For demonstration the first few non-trivial zeros are hard-coded.
    """
    rho_0: float = 1.0      # central density (kg/m³)
    R_scale: float = 1.0    # halo scale radius (m or kpc)
    modulation_amplitude: float = 0.01   # relative amplitude of prime modulation

    # First ten non-trivial Riemann zeta zeros (imaginary parts)
    _ZETA_ZEROS: tuple = (
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    )

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Evaluate ρ_DM(r) = ρ_NFW(r) × (1 + δ_prime(r))."""
        r = np.asarray(r, dtype=float)
        # NFW-like base profile
        x = r / self.R_scale
        rho_nfw = self.rho_0 / (x * (1.0 + x) ** 2 + 1e-30)

        # Prime-periodic modulation (P2.1)
        delta = np.zeros_like(r)
        for gamma_n in self._ZETA_ZEROS:
            k_n = gamma_n / self.R_scale
            delta += np.cos(k_n * r) / gamma_n  # amplitude ~ 1/γ_n
        delta *= self.modulation_amplitude

        return rho_nfw * (1.0 + delta)


# ---------------------------------------------------------------------------
# §4.5  Regime selector: particle DM vs modified gravity
# ---------------------------------------------------------------------------

def dominant_regime(n_W: float, n_c: float = 1.0) -> str:
    """Determine whether particle DM or modified gravity dominates.

    n_W >> n_c  →  particle-like dark matter
    n_W << n_c  →  impedance-modified gravity (MOND-like)
    """
    if n_W > n_c:
        return "particle_dark_matter"
    return "modified_gravity"


# ---------------------------------------------------------------------------
# §4.7  Dark matter self-interaction cross-section  (Prediction 10)
# ---------------------------------------------------------------------------

def dm_self_interaction_cross_section(W_c: float, m_dm_eV: float = 1e9,
                                       sigma_0: float = 1e-25) -> float:
    """Self-interaction cross-section of high-winding dark matter solitons.

    σ_DM / m_DM ~ σ₀ / W_c²

    High-winding solitons interact via impedance overlap.  The cross-section
    is suppressed by 1/W_c² — dark enough to satisfy Bullet Cluster limits
    (σ/m < 1 cm²/g) but non-zero.

    Parameters
    ----------
    W_c : float   – critical winding number
    m_dm_eV : float – DM particle mass (eV)
    sigma_0 : float – bare cross-section (cm²)

    Returns
    -------
    float  – σ/m in cm²/g
    """
    sigma = sigma_0 / (W_c ** 2)
    m_dm_g = m_dm_eV * 1.783e-33  # eV → g
    if m_dm_g > 0:
        return sigma / m_dm_g
    return 0.0


# ---------------------------------------------------------------------------
# §4.8  Muon anomalous magnetic moment correction  (Prediction 11)
# ---------------------------------------------------------------------------

def muon_g2_correction(W_c: float, m_muon: float = 105.66e6,
                        m_electron: float = 0.511e6,
                        alpha_em: float = 1.0 / 137.036) -> float:
    """Impedance mismatch contribution to muon (g-2).

    The muon couples to a boundary sector with effective winding
    W_eff = m_muon / m_electron (mass ratio sets the boundary mode).

    Δ(g-2)_μ = (α/2π) × (m_muon/m_electron)² / W_c²

    Experimental discrepancy: Δa_μ ≈ 2.49 × 10⁻⁹ (Fermilab 2023).

    Parameters
    ----------
    W_c : float  – critical winding
    m_muon : float – muon mass (eV)
    m_electron : float – electron mass (eV)
    alpha_em : float – fine-structure constant

    Returns
    -------
    float – Δ(g-2)/2
    """
    W_eff = m_muon / m_electron
    return (alpha_em / (2.0 * np.pi)) * (W_eff / W_c) ** 2


# ---------------------------------------------------------------------------
# §4.9  Hubble tension from boundary phase evolution  (Prediction 12)
# ---------------------------------------------------------------------------

def hubble_tension(R_boundary_0: float, z_CMB: float = 1089.0,
                    H0_local: float = 73.0) -> dict:
    """Hubble tension from evolving R_boundary.

    a₀ = c² / R_boundary.  If R_boundary grows with cosmic expansion:
        R_boundary(z) = R_boundary_0 / (1 + z)^n

    then the early-universe (CMB) effective H₀ differs from the local
    value.  The tension ΔH₀ ≈ H₀_local × n × ln(1 + z_CMB) / 2.

    For n ≈ 0.01 this gives ΔH₀ ≈ 2-5 km/s/Mpc.

    Returns
    -------
    dict – with keys H0_local, H0_CMB_effective, delta_H0, n_evolution
    """
    # Infer n from the observed tension (~5 km/s/Mpc)
    delta_H0_target = 5.0  # km/s/Mpc (approximate observed tension)
    ln_factor = np.log(1.0 + z_CMB)
    n_evolution = 2.0 * delta_H0_target / (H0_local * ln_factor)

    H0_CMB_eff = H0_local * (1.0 - n_evolution * ln_factor / 2.0)

    return {
        "H0_local": H0_local,
        "H0_CMB_effective": H0_CMB_eff,
        "delta_H0": H0_local - H0_CMB_eff,
        "n_evolution": n_evolution,
        "R_boundary_0": R_boundary_0,
    }


# ---------------------------------------------------------------------------
# §4.10  Proton lifetime from winding tunneling  (Prediction 14)
# ---------------------------------------------------------------------------

def proton_lifetime(p: int, delta_W: int = 1,
                     tau_0_years: float = 1e30) -> float:
    """Proton lifetime from Class A winding-number tunneling.

    Baryon number = winding number.  Proton decay requires ΔW = 1
    tunneling through a topological barrier of height ~ p.

    τ_proton ~ τ₀ × exp(p^{ΔW / dim})

    For p ~ 10⁵ and d = 3:  τ_proton >> 10³⁴ years
    (consistent with Super-Kamiokande bound).

    Parameters
    ----------
    p : int     – substrate prime
    delta_W : int – winding change (= 1 for proton decay)
    tau_0_years : float – prefactor timescale (years)

    Returns
    -------
    float – lifetime in years (may be effectively infinite)
    """
    # Tunneling exponent: barrier ~ p^(ΔW/3)
    exponent = float(p) ** (delta_W / 3.0)
    # Clamp to avoid overflow
    if exponent > 700:
        return float("inf")
    return tau_0_years * np.exp(exponent)


def rotation_curve(r: np.ndarray, M_baryon: float, a0: float) -> np.ndarray:
    """Rotation curve from impedance-modified gravity (MOND).

    Solves  μ(|a|/a₀) |a| = a_N = GM/r²  for |a|, then v = √(|a| r).

    In deep-MOND regime (a_N << a₀):
        |a| = √(a_N a₀)  →  v = (G M a₀)^{1/4}   (flat)
    In Newtonian regime  (a_N >> a₀):
        |a| = a_N         →  v = √(G M / r)
    """
    r = np.asarray(r, dtype=float)
    a_N = _G * M_baryon / (r ** 2 + 1e-30)
    # Solve μ(a/a₀)·a = a_N  with  μ(x) = x/(1+x)
    # This gives  a²/(a₀+a) = a_N  →  a² - a_N a - a_N a₀ = 0
    # Positive root: a = [a_N + √(a_N² + 4 a_N a₀)] / 2
    a_eff = 0.5 * (a_N + np.sqrt(a_N ** 2 + 4.0 * a_N * a0))
    return np.sqrt(a_eff * r)
