"""
Decay Length (ξ) and Oscillation Period (Λ) Derivation

This module derives the remaining parameters in the BPR Casimir formula:

    ΔF/F = g · λ · exp(-a/ξ) · cos(2πa/Λ)

ξ (xi): Characteristic decay length
    - Controls exponential suppression at large separation
    - Should emerge from correlation length in substrate

Λ (Lambda): Oscillation wavelength
    - Controls interference pattern
    - Should emerge from eigenmode spacing

DERIVATION STATUS: Week 3 of Parameter-Free Sprint

Mathematical Chain:
1. Correlation length from substrate phase correlations
2. Eigenmode spacing from boundary spectrum
3. Connect to physical separation scales
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import curve_fit

from .boundary_energy import SubstrateParameters, LatticeGeometry, L_PLANCK


@dataclass
class DecayOscillationParams:
    """Derived decay and oscillation parameters."""
    xi: float           # Correlation/decay length [m]
    Lambda: float       # Oscillation wavelength [m]
    xi_over_a: float    # ξ/a ratio (should be O(1) for measurable effect)
    Lambda_over_a: float  # Λ/a ratio


def derive_xi_from_correlation(
    params: SubstrateParameters,
    temperature_ratio: float = 1.0
) -> float:
    """
    Derive correlation length ξ from substrate phase correlations.

    The correlation function of the boundary phase field decays as:
        ⟨φ(x)φ(0)⟩ ~ exp(-|x|/ξ)

    For a 2D XY-like model on a lattice:
        ξ ~ a × exp(const × J/T)  at low T
        ξ ~ a × √(J/T)            at high T

    For RPST, the "effective temperature" comes from coarse-graining
    over p discrete states.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    temperature_ratio : float
        Effective T/J ratio (default 1.0)

    Returns
    -------
    float
        Correlation length ξ [m]
    """
    a = params.lattice_spacing
    p = params.p
    J = params.J

    # Effective temperature from discrete averaging
    # Heuristic: T_eff ~ J / ln(p)
    # This comes from entropy of p states per site
    T_eff_over_J = 1.0 / np.log(p) * temperature_ratio

    # For 2D XY model in high-T regime:
    # ξ ~ a × √(J/T) = a × √(1/T_eff_over_J)
    # This is a mean-field estimate

    if T_eff_over_J > 0:
        xi = a * np.sqrt(1.0 / T_eff_over_J)
    else:
        xi = a * 1000  # Very large (quasi-ordered)

    return xi


def derive_xi_from_coarse_graining(
    params: SubstrateParameters,
    coarse_graining_scale: Optional[float] = None
) -> float:
    """
    Derive ξ from the coarse-graining procedure itself.

    When we smooth the discrete substrate to get a continuum field,
    we introduce a correlation length ~ smoothing scale.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    coarse_graining_scale : float, optional
        Smoothing scale ε [m]. If None, use lattice spacing.

    Returns
    -------
    float
        Correlation length ξ [m]
    """
    a = params.lattice_spacing

    if coarse_graining_scale is None:
        # Natural scale: geometric mean of lattice spacing and system size
        epsilon = np.sqrt(a * params.radius)
    else:
        epsilon = coarse_graining_scale

    # Correlation length is at least the smoothing scale
    # With logarithmic correction for large p
    xi = epsilon * np.sqrt(np.log(params.p))

    return xi


def derive_Lambda_from_eigenspacing(
    params: SubstrateParameters,
    n_ref: int = 1
) -> float:
    """
    Derive oscillation wavelength Λ from eigenmode spacing.

    The oscillatory term cos(2πa/Λ) arises from interference
    between adjacent eigenmodes. The spacing between modes
    determines the interference pattern.

    For a spherical boundary:
        λ_l = l(l+1)/R²  → k_l = √(l(l+1))/R
        Δk = k_{l+1} - k_l ≈ 1/R  for large l

    The oscillation wavelength is:
        Λ = 2π/Δk ≈ 2πR

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    n_ref : int
        Reference mode number for spacing calculation

    Returns
    -------
    float
        Oscillation wavelength Λ [m]
    """
    R = params.radius

    if params.geometry == LatticeGeometry.SPHERE:
        # Spherical harmonics: k_l = √(l(l+1))/R
        # For l >> 1: Δk ≈ 1/R
        # So Λ = 2π/Δk ≈ 2πR
        Lambda = 2 * np.pi * R

    elif params.geometry == LatticeGeometry.RING:
        # 1D ring: k_n = n/R
        # Δk = 1/R
        # Λ = 2πR
        Lambda = 2 * np.pi * R

    elif params.geometry == LatticeGeometry.SQUARE:
        # Square lattice on torus: modes at k = (2πn/L, 2πm/L)
        # Spacing ~ 2π/L
        # L = 2πR, so Δk ~ 1/R
        Lambda = 2 * np.pi * R

    else:
        # Default
        Lambda = 2 * np.pi * R

    return Lambda


def derive_Lambda_from_vacuum_modes(
    separation: float,
    plate_size: float
) -> float:
    """
    Derive Λ from vacuum mode structure between plates.

    For parallel plates separated by a, the vacuum modes have:
        k_n = nπ/a

    The mode spacing is Δk = π/a, giving:
        Λ = 2π/Δk = 2a

    This predicts oscillations with period 2a.

    Parameters
    ----------
    separation : float
        Plate separation [m]
    plate_size : float
        Plate size [m] (affects transverse modes)

    Returns
    -------
    float
        Oscillation wavelength Λ [m]
    """
    # Perpendicular mode spacing: Δk_perp = π/a
    # Transverse mode spacing: Δk_trans = 2π/L

    # The dominant oscillation comes from perpendicular modes
    Lambda = 2 * separation

    return Lambda


def derive_all_decay_oscillation(
    params: SubstrateParameters,
    separation: float
) -> DecayOscillationParams:
    """
    Derive both ξ and Λ from substrate parameters.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    separation : float
        Casimir gap [m]

    Returns
    -------
    DecayOscillationParams
        Derived parameters
    """
    # Derive ξ (use geometric mean of two methods)
    xi_corr = derive_xi_from_correlation(params)
    xi_coarse = derive_xi_from_coarse_graining(params)
    xi = np.sqrt(xi_corr * xi_coarse)

    # Derive Λ (from eigenmode spacing)
    Lambda = derive_Lambda_from_eigenspacing(params)

    return DecayOscillationParams(
        xi=xi,
        Lambda=Lambda,
        xi_over_a=xi / separation,
        Lambda_over_a=Lambda / separation
    )


def analyze_measurability(
    params: SubstrateParameters,
    separation: float
) -> dict:
    """
    Analyze whether the derived ξ and Λ give measurable effects.

    For the correction exp(-a/ξ)·cos(2πa/Λ) to be significant:
    1. Need ξ ≳ a (otherwise exponentially suppressed)
    2. Need Λ ~ a (otherwise oscillation doesn't matter)

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    separation : float
        Casimir gap [m]

    Returns
    -------
    dict
        Analysis results
    """
    derived = derive_all_decay_oscillation(params, separation)

    # Exponential suppression factor
    exp_factor = np.exp(-separation / derived.xi)

    # Oscillation factor (worst case is cos = 0)
    osc_factor_max = 1.0
    osc_factor_min = 0.0

    # Combined factor
    combined_max = exp_factor * osc_factor_max
    combined_min = exp_factor * osc_factor_min

    return {
        'xi': derived.xi,
        'Lambda': derived.Lambda,
        'xi_over_a': derived.xi_over_a,
        'Lambda_over_a': derived.Lambda_over_a,
        'exp_suppression': exp_factor,
        'combined_factor_max': combined_max,
        'combined_factor_min': combined_min,
        'measurable': exp_factor > 1e-3  # Threshold for current precision
    }


def xi_Lambda_vs_separation(
    params: SubstrateParameters,
    separations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ξ/a and Λ/a ratios as function of separation.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration
    separations : np.ndarray
        Array of separations [m]

    Returns
    -------
    tuple
        (separations, xi_over_a, Lambda_over_a)
    """
    xi_over_a = np.zeros_like(separations)
    Lambda_over_a = np.zeros_like(separations)

    for i, a in enumerate(separations):
        derived = derive_all_decay_oscillation(params, a)
        xi_over_a[i] = derived.xi_over_a
        Lambda_over_a[i] = derived.Lambda_over_a

    return separations, xi_over_a, Lambda_over_a


if __name__ == "__main__":
    print("Decay Length and Oscillation Period Derivation")
    print("=" * 60)

    # Create substrate parameters
    params = SubstrateParameters(
        p=104729,  # Large prime
        N=10000,
        J=1.602e-19,  # 1 eV
        geometry=LatticeGeometry.SPHERE,
        radius=0.01  # 1 cm
    )

    print(f"\nSubstrate Parameters:")
    print(f"  p = {params.p}")
    print(f"  N = {params.N}")
    print(f"  J = {params.J:.3e} J")
    print(f"  Radius = {params.radius*100:.1f} cm")
    print(f"  Lattice spacing = {params.lattice_spacing:.3e} m")

    # Typical Casimir separation
    a = 100e-9  # 100 nm

    print(f"\nCasimir separation: {a*1e9:.0f} nm")

    # Derive parameters
    derived = derive_all_decay_oscillation(params, a)

    print(f"\nDerived Parameters:")
    print(f"  ξ = {derived.xi:.3e} m = {derived.xi*1e6:.1f} μm")
    print(f"  Λ = {derived.Lambda:.3e} m = {derived.Lambda*100:.1f} cm")
    print(f"  ξ/a = {derived.xi_over_a:.1f}")
    print(f"  Λ/a = {derived.Lambda_over_a:.1e}")

    # Analyze measurability
    analysis = analyze_measurability(params, a)

    print(f"\nMeasurability Analysis:")
    print(f"  exp(-a/ξ) = {analysis['exp_suppression']:.3e}")
    print(f"  Combined factor (max): {analysis['combined_factor_max']:.3e}")
    print(f"  Measurable (factor > 10^-3): {analysis['measurable']}")

    # Different separations
    print(f"\n--- Variation with Separation ---")
    seps = np.array([10, 50, 100, 500, 1000]) * 1e-9  # nm

    print(f"  {'a (nm)':<10} {'ξ/a':<10} {'Λ/a':<12} {'exp(-a/ξ)':<12}")
    for sep in seps:
        d = derive_all_decay_oscillation(params, sep)
        exp_factor = np.exp(-sep / d.xi)
        print(f"  {sep*1e9:<10.0f} {d.xi_over_a:<10.1f} {d.Lambda_over_a:<12.1e} {exp_factor:<12.3e}")

    # Key finding
    print(f"\nKey Finding:")
    if derived.xi > a:
        print(f"  ξ > a: Exponential suppression is mild ({analysis['exp_suppression']:.2e})")
    else:
        print(f"  ξ < a: Exponential suppression is severe ({analysis['exp_suppression']:.2e})")

    if derived.Lambda > 100 * a:
        print(f"  Λ >> a: Oscillation occurs over much larger scale than experiment")
    else:
        print(f"  Λ ~ a: Oscillation might be observable within experimental range")
