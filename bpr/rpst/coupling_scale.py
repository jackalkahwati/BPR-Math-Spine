"""
Analysis of Coupling Scale J for BPR Effects

The key question: What is the physical coupling J in the RPST Hamiltonian?

Options:
1. J ~ eV (atomic scale) → λ_BPR ~ 10^-90 (unmeasurable)
2. J ~ E_Planck (Planck scale) → λ_BPR ~ 10^-70 × E_P (still tiny)
3. J ~ Casimir energy density × volume → needs calculation

This module explores option 3: J derived from vacuum energy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Physical constants
HBAR = 1.054571817e-34  # J·s
C = 299792458  # m/s
G = 6.67430e-11  # m³/(kg·s²)
L_PLANCK = np.sqrt(HBAR * G / C**3)  # ≈ 1.616e-35 m
E_PLANCK = np.sqrt(HBAR * C**5 / G)  # ≈ 1.956e9 J


def casimir_energy_density(separation: float) -> float:
    """
    Compute Casimir energy density between parallel plates.

    The Casimir energy per unit area is:
        E/A = -π² ℏc / (720 a³)

    Parameters
    ----------
    separation : float
        Plate separation a [m]

    Returns
    -------
    float
        Energy per unit area [J/m²]
    """
    return -np.pi**2 * HBAR * C / (720 * separation**3)


def casimir_force_density(separation: float) -> float:
    """
    Compute Casimir force per unit area (pressure).

    F/A = -dE/da / A = -π² ℏc / (240 a⁴)

    Parameters
    ----------
    separation : float
        Plate separation a [m]

    Returns
    -------
    float
        Force per unit area [N/m²]
    """
    return -np.pi**2 * HBAR * C / (240 * separation**4)


@dataclass
class CouplingAnalysis:
    """Results of coupling scale analysis."""
    J_atomic: float  # 1 eV reference
    J_casimir: float  # From Casimir energy
    J_vacuum: float  # From vacuum energy density
    enhancement_casimir: float  # J_casimir / J_atomic
    enhancement_vacuum: float  # J_vacuum / J_atomic


def analyze_coupling_scales(
    separation: float = 100e-9,  # 100 nm typical Casimir experiment
    plate_area: float = 1e-6,  # 1 mm² plate area
    N_modes: int = 10000
) -> CouplingAnalysis:
    """
    Analyze what coupling scale J is appropriate.

    The question: If the boundary substrate mediates vacuum energy,
    what is the appropriate J?

    Hypothesis: J should be related to the vacuum energy per mode
    that the substrate is "managing".

    Parameters
    ----------
    separation : float
        Plate separation [m]
    plate_area : float
        Plate area [m²]
    N_modes : int
        Number of substrate modes

    Returns
    -------
    CouplingAnalysis
        Comparison of coupling scales
    """
    # Reference: atomic scale
    J_atomic = 1.602e-19  # 1 eV in Joules

    # From Casimir energy
    E_casimir_per_area = abs(casimir_energy_density(separation))
    E_casimir_total = E_casimir_per_area * plate_area
    J_casimir = E_casimir_total / N_modes  # Energy per mode

    # From vacuum energy density
    # QFT predicts vacuum energy ~ (cutoff)⁴
    # With Planck cutoff: ρ_vac ~ E_P / ℓ_P³ (way too large)
    # With Casimir cutoff (1/a): ρ_vac ~ ℏc/a⁴ × volume
    #
    # For a slab of thickness a:
    vacuum_energy_density = HBAR * C / separation**4  # [J/m³]
    vacuum_volume = plate_area * separation
    E_vacuum = vacuum_energy_density * vacuum_volume
    J_vacuum = E_vacuum / N_modes

    return CouplingAnalysis(
        J_atomic=J_atomic,
        J_casimir=J_casimir,
        J_vacuum=J_vacuum,
        enhancement_casimir=J_casimir / J_atomic,
        enhancement_vacuum=J_vacuum / J_atomic
    )


def lambda_bpr_with_casimir_coupling(
    separation: float,
    plate_area: float,
    N_modes: int
) -> float:
    """
    Compute λ_BPR using Casimir-derived coupling.

    Instead of J = 1 eV, use J = E_casimir / N_modes.

    This tests whether the signal becomes measurable if
    the coupling is set by the vacuum energy itself.

    Returns
    -------
    float
        λ_BPR with Casimir-derived coupling [J·m²]
    """
    analysis = analyze_coupling_scales(separation, plate_area, N_modes)
    J = analysis.J_casimir

    # From boundary_energy.py:
    # κ = z/2 where z = 4 (square lattice)
    # λ_BPR = (ℓ_P²/8π) × κ × J
    kappa = 2.0
    lambda_bpr = (L_PLANCK**2 / (8 * np.pi)) * kappa * J

    return lambda_bpr


def relative_casimir_correction(
    separation: float,
    plate_area: float = 1e-6,
    N_modes: int = 10000
) -> float:
    """
    Estimate relative BPR correction to Casimir force.

    ΔF/F ~ λ_BPR × (geometry factor) / (standard Casimir)

    This is a rough order-of-magnitude estimate.

    Parameters
    ----------
    separation : float
        Plate separation [m]
    plate_area : float
        Plate area [m²]
    N_modes : int
        Number of substrate modes

    Returns
    -------
    float
        Order of magnitude of ΔF/F
    """
    lambda_bpr = lambda_bpr_with_casimir_coupling(separation, plate_area, N_modes)

    # Standard Casimir energy
    E_casimir = abs(casimir_energy_density(separation)) * plate_area

    # The BPR correction scales as:
    # ΔE ~ λ_BPR × (boundary gradient)² × (area)
    # where (boundary gradient)² ~ 1/ξ² and ξ ~ lattice spacing
    #
    # Very rough: ΔE/E ~ λ_BPR × N_modes / (E_casimir × ξ²)

    # Lattice spacing
    lattice_spacing = np.sqrt(plate_area / N_modes)

    # Correlation length (from boundary_energy.py)
    # ξ ~ a × √ln(p) ≈ a × 3 for p ~ 10^5
    xi = lattice_spacing * 3

    # Rough estimate
    delta_E = lambda_bpr * N_modes / xi**2
    relative_correction = delta_E / E_casimir

    return relative_correction


if __name__ == "__main__":
    print("Coupling Scale Analysis")
    print("=" * 60)

    # Typical Casimir experiment parameters
    a = 100e-9  # 100 nm separation
    A = 1e-6    # 1 mm² plate area
    N = 10000   # Substrate modes

    print(f"\nExperimental Parameters:")
    print(f"  Separation: {a*1e9:.0f} nm")
    print(f"  Plate area: {A*1e6:.0f} mm²")
    print(f"  N_modes: {N}")

    # Standard Casimir values
    F_casimir = casimir_force_density(a)
    E_casimir = casimir_energy_density(a)

    print(f"\nStandard Casimir:")
    print(f"  Energy/area: {E_casimir:.3e} J/m²")
    print(f"  Force/area: {F_casimir:.3e} N/m² = {F_casimir:.3e} Pa")

    # Coupling analysis
    analysis = analyze_coupling_scales(a, A, N)

    print(f"\nCoupling Scales:")
    print(f"  J_atomic (1 eV):    {analysis.J_atomic:.3e} J")
    print(f"  J_casimir:          {analysis.J_casimir:.3e} J")
    print(f"  J_vacuum:           {analysis.J_vacuum:.3e} J")
    print(f"\nEnhancement over atomic:")
    print(f"  Casimir coupling:   {analysis.enhancement_casimir:.3e}×")
    print(f"  Vacuum coupling:    {analysis.enhancement_vacuum:.3e}×")

    # λ_BPR with different couplings
    print(f"\nλ_BPR with Different Couplings:")

    # Atomic coupling
    lambda_atomic = (L_PLANCK**2 / (8 * np.pi)) * 2.0 * analysis.J_atomic
    print(f"  J = 1 eV:           λ_BPR = {lambda_atomic:.3e} J·m²")

    # Casimir coupling
    lambda_casimir = lambda_bpr_with_casimir_coupling(a, A, N)
    print(f"  J = E_casimir/N:    λ_BPR = {lambda_casimir:.3e} J·m²")

    # Vacuum coupling
    lambda_vacuum = (L_PLANCK**2 / (8 * np.pi)) * 2.0 * analysis.J_vacuum
    print(f"  J = E_vacuum/N:     λ_BPR = {lambda_vacuum:.3e} J·m²")

    # Relative correction estimate
    rel_corr = relative_casimir_correction(a, A, N)
    print(f"\nRelative Casimir Correction (rough estimate):")
    print(f"  ΔF/F ~ {rel_corr:.3e}")

    print(f"\nConclusion:")
    if abs(rel_corr) > 1e-3:
        print(f"  ✓ Potentially measurable (current precision ~10^-3)")
    elif abs(rel_corr) > 1e-10:
        print(f"  △ Below current precision but not hopeless")
    else:
        print(f"  ✗ Far below any conceivable measurement")

    # What would make it measurable?
    print(f"\nTo reach ΔF/F ~ 10^-3, would need:")
    target = 1e-3
    boost_needed = target / abs(rel_corr) if rel_corr != 0 else np.inf
    print(f"  Enhancement factor: {boost_needed:.3e}×")
