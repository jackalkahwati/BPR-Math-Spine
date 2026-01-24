"""
Vacuum-Boundary Coupling Derivation (g parameter)

This module derives the coupling constant g that appears in the BPR
Casimir correction formula:

    ΔF/F = g · λ · exp(-a/ξ) · cos(2πa/Λ)

The parameter g represents the overlap between:
1. Boundary eigenmodes (from RPST substrate)
2. Vacuum fluctuation modes (from QED)

DERIVATION STATUS: Week 2 of Parameter-Free Sprint

Mathematical Chain:
1. Boundary eigenmodes on geometry (sphere, plate, etc.)
2. Vacuum fluctuation spectrum between boundaries
3. Overlap integral gives coupling strength
4. Sum over modes gives effective g

References
----------
[1] Casimir, H.B.G. (1948) - Original Casimir calculation
[2] Lifshitz (1956) - Dielectric Casimir theory
[3] Bordag et al. "Advances in the Casimir Effect" (2009)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
from scipy.special import spherical_jn, sph_harm
from scipy.integrate import quad, dblquad
from enum import Enum


class Geometry(Enum):
    """Supported boundary geometries."""
    PARALLEL_PLATES = "parallel_plates"
    SPHERE_PLATE = "sphere_plate"
    TWO_SPHERES = "two_spheres"
    CYLINDER_PLATE = "cylinder_plate"


@dataclass
class VacuumCouplingResult:
    """Results of vacuum-boundary coupling calculation."""
    g: float                    # Total coupling constant
    g_per_mode: np.ndarray      # Coupling per eigenmode
    dominant_mode: int          # Most strongly coupled mode
    mode_sum_converged: bool    # Whether sum converged
    n_modes_used: int           # Number of modes in sum


# Physical constants
HBAR = 1.054571817e-34  # J·s
C = 299792458  # m/s
EPSILON_0 = 8.854187817e-12  # F/m


def vacuum_mode_spectrum(k: float, separation: float) -> float:
    """
    Vacuum fluctuation mode amplitude at wavenumber k.

    For the electromagnetic vacuum between parallel plates,
    the mode density is modified from free space.

    Parameters
    ----------
    k : float
        Wavenumber [1/m]
    separation : float
        Plate separation a [m]

    Returns
    -------
    float
        Mode amplitude squared |ψ_vac(k)|²
    """
    # Free space vacuum fluctuation amplitude
    # |E_vac|² ~ ℏω/(2ε₀V) for each mode
    # In k-space: |ψ(k)|² ~ 1/(k² + m²) for massive, ~ 1/k² for massless EM

    # For massless photons between plates:
    # Modes are quantized: k_n = nπ/a for n = 1, 2, 3, ...
    # Amplitude: |ψ_n|² ~ 1/k_n² = a²/(nπ)²

    if k <= 0:
        return 0.0

    return 1.0 / k**2


def plate_eigenvalues(n_max: int, separation: float) -> np.ndarray:
    """
    Eigenvalues for parallel plate geometry.

    The allowed wavenumbers perpendicular to plates are:
        k_n = nπ/a  for n = 1, 2, 3, ...

    Parameters
    ----------
    n_max : int
        Maximum mode number
    separation : float
        Plate separation [m]

    Returns
    -------
    np.ndarray
        Array of eigenvalues k_n
    """
    n = np.arange(1, n_max + 1)
    return n * np.pi / separation


def sphere_eigenvalues(l_max: int, radius: float) -> np.ndarray:
    """
    Eigenvalues for spherical boundary.

    On a sphere of radius R, the Laplacian eigenvalues are:
        λ_l = l(l+1)/R²  for l = 0, 1, 2, ...

    The corresponding wavenumbers are:
        k_l = √λ_l = √(l(l+1))/R

    Parameters
    ----------
    l_max : int
        Maximum angular momentum
    radius : float
        Sphere radius [m]

    Returns
    -------
    np.ndarray
        Array of eigenvalues k_l
    """
    l = np.arange(0, l_max + 1)
    # Avoid l=0 which gives k=0
    l = l[1:]  # Start from l=1
    return np.sqrt(l * (l + 1)) / radius


def boundary_mode_amplitude(k_boundary: float, k_vacuum: float,
                            geometry: Geometry) -> float:
    """
    Overlap between boundary eigenmode and vacuum mode.

    This computes |⟨φ_boundary(k_b) | ψ_vacuum(k_v)⟩|²

    The overlap depends on how well the boundary mode "resonates"
    with the vacuum fluctuation at that wavenumber.

    Parameters
    ----------
    k_boundary : float
        Boundary eigenmode wavenumber
    k_vacuum : float
        Vacuum mode wavenumber
    geometry : Geometry
        Boundary geometry

    Returns
    -------
    float
        Overlap amplitude squared
    """
    if k_boundary <= 0 or k_vacuum <= 0:
        return 0.0

    # Resonance when k_boundary ≈ k_vacuum
    # Width of resonance depends on geometry

    if geometry == Geometry.PARALLEL_PLATES:
        # Sharp resonance for plates (discrete modes)
        # Overlap ~ δ(k_b - k_v) in continuum limit
        # For discrete: Gaussian approximation
        delta_k = abs(k_boundary - k_vacuum)
        width = k_boundary * 0.1  # 10% width
        return np.exp(-delta_k**2 / (2 * width**2))

    elif geometry == Geometry.SPHERE_PLATE:
        # Broader resonance due to curvature
        delta_k = abs(k_boundary - k_vacuum)
        width = k_boundary * 0.2  # 20% width
        return np.exp(-delta_k**2 / (2 * width**2))

    else:
        # Default: moderate resonance
        delta_k = abs(k_boundary - k_vacuum)
        width = k_boundary * 0.15
        return np.exp(-delta_k**2 / (2 * width**2))


def compute_g_parallel_plates(
    separation: float,
    plate_size: float,
    n_modes: int = 100
) -> VacuumCouplingResult:
    """
    Compute vacuum coupling g for parallel plate geometry.

    The coupling is the sum over all mode overlaps:
        g = Σₙ |⟨φₙ | ψ_vac⟩|² × (mode weight)

    Parameters
    ----------
    separation : float
        Plate separation a [m]
    plate_size : float
        Characteristic plate size L [m]
    n_modes : int
        Number of modes to include

    Returns
    -------
    VacuumCouplingResult
        Coupling constant and diagnostics
    """
    # Boundary eigenvalues (perpendicular modes)
    k_boundary = plate_eigenvalues(n_modes, separation)

    # For each boundary mode, compute coupling to vacuum
    g_per_mode = np.zeros(n_modes)

    for i, k_b in enumerate(k_boundary):
        # Vacuum mode at same wavenumber
        psi_vac_sq = vacuum_mode_spectrum(k_b, separation)

        # Overlap (for parallel plates, modes are orthonormal)
        # The overlap is 1 for matching modes, 0 otherwise
        # Weight by vacuum amplitude
        overlap = 1.0  # Perfect overlap for matching mode number

        # Mode degeneracy: parallel momentum modes
        # Number of transverse modes ~ (k_b × L)²
        degeneracy = (k_b * plate_size)**2 / (4 * np.pi**2)
        degeneracy = max(degeneracy, 1.0)

        g_per_mode[i] = overlap * psi_vac_sq * degeneracy

    # Total coupling (normalized)
    # Normalize by total number of modes to get dimensionless g
    g_total = np.sum(g_per_mode)

    # Normalize to make g dimensionless and O(1) for standard Casimir
    # Standard Casimir ~ π²/720, so normalize by this
    normalization = np.pi**2 / 720
    g_normalized = g_total * separation**2 / normalization

    # Find dominant mode
    dominant = np.argmax(g_per_mode)

    # Check convergence (last mode should be small)
    converged = g_per_mode[-1] < 0.01 * g_per_mode[0] if g_per_mode[0] > 0 else True

    return VacuumCouplingResult(
        g=g_normalized,
        g_per_mode=g_per_mode,
        dominant_mode=dominant + 1,  # 1-indexed
        mode_sum_converged=converged,
        n_modes_used=n_modes
    )


def compute_g_sphere_plate(
    sphere_radius: float,
    separation: float,
    l_max: int = 50
) -> VacuumCouplingResult:
    """
    Compute vacuum coupling g for sphere-plate geometry.

    This is more complex due to the curved boundary.
    The sphere modes (spherical harmonics) couple to the
    plate-modified vacuum.

    Parameters
    ----------
    sphere_radius : float
        Sphere radius R [m]
    separation : float
        Minimum separation a [m]
    l_max : int
        Maximum angular momentum to include

    Returns
    -------
    VacuumCouplingResult
        Coupling constant and diagnostics
    """
    # Sphere eigenvalues
    k_sphere = sphere_eigenvalues(l_max, sphere_radius)
    n_modes = len(k_sphere)

    # Vacuum mode at the gap
    k_gap = np.pi / separation  # Dominant vacuum mode

    g_per_mode = np.zeros(n_modes)

    for i, k_s in enumerate(k_sphere):
        l = i + 1  # Angular momentum (starting from l=1)

        # Overlap with vacuum mode
        overlap = boundary_mode_amplitude(k_s, k_gap, Geometry.SPHERE_PLATE)

        # Vacuum amplitude at this k
        psi_vac_sq = vacuum_mode_spectrum(k_s, separation)

        # Degeneracy: 2l+1 for each l
        degeneracy = 2 * l + 1

        g_per_mode[i] = overlap * psi_vac_sq * degeneracy

    # Total and normalize
    g_total = np.sum(g_per_mode)

    # Sphere-plate Casimir has different geometry factor
    # F_sp ~ -π³ℏcR/(360a³) for R >> a
    # Normalize accordingly
    geometry_factor = np.pi**3 / 360
    g_normalized = g_total * separation**2 * sphere_radius / geometry_factor

    dominant = np.argmax(g_per_mode)
    converged = g_per_mode[-1] < 0.01 * np.max(g_per_mode) if np.max(g_per_mode) > 0 else True

    return VacuumCouplingResult(
        g=g_normalized,
        g_per_mode=g_per_mode,
        dominant_mode=dominant + 1,
        mode_sum_converged=converged,
        n_modes_used=n_modes
    )


def compute_g(
    geometry: Geometry,
    separation: float,
    boundary_size: float,
    n_modes: int = 100
) -> VacuumCouplingResult:
    """
    Compute vacuum coupling g for given geometry.

    This is the main interface for g derivation.

    Parameters
    ----------
    geometry : Geometry
        Boundary geometry type
    separation : float
        Gap/separation distance [m]
    boundary_size : float
        Characteristic boundary size [m]
        (plate size for plates, radius for spheres)
    n_modes : int
        Number of modes to include

    Returns
    -------
    VacuumCouplingResult
        Coupling constant and diagnostics
    """
    if geometry == Geometry.PARALLEL_PLATES:
        return compute_g_parallel_plates(separation, boundary_size, n_modes)
    elif geometry == Geometry.SPHERE_PLATE:
        return compute_g_sphere_plate(boundary_size, separation, n_modes)
    else:
        # Default to parallel plates
        return compute_g_parallel_plates(separation, boundary_size, n_modes)


def g_scaling_with_separation(
    geometry: Geometry,
    boundary_size: float,
    separations: np.ndarray,
    n_modes: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute how g scales with separation.

    Parameters
    ----------
    geometry : Geometry
        Boundary geometry
    boundary_size : float
        Characteristic size [m]
    separations : np.ndarray
        Array of separations [m]
    n_modes : int
        Number of modes

    Returns
    -------
    tuple
        (separations, g_values)
    """
    g_values = np.zeros_like(separations)

    for i, a in enumerate(separations):
        result = compute_g(geometry, a, boundary_size, n_modes)
        g_values[i] = result.g

    return separations, g_values


def verify_casimir_geometry_factors() -> dict:
    """
    Verify that derived g reproduces standard Casimir geometry dependence.

    Standard results:
    - Parallel plates: F ~ 1/a⁴
    - Sphere-plate: F ~ R/a³ (for R >> a)

    Our g should give these scalings when combined with the formula.

    Returns
    -------
    dict
        Verification results
    """
    results = {}

    # Test 1: Parallel plates scaling
    # g should be ~constant (geometry factor absorbed in formula)
    L = 0.01  # 1 cm plate
    separations = np.array([50e-9, 100e-9, 200e-9, 500e-9])

    g_plates = []
    for a in separations:
        result = compute_g_parallel_plates(a, L, n_modes=50)
        g_plates.append(result.g)

    g_plates = np.array(g_plates)
    # Check if g is roughly constant (within factor of 2)
    g_variation = np.std(g_plates) / np.mean(g_plates)
    results['plates_g_constant'] = g_variation < 0.5
    results['plates_g_values'] = g_plates

    # Test 2: Sphere-plate scaling
    R = 0.001  # 1 mm sphere

    g_sphere = []
    for a in separations:
        result = compute_g_sphere_plate(R, a, l_max=30)
        g_sphere.append(result.g)

    g_sphere = np.array(g_sphere)
    results['sphere_g_values'] = g_sphere

    # For sphere-plate, g might scale with a
    # Check if scaling is consistent
    results['sphere_g_ratio'] = g_sphere[-1] / g_sphere[0] if g_sphere[0] != 0 else np.inf

    return results


def derive_g_from_substrate(
    substrate_eigenvalues: np.ndarray,
    separation: float,
    geometry: Geometry = Geometry.PARALLEL_PLATES
) -> float:
    """
    Derive g using eigenvalues from actual RPST substrate simulation.

    This connects the abstract derivation to concrete substrate data.

    Parameters
    ----------
    substrate_eigenvalues : np.ndarray
        Eigenvalues from RPST boundary Hamiltonian
    separation : float
        Casimir gap [m]
    geometry : Geometry
        Boundary geometry

    Returns
    -------
    float
        Derived coupling g
    """
    # Vacuum mode at gap
    k_vac = np.pi / separation

    # Convert substrate eigenvalues to wavenumbers
    # Eigenvalues have units [1/Length²], so k = √λ
    k_substrate = np.sqrt(np.abs(substrate_eigenvalues))

    # Compute overlap with vacuum mode
    g = 0.0
    for k_s in k_substrate:
        if k_s > 0:
            overlap = boundary_mode_amplitude(k_s, k_vac, geometry)
            psi_vac_sq = vacuum_mode_spectrum(k_s, separation)
            g += overlap * psi_vac_sq

    # Normalize
    g *= separation**2

    return g


if __name__ == "__main__":
    print("Vacuum-Boundary Coupling Derivation")
    print("=" * 60)

    # Test parameters
    a = 100e-9  # 100 nm separation
    L = 0.01    # 1 cm plate
    R = 0.001   # 1 mm sphere

    print(f"\nParameters:")
    print(f"  Separation: {a*1e9:.0f} nm")
    print(f"  Plate size: {L*100:.1f} cm")
    print(f"  Sphere radius: {R*1000:.1f} mm")

    # Parallel plates
    print(f"\n--- Parallel Plates ---")
    result_pp = compute_g_parallel_plates(a, L, n_modes=100)
    print(f"  g = {result_pp.g:.4f}")
    print(f"  Dominant mode: n = {result_pp.dominant_mode}")
    print(f"  Converged: {result_pp.mode_sum_converged}")

    # Sphere-plate
    print(f"\n--- Sphere-Plate ---")
    result_sp = compute_g_sphere_plate(R, a, l_max=50)
    print(f"  g = {result_sp.g:.6f}")
    print(f"  Dominant mode: l = {result_sp.dominant_mode}")
    print(f"  Converged: {result_sp.mode_sum_converged}")

    # Scaling with separation
    print(f"\n--- Scaling with Separation ---")
    seps = np.array([50, 100, 200, 500]) * 1e-9
    _, g_vs_a = g_scaling_with_separation(Geometry.PARALLEL_PLATES, L, seps)

    print(f"  {'a (nm)':<10} {'g':>10}")
    for s, g in zip(seps, g_vs_a):
        print(f"  {s*1e9:<10.0f} {g:>10.4f}")

    # Verify geometry factors
    print(f"\n--- Geometry Factor Verification ---")
    verification = verify_casimir_geometry_factors()
    print(f"  Plates g constant: {verification['plates_g_constant']}")
    print(f"  Plates g values: {verification['plates_g_values']}")
    print(f"  Sphere g ratio (500nm/50nm): {verification['sphere_g_ratio']:.2f}")
