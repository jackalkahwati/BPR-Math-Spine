"""
Boundary Energy Density from RPST Substrate

This module derives the effective coupling constant κ that appears in the
boundary energy density u = ½κ|∇φ|² from first principles of the discrete
RPST substrate dynamics.

DERIVATION STATUS: In Progress (Week 1 of Parameter-Free Sprint)

Mathematical Chain:
1. Discrete Hamiltonian on ℤₚ lattice
2. Continuum limit via coarse-graining
3. Extract coefficient of gradient term
4. Connect to physical units via ℓ_P

References
----------
[1] This derivation follows from standard lattice field theory
[2] Coarse-graining: Kadanoff (1966), Wilson (1971)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class LatticeGeometry(Enum):
    """Supported boundary geometries."""
    RING = "ring"           # 1D circle
    SQUARE = "square"       # 2D flat torus
    SPHERE = "sphere"       # 2D sphere (S²)


@dataclass
class SubstrateParameters:
    """
    Parameters defining the RPST substrate.

    Attributes
    ----------
    p : int
        Prime modulus for ℤₚ arithmetic. Must be prime.
    N : int
        Number of lattice nodes on boundary.
    J : float
        Nearest-neighbor coupling strength [Energy].
    geometry : LatticeGeometry
        Boundary geometry type.
    radius : float
        Characteristic size of boundary [Length].
        For ring: circumference/2π. For sphere: radius.
    """
    p: int
    N: int
    J: float
    geometry: LatticeGeometry
    radius: float

    def __post_init__(self):
        """Validate parameters."""
        if not self._is_prime(self.p):
            raise ValueError(f"p={self.p} must be prime")
        if self.N < 3:
            raise ValueError(f"N={self.N} must be >= 3")
        if self.J <= 0:
            raise ValueError(f"J={self.J} must be positive")
        if self.radius <= 0:
            raise ValueError(f"radius={self.radius} must be positive")

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    @property
    def lattice_spacing(self) -> float:
        """
        Compute lattice spacing a from geometry.

        Returns
        -------
        float
            Lattice spacing [Length]
        """
        if self.geometry == LatticeGeometry.RING:
            # N nodes on circle of circumference 2πR
            return 2 * np.pi * self.radius / self.N
        elif self.geometry == LatticeGeometry.SQUARE:
            # N nodes on square of side L = 2πR (for comparison)
            L = 2 * np.pi * self.radius
            return L / np.sqrt(self.N)
        elif self.geometry == LatticeGeometry.SPHERE:
            # N nodes on sphere of radius R
            # Area = 4πR², so a² ≈ 4πR²/N
            return self.radius * np.sqrt(4 * np.pi / self.N)
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")

    @property
    def coordination_number(self) -> int:
        """
        Return coordination number z (number of nearest neighbors).

        Returns
        -------
        int
            Number of nearest neighbors per node
        """
        if self.geometry == LatticeGeometry.RING:
            return 2  # Left and right neighbors
        elif self.geometry == LatticeGeometry.SQUARE:
            return 4  # Square lattice
        elif self.geometry == LatticeGeometry.SPHERE:
            return 6  # Approximate triangular lattice on sphere
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")

    @property
    def boundary_area(self) -> float:
        """
        Compute total boundary area/length.

        Returns
        -------
        float
            Total area [Length² for 2D, Length for 1D]
        """
        if self.geometry == LatticeGeometry.RING:
            return 2 * np.pi * self.radius
        elif self.geometry == LatticeGeometry.SQUARE:
            L = 2 * np.pi * self.radius
            return L * L
        elif self.geometry == LatticeGeometry.SPHERE:
            return 4 * np.pi * self.radius**2
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")


@dataclass
class DerivedCouplings:
    """
    Coupling constants derived from substrate parameters.

    Attributes
    ----------
    kappa : float
        Dimensionless boundary rigidity coefficient.
    kappa_dimensional : float
        Dimensional boundary rigidity [Energy].
    xi : float
        Correlation length [Length].
    lambda_bpr : float
        Stress-energy coupling including ℓ_P² [Energy·Length²].
    """
    kappa: float
    kappa_dimensional: float
    xi: float
    lambda_bpr: float


# Physical constants
HBAR = 1.054571817e-34  # J·s
C = 299792458  # m/s
G = 6.67430e-11  # m³/(kg·s²)
L_PLANCK = np.sqrt(HBAR * G / C**3)  # ≈ 1.616e-35 m
E_PLANCK = np.sqrt(HBAR * C**5 / G)  # ≈ 1.956e9 J


def derive_kappa(params: SubstrateParameters) -> float:
    """
    Derive the dimensionless boundary rigidity κ from substrate.

    The derivation proceeds:
    1. Start with discrete potential: V(Δq) = 1 - cos(2πΔq/p)
    2. Expand for small Δq: V ≈ (2π/p)² (Δq)²/2
    3. Sum over bonds: Σ_{⟨i,j⟩} J·V → (z/2)·J·(2π/p)² ∫|∇φ|² dA
    4. Extract coefficient: κ = z·(2π/p)²

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    float
        Dimensionless rigidity κ

    Notes
    -----
    This is κ in: u = ½ κ J |∇θ|²
    where θ = 2πq/p is the angular phase.

    The factor J sets the energy scale; κ is purely geometric.
    """
    z = params.coordination_number
    p = params.p

    # From the derivation:
    # V(Δq) ≈ (2π/p)² (Δq)²/2 for small Δq
    # Energy = (z/2) N J (2π/p)² ⟨(Δq)²⟩
    # In continuum: ∫ ½ κ J |∇θ|² dA where θ = 2πq/p
    # So κ = z (for proper normalization)

    # Actually, let's be more careful:
    # V(Δθ) = 1 - cos(Δθ) ≈ (Δθ)²/2 for small Δθ
    # where Δθ = 2π(q_j - q_i)/p
    #
    # Sum over bonds:
    # (z/2) N J (Δθ)²/2 = (z/4) N J (Δθ)²
    #
    # In continuum, |∇θ|² ~ (Δθ/a)², so (Δθ)² ~ a² |∇θ|²
    # And ∫ dA/a² = N (number of area elements)
    #
    # Energy = (z/4) J ∫ |∇θ|² dA
    #
    # Therefore: κ = z/2

    kappa = z / 2.0

    return kappa


def derive_kappa_dimensional(params: SubstrateParameters) -> float:
    """
    Derive dimensional boundary rigidity κ_dim [Energy].

    This is the coefficient in: u = ½ κ_dim |∇θ|²

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    float
        Dimensional rigidity [Energy]
    """
    kappa = derive_kappa(params)
    return kappa * params.J


def derive_correlation_length(params: SubstrateParameters) -> float:
    """
    Derive correlation length ξ from substrate parameters.

    For a discrete system with nearest-neighbor coupling J,
    the correlation length is related to the lattice spacing
    and the coupling strength.

    At zero temperature (ground state):
        ξ → ∞ (long-range order)

    At finite effective temperature T_eff:
        ξ ~ a / √(T_eff / J)

    For RPST, the "effective temperature" comes from
    the coarse-graining procedure over p states.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    float
        Correlation length [Length]

    Notes
    -----
    This derivation assumes the coarse-graining introduces
    an effective temperature T_eff ~ J/ln(p).

    CONJECTURE: This specific form needs verification.
    """
    a = params.lattice_spacing
    p = params.p

    # Effective temperature from coarse-graining over p states
    # Heuristic: averaging over p discrete states ≈ thermal average at T ~ J/ln(p)
    #
    # More precisely: The entropy of the discrete system is ~ N ln(p)
    # Thermal equilibration at energy J per bond gives T ~ J/ln(p)

    T_eff_over_J = 1.0 / np.log(p)

    # Standard result for 2D XY model:
    # ξ ~ a · exp(const / √(T/J)) at low T
    #
    # But for simplicity, use mean-field estimate:
    # ξ ~ a / √(T_eff/J) = a · √(ln(p))

    xi = a * np.sqrt(np.log(p))

    return xi


def derive_lambda_bpr(params: SubstrateParameters) -> float:
    """
    Derive the BPR stress-energy coupling λ_BPR.

    This is the coefficient in:
        T^μν = λ_BPR P^{ab}_{μν} ∂_a φ ∂_b φ

    The derivation connects boundary stress to bulk metric via:
        λ_BPR = (ℓ_P² / 8π) × κ_eff

    where κ_eff incorporates the boundary-to-bulk propagator.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    float
        Stress-energy coupling [Energy · Length²]

    Notes
    -----
    DERIVATION IN PROGRESS.

    The factor ℓ_P²/8π comes from Einstein's equation:
        G_μν = (8πG/c⁴) T_μν

    Rearranging: metric perturbation ~ (G/c⁴) × stress
                                    ~ (ℓ_P²/ℏc) × stress

    The boundary-bulk coupling κ_eff involves:
    1. Boundary rigidity κ (derived above)
    2. Propagator from boundary to bulk
    3. Mode overlap integrals

    For now, we use: λ_BPR = (ℓ_P²/8π) × κ_dimensional × (some factor)
    """
    kappa_dim = derive_kappa_dimensional(params)

    # Planck length squared
    ell_P_sq = L_PLANCK**2

    # The factor 8π comes from Einstein's equation
    # Additional factor accounts for boundary-bulk coupling
    #
    # CONJECTURE: The coupling is simply (ℓ_P²/8π) × κ_dim
    # This assumes boundary stress directly sources bulk metric
    # without additional suppression factors.

    lambda_bpr = (ell_P_sq / (8 * np.pi)) * kappa_dim

    return lambda_bpr


def derive_all_couplings(params: SubstrateParameters) -> DerivedCouplings:
    """
    Derive all coupling constants from substrate parameters.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    DerivedCouplings
        All derived coupling constants
    """
    return DerivedCouplings(
        kappa=derive_kappa(params),
        kappa_dimensional=derive_kappa_dimensional(params),
        xi=derive_correlation_length(params),
        lambda_bpr=derive_lambda_bpr(params)
    )


def verify_dimensional_consistency(params: SubstrateParameters) -> dict:
    """
    Verify that all derived quantities have correct dimensions.

    Parameters
    ----------
    params : SubstrateParameters
        Substrate configuration

    Returns
    -------
    dict
        Dimensional analysis results
    """
    couplings = derive_all_couplings(params)

    # Check dimensions by varying input scale
    R_test = params.radius
    params_2R = SubstrateParameters(
        p=params.p,
        N=params.N,
        J=params.J,
        geometry=params.geometry,
        radius=2 * R_test
    )
    couplings_2R = derive_all_couplings(params_2R)

    # κ (dimensionless) should not change with R
    kappa_scaling = couplings_2R.kappa / couplings.kappa

    # κ_dimensional [Energy] should not change with R
    kappa_dim_scaling = couplings_2R.kappa_dimensional / couplings.kappa_dimensional

    # ξ [Length] should scale as R (since a ~ R/√N)
    xi_scaling = couplings_2R.xi / couplings.xi

    # λ_BPR [Energy·Length²] should scale as...
    # ℓ_P² doesn't change, κ_dim doesn't change
    # So λ_BPR shouldn't change with R
    lambda_scaling = couplings_2R.lambda_bpr / couplings.lambda_bpr

    return {
        'kappa_scaling': kappa_scaling,  # Expected: 1.0
        'kappa_dim_scaling': kappa_dim_scaling,  # Expected: 1.0
        'xi_scaling': xi_scaling,  # Expected: 2.0 (linear in R)
        'lambda_scaling': lambda_scaling,  # Expected: 1.0
        'all_consistent': (
            np.isclose(kappa_scaling, 1.0, rtol=0.01) and
            np.isclose(kappa_dim_scaling, 1.0, rtol=0.01) and
            np.isclose(xi_scaling, 2.0, rtol=0.01) and
            np.isclose(lambda_scaling, 1.0, rtol=0.01)
        )
    }


# Convenience function for typical Casimir setup
def casimir_substrate_params(
    plate_radius: float = 0.01,  # 1 cm
    N: int = 10000,
    p: int = 104729,  # 10000th prime
    J_eV: float = 1.0  # Coupling in eV
) -> SubstrateParameters:
    """
    Create substrate parameters for Casimir geometry.

    Parameters
    ----------
    plate_radius : float
        Plate radius [m]
    N : int
        Number of substrate nodes
    p : int
        Prime modulus
    J_eV : float
        Coupling strength [eV]

    Returns
    -------
    SubstrateParameters
        Parameters for Casimir calculation
    """
    J_joules = J_eV * 1.602176634e-19  # eV to J

    return SubstrateParameters(
        p=p,
        N=N,
        J=J_joules,
        geometry=LatticeGeometry.SQUARE,  # Parallel plate approximation
        radius=plate_radius
    )


if __name__ == "__main__":
    print("RPST Boundary Energy Derivation")
    print("=" * 50)

    # Create test configuration
    params = casimir_substrate_params()

    print(f"\nSubstrate Parameters:")
    print(f"  p = {params.p}")
    print(f"  N = {params.N}")
    print(f"  J = {params.J:.3e} J = {params.J/1.602e-19:.3f} eV")
    print(f"  Geometry: {params.geometry.value}")
    print(f"  Radius: {params.radius*100:.1f} cm")
    print(f"  Lattice spacing: {params.lattice_spacing:.3e} m")
    print(f"  Coordination: z = {params.coordination_number}")

    # Derive couplings
    couplings = derive_all_couplings(params)

    print(f"\nDerived Couplings:")
    print(f"  κ (dimensionless) = {couplings.kappa:.4f}")
    print(f"  κ_dim = {couplings.kappa_dimensional:.3e} J")
    print(f"  ξ = {couplings.xi:.3e} m")
    print(f"  λ_BPR = {couplings.lambda_bpr:.3e} J·m²")

    # Verify dimensional consistency
    consistency = verify_dimensional_consistency(params)

    print(f"\nDimensional Consistency:")
    print(f"  κ scaling (expect 1.0): {consistency['kappa_scaling']:.4f}")
    print(f"  κ_dim scaling (expect 1.0): {consistency['kappa_dim_scaling']:.4f}")
    print(f"  ξ scaling (expect 2.0): {consistency['xi_scaling']:.4f}")
    print(f"  λ scaling (expect 1.0): {consistency['lambda_scaling']:.4f}")
    print(f"  All consistent: {consistency['all_consistent']}")

    # Compare to Planck scale
    print(f"\nComparison to Planck Scale:")
    print(f"  ℓ_P = {L_PLANCK:.3e} m")
    print(f"  ℓ_P² = {L_PLANCK**2:.3e} m²")
    print(f"  λ_BPR / ℓ_P² = {couplings.lambda_bpr / L_PLANCK**2:.3e} J")
    print(f"  This ratio = κ_dim / 8π = {couplings.kappa_dimensional / (8*np.pi):.3e} J")
