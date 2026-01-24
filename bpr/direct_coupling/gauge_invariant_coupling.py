"""
Gauge-Invariant EM Coupling

CRITICAL CORRECTION: Electrons don't couple to φ directly.
They couple to GAUGE-INVARIANT quantities built from φ.

The gauge-invariant content of a U(1) phase field:
1. Curvature: F_ij = ∂_i A_j - ∂_j A_i (for A ~ ∇φ: F = 0 if smooth!)
2. Topological charge: W = (1/2π) ∮ ∇φ · dl (winding number)
3. Phase gradients squared: (∇φ)² (gauge invariant)

For smooth boundary phases: F = 0, W = 0
→ NO A-B effect from smooth phases!

Observable effects require:
- Vortices (point defects with W ≠ 0)
- Boundaries where φ is discontinuous
- Interfaces between regions with different φ

This explains why we don't see huge phase shifts everywhere.

SPRINT: Week 3 of EM Coupling Search
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import scipy.constants as const

HBAR = const.hbar
C = const.c
E_CHARGE = const.e
ALPHA = const.alpha


@dataclass
class GaugeInvariantSource:
    """Sources that can couple gauge-invariantly."""
    curvature_density: float  # ∫ F² d²x
    vortex_count: int
    total_winding: int
    gradient_energy: float  # ∫ (∇φ)² d²x


@dataclass
class CorrectedEMPrediction:
    """Corrected EM prediction using gauge-invariant coupling."""
    mechanism: str
    effect_size: float
    detectable: bool
    explanation: str


def compute_curvature_from_phase(phi: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute gauge curvature from phase field.

    For a 2D phase field:
        F_12 = ∂_x A_y - ∂_y A_x

    where A ~ ∇φ.

    For smooth φ: F = 0 everywhere
    For vortex at origin: F has delta-function spike

    Parameters
    ----------
    phi : np.ndarray
        2D phase field [N, N]
    dx : float
        Grid spacing

    Returns
    -------
    np.ndarray
        Curvature field F(x, y)
    """
    # A = ∇φ (up to constant)
    # A_x = ∂φ/∂x, A_y = ∂φ/∂y

    # F = ∂_x A_y - ∂_y A_x = ∂_x ∂_y φ - ∂_y ∂_x φ

    # For smooth φ: mixed partials equal, so F = 0

    # But for singular φ (vortex), F ≠ 0 at singularity

    # Compute numerically
    N = phi.shape[0]
    F = np.zeros((N-1, N-1))

    for i in range(N-1):
        for j in range(N-1):
            # A_x at (i+1/2, j)
            Ax_ij = (phi[i+1, j] - phi[i, j]) / dx
            Ax_ij1 = (phi[i+1, j+1] - phi[i, j+1]) / dx

            # A_y at (i, j+1/2)
            Ay_ij = (phi[i, j+1] - phi[i, j]) / dx
            Ay_i1j = (phi[i+1, j+1] - phi[i+1, j]) / dx

            # F = ∂_x A_y - ∂_y A_x
            d_x_Ay = (Ay_i1j - Ay_ij) / dx
            d_y_Ax = (Ax_ij1 - Ax_ij) / dx

            F[i, j] = d_x_Ay - d_y_Ax

    return F


def count_vortices(phi: np.ndarray, dx: float) -> Tuple[int, int]:
    """
    Count vortices in phase field.

    A vortex is where the winding around a plaquette is ±2π.

    Parameters
    ----------
    phi : np.ndarray
        2D phase field
    dx : float
        Grid spacing

    Returns
    -------
    tuple
        (n_positive, n_negative) vortex counts
    """
    N = phi.shape[0]
    n_pos = 0
    n_neg = 0

    for i in range(N-1):
        for j in range(N-1):
            # Winding around plaquette (i,j) → (i+1,j) → (i+1,j+1) → (i,j+1) → (i,j)
            dphi1 = phi[i+1, j] - phi[i, j]
            dphi2 = phi[i+1, j+1] - phi[i+1, j]
            dphi3 = phi[i, j+1] - phi[i+1, j+1]
            dphi4 = phi[i, j] - phi[i, j+1]

            # Wrap each to [-π, π]
            dphi1 = np.arctan2(np.sin(dphi1), np.cos(dphi1))
            dphi2 = np.arctan2(np.sin(dphi2), np.cos(dphi2))
            dphi3 = np.arctan2(np.sin(dphi3), np.cos(dphi3))
            dphi4 = np.arctan2(np.sin(dphi4), np.cos(dphi4))

            winding = dphi1 + dphi2 + dphi3 + dphi4

            if winding > np.pi:
                n_pos += 1
            elif winding < -np.pi:
                n_neg += 1

    return n_pos, n_neg


def create_vortex_field(N: int, vortex_pos: Tuple[float, float],
                        charge: int = 1) -> np.ndarray:
    """
    Create a phase field with a vortex.

    φ(r) = charge × arg(z - z_vortex)

    Parameters
    ----------
    N : int
        Grid size
    vortex_pos : tuple
        (x, y) of vortex center (in grid units)
    charge : int
        Vortex charge (+1 or -1)

    Returns
    -------
    np.ndarray
        Phase field [N, N]
    """
    phi = np.zeros((N, N))
    x0, y0 = vortex_pos

    for i in range(N):
        for j in range(N):
            dx = i - x0
            dy = j - y0
            if dx != 0 or dy != 0:
                phi[i, j] = charge * np.arctan2(dy, dx)
            else:
                phi[i, j] = 0  # Undefined at core

    return phi


def ab_phase_from_vortex(vortex_charge: int, g_phi: float) -> float:
    """
    A-B phase from passing around a vortex.

    For a vortex with charge W, the enclosed "flux" is:
        Φ = (ℏ/g_φ) × 2π × W

    The A-B phase is:
        Δφ = (e/ℏ) × Φ = (e/g_φ) × 2π × W

    Parameters
    ----------
    vortex_charge : int
        Vortex winding number
    g_phi : float
        Coupling constant

    Returns
    -------
    float
        A-B phase in radians
    """
    return (E_CHARGE / g_phi) * 2 * np.pi * vortex_charge


def birefringence_from_gradient_squared(
    grad_phi_squared: float,
    area: float,
    path_length: float,
    g_phi: float
) -> float:
    """
    Birefringence from phase gradient energy.

    The gauge-invariant observable is (∇φ)².
    This couples to photons through vacuum polarization.

    δn ~ (g_φ² / ε₀ ℏ² c²) × ⟨(∇φ)²⟩ × (area/path²)

    Parameters
    ----------
    grad_phi_squared : float
        Average |∇φ|² [rad²/m²]
    area : float
        Boundary area [m²]
    path_length : float
        Optical path [m]
    g_phi : float
        Coupling constant [C]

    Returns
    -------
    float
        Refractive index change
    """
    eps0 = const.epsilon_0

    # This is a dimensional estimate
    # The actual coupling would come from QFT calculation
    # Here we use the scale set by g_φ

    # Energy scale from coupling: E_scale = g_φ² / (4πε₀ a)
    # where a is some length scale (Bohr radius?)

    a_Bohr = const.physical_constants['Bohr radius'][0]
    E_scale = g_phi**2 / (4 * np.pi * eps0 * a_Bohr)

    # δn ~ (E_scale / E_photon) × (gradient factor)
    # For optical photon: E_photon ~ 1 eV
    E_photon = const.e

    delta_n = (E_scale / E_photon) * grad_phi_squared * area / path_length**2

    return delta_n


def corrected_em_predictions(g_phi: float = ALPHA * E_CHARGE):
    """
    Corrected predictions using gauge-invariant coupling.
    """
    print("=" * 70)
    print("CORRECTED EM PREDICTIONS (GAUGE-INVARIANT)")
    print("=" * 70)

    print(f"\nCoupling: g_φ = {g_phi:.3e} C")
    print(f"         g_φ / e = {g_phi/E_CHARGE:.3e} (= α)")

    print(f"\n{'='*70}")
    print("AHARONOV-BOHM FROM VORTICES")
    print(f"{'='*70}")

    print("""
CRITICAL POINT: Smooth phase fields give ZERO A-B effect!
Gauge invariance means F = 0 for smooth φ.

A-B effect requires VORTICES (topological defects).
Each vortex with charge W gives:
    Δφ_AB = (e/g_φ) × 2π × W = (1/α) × 2π × W ≈ 137 × 2π × W
""")

    for W in [1, -1]:
        delta_phi = ab_phase_from_vortex(W, g_phi)
        print(f"  Vortex W={W:+d}: Δφ_AB = {delta_phi:.1f} rad = {delta_phi/(2*np.pi):.1f} × 2π")

    print("""
But vortices are:
- Rare in thermal equilibrium (energy cost ~ J × W²)
- Localized (nm-scale core)
- Hard to probe with electrons (need nm-resolution)

CONCLUSION: A-B from BPR vortices is THEORETICALLY LARGE
            but PRACTICALLY DIFFICULT to observe.
""")

    print(f"\n{'='*70}")
    print("BIREFRINGENCE FROM PHASE GRADIENTS")
    print(f"{'='*70}")

    print("""
The gauge-invariant observable is (∇φ)².
This couples to photons and causes birefringence.

Unlike A-B, this doesn't require vortices - any spatial
variation in φ contributes.
""")

    # Estimate gradient energy in typical system
    # For thermal boundary: |∇φ|² ~ (kT/J) / a² where a is lattice spacing

    a = 1e-9  # 1 nm
    J = const.e  # 1 eV
    T = 300  # Room temperature

    grad_phi_sq_thermal = (const.k * T / J) / a**2
    print(f"  Thermal gradient: |∇φ|² ~ {grad_phi_sq_thermal:.3e} rad²/m²")

    # Birefringence estimate
    area = (1e-2)**2  # 1 cm²
    path = 0.1  # 10 cm

    delta_n = birefringence_from_gradient_squared(
        grad_phi_sq_thermal, area, path, g_phi
    )

    print(f"  Boundary area: {np.sqrt(area)*100:.0f} cm × {np.sqrt(area)*100:.0f} cm")
    print(f"  Path length: {path*100:.0f} cm")
    print(f"  Predicted Δn: {delta_n:.3e}")
    print(f"  Current sensitivity: ~10^-10")

    if delta_n > 1e-10:
        print(f"  STATUS: DETECTABLE")
    elif delta_n > 1e-15:
        print(f"  STATUS: Potentially detectable with improvements")
    else:
        print(f"  STATUS: Not detectable (need {1e-10/delta_n:.0e}× improvement)")

    print(f"\n{'='*70}")
    print("SUMMARY: WHERE TO LOOK")
    print(f"{'='*70}")

    print("""
1. VORTEX A-B EFFECT
   - Large in principle (137 × 2π per vortex)
   - Requires: Creating/finding vortices
   - Requires: Electron holography with nm resolution near vortex
   - Challenge: Vortices are rare and localized

2. GRADIENT BIREFRINGENCE
   - Small effect (gauge-invariant suppression)
   - Requires: Very sensitive polarimetry
   - Advantage: Works with any boundary, no special structure needed
   - Challenge: Effect too small for current tech

3. CASIMIR (REVISITED)
   - The EM coupling also affects Casimir forces
   - Coupling g ~ α × e instead of gravitational λ ~ ℓ_P²
   - Need to redo Casimir calculation with EM mechanism

4. CAVITY QED
   - Photon modes in cavity couple to boundary phases
   - Could show up as cavity frequency shifts
   - Worth investigating...
""")

    print(f"\n{'='*70}")
    print("CRITICAL QUESTION: CASIMIR WITH EM COUPLING")
    print(f"{'='*70}")

    print("""
The gravitational Casimir gave ΔF/F ~ 10^-94 (Planck suppressed).

What if the Casimir effect couples through EM channel?

New mechanism:
    Boundary phase φ → Effective permittivity ε(x)
    ε(x) modifies photon modes
    Modified modes → Modified Casimir force

Scaling:
    If ε(φ) = ε₀(1 + α × F[φ])
    Then δF/F ~ α × (boundary factor)
           ~ 10^-2 × (TBD)

This could be MUCH larger than gravitational channel!

NEXT STEP: Derive Casimir with EM coupling mechanism.
""")


def investigate_em_casimir():
    """
    Preliminary investigation of Casimir through EM channel.
    """
    print(f"\n{'='*70}")
    print("EM CASIMIR COUPLING - PRELIMINARY")
    print(f"{'='*70}")

    # If boundary phase modifies ε:
    # ε(φ) = ε₀(1 + δε/ε₀)
    # where δε/ε₀ ~ α × (phase factor)

    # Casimir force scales as:
    # F ~ ℏc/a⁴ × (ε factor)

    # The ε factor for metallic plates is ~1
    # BPR modification: δF/F ~ δε/ε

    # Phase factor: depends on |∇φ|² at boundary
    # For thermal boundary: |∇φ|² ~ kT/(J × a²)

    a_lattice = 1e-9
    J = const.e
    T = 300
    grad_phi_sq = (const.k * T / J) / a_lattice**2

    # Dimensionless combination
    # α × (grad_phi)² × (some length)²

    # The "some length" is where the physics enters
    # Could be: a (separation), λ_photon, a_lattice

    # Most optimistic: use a_lattice
    phase_factor = grad_phi_sq * a_lattice**2
    delta_F_over_F_optimistic = ALPHA * phase_factor

    print(f"\nOptimistic estimate (using lattice scale):")
    print(f"  Phase factor: {phase_factor:.3e}")
    print(f"  δF/F ~ α × (phase factor) = {delta_F_over_F_optimistic:.3e}")

    # More realistic: use separation a
    a_sep = 100e-9
    phase_factor_realistic = grad_phi_sq * a_sep**2
    delta_F_over_F_realistic = ALPHA * phase_factor_realistic

    print(f"\nRealistic estimate (using separation):")
    print(f"  Separation: {a_sep*1e9:.0f} nm")
    print(f"  Phase factor: {phase_factor_realistic:.3e}")
    print(f"  δF/F ~ α × (phase factor) = {delta_F_over_F_realistic:.3e}")

    print(f"\nExperimental precision: 10^-3")

    if delta_F_over_F_realistic > 1e-3:
        print(f"STATUS: DETECTABLE!")
    elif delta_F_over_F_realistic > 1e-6:
        print(f"STATUS: Near future detectable")
    else:
        print(f"STATUS: Need {1e-3/delta_F_over_F_realistic:.0e}× improvement")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"""
EM coupling gives δF/F ~ {delta_F_over_F_realistic:.0e}

This is MUCH BETTER than gravitational (10^-94)!
But still below current precision (10^-3).

Gap: {1e-3/delta_F_over_F_realistic:.0e}×

PATH FORWARD:
1. Derive δε(φ) from first principles
2. Compute Casimir with modified ε
3. Check if structured boundaries enhance effect
4. Look for geometries that maximize |∇φ|²
""")


if __name__ == "__main__":
    corrected_em_predictions()
    investigate_em_casimir()
