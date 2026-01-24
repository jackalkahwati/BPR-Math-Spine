"""
Electromagnetic Coupling Derivation

THE CRITICAL CALCULATION: What is g_φ, the coupling of boundary
phase to electromagnetic fields?

This determines whether BPR is testable:
- If g_φ ~ e (elementary charge): TESTABLE at lab scale
- If g_φ ~ e × (ℓ_P/ℓ_lab): Back to Planck suppression

Key insight from gauge symmetry:
- RPST has U(1) structure (proven in gauge_symmetry.py)
- The gauge connection A_ij = ∂φ can couple to photons
- But WHAT IS THE COEFFICIENT?

SPRINT: Week 2-3 of EM Coupling Search
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import scipy.constants as const

# Physical constants
HBAR = const.hbar
C = const.c
E_CHARGE = const.e
EPSILON_0 = const.epsilon_0
ALPHA = const.alpha  # Fine structure constant ~ 1/137
L_PLANCK = np.sqrt(const.hbar * const.G / const.c**3)


class CouplingScale(Enum):
    """Classification of coupling strength."""
    PLANCK_SUPPRESSED = "planck"      # g ~ e × (ℓ_P/ℓ)² ~ unmeasurable
    ELECTROWEAK = "electroweak"        # g ~ e × (ℓ_EW/ℓ) ~ potentially measurable
    FINE_STRUCTURE = "fine_structure"  # g ~ e × α ~ measurable
    FULL_CHARGE = "full_charge"        # g ~ e ~ strongly measurable


@dataclass
class EMCouplingResult:
    """Result of EM coupling derivation."""
    g_phi: float                      # Coupling constant [Coulombs]
    lambda_em: float                  # Characteristic length [meters]
    coupling_scale: CouplingScale     # Classification
    dimensionless_factor: float       # f in g_φ = e × f
    derivation_path: str              # Which mechanism
    is_testable: bool                 # At current experimental precision


def derive_coupling_from_gauge_kinetic_term(
    substrate_lattice_spacing: float,
    coordination_number: int,
    coupling_J: float
) -> Tuple[float, str]:
    """
    Attempt 1: Derive g_φ from gauge kinetic term normalization.

    The gauge kinetic term in RPST is:
        T = Σᵢⱼ J (1 - cos(∂ᵢφ))

    In continuum limit with small gradients:
        T → (J a² z / 2) |∇φ|²

    where a = lattice spacing, z = coordination number.

    For this to couple to EM, we need:
        L_interaction = g_φ (∂_μ φ) A^μ

    The natural scale for g_φ comes from matching dimensions.

    Parameters
    ----------
    substrate_lattice_spacing : float
        Lattice spacing a [meters]
    coordination_number : int
        z (number of neighbors)
    coupling_J : float
        Coupling energy J [Joules]

    Returns
    -------
    tuple
        (g_phi [Coulombs], derivation_notes)
    """
    a = substrate_lattice_spacing
    z = coordination_number
    J = coupling_J

    # The coefficient of |∇φ|² is:
    # κ = J × a² × z / 2

    # To couple to A_μ, we need [g_φ ∂φ A] = [Energy]
    # [g_φ] [1/Length] [Voltage × Length] = [Energy]
    # [g_φ] = [Charge]

    # The natural guess is g_φ = e (elementary charge)
    # But there could be a dimensionless suppression factor

    # Key question: Is φ normalized like a standard U(1) phase?

    # In QED: A_μ → A_μ + (ℏc/e) ∂_μ χ under gauge transformation
    # In RPST: φ → φ + α under gauge transformation

    # For consistent coupling:
    # g_φ ∂φ must match (ℏc/e) ∂χ when φ = χ

    # This gives: g_φ = e × (φ_scale / χ_scale)

    # If φ is in radians (as in our code) and χ is dimensionless:
    # g_φ = e (no suppression!)

    # BUT: φ comes from q_i ∈ ℤ_p via φ = 2πq/p
    # The range of φ is [0, 2π)
    # This is correct for a U(1) phase

    # CONCLUSION: g_φ = e (to leading order)

    # However, there may be a suppression from the substrate
    # The "stiffness" of the phase field matters

    # Stiffness: κ = J × a² × z / 2
    # Characteristic phase gradient: |∇φ|_typical ~ 1/ξ
    # where ξ is correlation length

    # The coupling involves how phase gradients couple to vacuum
    # This is where suppression could enter

    notes = f"""
    Gauge kinetic term: T = (J a² z / 2) |∇φ|²
    Lattice spacing: a = {a:.2e} m
    Coordination: z = {z}
    Coupling J = {J:.2e} J

    Stiffness: κ = {J * a**2 * z / 2:.2e} J·m²

    If φ is standard U(1) phase: g_φ = e = {E_CHARGE:.2e} C
    """

    # Leading order: g_φ = e
    g_phi = E_CHARGE

    return g_phi, notes


def derive_coupling_from_vacuum_polarization(
    separation: float,
    boundary_radius: float
) -> Tuple[float, str]:
    """
    Attempt 2: Derive g_φ from vacuum polarization.

    The QED vacuum is polarizable. Boundary conditions affect it.
    If φ acts like an external field, it induces vacuum polarization.

    The coupling can be estimated from:
        δε/ε₀ = α × (characteristic potential)

    where the characteristic potential involves φ.

    Parameters
    ----------
    separation : float
        Distance from boundary [meters]
    boundary_radius : float
        Boundary size [meters]

    Returns
    -------
    tuple
        (g_phi estimate, derivation_notes)
    """
    # Vacuum polarization in QED gives corrections of order α
    # The question is: how does φ enter?

    # If φ creates an effective potential V_eff = g_φ φ / ε₀
    # Then vacuum response: δε/ε₀ ~ α × (e V_eff / m_e c²)

    # For this to be measurable:
    # δε/ε₀ > 10^-6 (current polarimetry sensitivity)
    # α × (e × g_φ × φ / ε₀) / (m_e c²) > 10^-6

    # With φ ~ 1 radian, ε₀ ~ 10^-11 F/m, m_e c² ~ 0.5 MeV:
    # g_φ > 10^-6 × (0.5 MeV) × ε₀ / (α × e × 1)
    # g_φ > 10^-6 × (8 × 10^-14 J) × (10^-11 F/m) / (10^-2 × 1.6 × 10^-19 C)
    # g_φ > ...

    # This is getting complicated. Let me think differently.

    # The vacuum polarization diagram gives:
    # Effective Lagrangian: L_eff = α²/(90 m_e⁴) (F² F̃²)

    # If φ couples as φ F_μν F^μν, the coefficient must be:
    # [g_φ² / energy scale⁴]

    # Matching dimensions:
    # g_φ² / Λ⁴ ~ α² / m_e⁴
    # g_φ ~ α × (Λ² / m_e²) × m_e

    # If Λ is the boundary-vacuum coupling scale...
    # This is where the suppression could enter.

    # Most optimistic: Λ ~ m_e, giving g_φ ~ α × m_e ~ small but not Planck

    m_e_energy = const.m_e * C**2  # Electron mass energy

    # Estimate: g_φ ~ e × α (one loop suppression)
    g_phi_estimate = E_CHARGE * ALPHA

    notes = f"""
    Vacuum polarization approach:
    One-loop QED gives corrections ~ α²
    If boundary phase couples at one-loop: g_φ ~ e × α

    Estimate: g_φ ~ {g_phi_estimate:.2e} C
    This is {ALPHA:.2e} times smaller than elementary charge
    Characteristic length: λ_em = ℏc/g_φ ~ {HBAR * C / g_phi_estimate:.2e} m
    """

    return g_phi_estimate, notes


def derive_coupling_from_topological_term(
    winding_number: int,
    boundary_area: float
) -> Tuple[float, str]:
    """
    Attempt 3: Derive g_φ from topological (Chern-Simons) coupling.

    Topological couplings are special because they're:
    - Quantized (no continuous parameters)
    - Independent of scale
    - Protected by topology

    If BPR has topological EM coupling:
        L_top = (θ / 2π) × (e² / 2πℏ) × E·B

    where θ = ∮ ∇φ·dl is winding number.

    Parameters
    ----------
    winding_number : int
        Topological charge
    boundary_area : float
        Boundary area [m²]

    Returns
    -------
    tuple
        (effective_g_phi, derivation_notes)
    """
    # The θ term in QED:
    # L_θ = (α/π) θ E·B

    # If θ = winding number × 2π:
    # L_θ = 2α × W × E·B

    # This doesn't directly give g_φ, but it tells us:
    # - Topological couplings involve α (fine structure)
    # - They're quantized in units of 2π
    # - The effect is NOT Planck suppressed!

    # Effective coupling from θ-term perspective:
    # g_eff ~ α × e (the coefficient in front)

    g_phi_topological = ALPHA * E_CHARGE

    notes = f"""
    Topological (Chern-Simons) approach:
    L_θ = (α/π) θ E·B where θ = 2π × winding

    For W = {winding_number}:
    Effective phase: θ = {2 * np.pi * winding_number:.4f}

    Topological coupling: g_eff ~ α × e = {g_phi_topological:.2e} C
    Characteristic length: λ_top = ℏc/g ~ {HBAR * C / g_phi_topological:.2e} m

    KEY: Topological coupling is NOT Planck suppressed!
    Scale is set by α, not ℓ_P.
    """

    return g_phi_topological, notes


def derive_coupling_comprehensive(
    substrate_params: dict,
    experimental_params: dict
) -> EMCouplingResult:
    """
    Comprehensive derivation attempting all approaches.

    Returns the most conservative (smallest) coupling that's
    still physically motivated.

    Parameters
    ----------
    substrate_params : dict
        {'lattice_spacing': float, 'coordination': int, 'coupling_J': float}
    experimental_params : dict
        {'separation': float, 'boundary_radius': float}

    Returns
    -------
    EMCouplingResult
        Complete coupling analysis
    """
    results = []

    # Attempt 1: Gauge kinetic term
    g1, notes1 = derive_coupling_from_gauge_kinetic_term(
        substrate_params.get('lattice_spacing', 1e-9),
        substrate_params.get('coordination', 4),
        substrate_params.get('coupling_J', const.e)  # Use eV scale
    )
    results.append(('gauge_kinetic', g1, notes1))

    # Attempt 2: Vacuum polarization
    g2, notes2 = derive_coupling_from_vacuum_polarization(
        experimental_params.get('separation', 100e-9),
        experimental_params.get('boundary_radius', 0.01)
    )
    results.append(('vacuum_polarization', g2, notes2))

    # Attempt 3: Topological
    g3, notes3 = derive_coupling_from_topological_term(
        winding_number=1,
        boundary_area=np.pi * experimental_params.get('boundary_radius', 0.01)**2
    )
    results.append(('topological', g3, notes3))

    # Choose the derivation path
    # Conservative: Use smallest physically motivated value
    # That's vacuum polarization or topological (both ~ α × e)

    # But the honest answer is: we don't know which is right
    # Let's report all three with their implications

    # For now, use the topological value (it's well-defined)
    g_phi = g3
    derivation = 'topological'

    # Characteristic length
    lambda_em = HBAR * C / g_phi

    # Dimensionless factor
    f = g_phi / E_CHARGE

    # Classification
    if f < 1e-30:
        scale = CouplingScale.PLANCK_SUPPRESSED
    elif f < 1e-10:
        scale = CouplingScale.ELECTROWEAK
    elif f < 0.1:
        scale = CouplingScale.FINE_STRUCTURE
    else:
        scale = CouplingScale.FULL_CHARGE

    # Is it testable?
    # Current vacuum birefringence sensitivity: δn ~ 10^-10
    # For g ~ α × e, effect size δn ~ g²/(ε₀ ℏ c) × (boundary factor)
    # Need detailed calculation...
    is_testable = (scale != CouplingScale.PLANCK_SUPPRESSED)

    return EMCouplingResult(
        g_phi=g_phi,
        lambda_em=lambda_em,
        coupling_scale=scale,
        dimensionless_factor=f,
        derivation_path=derivation,
        is_testable=is_testable
    )


def compute_birefringence_prediction(
    g_phi: float,
    phase_gradient: float,
    path_length: float
) -> float:
    """
    Compute predicted vacuum birefringence from BPR.

    Δn = (g_φ / ε₀ ℏ c) × |∇φ| × (geometry factor)

    Parameters
    ----------
    g_phi : float
        Coupling constant [C]
    phase_gradient : float
        Typical |∇φ| [rad/m]
    path_length : float
        Optical path [m]

    Returns
    -------
    float
        Refractive index change Δn
    """
    # The refractive index change from vacuum polarization is:
    # Δn ~ (α / 3π) × (external field / critical field)²

    # If BPR creates effective field F_eff ~ g_φ × |∇φ|:
    # Δn ~ (α / 3π) × (g_φ |∇φ| / E_crit)²

    # Critical field E_crit = m_e² c³ / (e ℏ) ~ 10^18 V/m

    E_crit = const.m_e**2 * C**3 / (E_CHARGE * HBAR)

    # Effective field from boundary phase gradient
    # [g_φ] = C, [∇φ] = 1/m
    # For dimensional consistency, need:
    # F_eff = g_φ × |∇φ| × (1/ε₀) ~ V/m²... this doesn't work

    # Better approach: Use the coupling directly
    # If the phase enters as a scalar potential:
    # V_eff = (ℏc/g_φ) × φ

    # Then electric field analog:
    # E_eff = (ℏc/g_φ) × |∇φ|

    E_eff = (HBAR * C / g_phi) * phase_gradient

    # Birefringence:
    delta_n = (ALPHA / (3 * np.pi)) * (E_eff / E_crit)**2

    return delta_n


def compute_aharonov_bohm_prediction(
    g_phi: float,
    winding_number: int
) -> float:
    """
    Compute predicted Aharonov-Bohm phase from BPR.

    Δφ_AB = (g_φ / ℏ) × ∮ A_eff · dl

    If A_eff comes from boundary phase winding:
        A_eff ~ (ℏ/g_φ) × ∇φ
        ∮ A_eff · dl = (ℏ/g_φ) × 2π × W

    Parameters
    ----------
    g_phi : float
        Coupling constant [C]
    winding_number : int
        Topological winding

    Returns
    -------
    float
        Phase shift [radians]
    """
    # Aharonov-Bohm phase:
    # Δφ = (e/ℏ) × ∮ A · dl = (e/ℏ) × Φ_B

    # If boundary phase creates effective flux:
    # Φ_eff = (ℏ/g_φ) × 2π × W

    # Then:
    # Δφ_AB = (e/ℏ) × (ℏ/g_φ) × 2π × W = (e/g_φ) × 2π × W

    delta_phi = (E_CHARGE / g_phi) * 2 * np.pi * winding_number

    return delta_phi


def summary_em_coupling():
    """Print summary of EM coupling analysis."""
    print("=" * 70)
    print("ELECTROMAGNETIC COUPLING ANALYSIS - BPR DIRECT CHANNEL")
    print("=" * 70)

    # Use typical parameters
    substrate = {
        'lattice_spacing': 1e-9,  # 1 nm
        'coordination': 4,
        'coupling_J': const.e  # 1 eV
    }
    experimental = {
        'separation': 100e-9,  # 100 nm
        'boundary_radius': 0.01  # 1 cm
    }

    result = derive_coupling_comprehensive(substrate, experimental)

    print(f"\nDerived coupling constant:")
    print(f"  g_φ = {result.g_phi:.3e} C")
    print(f"  g_φ / e = {result.dimensionless_factor:.3e}")
    print(f"  λ_EM = {result.lambda_em:.3e} m")
    print(f"  Coupling scale: {result.coupling_scale.value}")
    print(f"  Derivation path: {result.derivation_path}")
    print(f"  Testable at current precision: {result.is_testable}")

    print(f"\n{'='*70}")
    print("PREDICTED OBSERVABLE EFFECTS")
    print(f"{'='*70}")

    # Birefringence
    phase_grad = 1e3  # 1 rad/mm
    path = 0.1  # 10 cm
    delta_n = compute_birefringence_prediction(result.g_phi, phase_grad, path)
    print(f"\nVacuum birefringence:")
    print(f"  Phase gradient: {phase_grad:.0e} rad/m")
    print(f"  Path length: {path*100:.0f} cm")
    print(f"  Δn = {delta_n:.3e}")
    print(f"  Current sensitivity: ~10^-10")
    if delta_n > 1e-10:
        print(f"  STATUS: DETECTABLE")
    elif delta_n > 1e-15:
        print(f"  STATUS: Near-term detectable with improvements")
    else:
        print(f"  STATUS: Not detectable")

    # Aharonov-Bohm
    W = 1
    delta_phi_ab = compute_aharonov_bohm_prediction(result.g_phi, W)
    print(f"\nAharonov-Bohm phase (W={W}):")
    print(f"  Δφ_AB = {delta_phi_ab:.3e} rad = {delta_phi_ab/(2*np.pi):.3e} × 2π")
    print(f"  Current electron holography: ~0.01 rad")
    if abs(delta_phi_ab) > 0.01:
        print(f"  STATUS: DETECTABLE")
    elif abs(delta_phi_ab) > 1e-6:
        print(f"  STATUS: Near-term detectable")
    else:
        print(f"  STATUS: Not detectable")

    print(f"\n{'='*70}")
    print("CRITICAL ASSESSMENT")
    print(f"{'='*70}")

    if result.coupling_scale == CouplingScale.PLANCK_SUPPRESSED:
        print("\n❌ PLANCK SUPPRESSED")
        print("   EM channel gives same result as gravitational.")
        print("   BPR is cosmology-only theory.")
    elif result.coupling_scale == CouplingScale.ELECTROWEAK:
        print("\n⚠️  ELECTROWEAK SCALE")
        print("   Better than Planck but still very small.")
        print("   May be testable with significant improvements.")
    elif result.coupling_scale == CouplingScale.FINE_STRUCTURE:
        print("\n✓ FINE STRUCTURE SCALE")
        print("   Coupling ~ α × e")
        print("   Effects scale as α² ~ 10^-5")
        print("   POTENTIALLY TESTABLE with precision experiments.")
    else:
        print("\n✓✓ FULL CHARGE SCALE")
        print("   Coupling ~ e")
        print("   Effects should be easily measurable.")
        print("   (This would be surprising...)")


if __name__ == "__main__":
    summary_em_coupling()
