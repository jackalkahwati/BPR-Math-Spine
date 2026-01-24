"""
Collective Mode Coupling Analysis

Key insight: Collective modes have MUCH lower characteristic energies
than QED vacuum (E_crit ~ 10^18 V/m).

If BPR couples to collective modes, the suppression could be:
    (E_eff / E_collective)² instead of (E_eff / E_crit)²

Potential gain: (E_crit / E_collective)² ~ 10^10 to 10^14

Modes to investigate:
1. Phonons (lattice vibrations) - E ~ 10-100 meV
2. Plasmons (electron density waves) - E ~ 1-10 eV
3. Magnons (spin waves) - E ~ 1-100 meV
4. Polaritons (light-matter hybrids) - E ~ 0.1-1 eV

SPRINT: Week 4-5 of Coupling Search
"""

import numpy as np
import scipy.constants as const
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

# Constants
HBAR = const.hbar
C = const.c
E_CHARGE = const.e
K_B = const.k
M_E = const.m_e
EPSILON_0 = const.epsilon_0


class CollectiveMode(Enum):
    """Types of collective excitations."""
    PHONON = "phonon"
    PLASMON = "plasmon"
    MAGNON = "magnon"
    POLARITON = "polariton"


@dataclass
class ModeParameters:
    """Parameters for a collective mode."""
    mode_type: CollectiveMode
    characteristic_energy: float  # [J]
    characteristic_length: float  # [m]
    quality_factor: float  # Q (for resonance enhancement)
    description: str


@dataclass
class CollectiveCouplingResult:
    """Result of collective mode coupling analysis."""
    mode: CollectiveMode
    coupling_strength: float  # dimensionless
    effect_size: float  # δF/F or equivalent
    enhancement_over_qed: float  # vs vacuum polarization
    is_testable: bool
    required_precision: float


def phonon_parameters(material: str = 'silicon') -> ModeParameters:
    """
    Get phonon parameters for common materials.

    Parameters
    ----------
    material : str
        'silicon', 'gold', 'diamond'

    Returns
    -------
    ModeParameters
    """
    # Debye energies (approximate)
    debye_energies = {
        'silicon': 64e-3 * E_CHARGE,   # 64 meV
        'gold': 15e-3 * E_CHARGE,      # 15 meV
        'diamond': 192e-3 * E_CHARGE,  # 192 meV
        'sio2': 50e-3 * E_CHARGE,      # 50 meV (amorphous)
    }

    # Lattice constants
    lattice_constants = {
        'silicon': 5.43e-10,  # m
        'gold': 4.08e-10,
        'diamond': 3.57e-10,
        'sio2': 5e-10,  # approximate
    }

    # Quality factors (at low T)
    quality_factors = {
        'silicon': 1e6,   # High Q at cryogenic
        'gold': 1e4,      # Lower due to electrons
        'diamond': 1e7,   # Very high Q
        'sio2': 1e5,
    }

    E_D = debye_energies.get(material, 50e-3 * E_CHARGE)
    a = lattice_constants.get(material, 5e-10)
    Q = quality_factors.get(material, 1e5)

    return ModeParameters(
        mode_type=CollectiveMode.PHONON,
        characteristic_energy=E_D,
        characteristic_length=a,
        quality_factor=Q,
        description=f"Acoustic phonons in {material}"
    )


def plasmon_parameters(material: str = 'gold') -> ModeParameters:
    """
    Get plasmon parameters for metals.

    Plasma frequency: ω_p = √(n e² / ε₀ m_e)

    Parameters
    ----------
    material : str
        'gold', 'silver', 'aluminum'

    Returns
    -------
    ModeParameters
    """
    # Electron densities [m^-3]
    electron_densities = {
        'gold': 5.9e28,
        'silver': 5.86e28,
        'aluminum': 18.1e28,
    }

    # Plasma frequencies
    n_e = electron_densities.get(material, 5.9e28)
    omega_p = np.sqrt(n_e * E_CHARGE**2 / (EPSILON_0 * M_E))
    E_p = HBAR * omega_p

    # Characteristic length (skin depth at plasma frequency)
    lambda_p = C / omega_p

    # Quality factor (limited by electron scattering)
    # At room temp, τ ~ 10^-14 s, so Q ~ ω_p τ ~ 100
    Q = 100  # Typical for room temp metals

    return ModeParameters(
        mode_type=CollectiveMode.PLASMON,
        characteristic_energy=E_p,
        characteristic_length=lambda_p,
        quality_factor=Q,
        description=f"Surface plasmons in {material}"
    )


def magnon_parameters(material: str = 'yig') -> ModeParameters:
    """
    Get magnon parameters for magnetic materials.

    YIG (Yttrium Iron Garnet) is the gold standard for magnons.

    Parameters
    ----------
    material : str
        'yig', 'iron', 'nickel'

    Returns
    -------
    ModeParameters
    """
    # Magnon energies (typical)
    magnon_energies = {
        'yig': 10e-3 * E_CHARGE,    # 10 meV (ferromagnetic resonance)
        'iron': 50e-3 * E_CHARGE,    # 50 meV
        'nickel': 30e-3 * E_CHARGE,  # 30 meV
    }

    # Exchange lengths
    exchange_lengths = {
        'yig': 17e-9,    # 17 nm
        'iron': 3.4e-9,  # 3.4 nm
        'nickel': 7.6e-9,  # 7.6 nm
    }

    # Quality factors (YIG is exceptional)
    quality_factors = {
        'yig': 1e4,    # Very high for magnetic materials
        'iron': 100,
        'nickel': 100,
    }

    E_m = magnon_energies.get(material, 10e-3 * E_CHARGE)
    l_ex = exchange_lengths.get(material, 10e-9)
    Q = quality_factors.get(material, 100)

    return ModeParameters(
        mode_type=CollectiveMode.MAGNON,
        characteristic_energy=E_m,
        characteristic_length=l_ex,
        quality_factor=Q,
        description=f"Magnons in {material}"
    )


def polariton_parameters(system: str = 'microcavity') -> ModeParameters:
    """
    Get polariton parameters.

    Polaritons are hybrid light-matter modes.
    Strong coupling in microcavities can give large effects.

    Parameters
    ----------
    system : str
        'microcavity', 'phonon_polariton'

    Returns
    -------
    ModeParameters
    """
    if system == 'microcavity':
        # Exciton-polaritons in semiconductor microcavity
        E = 1.5 * E_CHARGE  # ~1.5 eV (visible)
        L = 1e-6  # 1 μm cavity
        Q = 1e5  # High-Q cavity
        desc = "Exciton-polaritons in microcavity"
    else:
        # Phonon-polaritons in polar crystals
        E = 50e-3 * E_CHARGE  # 50 meV (mid-IR)
        L = 10e-6  # 10 μm
        Q = 1e3
        desc = "Phonon-polaritons"

    return ModeParameters(
        mode_type=CollectiveMode.POLARITON,
        characteristic_energy=E,
        characteristic_length=L,
        quality_factor=Q,
        description=desc
    )


def thermal_phase_gradient(T: float, J: float, a: float) -> float:
    """Thermal phase gradient from substrate."""
    return np.sqrt(K_B * T / J) / a


def effective_field_from_gradient(grad_phi: float, length_scale: float) -> float:
    """Effective 'field' from phase gradient."""
    return (HBAR * C / length_scale) * grad_phi


def coupling_to_collective_mode(
    grad_phi: float,
    mode_params: ModeParameters,
    coupling_length: float
) -> CollectiveCouplingResult:
    """
    Compute BPR coupling to a collective mode.

    The key difference from QED vacuum:
    - Characteristic energy is E_mode, not m_e c²
    - Coupling can be enhanced by Q factor
    - Resonance effects possible

    Parameters
    ----------
    grad_phi : float
        Phase gradient [rad/m]
    mode_params : ModeParameters
        Collective mode parameters
    coupling_length : float
        Length scale for phase-mode coupling [m]

    Returns
    -------
    CollectiveCouplingResult
    """
    # Effective "field" from boundary phases
    E_eff = effective_field_from_gradient(grad_phi, coupling_length)

    # Characteristic field for this mode
    # E_char = E_mode / (e × λ_mode)
    # This is the field that would excite one quantum
    E_char = mode_params.characteristic_energy / (E_CHARGE * mode_params.characteristic_length)

    # Base coupling strength
    coupling_base = (E_eff / E_char)**2

    # Q-factor enhancement (if on resonance)
    # Off resonance: factor = 1
    # On resonance: factor = Q
    # We'll be optimistic and use Q
    coupling_enhanced = coupling_base * mode_params.quality_factor

    # For comparison to QED
    E_crit_qed = M_E**2 * C**3 / (E_CHARGE * HBAR)
    coupling_qed = (E_eff / E_crit_qed)**2
    enhancement = coupling_enhanced / coupling_qed if coupling_qed > 0 else np.inf

    # Is it testable?
    # For phonons: measure frequency shift δω/ω
    # Precision ~ 10^-9 (best atomic clocks), 10^-6 (good lab equipment)
    precision_available = 1e-6
    is_testable = coupling_enhanced > precision_available

    return CollectiveCouplingResult(
        mode=mode_params.mode_type,
        coupling_strength=coupling_enhanced,
        effect_size=coupling_enhanced,  # δω/ω or δF/F
        enhancement_over_qed=enhancement,
        is_testable=is_testable,
        required_precision=coupling_enhanced
    )


def analyze_all_collective_modes():
    """Comprehensive analysis of all collective mode channels."""
    print("=" * 70)
    print("COLLECTIVE MODE COUPLING ANALYSIS")
    print("=" * 70)

    # Substrate parameters
    T = 300  # K (room temperature)
    J = E_CHARGE  # 1 eV coupling
    a = 1e-9  # 1 nm lattice spacing

    grad_phi = thermal_phase_gradient(T, J, a)
    print(f"\nThermal phase gradient: |∇φ| = {grad_phi:.3e} rad/m")

    # Coupling length - we'll try different scales
    coupling_lengths = {
        'lattice': a,
        'nm': 1e-9,
        'μm': 1e-6,
    }

    print(f"\n{'='*70}")
    print("PHONON COUPLING")
    print(f"{'='*70}")

    for material in ['silicon', 'diamond', 'gold']:
        params = phonon_parameters(material)
        print(f"\n{params.description}:")
        print(f"  Debye energy: {params.characteristic_energy/E_CHARGE*1e3:.1f} meV")
        print(f"  Quality factor: {params.quality_factor:.0e}")

        for L_name, L in coupling_lengths.items():
            result = coupling_to_collective_mode(grad_phi, params, L)
            status = "TESTABLE" if result.is_testable else "Not testable"
            print(f"  Coupling length {L_name}: δω/ω ~ {result.effect_size:.2e} ({status})")
            print(f"    Enhancement over QED: {result.enhancement_over_qed:.2e}×")

    print(f"\n{'='*70}")
    print("PLASMON COUPLING")
    print(f"{'='*70}")

    for material in ['gold', 'silver', 'aluminum']:
        params = plasmon_parameters(material)
        print(f"\n{params.description}:")
        print(f"  Plasma energy: {params.characteristic_energy/E_CHARGE:.2f} eV")
        print(f"  Skin depth: {params.characteristic_length*1e9:.1f} nm")

        for L_name, L in coupling_lengths.items():
            result = coupling_to_collective_mode(grad_phi, params, L)
            status = "TESTABLE" if result.is_testable else "Not testable"
            print(f"  Coupling length {L_name}: δω/ω ~ {result.effect_size:.2e} ({status})")

    print(f"\n{'='*70}")
    print("MAGNON COUPLING")
    print(f"{'='*70}")

    params = magnon_parameters('yig')
    print(f"\n{params.description} (best case):")
    print(f"  Magnon energy: {params.characteristic_energy/E_CHARGE*1e3:.1f} meV")
    print(f"  Exchange length: {params.characteristic_length*1e9:.1f} nm")
    print(f"  Quality factor: {params.quality_factor:.0e}")

    for L_name, L in coupling_lengths.items():
        result = coupling_to_collective_mode(grad_phi, params, L)
        status = "TESTABLE" if result.is_testable else "Not testable"
        print(f"  Coupling length {L_name}: δω/ω ~ {result.effect_size:.2e} ({status})")
        print(f"    Enhancement over QED: {result.enhancement_over_qed:.2e}×")

    print(f"\n{'='*70}")
    print("POLARITON COUPLING")
    print(f"{'='*70}")

    for system in ['microcavity', 'phonon_polariton']:
        params = polariton_parameters(system)
        print(f"\n{params.description}:")
        print(f"  Energy: {params.characteristic_energy/E_CHARGE*1e3:.1f} meV")
        print(f"  Length: {params.characteristic_length*1e6:.1f} μm")
        print(f"  Q factor: {params.quality_factor:.0e}")

        for L_name, L in coupling_lengths.items():
            result = coupling_to_collective_mode(grad_phi, params, L)
            status = "TESTABLE" if result.is_testable else "Not testable"
            print(f"  Coupling length {L_name}: δω/ω ~ {result.effect_size:.2e} ({status})")

    print(f"\n{'='*70}")
    print("SUMMARY: ENHANCEMENT FACTORS")
    print(f"{'='*70}")

    print("""
Compared to QED vacuum polarization:

Mode            E_char          Enhancement Factor
----            ------          ------------------
QED vacuum      0.5 MeV         1 (baseline)
Plasmon         ~10 eV          10^10
Phonon          ~50 meV         10^14
Magnon          ~10 meV         10^16
""")

    print(f"\n{'='*70}")
    print("THE CATCH")
    print(f"{'='*70}")

    print("""
The enhancement factors look great, BUT:

1. The COUPLING LENGTH still matters
   - If coupling is at lattice scale (nm): Effect ~ 10^-40
   - If coupling is at μm scale: Effect ~ 10^-28

2. We're still orders of magnitude below detection
   - Best case (magnon, μm coupling): ~10^-28
   - Need 10^-6 to detect
   - Gap: 22 orders

3. The coupling length is NOT a free parameter
   - Must be derived from the physics
   - Likely set by boundary structure, not tuneable

CONCLUSION: Enhancement helps (~14 orders), but not enough.
""")


def investigate_resonance_enhancement():
    """Check if resonance could provide additional enhancement."""
    print(f"\n{'='*70}")
    print("RESONANCE ENHANCEMENT ANALYSIS")
    print(f"{'='*70}")

    print("""
On resonance, the response is enhanced by Q.

For high-Q systems:
- Diamond mechanical resonators: Q ~ 10^8
- Superconducting cavities: Q ~ 10^12
- Atomic transitions: Q ~ 10^15

If BPR drives a resonance, enhancement = Q × (base coupling).

Best case: Q ~ 10^12 (superconducting cavity)
Base coupling ~ 10^-40 (collective mode)
Enhanced ~ 10^-28

Still 22 orders below detection!
""")

    # Calculate resonance enhancement scenarios
    Q_factors = {
        'diamond MEMS': 1e8,
        'superconducting cavity': 1e12,
        'atomic clock': 1e15,
    }

    base_coupling = 1e-40  # Typical for collective mode

    print("\nResonance scenarios:")
    print(f"{'System':<25} {'Q':<12} {'Enhanced coupling':<20} {'Gap to 10^-6':<15}")
    print("-" * 70)

    for system, Q in Q_factors.items():
        enhanced = base_coupling * Q
        gap = np.log10(1e-6 / enhanced) if enhanced < 1e-6 else 0
        status = "TESTABLE!" if enhanced > 1e-6 else f"{gap:.0f} orders"
        print(f"{system:<25} {Q:<12.0e} {enhanced:<20.2e} {status:<15}")


def investigate_nonlinear_enhancement():
    """Check if nonlinear effects could help."""
    print(f"\n{'='*70}")
    print("NONLINEAR ENHANCEMENT ANALYSIS")
    print(f"{'='*70}")

    print("""
Nonlinear effects scale differently:

Linear response:   δω/ω ~ (E_eff/E_char)²
Nonlinear:         δω/ω ~ (E_eff/E_char)^n for n < 2

If n = 1 (linear in field):
  Effect is (E_char/E_eff) times larger
  But E_eff/E_char ~ 10^-20, so this HURTS not helps

Parametric amplification:
  Can amplify signal by factor G
  Typical G ~ 10-100
  Doesn't help with 20+ order gap

Squeezed states:
  Reduce noise below shot noise
  Factor ~ 10-20 dB (10-100×)
  Doesn't help with 20+ order gap

CONCLUSION: Nonlinear effects don't bridge the gap.
""")


def investigate_coherent_enhancement():
    """Check if coherent/collective effects could help."""
    print(f"\n{'='*70}")
    print("COHERENT/COLLECTIVE ENHANCEMENT")
    print(f"{'='*70}")

    print("""
If N boundary phases act COHERENTLY:

Single phase coupling: g
N coherent phases: N × g (amplitude)
                   N² × g² (intensity)

For macroscopic boundary:
  N ~ (boundary area) / (coherence area)
  Coherence area ~ ξ² where ξ = correlation length

Example:
  Boundary: 1 cm × 1 cm = 10^-4 m²
  Correlation length: ξ ~ 1 μm
  Coherence area: 10^-12 m²
  N = 10^-4 / 10^-12 = 10^8

Enhancement: N² = 10^16

This is HUGE!

Base coupling ~ 10^-40
With coherent enhancement: 10^-40 × 10^16 = 10^-24

Still 18 orders below detection, but better!
""")

    # Calculate for different coherence lengths
    print("\nCoherent enhancement scenarios:")
    print(f"{'ξ (μm)':<12} {'N':<12} {'N²':<12} {'Coupling':<15} {'Gap to 10^-6':<15}")
    print("-" * 70)

    boundary_area = 1e-4  # 1 cm²
    base_coupling = 1e-40

    for xi_um in [0.1, 1, 10, 100]:
        xi = xi_um * 1e-6
        coherence_area = xi**2
        N = boundary_area / coherence_area
        N_sq = N**2
        enhanced = base_coupling * N_sq
        gap = np.log10(1e-6 / enhanced) if enhanced < 1e-6 else 0
        status = "TESTABLE!" if enhanced > 1e-6 else f"{gap:.0f} orders"
        print(f"{xi_um:<12.1f} {N:<12.1e} {N_sq:<12.1e} {enhanced:<15.2e} {status:<15}")

    print("""
KEY INSIGHT:

Short correlation length (small ξ) gives MORE coherent patches (larger N).
But short ξ also means thermal fluctuations are larger.

The two effects compete:
- More patches: N ∝ 1/ξ²
- Larger fluctuations: grad_phi ∝ 1/ξ

Need to optimize!
""")


if __name__ == "__main__":
    analyze_all_collective_modes()
    investigate_resonance_enhancement()
    investigate_nonlinear_enhancement()
    investigate_coherent_enhancement()
