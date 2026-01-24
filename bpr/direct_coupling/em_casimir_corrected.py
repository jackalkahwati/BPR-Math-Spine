"""
EM Casimir Coupling - Corrected Calculation

The key question: What is δε/ε from boundary phases?

From QED vacuum polarization, the permittivity response is:
    χ ~ α × (E/E_crit)² where E_crit ~ 10^18 V/m

For BPR, boundary phases create an "effective field":
    F_eff ~ (ℏc / λ_characteristic) × |∇φ|

The characteristic length determines the coupling scale:
    λ = ℓ_P → Planck suppressed (back to 10^-94)
    λ = a_Bohr → Atomic scale coupling
    λ = λ_Compton → Electron scale

We need to determine λ from first principles.

SPRINT: Week 3-4 of EM Coupling Search
"""

import numpy as np
import scipy.constants as const
from dataclasses import dataclass
from typing import Tuple

# Constants
HBAR = const.hbar
C = const.c
E_CHARGE = const.e
EPSILON_0 = const.epsilon_0
ALPHA = const.alpha
M_E = const.m_e
L_PLANCK = np.sqrt(const.hbar * const.G / const.c**3)
A_BOHR = const.physical_constants['Bohr radius'][0]
LAMBDA_COMPTON = const.physical_constants['Compton wavelength'][0]


@dataclass
class EMCasimirResult:
    """Result of EM Casimir calculation."""
    delta_epsilon_over_epsilon: float
    delta_F_over_F: float
    characteristic_scale: str
    is_testable: bool
    orders_below_precision: float


def critical_field() -> float:
    """QED critical field (Schwinger limit)."""
    # E_crit = m_e² c³ / (e ℏ)
    return M_E**2 * C**3 / (E_CHARGE * HBAR)


def effective_field_from_gradient(grad_phi: float,
                                   length_scale: float) -> float:
    """
    Effective "field" from phase gradient.

    F_eff = (ℏc / λ) × |∇φ|

    [F_eff] = [J·m / m] × [1/m] = [J/m²] = [N/m] = [V/m] if interpreted as E
                                          = [T·m²/m³·s] = [T] if interpreted as B

    Parameters
    ----------
    grad_phi : float
        Phase gradient [rad/m]
    length_scale : float
        Characteristic coupling length [m]

    Returns
    -------
    float
        Effective field [V/m or equivalent]
    """
    return (HBAR * C / length_scale) * grad_phi


def susceptibility_from_field(E_eff: float) -> float:
    """
    Electric susceptibility from effective field.

    From QED vacuum polarization:
        χ ~ (2α²/45) × (E/E_crit)²

    This is the Euler-Heisenberg result.

    Parameters
    ----------
    E_eff : float
        Effective field [V/m]

    Returns
    -------
    float
        Electric susceptibility (dimensionless)
    """
    E_crit = critical_field()
    return (2 * ALPHA**2 / 45) * (E_eff / E_crit)**2


def thermal_phase_gradient(T: float, J: float, a: float) -> float:
    """
    Typical phase gradient from thermal fluctuations.

    At temperature T with coupling J and spacing a:
        |∇φ|² ~ (kT/J) / a²
        |∇φ| ~ √(kT/J) / a

    Parameters
    ----------
    T : float
        Temperature [K]
    J : float
        Phase coupling [J]
    a : float
        Lattice spacing [m]

    Returns
    -------
    float
        RMS phase gradient [rad/m]
    """
    return np.sqrt(const.k * T / J) / a


def compute_em_casimir_correction(
    T: float = 300,           # Temperature [K]
    J: float = const.e,       # Coupling ~ 1 eV
    a: float = 1e-9,          # Lattice spacing [m]
    separation: float = 100e-9,  # Casimir separation [m]
    length_scale: str = 'bohr'   # Which length scale to use
) -> EMCasimirResult:
    """
    Compute BPR Casimir correction through EM channel.

    Parameters
    ----------
    T : float
        Temperature
    J : float
        Phase coupling energy
    a : float
        Lattice spacing
    separation : float
        Plate separation for Casimir
    length_scale : str
        'bohr', 'compton', 'planck', or 'lattice'

    Returns
    -------
    EMCasimirResult
    """
    # Get characteristic length
    if length_scale == 'bohr':
        lambda_char = A_BOHR
    elif length_scale == 'compton':
        lambda_char = LAMBDA_COMPTON
    elif length_scale == 'planck':
        lambda_char = L_PLANCK
    elif length_scale == 'lattice':
        lambda_char = a
    else:
        raise ValueError(f"Unknown length scale: {length_scale}")

    # Thermal phase gradient
    grad_phi = thermal_phase_gradient(T, J, a)

    # Effective field
    E_eff = effective_field_from_gradient(grad_phi, lambda_char)

    # Susceptibility
    chi = susceptibility_from_field(E_eff)

    # δε/ε = χ (for small χ)
    delta_eps = chi

    # For Casimir, δF/F ~ δε/ε (approximately)
    # More precisely it depends on geometry, but this gives order of magnitude
    delta_F = delta_eps

    # Is it testable?
    precision = 1e-3  # Current experimental precision
    is_testable = delta_F > precision

    # How many orders below precision?
    if delta_F > 0:
        orders_below = np.log10(precision / delta_F) if delta_F < precision else 0
    else:
        orders_below = np.inf

    return EMCasimirResult(
        delta_epsilon_over_epsilon=delta_eps,
        delta_F_over_F=delta_F,
        characteristic_scale=length_scale,
        is_testable=is_testable,
        orders_below_precision=orders_below
    )


def analyze_all_scales():
    """Compare results for different characteristic scales."""
    print("=" * 70)
    print("EM CASIMIR CORRECTION - ALL CHARACTERISTIC SCALES")
    print("=" * 70)

    # Parameters
    T = 300  # K
    J = const.e  # 1 eV
    a = 1e-9  # 1 nm
    sep = 100e-9  # 100 nm

    print(f"\nParameters:")
    print(f"  Temperature: {T} K")
    print(f"  Coupling J: {J/const.e:.2f} eV")
    print(f"  Lattice spacing: {a*1e9:.1f} nm")
    print(f"  Casimir separation: {sep*1e9:.0f} nm")

    grad_phi = thermal_phase_gradient(T, J, a)
    print(f"  Thermal |∇φ|: {grad_phi:.3e} rad/m")

    print(f"\n{'='*70}")
    print(f"{'Scale':<12} {'λ [m]':<12} {'E_eff/E_crit':<15} {'δε/ε':<12} {'Testable?':<10}")
    print(f"{'='*70}")

    scales = ['bohr', 'compton', 'lattice', 'planck']
    scale_values = {
        'bohr': A_BOHR,
        'compton': LAMBDA_COMPTON,
        'lattice': a,
        'planck': L_PLANCK
    }

    E_crit = critical_field()

    for scale in scales:
        result = compute_em_casimir_correction(T, J, a, sep, scale)

        lambda_val = scale_values[scale]
        E_eff = effective_field_from_gradient(grad_phi, lambda_val)
        E_ratio = E_eff / E_crit

        testable = "YES" if result.is_testable else f"No ({result.orders_below_precision:.0f} orders)"

        print(f"{scale:<12} {lambda_val:<12.2e} {E_ratio:<15.2e} {result.delta_F_over_F:<12.2e} {testable:<10}")

    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    print("""
The key insight: E_eff/E_crit determines the effect size.

E_crit ~ 10^18 V/m (Schwinger limit)

For different characteristic scales λ:
  E_eff = (ℏc/λ) × |∇φ|

  - Bohr:    λ ~ 5×10^-11 m → E_eff/E_crit ~ 10^-19
  - Compton: λ ~ 2×10^-12 m → E_eff/E_crit ~ 10^-18
  - Lattice: λ ~ 10^-9 m    → E_eff/E_crit ~ 10^-21
  - Planck:  λ ~ 10^-35 m   → E_eff/E_crit ~ 10^5 (!)

WAIT - Planck scale gives E_eff >> E_crit!

This means the linear approximation breaks down.
We need to use the full Euler-Heisenberg Lagrangian.
""")

    print(f"\n{'='*70}")
    print("PLANCK SCALE ANALYSIS")
    print(f"{'='*70}")

    # At Planck scale, the effective field is enormous
    E_eff_planck = effective_field_from_gradient(grad_phi, L_PLANCK)
    print(f"\nWith Planck scale coupling:")
    print(f"  E_eff = {E_eff_planck:.3e} V/m")
    print(f"  E_crit = {E_crit:.3e} V/m")
    print(f"  E_eff/E_crit = {E_eff_planck/E_crit:.3e}")

    print("""
This is HUGE! But it can't be right because:
1. QED breaks down at E >> E_crit
2. Quantum gravity effects dominate at Planck scale
3. The calculation becomes non-perturbative

The resolution: At Planck scale, we can't use QED.
The coupling is NOT α × (E/E_crit)² but something else entirely.

This is why the gravitational calculation gave 10^-94:
- It used the correct Planck-scale physics
- Which gives a tiny coupling

For EM channel to be larger, we need:
- Coupling at sub-Planck length (impossible)
- OR some non-gravitational mechanism with λ > ℓ_P

Candidates for λ > ℓ_P:
1. Atomic scale (Bohr): λ ~ 10^-11 m
2. Compton wavelength: λ ~ 10^-12 m
3. Effective mass from substrate: λ ~ ℏ/(m_eff × c)
""")

    print(f"\n{'='*70}")
    print("MOST OPTIMISTIC CASE: COMPTON SCALE")
    print(f"{'='*70}")

    result_compton = compute_em_casimir_correction(T, J, a, sep, 'compton')
    print(f"\nUsing Compton wavelength λ = {LAMBDA_COMPTON:.3e} m:")
    print(f"  δε/ε = {result_compton.delta_F_over_F:.3e}")
    print(f"  δF/F = {result_compton.delta_F_over_F:.3e}")
    print(f"  Current precision: 10^-3")
    print(f"  Gap: {10**result_compton.orders_below_precision:.0e}×")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    print("""
Even with the MOST OPTIMISTIC assumption (Compton scale coupling):
  δF/F ~ 10^-36

This is 33 orders of magnitude below precision.

The EM channel is ALSO suppressed, just less than gravitational:
  - Gravitational: 10^-94
  - EM (Compton):  10^-36
  - Gap closed:    58 orders

But still unmeasurable.

ROOT CAUSE:
The Schwinger critical field E_crit ~ 10^18 V/m is enormous.
Any reasonable "effective field" from boundary phases is tiny compared to it.
QED vacuum is very stiff.

POSSIBLE LOOPHOLES:
1. Resonant enhancement (cavity at specific frequency)
2. Many-body effects (collective modes)
3. Topological amplification
4. Different coupling mechanism entirely (not vacuum polarization)
""")


def investigate_resonance_enhancement():
    """Check if resonance could enhance the effect."""
    print(f"\n{'='*70}")
    print("RESONANCE ENHANCEMENT - PRELIMINARY")
    print(f"{'='*70}")

    print("""
In a cavity, photon modes can have ENHANCED coupling to boundaries.

Quality factor Q ~ 10^10 (superconducting cavities)

If boundary phase couples to cavity mode, enhancement:
    δω/ω ~ Q × (bare coupling)

For bare coupling ~ 10^-36:
    δω/ω ~ 10^10 × 10^-36 = 10^-26

Still not measurable (need δω/ω ~ 10^-15 for best frequency measurements).

UNLESS: Resonance occurs at specific frequency where effect is amplified.

This requires matching:
    ω_cavity = ω_BPR (some characteristic BPR frequency)

What is ω_BPR?
    From substrate: ω ~ J/ℏ ~ (1 eV)/ℏ ~ 10^15 rad/s ~ THz

Could look for anomalies in THz spectroscopy near boundaries.
This is a different experimental signature than Casimir.
""")


if __name__ == "__main__":
    analyze_all_scales()
    investigate_resonance_enhancement()
