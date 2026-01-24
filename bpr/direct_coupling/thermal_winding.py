"""
Thermal Winding Analysis

The EM coupling g_φ ~ α × e is NOT Planck suppressed.
But observable effects depend on WINDING NUMBER W.

At temperature T, winding is thermally suppressed:
    P(W) ~ exp(-E_W / kT)

This determines what effects are actually observable.

SPRINT: Week 2-3 of EM Coupling Search
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import scipy.constants as const

# Constants
K_B = const.k
HBAR = const.hbar
C = const.c
E_CHARGE = const.e
ALPHA = const.alpha


@dataclass
class WindingStatistics:
    """Thermal statistics of boundary winding."""
    mean_W: float
    rms_W: float
    prob_W1: float
    characteristic_T: float  # Temperature where W=1 becomes probable


def winding_energy(W: int, J: float, length_scale: float) -> float:
    """
    Energy cost of winding W.

    For a ring of circumference L:
        E_W = (J/2) × (2πW/L)² × L = (2π² J W²) / L

    For a sphere, scaling is similar.

    Parameters
    ----------
    W : int
        Winding number
    J : float
        Phase stiffness [J·m]
    length_scale : float
        System size [m]

    Returns
    -------
    float
        Winding energy [J]
    """
    return 2 * np.pi**2 * J * W**2 / length_scale


def thermal_winding_probability(W: int, T: float, J: float,
                                 length_scale: float) -> float:
    """
    Probability of winding W at temperature T.

    P(W) = exp(-E_W / kT) / Z
    where Z = Σ_W exp(-E_W / kT)

    Parameters
    ----------
    W : int
        Winding number
    T : float
        Temperature [K]
    J : float
        Phase stiffness [J·m]
    length_scale : float
        System size [m]

    Returns
    -------
    float
        Probability
    """
    if T <= 0:
        return 1.0 if W == 0 else 0.0

    kT = K_B * T
    E_W = winding_energy(W, J, length_scale)

    # Compute partition function (truncate at large W)
    Z = 0.0
    for w in range(-100, 101):
        E = winding_energy(w, J, length_scale)
        Z += np.exp(-E / kT)

    return np.exp(-E_W / kT) / Z


def compute_winding_statistics(T: float, J: float,
                                length_scale: float) -> WindingStatistics:
    """
    Compute thermal statistics of winding.

    Parameters
    ----------
    T : float
        Temperature [K]
    J : float
        Phase stiffness [J·m]
    length_scale : float
        System size [m]

    Returns
    -------
    WindingStatistics
    """
    if T <= 0:
        return WindingStatistics(
            mean_W=0.0,
            rms_W=0.0,
            prob_W1=0.0,
            characteristic_T=winding_energy(1, J, length_scale) / K_B
        )

    # Compute moments
    mean_W = 0.0
    mean_W2 = 0.0
    prob_W1 = 0.0

    for W in range(-100, 101):
        P_W = thermal_winding_probability(W, T, J, length_scale)
        mean_W += W * P_W
        mean_W2 += W**2 * P_W
        if abs(W) == 1:
            prob_W1 += P_W

    rms_W = np.sqrt(mean_W2)

    # Temperature where W=1 has reasonable probability
    E_1 = winding_energy(1, J, length_scale)
    T_char = E_1 / K_B  # P(W=1) ~ exp(-1) at this T

    return WindingStatistics(
        mean_W=mean_W,
        rms_W=rms_W,
        prob_W1=prob_W1,
        characteristic_T=T_char
    )


def effective_winding_for_observable(T: float, J: float,
                                      length_scale: float) -> float:
    """
    Effective winding number for computing observables.

    Observables scale as W or W².
    Returns RMS winding for typical effect size.
    """
    stats = compute_winding_statistics(T, J, length_scale)
    return stats.rms_W


def phase_stiffness_from_substrate(
    coupling_J: float,  # [J]
    lattice_spacing: float,  # [m]
    coordination: int
) -> float:
    """
    Phase stiffness from substrate parameters.

    κ = (J × a × z / 2)

    This has dimensions [J·m].

    Parameters
    ----------
    coupling_J : float
        Substrate coupling [J]
    lattice_spacing : float
        Lattice spacing [m]
    coordination : int
        Coordination number

    Returns
    -------
    float
        Phase stiffness [J·m]
    """
    return coupling_J * lattice_spacing * coordination / 2


def analyze_thermal_suppression():
    """
    Complete analysis of thermal suppression of winding.
    """
    print("=" * 70)
    print("THERMAL WINDING ANALYSIS")
    print("=" * 70)

    # Substrate parameters
    J_coupling = const.e  # 1 eV
    a = 1e-9  # 1 nm lattice spacing
    z = 4  # coordination

    J_stiffness = phase_stiffness_from_substrate(J_coupling, a, z)
    print(f"\nSubstrate parameters:")
    print(f"  Coupling: J = {J_coupling/const.e:.2f} eV")
    print(f"  Lattice spacing: a = {a*1e9:.1f} nm")
    print(f"  Coordination: z = {z}")
    print(f"  Phase stiffness: κ = {J_stiffness:.3e} J·m")

    # System sizes
    sizes = [1e-6, 1e-4, 1e-2]  # 1 μm, 100 μm, 1 cm
    size_names = ["1 μm", "100 μm", "1 cm"]

    # Temperatures
    temps = [4.0, 77.0, 300.0]  # 4K, 77K, room temp
    temp_names = ["4 K", "77 K (LN₂)", "300 K (room)"]

    print(f"\n{'='*70}")
    print("WINDING STATISTICS BY SIZE AND TEMPERATURE")
    print(f"{'='*70}")

    for L, L_name in zip(sizes, size_names):
        print(f"\n--- System size: {L_name} ---")

        E_1 = winding_energy(1, J_stiffness, L)
        T_char = E_1 / K_B

        print(f"  Winding energy E(W=1) = {E_1/const.e*1e3:.3f} meV")
        print(f"  Characteristic temperature = {T_char:.1f} K")

        for T, T_name in zip(temps, temp_names):
            stats = compute_winding_statistics(T, J_stiffness, L)
            print(f"\n  At {T_name}:")
            print(f"    ⟨W⟩ = {stats.mean_W:.3e}")
            print(f"    √⟨W²⟩ = {stats.rms_W:.3e}")
            print(f"    P(|W|=1) = {stats.prob_W1:.3e}")

    print(f"\n{'='*70}")
    print("IMPLICATIONS FOR EM COUPLING")
    print(f"{'='*70}")

    # A-B phase prediction with thermal winding
    g_phi = ALPHA * E_CHARGE

    print(f"\nCoupling: g_φ = α × e = {g_phi:.3e} C")
    print(f"A-B phase per unit winding: Δφ/W = {(E_CHARGE/g_phi):.1f} × 2π rad")

    print(f"\nEFFECTIVE A-B PHASE (using RMS winding):")

    for L, L_name in zip(sizes, size_names):
        print(f"\n  {L_name}:")
        for T, T_name in zip(temps, temp_names):
            W_eff = effective_winding_for_observable(T, J_stiffness, L)
            delta_phi = (E_CHARGE / g_phi) * 2 * np.pi * W_eff

            print(f"    {T_name}: W_eff = {W_eff:.3e}, Δφ = {delta_phi:.3e} rad")

            if delta_phi > 0.01:
                print(f"      STATUS: DETECTABLE (current: 0.01 rad)")
            elif delta_phi > 1e-6:
                print(f"      STATUS: Near-term possible")
            else:
                print(f"      STATUS: Not detectable")

    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print(f"{'='*70}")
    print("""
The coupling g_φ ~ α × e is NOT Planck suppressed.
BUT thermal fluctuations suppress non-trivial winding.

To observe EM effects:
1. Cool to mK temperatures (superconducting systems)
2. Use small systems (larger winding fluctuations)
3. Artificially create winding (vortex injection)
4. Use non-equilibrium boundary configurations

BOTTOM LINE:
- Coupling constant: GOOD (α scale, not Planck)
- Thermal state: PROBLEM (winding suppressed at room T)
- Solution: Low T, small systems, or artificial winding
""")


if __name__ == "__main__":
    analyze_thermal_suppression()
