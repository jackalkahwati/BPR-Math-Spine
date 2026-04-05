"""
BPR Optics: Boundary Phase Resonance in Light and Metamaterials
================================================================

Derives optical phenomena from boundary phase dynamics:
photon propagation, nonlinear optics, metamaterial design,
and Casimir-Polder effects.

Key idea: light propagation = phase coherence transport across
boundaries. Metamaterials = engineered boundary impedance.

Energy scale: ~1 eV (visible), ~0.01-10 eV (IR to UV)
Abstraction: 3-5 (phenomenology)

References: Al-Kahwati (2026), BPR extensions
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .constants import (
    C, HBAR, K_B, EPSILON_0, Z_0, P_DEFAULT, GAMMA_ZEROS,
    H_PLANCK, E_CHARGE,
)

# ---------------------------------------------------------------------------
# Shorthand
# ---------------------------------------------------------------------------
_c = C
_hbar = HBAR
_k_B = K_B
_eps0 = EPSILON_0
_Z0 = Z_0
_p = P_DEFAULT
_gamma1 = GAMMA_ZEROS[0]  # 14.1347  (first Riemann zero)


# ═══════════════════════════════════════════════════════════════════════════
# §O.1  BPR Refractive Index
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LorentzOscillator:
    """Single Lorentz oscillator model for material impedance.

    Parameters
    ----------
    omega_p : float
        Plasma frequency [rad/s].
    omega_0 : float
        Resonance frequency [rad/s].
    gamma : float
        Damping rate [rad/s].
    """
    omega_p: float
    omega_0: float
    gamma: float


def bpr_refractive_index(
    omega: np.ndarray | float,
    Z_material: np.ndarray | float,
    Z_vacuum: float = _Z0,
) -> np.ndarray | float:
    """Refractive index from impedance matching.

    n(omega) = Z_vacuum / Z_material(omega)

    In BPR, light slows because boundary impedance differs from vacuum.
    n > 1 when Z_material < Z_vacuum (normal materials).
    n < 1 when Z_material > Z_vacuum (metamaterials, plasma).

    Parameters
    ----------
    omega : array_like
        Angular frequency [rad/s].
    Z_material : array_like
        Material impedance at *omega* [Ohm].  May be complex (lossy media).
    Z_vacuum : float
        Vacuum impedance (default 376.73 Ohm).

    Returns
    -------
    n : array_like
        Complex refractive index.  Re(n) = phase index, Im(n) = extinction.

    Examples
    --------
    >>> # Glass (Z ~ 200 Ohm) -> n ~ 1.88
    >>> float(np.real(bpr_refractive_index(3e15, 200.0)))  # doctest: +SKIP
    1.88...
    """
    omega = np.asarray(omega, dtype=complex)
    Z_material = np.asarray(Z_material, dtype=complex)
    return Z_vacuum / Z_material


def lorentz_impedance(
    omega: np.ndarray | float,
    osc: LorentzOscillator,
    Z_base: float = _Z0,
) -> np.ndarray:
    """Material impedance from a single Lorentz oscillator.

    Z_material(omega) = Z_base * sqrt(1 + omega_p^2 / (omega^2 - omega_0^2 + i*gamma*omega))

    Parameters
    ----------
    omega : array_like
        Angular frequency [rad/s].
    osc : LorentzOscillator
        Oscillator parameters.
    Z_base : float
        Background impedance [Ohm].

    Returns
    -------
    Z : ndarray (complex)
        Material impedance at each frequency.
    """
    omega = np.asarray(omega, dtype=complex)
    denom = omega**2 - osc.omega_0**2 + 1j * osc.gamma * omega
    epsilon_r = 1.0 + osc.omega_p**2 / denom
    return Z_base / np.sqrt(epsilon_r)


# ═══════════════════════════════════════════════════════════════════════════
# §O.2  Metamaterial Design from Impedance
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetamaterialDesign:
    """Result of a metamaterial impedance design.

    Attributes
    ----------
    target_n : complex
        Desired refractive index.
    Z_required : complex
        Required impedance [Ohm].
    epsilon_r : complex
        Required relative permittivity.
    mu_r : complex
        Required relative permeability.
    is_negative_index : bool
        True if Re(n) < 0 (needs active boundary / double-negative).
    """
    target_n: complex
    Z_required: complex
    epsilon_r: complex
    mu_r: complex
    is_negative_index: bool


def metamaterial_impedance(
    target_n: complex,
    omega: float,
    Z_vacuum: float = _Z0,
) -> MetamaterialDesign:
    """Design a metamaterial with target refractive index.

    Z_required = Z_vacuum / target_n.
    For n < 0 (negative index): need Z < 0 (active boundary).

    Parameters
    ----------
    target_n : complex
        Target refractive index.
    omega : float
        Design frequency [rad/s].
    Z_vacuum : float
        Vacuum impedance [Ohm].

    Returns
    -------
    MetamaterialDesign
        Required impedance and material parameters.

    Examples
    --------
    >>> d = metamaterial_impedance(-1.0, 2*np.pi*300e12)
    >>> d.is_negative_index
    True
    >>> abs(d.Z_required + Z_0) < 1.0  # Z ≈ -376.73
    True
    """
    n = complex(target_n)
    Z_req = Z_vacuum / n

    # n = sqrt(eps_r * mu_r), Z = Z_0 * sqrt(mu_r / eps_r)
    # => eps_r = n / (Z_req / Z_vacuum) = n^2 / mu_r
    # and mu_r = (Z_req / Z_vacuum) * n
    mu_r = (Z_req / Z_vacuum) * n
    eps_r = n / mu_r if mu_r != 0 else complex(float("inf"))

    return MetamaterialDesign(
        target_n=n,
        Z_required=Z_req,
        epsilon_r=eps_r,
        mu_r=mu_r,
        is_negative_index=(np.real(n) < 0),
    )


@dataclass
class CloakLayer:
    """Single layer of an impedance cloak."""
    r_inner: float
    r_outer: float
    impedance: float
    n_eff: float


@dataclass
class CloakDesign:
    """Impedance cloak design result.

    Attributes
    ----------
    R_inner : float
        Inner radius [m].
    R_outer : float
        Outer radius [m].
    layers : list[CloakLayer]
        Layer-by-layer impedance profile.
    scattering_cross_section : float
        Predicted scattering cross-section [m^2].
    """
    R_inner: float
    R_outer: float
    layers: List[CloakLayer] = field(default_factory=list)
    scattering_cross_section: float = 0.0


def impedance_cloak(
    R_inner: float,
    R_outer: float,
    n_layers: int = 10,
) -> CloakDesign:
    """Invisibility cloak as radial impedance gradient.

    Z(r) = Z_0 * (r / R_outer)  for r in [R_inner, R_outer].
    Perfect cloak when Z(R_inner) -> 0 (zero impedance at inner boundary).

    Parameters
    ----------
    R_inner : float
        Inner (cloaked object) radius [m].
    R_outer : float
        Outer cloak radius [m].
    n_layers : int
        Number of discrete layers.

    Returns
    -------
    CloakDesign
        Layer impedances and predicted scattering cross-section.

    Notes
    -----
    Predicted scattering cross-section scales as:
        sigma ~ pi * R_inner^2 * (R_inner / R_outer)^2 * (1 / n_layers)^2
    vanishing for thick cloak and many layers.
    """
    edges = np.linspace(R_inner, R_outer, n_layers + 1)
    layers: List[CloakLayer] = []
    for i in range(n_layers):
        r_mid = 0.5 * (edges[i] + edges[i + 1])
        Z_layer = _Z0 * (r_mid / R_outer)
        n_eff = _Z0 / Z_layer if Z_layer > 0 else float("inf")
        layers.append(CloakLayer(
            r_inner=edges[i],
            r_outer=edges[i + 1],
            impedance=Z_layer,
            n_eff=n_eff,
        ))

    # Residual scattering from discretisation
    ratio = R_inner / R_outer
    sigma = np.pi * R_inner**2 * ratio**2 / n_layers**2

    return CloakDesign(
        R_inner=R_inner,
        R_outer=R_outer,
        layers=layers,
        scattering_cross_section=sigma,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §O.3  Nonlinear Optics from Boundary Phase
# ═══════════════════════════════════════════════════════════════════════════

def bpr_kerr_coefficient(
    chi_3: float,
    Z_material: float,
    n_0: Optional[float] = None,
) -> float:
    """Kerr coefficient n_2 from third-order susceptibility.

    n_2 = 3 chi^(3) / (4 epsilon_0 c n_0^2)

    BPR prediction: chi^(3) ~ 1/Z^3 (stronger nonlinearity at lower impedance).

    Parameters
    ----------
    chi_3 : float
        Third-order nonlinear susceptibility [m^2/V^2].
    Z_material : float
        Material impedance [Ohm].
    n_0 : float, optional
        Linear refractive index.  If None, computed as Z_0 / Z_material.

    Returns
    -------
    n_2 : float
        Kerr coefficient [m^2/W].

    Examples
    --------
    >>> # Fused silica: chi^(3) ~ 2e-22 m^2/V^2, n ~ 1.45
    >>> n2 = bpr_kerr_coefficient(2e-22, _Z0 / 1.45)
    >>> 2e-20 < n2 < 4e-20
    True
    """
    if n_0 is None:
        n_0 = _Z0 / Z_material
    return 3.0 * chi_3 / (4.0 * _eps0 * _c * n_0**2)


def second_harmonic_generation(
    omega: float,
    chi_2: float,
    L_crystal: float,
    n_omega: float,
    n_2omega: float,
) -> dict:
    """Second harmonic generation efficiency from phase matching.

    BPR prediction: optimal phase matching when Z(omega) = Z(2*omega),
    i.e. impedance matching between fundamental and harmonic.

    Parameters
    ----------
    omega : float
        Fundamental angular frequency [rad/s].
    chi_2 : float
        Second-order susceptibility [m/V].
    L_crystal : float
        Crystal length [m].
    n_omega : float
        Refractive index at omega.
    n_2omega : float
        Refractive index at 2*omega.

    Returns
    -------
    dict
        Keys: 'efficiency', 'coherence_length', 'delta_k',
        'Z_omega', 'Z_2omega', 'impedance_mismatch'.

    Examples
    --------
    >>> # KDP crystal: chi_2 ~ 0.4 pm/V, L = 1 cm
    >>> r = second_harmonic_generation(2*np.pi*3e14, 0.4e-12, 1e-2, 1.49, 1.51)
    >>> r['coherence_length'] > 0
    True
    """
    # Phase mismatch
    delta_k = 2.0 * omega * (n_2omega - n_omega) / _c

    # Coherence length
    L_c = np.pi / abs(delta_k) if delta_k != 0 else float("inf")

    # SHG efficiency: eta ~ (chi_2 * L * omega / (n_omega * n_2omega * c))^2 * sinc^2(delta_k * L / 2)
    prefactor = chi_2 * L_crystal * omega / (n_omega * n_2omega * _c)
    phase_arg = delta_k * L_crystal / 2.0
    sinc_val = np.sinc(phase_arg / np.pi) if phase_arg != 0 else 1.0
    efficiency = prefactor**2 * sinc_val**2

    # BPR impedances
    Z_omega = _Z0 / n_omega
    Z_2omega = _Z0 / n_2omega
    impedance_mismatch = abs(Z_omega - Z_2omega) / _Z0

    return {
        "efficiency": float(efficiency),
        "coherence_length": float(L_c),
        "delta_k": float(delta_k),
        "Z_omega": float(Z_omega),
        "Z_2omega": float(Z_2omega),
        "impedance_mismatch": float(impedance_mismatch),
    }


@dataclass
class SolitonResult:
    """Optical soliton analysis result."""
    soliton_number: float
    is_stable: bool
    P_peak: float
    P_soliton: float
    dispersion_length: float
    nonlinear_length: float


def bpr_soliton_condition(
    P_peak: float,
    n_2: float,
    A_eff: float,
    beta_2: float,
    T_0: float = 1e-13,
) -> SolitonResult:
    """Optical soliton condition: N^2 = gamma * P_0 * T_0^2 / |beta_2| = 1.

    BPR: solitons are boundary-stabilised phase coherent pulses.

    Parameters
    ----------
    P_peak : float
        Peak power [W].
    n_2 : float
        Kerr coefficient [m^2/W].
    A_eff : float
        Effective mode area [m^2].
    beta_2 : float
        Group velocity dispersion [s^2/m].  Must be negative for bright soliton.
    T_0 : float
        Pulse duration [s].

    Returns
    -------
    SolitonResult
        Soliton number N and stability assessment.

    Examples
    --------
    >>> # Typical SMF-28 soliton: n_2=2.6e-20, A_eff=80e-12, beta_2=-21e-27
    >>> r = bpr_soliton_condition(50e-3, 2.6e-20, 80e-12, -21e-27, T_0=1e-12)
    >>> 0.5 < r.soliton_number < 2.0
    True
    """
    omega_0 = 2.0 * np.pi * _c / 1550e-9  # reference: 1550 nm
    gamma = n_2 * omega_0 / (_c * A_eff)

    L_D = T_0**2 / abs(beta_2)       # dispersion length
    L_NL = 1.0 / (gamma * P_peak)    # nonlinear length

    N_sq = gamma * P_peak * T_0**2 / abs(beta_2)
    N = np.sqrt(N_sq)

    # Fundamental soliton power
    P_sol = abs(beta_2) / (gamma * T_0**2)

    # Stable if 0.5 < N < 1.5 (fundamental); higher-order solitons are unstable
    is_stable = 0.5 < N < 1.5

    return SolitonResult(
        soliton_number=float(N),
        is_stable=is_stable,
        P_peak=P_peak,
        P_soliton=float(P_sol),
        dispersion_length=float(L_D),
        nonlinear_length=float(L_NL),
    )


# ═══════════════════════════════════════════════════════════════════════════
# §O.4  Photonic Crystal from Boundary Periodicity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicBandgapResult:
    """Photonic bandgap analysis."""
    center_frequency: float       # omega_0 [rad/s]
    gap_width: float              # Delta_omega [rad/s]
    relative_gap: float           # Delta_omega / omega_0
    lower_edge: float             # omega_lower [rad/s]
    upper_edge: float             # omega_upper [rad/s]
    impedance_contrast: float     # |Z_1 - Z_2| / (Z_1 + Z_2)


def photonic_bandgap(
    n1: float,
    n2: float,
    d1: float,
    d2: float,
) -> PhotonicBandgapResult:
    """Photonic bandgap from periodic impedance boundaries.

    Relative gap: Delta_omega/omega_0 = (4/pi) * arcsin(|n_1 - n_2| / (n_1 + n_2))

    BPR: the bandgap is a boundary-induced spectral gap identical
    to the eigenvalue gap in resonance_algebra.

    Parameters
    ----------
    n1 : float
        Refractive index of layer 1.
    n2 : float
        Refractive index of layer 2.
    d1 : float
        Thickness of layer 1 [m].
    d2 : float
        Thickness of layer 2 [m].

    Returns
    -------
    PhotonicBandgapResult
        Gap centre, width, and impedance contrast.

    Examples
    --------
    >>> # Si/SiO2 stack: n1=3.5, n2=1.45, quarter-wave at 1550 nm
    >>> d1 = 1550e-9 / (4*3.5)
    >>> d2 = 1550e-9 / (4*1.45)
    >>> r = photonic_bandgap(3.5, 1.45, d1, d2)
    >>> r.relative_gap > 0.3  # large contrast => wide gap
    True
    """
    period = d1 + d2
    # Bragg centre: n1*d1 + n2*d2 = lambda_B / 2
    lambda_B = 2.0 * (n1 * d1 + n2 * d2)
    omega_0 = 2.0 * np.pi * _c / lambda_B

    # Relative gap width
    contrast = abs(n1 - n2) / (n1 + n2)
    relative_gap = (4.0 / np.pi) * np.arcsin(contrast)

    delta_omega = relative_gap * omega_0
    omega_lower = omega_0 - delta_omega / 2.0
    omega_upper = omega_0 + delta_omega / 2.0

    return PhotonicBandgapResult(
        center_frequency=float(omega_0),
        gap_width=float(delta_omega),
        relative_gap=float(relative_gap),
        lower_edge=float(omega_lower),
        upper_edge=float(omega_upper),
        impedance_contrast=float(contrast),
    )


@dataclass
class TopologicalEdgeMode:
    """Topological photonic edge mode result."""
    frequency: float              # omega_edge [rad/s]
    decay_length: float           # penetration into bulk [m]
    is_topological: bool          # robust to disorder
    winding_number: int           # BPR Class A winding


def topological_photonic_edge_mode(
    n_periods: int,
    n1: float,
    n2: float,
    d1: float,
    d2: float,
) -> TopologicalEdgeMode:
    """Topological edge mode in photonic crystal.

    BPR: these are Class A (winding) boundary modes.
    Prediction: edge mode frequency = midgap, robust to disorder.

    Parameters
    ----------
    n_periods : int
        Number of unit cells on each side of the interface.
    n1 : float
        Refractive index of layer 1.
    n2 : float
        Refractive index of layer 2.
    d1 : float
        Thickness of layer 1 [m].
    d2 : float
        Thickness of layer 2 [m].

    Returns
    -------
    TopologicalEdgeMode
        Edge mode properties.

    Examples
    --------
    >>> m = topological_photonic_edge_mode(20, 3.5, 1.45, 111e-9, 267e-9)
    >>> m.is_topological
    True
    """
    bg = photonic_bandgap(n1, n2, d1, d2)

    # Edge mode sits at midgap
    omega_edge = bg.center_frequency

    # Decay length ~ period / ln(n1/n2)
    period = d1 + d2
    ratio = max(n1, n2) / min(n1, n2)
    decay_length = period / np.log(ratio) if ratio > 1 else float("inf")

    # Topological if Zak phase difference across interface is pi
    # i.e. when n1 != n2 (non-trivial contrast)
    is_topological = abs(n1 - n2) > 1e-10

    # BPR winding number: +1 for the interface mode
    winding = 1 if is_topological else 0

    return TopologicalEdgeMode(
        frequency=float(omega_edge),
        decay_length=float(decay_length),
        is_topological=is_topological,
        winding_number=winding,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §O.5  Casimir-Polder from BPR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CasimirPolderResult:
    """Casimir-Polder atom-surface potential."""
    d: float                    # separation [m]
    U_standard: float           # standard potential [J]
    U_bpr: float                # BPR-corrected potential [J]
    regime: str                 # "retarded" or "thermal"
    bpr_correction_frac: float  # delta_U / U_standard


def casimir_polder_potential(
    alpha_atom: float,
    d: float,
    T: float = 300.0,
    p: int = P_DEFAULT,
) -> CasimirPolderResult:
    """Casimir-Polder atom-surface potential.

    Retarded (d << lambda_thermal):
        U(d) = -(3 hbar c alpha) / (8 pi^2 epsilon_0 d^4)

    Thermal (d >> lambda_thermal):
        U(d) = -(3 k_B T alpha) / (4 pi epsilon_0 d^3)

    BPR correction: U_BPR = U * (1 + delta/p) from substrate granularity,
    where delta = ln(p) / (2 pi).

    Parameters
    ----------
    alpha_atom : float
        Static atomic polarisability [C^2 s^2 / (kg m^3)] (SI).
        For Rb: alpha ~ 4.7e-39.
    d : float
        Atom-surface separation [m].
    T : float
        Temperature [K].
    p : int
        BPR substrate prime.

    Returns
    -------
    CasimirPolderResult
        Potential in both regimes with BPR correction.

    Examples
    --------
    >>> # Rb atom at 1 micron from gold surface
    >>> r = casimir_polder_potential(4.7e-39, 1e-6, T=300)
    >>> r.U_bpr < 0  # attractive
    True
    >>> abs(r.bpr_correction_frac) < 0.01  # small BPR correction
    True
    """
    lambda_thermal = _hbar * _c / (_k_B * T)
    delta_bpr = np.log(p) / (2.0 * np.pi)

    if d < lambda_thermal:
        # Retarded (van der Waals-Casimir) regime
        U_std = -(3.0 * _hbar * _c * alpha_atom) / (
            8.0 * np.pi**2 * _eps0 * d**4
        )
        regime = "retarded"
    else:
        # Thermal regime
        U_std = -(3.0 * _k_B * T * alpha_atom) / (
            4.0 * np.pi * _eps0 * d**3
        )
        regime = "thermal"

    correction = delta_bpr / p
    U_bpr = U_std * (1.0 + correction)

    return CasimirPolderResult(
        d=d,
        U_standard=float(U_std),
        U_bpr=float(U_bpr),
        regime=regime,
        bpr_correction_frac=float(correction),
    )


# ═══════════════════════════════════════════════════════════════════════════
# §O.6  Slow Light / EIT from Impedance Matching
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SlowLightResult:
    """Slow light / group velocity result."""
    v_group: float            # group velocity [m/s]
    slowdown_factor: float    # c / v_g
    n_group: float            # group index
    dn_domega: float          # dispersion [s]


def slow_light_group_velocity(
    dn_domega: float,
    n: float = 1.0,
    omega: float = 2.0 * np.pi * 3e14,
    c: float = _c,
) -> SlowLightResult:
    """Group velocity from dispersion.

    v_g = c / (n + omega * dn/domega)

    BPR: slow light = high impedance gradient dZ/d(omega).
    EIT: impedance window opens -> v_g -> 0.

    Parameters
    ----------
    dn_domega : float
        Derivative of refractive index w.r.t. angular frequency [s].
    n : float
        Phase refractive index at the operating frequency.
    omega : float
        Angular frequency [rad/s].
    c : float
        Speed of light [m/s].

    Returns
    -------
    SlowLightResult
        Group velocity, slowdown factor, and group index.

    Examples
    --------
    >>> # Typical EIT: dn/domega ~ 1e-7 s at n ~ 1
    >>> r = slow_light_group_velocity(1e-7, n=1.0, omega=2*np.pi*3e14)
    >>> r.v_group < 1e3  # ~ 600 m/s
    True
    >>> r.slowdown_factor > 1e5
    True
    """
    n_group = n + omega * dn_domega
    if abs(n_group) < 1e-30:
        v_g = float("inf")
    else:
        v_g = c / n_group

    slowdown = c / abs(v_g) if v_g != 0 else float("inf")

    return SlowLightResult(
        v_group=float(v_g),
        slowdown_factor=float(slowdown),
        n_group=float(n_group),
        dn_domega=float(dn_domega),
    )
