"""
Substrate Thermal Heat Flow
============================

Computes predicted steady-state heat flux from a hypothesized hot RPST
substrate (at effective temperature T_eff) into a cold physical device,
through the BPR boundary coupling channel.

**Key physics distinction from energy_balance.py:**
energy_balance.py models a *closed-cycle* process on a conservative
free-energy functional F(W, T, B).  A cycle returns ΔF = 0.

This module models *steady-state heat transfer* from an *external*
thermal reservoir (the substrate at T_eff) through a weak coupling
channel (the BPR boundary).  This is NOT a cycle—it is a continuous
heat flow from hot to cold, limited by the thermal conductance of
the boundary coupling.

Three transfer-law backends avoid dependence on any single model:

  1. ``"landauer"`` — spectral integral with explicit T(ω):

         Q̇ = ∫₀^∞ (dω / 2π) ℏω T(ω) [n(ω, T_eff) − n(ω, T_cold)]

  2. ``"ohmic"``   — conservative Kubo-like:

         Q̇ = α k_B (T_eff − T_cold) Γ_ex

  3. ``"sb_proxy"`` — Stefan–Boltzmann proxy (sanity-check upper bound):

         Q̇ = λ_eff σ (T_eff⁴ − T_cold⁴) A

T_eff is treated as a **hypothesis parameter**, not a fact:

  - ``"entropy_mapping"``: T_eff = J / (k_B ln p)  [CONJECTURE]
  - ``"free_parameter"``:  user-supplied value
  - ``"cosmological"``:    uses p ∼ 10⁶⁰

Usage
-----
>>> from bpr.substrate_heat_flow import SubstrateBath, BoundaryCoupling
>>> from bpr.substrate_heat_flow import compute_heat_flow
>>> bath = SubstrateBath(J_eV=1.0, p=1e5)
>>> coupling = BoundaryCoupling(lambda_eff=6e-7, A=1e-4)
>>> result = compute_heat_flow(bath, coupling, T_cold=4.0, method="landauer")
>>> print(result.summary())

References
----------
Al-Kahwati (2026), BPR-Math-Spine, Test 1C Specification.
Landauer heat transport: Pendry (1983), J. Phys. A.
Fluctuation-dissipation: Kubo (1966), Rep. Prog. Phys.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

try:
    from scipy.integrate import quad as _quad
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover – fallback for minimal envs
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_K_B = 1.380649e-23          # J/K   (Boltzmann)
_HBAR = 1.054571817e-34      # J·s
_C = 299_792_458.0           # m/s
_SIGMA_SB = 5.670374419e-8   # W m⁻² K⁻⁴  (Stefan–Boltzmann)
_EV = 1.602176634e-19        # J per eV


# ═══════════════════════════════════════════════════════════════════════════
#  §1  T_eff computation — hypothesis parameter with multiple models
# ═══════════════════════════════════════════════════════════════════════════

class TeffModel(Enum):
    """Models for the substrate effective temperature."""
    ENTROPY_MAPPING = "entropy_mapping"
    FREE_PARAMETER = "free_parameter"
    COSMOLOGICAL = "cosmological"


@dataclass
class TeffResult:
    """Result of a T_eff computation."""
    T_eff: float               # [K]
    model: str                 # which model produced this
    is_conjecture: bool        # True if based on unverified mapping
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        tag = " [CONJECTURE]" if self.is_conjecture else ""
        return (f"T_eff = {self.T_eff:.2f} K  (model={self.model}{tag})")


def compute_Teff(
    J_eV: float = 1.0,
    p: float = 1e5,
    model: str = "entropy_mapping",
    Teff_override: Optional[float] = None,
) -> TeffResult:
    """Compute substrate effective temperature under a chosen model.

    Parameters
    ----------
    J_eV : float
        Nearest-neighbour coupling energy [eV].
    p : float
        Prime modulus of the ℤ_p substrate.
    model : str
        One of ``"entropy_mapping"``, ``"free_parameter"``, ``"cosmological"``.
    Teff_override : float | None
        Used when ``model="free_parameter"``.

    Returns
    -------
    TeffResult
        Contains T_eff [K], model name, conjecture flag, and warnings.

    Notes
    -----
    ``"entropy_mapping"`` uses  T_eff = J / (k_B ln p).

    This equivalence **only** holds if:
      (a) the coarse-grained substrate degrees of freedom actually thermalise,
      (b) the substrate provides a stationary KMS-like state, and
      (c) the boundary couples to the same d.o.f. that carry T_eff.

    None of these are proven; the formula is a heuristic from boundary_energy.py.
    """
    warns: List[str] = []

    if model == "entropy_mapping":
        if p <= 1:
            raise ValueError(f"p must be > 1, got {p}")
        J_si = J_eV * _EV
        T_eff = J_si / (_K_B * np.log(p))
        warns.append(
            "T_eff derived from entropy mapping (T = J/k_B ln p). "
            "CONJECTURE — requires KMS-state verification."
        )
        return TeffResult(T_eff=T_eff, model=model, is_conjecture=True,
                          warnings=warns)

    elif model == "free_parameter":
        if Teff_override is None:
            raise ValueError(
                "model='free_parameter' requires Teff_override to be set."
            )
        if Teff_override <= 0:
            raise ValueError(f"Teff_override must be > 0 K, got {Teff_override}")
        return TeffResult(T_eff=Teff_override, model=model,
                          is_conjecture=False, warnings=warns)

    elif model == "cosmological":
        # Use p ~ 10^60 as required by dark-energy scaling (impedance.py §4.3)
        p_cosmo = 1e60
        J_si = J_eV * _EV
        T_eff = J_si / (_K_B * np.log(p_cosmo))
        warns.append(
            f"T_eff using cosmological p = {p_cosmo:.0e}. "
            f"Gives T_eff = {T_eff:.2f} K. "
            "CONJECTURE — same caveats as entropy_mapping."
        )
        return TeffResult(T_eff=T_eff, model=model, is_conjecture=True,
                          warnings=warns)

    else:
        raise ValueError(
            f"Unknown Teff model '{model}'. "
            f"Use 'entropy_mapping', 'free_parameter', or 'cosmological'."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Substrate bath and boundary coupling dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubstrateBath:
    """Hypothesised substrate thermal reservoir.

    Attributes
    ----------
    Teff_model : str
        How T_eff is determined (see :func:`compute_Teff`).
    J_eV : float
        Substrate nearest-neighbour coupling [eV].
    p : float
        Prime modulus (lab-scale default 10⁵; cosmological ∼ 10⁶⁰).
    Teff_override : float | None
        User-supplied T_eff when ``Teff_model="free_parameter"``.
    """
    Teff_model: str = "entropy_mapping"
    J_eV: float = 1.0
    p: float = 1e5
    Teff_override: Optional[float] = None

    def T_eff(self) -> TeffResult:
        """Compute T_eff under the selected model."""
        return compute_Teff(
            J_eV=self.J_eV, p=self.p,
            model=self.Teff_model, Teff_override=self.Teff_override,
        )


@dataclass
class BoundaryCoupling:
    """Boundary coupling channel parameters.

    Attributes
    ----------
    lambda_eff : float
        Effective dimensionless coupling (signal transduction strength).
        From stacked_enhancement.py, the phonon channel gives ∼ 6×10⁻⁷
        after derating.
    A : float
        Boundary area [m²].  Default 10⁻⁴ m² (1 cm²).
    spectrum_model : str
        Spectral shape of T(ω): ``"flat"``, ``"ohmic"``, ``"resonant"``.
    omega_c : float | None
        Cutoff angular frequency [rad/s].  Defaults to k_B T_eff / ℏ.
    omega_res : float | None
        Resonance angular frequency [rad/s], used if ``spectrum_model="resonant"``.
    Q_res : float | None
        Quality factor of device resonance.
    transmission_cap : float
        Unitarity cap: T(ω) ≤ transmission_cap ≤ 1.
    """
    lambda_eff: float = 6e-7
    A: float = 1e-4
    spectrum_model: str = "ohmic"
    omega_c: Optional[float] = None
    omega_res: Optional[float] = None
    Q_res: Optional[float] = None
    transmission_cap: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Transmission functions  T(ω)
# ═══════════════════════════════════════════════════════════════════════════

def _bose_einstein(omega: float, T: float) -> float:
    """Bose–Einstein occupation n(ω, T)."""
    if T <= 0 or omega <= 0:
        return 0.0
    x = _HBAR * omega / (_K_B * T)
    if x > 500:
        return 0.0
    return 1.0 / (np.expm1(x))


def transmission_flat(
    omega: float | np.ndarray,
    coupling: BoundaryCoupling,
    omega_c: float,
) -> float | np.ndarray:
    r"""Flat (frequency-independent) transmission.

    .. math::
        \mathcal{T}(\omega) = \min(\mathrm{cap},\; \lambda_\mathrm{eff})
        \quad\text{for } \omega < \omega_c

    This is a bounding estimate: maximum spectral weight.
    """
    omega = np.asarray(omega, dtype=float)
    T_val = min(coupling.transmission_cap, coupling.lambda_eff)
    return np.where(omega < omega_c, T_val, 0.0)


def transmission_ohmic(
    omega: float | np.ndarray,
    coupling: BoundaryCoupling,
    omega_c: float,
) -> float | np.ndarray:
    r"""Ohmic (Drude) transmission.

    .. math::
        \mathcal{T}(\omega) = \lambda_\mathrm{eff}
            \,\frac{\omega}{\omega_c}\,
            e^{-\omega/\omega_c}

    Standard form for dissipative coupling to a bosonic bath.
    Peaks at ω = ω_c with maximum value λ_eff / e.
    """
    omega = np.asarray(omega, dtype=float)
    with np.errstate(over="ignore"):
        raw = coupling.lambda_eff * (omega / omega_c) * np.exp(-omega / omega_c)
    return np.minimum(raw, coupling.transmission_cap)


def transmission_resonant(
    omega: float | np.ndarray,
    coupling: BoundaryCoupling,
    omega_c: float,
) -> float | np.ndarray:
    r"""Resonant (Lorentzian) transmission.

    .. math::
        \mathcal{T}(\omega) = \lambda_\mathrm{eff}
            \frac{(\Gamma/2)^2}{(\omega - \omega_0)^2 + (\Gamma/2)^2}

    where Γ = ω₀/Q.  Captures narrowband coupling through a high-Q
    device resonance.
    """
    omega = np.asarray(omega, dtype=float)
    omega_0 = coupling.omega_res
    Q = coupling.Q_res
    if omega_0 is None or Q is None:
        raise ValueError(
            "spectrum_model='resonant' requires omega_res and Q_res to be set."
        )
    gamma = omega_0 / Q
    half_gamma = gamma / 2.0
    lorentz = half_gamma**2 / ((omega - omega_0)**2 + half_gamma**2)
    raw = coupling.lambda_eff * lorentz
    return np.minimum(raw, coupling.transmission_cap)


def transmission_T(
    omega: float | np.ndarray,
    coupling: BoundaryCoupling,
    omega_c: float,
) -> float | np.ndarray:
    """Dispatch to the selected transmission model.

    Parameters
    ----------
    omega : float | ndarray
        Angular frequency [rad/s].
    coupling : BoundaryCoupling
        Channel parameters.
    omega_c : float
        Cutoff frequency [rad/s].

    Returns
    -------
    float | ndarray
        Dimensionless transmission T(ω) ∈ [0, cap].
    """
    model = coupling.spectrum_model
    if model == "flat":
        return transmission_flat(omega, coupling, omega_c)
    elif model == "ohmic":
        return transmission_ohmic(omega, coupling, omega_c)
    elif model == "resonant":
        return transmission_resonant(omega, coupling, omega_c)
    else:
        raise ValueError(f"Unknown spectrum_model '{model}'.")


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Heat current computations
# ═══════════════════════════════════════════════════════════════════════════

def _default_omega_c(T_eff: float) -> float:
    """Natural cutoff: thermal frequency of substrate, k_B T_eff / ℏ."""
    return _K_B * T_eff / _HBAR


def heat_current_landauer(
    T_eff: float,
    T_cold: float,
    coupling: BoundaryCoupling,
    omega_c: Optional[float] = None,
    n_points: int = 2000,
) -> float:
    r"""Landauer-type spectral heat current.

    .. math::
        \dot{Q} = \int_0^\infty \frac{d\omega}{2\pi}\;
            \hbar\omega\;\mathcal{T}(\omega)\;
            \bigl[n(\omega,T_\mathrm{eff}) - n(\omega,T_\mathrm{cold})\bigr]

    This is the number-of-channels = 1 form.  The total power for the
    device is obtained by multiplying by the effective channel count,
    which here is folded into λ_eff (from stacked_enhancement.py, which
    already includes N², phonon mode ratio, Q, and derating).

    Parameters
    ----------
    T_eff : float
        Substrate effective temperature [K].
    T_cold : float
        Device / cryostat temperature [K].
    coupling : BoundaryCoupling
        Coupling channel parameters.
    omega_c : float | None
        Cutoff frequency.  Defaults to k_B T_eff / ℏ.
    n_points : int
        Number of quadrature points (used if scipy unavailable).

    Returns
    -------
    float
        Heat current Q̇ [W].
    """
    if T_eff <= T_cold:
        return 0.0

    if omega_c is None:
        omega_c = _default_omega_c(T_eff)

    # Integration upper limit: beyond ~10 × ω_c, integrand is negligible
    omega_max = 10.0 * omega_c

    def integrand(omega: float) -> float:
        if omega <= 0:
            return 0.0
        T_w = float(transmission_T(omega, coupling, omega_c))
        dn = _bose_einstein(omega, T_eff) - _bose_einstein(omega, T_cold)
        return (1.0 / (2.0 * np.pi)) * _HBAR * omega * T_w * dn

    if _HAS_SCIPY:
        result, _ = _quad(integrand, 0.0, omega_max, limit=200)
        return float(result)
    else:
        # Trapezoidal fallback
        omegas = np.linspace(1e6, omega_max, n_points)
        vals = np.array([integrand(w) for w in omegas])
        return float(np.trapz(vals, omegas))


def heat_current_ohmic_kubo(
    T_eff: float,
    T_cold: float,
    coupling: BoundaryCoupling,
    alpha: float = 1.0,
    omega_c: Optional[float] = None,
) -> float:
    r"""Conservative Kubo-like heat current estimate.

    .. math::
        \dot{Q} = \alpha\, k_B\, (T_\mathrm{eff} - T_\mathrm{cold})\,
                  \Gamma_\mathrm{ex}

    where :math:`\Gamma_\mathrm{ex} = \lambda_\mathrm{eff} \times \omega_c`
    is the energy exchange rate through the coupling channel.

    This is the bridge from decoherence rate to energy flux:

    - Γ_dec (from decoherence.py) = (k_B T / ℏ)(ΔZ/Z₀)²(A/λ_dB²)  — phase rate
    - Γ_ex  = λ_eff × ω_c  — energy exchange rate, carrying ⟨ℏω⟩ per event
    - α     — O(1) numerical factor, carried explicitly

    Parameters
    ----------
    T_eff, T_cold : float
        Temperatures [K].
    coupling : BoundaryCoupling
        Channel parameters (uses lambda_eff only).
    alpha : float
        Order-one prefactor, default 1.0.
    omega_c : float | None
        Cutoff frequency.  Defaults to k_B T_eff / ℏ.

    Returns
    -------
    float
        Heat current Q̇ [W].
    """
    if T_eff <= T_cold:
        return 0.0
    if omega_c is None:
        omega_c = _default_omega_c(T_eff)
    gamma_ex = coupling.lambda_eff * omega_c
    return alpha * _K_B * (T_eff - T_cold) * gamma_ex


def heat_current_sb_proxy(
    T_eff: float,
    T_cold: float,
    coupling: BoundaryCoupling,
) -> float:
    r"""Stefan–Boltzmann proxy (sanity-check upper bound).

    .. math::
        \dot{Q} = \lambda_\mathrm{eff}\, \sigma\,
                  (T_\mathrm{eff}^4 - T_\mathrm{cold}^4)\, A

    **WARNING**: This assumes blackbody-like coupling (EM modes in free
    space with unit density-of-states).  It is almost certainly the wrong
    transfer law for a weak boundary coupling, but serves as a sanity
    check and an upper bound.

    Parameters
    ----------
    T_eff, T_cold : float
        Temperatures [K].
    coupling : BoundaryCoupling
        Uses lambda_eff and A.

    Returns
    -------
    float
        Heat current Q̇ [W].
    """
    if T_eff <= T_cold:
        return 0.0
    return (
        coupling.lambda_eff
        * _SIGMA_SB
        * (T_eff**4 - T_cold**4)
        * coupling.A
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Unified dispatcher
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeatFlowResult:
    """Full result of a heat-flow prediction.

    Attributes
    ----------
    Qdot_W : float
        Total predicted heat current [W].
    Qdot_per_area : float
        Heat current per unit boundary area [W/m²].
    T_eff : float
        Effective substrate temperature used [K].
    T_cold : float
        Device / cryostat temperature [K].
    method : str
        Transfer-law backend used.
    Teff_result : TeffResult
        Full T_eff computation result (includes warnings / conjecture flag).
    scaling_summary : Dict[str, float]
        Key parameters for scaling analysis.
    warnings : List[str]
        Accumulated warnings.
    params : Dict[str, Any]
        Snapshot of all input parameters.
    """
    Qdot_W: float
    Qdot_per_area: float
    T_eff: float
    T_cold: float
    method: str
    Teff_result: TeffResult
    scaling_summary: Dict[str, float]
    warnings: List[str]
    params: Dict[str, Any]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 72,
            "SUBSTRATE HEAT FLOW PREDICTION",
            "=" * 72,
            "",
            f"  T_eff model     : {self.Teff_result.model}"
            + (" [CONJECTURE]" if self.Teff_result.is_conjecture else ""),
            f"  T_eff           : {self.T_eff:.2f} K",
            f"  T_cold          : {self.T_cold:.2f} K",
            f"  ΔT              : {self.T_eff - self.T_cold:.2f} K",
            f"  Carnot limit    : {1 - self.T_cold / self.T_eff:.4f}"
            if self.T_eff > 0 else "  Carnot limit    : N/A",
            "",
            f"  Transfer law    : {self.method}",
            f"  λ_eff           : {self.params.get('lambda_eff', '?'):.2e}",
            f"  Boundary area   : {self.params.get('A', '?'):.2e} m²",
            f"  Spectrum model  : {self.params.get('spectrum_model', '?')}",
            f"  ω_c             : {self.scaling_summary.get('omega_c', 0):.3e} rad/s",
            "",
            "─" * 40,
            f"  Q̇ (total)       : {self.Qdot_W:.4e} W",
            f"  Q̇ / area        : {self.Qdot_per_area:.4e} W/m²",
            "─" * 40,
            "",
        ]
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    • {w}")
            lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


def compute_heat_flow(
    bath: SubstrateBath,
    coupling: BoundaryCoupling,
    T_cold: float = 4.0,
    method: str = "landauer",
    alpha_ohmic: float = 1.0,
) -> HeatFlowResult:
    """Compute predicted heat flow for given bath + coupling + cold temperature.

    Parameters
    ----------
    bath : SubstrateBath
        Substrate reservoir specification.
    coupling : BoundaryCoupling
        Boundary coupling channel.
    T_cold : float
        Device / cryostat temperature [K].
    method : str
        ``"landauer"``, ``"ohmic"``, or ``"sb_proxy"``.
    alpha_ohmic : float
        O(1) prefactor for the ohmic backend.

    Returns
    -------
    HeatFlowResult
    """
    teff_res = bath.T_eff()
    T_eff = teff_res.T_eff
    all_warnings = list(teff_res.warnings)

    omega_c = coupling.omega_c
    if omega_c is None:
        omega_c = _default_omega_c(T_eff)

    if method == "landauer":
        Qdot = heat_current_landauer(T_eff, T_cold, coupling, omega_c)
    elif method == "ohmic":
        Qdot = heat_current_ohmic_kubo(
            T_eff, T_cold, coupling, alpha=alpha_ohmic, omega_c=omega_c,
        )
    elif method == "sb_proxy":
        Qdot = heat_current_sb_proxy(T_eff, T_cold, coupling)
        all_warnings.append(
            "SB proxy assumes blackbody-like DoS; treat as upper bound only."
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'landauer', 'ohmic', or 'sb_proxy'."
        )

    Qdot_per_area = Qdot / coupling.A if coupling.A > 0 else 0.0

    scaling = {
        "omega_c": omega_c,
        "T_eff": T_eff,
        "T_cold": T_cold,
        "delta_T": T_eff - T_cold,
        "lambda_eff": coupling.lambda_eff,
        "A": coupling.A,
        "Qdot_W": Qdot,
    }

    params = {
        "lambda_eff": coupling.lambda_eff,
        "A": coupling.A,
        "spectrum_model": coupling.spectrum_model,
        "J_eV": bath.J_eV,
        "p": bath.p,
        "method": method,
    }

    return HeatFlowResult(
        Qdot_W=Qdot,
        Qdot_per_area=Qdot_per_area,
        T_eff=T_eff,
        T_cold=T_cold,
        method=method,
        Teff_result=teff_res,
        scaling_summary=scaling,
        warnings=all_warnings,
        params=params,
    )


def compute_all_methods(
    bath: SubstrateBath,
    coupling: BoundaryCoupling,
    T_cold: float = 4.0,
) -> Dict[str, HeatFlowResult]:
    """Run all three backends and return a comparison dict."""
    results: Dict[str, HeatFlowResult] = {}
    for method in ("landauer", "ohmic", "sb_proxy"):
        # For Landauer, match the coupling's spectrum model.
        # For ohmic and sb_proxy, spectrum_model is irrelevant but harmless.
        results[method] = compute_heat_flow(bath, coupling, T_cold, method=method)
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Null configurations (kill switches for steady-state test)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NullConfiguration:
    """A null / control configuration for the heat-flow test."""
    name: str
    description: str
    bath: SubstrateBath
    coupling: BoundaryCoupling
    expected_signal: str     # "zero", "reduced", "unchanged"


def generate_nulls_for_heat_flow(
    baseline_bath: Optional[SubstrateBath] = None,
    baseline_coupling: Optional[BoundaryCoupling] = None,
) -> List[NullConfiguration]:
    """Generate null configurations that a genuine substrate heat signal
    must survive.

    Returns
    -------
    list of NullConfiguration
        Five kill switches.
    """
    if baseline_bath is None:
        baseline_bath = SubstrateBath()
    if baseline_coupling is None:
        baseline_coupling = BoundaryCoupling()

    nulls: List[NullConfiguration] = []

    # NULL-1: No coupling (λ_eff = 0)
    null_coupling_zero = BoundaryCoupling(
        lambda_eff=0.0,
        A=baseline_coupling.A,
        spectrum_model=baseline_coupling.spectrum_model,
        omega_c=baseline_coupling.omega_c,
        omega_res=baseline_coupling.omega_res,
        Q_res=baseline_coupling.Q_res,
    )
    nulls.append(NullConfiguration(
        name="NULL-1: No coupling",
        description=(
            "Set λ_eff = 0 (equivalent to removing boundary-substrate link). "
            "Any residual heat is parasitic."
        ),
        bath=baseline_bath,
        coupling=null_coupling_zero,
        expected_signal="zero",
    ))

    # NULL-2: Inert boundary spectrum (no spectral overlap)
    null_coupling_no_overlap = BoundaryCoupling(
        lambda_eff=baseline_coupling.lambda_eff,
        A=baseline_coupling.A,
        spectrum_model="resonant",
        omega_c=baseline_coupling.omega_c,
        # Resonance far above thermal cutoff → no overlap
        omega_res=1e18,   # ω_res >> ω_c
        Q_res=1e6,
    )
    nulls.append(NullConfiguration(
        name="NULL-2: Inert boundary spectrum",
        description=(
            "Move device resonance far from substrate thermal band. "
            "Coupling exists but spectral overlap → 0."
        ),
        bath=baseline_bath,
        coupling=null_coupling_no_overlap,
        expected_signal="zero",
    ))

    # NULL-3: Normal metal (no superconductivity, no coherent boundary)
    null_coupling_normal_metal = BoundaryCoupling(
        lambda_eff=baseline_coupling.lambda_eff * 1e-10,
        A=baseline_coupling.A,
        spectrum_model=baseline_coupling.spectrum_model,
        omega_c=baseline_coupling.omega_c,
    )
    nulls.append(NullConfiguration(
        name="NULL-3: Normal metal boundary",
        description=(
            "Replace superconducting film with normal metal of same geometry. "
            "No coherent boundary → λ_eff drops by ~10 orders "
            "(loss of N² coherence and Q enhancement)."
        ),
        bath=baseline_bath,
        coupling=null_coupling_normal_metal,
        expected_signal="zero",
    ))

    # NULL-4: Q-suppressed resonator
    null_coupling_low_Q = BoundaryCoupling(
        lambda_eff=baseline_coupling.lambda_eff / 1e8,  # remove Q enhancement
        A=baseline_coupling.A,
        spectrum_model=baseline_coupling.spectrum_model,
        omega_c=baseline_coupling.omega_c,
    )
    nulls.append(NullConfiguration(
        name="NULL-4: Q-suppressed resonator",
        description=(
            "Intentionally degrade resonator Q (e.g., add lossy element). "
            "λ_eff drops by the Q factor."
        ),
        bath=baseline_bath,
        coupling=null_coupling_low_Q,
        expected_signal="reduced",
    ))

    # NULL-5: Swapped area scaling test
    null_coupling_half_area = BoundaryCoupling(
        lambda_eff=baseline_coupling.lambda_eff,
        A=baseline_coupling.A / 2.0,
        spectrum_model=baseline_coupling.spectrum_model,
        omega_c=baseline_coupling.omega_c,
        omega_res=baseline_coupling.omega_res,
        Q_res=baseline_coupling.Q_res,
    )
    nulls.append(NullConfiguration(
        name="NULL-5: Area scaling",
        description=(
            "Halve boundary area.  If signal is substrate heat, Q̇ should "
            "halve (for SB proxy) or change predictably with area."
        ),
        bath=baseline_bath,
        coupling=null_coupling_half_area,
        expected_signal="reduced",
    ))

    return nulls


# ═══════════════════════════════════════════════════════════════════════════
#  §7  Scaling predictions (DOE output curves)
# ═══════════════════════════════════════════════════════════════════════════

def generate_scaling_predictions(
    bath: SubstrateBath,
    coupling: BoundaryCoupling,
    T_cold: float = 4.0,
    method: str = "landauer",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate scaling curves for DOE deliverables.

    Returns a dict with keys:

    - ``"area"``:      Q̇ vs boundary area
    - ``"Teff"``:      Q̇ vs T_eff (scanning J and p)
    - ``"frequency"``: Q̇ vs device resonance frequency
    - ``"Q_factor"``:  Q̇ vs resonator quality factor

    Each value is a dict with ``"x"`` and ``"y"`` arrays.
    """
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    # ── Area scaling ──
    areas = np.logspace(-6, -2, 30)  # 1 mm² → 100 cm²
    Qs_area = []
    for a in areas:
        c = BoundaryCoupling(
            lambda_eff=coupling.lambda_eff, A=a,
            spectrum_model=coupling.spectrum_model,
            omega_c=coupling.omega_c,
            omega_res=coupling.omega_res,
            Q_res=coupling.Q_res,
        )
        r = compute_heat_flow(bath, c, T_cold, method=method)
        Qs_area.append(r.Qdot_W)
    predictions["area"] = {"x": areas, "y": np.array(Qs_area),
                           "xlabel": "A [m²]", "ylabel": "Q̇ [W]"}

    # ── T_eff scaling (via J at fixed p) ──
    J_values = np.logspace(-1, 1, 25)  # 0.1 eV → 10 eV
    Teff_vals = []
    Qs_teff = []
    for J in J_values:
        b = SubstrateBath(Teff_model=bath.Teff_model, J_eV=J, p=bath.p)
        r = compute_heat_flow(b, coupling, T_cold, method=method)
        Teff_vals.append(r.T_eff)
        Qs_teff.append(r.Qdot_W)
    predictions["Teff"] = {"x": np.array(Teff_vals), "y": np.array(Qs_teff),
                           "xlabel": "T_eff [K]", "ylabel": "Q̇ [W]"}

    # ── Resonance frequency scaling ──
    if coupling.spectrum_model == "resonant" and coupling.Q_res is not None:
        freqs = np.logspace(8, 15, 30)  # 100 MHz → PHz
        Qs_freq = []
        for f in freqs:
            c = BoundaryCoupling(
                lambda_eff=coupling.lambda_eff, A=coupling.A,
                spectrum_model="resonant", omega_c=coupling.omega_c,
                omega_res=2.0 * np.pi * f, Q_res=coupling.Q_res,
            )
            r = compute_heat_flow(bath, c, T_cold, method=method)
            Qs_freq.append(r.Qdot_W)
        predictions["frequency"] = {
            "x": freqs, "y": np.array(Qs_freq),
            "xlabel": "f_res [Hz]", "ylabel": "Q̇ [W]",
        }

    # ── Q-factor scaling ──
    Q_factors = np.logspace(1, 12, 30)
    Qs_q = []
    base_lambda_no_Q = coupling.lambda_eff / (coupling.Q_res or 1e8)
    for Q in Q_factors:
        c = BoundaryCoupling(
            lambda_eff=base_lambda_no_Q * Q,
            A=coupling.A,
            spectrum_model=coupling.spectrum_model,
            omega_c=coupling.omega_c,
            omega_res=coupling.omega_res,
            Q_res=Q,
        )
        r = compute_heat_flow(bath, c, T_cold, method=method)
        Qs_q.append(r.Qdot_W)
    predictions["Q_factor"] = {"x": Q_factors, "y": np.array(Qs_q),
                               "xlabel": "Q", "ylabel": "Q̇ [W]"}

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Full analysis report
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    bath: Optional[SubstrateBath] = None,
    coupling: Optional[BoundaryCoupling] = None,
    T_cold: float = 4.0,
) -> str:
    """Run all backends, all nulls, and produce a comprehensive report.

    Returns
    -------
    str
        Multi-section report suitable for inclusion in a DOE milestone doc.
    """
    if bath is None:
        bath = SubstrateBath()
    if coupling is None:
        coupling = BoundaryCoupling()

    lines: List[str] = []

    # ── Header ──
    lines.append("=" * 72)
    lines.append("SUBSTRATE HEAT FLOW — FULL ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")

    # ── All methods comparison ──
    lines.append("§1  THREE-BACKEND COMPARISON")
    lines.append("─" * 40)
    results = compute_all_methods(bath, coupling, T_cold)
    for method, r in results.items():
        lines.append(f"  {method:12s}:  Q̇ = {r.Qdot_W:.4e} W  "
                      f"({r.Qdot_per_area:.4e} W/m²)")
    span = max(r.Qdot_W for r in results.values() if r.Qdot_W > 0)
    floor = min(r.Qdot_W for r in results.values() if r.Qdot_W > 0)
    if floor > 0:
        lines.append(f"\n  Spread: {span / floor:.1e}× "
                      f"(backend uncertainty)")
    lines.append("")

    # ── T_eff details ──
    teff_res = bath.T_eff()
    lines.append("§2  EFFECTIVE TEMPERATURE")
    lines.append("─" * 40)
    lines.append(f"  {teff_res}")
    lines.append(f"  J = {bath.J_eV} eV,  p = {bath.p:.2e}")
    lines.append(f"  T_cold = {T_cold} K")
    if teff_res.is_conjecture:
        lines.append("  ⚠  T_eff IS A CONJECTURE — requires KMS verification.")
    lines.append("")

    # ── Null configurations ──
    lines.append("§3  NULL CONFIGURATIONS")
    lines.append("─" * 40)
    nulls = generate_nulls_for_heat_flow(bath, coupling)
    for null in nulls:
        r = compute_heat_flow(null.bath, null.coupling, T_cold, method="ohmic")
        lines.append(f"  {null.name}")
        lines.append(f"    {null.description}")
        lines.append(f"    Expected: {null.expected_signal}")
        lines.append(f"    Predicted Q̇ = {r.Qdot_W:.4e} W")
        lines.append("")

    # ── Detectability bounds ──
    lines.append("§4  DETECTABILITY")
    lines.append("─" * 40)
    best = results["sb_proxy"]
    conservative = results["ohmic"]
    lines.append(f"  Best-case (SB proxy):    {best.Qdot_W:.4e} W")
    lines.append(f"  Conservative (ohmic):    {conservative.Qdot_W:.4e} W")
    lines.append(f"  Modern calorimetry NEP:  ~10⁻¹⁵ W (TES bolometer)")
    lines.append(f"  SQUID-based:             ~10⁻¹⁸ W")
    lines.append("")
    for label, Qdot in [("best-case", best.Qdot_W),
                         ("conservative", conservative.Qdot_W)]:
        if Qdot > 1e-18:
            lines.append(f"  {label}: ABOVE SQUID noise floor ✓")
        elif Qdot > 0:
            gap = np.log10(1e-18 / Qdot)
            lines.append(f"  {label}: {gap:.1f} orders BELOW SQUID floor")
        else:
            lines.append(f"  {label}: zero signal")
    lines.append("")

    # ── Key caveats ──
    lines.append("§5  CAVEATS AND OPEN QUESTIONS")
    lines.append("─" * 40)
    lines.append("  1. T_eff = J/(k_B ln p) is CONJECTURE. Requires KMS test.")
    lines.append("  2. Ohmic and Landauer results differ by orders of magnitude.")
    lines.append("     The correct answer depends on T(ω), which is unknown.")
    lines.append("  3. λ_eff from stacked_enhancement.py assumes independent,")
    lines.append("     multiplicative enhancement factors — reviewer target #1.")
    lines.append("  4. Steady-state parasitic heat leaks must be bounded below")
    lines.append("     the predicted signal for any detection claim.")
    lines.append("  5. Spectral structure of the signal (if measurable) is the")
    lines.append("     strongest discriminant vs. mundane thermal leaks.")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)
