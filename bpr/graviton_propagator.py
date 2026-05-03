"""
Finite-p graviton propagator scaffold for BPR induced gravity.

This module implements the conservative, leading-order bridge:

    boundary TT correlator -> induced Einstein-Hilbert normalization
    -> transverse-traceless spin-2 propagator

The finite-p TT normalization below is the derived Hopf/S² mode-count factor.
The optional energy-dependent EFT term is deliberately off by default: the
known induced R² sector creates a scalar mode, not a transverse-traceless
spin-2 correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from bpr.constants import C, HBAR, L_PLANCK, P_DEFAULT


def _as_nonzero_k_vector(k_vec: np.ndarray) -> np.ndarray:
    """Return ``k_vec`` as a valid 3-vector with nonzero norm."""
    k = np.asarray(k_vec, dtype=float)
    if k.shape != (3,):
        raise ValueError("k_vec must be a 3-vector")
    if not np.all(np.isfinite(k)):
        raise ValueError("k_vec must contain finite values")
    if np.linalg.norm(k) == 0.0:
        raise ValueError("k_vec must be nonzero")
    return k


def transverse_projector(k_vec: np.ndarray) -> np.ndarray:
    """Project spatial tensors onto the plane transverse to ``k_vec``."""
    k = _as_nonzero_k_vector(k_vec)
    k_hat = k / np.linalg.norm(k)
    return np.eye(3) - np.outer(k_hat, k_hat)


def transverse_traceless_projector(k_vec: np.ndarray) -> np.ndarray:
    """Return the 3D transverse-traceless spin-2 projector.

    The projector is:

        P^TT_ij,kl = 1/2(P_ik P_jl + P_il P_jk - P_ij P_kl)

    where ``P_ij = delta_ij - k_i k_j / |k|^2`` is the transverse projector.
    """
    p_transverse = transverse_projector(k_vec)
    return 0.5 * (
        np.einsum("ik,jl->ijkl", p_transverse, p_transverse)
        + np.einsum("il,jk->ijkl", p_transverse, p_transverse)
        - np.einsum("ij,kl->ijkl", p_transverse, p_transverse)
    )


def finite_p_correction(
    k_over_cutoff: float,
    p: int = P_DEFAULT,
    coefficient: float = 0.0,
) -> float:
    """Public finite-p correction factor for the induced propagator.

    This uses the derived Hopf/S² stress-tensor mode-count factor by default.
    ``coefficient`` controls the still-undetermined EFT term
    ``coefficient * k^2/Lambda_b^2`` and may have either sign.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if not np.isfinite(k_over_cutoff):
        raise ValueError("k_over_cutoff must be finite")
    if k_over_cutoff < 0:
        raise ValueError("k_over_cutoff must be nonnegative")
    if not np.isfinite(coefficient):
        raise ValueError("coefficient must be finite")

    return derived_finite_p_correction(
        k_over_cutoff=k_over_cutoff,
        p=p,
        eft_coefficient=coefficient,
    )


@dataclass(frozen=True)
class BoundaryStressTensorCorrection:
    """Finite-cutoff correction to the boundary stress-tensor normalization."""

    p: int
    L_max: int
    s2_modes: int

    @property
    def central_charge_ratio(self) -> float:
        """Finite S² mode count divided by the ideal CS sector count p."""
        return self.s2_modes / float(self.p)

    @property
    def propagator_ratio(self) -> float:
        """Propagator correction from the inverse induced Planck coefficient."""
        return float(self.p) / self.s2_modes

    @property
    def eta_equivalent(self) -> float:
        """Equivalent eta in the older low-energy form ``1 + eta/p``."""
        return self.p * (self.propagator_ratio - 1.0)


@dataclass(frozen=True)
class CurvatureSquaredCorrection:
    """Boundary-induced curvature-squared correction summary."""

    p: int
    z: int
    alpha_R2_minimal: float

    @property
    def spin2_eft_coefficient(self) -> float:
        """R² changes the scalar sector, not physical TT spin-2 propagation."""
        return 0.0

    @property
    def scalaron_mass_over_planck(self) -> float:
        """Scalaron mass M/M_Pl from the action convention alpha/2 * R²."""
        return 1.0 / (2.0 * np.sqrt(self.alpha_R2_minimal))


@dataclass(frozen=True)
class ScalaronSector:
    """Einstein-frame scalar sector induced by the BPR boundary R² term."""

    p: int
    z: int
    spatial_dimensions: int
    observed_scalar_amplitude: float
    alpha_R2_minimal: float
    n_efolds: float

    @property
    def scalaron_mass_over_planck(self) -> float:
        """Starobinsky mass parameter ``M / M_Pl`` in the repo convention."""
        return 1.0 / (2.0 * np.sqrt(self.alpha_R2_minimal))

    @property
    def potential_plateau_over_planck4(self) -> float:
        """Einstein-frame plateau height ``V0 / M_Pl^4``."""
        return 3.0 / (16.0 * self.alpha_R2_minimal)

    @property
    def trace_coupling_over_planck(self) -> float:
        """Universal scalaron coupling coefficient to trace stress-energy."""
        return 1.0 / np.sqrt(6.0)

    @property
    def scalar_amplitude_minimal(self) -> float:
        """Slow-roll scalar amplitude from the minimal boundary R² term."""
        return self.n_efolds ** 2 / (96.0 * np.pi ** 2 * self.alpha_R2_minimal)

    @property
    def alpha_required_for_observed_amplitude(self) -> float:
        """R² coefficient required to match the observed scalar amplitude."""
        return self.n_efolds ** 2 / (
            96.0 * np.pi ** 2 * self.observed_scalar_amplitude
        )

    @property
    def alpha_gap_factor(self) -> float:
        """Extra enhancement needed beyond the minimal boundary R² coefficient."""
        return self.alpha_required_for_observed_amplitude / self.alpha_R2_minimal

    @property
    def n_s(self) -> float:
        """Leading Starobinsky scalar spectral index."""
        return 1.0 - 2.0 / self.n_efolds

    @property
    def tensor_to_scalar_ratio(self) -> float:
        """Leading Starobinsky tensor-to-scalar ratio."""
        return 12.0 / self.n_efolds ** 2


@dataclass(frozen=True)
class ScalaronNormalizationDiagnostic:
    """Audit candidate normalizations for the scalaron R² amplitude gap."""

    scalaron_sector: ScalaronSector
    candidate_factors: dict[str, float]
    close_threshold: float = 0.1

    @property
    def required_alpha_gap(self) -> float:
        """Required enhancement of the minimal boundary R² coefficient."""
        return self.scalaron_sector.alpha_gap_factor

    @property
    def effective_boundary_sector_count(self) -> float:
        """Effective sector count implied by matching the observed amplitude."""
        return self.scalaron_sector.p * self.required_alpha_gap

    @property
    def best_candidate_name(self) -> str:
        """Candidate with the smallest relative error against the required gap."""
        return min(
            self.candidate_factors,
            key=lambda name: abs(self.candidate_factors[name] / self.required_alpha_gap - 1.0),
        )

    @property
    def best_candidate_relative_error(self) -> float:
        """Relative error of the closest candidate factor."""
        best = self.candidate_factors[self.best_candidate_name]
        return abs(best / self.required_alpha_gap - 1.0)

    @property
    def status(self) -> str:
        """Whether an existing simple candidate closes the amplitude gap."""
        if self.best_candidate_relative_error <= self.close_threshold:
            return "candidate"
        return "open"


@dataclass(frozen=True)
class Spin2CurvatureSquaredCorrection:
    """Universal Weyl/Ricci-squared correction to the TT spin-2 sector."""

    p: int
    n_modes: int
    renormalization_log: float
    beta_weyl: float

    @property
    def ricci_squared_equivalent(self) -> float:
        """Equivalent TT coefficient in a ``beta_Ricci/2 * R_mn R^mn`` basis."""
        return 2.0 * self.beta_weyl

    @property
    def spin2_eft_coefficient(self) -> float:
        """Coefficient of ``k^2/Lambda_b^2`` in the low-energy TT propagator."""
        return -self.beta_weyl * (48.0 * np.pi ** 2 / self.p)


@dataclass(frozen=True)
class Spin2ProbeEnergyCorrection:
    """Spin-2 curvature correction evaluated at a physical probe energy."""

    p: int
    probe_energy_J: float
    cutoff_energy_J: float
    curvature_correction: Spin2CurvatureSquaredCorrection

    @property
    def energy_ratio(self) -> float:
        """Probe energy divided by the BPR boundary cutoff energy."""
        return self.probe_energy_J / self.cutoff_energy_J

    @property
    def renormalization_log(self) -> float:
        """Wilsonian RG window ``log(Lambda_b / mu)``."""
        return self.curvature_correction.renormalization_log

    @property
    def spin2_eft_coefficient(self) -> float:
        """Coefficient multiplying ``(probe_energy / Lambda_b)^2``."""
        return self.curvature_correction.spin2_eft_coefficient

    @property
    def fractional_spin2_shift(self) -> float:
        """Energy-dependent fractional TT propagator shift at this probe."""
        return self.spin2_eft_coefficient * self.energy_ratio ** 2


@dataclass(frozen=True)
class Spin2FrequencyCorrection:
    """Spin-2 correction evaluated for a gravitational-wave frequency."""

    frequency_hz: float
    angular_frequency_rad_s: float
    probe_correction: Spin2ProbeEnergyCorrection

    @property
    def probe_energy_J(self) -> float:
        """Quantum probe energy ``hbar * omega``."""
        return self.probe_correction.probe_energy_J

    @property
    def fractional_spin2_shift(self) -> float:
        """Energy-dependent fractional TT propagator shift at this frequency."""
        return self.probe_correction.fractional_spin2_shift


@dataclass(frozen=True)
class Spin2MaximumCorrection:
    """Maximum EFT spin-2 correction allowed within the cutoff domain."""

    p: int
    cutoff_energy_J: float
    energy_ratio: float
    renormalization_log: float
    fractional_spin2_shift: float

    @property
    def probe_energy_J(self) -> float:
        """Probe energy where ``|delta_TT|`` is maximized."""
        return self.energy_ratio * self.cutoff_energy_J

    @property
    def frequency_hz(self) -> float:
        """Equivalent wave frequency for the maximizing probe energy."""
        return self.probe_energy_J / (2.0 * np.pi * HBAR)


def boundary_stress_tensor_correction(
    p: int = P_DEFAULT,
) -> BoundaryStressTensorCorrection:
    """Derive the finite-p TT normalization from the Hopf/S² mode count.

    The c=1 boundary stress-tensor two-point coefficient is additive in the
    number of retained compact-boson sectors.  At finite CS level, the Hopf
    map keeps spherical harmonic modes up to ``L_max = floor(sqrt(p))``, so
    the retained S² mode count is ``(L_max + 1)^2`` rather than exactly ``p``.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    L_max = math.isqrt(p)
    return BoundaryStressTensorCorrection(
        p=p,
        L_max=L_max,
        s2_modes=(L_max + 1) ** 2,
    )


def curvature_squared_correction(
    p: int = P_DEFAULT,
    z: int = 6,
) -> CurvatureSquaredCorrection:
    """Return the derived minimal R² correction from boundary-mode integration.

    The established curvature-squared term in the repo is the Starobinsky
    ``R²`` term, with ``alpha = (p / 384 pi²) * kappa²`` and
    ``kappa = z/2``.  Around flat space, this term adds a scalar mode but does
    not shift the physical transverse-traceless spin-2 propagator.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if z <= 0:
        raise ValueError("z must be positive")

    kappa = z / 2.0
    alpha_R2 = (p / (384.0 * np.pi ** 2)) * kappa ** 2
    return CurvatureSquaredCorrection(
        p=p,
        z=z,
        alpha_R2_minimal=alpha_R2,
    )


def scalaron_sector_from_boundary_r2(
    p: int = P_DEFAULT,
    z: int = 6,
    spatial_dimensions: int = 3,
    observed_scalar_amplitude: float = 2.1e-9,
) -> ScalaronSector:
    """Derive scalaron-sector quantities from the BPR boundary R² term.

    This uses the same action convention as ``curvature_squared_correction``:
    ``S includes (M_Pl²/2) R + (alpha/2) R²``.  It deliberately reports the
    minimal boundary ``alpha`` separately from the ``alpha`` required by the
    observed scalar amplitude, because the latter still depends on the open
    winding/anyon-loop normalization.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if z <= 0:
        raise ValueError("z must be positive")
    if spatial_dimensions <= 0:
        raise ValueError("spatial_dimensions must be positive")
    if (
        not np.isfinite(observed_scalar_amplitude)
        or observed_scalar_amplitude <= 0.0
    ):
        raise ValueError("observed_scalar_amplitude must be positive and finite")

    curvature = curvature_squared_correction(p=p, z=z)
    n_efolds = p ** (1.0 / 3.0) * (1.0 + 1.0 / spatial_dimensions)
    return ScalaronSector(
        p=p,
        z=z,
        spatial_dimensions=spatial_dimensions,
        observed_scalar_amplitude=observed_scalar_amplitude,
        alpha_R2_minimal=curvature.alpha_R2_minimal,
        n_efolds=n_efolds,
    )


def scalaron_normalization_diagnostic(
    p: int = P_DEFAULT,
    z: int = 6,
    spatial_dimensions: int = 3,
    observed_scalar_amplitude: float = 2.1e-9,
) -> ScalaronNormalizationDiagnostic:
    """Compare known/simple BPR factors to the scalaron amplitude gap.

    This is an audit helper, not a derivation of the missing coefficient.  It
    keeps the old winding estimate visible and checks nearby simple structures
    so the scalar amplitude is not accidentally marked as closed.
    """
    sector = scalaron_sector_from_boundary_r2(
        p=p,
        z=z,
        spatial_dimensions=spatial_dimensions,
        observed_scalar_amplitude=observed_scalar_amplitude,
    )
    ln_p = np.log(p)
    winding_critical = np.sqrt(z / 2.0)
    winding_bare = np.sqrt(ln_p)
    alpha_inv_like = ln_p ** 2 + z / 2.0 + 0.5772156649015329 - 1.0 / (2.0 * np.pi)
    candidate_factors = {
        "previous_winding_factor": 1.0 + winding_critical / winding_bare,
        "p": float(p),
        "p_log_p": float(p * ln_p),
        "p_log_p_squared": float(p * ln_p ** 2),
        "p_alpha_inverse_like": float(p * alpha_inv_like),
        "p_four_thirds": float(p ** (4.0 / 3.0)),
        "p_three_halves": float(p ** 1.5),
        "p_z_squared": float(p * z ** 2),
    }
    return ScalaronNormalizationDiagnostic(
        scalaron_sector=sector,
        candidate_factors=candidate_factors,
    )


def spin2_curvature_squared_correction(
    p: int = P_DEFAULT,
    renormalization_log: float = 1.0,
) -> Spin2CurvatureSquaredCorrection:
    """Return the universal spin-2 curvature-squared coefficient per RG log.

    For each real compact-boson sector, the four-dimensional heat-kernel /
    trace-anomaly coefficient contains

        Gamma_1-loop ⊃ [1 / (120 (4 pi)^2)] log(Λ/μ) ∫ C².

    In this module's action convention, ``S ⊃ (beta_weyl / 2) ∫ C²``, so
    ``beta_weyl = N_modes log(Λ/μ) / (960 pi²)``.  In the TT sector, this is
    equivalent to a Ricci-squared term up to the topological Gauss-Bonnet
    density and scalar ``R²`` pieces.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if not np.isfinite(renormalization_log) or renormalization_log < 0.0:
        raise ValueError("renormalization_log must be finite and nonnegative")

    n_modes = boundary_stress_tensor_correction(p).s2_modes
    beta_weyl = n_modes * renormalization_log / (960.0 * np.pi ** 2)
    return Spin2CurvatureSquaredCorrection(
        p=p,
        n_modes=n_modes,
        renormalization_log=renormalization_log,
        beta_weyl=beta_weyl,
    )


def spin2_correction_for_probe_energy(
    probe_energy_J: float,
    p: int = P_DEFAULT,
    Lambda_b_J: Optional[float] = None,
) -> Spin2ProbeEnergyCorrection:
    """Evaluate the derived spin-2 correction for a physical probe energy.

    ``Lambda_b`` is the BPR boundary cutoff.  If it is omitted, the code uses
    the same Sakharov cutoff convention as ``BoundaryGravitonPropagator``.
    The EFT expression is valid only for probes at or below the cutoff.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if not np.isfinite(probe_energy_J) or probe_energy_J <= 0.0:
        raise ValueError("probe_energy_J must be positive and finite")
    if Lambda_b_J is not None and (
        not np.isfinite(Lambda_b_J) or Lambda_b_J <= 0.0
    ):
        raise ValueError("Lambda_b_J must be positive and finite")

    cutoff_energy_J = (
        BoundaryGravitonPropagator(p=p, Lambda_b_J=Lambda_b_J).boundary_cutoff_J
    )
    if probe_energy_J > cutoff_energy_J:
        raise ValueError("probe_energy_J must not exceed the boundary cutoff")

    renormalization_log = float(np.log(cutoff_energy_J / probe_energy_J))
    curvature_correction = spin2_curvature_squared_correction(
        p=p,
        renormalization_log=renormalization_log,
    )
    return Spin2ProbeEnergyCorrection(
        p=p,
        probe_energy_J=probe_energy_J,
        cutoff_energy_J=cutoff_energy_J,
        curvature_correction=curvature_correction,
    )


def spin2_correction_for_gw_frequency(
    frequency_hz: float,
    p: int = P_DEFAULT,
    Lambda_b_J: Optional[float] = None,
) -> Spin2FrequencyCorrection:
    """Evaluate the spin-2 correction for a GW/ringdown frequency.

    A classical gravitational wave at frequency ``f`` probes the propagator at
    angular frequency ``omega = 2 pi f``.  The corresponding EFT energy scale
    is ``E_probe = hbar * omega``.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")

    angular_frequency_rad_s = 2.0 * np.pi * frequency_hz
    probe_energy_J = HBAR * angular_frequency_rad_s
    probe_correction = spin2_correction_for_probe_energy(
        probe_energy_J=probe_energy_J,
        p=p,
        Lambda_b_J=Lambda_b_J,
    )
    return Spin2FrequencyCorrection(
        frequency_hz=frequency_hz,
        angular_frequency_rad_s=angular_frequency_rad_s,
        probe_correction=probe_correction,
    )


def spin2_max_fractional_shift(
    p: int = P_DEFAULT,
    Lambda_b_J: Optional[float] = None,
) -> Spin2MaximumCorrection:
    """Return the largest derived spin-2 EFT shift below the boundary cutoff.

    With ``x = E_probe / Lambda_b``, the magnitude is proportional to
    ``x^2 log(1/x)``.  This peaks at ``x = exp(-1/2)``.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if Lambda_b_J is not None and (
        not np.isfinite(Lambda_b_J) or Lambda_b_J <= 0.0
    ):
        raise ValueError("Lambda_b_J must be positive and finite")

    cutoff_energy_J = (
        BoundaryGravitonPropagator(p=p, Lambda_b_J=Lambda_b_J).boundary_cutoff_J
    )
    n_modes = boundary_stress_tensor_correction(p).s2_modes
    energy_ratio = float(np.exp(-0.5))
    renormalization_log = 0.5
    fractional_spin2_shift = -n_modes / (40.0 * np.e * p)
    return Spin2MaximumCorrection(
        p=p,
        cutoff_energy_J=cutoff_energy_J,
        energy_ratio=energy_ratio,
        renormalization_log=renormalization_log,
        fractional_spin2_shift=fractional_spin2_shift,
    )


def spin2_energy_ratio_for_fractional_shift(
    target_abs_shift: float,
    p: int = P_DEFAULT,
) -> float:
    """Return the low-energy ``E/Lambda_b`` needed for a target shift.

    The equation ``target = (N_S2/(20p)) x^2 log(1/x)`` has two solutions
    below the cutoff when the target is below the maximum.  This helper returns
    the lower-energy branch, i.e. the first scale where the target appears.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    if not np.isfinite(target_abs_shift) or target_abs_shift <= 0.0:
        raise ValueError("target_abs_shift must be positive and finite")

    max_shift = abs(spin2_max_fractional_shift(p=p).fractional_spin2_shift)
    if target_abs_shift >= max_shift:
        raise ValueError("target_abs_shift must be below the maximum spin-2 shift")

    n_modes = boundary_stress_tensor_correction(p).s2_modes
    amplitude = n_modes / (20.0 * p)
    # Solve on the low-energy branch in y = log(1/x), where
    # target = amplitude * y * exp(-2y), y >= 1/2.  This avoids the absolute
    # precision floor that direct bisection in x hits for tiny targets.
    lo = 0.5
    hi = 1.0
    while amplitude * hi * np.exp(-2.0 * hi) > target_abs_shift:
        hi *= 2.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = amplitude * mid * np.exp(-2.0 * mid)
        if value > target_abs_shift:
            lo = mid
        else:
            hi = mid
    return float(np.exp(-0.5 * (lo + hi)))


def derived_finite_p_correction(
    k_over_cutoff: float,
    p: int = P_DEFAULT,
    eft_coefficient: float = 0.0,
) -> float:
    """Finite-p propagator factor from the boundary TT mode-count correction.

    The low-energy factor is ``p / N_S2`` because the propagator is inverse to
    the induced Einstein-Hilbert coefficient.  The separate EFT correction is
    optional; callers can use ``spin2_correction_for_probe_energy`` to derive
    its coefficient for a physical probe scale.
    """
    if not np.isfinite(k_over_cutoff):
        raise ValueError("k_over_cutoff must be finite")
    if k_over_cutoff < 0:
        raise ValueError("k_over_cutoff must be nonnegative")
    if not np.isfinite(eft_coefficient):
        raise ValueError("eft_coefficient must be finite")
    tt_correction = boundary_stress_tensor_correction(p)
    return tt_correction.propagator_ratio * (
        1.0 + eft_coefficient * k_over_cutoff ** 2
    )


@dataclass
class BoundaryGravitonPropagator:
    """Leading BPR graviton propagator from induced boundary gravity.

    Parameters
    ----------
    p:
        BPR prime / Chern-Simons level.
    Lambda_b_J:
        Boundary UV cutoff energy.  If omitted, the remaining dimensionful
        anchor is the observed Planck length and the Sakharov relation fixes
        the cutoff.
    eft_coefficient:
        Optional coefficient for the EFT cutoff envelope ``k^2/Lambda_b^2``.
        Defaults to zero so the public propagator uses only the derived Hopf/S²
        stress-tensor normalization.  Use
        ``spin2_curvature_squared_correction(...).spin2_eft_coefficient`` to
        opt into the universal Weyl/Ricci-squared coefficient for a chosen
        renormalization log.
    """

    p: int = P_DEFAULT
    Lambda_b_J: Optional[float] = None
    eft_coefficient: float = 0.0

    def __post_init__(self) -> None:
        """Validate physical model parameters early."""
        if self.p <= 0:
            raise ValueError("p must be positive")
        if self.Lambda_b_J is not None and (
            not np.isfinite(self.Lambda_b_J) or self.Lambda_b_J <= 0.0
        ):
            raise ValueError("Lambda_b_J must be positive and finite")
        if (
            not np.isfinite(self.eft_coefficient)
        ):
            raise ValueError("eft_coefficient must be finite")

    @property
    def hbar_c_J_m(self) -> float:
        """Return hbar*c in SI units."""
        return HBAR * C

    @property
    def boundary_spacing_m(self) -> float:
        """Boundary lattice spacing from the Sakharov relation."""
        if self.Lambda_b_J is not None:
            return self.hbar_c_J_m / self.Lambda_b_J
        return L_PLANCK * np.sqrt(self.p / (48.0 * np.pi ** 2))

    @property
    def boundary_cutoff_J(self) -> float:
        """Boundary UV cutoff energy Lambda_b = hbar*c/a."""
        return self.hbar_c_J_m / self.boundary_spacing_m

    @property
    def planck_energy_J(self) -> float:
        """Induced Planck energy from M_Pl = Lambda_b sqrt(p/(48 pi^2))."""
        return self.boundary_cutoff_J * np.sqrt(self.p / (48.0 * np.pi ** 2))

    @property
    def cutoff_wavenumber_m_inv(self) -> float:
        """Boundary cutoff as a wavenumber, 1/a."""
        return 1.0 / self.boundary_spacing_m

    def spin2_projector(self, k_vec: np.ndarray) -> np.ndarray:
        """Return the transverse-traceless spin-2 projector for ``k_vec``."""
        return transverse_traceless_projector(k_vec)

    def correction_for_wavevector(self, k_vec: np.ndarray) -> float:
        """Return the finite-p correction factor for ``k_vec``."""
        k = _as_nonzero_k_vector(k_vec)
        k_over_cutoff = np.linalg.norm(k) / self.cutoff_wavenumber_m_inv
        return derived_finite_p_correction(
            k_over_cutoff,
            p=self.p,
            eft_coefficient=self.eft_coefficient,
        )

    def propagator(
        self,
        k_vec: np.ndarray,
        include_finite_p: bool = True,
    ) -> np.ndarray:
        """Return the leading spatial TT graviton propagator in momentum space.

        The natural-units GR limit is ``P_TT / (M_Pl^2 |k|^2)``.  Since this
        API accepts SI wavevectors in ``m^-1``, ``|k|`` is converted to energy
        by ``hbar*c*|k|``.
        """
        k = _as_nonzero_k_vector(k_vec)
        k_energy_squared = self.hbar_c_J_m ** 2 * float(np.dot(k, k))
        propagator = self.spin2_projector(k) / (
            self.planck_energy_J ** 2 * k_energy_squared
        )
        if include_finite_p:
            propagator = propagator * self.correction_for_wavevector(k)
        return propagator
