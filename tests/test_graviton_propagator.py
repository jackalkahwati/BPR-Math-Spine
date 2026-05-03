"""
Tests for the BPR induced-gravity graviton propagator scaffold.

These tests verify the narrow MVP: correct spin-2 tensor structure,
normalization against the Sakharov-induced Planck scale, and a finite-p
correction model that vanishes in the GR/continuum limit.
"""

import numpy as np
import pytest


def test_tt_projector_is_transverse_and_traceless():
    from bpr.graviton_propagator import transverse_traceless_projector

    k_vec = np.array([1.0, 2.0, 3.0])
    projector = transverse_traceless_projector(k_vec)

    assert projector.shape == (3, 3, 3, 3)
    assert np.einsum("i,ijkl->jkl", k_vec, projector) == pytest.approx(0.0)
    assert np.einsum("ij,ijkl->kl", np.eye(3), projector) == pytest.approx(0.0)


def test_tt_projector_has_spin2_symmetries():
    from bpr.graviton_propagator import transverse_traceless_projector

    projector = transverse_traceless_projector(np.array([0.0, 0.0, 1.0]))

    assert projector == pytest.approx(np.swapaxes(projector, 0, 1))
    assert projector == pytest.approx(np.swapaxes(projector, 2, 3))
    assert projector == pytest.approx(np.transpose(projector, (2, 3, 0, 1)))


def test_induced_planck_energy_matches_boundary_cutoff_relation():
    from bpr.graviton_propagator import BoundaryGravitonPropagator

    model = BoundaryGravitonPropagator(p=104761)

    assert model.planck_energy_J == pytest.approx(
        model.boundary_cutoff_J * np.sqrt(model.p / (48.0 * np.pi**2)),
        rel=1e-12,
    )
    assert model.boundary_spacing_m == pytest.approx(
        model.hbar_c_J_m / model.boundary_cutoff_J,
        rel=1e-12,
    )


def test_propagator_recovers_gr_limit_when_corrections_disabled():
    from bpr.graviton_propagator import BoundaryGravitonPropagator

    model = BoundaryGravitonPropagator(p=104761)
    k_vec = np.array([0.0, 0.0, 1.0e20])

    corrected = model.propagator(k_vec, include_finite_p=False)
    k_energy_squared = (model.hbar_c_J_m**2) * np.dot(k_vec, k_vec)
    expected = model.spin2_projector(k_vec) / (model.planck_energy_J**2 * k_energy_squared)

    assert corrected == pytest.approx(expected)


def test_public_finite_p_correction_uses_derived_mode_count():
    from bpr.graviton_propagator import (
        boundary_stress_tensor_correction,
        finite_p_correction,
    )

    small_p = finite_p_correction(k_over_cutoff=0.0, p=101)
    large_p = finite_p_correction(k_over_cutoff=0.0, p=10_000_019)
    low_energy = finite_p_correction(k_over_cutoff=0.0, p=10_000_019)

    assert abs(large_p - 1.0) < abs(small_p - 1.0)
    assert low_energy == pytest.approx(
        boundary_stress_tensor_correction(10_000_019).propagator_ratio
    )


def test_boundary_stress_tensor_correction_comes_from_mode_count():
    from bpr.graviton_propagator import boundary_stress_tensor_correction

    correction = boundary_stress_tensor_correction(p=104761)

    assert correction.L_max == int(np.sqrt(104761))
    assert correction.s2_modes == (correction.L_max + 1) ** 2
    assert correction.central_charge_ratio == pytest.approx(
        correction.s2_modes / correction.p
    )
    assert correction.propagator_ratio == pytest.approx(
        correction.p / correction.s2_modes
    )
    assert correction.eta_equivalent == pytest.approx(
        correction.p * (correction.propagator_ratio - 1.0)
    )


def test_derived_tt_correction_approaches_gr_limit():
    from bpr.graviton_propagator import boundary_stress_tensor_correction

    small_p = boundary_stress_tensor_correction(p=101)
    default_p = boundary_stress_tensor_correction(p=104761)
    large_p = boundary_stress_tensor_correction(p=10_000_019)

    assert abs(large_p.propagator_ratio - 1.0) < abs(default_p.propagator_ratio - 1.0)
    assert abs(default_p.propagator_ratio - 1.0) < abs(small_p.propagator_ratio - 1.0)


def test_model_uses_derived_tt_correction_by_default():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        derived_finite_p_correction,
    )

    model = BoundaryGravitonPropagator(p=104761)
    k_vec = np.array([0.0, 0.0, 1.0e20])
    k_over_cutoff = np.linalg.norm(k_vec) / model.cutoff_wavenumber_m_inv

    assert model.correction_for_wavevector(k_vec) == pytest.approx(
        derived_finite_p_correction(k_over_cutoff=k_over_cutoff, p=model.p)
    )


def test_eft_coefficient_retains_derived_mode_count_factor():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        boundary_stress_tensor_correction,
    )

    model = BoundaryGravitonPropagator(p=104761, eft_coefficient=2.0)
    k_vec = np.array([0.0, 0.0, 1.0e20])
    k_over_cutoff = np.linalg.norm(k_vec) / model.cutoff_wavenumber_m_inv
    mode_count_ratio = boundary_stress_tensor_correction(model.p).propagator_ratio

    assert model.correction_for_wavevector(k_vec) == pytest.approx(
        mode_count_ratio * (1.0 + 2.0 * k_over_cutoff**2)
    )


def test_negative_eft_coefficient_is_allowed_because_sign_is_open():
    from bpr.graviton_propagator import BoundaryGravitonPropagator

    model = BoundaryGravitonPropagator(p=104761, eft_coefficient=-0.5)
    k_vec = np.array([0.0, 0.0, 1.0e20])

    assert np.isfinite(model.correction_for_wavevector(k_vec))


def test_curvature_squared_r2_does_not_shift_spin2_tt_sector():
    from bpr.graviton_propagator import curvature_squared_correction

    correction = curvature_squared_correction(p=104761, z=8)
    expected_alpha = (104761 / (384.0 * np.pi**2)) * (8 / 2.0) ** 2

    assert correction.alpha_R2_minimal == pytest.approx(expected_alpha)
    assert correction.spin2_eft_coefficient == 0.0
    assert correction.scalaron_mass_over_planck == pytest.approx(
        1.0 / (2.0 * np.sqrt(correction.alpha_R2_minimal))
    )


def test_scalaron_sector_derives_starobinsky_quantities_from_r2():
    from bpr.graviton_propagator import scalaron_sector_from_boundary_r2

    sector = scalaron_sector_from_boundary_r2(p=104761, z=6)
    expected_alpha = (104761 / (384.0 * np.pi**2)) * (6 / 2.0) ** 2
    expected_n = 104761 ** (1.0 / 3.0) * (1.0 + 1.0 / 3.0)

    assert sector.alpha_R2_minimal == pytest.approx(expected_alpha)
    assert sector.n_efolds == pytest.approx(expected_n)
    assert sector.scalaron_mass_over_planck == pytest.approx(
        1.0 / (2.0 * np.sqrt(expected_alpha))
    )
    assert sector.potential_plateau_over_planck4 == pytest.approx(
        3.0 / (16.0 * expected_alpha)
    )
    assert sector.trace_coupling_over_planck == pytest.approx(1.0 / np.sqrt(6.0))
    assert sector.n_s == pytest.approx(1.0 - 2.0 / expected_n)
    assert sector.tensor_to_scalar_ratio == pytest.approx(12.0 / expected_n**2)


def test_scalaron_minimal_r2_overpredicts_scalar_amplitude():
    from bpr.graviton_propagator import scalaron_sector_from_boundary_r2

    sector = scalaron_sector_from_boundary_r2(p=104761, z=6)
    expected_amplitude = sector.n_efolds**2 / (
        96.0 * np.pi**2 * sector.alpha_R2_minimal
    )

    assert sector.scalar_amplitude_minimal == pytest.approx(expected_amplitude)
    assert sector.scalar_amplitude_minimal > sector.observed_scalar_amplitude
    assert sector.alpha_required_for_observed_amplitude == pytest.approx(
        sector.n_efolds**2 / (96.0 * np.pi**2 * sector.observed_scalar_amplitude)
    )
    assert sector.alpha_gap_factor == pytest.approx(
        sector.alpha_required_for_observed_amplitude / sector.alpha_R2_minimal
    )
    assert sector.alpha_gap_factor > 1.0e6


def test_scalaron_normalization_diagnostic_marks_winding_gap_open():
    from bpr.graviton_propagator import scalaron_normalization_diagnostic

    diagnostic = scalaron_normalization_diagnostic(p=104761, z=6)
    previous = diagnostic.candidate_factors["previous_winding_factor"]

    assert diagnostic.required_alpha_gap == pytest.approx(
        diagnostic.scalaron_sector.alpha_gap_factor
    )
    assert previous == pytest.approx(1.0 + np.sqrt(6 / 2.0) / np.sqrt(np.log(104761)))
    assert previous < 2.0
    assert diagnostic.required_alpha_gap > 1.0e6
    assert diagnostic.best_candidate_name in diagnostic.candidate_factors
    assert diagnostic.best_candidate_relative_error > 0.1
    assert diagnostic.status == "open"


def test_scalaron_normalization_diagnostic_effective_sector_count():
    from bpr.graviton_propagator import scalaron_normalization_diagnostic

    diagnostic = scalaron_normalization_diagnostic(p=104761, z=6)

    assert diagnostic.effective_boundary_sector_count == pytest.approx(
        diagnostic.required_alpha_gap * diagnostic.scalaron_sector.p
    )
    assert diagnostic.effective_boundary_sector_count > 1.0e11


def test_scalaron_sector_rejects_invalid_inputs():
    from bpr.graviton_propagator import scalaron_sector_from_boundary_r2

    for kwargs in (
        {"p": 0},
        {"z": 0},
        {"spatial_dimensions": 0},
        {"observed_scalar_amplitude": 0.0},
        {"observed_scalar_amplitude": np.nan},
    ):
        with pytest.raises(ValueError):
            scalaron_sector_from_boundary_r2(**kwargs)


def test_model_can_use_derived_zero_spin2_eft_coefficient():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        curvature_squared_correction,
    )

    r2 = curvature_squared_correction(p=104761)
    model = BoundaryGravitonPropagator(
        p=104761,
        eft_coefficient=r2.spin2_eft_coefficient,
    )
    default_model = BoundaryGravitonPropagator(p=104761)
    k_vec = np.array([0.0, 0.0, 1.0e20])

    assert model.correction_for_wavevector(k_vec) == pytest.approx(
        default_model.correction_for_wavevector(k_vec)
    )


def test_weyl_squared_spin2_coefficient_is_universal_per_log_interval():
    from bpr.graviton_propagator import (
        boundary_stress_tensor_correction,
        spin2_curvature_squared_correction,
    )

    correction = spin2_curvature_squared_correction(p=104761, renormalization_log=1.0)
    n_modes = boundary_stress_tensor_correction(104761).s2_modes
    expected_beta = n_modes / (960.0 * np.pi**2)
    expected_eta = -expected_beta * (48.0 * np.pi**2 / 104761)

    assert correction.n_modes == n_modes
    assert correction.beta_weyl == pytest.approx(expected_beta)
    assert correction.ricci_squared_equivalent == pytest.approx(2.0 * expected_beta)
    assert correction.spin2_eft_coefficient == pytest.approx(expected_eta)
    assert correction.spin2_eft_coefficient < 0.0


def test_weyl_squared_spin2_coefficient_scales_with_renormalization_log():
    from bpr.graviton_propagator import spin2_curvature_squared_correction

    one_log = spin2_curvature_squared_correction(p=104761, renormalization_log=1.0)
    half_log = spin2_curvature_squared_correction(p=104761, renormalization_log=0.5)

    assert half_log.beta_weyl == pytest.approx(0.5 * one_log.beta_weyl)
    assert half_log.spin2_eft_coefficient == pytest.approx(
        0.5 * one_log.spin2_eft_coefficient
    )


def test_model_can_use_derived_weyl_spin2_eft_coefficient_explicitly():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        spin2_curvature_squared_correction,
    )

    spin2 = spin2_curvature_squared_correction(p=104761, renormalization_log=1.0)
    model = BoundaryGravitonPropagator(
        p=104761,
        eft_coefficient=spin2.spin2_eft_coefficient,
    )
    default_model = BoundaryGravitonPropagator(p=104761)
    k_vec = np.array([0.0, 0.0, 0.1 * default_model.cutoff_wavenumber_m_inv])

    assert model.correction_for_wavevector(k_vec) < default_model.correction_for_wavevector(
        k_vec
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 0},
        {"renormalization_log": -1.0},
        {"renormalization_log": np.inf},
    ],
)
def test_spin2_curvature_squared_rejects_invalid_inputs(kwargs):
    from bpr.graviton_propagator import spin2_curvature_squared_correction

    with pytest.raises(ValueError):
        spin2_curvature_squared_correction(**kwargs)


def test_probe_energy_sets_physical_rg_window():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        spin2_correction_for_probe_energy,
    )

    model = BoundaryGravitonPropagator(p=104761)
    probe_energy_J = 0.1 * model.boundary_cutoff_J
    correction = spin2_correction_for_probe_energy(
        probe_energy_J=probe_energy_J,
        p=model.p,
    )

    assert correction.cutoff_energy_J == pytest.approx(model.boundary_cutoff_J)
    assert correction.energy_ratio == pytest.approx(0.1)
    assert correction.renormalization_log == pytest.approx(np.log(10.0))
    assert correction.fractional_spin2_shift == pytest.approx(
        correction.spin2_eft_coefficient * correction.energy_ratio**2
    )


def test_probe_energy_at_cutoff_has_zero_rg_window():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        spin2_correction_for_probe_energy,
    )

    model = BoundaryGravitonPropagator(p=104761)
    correction = spin2_correction_for_probe_energy(
        probe_energy_J=model.boundary_cutoff_J,
        p=model.p,
    )

    assert correction.renormalization_log == pytest.approx(0.0)
    assert correction.spin2_eft_coefficient == pytest.approx(0.0)
    assert correction.fractional_spin2_shift == pytest.approx(0.0)


def test_probe_energy_helper_can_use_explicit_cutoff():
    from bpr.graviton_propagator import spin2_correction_for_probe_energy

    correction = spin2_correction_for_probe_energy(
        probe_energy_J=2.0,
        p=104761,
        Lambda_b_J=10.0,
    )

    assert correction.cutoff_energy_J == pytest.approx(10.0)
    assert correction.energy_ratio == pytest.approx(0.2)
    assert correction.renormalization_log == pytest.approx(np.log(5.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probe_energy_J": 0.0},
        {"probe_energy_J": np.nan},
        {"probe_energy_J": 2.0, "Lambda_b_J": 1.0},
        {"probe_energy_J": 1.0, "Lambda_b_J": 0.0},
    ],
)
def test_probe_energy_rg_window_rejects_invalid_inputs(kwargs):
    from bpr.graviton_propagator import spin2_correction_for_probe_energy

    with pytest.raises(ValueError):
        spin2_correction_for_probe_energy(**kwargs)


def test_gravitational_wave_frequency_maps_to_probe_energy():
    from bpr.constants import HBAR
    from bpr.graviton_propagator import spin2_correction_for_gw_frequency

    correction = spin2_correction_for_gw_frequency(
        frequency_hz=100.0,
        p=104761,
    )

    assert correction.frequency_hz == pytest.approx(100.0)
    assert correction.angular_frequency_rad_s == pytest.approx(2.0 * np.pi * 100.0)
    assert correction.probe_energy_J == pytest.approx(HBAR * 2.0 * np.pi * 100.0)
    assert correction.fractional_spin2_shift < 0.0
    assert abs(correction.fractional_spin2_shift) < 1.0e-70


def test_gw_frequency_helper_accepts_explicit_cutoff():
    from bpr.constants import HBAR
    from bpr.graviton_propagator import spin2_correction_for_gw_frequency

    cutoff_J = 10.0
    frequency_hz = cutoff_J / (20.0 * np.pi * HBAR)
    correction = spin2_correction_for_gw_frequency(
        frequency_hz=frequency_hz,
        p=104761,
        Lambda_b_J=cutoff_J,
    )

    assert correction.probe_energy_J == pytest.approx(0.1 * cutoff_J)
    assert correction.probe_correction.energy_ratio == pytest.approx(0.1)
    assert correction.probe_correction.renormalization_log == pytest.approx(np.log(10.0))


@pytest.mark.parametrize(
    "frequency_hz",
    [0.0, -1.0, np.nan, np.inf],
)
def test_gw_frequency_helper_rejects_invalid_frequency(frequency_hz):
    from bpr.graviton_propagator import spin2_correction_for_gw_frequency

    with pytest.raises(ValueError):
        spin2_correction_for_gw_frequency(frequency_hz=frequency_hz)


def test_spin2_correction_has_near_cutoff_maximum():
    from bpr.constants import HBAR
    from bpr.graviton_propagator import (
        boundary_stress_tensor_correction,
        spin2_max_fractional_shift,
    )

    maximum = spin2_max_fractional_shift(p=104761)
    n_modes = boundary_stress_tensor_correction(104761).s2_modes
    expected_abs_shift = n_modes / (40.0 * np.e * 104761)

    assert maximum.energy_ratio == pytest.approx(np.exp(-0.5))
    assert maximum.renormalization_log == pytest.approx(0.5)
    assert maximum.fractional_spin2_shift == pytest.approx(-expected_abs_shift)
    assert 0.009 < abs(maximum.fractional_spin2_shift) < 0.01
    assert maximum.probe_energy_J == pytest.approx(
        maximum.energy_ratio * maximum.cutoff_energy_J
    )
    assert maximum.frequency_hz == pytest.approx(
        maximum.probe_energy_J / (2.0 * np.pi * HBAR)
    )


def test_energy_ratio_for_target_spin2_shift_uses_low_energy_branch():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        spin2_correction_for_probe_energy,
        spin2_energy_ratio_for_fractional_shift,
    )

    target = 1.0e-3
    ratio = spin2_energy_ratio_for_fractional_shift(
        target_abs_shift=target,
        p=104761,
    )
    model = BoundaryGravitonPropagator(p=104761)
    correction = spin2_correction_for_probe_energy(
        probe_energy_J=ratio * model.boundary_cutoff_J,
        p=104761,
    )

    assert 0.0 < ratio < np.exp(-0.5)
    assert abs(correction.fractional_spin2_shift) == pytest.approx(target)


def test_energy_ratio_for_tiny_target_spin2_shift_is_stable():
    from bpr.graviton_propagator import (
        BoundaryGravitonPropagator,
        spin2_correction_for_probe_energy,
        spin2_energy_ratio_for_fractional_shift,
    )

    target = 1.0e-78
    ratio = spin2_energy_ratio_for_fractional_shift(
        target_abs_shift=target,
        p=104761,
    )
    model = BoundaryGravitonPropagator(p=104761)
    correction = spin2_correction_for_probe_energy(
        probe_energy_J=ratio * model.boundary_cutoff_J,
        p=104761,
    )

    assert ratio > 0.0
    assert abs(correction.fractional_spin2_shift) / target == pytest.approx(
        1.0,
        rel=1e-10,
    )


@pytest.mark.parametrize(
    "target_abs_shift",
    [0.0, -1.0e-3, np.nan, np.inf, 1.0],
)
def test_energy_ratio_for_target_spin2_shift_rejects_invalid_targets(target_abs_shift):
    from bpr.graviton_propagator import spin2_energy_ratio_for_fractional_shift

    with pytest.raises(ValueError):
        spin2_energy_ratio_for_fractional_shift(target_abs_shift=target_abs_shift)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_over_cutoff": np.nan},
        {"k_over_cutoff": np.inf},
        {"k_over_cutoff": 0.01, "coefficient": np.nan},
        {"k_over_cutoff": 0.01, "coefficient": np.inf},
    ],
)
def test_finite_p_correction_rejects_non_finite_inputs(kwargs):
    from bpr.graviton_propagator import finite_p_correction

    with pytest.raises(ValueError):
        finite_p_correction(**kwargs)


def test_zero_wavevector_is_rejected():
    from bpr.graviton_propagator import transverse_traceless_projector

    with pytest.raises(ValueError, match="nonzero"):
        transverse_traceless_projector(np.zeros(3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 0},
        {"p": -7},
        {"Lambda_b_J": 0.0},
        {"Lambda_b_J": -1.0},
        {"eft_coefficient": np.nan},
    ],
)
def test_invalid_model_parameters_are_rejected(kwargs):
    from bpr.graviton_propagator import BoundaryGravitonPropagator

    with pytest.raises(ValueError):
        BoundaryGravitonPropagator(**kwargs)
