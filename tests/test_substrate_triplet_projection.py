"""Independent finite-ring checks; fixed algebraic controls, not a parameter scan."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr.condensate_mechanism import lattice_dispersion
from bpr.condensate_regime import energy as ring_energy
from bpr.chiral_flavor_prototype import overlap_matrix
from bpr.quantum_flavor_sources import overlap_operators
from bpr import substrate_triplet_projection as s

PS = (5, 6, 7, 11)
Z = np.array([0.3+0.7j, -0.4+0.2j, 0.6-0.1j])


def independent_u(p):
    return np.array([[np.exp(2j*np.pi*x*k/p)/np.sqrt(p) for k in (-1, 0, 1)] for x in range(p)])


def site_gradient(psi, C, g):
    return -C*(np.roll(psi, 1)+np.roll(psi, -1))+g*abs(psi)**2*psi


@pytest.mark.parametrize("p", PS)
def test_isometry_projector_and_free_invariance(p):
    u = independent_u(p)
    np.testing.assert_allclose(s.fourier_isometry(p), u, atol=3e-15)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(3), atol=1e-15)
    vector = np.exp(0.3j*np.arange(p)) + 0.12*np.arange(p)
    expected = u @ (u.conj().T @ vector)
    np.testing.assert_allclose(s.project(vector), expected, atol=5e-15)
    np.testing.assert_allclose(s.project(expected), expected, atol=5e-15)
    C = 1.3
    hu = -C*(np.roll(u, 1, axis=0)+np.roll(u, -1, axis=0))
    restricted = np.diag(C*lattice_dispersion(p)[[-1, 0, 1]])
    np.testing.assert_allclose(hu, u @ restricted, atol=4e-15)
    np.testing.assert_allclose(u.conj().T @ hu, restricted, atol=4e-15)


@pytest.mark.parametrize("p", PS)
def test_symmetry_conventions_and_singlet_doublet(p):
    u = independent_u(p)
    sym = s.restricted_symmetries(p)
    t, r = sym["translation"], sym["reflection"]
    np.testing.assert_allclose(np.roll(u, -1, axis=0), u @ t, atol=4e-15)
    np.testing.assert_allclose(u[(-np.arange(p)) % p], u @ r, atol=4e-15)
    np.testing.assert_allclose(r @ t @ r, t.conj().T, atol=1e-15)
    np.testing.assert_allclose(np.linalg.matrix_power(t, p), np.eye(3), atol=5e-15)
    np.testing.assert_allclose(r @ r, np.eye(3), atol=1e-15)
    window = s.free_window(p, 1.3)
    energies = 1.3*lattice_dispersion(p)
    assert window["bandwidth"] == pytest.approx(energies[1]-energies[0])
    assert window["isolation_gap"] == pytest.approx(energies[2]-energies[1])
    assert window["gap_to_bandwidth"] == pytest.approx(window["isolation_gap"]/window["bandwidth"])
    assert not window["exactly_degenerate_triplet"]


def test_stable_gap_reports_unshifted_cancellation_without_large_allocation():
    window = s.free_window(10**10)
    assert window["bandwidth"] > 0 and window["isolation_gap"] > 0
    assert not window["unshifted_split_resolved"]
    assert window["unshifted_energies"] == [-2.0]*3
    assert window["gap_to_bandwidth"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="unresolved"):
        s.free_window(10**200)


@pytest.mark.parametrize("p", PS)
def test_source_image_rank_and_toeplitz_independently(p):
    u = independent_u(p)
    # All p site sources, not just the five analytic generators.
    site_sources = np.array([np.outer(row.conj(), row) for row in u])
    real_coordinates = np.concatenate([site_sources.real.reshape(p, 9), site_sources.imag.reshape(p, 9)], axis=1)
    assert np.linalg.matrix_rank(real_coordinates) == 5
    fields = s.source_generators(p)
    assert np.linalg.matrix_rank(fields) == 5
    assert s.real_gram_rank(np.array([s.compress_local_source(f) for f in fields])) == 5
    f = np.sin(np.arange(p)*0.4)+np.arange(p)/3
    matrix = s.compress_local_source(f)
    np.testing.assert_allclose(matrix, u.conj().T @ np.diag(f) @ u, atol=2e-15)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-15)
    for offset in range(-2, 3):
        diagonal = np.diag(matrix, offset)
        np.testing.assert_allclose(diagonal, diagonal[0], atol=1e-15)
    witness = np.diag([1., 0., -1.])
    np.testing.assert_allclose(np.einsum("aij,ij->a", site_sources.conj(), witness), 0, atol=2e-16)
    # Rank is unchanged under a generic complex unitary change of basis.
    q, _ = np.linalg.qr(np.array([[1, 2j, 3], [4j, 5, 6], [7, 8, 10j]]))
    rotated = np.array([q.conj().T @ t @ q for t in site_sources])
    assert s.real_gram_rank(rotated) == 5


def test_all_nine_target_overlaps_independent_quadrature_and_missing_direction():
    direct = np.array([overlap_matrix(row) for row in np.eye(9)])
    np.testing.assert_allclose(direct, overlap_operators(), atol=2e-14)
    assert s.real_gram_rank(direct) == 9
    witness = np.diag([1., 0., -1.])
    coefficients = np.linalg.lstsq(direct.reshape(9, 9).T, witness.reshape(9), rcond=None)[0]
    np.testing.assert_allclose(coefficients.imag, 0, atol=1e-13)
    np.testing.assert_allclose(np.einsum("a,aij->ij", coefficients.real, direct), witness, atol=1e-13)


@pytest.mark.parametrize("p", PS)
def test_generated_associative_algebra_is_full_not_linear_source_image(p):
    units = s.generated_matrix_units(p)
    np.testing.assert_allclose(units, np.eye(9).reshape(9, 3, 3), atol=4e-15)
    assert np.linalg.matrix_rank(units.reshape(9, 9)) == 9
    diag = s.source_matching_diagnostic(p)
    assert diag["direct_source_real_rank"] == 5
    assert diag["target_source_real_rank"] == 9
    assert diag["generated_algebra_complex_rank"] == 9
    assert not diag["direct_matching"]


@pytest.mark.parametrize("p", PS)
def test_local_source_preservation_requires_constant(p):
    u = independent_u(p)
    # Stack the real and imaginary matrix residual of every site generator.
    columns = []
    for f in np.eye(p):
        fu = f[:, None]*u
        columns.append((fu-u @ (u.conj().T @ fu)).ravel())
    linear_map = np.array(columns).T
    assert np.linalg.matrix_rank(np.vstack([linear_map.real, linear_map.imag])) == p-1
    assert s.local_source_closure_residual(np.ones(p))["status"] == "structural_zero"
    diag = s.local_source_closure_residual(s.source_generators(p)[1])
    assert diag["status"] == "resolved_nonzero"
    fu = s.source_generators(p)[1, :, None]*u
    assert diag["raw_norm"] == pytest.approx(np.linalg.norm(fu-u @ (u.conj().T @ fu)))
    almost_constant = np.ones(p)
    almost_constant[0] += np.finfo(float).eps
    assert s.local_source_closure_residual(almost_constant)["status"] == "resolved_nonzero"
    # Removing the exactly invariant constant avoids catastrophic subtraction.
    large_baseline = 1e10+s.source_generators(p)[1]
    centered = large_baseline-large_baseline[0]
    assert s.local_source_closure_residual(large_baseline)["raw_norm"] == pytest.approx(
        s.local_source_closure_residual(centered)["raw_norm"])


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("g", [0., 0.5, 1.7])
def test_quartic_energy_and_gradient_against_existing_ring(p, g):
    u, C = independent_u(p), 1.2
    psi = u @ Z
    assert s.projected_energy(Z, p, C, g) == pytest.approx(ring_energy(psi, C, g), abs=3e-15)
    expected = u.conj().T @ site_gradient(psi, C, g)
    np.testing.assert_allclose(s.projected_gradient(Z, p, C, g), expected, atol=3e-15)
    # Independent Wirtinger-gradient check using the PREVIOUS ring energy API.
    delta = 1e-6
    gradient = s.projected_gradient(Z, p, C, g)
    for i in range(3):
        for phase, component in ((1., gradient[i].real), (1j, gradient[i].imag)):
            direction = np.eye(3)[i]*phase*delta
            derivative = (ring_energy(u @ (Z+direction), C, g)-ring_energy(u @ (Z-direction), C, g))/(2*delta)
            assert derivative == pytest.approx(2*component, abs=8e-10)
    assert np.vdot(Z, gradient).imag == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("p", PS)
def test_gradient_by_independent_fourier_convolution(p):
    spectrum = np.zeros(p, complex)
    spectrum[np.array([-1, 0, 1]) % p] = Z
    cubic = np.zeros(p, complex)
    for i in range(p):
        for j in range(p):
            for k in range(p):
                cubic[(i-j+k) % p] += spectrum[i]*spectrum[j].conjugate()*spectrum[k]
    kinetic = lattice_dispersion(p)[[-1, 0, 1]]*Z
    np.testing.assert_allclose(s.projected_gradient(Z, p, g=0.7), kinetic+0.7/p*cubic[[-1, 0, 1]], atol=1e-15)
    off = s.offband_coefficients(Z, p, 0.7)
    for residue in set(range(p))-{p-1, 0, 1}:
        assert off.get(residue, 0) == pytest.approx(0.7/p*cubic[residue], abs=1e-16)


@pytest.mark.parametrize("p", PS)
def test_leakage_against_direct_site_residual_and_fourier_sum(p):
    u, g = independent_u(p), 0.6
    psi = u @ Z
    nonlinear = g*abs(psi)**2*psi
    residual = nonlinear-u @ (u.conj().T @ nonlinear)
    coeffs = s.offband_coefficients(Z, p, g)
    full_basis = np.exp(2j*np.pi*np.outer(np.arange(p), np.arange(p))/p)/np.sqrt(p)
    dft = full_basis.conj().T @ nonlinear
    for k, value in coeffs.items():
        assert value == pytest.approx(dft[k], abs=1e-15)
    lifted = sum(full_basis[:, k]*value for k, value in coeffs.items())
    np.testing.assert_allclose(lifted, residual, atol=1e-15)
    diag = s.leakage_diagnostic(Z, p, g)
    assert diag["raw_norm"] == pytest.approx(np.linalg.norm(residual))
    assert diag["raw_norm"] <= diag["uniform_residual_bound"]
    assert diag["status"] == "resolved_nonzero"
    assert len(coeffs) == {5: 2, 6: 3}.get(p, 4)


@pytest.mark.parametrize("p", PS)
def test_conjugation_and_global_phase_covariance(p):
    phase = np.exp(0.47j)
    # Conjugating the SITE field reverses Fourier order as well as conjugating.
    zc = Z[::-1].conj()
    np.testing.assert_allclose(independent_u(p) @ zc, (independent_u(p) @ Z).conj(), atol=1e-15)
    np.testing.assert_allclose(s.projected_gradient(zc, p), s.projected_gradient(Z, p)[::-1].conj(), atol=1e-15)
    np.testing.assert_allclose(s.projected_gradient(phase*Z, p), phase*s.projected_gradient(Z, p), atol=1e-15)
    assert s.projected_energy(zc, p) == pytest.approx(s.projected_energy(Z, p))
    assert s.projected_energy(phase*Z, p) == pytest.approx(s.projected_energy(Z, p))
    assert s.leakage_diagnostic(zc, p)["raw_norm"] == pytest.approx(s.leakage_diagnostic(Z, p)["raw_norm"])
    coeff, cc = s.offband_coefficients(Z, p), s.offband_coefficients(zc, p)
    for k in cc:
        assert cc[k] == pytest.approx(coeff[(-k) % p].conjugate())


@pytest.mark.parametrize("p", PS)
def test_exact_zero_controls_and_adjacent_two_mode_nonclosure(p):
    for z in [np.zeros(3), *np.eye(3, dtype=complex)*(0.2+0.7j)]:
        diag = s.leakage_diagnostic(z, p)
        assert diag["status"] == "structural_zero" and diag["raw_norm"] == 0
    assert s.leakage_diagnostic(Z, p, g=0)["raw_norm"] == 0
    z = np.array([0., 1., 1j])/np.sqrt(2)
    diag = s.leakage_diagnostic(z, p)
    assert diag["status"] == "resolved_nonzero"
    assert diag["raw_norm"] == pytest.approx(0.5/(2*np.sqrt(2)*p))


def test_p6_opposite_mode_cancellation_is_not_a_generic_multimode_leakage_claim():
    # a=1,c=exp(i*pi/3): c² a* + a² c* = 0 analytically.
    z = np.array([1., 0., np.exp(1j*np.pi/3)])
    diag = s.leakage_diagnostic(z, 6)
    assert diag["status"] == "unresolved_cancellation"
    assert diag["resolved_norm"] is None
    assert diag["raw_norm"] < 1e-15
    assert not diag["exact_invariant_truncation"]
    assert s.leakage_diagnostic(z, 7)["status"] == "resolved_nonzero"


def test_p5_aliases_are_added_before_norm():
    coeff = s.offband_coefficients(Z, 5, 1.)
    a, b, c = Z
    assert coeff[2] == pytest.approx((c*c*b.conjugate()+2*b*c*a.conjugate()+a*a*c.conjugate())/5)
    assert coeff[3] == pytest.approx((a*a*b.conjugate()+2*b*a*c.conjugate()+c*c*a.conjugate())/5)


@pytest.mark.parametrize("t", [-2., -0.1, 0., 0.1, 1., 2.])
def test_finite_time_bound_formula(t):
    N, p, g = 1.2, 7, 0.5
    diag = s.finite_time_error_bound(N, p, g, t)
    expected = min(2., np.expm1(3*g*N*abs(t))/p)
    assert diag["relative_bound"] == pytest.approx(expected)
    assert diag["absolute_bound"] == pytest.approx(expected*np.sqrt(N))
    assert diag["capped"] == (expected == 2.)


def test_bound_controls_log_cap_and_unresolved_positive_scale():
    assert s.finite_time_error_bound(0, 7, 1, 1e308)["relative_bound"] is None
    assert s.finite_time_error_bound(0, 7, 1, 1e308)["absolute_bound"] == 0
    assert s.finite_time_error_bound(2, 7, 0, 1e308)["relative_bound"] == 0
    enormous = s.finite_time_error_bound(1e308, 7, 1e308, 1e308)
    assert enormous["relative_bound"] == 2 and enormous["capped"]
    assert np.isfinite(enormous["absolute_bound"])
    with pytest.raises(ValueError, match="unresolved"):
        s.finite_time_error_bound(1e-300, 7, 1e-300, 1e-300)
    assert s.interaction_gap_diagnostic(1., 7) == pytest.approx(.5/(7*s.free_window(7)["isolation_gap"]))


@pytest.mark.parametrize("p", [True, 5.0, 4, -1, 0])
def test_invalid_ring_size(p):
    with pytest.raises((TypeError, ValueError)):
        s.fourier_isometry(p)
    with pytest.raises((TypeError, ValueError)):
        s.projected_energy(Z, p)


@pytest.mark.parametrize("C", [0., -1., np.nan, np.inf, True, 1j])
def test_invalid_C(C):
    with pytest.raises((TypeError, ValueError)):
        s.free_window(7, C)


@pytest.mark.parametrize("g", [-1., np.nan, np.inf, True, 1j])
def test_invalid_g(g):
    with pytest.raises((TypeError, ValueError)):
        s.projected_gradient(Z, 7, g=g)
    with pytest.raises((TypeError, ValueError)):
        s.offband_coefficients(Z, 7, g=g)


@pytest.mark.parametrize("z", [[1, 2], [1, 2, 3, 4], [1, np.nan, 2], [1j*np.inf, 0, 0], [1e308, 0, 0], [1e-300, 0, 0]])
def test_invalid_or_unresolved_amplitudes(z):
    for function in (s.projected_energy, s.projected_gradient, s.leakage_diagnostic):
        with pytest.raises((TypeError, ValueError)):
            function(z, 7)


@pytest.mark.parametrize("kwargs", [{"N": -1}, {"N": np.inf}, {"t": np.nan}, {"t": np.inf}, {"g": -1}, {"p": 4}])
def test_invalid_bound_input(kwargs):
    inputs = dict(N=1., p=7, g=.5, t=1.)
    inputs.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        s.finite_time_error_bound(**inputs)


def test_invalid_sources_and_vectors():
    for source in (np.ones(7, complex), [1, 2, np.nan, 4, 5], np.ones((5, 1))):
        with pytest.raises((TypeError, ValueError)):
            s.compress_local_source(source)
    for vector in ([0, np.inf, 0, 0, 0], np.ones((5, 1))):
        with pytest.raises(ValueError):
            s.project(vector)


def test_scaled_gram_and_complex_not_real_associative_algebra():
    matrices = np.array([s.compress_local_source(f) for f in s.source_generators(7)])
    for scale in (1e-200, 1., 1e200):
        assert s.real_gram_rank(scale*matrices) == 5
    reflection = np.eye(3)[::-1]
    # The real generators preserve RK; complex scalar multiplication need not.
    for matrix in matrices:
        np.testing.assert_allclose(reflection @ matrix.conj() @ reflection, matrix, atol=1e-15)
    unit = s.generated_matrix_units(7)[0]
    assert not np.allclose(reflection @ unit.conj() @ reflection, unit)


def test_energy_terms_keep_small_corrections_and_report_total_cancellation():
    terms = s.projected_energy_terms([1, 0, 0], 10**10, g=0)
    assert terms["shifted_kinetic"] > 0 and terms["total"] == -2
    assert not terms["corrections_resolved_in_total"]
    # A pure zero-momentum mode has H=-2N+gN²/(2p), exactly zero for g=4p,N=1.
    terms = s.projected_energy_terms([0, 1, 0], 7, g=28)
    assert terms["total"] == pytest.approx(0, abs=2e-15)
    assert terms["total_cancellation_warning"]


def test_large_free_amplitudes_do_not_form_unneeded_cubics():
    for amplitude in (1e100, 1e150):
        z = [amplitude, 0, 0]
        assert np.isfinite(s.projected_energy(z, 7, g=0))
        assert np.all(np.isfinite(s.projected_gradient(z, 7, g=0)))
        assert s.leakage_diagnostic(z, 7, g=0)["raw_norm"] == 0


def test_tiny_leakage_has_stable_norm_or_explicit_resolution_failure():
    diag = s.leakage_diagnostic([1e-90, 1e-90, 0], 7)
    assert diag["raw_norm"] == pytest.approx(0.5/7*1e-270, rel=1e-12, abs=0)
    assert diag["status"] == "resolved_nonzero"
    for function in (s.offband_coefficients, s.leakage_diagnostic):
        with pytest.raises(ValueError, match="unresolved"):
            function([1e-110, 1e-110, 0], 7)
    with pytest.raises(ValueError, match="unresolved"):
        s.interaction_gap_diagnostic(1e-300, 7, g=1e-300)


def test_mixed_scale_cubic_underflow_is_not_reported_as_closure():
    for function in (s.projected_gradient, s.offband_coefficients, s.leakage_diagnostic):
        with pytest.raises(ValueError, match="unresolved"):
            function([0, 1, 1e-200], 7, g=1e300)


def test_uncapped_finite_time_bound_survives_large_exponential():
    result = s.finite_time_error_bound(1., 10**308, 1., 709.8/3)
    assert not result["capped"]
    assert result["relative_bound"] == pytest.approx(np.exp(709.8-np.log(1e308)), rel=5e-13)
    assert 1 < result["relative_bound"] < 2


def test_json_safe_frozen_demonstration():
    result = json.loads(json.dumps(s.demonstration(), allow_nan=False))
    assert result["parameters"] == {"p": 7, "C": 1., "g": .5, "N": 1.}
    assert result["physical_masses"] is None and result["physical_mixing"] is None
    assert result["source_matching"]["direct_matching"] is False
    assert len(result["examples"]) == 3
    for case in result["examples"]:
        assert case["norm"] == pytest.approx(1.)
    assert result["examples"][0]["leakage"]["status"] == "structural_zero"
    assert all(case["leakage"]["status"] == "resolved_nonzero" for case in result["examples"][1:])


def test_stdout_only_demo_cli_strict_json_and_text(tmp_path):
    script = Path(__file__).resolve().parents[1]/"scripts"/"demo_substrate_triplet_projection.py"
    completed = subprocess.run([sys.executable, str(script), "--json"], cwd=tmp_path,
                               text=True, capture_output=True, check=True)
    def reject_constant(value):
        raise AssertionError(f"non-strict JSON token {value}")
    parsed = json.loads(completed.stdout, parse_constant=reject_constant)
    assert parsed == s.demonstration()
    assert completed.stderr == ""
    text = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                          text=True, capture_output=True, check=True)
    assert text.stdout and text.stderr == ""
    assert not list(tmp_path.iterdir())
