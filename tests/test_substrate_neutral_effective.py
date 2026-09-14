"""Independent SW coefficients and finite-system bounds, not empirical tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from bpr import substrate_neutral_effective as e
from bpr import substrate_neutral_response as old
from bpr import substrate_neutral_continuum as continuum


def comm(a, b):
    return a @ b - b @ a


@pytest.mark.parametrize('L', [3, 4, 5, 6])
def test_generators_and_exact_bch_coefficients(L):
    a = old.neutral_model(L)
    c = e.sw_coefficients(a)
    D = np.diag(a.model.D)
    same = a.model.D[:, None] == a.model.D[None, :]
    for S in (c.S1, c.S2):
        np.testing.assert_allclose(S + S.conj().T, 0., atol=2e-14)
        np.testing.assert_array_equal(S[same], 0.)
    np.testing.assert_allclose(comm(c.S1, D), -c.To, atol=2e-14)
    np.testing.assert_allclose(c.Td + c.To, a.model.V_unit, atol=1e-15)
    second = comm(c.S2, D) + comm(c.S1, a.model.V_unit)
    second += .5 * comm(c.S1, comm(c.S1, D))
    np.testing.assert_allclose(second, c.K, atol=8e-14)
    np.testing.assert_allclose(c.K, c.K.conj().T, atol=3e-14)
    np.testing.assert_array_equal(c.K[~same], 0.)
    assert c.K[a.omega_index, a.omega_index] == pytest.approx(-4*L)
    translation = old.translation(a.model)
    for matrix in (c.S1, c.S2, c.K):
        np.testing.assert_allclose(comm(translation, matrix), 0., atol=4e-14)


@pytest.mark.parametrize('L', [3, 4, 5])
def test_signed_virtual_paths_independently(L):
    a = old.neutral_model(L)
    c = e.sw_coefficients(a)
    p = a.p1_indices
    T = a.model.V_unit
    independent = np.zeros((len(p), len(p)))
    for q in a.q1_indices:
        independent += np.outer(T[p, q], T[q, p])/(1-a.model.D[q])
    np.testing.assert_allclose(c.K[np.ix_(p, p)], independent, atol=6e-14)
    combined = np.zeros_like(independent)
    for channel, matrix in c.p1_channels.items():
        q = np.flatnonzero(a.model.D == channel)
        expected = T[np.ix_(p, q)] @ T[np.ix_(q, p)]/(1-channel)
        np.testing.assert_allclose(matrix, expected, atol=5e-14)
        combined += matrix
        values = np.linalg.eigvalsh(matrix)
        if channel == 0:
            assert values.min() >= -3e-14
            assert values.max() > 0
        else:
            assert values.max() <= 4e-14
    np.testing.assert_allclose(combined, independent, atol=8e-14)
    # D=3 intermediates exist already on the smallest complete ring.
    assert 3 in c.p1_channels


@pytest.mark.parametrize('L', [3, 4, 5])
def test_dressed_density_leading_source_and_orders(L):
    model = e.effective_model(L, .8, 40.)
    a = model.adapter
    for m in range(L):
        s = e.dressed_source(model, m)
        rho = np.diag(old.density_diagonal(a.model, m))
        np.testing.assert_allclose(s.rho, rho, atol=1e-15)
        first = comm(model.coefficients.S1, rho)
        second = comm(model.coefficients.S2, rho)
        second += .5*comm(model.coefficients.S1, first)
        np.testing.assert_allclose(s.rho1, first, atol=3e-14)
        np.testing.assert_allclose(s.rho2, second, atol=5e-14)
        source1 = model.lam*s.z1
        inherited = old.density_diagonal(a.model, m)*old.virtual_ground_component(a)
        np.testing.assert_allclose(source1, inherited, atol=2e-16)
        U = old.momentum_isometry(a, m)
        endpoints = continuum.endpoint_source(L, m, .8, 40.)
        expected = np.zeros(L-1, dtype=complex)
        expected[0], expected[-1] = endpoints
        np.testing.assert_allclose(U.conj().T @ source1, expected, atol=3e-16)
        np.testing.assert_allclose(s.z, model.lam*s.z1+model.lam**2*s.z2, atol=1e-16)
        assert s.weight_order2 == pytest.approx(model.lam**2*np.vdot(s.z1, s.z1).real)
        assert s.weight_order3 == pytest.approx(2*model.lam**3*np.vdot(s.z1, s.z2).real, abs=1e-16)
        assert s.weight_partial_order4 == pytest.approx(model.lam**4*np.vdot(s.z2, s.z2).real)
        assert s.truncated_weight == pytest.approx(np.vdot(s.z, s.z).real)
        if m == 0:
            np.testing.assert_array_equal(s.z, 0.)
            np.testing.assert_array_equal(s.rho2, 0.)


@pytest.mark.parametrize('L', [3, 4])
@pytest.mark.parametrize('g', [40., 80., 160.])
def test_unitary_remainders_spectra_and_dynamics(L, g):
    model = e.effective_model(L, 1., g)
    cert = e.remainder_certificate(model, 1)
    H = model.adapter.model.H
    U = expm(model.S)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(len(H)), atol=2e-15)
    residual = np.linalg.norm(U @ H @ U.conj().T - model.H2, 2)
    assert residual <= cert['energy_bound'] + 2e-12
    assert cert['energy_bound'] == pytest.approx(g*cert['RH'])
    exact_e = np.linalg.eigvalsh(H)
    effective_e = np.linalg.eigvalsh(model.H2)
    assert max(abs(exact_e-effective_e)) <= cert['energy_bound']+2e-12
    source = e.dressed_source(model, 1)
    observable = source.rho+model.lam*source.rho1+model.lam**2*source.rho2
    actual_rho = U @ source.rho @ U.conj().T
    assert np.linalg.norm(actual_rho-observable, 2) <= cert['Rrho']+2e-13
    time = .2
    exact = U @ expm(-1j*time*H) @ U.conj().T
    approx = expm(-1j*time*model.H2)
    assert np.linalg.norm(exact-approx, 2) <= time*cert['energy_bound']+3e-13
    assert model.E0 == pytest.approx(-4*L/g)
    np.testing.assert_allclose(model.gaps1, model.H1-model.E0*np.eye(L*(L-1)), atol=1e-14)


def test_exact_total_density_weight_has_independent_cubic_term():
    for g in (40., 80., 160.):
        model = e.effective_model(3, 1., g)
        _, vec = np.linalg.eigh(model.adapter.model.H)
        rho = old.density_diagonal(model.adapter.model, 1)
        actual = np.linalg.norm(rho*vec[:, 0])**2
        predicted = 12/g**2+72/g**3
        source = e.dressed_source(model, 1)
        error = e.remainder_certificate(model, 1)['source_error_bound']
        # Difference of squared norms plus the explicit partial quartic term.
        bound = error*(2*np.linalg.norm(source.z)+error)+source.weight_partial_order4
        assert abs(actual-predicted) <= bound+1e-14
        assert source.weight_order2+source.weight_order3 == pytest.approx(predicted)


def test_cubic_residual_scaling_without_fitted_exponent():
    residuals = []
    for g in (40., 80., 160.):
        model = e.effective_model(3, 1., g)
        U = expm(model.S)
        residuals.append(np.linalg.norm(U @ model.adapter.model.H @ U.T-model.H2, 2)/g)
    # Fixed asymptotic control; broad inequalities do not fit a power law.
    assert 5 < residuals[0]/residuals[1] < 11
    assert 5 < residuals[1]/residuals[2] < 11


@pytest.mark.parametrize('L,leading,cubic,higher_norm', [(3, 12., 72., np.sqrt(1.5)),
                                                        (4, 8., 0., np.sqrt(2/3))])
def test_independent_ground_series_and_higher_band_witness(L, leading, cubic, higher_norm):
    model = e.effective_model(L)
    a, c = model.adapter, model.coefficients
    T, D = a.model.V_unit, a.model.D
    omega = np.eye(len(D))[:, a.omega_index]
    first = -T @ omega
    second = T @ T @ omega
    second[D > 0] /= D[D > 0]
    second[a.omega_index] = -2*L
    np.testing.assert_allclose(first+c.S1 @ omega, 0., atol=2e-15)
    transformed_second = second+c.S1 @ first+c.S2 @ omega+.5*c.S1 @ c.S1 @ omega
    np.testing.assert_allclose(transformed_second, 0., atol=4e-14)
    source = e.dressed_source(model, 1)
    independent_z2 = source.rho @ second+c.S1 @ source.rho @ first
    np.testing.assert_allclose(source.z2, independent_z2, atol=8e-14)
    assert np.vdot(source.z1, source.z1).real == pytest.approx(leading)
    assert 2*np.vdot(source.z1, source.z2).real == pytest.approx(cubic, abs=2e-13)
    assert np.linalg.norm(source.z2[D == 3]) == pytest.approx(higher_norm)


@pytest.mark.parametrize('L', [3, 4, 5])
def test_phase_aligned_full_ground_source_bound(L):
    model = e.effective_model(L)
    cert = e.remainder_certificate(model, 1)
    assert cert['ground_vector_error_bound'] is not None
    _, vectors = np.linalg.eigh(model.adapter.model.H)
    U = expm(model.S)
    transformed = U @ vectors[:, 0]
    index = model.adapter.omega_index
    phase = np.conj(transformed[index])/abs(transformed[index])
    ground = vectors[:, 0]*phase
    omega = np.eye(len(ground))[:, index]
    assert np.linalg.norm(U @ ground-omega) <= cert['ground_vector_error_bound']+2e-13
    source = e.dressed_source(model, 1)
    assert np.linalg.norm(U @ source.rho @ ground-source.z) <= cert['source_error_bound']+2e-13


@pytest.mark.parametrize('m', [-4, 6, 10**100+1])
def test_integer_alias_and_time_reversal(m):
    model = e.effective_model(5)
    one = e.dressed_source(model, 1)
    alias = e.dressed_source(model, m)
    reverse = e.dressed_source(model, -1)
    np.testing.assert_array_equal(alias.z, one.z)
    np.testing.assert_allclose(reverse.z, one.z.conj(), atol=2e-15)


@pytest.mark.parametrize('scale', [1e-60, 1e60])
def test_common_energy_scale_covariance(scale):
    base = e.effective_model(3, 1., 40.)
    scaled = e.effective_model(3, scale, 40*scale)
    np.testing.assert_allclose(scaled.H2/scale, base.H2, atol=1e-13)
    np.testing.assert_allclose(scaled.gaps1/scale, base.gaps1, atol=1e-13)
    np.testing.assert_allclose(e.dressed_source(scaled, 1).z,
                               e.dressed_source(base, 1).z, atol=3e-16)
    one = e.remainder_certificate(base, 1)
    two = e.remainder_certificate(scaled, 1)
    assert two['RH'] == pytest.approx(one['RH'])
    assert two['energy_bound']/scale == pytest.approx(one['energy_bound'])


def test_frozen_weak_case_does_not_gain_cluster_certificate():
    strong = e.remainder_certificate(e.effective_model(5, 1., 40.), 1)
    weak = e.remainder_certificate(e.effective_model(5, 1., .7), 1)
    assert strong['block_identification_available']
    assert not weak['block_identification_available']
    assert weak['ground_vector_error_bound'] is None
    assert weak['source_error_bound'] is None


def test_allocation_cap_checked_before_basis(monkeypatch):
    from bpr import substrate_fermionization as inherited
    def fail(*args, **kwargs):
        raise AssertionError('occupation enumeration reached')
    monkeypatch.setattr(inherited, '_occupations', fail)
    with pytest.raises(ValueError):
        e.effective_model(7)


@pytest.mark.parametrize('L,C,g', [(2, 1., 40.), (7, 1., 40.), (True, 1., 40.),
                                  (3, 0., 40.), (3, 1., 0.), (3, 1., -1.),
                                  (3, np.inf, 40.), (3, 1., np.nan), (3, 1e-320, 40.)])
def test_invalid_or_uncapped_inputs(L, C, g):
    with pytest.raises((ValueError, TypeError)):
        e.effective_model(L, C, g)


@pytest.mark.parametrize('g,m', [(40., 1), (.7, 1), (40., 0), (.7, 0), (0., 1), (0., 0)])
def test_report_scopes_and_sum_rules(g, m):
    r = e.case_report(3, m, 1., g)
    json.dumps(r, allow_nan=False)
    assert all(x is None for x in r['physical_predictions'].values())
    assert r['exact_sum_rules']['total_weight_residual'] < 1e-12
    assert r['exact_sum_rules']['first_moment_residual'] < 1e-10
    if g == 0:
        assert not r['perturbation_available']
        assert r['effective'] is None
    else:
        effective = r['effective']
        assert effective['partially_resummed']
        assert not effective['individual_weights_certified']
        assert not effective['source_weights']['complete_fourth_order']
        assert not r['remainder_certificate']['roundoff_included']
        for key in ('full_sum_rules', 'P1_sum_rules'):
            assert effective[key]['total_weight_residual'] < 1e-9
            assert effective[key]['first_moment_residual'] < 1e-7
    if m == 0:
        assert r['exact_density_measure']['total_weight'] == 0.


@pytest.mark.parametrize('time', [0., .001, -1., 1e100])
def test_dynamics_bound_saturation(time):
    model = e.effective_model(3)
    expected = min(2., abs(time)*e.remainder_certificate(model)['energy_bound'])
    assert e.dynamics_error_bound(model, time) == pytest.approx(expected)


def test_dynamics_cap_without_unrepresentable_reciprocal():
    large = e.effective_model(3, 2e303, 1.4e303)
    assert e.remainder_certificate(large)['energy_bound'] > 1e308
    assert e.dynamics_error_bound(large, 1.) == 2.
    tiny = e.effective_model(3, 1e-295, 4e-288)
    bound = e.remainder_certificate(tiny)['energy_bound']
    assert e.dynamics_error_bound(tiny, 1.) == pytest.approx(bound, rel=1e-14, abs=0.)
    assert e.dynamics_error_bound(tiny, 1e308) == 2.


def test_diagonalize_without_reciprocal_underflow():
    a = old.neutral_model(3, 1e306, 4e307)
    values, vectors, scale = e._diagonalize(a.model.H)
    expected = np.linalg.eigvalsh(a.model.H/scale)
    np.testing.assert_allclose(values/scale, expected, atol=3e-15)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=3e-15)
    assert np.isfinite(values).all()
    report = e.case_report(3, 1, 1e306, 4e307)
    baseline = e.case_report(3, 1, 1., 40.)
    assert report['exact_density_measure']['total_weight'] == pytest.approx(
        baseline['exact_density_measure']['total_weight'], rel=2e-12)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('flags', [[], ['--json']])
def test_stdout_demo(tmp_path, flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_neutral_effective.py'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    proc = subprocess.run([sys.executable, str(script)]+flags, cwd=tmp_path,
                          env=env, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert not proc.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(proc.stdout, parse_constant=lambda x: pytest.fail(x))
    else:
        assert 'neutral' in proc.stdout.lower()
