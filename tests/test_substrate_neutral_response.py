"""Independent finite-ring excitation and observable checks, not empirical tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr.substrate_neutral_response import (
    neutral_model, translation, momentum_isometry, density_diagonal, bilinear,
    defect_compression, momentum_block, compression_eigenvalues,
    excited_certificate, excited_schur_correction, virtual_ground_component,
    density_certificate, grouped_spectral_measure, case_report, bilinear_diagnostics,
)
from bpr.substrate_fermionization import fixed_number_model


def kron_all(operators):
    result = np.ones((1, 1))
    for operator in operators:
        result = np.kron(result, operator)
    return result


def full_tensor_three(C, g):
    a = np.diag(np.sqrt(np.arange(1, 4)), 1)
    ops = [kron_all([a if x == y else np.eye(4) for x in range(3)])
           for y in range(3)]
    nums = [a.T @ a for a in ops]
    return sum(-C * (ops[x].T @ ops[(x+1) % 3]
                    + ops[(x+1) % 3].T @ ops[x]) for x in range(3)) + sum(
        g/2 * n @ (n-np.eye(64)) for n in nums)


@pytest.mark.parametrize('C,g', [(1., 40.), (.3, .7), (2., 0.)])
def test_complete_three_site_tensor(C, g):
    a = neutral_model(3, C, g)
    indices = [16*s[0]+4*s[1]+s[2] for s in a.model.basis]
    oracle = full_tensor_three(C, g)[np.ix_(indices, indices)]
    np.testing.assert_allclose(a.model.H, oracle, atol=4e-14)
    assert len(indices) == 10


@pytest.mark.parametrize('L,dim', [(3, 10), (4, 35), (5, 126), (6, 462)])
def test_complete_dimensions(L, dim):
    a = neutral_model(L)
    assert len(a.model.basis) == dim
    assert len(a.p1_indices) == L*(L-1)
    assert a.model.basis[a.omega_index] == (1,)*L
    assert all(sum(s) == L for s in a.model.basis)
    assert all(a.model.D[i] == 1 for i in a.p1_indices)
    assert np.linalg.norm(a.B1) > 0


@pytest.mark.parametrize('L', [3, 4, 5])
def test_defect_hopping_independent(L):
    a = neutral_model(L, .8, 40.)
    states = [a.model.basis[i] for i in a.p1_indices]
    expected = 40.*np.eye(len(states))
    for col, s in enumerate(states):
        d, h = s.index(2), s.index(0)
        for sign in (-1, 1):
            nd, nh = (d+sign) % L, (h+sign) % L
            if nd != h:
                target = list(s)
                target[d], target[nd] = 1, 2
                expected[states.index(tuple(target)), col] -= 1.6
            if nh != d:
                target = list(s)
                target[h], target[nh] = 1, 0
                expected[states.index(tuple(target)), col] -= .8
    np.testing.assert_allclose(a.A1, expected, atol=1e-14)
    np.testing.assert_allclose(defect_compression(a), expected, atol=1e-14)


@pytest.mark.parametrize('L', [3, 4, 5])
def test_translation_and_momentum(L):
    a = neutral_model(L)
    T = translation(a.model)
    np.testing.assert_allclose(T.T @ T, np.eye(len(T)), atol=0)
    np.testing.assert_allclose(np.linalg.matrix_power(T, L), np.eye(len(T)), atol=0)
    np.testing.assert_allclose(T @ a.model.H, a.model.H @ T, atol=1e-14)
    pieces = []
    for m in range(L):
        U = momentum_isometry(a, m)
        pieces.append(U)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(L-1), atol=8e-16)
        np.testing.assert_allclose(T @ U, np.exp(2j*np.pi*m/L)*U, atol=1e-15)
        block = U.conj().T @ a.model.H @ U
        np.testing.assert_allclose(block, momentum_block(a, m), atol=3e-14)
        expected = np.sort([40-2*np.sqrt(5+4*np.cos(2*np.pi*m/L))*
                            np.cos(np.pi*j/L) for j in range(1, L)])
        np.testing.assert_allclose(np.linalg.eigvalsh(block), expected, atol=5e-14)
        np.testing.assert_allclose(compression_eigenvalues(a, m), expected, atol=5e-14)
    whole = np.column_stack(pieces)
    np.testing.assert_allclose(whole.conj().T @ whole, np.eye(L*(L-1)), atol=2e-15)
    np.testing.assert_allclose(whole @ whole.conj().T,
                               np.diag((a.model.D == 1).astype(float)), atol=2e-15)


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('g', [40., 80.])
def test_excited_cluster_schur_and_leakage(L, g):
    a = neutral_model(L, 1., g)
    cert = excited_certificate(a)
    assert cert['available']
    v, delta = 2.*L, g-4.*L
    assert cert['v'] == v
    assert cert['delta'] == delta
    values, vectors = np.linalg.eigh(a.model.H)
    mask = (values >= g-v) & (values <= g+v)
    assert mask.sum() == L*(L-1)
    assert np.count_nonzero(values <= v) == 1
    assert np.all(values[1+L*(L-1):] >= 2*g-v)
    for E in [g-v, g, g+v]:
        excluded = a.K1-E*np.eye(len(a.K1))
        assert np.linalg.eigvalsh(excluded).min() < 0
        assert np.linalg.eigvalsh(excluded).max() > 0
        oracle = -a.B1 @ np.linalg.solve(excluded, a.B1.T)
        correction = excited_schur_correction(a, E)
        np.testing.assert_allclose(correction, oracle, atol=2e-14)
        assert np.linalg.norm(correction, 2) <= cert['schur_correction_bound']*(1+1e-13)
    for E, psi in zip(values[mask], vectors[:, mask].T):
        p, q = psi[a.p1_indices], psi[a.q1_indices]
        assert np.linalg.norm(q) <= cert['leakage_ratio_bound']*np.linalg.norm(p)+1e-13
        np.testing.assert_allclose(q, -np.linalg.solve(a.K1-E*np.eye(len(a.K1)),
                                                      a.B1.T @ p), atol=1e-13)
        np.testing.assert_allclose((a.A1+excited_schur_correction(a, E)) @ p,
                                   E*p, atol=2e-12)


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('g', [0., .7, 12.])
def test_unavailable_not_disproved(L, g):
    a = neutral_model(L, 1., g)
    assert not excited_certificate(a)['available']
    c = density_certificate(a, 1)
    assert not c['available']
    if g == 0:
        assert c['leading_weight'] is None
        with pytest.raises((ValueError, TypeError)):
            virtual_ground_component(a)


@pytest.mark.parametrize('L', [3, 4, 5])
def test_density_leading_and_exact_bounds(L):
    a = neutral_model(L)
    omega = np.eye(len(a.model.basis))[:, a.omega_index]
    w = -a.model.V @ omega / a.model.g
    np.testing.assert_allclose(virtual_ground_component(a), w, atol=1e-17)
    assert np.linalg.norm(a.model.V @ omega) == pytest.approx(2*np.sqrt(L))
    E, vec = np.linalg.eigh(a.model.H)
    G = vec[:, 0] * np.sign(vec[a.omega_index, 0])
    chi = G.copy()
    chi[a.omega_index] = 0
    T = translation(a.model)
    for m in range(L):
        rho = density_diagonal(a.model, m)
        expected = np.array([sum(np.exp(-2j*np.pi*m*x/L)*(s[x]-1)
                                  for x in range(L))/np.sqrt(L)
                             for s in a.model.basis])
        np.testing.assert_allclose(rho, expected, atol=2e-15)
        assert rho[a.omega_index] == 0
        np.testing.assert_allclose(T @ (rho*G), np.exp(2j*np.pi*m/L)*(rho*G), atol=1e-14)
        c = density_certificate(a, m)
        leading = 16/40**2*np.sin(np.pi*m/L)**2
        assert c['leading_weight'] == pytest.approx(leading, rel=2e-14, abs=1e-18)
        assert np.linalg.norm(rho*w)**2 == pytest.approx(leading, rel=2e-14, abs=1e-18)
        assert np.linalg.norm(chi-w) <= c['eta']+1e-14
        assert abs(np.linalg.norm(rho*G)**2-leading) <= c['total_weight_error_bound']+1e-14
        assert abs(np.vdot(G, rho*G)) < 1e-13
        if m == 0:
            assert np.count_nonzero(rho) == 0
            assert c['total_weight_error_bound'] == 0
        else:
            assert max(abs(rho)) == pytest.approx(np.sqrt(L))


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('g', [0., .7, 40.])
def test_full_spectral_sum_rules(L, g):
    a = neutral_model(L, 1., g)
    energies, vectors = np.linalg.eigh(a.model.H)
    gaps = energies-energies[0]
    source = density_diagonal(a.model, 1)*vectors[:, 0]
    s = grouped_spectral_measure(gaps, vectors, source)
    assert s['total_weight'] == pytest.approx(np.vdot(source, source).real, abs=1e-13)
    moment = np.vdot(source, (a.model.H-energies[0]*np.eye(len(source))) @ source).real
    assert s['first_moment'] == pytest.approx(moment, abs=3e-12)
    assert sum(x['weight'] for x in s['groups']) == pytest.approx(s['total_weight'])
    assert sum(x['first_moment'] for x in s['groups']) == pytest.approx(moment, abs=3e-12)


def test_degenerate_group_rotation_and_nontransitive_grouping():
    E = np.array([0., 1., 1., 3.])
    U = np.eye(4, dtype=complex)
    z = np.array([.1j, 2., 3j, .4])
    V = U.copy()
    V[1:3, 1:3] = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    one = grouped_spectral_measure(E, U, z, tolerance=1e-12)
    two = grouped_spectral_measure(E, V, z, tolerance=1e-12)
    assert len(one['groups']) == 3
    np.testing.assert_allclose([x['weight'] for x in one['groups']],
                               [x['weight'] for x in two['groups']], atol=1e-14)
    s = grouped_spectral_measure(np.array([0., .75, 1.5]), np.eye(3),
                                 np.ones(3), tolerance=1.)
    assert len(s['groups']) == 2


@pytest.mark.parametrize('L', [3, 4])
def test_neutral_bilinear_algebra_not_car(L):
    a = neutral_model(L)
    ops = {(d, h): bilinear(a.model, d, h) for d in range(L) for h in range(L)}
    zero = np.zeros_like(a.model.H)
    for d in range(L):
        for h in range(L):
            for u in range(L):
                for v in range(L):
                    X, Y = ops[d, h], ops[u, v]
                    rhs = (ops[d, v] if h == u else zero)-(ops[u, h] if d == v else zero)
                    np.testing.assert_allclose(X @ Y-Y @ X, rhs, atol=7e-15)
    np.testing.assert_allclose(sum(ops[x, x] for x in range(L)), L*np.eye(len(zero)), atol=0)
    initial = (0, 2, L-2)+(0,)*(L-3)
    final = (2, 0, L-2)+(0,)*(L-3)
    X = ops[0, 1]
    assert (X @ X)[a.model.basis.index(final), a.model.basis.index(initial)] == pytest.approx(2.)
    assert np.linalg.norm(X @ X.T+X.T @ X-np.eye(len(X))) > 1


@pytest.mark.parametrize('N', [0, 1, 2, 3])
def test_bilinear_diagnostics_small_number_sectors(N):
    report = bilinear_diagnostics(fixed_number_model(3, N))
    assert report['chain_commutator_residual'] < 1e-14
    assert report['reverse_commutator_residual'] < 1e-14
    if N < 2:
        assert report['non_car_witness'] is None
        assert 'N>=2' in report['non_car_witness_unavailable_reason']
    else:
        assert report['non_car_witness_unavailable_reason'] is None
        assert report['non_car_witness']['matrix_element'] == pytest.approx(np.sqrt(2*N*(N-1)))
    assert json.loads(json.dumps(report, allow_nan=False))


@pytest.mark.parametrize('m', [True, 1.5, float('nan'), '1'])
def test_bad_momentum(m):
    a = neutral_model(3)
    for func, arg in [(momentum_isometry, a), (density_diagonal, a.model),
                      (density_certificate, a)]:
        with pytest.raises((ValueError, TypeError)):
            func(arg, m)


@pytest.mark.parametrize('L', [3, 4, 5])
def test_leading_source_lives_in_selected_momentum_block(L):
    a = neutral_model(L)
    w = virtual_ground_component(a)
    for m in range(L):
        U = momentum_isometry(a, m)
        z = density_diagonal(a.model, m)*w
        np.testing.assert_allclose(U @ (U.conj().T @ z), z, atol=2e-16)
        energies, vectors = np.linalg.eigh(momentum_block(a, m))
        measure = grouped_spectral_measure(energies, vectors, U.conj().T @ z)
        assert measure['total_weight'] == pytest.approx(
            density_certificate(a, m)['leading_weight'], abs=1e-17)


def test_integer_momentum_aliases():
    a = neutral_model(3)
    for m in [-2, 10**100, np.int64(4)]:
        np.testing.assert_array_equal(density_diagonal(a.model, m),
                                      density_diagonal(a.model, 1))


@pytest.mark.parametrize('L,C,g', [(7, 1., 40.), (2, 1., 40.), (True, 1., 40.),
                                  (3, 0., 40.), (3, -1., 40.), (3, 1., -1.),
                                  (3, float('inf'), 40.), (3, 1., float('nan')),
                                  (3, 1e-320, 40.)])
def test_model_validation(L, C, g):
    with pytest.raises((ValueError, TypeError)):
        neutral_model(L, C, g)


@pytest.mark.parametrize('scale', [1e-60, 1e60])
def test_common_scale_covariance(scale):
    a, b = neutral_model(3), neutral_model(3, scale, 40*scale)
    np.testing.assert_allclose(b.A1/scale, a.A1, atol=1e-14)
    ca, cb = density_certificate(a, 1), density_certificate(b, 1)
    for key in ['leading_weight', 'eta', 'total_weight_error_bound']:
        assert cb[key] == pytest.approx(ca[key], rel=2e-14)
    np.testing.assert_allclose(excited_schur_correction(b, 40*scale)/scale,
                               excited_schur_correction(a, 40), atol=1e-13)


def test_singular_schur_and_frozen_bound():
    a = neutral_model(5)
    with pytest.raises((ValueError, TypeError)):
        excited_schur_correction(a, 0.)
    c = density_certificate(a, 1)
    assert c['leading_weight'] == pytest.approx(.003454915028125262)
    assert c['total_weight_error_bound'] == pytest.approx(.25625*(.2*np.sin(np.pi/5)+.25625), rel=2e-14)
    assert not c['leading_signal_resolved_by_bound']


def test_near_zero_margin_rejected_not_certified():
    a = neutral_model(3, 1., np.nextafter(12., np.inf))
    with pytest.raises(ValueError):
        excited_certificate(a)


@pytest.mark.parametrize('E', [float('nan'), float('inf')])
def test_schur_nonfinite_energy(E):
    with pytest.raises(ValueError):
        excited_schur_correction(neutral_model(3), E)


@pytest.mark.parametrize('g', [0., .7, 40.])
@pytest.mark.parametrize('m', [0, 1])
def test_report_contract(g, m):
    report = case_report(3, 1., g, m)
    assert report['parameters'] == dict(L=3, N=3, q=3, C=1., g=g, m=m)
    assert report['physical_predictions'] == dict(masses=None, mixing=None)
    assert report['neutrality_assumed'] is True
    assert report['excited_certificate']['available'] == (g > 12)
    assert (report['leading_compression_measure'] is None) == (g <= 12)
    checks = report['numerical_checks']
    assert checks['total_weight_sum_rule_residual'] < 1e-13
    assert checks['first_moment_sum_rule_residual'] < 1e-12
    if m == 0:
        assert report['exact_density_measure']['total_weight'] == 0
    if g == 40:
        assert checks['cluster_counts'] == dict(ground=1, excited=6, higher=3)
    assert json.loads(json.dumps(report, allow_nan=False))


def test_strict_report_json_numpy_and_extreme_resolution():
    report = case_report(np.int64(3), np.float64(1), np.float64(40), np.int64(1))
    assert json.loads(json.dumps(report, allow_nan=False))
    with pytest.raises((ValueError, TypeError)):
        case_report(3, 1., 1e20, 1)


@pytest.mark.parametrize('flag', [[], ['--json']])
def test_stdout_demo_outside_repository(tmp_path, flag):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_neutral_response.py'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')
    result = subprocess.run([sys.executable, str(script)]+flag, cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flag:
        report = json.loads(result.stdout, parse_constant=lambda x: pytest.fail(x))
        assert report
    else:
        assert 'neutral' in result.stdout.lower()
