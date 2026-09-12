"""Independent finite-chain and weak-limit checks, not empirical validation."""
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from bpr import substrate_neutral_continuum as c
from bpr import substrate_neutral_response as old


@pytest.mark.parametrize('L', [3, 4, 5, 6])
def test_full_fock_source_and_independent_chain(L):
    adapter = old.neutral_model(L, .8, 40.)
    w = old.virtual_ground_component(adapter)
    for m in range(1, L):
        U = old.momentum_isometry(adapter, m)
        source = U.conj().T @ (old.density_diagonal(adapter.model, m)*w)
        endpoints = c.endpoint_source(L, m, .8, 40.)
        expected = np.zeros(L-1, dtype=complex)
        expected[0], expected[-1] = endpoints
        np.testing.assert_allclose(source, expected, atol=3e-16)
        # The existing full Fock compression is independent of the new formula.
        block = U.conj().T @ adapter.model.H @ U
        energies, vectors = np.linalg.eigh(block)
        weights = np.abs(vectors.conj().T @ source)**2
        r = c.finite_compression_measure(L, m, .8, 40.)
        np.testing.assert_allclose(r['energies'], energies, atol=5e-13)
        np.testing.assert_allclose(r['absolute_weights'], weights, atol=2e-16)
        assert r['total_weight'] == pytest.approx(np.vdot(source, source).real)


@pytest.mark.parametrize('m', [1, 2])
def test_three_site_interference_is_asymmetric(m):
    r = c.finite_compression_measure(3, m)
    np.testing.assert_allclose(r['probabilities'],
                               [(2+np.sqrt(3))/4, (2-np.sqrt(3))/4], atol=1e-15)
    assert r['probabilities'] @ r['dimensionless_offsets'] == pytest.approx(-1.5)
    assert r['total_weight'] == pytest.approx(12/40**2)


@pytest.mark.parametrize('L', [3, 4, 5, 6, 16, 64, 256])
def test_mass_maximum_time_reversal_and_moments(L):
    r = c.finite_compression_measure(L, 1)
    p, s = r['probabilities'], r['dimensionless_offsets']
    assert np.sum(p) == pytest.approx(1., abs=2e-14)
    assert np.all(p >= 0)
    assert max(p) <= 4/L+2e-15
    assert max(r['absolute_weights']) <= 4*r['total_weight']/L+1e-17
    reverse = c.finite_compression_measure(L, L-1)
    np.testing.assert_allclose(reverse['probabilities'], p, atol=2e-14)
    for d in range(min(L-2, 7)):
        expected = c.semicircle_moment(d, r['k'])
        assert p @ s**d == pytest.approx(expected, rel=2e-12, abs=2e-10)


@pytest.mark.parametrize('L', [3, 4, 5, 6, 7])
def test_first_nonmatching_moment(L):
    r = c.finite_compression_measure(L, 1)
    d = L-2
    observed = r['probabilities'] @ r['dimensionless_offsets']**d
    expected = c.semicircle_moment(d, r['k'])+r['beta']*(-r['a'])**d
    assert observed == pytest.approx(expected, abs=3e-11)


@pytest.mark.parametrize('L', [4, 6, 16, 64])
def test_pi_structural_dark_parity(L):
    r = c.finite_compression_measure(L, L//2)
    np.testing.assert_array_equal(r['probabilities'][1::2], 0.)
    np.testing.assert_array_equal(r['structural_dark_indices'], np.arange(1, L-1, 2))
    assert r['a'] == 1.
    assert sum(r['probabilities']) == pytest.approx(1.)


@pytest.mark.parametrize('k', [0., .3, np.pi/2, np.pi, -np.pi])
def test_semicircle_kernel_independent_quadrature(k):
    a = abs(2+np.exp(-1j*k))
    assert c.semicircle_density(-2*a-1, k) == 0.
    assert c.semicircle_density(2*a+1, k) == 0.
    for d in range(5):
        integral = quad(lambda s: s**d*c.semicircle_density(s, k),
                        -2*a, 2*a, epsabs=2e-10)[0]
        assert integral == pytest.approx(c.semicircle_moment(d, k), abs=2e-9)
    absolute = quad(lambda s: abs(s)*c.semicircle_density(s, k), -2*a, 2*a)[0]
    assert absolute == pytest.approx(8*a/(3*np.pi))
    assert c.semicircle_density(0., k) == pytest.approx(1/(np.pi*a))
    json.dumps(c.limiting_shape(k), allow_nan=False)


@pytest.mark.parametrize('L', [16, 64, 256])
@pytest.mark.parametrize('fixed_mode', [False, True])
def test_lipschitz_bound_and_momentum_mismatch(L, fixed_mode):
    m = 1 if fixed_mode else L//4
    r = c.finite_compression_measure(L, m)
    target = 0. if fixed_mode else np.pi/2
    a = abs(2+np.exp(-1j*target))
    p, s = r['probabilities'], r['dimensionless_offsets']
    for values, expected, K in [(s, 0., 1.), (s*s, a*a, 12.),
                                (abs(s), 8*a/(3*np.pi), 1.)]:
        bound = c.weak_limit_bound(L, m, target, K)
        assert abs(p @ values-expected) <= bound+2e-12
    distance = abs(r['k']-target)
    assert c.weak_limit_bound(L, m, target) == pytest.approx(
        32*r['a']/(3*L)+2*distance)
    if not fixed_mode:
        assert r['total_weight'] == pytest.approx(.005)
    else:
        leading = 16*np.pi**2/(40**2*L**2)
        assert r['total_weight']/leading == pytest.approx(np.sinc(1/L)**2)


@pytest.mark.parametrize('k', [0., np.pi/2, np.pi])
def test_density_support_edges(k):
    a = c.limiting_shape(k)['a']
    for edge in [-2*a, 2*a]:
        assert c.semicircle_density(edge, k) == 0.
        inner = np.nextafter(edge, 0.)
        assert c.semicircle_density(inner, k) > 0.


def test_circular_distance_and_zero_lipschitz():
    L, m = 64, 31
    r = c.finite_compression_measure(L, m)
    bound = c.weak_limit_bound(L, m, -np.pi)
    assert bound == pytest.approx(32*r['a']/(3*L)+2*np.pi/32)
    assert c.weak_limit_bound(L, m, 0., 0.) == 0.


def test_near_seam_distance_does_not_cancel_to_zero():
    L, m = 10**17, 10**17//2-4
    k = float(Fraction(m, L)*Fraction.from_float(2*np.pi))
    distance = float(Fraction.from_float(np.pi)-Fraction.from_float(k))
    assert distance > 0
    a = c.limiting_shape(k)['a']
    expected = 32*a/(3*L)+2*distance
    assert c.weak_limit_bound(L, m, -np.pi) == pytest.approx(expected, rel=2e-15, abs=0.)
    assert c.weak_limit_bound(L, -m, np.pi) == pytest.approx(expected, rel=2e-15, abs=0.)


@pytest.mark.parametrize('which', ['shape', 'density', 'moment'])
def test_giant_real_input_standardized_error(which):
    with pytest.raises(ValueError, match='unresolved'):
        if which == 'shape':
            c.limiting_shape(10**400)
        elif which == 'density':
            c.semicircle_density(10**400, 0.)
        else:
            c.semicircle_moment(2, 10**400)


@pytest.mark.parametrize('m', [-4, 10**100+1, np.int64(6)])
def test_exact_integer_aliases(m):
    a = c.finite_compression_measure(5, m)
    b = c.finite_compression_measure(5, 1)
    np.testing.assert_array_equal(a['probabilities'], b['probabilities'])
    np.testing.assert_array_equal(c.endpoint_source(5, m), c.endpoint_source(5, 1))


def test_zero_mode_and_free_source_are_distinct():
    zero = c.finite_compression_measure(5, 0)
    assert zero['total_weight'] == 0
    assert zero['probabilities'] is None
    assert not zero['normalization_available']
    np.testing.assert_array_equal(zero['absolute_weights'], 0.)
    np.testing.assert_array_equal(c.endpoint_source(5, 0), 0.)
    for m in [0, 1]:
        free = c.finite_compression_measure(5, m, 1., 0.)
        assert free['probabilities'] is None
        assert free['absolute_weights'] is None
        assert free['total_weight'] is None
        with pytest.raises(ValueError):
            c.endpoint_source(5, m, 1., 0.)
    with pytest.raises(ValueError):
        c.weak_limit_bound(5, 0, 0.)


@pytest.mark.parametrize('scale', [1e-60, 1e60])
def test_scale_covariance(scale):
    a = c.finite_compression_measure(5, 1)
    b = c.finite_compression_measure(5, 1, scale, 40*scale)
    np.testing.assert_array_equal(a['dimensionless_offsets'], b['dimensionless_offsets'])
    np.testing.assert_array_equal(a['probabilities'], b['probabilities'])
    np.testing.assert_allclose(a['energies'], b['energies']/scale)
    np.testing.assert_allclose(a['absolute_weights'], b['absolute_weights'])


@pytest.mark.parametrize('C,g', [(1e100, 1e-200), (1e-150, 1e5)])
def test_unrepresentable_absolute_scale_preserves_probabilities(C, g):
    r = c.finite_compression_measure(5, 1, C, g)
    assert r['total_weight'] is None
    assert r['absolute_weights'] is None
    assert r['absolute_weight_reason']
    assert sum(r['probabilities']) == pytest.approx(1.)


def test_raw_underflow_and_energy_collapse_preserve_shape():
    r = c.finite_compression_measure(5, 1, 1e-200, 1e100)
    assert r['probabilities'] is not None
    assert sum(r['probabilities']) == pytest.approx(1.)
    assert r['total_weight'] is None
    assert r['absolute_weights'] is None
    assert r['absolute_weight_reason']
    assert r['energies'] is None
    assert r['energy_reason']
    assert np.ptp(r['dimensionless_offsets']) > 1


@pytest.mark.parametrize('L', [3, 5, 9, 10, 16, 10**100])
def test_exact_threshold_audit(L):
    for g in [0., 4., 40., np.nextafter(40., np.inf)]:
        r = c.validity_audit(L, 1., g)
        assert r['population_selection_sufficient'] == (g >= 4)
        assert r['excitation_separation_sufficient'] == (Fraction.from_float(g) > 4*L)
        json.dumps(r, allow_nan=False)
        if L in [3, 5]:
            prior = old.excited_certificate(old.neutral_model(L, 1., g))
            assert r['excitation_separation_sufficient'] == prior['available']


def test_cap_checked_before_arrays(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('allocation reached')
    monkeypatch.setattr(np, 'arange', fail)
    with pytest.raises(ValueError):
        c.finite_compression_measure(4097, 1)
    assert c.validity_audit(4097)['population_selection_sufficient']


def test_finite_cap_boundary_and_uncapped_scalar():
    r = c.finite_compression_measure(4096, 1)
    assert len(r['probabilities']) == 4095
    assert sum(r['probabilities']) == pytest.approx(1., abs=1e-13)
    assert c.weak_limit_bound(8192, 1, 0.) > 0


@pytest.mark.parametrize('args', [(2, 1, 1., 40.), (True, 1, 1., 40.),
                                  (3, True, 1., 40.), (3, 1.5, 1., 40.),
                                  (3, 1, 0., 40.), (3, 1, 1., -1.),
                                  (3, 1, np.inf, 40.), (3, 1, 1., np.nan),
                                  (3, 1, 1e-320, 40.)])
def test_invalid_finite_arguments(args):
    with pytest.raises((TypeError, ValueError)):
        c.finite_compression_measure(*args)


@pytest.mark.parametrize('k', [np.inf, np.nan, 10., True, 1e-320])
def test_invalid_continuous_momentum(k):
    with pytest.raises((TypeError, ValueError)):
        c.limiting_shape(k)


@pytest.mark.parametrize('degree', [-1, 33, True, 1.5])
def test_invalid_moment_degree(degree):
    with pytest.raises((TypeError, ValueError)):
        c.semicircle_moment(degree, 0.)


@pytest.mark.parametrize('K', [-1., np.inf, True, 1e-320])
def test_invalid_lipschitz_constant(K):
    with pytest.raises((TypeError, ValueError)):
        c.weak_limit_bound(5, 1, 0., K)


@pytest.mark.parametrize('g,m', [(40., 1), (40., 0), (0., 1), (0., 0)])
def test_report_claims_and_availability(g, m):
    r = c.case_report(16, m, 1., g)
    audit = r['validity_audit']
    assert not audit['schur_norm_computed']
    assert not audit['full_hamiltonian_thermodynamic_certificate_available']
    assert not audit['excitation_separation_sufficient']
    assert r['compression_measure']['normalization_available'] == bool(g and m)
    assert (r['weak_limit_bound'] is not None) == bool(g and m)
    json.dumps(r, allow_nan=False)


def test_demo_frozen_thresholds_and_no_physical_predictions():
    r = c.demonstration_report()
    rows = r['frozen_threshold_audits']
    assert [x['parameters']['L'] for x in rows] == [9, 10]
    assert [x['audit']['excitation_separation_sufficient'] for x in rows] == [True, False]
    assert all(value is None for value in r['physical_predictions'].values())
    assert len(r['cases']) == 10
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('L,m', [(10**400, 1), (10**100, 10**100//2-1)])
def test_unrepresentable_scalar_angle_is_not_false_zero_or_pi(L, m):
    with pytest.raises(ValueError):
        c.weak_limit_bound(L, m, 0.)
    assert c.validity_audit(L)['population_selection_sufficient']


@pytest.mark.parametrize('flags', [[], ['--json']])
def test_stdout_demo(tmp_path, flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_neutral_continuum.py'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')
    result = subprocess.run([sys.executable, str(script)]+flags, cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout, parse_constant=lambda x: pytest.fail(x))
    else:
        assert 'compression' in result.stdout.lower()
