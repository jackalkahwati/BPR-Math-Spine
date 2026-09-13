"""Full-source joint-limit checks with independent finite-matrix oracles."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from bpr import substrate_neutral_scaling as s
from bpr import substrate_neutral_response as old
from bpr import substrate_neutral_effective as effective
from bpr import substrate_neutral_continuum as continuum


def semicircle_characteristic(t, k):
    a = abs(2+np.exp(-1j*k))
    return quad(lambda u: 2/np.pi*np.sin(u)**2*np.cos(2*a*t*np.cos(u)),
                0., np.pi, epsabs=1e-12)[0]


@pytest.mark.parametrize('n', [-4, -2, -1, 1, 2, 4])
def test_sawtooth_inverse_commutator_sign(n):
    # The imaginary part cancels; the real part is -sin(nt)*(t-pi).
    real = quad(lambda t: -(t-np.pi)*np.sin(n*t)/(2*np.pi), 0., 2*np.pi)[0]
    imag = quad(lambda t: (t-np.pi)*np.cos(n*t)/(2*np.pi), 0., 2*np.pi)[0]
    assert real == pytest.approx(1/n, abs=1e-14)
    assert abs(imag) < 1e-14
    norm_kernel = quad(lambda t: abs(t-np.pi)/(2*np.pi), 0., 2*np.pi)[0]
    assert norm_kernel == pytest.approx(np.pi/2)


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('g', [40., 4000.])
def test_first_order_unitary_generator_and_source_bounds(L, g):
    a = old.neutral_model(L, 1., g)
    coeff = effective.sw_coefficients(a)
    lam = 1/g
    U = expm(lam*coeff.S1)
    assert np.linalg.norm(coeff.S1, 2) <= np.pi*L
    Hbd = g*np.diag(a.model.D)+coeff.Td
    assert np.linalg.norm(U @ a.model.H @ U.T-Hbd, 2) <= 8*np.pi*lam*L**2+2e-11
    values, vectors = np.linalg.eigh(a.model.H)
    G = vectors[:, 0]
    if G[a.omega_index] < 0:
        G = -G
    assert abs(values[0]) <= 4*lam*L/(1-4*lam*L)+1e-12
    w = old.virtual_ground_component(a)
    old_cert = old.density_certificate(a, 1)
    x = old.density_diagonal(a.model, 1)*G
    y = old.density_diagonal(a.model, 1)*w
    delta = np.sqrt(L)*old_cert['eta']
    assert np.linalg.norm(x-y) <= delta+2e-14
    if delta < np.linalg.norm(y):
        vector_bound = 2*delta/np.linalg.norm(y)+np.pi*lam*L
        assert np.linalg.norm(U @ x/np.linalg.norm(x)-y/np.linalg.norm(y)) <= vector_bound+2e-13


@pytest.mark.parametrize('L', [3, 4, 5])
@pytest.mark.parametrize('g', [40., 4000.])
@pytest.mark.parametrize('time', [0., .5, 1.])
def test_independent_full_characteristic_bound(L, g, time):
    a = old.neutral_model(L, 1., g)
    values, vectors = np.linalg.eigh(a.model.H)
    x = old.density_diagonal(a.model, 1)*vectors[:, 0]
    probabilities = abs(vectors.conj().T @ x)**2/np.vdot(x, x).real
    exact = probabilities @ np.exp(1j*time*(values-values[0]-g))
    compressed = continuum.finite_compression_measure(L, 1, 1., g)
    leading = compressed['probabilities'] @ np.exp(1j*time*compressed['dimensionless_offsets'])
    delta = np.sqrt(L)*old.density_certificate(a, 1)['eta']
    norm = 4/g*abs(np.sin(np.pi/L))
    report = s.comparison_bound(L, 1, 1., g, time, 0.)
    assert report['bound_available'] == bool(delta < norm)
    assert report['source_norm'] == pytest.approx(norm)
    assert report['source_error_bound'] == pytest.approx(delta)
    if delta < norm:
        bound = 4*delta/norm+2*np.pi*L/g+abs(time)*(8*np.pi*L**2/g+4*L/g/(1-4*L/g))
        expected_full = min(2., bound) if time else 0.
        assert report['full_compression_bound'] == pytest.approx(expected_full)
        assert abs(exact-leading) <= report['full_compression_bound']+4e-11
        assert abs(exact-leading) <= min(2., bound)+4e-11
        target = semicircle_characteristic(time, 0.)
        total = min(2., bound+2*continuum.weak_limit_bound(L, 1, 0., abs(time)))
        assert abs(exact-target) <= total+4e-11
        assert report['full_limit_bound'] == pytest.approx(total if time else 0.)
    else:
        assert report['full_limit_bound'] is None


@pytest.mark.parametrize('L', [16, 64, 256])
@pytest.mark.parametrize('fixed_m', [False, True])
def test_scalar_joint_sequences_and_compound_bound(L, fixed_m):
    power = 4 if fixed_m else 3
    m = 1 if fixed_m else L//4
    target = 0. if fixed_m else np.pi/2
    r = s.comparison_bound(L, m, 1., float(L**power), 1., target)
    assert r['bound_available']
    assert not r['roundoff_included']
    assert r['generator_bound'] == pytest.approx(8*np.pi*L**(2-power))
    assert r['ground_shift_bound'] == pytest.approx(4*L**(1-power)/(1-4*L**(1-power)))
    assert r['full_limit_bound'] <= 2.
    raw = r['full_compression_bound_uncapped']+2*continuum.weak_limit_bound(L, m, target)
    assert r['full_limit_bound'] == pytest.approx(min(2., raw))
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('L', [16, 64, 256])
def test_spatial_threshold_and_physical_generator_scaling(L):
    r = s.spatial_scaling(L, ell=2., kappa=3.)
    h = 2/L
    p = np.pi
    C = 3/h**2
    k = h*p
    expected = 16*C*np.sin(k/2)**2/(3+abs(2+np.exp(1j*k)))
    assert r['threshold_shift'] == pytest.approx(expected)
    assert r['quadratic_threshold_shift'] == pytest.approx(2*3*p*p/3)
    assert abs(expected-r['quadratic_threshold_shift']) <= r['threshold_remainder_bound']+2e-13
    assert r['threshold_remainder_bound'] == pytest.approx(5*C*k**4/36)
    shape = abs(2+np.exp(1j*k))
    edge_offset = 4*C*shape*np.sin(np.pi/(2*L))**2
    assert r['finite_compression_edge_offset'] == pytest.approx(edge_offset)
    assert 'support edge' in r['threshold_scope']
    # The finite-size offset survives the physical C proportional L² scaling.
    assert edge_offset > 15.
    assert r['physical_generator_bound'] == pytest.approx(C*8*np.pi/L**3)
    assert r['physical_ground_shift_bound'] == pytest.approx(C*4/L**4/(1-4/L**4))
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('g,m', [(40., 1), (.7, 1), (0., 1), (0., 0), (40., 0)])
def test_frozen_unavailable_controls(g, m):
    r = s.comparison_bound(5, m, 1., g, 1., 0.)
    assert not r['bound_available']
    assert r['unavailable_reason']
    assert r['full_limit_bound'] is None
    assert r['full_compression_bound'] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('g,m', [(40.,0),(0.,1),(0.,0),(.7,1),(40.,1)])
def test_zero_time_does_not_bypass_source_gate(g,m):
    r = s.comparison_bound(5,m,1.,g,0.)
    assert not r['bound_available']
    assert r['full_limit_bound'] is None


def test_opposite_momenta_and_times():
    a = s.comparison_bound(64,1,1.,float(64**4),.5,0.)
    b = s.comparison_bound(64,-1,1.,float(64**4),-.5,0.)
    for key in ('source_norm','source_error_bound','full_limit_bound'):
        assert a[key] == pytest.approx(b[key])


@pytest.mark.parametrize('m', [-4, 6, 10**100+1])
def test_scalar_aliases(m):
    a = s.comparison_bound(5, 1, 1., 4000., .5, 0.)
    b = s.comparison_bound(5, m, 1., 4000., .5, 0.)
    assert a == b


@pytest.mark.parametrize('scale', [1e-60, 1e60])
def test_scale_covariance(scale):
    a = s.comparison_bound(5, 1, 1., 4000.)
    b = s.comparison_bound(5, 1, scale, 4000*scale)
    for key in ('source_norm', 'source_error_bound', 'relative_source_error',
                'generator_bound', 'ground_shift_bound', 'full_limit_bound'):
        assert a[key] == pytest.approx(b[key], rel=2e-14)


@pytest.mark.parametrize('C,g', [(1e303,1e308),(1e-303,1e-298)])
def test_scalar_extreme_common_scales(C,g):
    r = s.comparison_bound(3,1,C,g,.5,0.)
    reference = s.comparison_bound(3,1,1.,1e5,.5,0.)
    assert r['bound_available']
    for key in ('source_norm','source_error_bound','generator_bound','full_limit_bound'):
        assert r[key] == pytest.approx(reference[key], rel=2e-14)
    json.dumps(r, allow_nan=False)


def test_scalar_no_full_fock_allocation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('full Fock allocation reached')
    monkeypatch.setattr(old, 'neutral_model', fail)
    r = s.comparison_bound(8192, 1, 1., float(8192**4))
    assert r['bound_available']
    assert r['full_limit_bound'] < .01


def test_separation_boundary_and_vanishing_absolute_source():
    for g in (20.,np.nextafter(20.,0.)):
        r = s.comparison_bound(5,1,1.,g)
        assert not r['separation_sufficient']
        assert not r['bound_available']
    r = s.comparison_bound(5,1,1.,np.nextafter(20.,np.inf))
    assert r['separation_sufficient']
    assert not r['source_normalization_sufficient']
    upper = []
    for L in (16,64,256):
        r = s.comparison_bound(L,1,1.,float(L**4))
        assert r['source_norm'] > r['source_error_bound']
        upper.append((r['source_norm']+r['source_error_bound'])**2)
    assert upper[2] < upper[1] < upper[0] < 1e-8


@pytest.mark.parametrize('L,m,C,g,t', [(2,1,1.,40.,1.), (True,1,1.,40.,1.),
    (3,True,1.,40.,1.), (3,1,0.,40.,1.), (3,1,1.,-1.,1.),
    (3,1,np.inf,40.,1.), (3,1,1.,np.nan,1.), (3,1,1.,40.,np.inf),
    (3,1,1e-320,40.,1.), (3,1,1.,40.,True)])
def test_invalid_bound_arguments(L,m,C,g,t):
    with pytest.raises((ValueError, TypeError)):
        s.comparison_bound(L,m,C,g,t)


@pytest.mark.parametrize('L,g,m', [(3,4000.,1), (4,4000.,2), (5,40.,1)])
def test_exact_oracle_report_independent_diagnostics(L,g,m):
    r = s.exact_case_report(L,m,1.,g,.5,0.)
    d = r['numerical_diagnostics']
    assert d is not None
    assert d['generator_error_over_C'] <= 8*np.pi*L**2/g+3e-11
    bound = r['comparison_bound']
    if bound['bound_available']:
        assert d['full_compression_error'] <= bound['full_compression_bound']+3e-11
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('time',[0.,1e-6,1.])
def test_oracle_rejects_noise_dominated_strong_coupling(time):
    # The scalar theorem remains meaningful, but double-precision full
    # diagonalization loses the source and centered energies at this scale.
    assert s.comparison_bound(3,1,1.,1e18,time)['bound_available']
    with pytest.raises(ValueError, match='unresolved'):
        s.exact_case_report(3,1,1.,1e18,time)


def test_oracle_cap_and_structural_undefined_controls():
    with pytest.raises(ValueError):
        s.exact_case_report(7,1)
    for g,m in [(40.,0),(0.,1),(0.,0)]:
        r = s.exact_case_report(3,m,1.,g)
        assert r['numerical_diagnostics'] is None
        assert r['unavailable_reason']


@pytest.mark.parametrize('flags', [[], ['--json']])
def test_stdout_demo(tmp_path, flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_neutral_scaling.py'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    proc = subprocess.run([sys.executable, str(script)]+flags, cwd=tmp_path,
                          env=env, text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert not proc.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(proc.stdout, parse_constant=lambda x: pytest.fail(x))
    else:
        assert 'limit' in proc.stdout.lower()
