"""Independent flux, charge and spectral-flow checks; no physical benchmark."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_chirality as c
from bpr import substrate_fermionization as f


def tensor(factors):
    result = np.ones((1,1))
    for factor in factors:
        result = np.kron(result,factor)
    return result


def tensor_hard_core(L,N,C,phi):
    """Local commuting spin lowerers; no fermion or dispersion helpers."""
    lower = np.array([[0.,1.],[0.,0.]])
    b = [tensor([lower if x==site else np.eye(2) for x in reversed(range(L))])
         for site in range(L)]
    H = np.zeros((2**L,2**L),dtype=complex)
    for x in range(L):
        y = (x+1)%L
        phase = np.exp(1j*phi) if x==L-1 else 1.
        term = phase*b[x].T@b[y]
        H -= C*(term+term.conj().T)
    indices = [bit for bit in range(2**L) if bin(bit).count('1')==N]
    return H[np.ix_(indices,indices)]


def independent_slater(L,N,C,phi):
    theta = 0. if N%2 else np.pi
    eps = -2*C*np.cos((2*np.pi*np.arange(L)+theta+phi)/L)
    return np.sort([sum(eps[list(indices)]) for indices in itertools.combinations(range(L),N)])


@pytest.mark.parametrize('L,N',[(3,0),(3,1),(3,2),(3,3),(5,0),(5,1),(5,2),(5,3),(5,5),(6,2),(6,3)])
@pytest.mark.parametrize('phi',[0.,np.pi/3,2*np.pi])
def test_spectra_against_independent_tensor_and_slater(L,N,phi):
    C = 1.3
    expected = tensor_hard_core(L,N,C,phi)
    np.testing.assert_allclose(c.hard_core_hopping(L,N,C,phi),expected,atol=3e-14)
    numerical = np.linalg.eigvalsh(expected)
    np.testing.assert_allclose(c.slater_energies(L,N,C,phi),numerical,atol=3e-13)
    np.testing.assert_allclose(independent_slater(L,N,C,phi),numerical,atol=3e-13)


@pytest.mark.parametrize('L,N',[(3,1),(4,2),(5,2),(5,3),(6,3)])
@pytest.mark.parametrize('phi',[0.,np.pi/3,2*np.pi])
def test_one_body_phase_eigenvectors_and_velocities(L,N,phi):
    C = .7
    h = c.one_body_hopping(L,N,C,phi)
    r = c.one_body_spectrum(L,N,C,phi)
    theta = np.pi if N%2==0 else 0.
    momenta = (2*np.pi*np.arange(L)+theta+phi)/L
    vectors = np.exp(1j*np.outer(np.arange(L),momenta))/np.sqrt(L)
    np.testing.assert_allclose(h@vectors,vectors*(-2*C*np.cos(momenta)),atol=3e-14)
    np.testing.assert_allclose(r['energies'],-2*C*np.cos(momenta),atol=3e-14)
    np.testing.assert_allclose(r['velocities'],2*C*np.sin(momenta),atol=3e-14)
    np.testing.assert_allclose(r['flux_slopes'],2*C*np.sin(momenta)/L,atol=3e-14)
    np.testing.assert_allclose(h,h.conj().T,atol=0.)


@pytest.mark.parametrize('L,N',[(4,1),(4,2),(5,2),(5,3)])
def test_large_gauge_cycle_is_permutation(L,N):
    np.testing.assert_array_equal(c.one_body_hopping(L,N,phi=0.),c.one_body_hopping(L,N,phi=2*np.pi))
    a = c.one_body_spectrum(L,N,phi=0.)['energies']
    b = c.one_body_spectrum(L,N,phi=2*np.pi)['energies']
    np.testing.assert_allclose(b,np.roll(a,-1),atol=3e-14)
    np.testing.assert_allclose(c.slater_energies(L,N,phi=0.),c.slater_energies(L,N,phi=2*np.pi),atol=3e-14)


def test_even_sector_requires_antiperiodic_fermions():
    exact = np.linalg.eigvalsh(tensor_hard_core(5,2,1.,0.))
    periodic = -2*np.cos(2*np.pi*np.arange(5)/5)
    wrong = np.sort([sum(periodic[list(pair)]) for pair in itertools.combinations(range(5),2)])
    assert np.max(abs(exact-wrong)) > .5
    np.testing.assert_allclose(c.slater_energies(5,2),exact,atol=2e-14)


@pytest.mark.parametrize('L,N',[(4,1),(5,2),(6,3)])
def test_uniform_link_gauge_equivalence(L,N):
    phi = np.pi/3
    alpha = (0. if N%2 else np.pi)+phi
    gauge = np.diag(np.exp(1j*alpha*np.arange(L)/L))
    h = c.one_body_hopping(L,N,1.,phi)
    uniform = np.zeros((L,L),complex)
    for x in range(L):
        y = (x+1)%L
        uniform[x,y] = -np.exp(1j*alpha/L)
        uniform[y,x] = -np.exp(-1j*alpha/L)
    np.testing.assert_allclose(gauge.conj().T@h@gauge,uniform,atol=2e-14)


@pytest.mark.parametrize('L,N',[(3,1),(4,1),(4,2),(5,2),(6,3)])
@pytest.mark.parametrize('mu',[-1.,0.,.7])
def test_transverse_flux_events_and_paired_flow(L,N,mu):
    r = c.flux_crossings(L,N,1.,mu)
    assert len(r['events']) == 2
    assert r['net_signed_flow'] == 0
    assert sorted(e['sign'] for e in r['events']) == [-1,1]
    for e in r['events']:
        assert 0 <= e['phi'] < 2*np.pi
        assert 0 <= e['mode'] < L
        theta = 0 if N%2 else np.pi
        k = (2*np.pi*e['mode']+theta+e['phi'])/L
        assert -2*np.cos(k) == pytest.approx(mu,abs=3e-14)
        assert e['velocity'] == pytest.approx(2*np.sin(k),abs=3e-14)
        assert e['flux_slope'] == pytest.approx(e['velocity']/L)
        assert np.sign(e['velocity']) == e['sign']
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('L,N',[(4,1),(4,2),(5,2)])
@pytest.mark.parametrize('mu',[-3.,-2.,2.,3.])
def test_band_edges_are_tangencies_not_chiral_crossings(L,N,mu):
    r = c.flux_crossings(L,N,1.,mu)
    assert r['net_signed_flow'] == 0
    if abs(mu)==2:
        assert len(r['events']) == 1
        assert r['events'][0]['sign'] == 0
        assert r['events'][0]['velocity'] == 0
    else:
        assert r['events'] == []


@pytest.mark.parametrize('mu',[np.nextafter(-2.,0.),np.nextafter(2.,0.)])
def test_near_band_edges_remain_transverse(mu):
    r = c.flux_crossings(5,2,1.,mu)
    assert len(r['events'])==2
    assert sorted(e['sign'] for e in r['events'])==[-1,1]
    assert all(e['velocity']!=0 for e in r['events'])


def test_endpoint_ties_retained_separately():
    for N,flux in [(1,0.),(2,np.pi)]:
        r = c.flux_crossings(4,N,1.,0.)
        assert len(r['events']) == 2
        assert [e['phi'] for e in r['events']] == pytest.approx([flux,flux])
        assert len({e['mode'] for e in r['events']}) == 2
        assert r['net_signed_flow'] == 0


@pytest.mark.parametrize('L,N',[(3,1),(3,2),(5,1),(5,2),(5,3),(5,4),(6,2),(6,3)])
def test_zero_flux_symmetric_chemical_potential_identity(L,N):
    E = [c.slater_energies(L,n)[0] for n in (N-1,N,N+1)]
    assert E[1] == pytest.approx(-2*np.sin(np.pi*N/L)/np.sin(np.pi/L),abs=3e-14)
    symmetric = (E[2]-E[0])/2
    reference = -2*np.cos(np.pi*N/L)
    assert symmetric == pytest.approx(reference,abs=3e-14)
    assert E[1]-E[0] <= reference+3e-14
    assert reference <= E[2]-E[1]+3e-14


def test_flux_induced_ground_degeneracy_is_not_hidden():
    # At phi=pi, L5,N2 has a singlet filled first and a tied +/- pair
    # at the occupation cut. Two distinct Slater ground states remain.
    r = c.ground_filling(5,2,1.,np.pi)
    assert r['cut_tie_numerical']
    assert r['ground_degeneracy_numerical']==2
    assert r['occupation_gap']==pytest.approx(0.,abs=3e-14)
    r = c.ground_filling(5,2,1.,0.)
    assert not r['cut_tie_numerical']
    assert r['ground_degeneracy_numerical']==1


@pytest.mark.parametrize('L,N',[(5,0),(5,1),(5,2),(5,3),(5,5),(6,2),(6,3)])
@pytest.mark.parametrize('phi',[0.,np.pi/3,2*np.pi])
def test_ground_energy_and_cut_accounting(L,N,phi):
    r = c.ground_filling(L,N,1.,phi)
    expected = independent_slater(L,N,1.,phi)
    assert r['ground_energy'] == pytest.approx(expected[0],abs=3e-14)
    if N in (0,L):
        assert r['occupation_gap'] is None
    else:
        eps = np.sort(-2*np.cos((2*np.pi*np.arange(L)+(0 if N%2 else np.pi)+phi)/L))
        assert r['occupation_gap'] == pytest.approx(eps[N]-eps[N-1],abs=3e-14)
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('L,q',[(3,3),(4,4),(5,5),(5,2)])
def test_full_binary_charge_and_neutrality_independent(L,q):
    ops = f.binary_operators(L)
    number = ops['number']
    P = np.diag([int(bin(bit).count('1')%q==0) for bit in range(2**L)])
    for op in ops['c']:
        create = op.T
        np.testing.assert_array_equal(number@create-create@number,create)
        np.testing.assert_array_equal(P@create@P,np.zeros_like(P))
    for x in range(L):
        for y in range(L):
            bilinear = ops['c'][x].T@ops['c'][y]
            np.testing.assert_array_equal(number@bilinear,bilinear@number)
    if q==L:
        assert np.trace(P)==2
    report = c.charge_neutrality_diagnostics(L,q)
    assert report['neutral_dimension']==np.trace(P)
    for key in ('creation_charge_one_residual','bilinear_charge_zero_residual',
                'projected_creation_max_abs','jw_string_number_commutator_residual'):
        assert report[key]==0.
    assert report['only_empty_full']==(q==L)
    assert report['projected_offsite_bilinear_max_abs']==(0. if q==L else 1.)
    assert not report['dynamical_confinement_derived']
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('L,N',[(5,0),(5,1),(5,2),(5,3),(5,5),(6,3)])
def test_default_branch_pair_or_structural_empty_full(L,N):
    r = c.branch_diagnostics(L,N)
    assert r['two_nonzero_velocity_branches']==(0<N<L)
    if 0<N<L:
        assert r['mu']==pytest.approx(-2*np.cos(np.pi*N/L),abs=3e-14)
        assert sorted(e['sign'] for e in r['branches'])==[-1,1]
        assert sum(e['velocity'] for e in r['branches'])==pytest.approx(0.,abs=3e-14)
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('g',[40.,.7])
def test_finite_g_interactions_remain_negative_control(g):
    model = f.fixed_number_model(5,2,1.,g)
    delta = f.second_order_effective(model)['correction']
    basis = model.hard_core_basis
    i = basis.index((1,1,0,0,0))
    j = basis.index((0,1,1,0,0))
    assert delta[i,i]==pytest.approx(-4/g)
    assert delta[j,i]==pytest.approx(-2/g)
    r = c.finite_g_witnesses(1.,g)
    assert not r['finite_g_exact_free_fermion_interpretation']
    assert not r['other_emergent_fermion_mechanisms_excluded']
    assert r['certificate']['available']==(g==40.)
    design = np.column_stack((np.ones(len(basis)),np.array(basis)))
    fit = design@np.linalg.lstsq(design,np.diag(delta),rcond=None)[0]
    residual = max(abs(np.diag(delta)-fit))
    assert residual==pytest.approx(2/g)
    assert r['additive_diagonal_fit']['residual_max_abs']==pytest.approx(residual)
    assert r['additive_diagonal_fit']['residual_max_abs_in_C_squared_over_g']==pytest.approx(2.)
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('L,N',[(5,0),(5,2),(5,5),(6,3)])
def test_case_report_combines_actual_spectra_without_physical_claims(L,N):
    r = c.case_report(L,N,1.,np.pi/3,q=L)
    assert r['dimensions']['hard_core']==len(independent_slater(L,N,1.,np.pi/3))
    np.testing.assert_allclose(r['spectra']['hard_core'],r['spectra']['slater'],atol=3e-13)
    assert not r['numerical_checks']['roundoff_certified']
    assert r['neutrality']['only_empty_full']
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('scale',[1e-80,1e80])
def test_common_energy_scaling_preserves_flux_events(scale):
    a = c.flux_crossings(5,2,1.,.7)
    b = c.flux_crossings(5,2,scale,.7*scale)
    assert a['net_signed_flow']==b['net_signed_flow']
    for x,y in zip(a['events'],b['events']):
        assert x['mode']==y['mode']
        assert x['phi']==pytest.approx(y['phi'],abs=2e-14)
        assert x['sign']==y['sign']
        assert y['velocity']/scale==pytest.approx(x['velocity'])
    np.testing.assert_allclose(c.slater_energies(5,2,scale,np.pi/3)/scale,
                               c.slater_energies(5,2,1.,np.pi/3),atol=2e-14)


@pytest.mark.parametrize('mu',[np.nan,np.inf,True])
def test_invalid_reference(mu):
    with pytest.raises((ValueError,TypeError)):
        c.flux_crossings(5,2,1.,mu)


@pytest.mark.parametrize('L,N',[(513,1),(True,1),(2,1),(4,True),(4,5)])
def test_one_body_invalid_and_preallocation_caps(L,N):
    with pytest.raises((TypeError,ValueError)):
        c.one_body_hopping(L,N)


def test_many_body_preallocation_caps():
    with pytest.raises(ValueError):
        c.hard_core_hopping(9,1)
    with pytest.raises(ValueError):
        c.slater_energies(12,6)


@pytest.mark.parametrize('C,phi',[(0.,0.),(-1.,0.),(np.inf,0.),(1e-320,0.),(1.,np.nan),(1.,-1.),(1.,2*np.pi+1.),(True,0.),(1.,True)])
def test_invalid_energy_and_flux(C,phi):
    with pytest.raises((TypeError,ValueError)):
        c.one_body_hopping(5,2,C,phi)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_stdout_demo(tmp_path,flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_chirality.py'
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    p = subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert p.returncode==0,p.stderr
    assert not p.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(p.stdout,parse_constant=lambda v:pytest.fail(v))
    else:
        assert 'chirality' in p.stdout.lower()
