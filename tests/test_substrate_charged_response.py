"""Independent complete-sector spectroscopy checks, not particle-mass evidence."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_charged_response as c
from bpr.substrate_vacuum_selection import all_number_model


def occupation_annihilator(lower, upper, m):
    """Independent bra/ket selection-rule construction, no moved-state lookup."""
    out = np.zeros((len(lower.basis), len(upper.basis)), dtype=complex)
    for row, bra in enumerate(lower.basis):
        for col, ket in enumerate(upper.basis):
            delta = np.array(ket) - np.array(bra)
            sites = np.flatnonzero(delta)
            if len(sites) == 1 and delta[sites[0]] == 1:
                x = sites[0]
                out[row, col] = np.sqrt(ket[x] / upper.L) * np.exp(-2j*np.pi*m*x/upper.L)
    return out


def independent_sector_data(L, g, m, kappa=0.):
    models = [all_number_model(L, N, 1., g) for N in (L-1, L, L+1)]
    matrices = [model.H + kappa*model.N*np.eye(len(model.basis)) for model in models]
    spectra = [np.linalg.eigh(H) for H in matrices]
    E0 = spectra[1][0][0]
    ground = spectra[1][1][:, 0]
    remove = occupation_annihilator(models[0], models[1], m) @ ground
    add = occupation_annihilator(models[1], models[2], m).conj().T @ ground
    return models, matrices, spectra, E0, ground, remove, add


@pytest.fixture(scope='module')
def reports():
    return {(L,g): c.spectroscopy(L,g=g) for L,g in itertools.product([3,4,5],[.7,40.])}


@pytest.mark.parametrize('L,N', [(3,1),(3,3),(3,4),(4,4),(5,6)])
def test_rectangular_occupation_and_site_maps(L,N):
    lower, upper = [all_number_model(L,n) for n in [N-1,N]]
    for m in range(L):
        actual = c.annihilation_map(L,N,m)
        expected = occupation_annihilator(lower,upper,m)
        np.testing.assert_allclose(actual,expected,atol=9e-16)
        local = sum(np.exp(-2j*np.pi*m*x/L)*c.local_annihilation_map(L,N,x)
                    for x in range(L))/np.sqrt(L)
        np.testing.assert_allclose(actual,local,atol=9e-16)


def test_three_site_tensor_annihilation_oracle():
    # Local occupation0..4 includes every vector in total-number sectors3,4.
    a=np.diag(np.sqrt(np.arange(1,5)),1)
    identity=np.eye(5)
    local=[np.kron(np.kron(a if x==0 else identity,a if x==1 else identity),
                   a if x==2 else identity) for x in range(3)]
    lower,upper=[all_number_model(3,N) for N in [3,4]]
    low_indices=[25*s[0]+5*s[1]+s[2] for s in lower.basis]
    high_indices=[25*s[0]+5*s[1]+s[2] for s in upper.basis]
    for m in range(3):
        operator=sum(np.exp(-2j*np.pi*m*x/3)*local[x] for x in range(3))/np.sqrt(3)
        oracle=operator[np.ix_(low_indices,high_indices)]
        np.testing.assert_allclose(c.annihilation_map(3,4,m),oracle,atol=8e-16)


def test_exact_degenerate_target_weight_is_basis_invariant():
    from bpr.substrate_neutral_response import grouped_spectral_measure
    L=3
    middle,upper=[all_number_model(L,N) for N in [L,L+1]]
    ground=np.zeros(len(middle.basis))
    ground[middle.basis.index((1,)*L)]=1.
    source=c.annihilation_map(L,L+1,1).conj().T@ground
    energies=40*upper.D
    vectors=np.eye(len(energies),dtype=complex)
    before=grouped_spectral_measure(energies,vectors,source)
    block=np.flatnonzero(energies==40)
    size=len(block)
    unitary=np.exp(2j*np.pi*np.outer(np.arange(size),np.arange(size))/size)/np.sqrt(size)
    vectors[np.ix_(block,block)]=unitary
    after=grouped_spectral_measure(energies,vectors,source)
    np.testing.assert_allclose([group['weight'] for group in before['groups']],
                               [group['weight'] for group in after['groups']],atol=2e-15)
    assert before['first_moment']==pytest.approx(after['first_moment'],abs=2e-13)


def test_translation_intertwines_rectangular_fourier_maps():
    def translate(model):
        T=np.zeros_like(model.H)
        for col,state in enumerate(model.basis):
            row=model.basis.index((state[-1],)+state[:-1])
            T[row,col]=1
        return T
    for N in [3,4]:
        lower,upper=[all_number_model(3,n) for n in [N-1,N]]
        for m in range(3):
            A=c.annihilation_map(3,N,m)
            np.testing.assert_allclose(translate(lower)@A,
                                       np.exp(2j*np.pi*m/3)*A@translate(upper),atol=2e-15)


@pytest.mark.parametrize('L',[3,4,5])
def test_commutator_on_complete_middle_sector(L):
    for m in range(L):
        down = c.annihilation_map(L,L,m)
        upper_down = c.annihilation_map(L,L+1,m)
        np.testing.assert_allclose(upper_down@upper_down.conj().T-down.conj().T@down,
                                   np.eye(down.shape[1]),atol=4e-15)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_signed_thresholds_weights_and_moments(reports,L,g):
    report=reports[L,g]
    for m,mode in enumerate(report['modes']):
        models, matrices, spectra, E0, ground, remove, add=independent_sector_data(L,g,m)
        mu_plus=spectra[2][0][0]-E0
        mu_minus=E0-spectra[0][0][0]
        for key,value in [('mu_plus',mu_plus),('mu_minus',mu_minus),
                          ('Delta_c',mu_plus-mu_minus),('removal_ground_cost',-mu_minus)]:
            assert report['thresholds'][key]==pytest.approx(value,abs=2e-10)
        totals=[]
        for name,index,source in [('removal',0,remove),('addition',2,add)]:
            data=mode[name]
            total=np.vdot(source,source).real
            moment=np.vdot(source,(matrices[index]-E0*np.eye(len(source)))@source).real
            totals.append(total)
            np.testing.assert_allclose(data['energies'],spectra[index][0]-E0,atol=2e-10)
            assert data['total_weight']==pytest.approx(total,abs=2e-12)
            assert data['first_moment']==pytest.approx(moment,abs=3e-10)
            assert data['direct_first_moment']==pytest.approx(moment,abs=3e-10)
            assert sum(data['weights'])==pytest.approx(total,abs=2e-12)
            assert sum(group['weight'] for group in data['groups'])==pytest.approx(total,abs=2e-12)
            assert all(w>=0 for w in data['weights'])
        assert totals[1]-totals[0]==pytest.approx(1.,abs=3e-12)
        # Exact double-commutator coefficient for the homogeneous fixed-N state.
        combined=mode['addition']['first_moment']+mode['removal']['first_moment']
        assert combined==pytest.approx(-2*np.cos(2*np.pi*m/L)+2*g,abs=6e-10)
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_green_function_against_independent_resolvents(reports,L,g):
    report=reports[L,g]
    m=1
    models,matrices,spectra,E0,ground,remove,add=independent_sector_data(L,g,m)
    for z in [.5j,1+.5j,4+1j]:
        plus=np.vdot(add,np.linalg.solve((z+E0)*np.eye(len(add))-matrices[2],add))
        minus=np.vdot(remove,np.linalg.solve((z-E0)*np.eye(len(remove))+matrices[0],remove))
        assert c.green_function(report,m,z)==pytest.approx(plus-minus,rel=3e-11,abs=3e-11)
    z=10000j
    mode=report['modes'][m]
    coefficient=mode['addition']['first_moment']+mode['removal']['first_moment']
    remainder_bound=0.
    for name in ['addition','removal']:
        for d,w in zip(mode[name]['energies'],mode[name]['weights']):
            remainder_bound+=w*d*d/(abs(z)**2*abs(z.imag))
    assert abs(c.green_function(report,m,z)-1/z-coefficient/z**2)<=remainder_bound+1e-17


@pytest.mark.parametrize('g',[.7,40.])
@pytest.mark.parametrize('kappa',[-3.,0.,2.5])
def test_actual_number_shift_and_green_translation(reports,g,kappa):
    base=reports[3,g]
    shifted=c.spectroscopy(3,g=g,kappa=kappa)
    for key in ['mu_plus','mu_minus']:
        assert shifted['thresholds'][key]==pytest.approx(base['thresholds'][key]+kappa,abs=3e-11)
    assert shifted['thresholds']['Delta_c']==pytest.approx(base['thresholds']['Delta_c'],abs=6e-11)
    for m in range(3):
        for name,sign in [('addition',1),('removal',-1)]:
            a,b=shifted['modes'][m][name],base['modes'][m][name]
            assert a['total_weight']==pytest.approx(b['total_weight'],abs=3e-12)
            np.testing.assert_allclose(a['energies'],np.array(b['energies'])+sign*kappa,atol=6e-11)
        z=1+.5j
        assert c.green_function(shifted,m,z)==pytest.approx(c.green_function(base,m,z-kappa),abs=5e-11)


@pytest.mark.parametrize('L',[3,4,5])
def test_atomic_oracle_from_diagonal_occupation_hamiltonian(L):
    g=40.
    lower, middle, upper=[all_number_model(L,N) for N in [L-1,L,L+1]]
    ground=np.zeros(len(middle.basis))
    ground[middle.basis.index((1,)*L)]=1.
    assert min(g*upper.D)==g and min(g*lower.D)==0
    for m in range(L):
        rm=c.annihilation_map(L,L,m)@ground
        ad=c.annihilation_map(L,L+1,m).conj().T@ground
        assert np.vdot(rm,rm).real==pytest.approx(1.)
        assert np.vdot(ad,ad).real==pytest.approx(2.)
        np.testing.assert_allclose(g*lower.D*rm,0.,atol=0.)
        np.testing.assert_allclose(g*upper.D*ad,g*ad,atol=1e-14)


@pytest.mark.parametrize('L',[3,4,5])
def test_free_condensate_signed_green_and_dark_removal(L):
    report=c.spectroscopy(L,g=0.)
    assert report['thresholds']['mu_plus']==pytest.approx(-2.,abs=1e-11)
    assert report['thresholds']['mu_minus']==pytest.approx(-2.,abs=1e-11)
    assert report['thresholds']['Delta_c']==pytest.approx(0.,abs=2e-11)
    for m,mode in enumerate(report['modes']):
        number=L if m==0 else 0
        assert mode['addition']['total_weight']==pytest.approx(number+1,abs=2e-12)
        assert mode['removal']['total_weight']==pytest.approx(number,abs=2e-12)
        for z in [.5j,1+.5j,4+1j]:
            assert c.green_function(report,m,z)==pytest.approx(1/(z+2*np.cos(2*np.pi*m/L)),abs=3e-11)


def test_free_dark_sector_threshold_is_explicit():
    report=c.spectroscopy(3,g=0.)
    for m,mode in enumerate(report['modes']):
        for name in ['addition','removal']:
            data=mode[name]
            expected=(4 if name=='addition' else 3) if m==0 else 0
            assert data['sector_ground_group_weight']==expected
            assert 'threshold_visibility' in data
            if m!=0:
                assert 'dark' in str(data['threshold_visibility']).lower()


def test_free_analytic_report_against_full_hamiltonian_resolvent():
    report=c.spectroscopy(3,g=0.)
    for m in range(3):
        models,matrices,spectra,E0,ground,remove,add=independent_sector_data(3,0.,m)
        z=1+.5j
        expected=np.vdot(add,np.linalg.solve((z+E0)*np.eye(len(add))-matrices[2],add))
        expected-=np.vdot(remove,np.linalg.solve((z-E0)*np.eye(len(remove))+matrices[0],remove))
        assert c.green_function(report,m,z)==pytest.approx(expected,abs=2e-12)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_reflection_of_complex_fourier_sources(reports,L,g):
    report=reports[L,g]
    for m in range(L):
        for z in [.5j,1+.5j,4+1j]:
            assert c.green_function(report,m,z)==pytest.approx(c.green_function(report,(-m)%L,z),abs=4e-11)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_isolated_stdout_demo(tmp_path,flags):
    script=Path(__file__).resolve().parents[1]/'scripts/demo_substrate_charged_response.py'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout,parse_constant=lambda value:pytest.fail(value))
    else:
        assert 'charged' in result.stdout.lower()


@pytest.mark.parametrize('L',[True,2,6,1000000,3.5])
def test_spectroscopy_site_cap(L):
    with pytest.raises((ValueError,TypeError)):
        c.spectroscopy(L)


@pytest.mark.parametrize('value',[True,np.nan,np.inf,-1.,1e-320])
def test_invalid_material_scalars(value):
    with pytest.raises((ValueError,TypeError)):
        c.spectroscopy(3,C=value)
    with pytest.raises((ValueError,TypeError)):
        c.spectroscopy(3,g=value)


def test_exact_tiny_scalar_cannot_become_zero():
    from fractions import Fraction
    with pytest.raises((ValueError,TypeError)):
        c.spectroscopy(3,g=Fraction(1,10**400))


@pytest.mark.parametrize('N',[True,0,-1,1000000,2.5])
def test_map_number_domain(N):
    with pytest.raises((ValueError,TypeError)):
        c.annihilation_map(3,N,0)


@pytest.mark.parametrize('m',[True,-1,3,1.5])
def test_map_momentum_domain(m):
    with pytest.raises((ValueError,TypeError)):
        c.annihilation_map(3,3,m)


@pytest.mark.parametrize('z',[True,0.,1.-1j,complex(np.inf,1),complex(1,np.nan),complex(1,1e-320)])
def test_green_frequency_domain(reports,z):
    with pytest.raises((ValueError,TypeError)):
        c.green_function(reports[3,.7],1,z)


@pytest.mark.parametrize('scale',[1e-80,1e80])
def test_energy_unit_rescaling_preserves_weights_and_green(reports,scale):
    reference=reports[3,40.]
    scaled=c.spectroscopy(3,C=scale,g=40*scale)
    for key in ['mu_plus','mu_minus','Delta_c']:
        assert scaled['thresholds'][key]/scale==pytest.approx(reference['thresholds'][key],rel=3e-10,abs=3e-11)
    for m in range(3):
        for name in ['addition','removal']:
            assert scaled['modes'][m][name]['total_weight']==pytest.approx(reference['modes'][m][name]['total_weight'],abs=3e-12)
        assert scale*c.green_function(scaled,m,scale*(1+.5j))==pytest.approx(c.green_function(reference,m,1+.5j),abs=3e-11)


def test_green_small_normal_real_component_is_not_erased():
    from fractions import Fraction
    z=complex(1e-17,1.)
    report={'parameters':{'L':3},'modes':[
        {'addition':{'energies':[-2.,2.],'weights':[.5,.5]},
         'removal':{'energies':[0.],'weights':[0.]}} for _ in range(3)]}
    real=Fraction.from_float(z.real)
    expected_real=sum((Fraction(1,2)*(real-d)/((real-d)**2+1) for d in [-2,2]),Fraction(0))
    expected_imag=-sum((Fraction(1,2)/((real-d)**2+1) for d in [-2,2]),Fraction(0))
    try:
        actual=c.green_function(report,0,z)
    except (ValueError,FloatingPointError):
        return
    assert actual.real!=0
    assert actual.real==pytest.approx(float(expected_real),rel=1e-13,abs=0.)
    assert actual.imag==pytest.approx(float(expected_imag),rel=1e-13,abs=0.)


def test_green_raw_spectrum_cap_precedes_conversion():
    class Oversize(list):
        def __array__(self,*args,**kwargs):
            pytest.fail('converted oversized raw spectrum before cap')
    bad=Oversize([0.]*513)
    report={'parameters':{'L':3},'modes':[
        {'addition':{'energies':bad,'weights':bad},'removal':{'energies':[1.],'weights':[1.]}}
        for _ in range(3)]}
    with pytest.raises((ValueError,TypeError)):
        c.green_function(report,0,1j)


def test_two_point_time_commutator_has_both_channel_signs(reports):
    from scipy.integrate import quad
    report=reports[3,.7]
    mode=report['modes'][1]
    dp,wp=np.array(mode['addition']['energies']),np.array(mode['addition']['weights'])
    dm,wm=np.array(mode['removal']['energies']),np.array(mode['removal']['weights'])
    z=1+.5j
    def integrand(t):
        commutator=np.sum(wp*np.exp(-1j*dp*t))-np.sum(wm*np.exp(1j*dm*t))
        return -1j*np.exp(1j*z*t)*commutator
    expected=quad(lambda t: integrand(t).real,0,80,epsabs=1e-10,limit=400)[0]
    expected+=1j*quad(lambda t: integrand(t).imag,0,80,epsabs=1e-10,limit=400)[0]
    assert c.green_function(report,1,z)==pytest.approx(expected,abs=1e-9)


def test_large_number_offset_does_not_fabricate_resolved_charge_gap():
    with pytest.raises((ValueError,TypeError,FloatingPointError)):
        c.spectroscopy(3,g=.7,kappa=1e20)


def test_public_cap_precedes_model_construction(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('allocated model before rejecting dimension/site cap')
    monkeypatch.setattr(c,'all_number_model',forbidden)
    for L in [6,1000000]:
        with pytest.raises((ValueError,TypeError)):
            c.spectroscopy(L)
    with pytest.raises((ValueError,TypeError)):
        c.annihilation_map(5,1000000,0)


def test_python38_grammar():
    import ast
    root=Path(__file__).resolve().parents[1]
    for path in ['bpr/substrate_charged_response.py','scripts/demo_substrate_charged_response.py']:
        ast.parse((root/path).read_text(),feature_version=(3,8))
