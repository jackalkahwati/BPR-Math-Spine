"""Independent canonical-mode and finite quantum-memory controls."""
import ast
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_collective_dynamics as c


def classical_energy(d, theta, nbar, C, g):
    psi=np.sqrt(nbar+d)*np.exp(1j*theta)
    return -2*C*np.vdot(psi,np.roll(psi,-1)).real+.5*g*np.sum(abs(psi)**4)


def laplacian(L):
    result=np.zeros((L,L))
    for x in range(L):
        edge=np.zeros(L); edge[x]=1; edge[(x+1)%L]=-1
        result+=np.outer(edge,edge)
    return result


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[0.,.7,40.]))
def test_classical_quadratic_energy_independent_differentiation(L,g):
    nbar=1.; C=1.
    lap=laplacian(L)
    Kn=g*np.eye(L)+C*lap/(2*nbar)
    Ktheta=2*C*nbar*lap
    d=np.cos(2*np.pi*np.arange(L)/L)
    theta=np.sin(2*np.pi*np.arange(L)/L)+.3*d
    d-=np.mean(d)
    predicted=d@Kn@d+theta@Ktheta@theta
    h=1e-3
    base=classical_energy(np.zeros(L),np.zeros(L),nbar,C,g)
    estimates=[]
    for step in [h,h/2]:
        plus=classical_energy(step*d,step*theta,nbar,C,g)
        minus=classical_energy(-step*d,-step*theta,nbar,C,g)
        estimates.append((plus+minus-2*base)/step**2)
    assert estimates[-1]==pytest.approx(predicted,rel=3e-6,abs=3e-6)
    # Cartesian DNLS linearization, not density/phase Hessian reuse.
    J=np.block([[np.zeros((L,L)),C*lap],[-C*lap-2*g*nbar*np.eye(L),np.zeros((L,L))]])
    eigen=np.linalg.eigvals(J)
    # Uniform numerical zero pair is not used as an oscillator.
    positive=sorted(v.imag for v in eigen if v.imag>1e-6)
    eps=4*C*np.sin(np.pi*np.arange(1,L)/L)**2
    np.testing.assert_allclose(positive,np.sort(np.sqrt(eps*(eps+2*g*nbar))),atol=2e-12)
    for m in range(1,L):
        z=1+.5j
        e=4*C*np.sin(np.pi*m/L)**2
        assert c.classical_response(L,m,z,nbar=nbar,C=C,g=g)==pytest.approx(
            2*nbar*e/(z*z-e*(e+2*g*nbar)),abs=2e-12)
    assert c.classical_response(L,0,1+.5j,nbar=nbar,C=C,g=g)==0j


@pytest.mark.parametrize('g',[0.,.7,40.])
def test_cartesian_linearization_against_nonlinear_dnls(g):
    L=4; nbar=1.; C=1.; mu=-2*C+g*nbar
    def rhs(u,w):
        psi=np.sqrt(nbar)+u+1j*w
        derivative=-1j*(-C*(np.roll(psi,1)+np.roll(psi,-1))+g*abs(psi)**2*psi-mu*psi)
        return np.concatenate([derivative.real,derivative.imag])
    step=1e-5
    derivative=np.empty((2*L,2*L))
    for j in range(2*L):
        delta=np.zeros(2*L); delta[j]=step
        derivative[:,j]=(rhs(delta[:L],delta[L:])-rhs(-delta[:L],-delta[L:]))/(2*step)
    lap=laplacian(L)
    expected=np.block([[np.zeros((L,L)),C*lap],[-C*lap-2*g*nbar*np.eye(L),np.zeros((L,L))]])
    np.testing.assert_allclose(derivative,expected,atol=2e-8)


def test_exact_two_line_projection_and_memory():
    H=np.diag([0.,1.,3.])
    source=np.array([0.,1.,1.])/np.sqrt(2)
    p=c.projected_system(H,source)
    for z in [.5j,1+.5j,4+1j]:
        assert c.self_energy(p,z)==pytest.approx(1/(z-2),abs=2e-13)
        assert c.source_resolvent(p,z)==pytest.approx((z-2)/((z-1)*(z-3)),abs=2e-13)
        assert c.source_resolvent(p,z)==pytest.approx(np.vdot(source,np.linalg.solve(z*np.eye(3)-H,source)),abs=2e-13)
    for t in [0.,.5,1.,2.]:
        assert c.memory_kernel(p,t)==pytest.approx(np.exp(-2j*t),abs=2e-13)


def test_memory_transform_sign_and_finite_time_tail_bound():
    from scipy.integrate import quad
    p=c.projected_system(np.diag([0.,1.,3.]),np.array([0.,1.,1.])/np.sqrt(2))
    T=12.
    for z in [.5j,1+.5j,4+1j]:
        def integrand(t):
            return -1j*np.exp(1j*z*t)*np.exp(-2j*t)
        partial=quad(lambda t:integrand(t).real,0,T,epsabs=1e-11)[0]
        partial+=1j*quad(lambda t:integrand(t).imag,0,T,epsabs=1e-11)[0]
        assert abs(c.self_energy(p,z)-partial)<=np.exp(-z.imag*T)/z.imag+2e-12


def test_exact_memory_convolution_sign():
    from scipy.integrate import quad
    p=c.projected_system(np.diag([0.,1.,3.]),np.array([0.,1.,1.])/np.sqrt(2))
    def survival(t):
        return .5*(np.exp(-1j*t)+np.exp(-3j*t))
    for t in [.5,1.,2.]:
        def integrand(u):
            return c.memory_kernel(p,t-u)*survival(u)
        integral=quad(lambda u:integrand(u).real,0,t,epsabs=1e-11)[0]
        integral+=1j*quad(lambda u:integrand(u).imag,0,t,epsabs=1e-11)[0]
        derivative=-.5j*(np.exp(-1j*t)+3*np.exp(-3j*t))
        assert derivative==pytest.approx(-2j*survival(t)-integral,abs=2e-12)


def test_single_transition_zero_memory_is_not_decay():
    p=c.projected_system(np.diag([0.,2.,5.]),np.array([0.,2.,0.]))
    for z in [.5j,1+.5j,4+1j]:
        assert c.self_energy(p,z)==0j
        assert c.source_resolvent(p,z)==pytest.approx(4/(z-2),abs=2e-13)
    for t in [0.,.5,1.,2.]:
        assert c.memory_kernel(p,t)==0j


def test_toy_passivity_memory_and_no_memory_bounds():
    H=np.diag([0.,1.,3.])
    source=np.array([0.,1.,1.])/np.sqrt(2)
    p=c.projected_system(H,source)
    for z in [.5j,1+.5j,4+1j]:
        F=c.source_resolvent(p,z); S=c.self_energy(p,z)
        assert F.imag<=1e-14 and S.imag<=1e-14
        assert abs(F)<=1/z.imag+1e-13
        assert abs(S)<=1/z.imag+1e-13
        assert abs(F-1/(z-2))<=1/z.imag**3+1e-13
    for t in [0.,.5,1.,2.]:
        survival=.5*(np.exp(-1j*t)+np.exp(-3j*t))
        assert abs(survival-np.exp(-2j*t))<=min(2.,t*t/2)+1e-13
        assert abs(c.memory_kernel(p,t))<=1+1e-13
    for eta,other in itertools.combinations([.25,.5,1.],2):
        bound=abs(eta-other)/(eta*other)
        assert abs(c.source_resolvent(p,1+1j*eta)-c.source_resolvent(p,1+1j*other))<=bound+1e-13
        assert abs(c.self_energy(p,1+1j*eta)-c.self_energy(p,1+1j*other))<=bound+1e-13


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_quantum_ring_projection_and_two_source_retarded_identity(L,g):
    from bpr.substrate_vacuum_selection import all_number_model
    from bpr.substrate_current_response import spectral_system, retarded_response
    model=all_number_model(L,L,1.,g)
    system=spectral_system(model.H)
    ground=system['ground']
    k=2*np.pi/L
    # Independent occupation-density Fourier source, no production rho helper.
    diagonal=np.array([sum(np.exp(-1j*k*x)*(state[x]-1) for x in range(L))/np.sqrt(L)
                       for state in model.basis])
    O=np.diag(diagonal)
    v=O@ground-np.vdot(ground,O@ground)*ground
    vd=O.conj().T@ground-np.vdot(ground,O.conj().T@ground)*ground
    p=c.projected_system(model.H,v)
    pd=c.projected_system(model.H,vd)
    A=model.H-system['energies'][0]*np.eye(len(model.H))
    for z in [.5j,1+.5j,4+1j]:
        actual=c.source_resolvent(p,z)
        direct=np.vdot(v,np.linalg.solve(z*np.eye(len(A))-A,v))
        assert actual==pytest.approx(direct,rel=2e-9,abs=2e-12)
        combined=c.source_resolvent(pd,z)+c.source_resolvent(p,-z.conjugate()).conjugate()
        assert combined==pytest.approx(retarded_response(system,O,O.conj().T,z),rel=2e-9,abs=2e-12)
        assert actual.imag<=2e-12
    beta=p['beta_squared']; s=p['s']; a=p['a']
    for z in [.5j,1+.5j,4+1j]:
        assert abs(c.self_energy(p,z))<=beta/z.imag+2e-10
        assert abs(c.source_resolvent(p,z)-s/(z-a))<=s*beta/(z.imag*abs(z-a)**2)+2e-10
    for t in [0.,.5,1.,2.]:
        assert abs(c.memory_kernel(p,t))<=beta+2e-10


def test_projection_basis_and_variance_identity():
    H=np.diag([0.,1.,3.,5.])
    source=np.array([0.,1.,2j,1.])
    p=c.projected_system(H,source)
    e=p['e']; W=p['W']; A=p['A']
    np.testing.assert_allclose(W.conj().T@W,np.eye(3),atol=2e-13)
    np.testing.assert_allclose(W.conj().T@e,0,atol=2e-13)
    np.testing.assert_allclose(W@W.conj().T+np.outer(e,e.conj()),np.eye(4),atol=2e-13)
    assert p['s']==pytest.approx(np.vdot(source,source).real,abs=1e-13)
    assert p['a']==pytest.approx(np.vdot(e,A@e).real,abs=2e-13)
    variance=np.vdot(A@e,A@e).real-p['a']**2
    assert p['beta_squared']==pytest.approx(variance,abs=2e-12)
    np.testing.assert_allclose(p['b'],W.conj().T@A@e,atol=2e-13)
    np.testing.assert_allclose(p['B'],W.conj().T@A@W,atol=2e-13)


def test_zero_source_has_no_normalized_memory():
    p=c.projected_system(np.diag([0.,2.,3.]),np.zeros(3))
    assert c.source_resolvent(p,1+.5j)==0j
    with pytest.raises(ValueError):
        c.self_energy(p,1+.5j)
    with pytest.raises(ValueError):
        c.memory_kernel(p,0.)
    with pytest.raises((ValueError,TypeError)):
        c.source_resolvent(p,0.)


def test_identity_energy_shift_leaves_projected_dynamics_unchanged():
    H=np.diag([0.,1.,3.])
    v=np.array([0.,1.,1j])/np.sqrt(2)
    base=c.projected_system(H,v)
    for offset in [-3.,2.5]:
        shifted=c.projected_system(H+offset*np.eye(3),v)
        for z in [.5j,1+.5j,4+1j]:
            assert c.source_resolvent(shifted,z)==pytest.approx(c.source_resolvent(base,z),abs=2e-12)
            assert c.self_energy(shifted,z)==pytest.approx(c.self_energy(base,z),abs=2e-12)
        assert c.memory_kernel(shifted,.5)==pytest.approx(c.memory_kernel(base,.5),abs=2e-12)


def test_nonconnected_generic_vector_includes_ground_weight():
    H=np.diag([0.,1.,3.])
    v=np.array([1.,1j,2.])
    p=c.projected_system(H,v)
    z=1+.5j
    assert c.source_resolvent(p,z)==pytest.approx(1/z+1/(z-1)+4/(z-3),abs=2e-12)


@pytest.mark.parametrize('scale',[1e-40,1e40])
def test_source_normalization_not_lost_in_projection(scale):
    H=np.diag([0.,1.,3.])
    v=np.array([0.,1.,1j])/np.sqrt(2)
    p=c.projected_system(H,scale*v)
    for z in [.5j,1+.5j,4+1j]:
        assert c.source_resolvent(p,z)/scale**2==pytest.approx(.5/(z-1)+.5/(z-3),abs=2e-12)
        assert c.self_energy(p,z)==pytest.approx(1/(z-2),abs=2e-12)
    for t in [0.,.5,1.,2.]:
        assert c.memory_kernel(p,-t)==pytest.approx(c.memory_kernel(p,t).conjugate(),abs=2e-13)


def test_complex_projected_source_matches_full_resolvent():
    # Independent unitary change of a diagonal Hamiltonian with complex source.
    from scipy.linalg import null_space, expm
    raw=np.array([[1.,1j,2.],[2j,1.,1j],[1.,2.,1.]])
    U,_=np.linalg.qr(raw)
    H=U@np.diag([0.,1.,3.])@U.conj().T
    H=(H+H.conj().T)/2
    source=U@np.array([0.,1.,2j])
    p=c.projected_system(H,source)
    E=np.linalg.eigvalsh(H)
    A=H-E[0]*np.eye(3)
    s=np.vdot(source,source).real
    e=source/np.sqrt(s)
    W=null_space(e.conj()[None,:])
    B=W.conj().T@A@W
    b=W.conj().T@A@e
    for z in [.5j,1+.5j,4+1j]:
        direct=np.vdot(source,np.linalg.solve(z*np.eye(3)-A,source))
        sigma=np.vdot(b,np.linalg.solve(z*np.eye(2)-B,b))
        assert c.source_resolvent(p,z)==pytest.approx(direct,abs=2e-11)
        assert c.self_energy(p,z)==pytest.approx(sigma,abs=2e-11)
    for t in [0.,.5,1.,2.]:
        assert c.memory_kernel(p,t)==pytest.approx(np.vdot(b,expm(-1j*t*B)@b),abs=2e-11)


@pytest.mark.parametrize('scale',[1e-60,1e60])
def test_projection_energy_unit_scaling(scale):
    source=np.array([0.,1.,1.])/np.sqrt(2)
    base=c.projected_system(np.diag([0.,1.,3.]),source)
    shifted=c.projected_system(scale*np.diag([0.,1.,3.]),source)
    z=1+.5j
    assert scale*c.source_resolvent(shifted,z*scale)==pytest.approx(c.source_resolvent(base,z),abs=3e-12)
    assert c.self_energy(shifted,z*scale)/scale==pytest.approx(c.self_energy(base,z),abs=3e-12)
    assert c.memory_kernel(shifted,.5/scale)/scale**2==pytest.approx(c.memory_kernel(base,.5),abs=3e-12)


@pytest.mark.parametrize('z',[True,0.,1-1j,complex(1,np.nan),complex(1,1e-320)])
def test_frequency_domain_validated_before_structural_return(z):
    p=c.projected_system(np.diag([0.,2.,3.]),np.array([0.,1.,0.]))
    for function in [c.source_resolvent,c.self_energy]:
        with pytest.raises((ValueError,TypeError)):
            function(p,z)
    with pytest.raises((ValueError,TypeError)):
        c.classical_response(3,0,z)


@pytest.mark.parametrize('kwargs',[{'C':1e-200,'g':0.},{'C':1e-100,'nbar':1e-230,'g':1.}])
def test_underflowed_classical_coefficients_are_not_fabricated_zero(kwargs):
    with pytest.raises((ValueError,TypeError)):
        c.classical_modes(3,**kwargs)
    with pytest.raises((ValueError,TypeError)):
        c.classical_response(3,1,1e-100j,**kwargs)


@pytest.mark.parametrize('L',[True,2,6,1000000,3.5])
def test_classical_site_domain(L):
    with pytest.raises((ValueError,TypeError)):
        c.classical_modes(L)


@pytest.mark.parametrize('kwargs',[{'nbar':0.},{'nbar':-1.},{'C':0.},{'g':-1.},{'g':True}])
def test_classical_material_domain(kwargs):
    with pytest.raises((ValueError,TypeError)):
        c.classical_modes(3,**kwargs)


@pytest.mark.parametrize('source',[[0.,1.],[[0.],[1.],[0.]],[0.,1.,float('nan')]])
def test_projected_source_shape_and_scalar_domain(source):
    with pytest.raises((ValueError,TypeError)):
        c.projected_system(np.diag([0.,1.,3.]),source)


def test_dense_cap_checked_before_conversion():
    class Oversize(list):
        def __array__(self,*args,**kwargs):
            pytest.fail('unbounded matrix converted before cap')
    with pytest.raises((ValueError,TypeError)):
        c.projected_system(Oversize([[0.]]*513),[0.]*513)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[0.,.7,40.]))
def test_classical_report_dispersion_and_zero_mode_contract(L,g):
    report=c.classical_modes(L,g=g)
    assert report['parameters']['L']==L
    assert len(report['modes'])==L
    for row in report['modes']:
        m=row['m']; eps=4*np.sin(np.pi*m/L)**2
        assert row['epsilon']==pytest.approx(eps,abs=2e-13)
        assert row['omega_squared']==pytest.approx(eps*(eps+2*g),abs=2e-11)
        if m:
            assert row['static_susceptibility']==pytest.approx(1/(g+eps/2),abs=2e-13)
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_quantum_report_frozen_counts_and_strict_json(L,g):
    report=c.quantum_case(L,g=g)
    assert report['parameters']['L']==L
    assert report['source']['s']>0
    assert len(report['responses'])==3
    assert len(report['memory'])==4
    assert report['source']['beta_squared']>=0
    for response in report['responses']:
        assert response['full_resolvent_residual']<2e-9
        assert response['spectral_residual']<2e-9
        assert response['retarded_identity_residual']<2e-9
        assert response['self_energy_direct_residual']<2e-9
        assert response['passivity_identity_residual']<2e-9
        assert response['F_abs']<=response['F_bound']+2e-10
        assert response['one_line_error']<=response['one_line_bound']+2e-10
    for row in report['memory']:
        assert row['K_abs']<=row['K_bound']+2e-10
        assert row['complementary_leakage']<=row['leakage_bound']+2e-10
        assert row['one_line_amplitude_error']<=row['one_line_amplitude_bound']+2e-10
    for row in report['regulator_comparisons']:
        assert row['F_difference']<=row['F_bound']+2e-10
        assert row['Sigma_difference']<=row['Sigma_bound']+2e-10
    assert report['uniform_density_control']['F']=={'real':0.,'imag':0.}
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_stdout_only_demo(tmp_path,flags):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(root/'scripts/demo_substrate_collective_dynamics.py')]+flags,
                          cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout,parse_constant=lambda x:pytest.fail(x))
    else:
        assert 'classical' in result.stdout.lower() and 'memory' in result.stdout.lower()


def test_python38_grammar():
    root=Path(__file__).resolve().parents[1]
    for name in ['bpr/substrate_collective_dynamics.py','scripts/demo_substrate_collective_dynamics.py']:
        ast.parse((root/name).read_text(),feature_version=(3,8))
