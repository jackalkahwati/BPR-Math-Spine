"""Independent conservation and source-response checks on the unchanged ring."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_current_response as c
from bpr.substrate_vacuum_selection import all_number_model
from bpr.substrate_neutral_response import density_diagonal


def transfer_oracle(model,x):
    """Bra/ket occupation selection, independent of production bilinear helper."""
    result=np.zeros_like(model.H,dtype=complex)
    y=(x+1)%model.L
    for row,bra in enumerate(model.basis):
        for col,ket in enumerate(model.basis):
            difference=np.array(bra)-np.array(ket)
            if difference[x]==-1 and difference[y]==1 and np.count_nonzero(difference)==2:
                result[row,col]=np.sqrt(ket[x]*(ket[y]+1))
    return result


def peierls_oracle(model,phases):
    result=np.diag(model.g*model.D).astype(complex)
    for x,phase in enumerate(phases):
        T=transfer_oracle(model,x)
        result-=model.C*(np.exp(1j*phase)*T+np.exp(-1j*phase)*T.conj().T)
    return result


def commutator(A,B):
    return A@B-B@A


def mixed_lehmann(H,A,B,z):
    energies,vectors=np.linalg.eigh(H)
    g=vectors[:,0]
    delta=energies-energies[0]
    # Ground elastic terms cancel; excited terms need distinct numerators.
    result=0j
    for i in range(1,len(energies)):
        state=vectors[:,i]
        first=np.vdot(g,A@state)*np.vdot(state,B@g)
        second=np.vdot(g,B@state)*np.vdot(state,A@g)
        result+=first/(z-delta[i])-second/(z+delta[i])
    return result


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_number_current_and_exact_continuity(L,g):
    model=all_number_model(L,L,1.,g)
    currents=c.bond_currents(model)
    assert len(currents)==L
    for x in range(L):
        T=transfer_oracle(model,x)
        expected=1j*(T-T.conj().T)
        np.testing.assert_allclose(currents[x],expected,atol=1e-14)
        np.testing.assert_allclose(currents[x],currents[x].conj().T,atol=0.)
        number=np.diag([state[x] for state in model.basis])
        np.testing.assert_allclose(1j*commutator(model.H,number),currents[x-1]-currents[x],atol=2e-13)
    for m in range(L):
        k=2*np.pi*m/L
        rho=np.diag(density_diagonal(model,m))
        current=sum(np.exp(-1j*k*x)*currents[x] for x in range(L))/np.sqrt(L)
        np.testing.assert_allclose(1j*commutator(model.H,rho),(np.exp(-1j*k)-1)*current,atol=4e-13)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_peierls_sources_and_gauge_covariance(L,g):
    model=all_number_model(L,L,1.,g)
    zero=np.zeros(L)
    np.testing.assert_array_equal(c.peierls_hamiltonian(model,zero),model.H)
    phases=np.full(L,.3/L)
    expected=peierls_oracle(model,phases)
    H=c.peierls_hamiltonian(model,phases)
    np.testing.assert_allclose(H,expected,atol=1e-14)
    theta=.2*np.cos(2*np.pi*np.arange(L)/L)
    shifted=phases+np.roll(theta,-1)-theta
    U=np.diag(np.exp(1j*np.array(model.basis)@theta))
    np.testing.assert_allclose(c.peierls_hamiltonian(model,shifted),U@H@U.conj().T,atol=3e-13)
    original=c.bond_currents(model,phases)
    changed=c.bond_currents(model,shifted)
    for x in range(L):
        number=np.diag([state[x] for state in model.basis])
        np.testing.assert_allclose(1j*commutator(H,number),original[x-1]-original[x],atol=3e-13)
    for a,b in zip(original,changed):
        np.testing.assert_allclose(b,U@a@U.conj().T,atol=2e-14)
    single=np.zeros(L); single[-1]=.3
    np.testing.assert_allclose(np.linalg.eigvalsh(c.peierls_hamiltonian(model,single)),np.linalg.eigvalsh(H),atol=6e-12)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_mixed_ward_contacts_and_fsum(L,g):
    model=all_number_model(L,L,1.,g)
    system=c.spectral_system(model.H)
    currents=c.bond_currents(model)
    E,U=np.linalg.eigh(model.H)
    ground=U[:,0]
    for m in range(L):
        k=2*np.pi*m/L
        rho=np.diag(density_diagonal(model,m))
        jk=sum(np.exp(-1j*k*x)*currents[x] for x in range(L))/np.sqrt(L)
        source=rho@ground
        moment=np.vdot(source,(model.H-E[0]*np.eye(len(E)))@source).real
        oracle=-(1-np.cos(k))*np.vdot(ground,model.V@ground).real/L
        assert moment==pytest.approx(oracle,abs=3e-11)
        double=np.vdot(ground,commutator(rho.conj().T,commutator(model.H,rho))@ground)/2
        assert double==pytest.approx(moment,abs=3e-11)
        B=jk.conj().T
        dA=1j*commutator(model.H,rho)
        dB=1j*commutator(model.H,B)
        contact=np.vdot(ground,commutator(rho,B)@ground)
        expected_contact=-1j*(np.exp(-1j*k)-1)*np.vdot(ground,model.V@ground)/L
        assert contact==pytest.approx(expected_contact,abs=3e-12)
        for z in [.5j,1+.5j,4+1j]:
            density_response=c.retarded_response(system,rho,rho.conj().T,z)
            current_response=c.retarded_response(system,jk,B,z)
            assert z*z*density_response==pytest.approx(2*(1-np.cos(k))*(current_response-np.vdot(ground,model.V@ground)/L),abs=1e-9)
            response=c.retarded_response(system,rho,B,z)
            assert response==pytest.approx(mixed_lehmann(model.H,rho,B,z),abs=3e-11)
            assert z*response==pytest.approx(contact+1j*c.retarded_response(system,dA,B,z),abs=5e-10)
            assert z*response==pytest.approx(contact-1j*c.retarded_response(system,rho,dB,z),abs=5e-10)
        if m!=0:
            assert abs(contact)>1e-5


@pytest.mark.parametrize('g',[.7,40.])
def test_uniform_conserved_sources_are_structural_zero(g):
    model=all_number_model(3,3,1.,g)
    system=c.spectral_system(model.H)
    B=c.bond_currents(model)[0]
    for A in [np.zeros_like(model.H),np.eye(len(model.H)),model.H]:
        assert c.retarded_response(system,A,B,1+.5j)==0
        assert c.retarded_response(system,B,A,1+.5j)==0


def test_generic_nonhermitian_mixed_observables_keep_distinct_numerators():
    H=np.diag([-.3,1.,2.5]).astype(complex)
    H[0,1]=.2j; H[1,0]=-.2j
    A=np.array([[0,1j,2],[3,0,-1j],[1,2,0]],dtype=complex)
    B=np.array([[1,2,0],[0,1j,3],[4,-1,2]],dtype=complex)
    system=c.spectral_system(H)
    for z in [.5j,1+.5j,4+1j]:
        actual=c.retarded_response(system,A,B,z)
        assert actual==pytest.approx(mixed_lehmann(H,A,B,z),abs=1e-12)
        E,U=np.linalg.eigh(H)
        ground=U[:,0]
        excitation=H-E[0]*np.eye(3)
        # Full-space matrix inverses, with equal elastic terms canceling.
        direct=np.vdot(ground,A@np.linalg.solve(z*np.eye(3)-excitation,B@ground))
        direct-=np.vdot(ground,B@np.linalg.solve(z*np.eye(3)+excitation,A@ground))
        assert actual==pytest.approx(direct,abs=1e-12)
        conjugate=c.retarded_response(system,A.conj().T,B.conj().T,-z.conjugate())
        assert actual.conjugate()==pytest.approx(conjugate,abs=1e-12)


def test_degenerate_excited_basis_rotation_does_not_change_response():
    H=np.diag([0.,2.,2.])
    A=np.array([[0.,1.,2j],[3.,1.,0.],[4j,0.,2.]])
    B=np.array([[0.,-2j,1.],[1j,0.,3.],[2.,1.,0.]])
    system=c.spectral_system(H)
    rotated=dict(system)
    W=np.array([[1.,1j],[1j,1.]])/np.sqrt(2)
    vectors=system['vectors'].copy()
    vectors[:,1:]=vectors[:,1:]@W
    rotated['vectors']=vectors
    for z in [.5j,1+.5j,4+1j]:
        assert c.retarded_response(rotated,A,B,z)==pytest.approx(
            c.retarded_response(system,A,B,z),abs=1e-13)


def test_phase_covariant_mixed_response_and_static_sources():
    model=all_number_model(3,3,1.,.7)
    theta=.2*np.cos(2*np.pi*np.arange(3)/3)
    phases=np.roll(theta,-1)-theta
    U=np.diag(np.exp(1j*np.array(model.basis)@theta))
    original=c.spectral_system(model.H)
    transformed=c.spectral_system(c.peierls_hamiltonian(model,phases))
    A=np.diag(density_diagonal(model,1))
    B=c.bond_currents(model)[0]
    for z in [.5j,1+.5j,4+1j]:
        assert c.retarded_response(original,A,B,z)==pytest.approx(
            c.retarded_response(transformed,U@A@U.conj().T,U@B@U.conj().T,z),abs=3e-12)


@pytest.mark.parametrize('L',[3,4,5])
def test_free_flux_curvature_and_frozen_refinement(L):
    model=all_number_model(L,L,1.,0.)
    system=c.spectral_system(model.H)
    ground=system['ground']
    J=sum(c.bond_currents(model))
    np.testing.assert_allclose(J@ground,0,atol=3e-14)
    diamagnetic=-np.vdot(ground,model.V@ground).real/L**2
    assert diamagnetic==pytest.approx(2/L,abs=1e-13)
    for step in [.04,.02,.01,.005]:
        energy=np.linalg.eigvalsh(c.peierls_hamiltonian(model,np.full(L,step/L)))[0]
        assert energy==pytest.approx(-2*L*np.cos(step/L),abs=1e-12)
        estimate=2*(energy-system['energies'][0])/step**2
        assert abs(estimate-2/L)<step**2/(5*L**3)+1e-7


def test_peierls_first_and_second_derivatives_use_total_flux():
    model=all_number_model(3,3,1.,.7)
    h=.0001
    plus=peierls_oracle(model,np.full(3,h/3))
    minus=peierls_oracle(model,np.full(3,-h/3))
    J=sum(c.bond_currents(model))
    np.testing.assert_allclose((plus-minus)/(2*h),-J/3,atol=3e-10)
    np.testing.assert_allclose((plus+minus-2*model.H)/h**2,-model.V/9,atol=2e-7)


def test_retarded_time_integral_has_both_mixed_signs():
    from scipy.integrate import quad
    H=np.diag([0.,2.])
    A=np.array([[0.,1.],[1.,0.]])
    B=np.array([[0.,-1j],[1j,0.]])
    system=c.spectral_system(H)
    for z in [.5j,1+.5j,4+1j]:
        # <[sigma_x(t),sigma_y]>=2i*cos(2t), directly in time domain.
        def integrand(t):
            return 2*np.exp(1j*z*t)*np.cos(2*t)
        value=quad(lambda t:integrand(t).real,0,np.inf,epsabs=1e-10,limit=300)[0]
        value+=1j*quad(lambda t:integrand(t).imag,0,np.inf,epsabs=1e-10,limit=300)[0]
        assert c.retarded_response(system,A,B,z)==pytest.approx(value,abs=1e-9)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_interacting_flux_hessian_independent_finite_difference(L,g):
    model=all_number_model(L,L,1.,g)
    E,U=np.linalg.eigh(model.H)
    ground=U[:,0]
    J=sum(c.bond_currents(model))
    overlaps=U[:,1:].conj().T@(J@ground)
    curvature=(-np.vdot(ground,model.V@ground).real-2*np.sum(abs(overlaps)**2/(E[1:]-E[0])))/L**2
    errors=[]
    for step in [.04,.02,.01,.005]:
        plus=np.linalg.eigvalsh(peierls_oracle(model,np.full(L,step/L)))[0]
        minus=np.linalg.eigvalsh(peierls_oracle(model,np.full(L,-step/L)))[0]
        estimate=(plus+minus-2*E[0])/step**2
        errors.append(abs(estimate-curvature))
    # Resolution of finite differences is finite-size numerical evidence only.
    assert min(errors)<2e-6
    assert errors[1]<errors[0]+2e-7


def test_two_level_cross_response_and_identity_shifts():
    H=np.diag([0.,2.])
    A=np.array([[0.,1.],[1.,0.]])
    B=np.array([[0.,-1j],[1j,0.]])
    system=c.spectral_system(H)
    for z in [.5j,1+.5j,4+1j]:
        expected=1j/(z-2)+1j/(z+2)
        assert c.retarded_response(system,A,B,z)==pytest.approx(expected,abs=2e-14)
        assert c.retarded_response(system,A+3*np.eye(2),B-2j*np.eye(2),z)==pytest.approx(expected,abs=2e-14)


@pytest.mark.parametrize('L,g',itertools.product([3,4,5],[.7,40.]))
def test_report_has_frozen_diagnostics_and_no_unavailable_fake_zero(L,g):
    r=c.case_report(L,g=g)
    assert r['parameters']['L']==L
    assert r['continuity']['local_max_residual']<1e-10
    assert r['continuity']['fourier_max_residual']<1e-10
    assert len(r['momenta'])==L
    for mode in r['momenta']:
        assert mode['fsum']['residual']<1e-9
        assert mode['contact_residual']<1e-9
        for response in mode['responses']:
            assert response['ward_first_residual']<1e-8
            assert response['ward_second_residual']<1e-8
            assert response['ward_combined_residual']<1e-8
    curvature=r['curvature']
    assert [row['step'] for row in curvature['finite_differences']]==[.04,.02,.01,.005]
    if curvature['value'] is not None:
        assert curvature['value']>=-1e-10
        assert curvature['value']<=curvature['diamagnetic']+1e-10
    for row in curvature['finite_differences']:
        if row['value'] is None:
            assert 'unresolved' in row['status']
    assert r['gauge']['hamiltonian_residual']<1e-10
    assert r['gauge']['current_max_residual']<1e-10
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('g',[.7,40.])
def test_diamagnetic_finite_flux_bounds_from_full_matrices(g):
    model=all_number_model(3,3,1.,g)
    E,U=np.linalg.eigh(model.H)
    K=-np.vdot(U[:,0],model.V@U[:,0]).real
    for phi in [.04,.02,.01,.005,.3]:
        energy=np.linalg.eigvalsh(peierls_oracle(model,np.full(3,phi/3)))[0]
        delta=energy-E[0]
        assert delta>=-3e-12
        assert delta<=K*(1-np.cos(phi/3))+3e-12


@pytest.mark.parametrize('flags',[[],['--json']])
def test_stdout_only_demo(tmp_path,flags):
    script=Path(__file__).resolve().parents[1]/'scripts/demo_substrate_current_response.py'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout,parse_constant=lambda x:pytest.fail(x))
    else:
        assert 'current' in result.stdout.lower()


@pytest.mark.parametrize('H',[np.zeros((2,2)),np.diag([0.,0.,1.]),np.array([[0.,1.],[0.,2.]])])
def test_unresolved_or_nonhermitian_ground_rejected(H):
    with pytest.raises((ValueError,TypeError)):
        c.spectral_system(H)


@pytest.mark.parametrize('L',[True,2,6,1000000,3.5])
def test_report_site_cap(L):
    with pytest.raises((ValueError,TypeError)):
        c.case_report(L)


@pytest.mark.parametrize('z',[True,0.,1-1j,complex(1,np.nan),complex(1,1e-320)])
def test_response_frequency_domain(z):
    system=c.spectral_system(np.diag([0.,1.]))
    with pytest.raises((ValueError,TypeError)):
        c.retarded_response(system,np.eye(2),np.eye(2),z)


def test_shape_cap_precedes_custom_conversion():
    class Oversize(list):
        def __array__(self,*args,**kwargs):
            pytest.fail('converted oversized operator before cap')
    value=Oversize([[0.]]*513)
    with pytest.raises((ValueError,TypeError)):
        c.spectral_system(value)
    system=c.spectral_system(np.diag([0.,1.]))
    with pytest.raises((ValueError,TypeError)):
        c.retarded_response(system,value,np.eye(2),1j)


def test_exact_tiny_material_parameter_rejected():
    from fractions import Fraction
    with pytest.raises((ValueError,TypeError)):
        c.case_report(3,g=Fraction(1,10**400))


@pytest.mark.parametrize('scale',[1e-80,1e80])
def test_generic_response_energy_rescaling(scale):
    H=np.diag([0.,2.])
    A=np.array([[0.,1.],[1.,0.]])
    B=np.array([[0.,-1j],[1j,0.]])
    system=c.spectral_system(H*scale)
    z=(1+.5j)*scale
    expected=1j/(1+.5j-2)+1j/(1+.5j+2)
    assert scale*c.retarded_response(system,A,B,z)==pytest.approx(expected,abs=3e-13)


@pytest.mark.parametrize('phase',[1.,1j])
def test_single_pole_denominator_erasure_keeps_small_component(phase):
    system=c.spectral_system(np.diag([0.,2.]))
    A=np.array([[0.,-2+1j],[0.,0.]])
    B=phase*np.array([[0.,0.],[1.,0.]])
    z=complex(1e-20,1.)
    # Analytic real/imaginary form avoids cancellation in the independent oracle.
    x=z.real
    denominator=(x-2)**2+1
    base=complex((5-2*x)/denominator,x/denominator)
    expected=phase*base
    try:
        actual=c.retarded_response(system,A,B,z)
    except c.NumericalUnavailable:
        return
    assert actual.real==pytest.approx(expected.real,rel=2e-12,abs=0.)
    assert actual.imag==pytest.approx(expected.imag,rel=2e-12,abs=0.)


def test_large_spectator_transition_cannot_mask_canceled_mixed_channel():
    H=np.diag(np.arange(9,dtype=float)+10)
    H[:2,:2]=[[0.,1.],[1.,0.]]
    system=c.spectral_system(H)
    A=np.zeros((9,9)); A[:2,:2]=[[0.,1e16],[1e16+2,0.]]
    A[0,2]=A[2,0]=1e16
    B=np.zeros((9,9)); B[:2,:2]=np.diag([1.,-1.])
    A0=np.zeros((9,9)); A0[1,0]=2.
    z=1+.5j
    expected=c.retarded_response(system,A0,B,z)
    try:
        actual=c.retarded_response(system,A,B,z)
    except c.NumericalUnavailable:
        return
    assert actual==pytest.approx(expected,rel=2e-12,abs=0.)


def test_disconnected_spectator_does_not_erase_active_diagonal_source():
    H=np.array([[10.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
    system=c.spectral_system(H)
    A=np.diag([1e16,1.,0.])
    B=np.diag([0.,1.,0.])
    z=1+.5j
    try:
        actual=c.retarded_response(system,A,B,z)
    except c.NumericalUnavailable:
        return
    assert actual==pytest.approx(1/(z*z-4),rel=2e-12,abs=0.)


@pytest.mark.parametrize('scale',[1e12,1e16])
def test_large_stationary_source_addition_does_not_erase_transition(scale):
    H=np.array([[0.,1.],[1.,0.]])
    system=c.spectral_system(H)
    A0=np.array([[0.,0.],[2.,0.]])
    A=A0+scale*H
    B=np.diag([1.,-1.])
    expected=c.retarded_response(system,A0,B,1+.5j)
    try:
        actual=c.retarded_response(system,A,B,1+.5j)
    except c.NumericalUnavailable:
        return
    assert actual==pytest.approx(expected,rel=2e-12,abs=0.)


def test_small_response_components_and_genuine_cancellation():
    system=c.spectral_system(np.diag([0.,2.]))
    A=np.array([[0.,1.],[1.,0.]])
    B=np.array([[0.,-1j],[1j,0.]])
    z=complex(1e-20,1.)
    expected=2j*z/(z*z-4)
    try:
        actual=c.retarded_response(system,A,B,z)
    except c.NumericalUnavailable:
        pass
    else:
        assert actual.real==pytest.approx(expected.real,rel=2e-12,abs=0.)
        assert actual.imag==pytest.approx(expected.imag,rel=2e-12,abs=0.)
    diagonal=c.spectral_system(np.diag([0.,2.,2.]))
    X=np.zeros((3,3)); X[0,1:]=[1.,-1.]
    Y=np.zeros((3,3)); Y[1:,0]=1.
    assert c.retarded_response(diagonal,X,Y,1j)==0j


def test_cancellation_between_excited_states_keeps_small_remainder():
    system=c.spectral_system(np.diag([0.,2.,2.,2.]))
    A=np.zeros((4,4)); A[0,1:]=[1e16,1.,-1e16]
    B=np.zeros((4,4)); B[1:,0]=1.
    try:
        actual=c.retarded_response(system,A,B,1j)
    except c.NumericalUnavailable:
        return
    assert actual==pytest.approx(1/(1j-2),rel=2e-12,abs=0.)


@pytest.mark.parametrize('real_z',[1e16,1e20])
@pytest.mark.parametrize('phase',[1.,1j])
def test_large_frequency_cancellation_is_resolved_or_explicitly_unavailable(real_z,phase):
    H=np.diag([0.,2.])
    A=np.array([[0.,1.],[1.,0.]])
    system=c.spectral_system(H)
    z=complex(real_z,1.)
    expected=phase*4/(z*z-4)
    try:
        actual=c.retarded_response(system,A,phase*A,z)
    except c.NumericalUnavailable:
        return
    # Check components separately; a large component must not mask loss of another.
    assert actual.real==pytest.approx(expected.real,rel=2e-12,abs=0.)
    assert actual.imag==pytest.approx(expected.imag,rel=2e-12,abs=0.)


def test_new_python38_grammar():
    import ast
    root=Path(__file__).resolve().parents[1]
    for p in ['bpr/substrate_current_response.py','scripts/demo_substrate_current_response.py']:
        ast.parse((root/p).read_text(),feature_version=(3,8))
