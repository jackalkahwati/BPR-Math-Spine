"""Independent determinant, occupation and symmetry checks for candidate matching."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_quantum_matching as q
from bpr import substrate_chirality as chirality


def occupation_basis(L,N):
    return [tuple(x for x in range(L) if bit&(1<<x))
            for bit in range(1<<L) if bin(bit).count('1')==N]


def small_determinant(a):
    # Independent Leibniz oracle: retained minors have order at most three.
    n = len(a)
    total = 0j
    for p in itertools.permutations(range(n)):
        inversions = sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))
        total += (-1)**inversions*np.prod([a[i,p[i]] for i in range(n)])
    return total


def determinant_embedding(L,N):
    theta = 0. if N%2 else np.pi
    U = np.exp(1j*np.outer(np.arange(L),2*np.pi*np.array([-1,0,1])+theta)/L)/np.sqrt(L)
    rows = occupation_basis(L,N)
    columns = list(itertools.combinations(range(3),N))
    W = np.zeros((len(rows),len(columns)),complex)
    for i,sites in enumerate(rows):
        for j,modes in enumerate(columns):
            W[i,j] = small_determinant(U[np.ix_(sites,modes)]) if N else 1.
    return U,W


def occupation_permutation(L,N,kind):
    basis = occupation_basis(L,N)
    index = {sites:i for i,sites in enumerate(basis)}
    P = np.zeros((len(basis),len(basis)))
    for j,sites in enumerate(basis):
        moved = tuple(sorted((x-1)%L if kind=='translation' else (-x)%L for x in sites))
        P[index[moved],j] = 1
    return P


def real_matrix_span(matrices):
    if not matrices or matrices[0].size==0:
        return 0
    columns = np.column_stack([np.concatenate((a.real.ravel(),a.imag.ravel())) for a in matrices])
    return np.linalg.matrix_rank(columns,tol=1e-10)


def exterior_operator(matrix,N):
    rows = list(itertools.combinations(range(matrix.shape[0]),N))
    columns = list(itertools.combinations(range(matrix.shape[1]),N))
    result = np.zeros((len(rows),len(columns)),complex)
    for i,a in enumerate(rows):
        for j,b in enumerate(columns):
            result[i,j] = small_determinant(matrix[np.ix_(a,b)]) if N else 1.
    return result


def independent_complex_commutant_dimension(matrices):
    if not matrices or matrices[0].size==0:
        return 0
    d = len(matrices[0])
    constraints = np.vstack([np.kron(np.eye(d),a)-np.kron(a.T,np.eye(d)) for a in matrices])
    return d*d-np.linalg.matrix_rank(constraints,tol=1e-10)


CASES = [(L,N) for L in (5,6,7) for N in (1,2,3)]+[(5,0),(3,3),(4,4),(5,5)]


@pytest.mark.parametrize('N',[0,1,2,3,4])
def test_exterior_product_and_additive_derivative(N):
    A = np.array([[1,.3j,-.2],[-.3j,2,.4],[-.2,.4,3]],complex)
    B = np.array([[.1,1,0],[0,.2,1],[1,0,.3]],complex)
    np.testing.assert_allclose(q.exterior_power(A,N),exterior_operator(A,N),atol=3e-14)
    np.testing.assert_allclose(q.exterior_power(A@B,N),q.exterior_power(A,N)@q.exterior_power(B,N),atol=3e-14)
    step = 1e-5
    derivative = (exterior_operator(np.eye(3)+step*A,N)-exterior_operator(np.eye(3)-step*A,N))/(2*step)
    np.testing.assert_allclose(q.exterior_lift(A,N),derivative,atol=1e-8)


@pytest.mark.parametrize('L,N',CASES)
def test_actual_determinant_embedding_and_dimensions(L,N):
    U,W = determinant_embedding(L,N)
    model = q.projected_model(L,N)
    np.testing.assert_allclose(model.U,U,atol=2e-14)
    np.testing.assert_allclose(model.W,W,atol=2e-14)
    assert model.W.shape==(len(occupation_basis(L,N)),len(list(itertools.combinations(range(3),N))))
    np.testing.assert_allclose(model.W.conj().T@model.W,np.eye(model.W.shape[1]),atol=2e-14)
    np.testing.assert_allclose(q.wedge_embedding(L,N),W,atol=2e-14)
    if model.complement.shape[1]:
        np.testing.assert_allclose(model.W.conj().T@model.complement,0.,atol=3e-14)
        np.testing.assert_allclose(model.complement.conj().T@model.complement,np.eye(model.complement.shape[1]),atol=3e-14)


@pytest.mark.parametrize('L,N',CASES)
def test_projected_dynamics_and_density_against_occupation_oracle(L,N):
    model = q.projected_model(L,N,C=1.7)
    _,W = determinant_embedding(L,N)
    H = chirality.hard_core_hopping(L,N,1.7)
    np.testing.assert_allclose(model.H,H,atol=2e-14)
    np.testing.assert_allclose(model.Hproj,W.conj().T@H@W,atol=3e-14)
    np.testing.assert_allclose(H@W,W@model.Hproj,atol=5e-14)
    basis = occupation_basis(L,N)
    for x in range(L):
        n = np.diag([float(x in row) for row in basis])
        projected = W.conj().T@n@W
        np.testing.assert_allclose(model.projected_densities[x],projected,atol=3e-14)
        np.testing.assert_allclose(model.density_leakages[x],n@W-W@projected,atol=3e-14)
    np.testing.assert_allclose(sum(model.projected_densities),N*np.eye(W.shape[1]),atol=3e-14)


@pytest.mark.parametrize('L,N',CASES)
def test_actual_many_body_spectral_separation_not_one_body_gap(L,N):
    model = q.projected_model(L,N)
    r = q.matching_diagnostics(L,N)
    assert r['candidate']['dimension']==model.W.shape[1]
    retained = np.linalg.eigvalsh(model.Hproj)
    omitted = np.linalg.eigvalsh(model.complement.conj().T@model.H@model.complement)
    gap = r['spectral_separation']['minimum_separation']
    if len(retained) and len(omitted):
        expected = np.min(abs(retained[:,None]-omitted[None,:]))
        assert gap==pytest.approx(expected,abs=3e-13)
    else:
        assert gap is None
    assert r['candidate']['hamiltonian_closure_norm']<3e-13
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('L,N',CASES)
def test_density_span_and_dynamical_commutant_independent(L,N):
    model = q.projected_model(L,N)
    r = q.matching_diagnostics(L,N)
    expected = real_matrix_span(model.projected_densities)
    assert r['sources']['direct_real_span_dimension']==expected
    hdim = independent_complex_commutant_dimension([model.Hproj])
    assert r['symmetries']['hamiltonian_commutant_complex_dimension']==hdim
    if N==1:
        assert expected==5
        assert r['sources']['unital_complex_associative_algebra_dimension']==9
        assert hdim==5
    if N==3:
        assert expected==1
    if N>3 or N==0:
        assert expected==0


@pytest.mark.parametrize('L,N',CASES)
def test_symmetry_lift_matches_occupation_after_phase(L,N):
    model = q.projected_model(L,N)
    r = q.matching_diagnostics(L,N)['symmetries']
    assert r['translation_lift_vs_occupation_residual']<3e-13
    assert r['reflection_lift_vs_occupation_phase_corrected_residual']<3e-13
    W = model.W
    literal_R = occupation_permutation(L,N,'reflection')
    if W.shape[1]:
        leakage = np.linalg.norm(literal_R@W-W@(W.conj().T@literal_R@W),2)
    else:
        leakage = 0.
    assert r['many_body_reflection_leakage']==pytest.approx(leakage,abs=3e-13)
    if N==2 and L>=5:
        assert r['one_body_window_reflection_leakage']>.9
        assert leakage>.9
    if N%2:
        assert leakage<3e-13


@pytest.mark.parametrize('L',[5,6,7])
def test_literal_vector_lift_vs_bilinear_identity(L):
    r = q.intertwiner_diagnostics(L)
    a = r['lifts']['proper_half_turn']
    b = r['lifts']['alternate_parity_twisted']
    assert a['solution_complex_dimension']==1
    assert a['maximum_rank']==2
    assert b['solution_complex_dimension']==2
    assert b['maximum_rank']==3
    assert r['bilinear_conjugation_identity_residual']<1e-13
    T = np.diag(np.exp(2j*np.pi*np.array([-1,0,1])/L))
    R = np.fliplr(np.eye(3))
    for target,expected in [(-R,1),(R,2)]:
        constraints = np.vstack((np.kron(np.eye(3),T)-np.kron(T.T,np.eye(3)),
                                 np.kron(np.eye(3),R)-np.kron(target.T,np.eye(3))))
        assert 9-np.linalg.matrix_rank(constraints,tol=1e-10)==expected
    for i in range(3):
        for j in range(3):
            E = np.zeros((3,3)); E[i,j]=1
            np.testing.assert_array_equal(R@E@R,(-R)@E@(-R))
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('L',[5,6,7])
@pytest.mark.parametrize('N',[1,2,3])
def test_exact_reviewed_gap_leakage_and_complement_identities(L,N):
    model = q.projected_model(L,N)
    r = q.matching_diagnostics(L,N)
    expected_gap = 0. if N==2 else 2*(np.cos(2*np.pi/L)-np.cos(4*np.pi/L))
    assert r['spectral_separation']['minimum_separation']==pytest.approx(expected_gap,abs=3e-13)
    p = 3/L
    for B,K in zip(model.projected_densities,model.density_leakages):
        np.testing.assert_allclose(K.conj().T@K,B-B@B,atol=3e-14)
        assert np.linalg.norm(K,2)==pytest.approx(np.sqrt(p*(1-p)),abs=3e-14)
    R = occupation_permutation(L,N,'reflection')
    B = model.W.conj().T@R@model.W
    if N==2:
        np.testing.assert_allclose(B,np.diag([1,0,0]),atol=3e-14)
        hodge = np.array([[0,0,1],[0,-1,0],[1,0,0]])
        for x,B in enumerate(model.projected_densities):
            A = np.outer(model.U[x].conj(),model.U[x])
            np.testing.assert_allclose(hodge@B@hodge.T,p*np.eye(3)-A.T,atol=3e-14)
        assert r['sources']['direct_real_span_dimension']==5
        assert r['sources']['unital_complex_associative_algebra_dimension']==9
    if N==3:
        np.testing.assert_allclose(B,[[1]],atol=3e-14)
    assert r['symmetries']['discrete_compression_commutant_complex_dimension']=={1:2,2:3,3:1}[N]


@pytest.mark.parametrize('scale',[1e-80,1e80])
@pytest.mark.parametrize('N',[1,2,3])
def test_common_scale_preserves_dimensions_and_separation(scale,N):
    a = q.matching_diagnostics(5,N)
    b = q.matching_diagnostics(5,N,scale)
    assert a['sources']['direct_real_span_dimension']==b['sources']['direct_real_span_dimension']
    assert a['symmetries']['hamiltonian_commutant_complex_dimension']==b['symmetries']['hamiltonian_commutant_complex_dimension']
    assert b['spectral_separation']['minimum_separation']/scale==pytest.approx(a['spectral_separation']['minimum_separation'],abs=3e-13)
    json.dumps(b,allow_nan=False)


@pytest.mark.parametrize('L,N',[(9,1),(2,1),(True,1),(5,True),(5,6),(5,-1)])
def test_preallocation_and_sector_rejection(L,N):
    with pytest.raises((ValueError,TypeError)):
        q.projected_model(L,N)


@pytest.mark.parametrize('L,N',[(3,0),(3,1),(3,3),(5,0),(5,1),(5,5)])
def test_neutrality_distinguishes_vacuum_from_unit_filling(L,N):
    r = q.matching_diagnostics(L,N)['neutrality']
    neutral = N%L==0
    d = len(list(itertools.combinations(range(3),N)))
    assert r['is_neutral']==neutral
    assert r['is_unit_filling']==(N==L)
    assert r['number_charge']==N
    assert r['modular_charge']==N%L
    assert r['charge_deviation_from_unit_filling']==N-L
    assert r['neutral_candidate_dimension']==(d if neutral else 0)
    assert r['neutral_hard_core_dimension']==(len(occupation_basis(L,N)) if neutral else 0)
    assert r['unit_filling_candidate_dimension']==int(L==3)
    assert r['unit_filling_hard_core_dimension']==1


@pytest.mark.parametrize('L,N',[(np.int64(5),1),(5,np.int64(1)),(np.int64(5),np.int64(1))])
def test_numpy_integer_inputs_remain_strict_json(L,N):
    r = q.matching_diagnostics(L,N)
    json.dumps(r,allow_nan=False)
    json.dumps(q.intertwiner_diagnostics(L),allow_nan=False)


@pytest.mark.parametrize('N',[1,2])
def test_complete_three_site_window_has_commuting_density_algebra(N):
    model = q.projected_model(3,N)
    matrices = model.projected_densities
    for A in matrices:
        for B in matrices:
            np.testing.assert_allclose(A@B,B@A,atol=2e-14)
    assert q.source_diagnostics(3,N)['unital_complex_associative_algebra_dimension']==3


@pytest.mark.parametrize('scale',[1e-100,1e100])
def test_commuting_algebra_survives_generator_rescaling(scale):
    matrices = q.projected_model(3,1).projected_densities
    assert q._algebra_dimension(tuple(scale*A for A in matrices),3)==3
    assert q._algebra_dimension(tuple(10.**(-40*x)*A for x,A in enumerate(matrices)),3)==3


@pytest.mark.parametrize('api',[q.exterior_power,q.exterior_lift])
@pytest.mark.parametrize('nested',[False,True])
def test_exterior_cap_precedes_complex_conversion(monkeypatch,api,nested):
    oversized = np.zeros((9,9))
    if nested:
        oversized = oversized.tolist()
    original = np.asarray
    def guarded(value,*args,**kwargs):
        assert value is not oversized,'oversized input converted before dimension guard'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(np,'asarray',guarded)
    with pytest.raises((ValueError,TypeError)):
        api(oversized,1)


@pytest.mark.parametrize('C',[0.,-1.,np.nan,np.inf,1e-320,True])
def test_invalid_energy(C):
    with pytest.raises((ValueError,TypeError)):
        q.projected_model(5,2,C)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_stdout_demo(tmp_path,flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_substrate_quantum_matching.py'
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    p = subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert p.returncode==0,p.stderr
    assert not p.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(p.stdout,parse_constant=lambda v:pytest.fail(v))
    else:
        assert 'matching' in p.stdout.lower()
