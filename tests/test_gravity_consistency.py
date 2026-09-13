"""Independent action, dimensional and identifiability checks; no data fit."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from bpr import gravity_consistency as g


HBAR = 1.054571817e-34
C = 299792458.


def test_jordan_auxiliary_elimination_and_einstein_curvature():
    M,alpha,chi,R,phi = sp.symbols('M alpha chi R phi',positive=True)
    jordan = (M**2/2+alpha*chi)*R-alpha*chi**2/2
    assert sp.simplify(sp.diff(jordan,chi)-alpha*(R-chi))==0
    F = sp.exp(sp.sqrt(sp.Rational(2,3))*phi/M)
    chi_E = M**2*(F-1)/(2*alpha)
    V = sp.simplify(alpha*chi_E**2/(2*F**2))
    expected = M**4/(8*alpha)*(1-1/F)**2
    assert sp.simplify(V-expected)==0
    assert sp.simplify(sp.diff(V,phi,2).subs(phi,0)-M**2/(6*alpha))==0
    N = sp.symbols('N',positive=True)
    epsilon = sp.Rational(3,4)/N**2
    amplitude = (M**4/(8*alpha))/(24*sp.pi**2*M**4*epsilon)
    assert sp.simplify(amplitude-N**2/(144*sp.pi**2*alpha))==0


@pytest.mark.parametrize('M,alpha',itertools.product([1.,2.],[.5,2.,10.]))
@pytest.mark.parametrize('ratio',[0.,.1,1.,5.])
def test_potential_mass_plateau(M,alpha,ratio):
    expected = M**4/(8*alpha)*np.expm1(-np.sqrt(2/3)*ratio)**2
    assert g.einstein_potential(M*ratio,M,alpha)==pytest.approx(expected)
    assert g.scalaron_mass_squared(M,alpha)==pytest.approx(M**2/(6*alpha))
    assert g.scalaron_mass(M,alpha)**2==pytest.approx(M**2/(6*alpha))
    assert g.plateau_energy_density(M,alpha)==pytest.approx(M**4/(8*alpha))


@pytest.mark.parametrize('ratio',[1e-12,-1e-12])
def test_stable_quadratic_small_field(ratio):
    V = g.einstein_potential(ratio,1.,2.)
    assert V/ratio**2==pytest.approx(1/24,rel=2e-12)


@pytest.mark.parametrize('N,alpha',itertools.product([40.,55.,70.],[.5,2.,10.]))
def test_leading_scalar_amplitude_from_plateau_and_epsilon(N,alpha):
    expected = (1/(8*alpha))/(24*np.pi**2*(3/(4*N**2)))
    assert g.leading_scalar_amplitude(N,alpha)==pytest.approx(expected)
    assert g.alpha_from_scalar_amplitude(N,expected)==pytest.approx(alpha)
    json.dumps(g.leading_slow_roll(N,alpha),allow_nan=False)


@pytest.mark.parametrize('N,amplitude',itertools.product([40.,55.,70.],[1e-3,1e-6]))
def test_inverse_is_synthetic_calibration_roundtrip(N,amplitude):
    alpha = g.alpha_from_scalar_amplitude(N,amplitude)
    assert alpha==pytest.approx(N**2/(144*np.pi**2*amplitude))
    assert g.leading_scalar_amplitude(N,alpha)==pytest.approx(amplitude)


@pytest.mark.parametrize('energy',[1.,2.,10.])
def test_reduced_unreduced_newton_energy_roundtrip(energy):
    unreduced = g.reduced_to_unreduced_energy(energy)
    assert unreduced==pytest.approx(np.sqrt(8*np.pi)*energy)
    assert g.unreduced_to_reduced_energy(unreduced)==pytest.approx(energy)
    G = g.newton_constant_from_reduced_energy(energy)
    assert G==pytest.approx(HBAR*C**5/(8*np.pi*energy**2),rel=2e-9)
    assert g.reduced_energy_from_newton_constant(G)==pytest.approx(energy)
    assert g.natural_newton_constant(energy)==pytest.approx(1/(8*np.pi*energy**2))


@pytest.mark.parametrize('p,cutoff',itertools.product([5,11],[1.,2.,10.]))
def test_induced_coefficient_spacing_and_wald_identity(p,cutoff):
    M2 = p*cutoff**2/(48*np.pi**2)
    assert g.induced_reduced_coefficient(p,cutoff)==pytest.approx(M2)
    assert g.induced_spacing_planck_ratio(p)==pytest.approx(np.sqrt(p/(6*np.pi)))
    a = g.cutoff_spacing_si(cutoff)
    assert a==pytest.approx(HBAR*C/cutoff,rel=2e-9,abs=0.)
    G = g.newton_constant_from_reduced_energy(np.sqrt(M2))
    lp2 = HBAR*G/C**3
    assert a/np.sqrt(lp2)==pytest.approx(np.sqrt(p/(6*np.pi)),rel=2e-9)
    area = 2.
    assert g.wald_entropy_natural(area,np.sqrt(M2))==pytest.approx(2*np.pi*area*M2)
    assert g.wald_entropy_si(area,np.sqrt(M2))==pytest.approx(area/(4*lp2),rel=2e-9)


@pytest.mark.parametrize('factor',[.25,4.])
@pytest.mark.parametrize('k',[.5,1.,3.])
def test_constant_normalization_is_degenerate_with_G(factor,k):
    Z,G = 2.,3.
    a = g.propagator_amplitude(Z,G,k)
    assert a==pytest.approx(Z*G/k**2)
    assert g.propagator_amplitude(Z*factor,G/factor,k)==pytest.approx(a)
    json.dumps(g.propagator_rescaling(Z,G,factor),allow_nan=False)


@pytest.mark.parametrize('shift',[-.25,.25])
def test_induced_bare_counterterm_null_direction(shift):
    b,c,p,cutoff = 2.,-.5,5,3.
    value = g.induced_effective_coefficient(b,c,p,cutoff)
    assert value==pytest.approx(b+c+p*cutoff**2/(48*np.pi**2))
    assert g.induced_effective_coefficient(b+shift,c-shift,p,cutoff)==pytest.approx(value)


@pytest.mark.parametrize('new_cutoff',[2.,4.])
def test_induced_cutoff_compensation(new_cutoff):
    b,c,p,cutoff = 2.,-.5,5,3.
    k = p/(48*np.pi**2)
    bprime = b+k*(cutoff**2-new_cutoff**2)
    assert g.induced_effective_coefficient(bprime,c,p,new_cutoff)==pytest.approx(g.induced_effective_coefficient(b,c,p,cutoff))
    J = np.array([1.,1.,2*k*cutoff])
    nulls = np.array([[1.,-1.,0.],[-2*k*cutoff,0.,1.]])
    np.testing.assert_allclose(J@nulls.T,0.,atol=1e-15)


def test_reported_identifiability_nulls_and_exact_transformations():
    b,c,p,cutoff = 2.,-.5,5,3.
    r = g.induced_identifiability_report(b,c,p,cutoff)
    J = np.array([1.,1.,2*p*cutoff/(48*np.pi**2)])
    np.testing.assert_allclose(r['jacobian'],J,atol=1e-14)
    np.testing.assert_allclose(np.asarray(r['null_directions'])@J,0.,atol=1e-14)
    assert r['M_eff_squared']==pytest.approx(b+c+p*cutoff**2/(48*np.pi**2))
    for shift in [-.25,.25]:
        bp,cp = g.induced_bare_counterterm_shift(b,c,p,cutoff,shift)
        assert bp==pytest.approx(b+shift)
        assert cp==pytest.approx(c-shift)
    for target in [2.,4.]:
        bp,cp,lp = g.induced_cutoff_shift(b,c,p,cutoff,target)
        assert bp==pytest.approx(b+p*(cutoff**2-target**2)/(48*np.pi**2))
        assert cp==c and lp==target
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('offset',[-3.,4.])
def test_flat_identity_shift_against_independent_exponential(offset):
    from scipy.linalg import expm
    H = np.array([[1.,.2],[.2,2.]])
    A = np.array([[0,1j],[-1j,0]])
    times = [0.,.3,1.]
    r = g.flat_identity_shift_report(H,A,offset,times)
    np.testing.assert_allclose(r['energies'],np.linalg.eigvalsh(H),atol=1e-14)
    np.testing.assert_allclose(r['shifted_energies'],np.linalg.eigvalsh(H+offset*np.eye(2)),atol=1e-14)
    np.testing.assert_allclose(r['pairwise_gaps'],r['shifted_pairwise_gaps'],atol=1e-13)
    assert r['source_commutator_norm']>0
    for t,event in zip(times,r['dynamics']):
        U = expm(-1j*t*H)
        V = expm(-1j*t*(H+offset*np.eye(2)))
        np.testing.assert_allclose(V,np.exp(-1j*t*offset)*U,atol=2e-14)
        np.testing.assert_allclose(V@A@V.conj().T,U@A@U.conj().T,atol=2e-14)
        assert event['conjugation_residual']<1e-12
        assert event['state_phase_operator_residual']<1e-12
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('offset',[1e20,np.inf,1j,True])
def test_flat_invalid_or_unresolved_offset(offset):
    with pytest.raises((ValueError,TypeError)):
        g.flat_identity_shift_report([[1.,.2],[.2,2.]],[[0.,1.],[1.,0.]],offset)


def test_flat_matrix_cap_before_conversion(monkeypatch):
    H = np.eye(33)
    source = H.copy()
    original = np.asarray
    def checked(value,*args,**kwargs):
        assert value is not H,'allocated before cap validation'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(np,'asarray',checked)
    with pytest.raises((ValueError,TypeError)):
        g.flat_identity_shift_report(H,source,1.)


def test_induced_cancellation_rejects_unresolved_physical_coefficient():
    cutoff = np.sqrt(48*np.pi**2*1e16)
    with pytest.raises(ValueError):
        g.induced_effective_coefficient(-1e16,3.,1,cutoff)
    with pytest.raises(ValueError):
        g.induced_identifiability_report(-1e16,3.,1,cutoff)


@pytest.mark.parametrize('transform,value',[(g.induced_bare_counterterm_shift,1e20),(g.induced_cutoff_shift,1e9)])
def test_finite_transform_cannot_silently_change_coefficient(transform,value):
    with pytest.raises(ValueError):
        transform(2.,-.5,5,3.,value)


def test_flat_original_collapsed_spectrum_is_not_zero_physical_gap():
    H = [[1e16,.2],[.2,1e16]]
    with pytest.raises(ValueError):
        g.flat_identity_shift_report(H,[[1.,0.],[0.,-1.]],-1e16,[.4])


@pytest.mark.parametrize('rho',[-3.,0.,4.])
def test_vacuum_shift_requires_separately_supplied_density(rho):
    assert g.cosmological_constant_shift(rho,2.)==pytest.approx(rho/2)
    with pytest.raises((ValueError,TypeError)):
        g.cosmological_constant_shift(rho,0.)


def test_global_shift_preserves_population_order_but_sector_shift_need_not():
    energies = np.array([1.,2.])
    np.testing.assert_array_equal(np.argsort(energies),np.argsort(energies-4))
    assert np.argmin(energies+np.array([3.,0.]))!=np.argmin(energies)


@pytest.mark.parametrize('p',[5,11,104761])
def test_legacy_induced_helpers_obey_reduced_convention(p):
    from bpr import emergent_spacetime as old
    cutoff = 2.
    reduced = old.planck_mass_from_boundary_cutoff(p,cutoff)
    assert reduced==pytest.approx(cutoff*np.sqrt(p/(48*np.pi**2)))
    assert old.boundary_cutoff_from_planck_mass(p,reduced)==pytest.approx(cutoff)
    assert old.newtons_constant_from_substrate(p,cutoff)==pytest.approx(HBAR*C**5/(8*np.pi*reduced**2),rel=2e-9)
    lp = old.planck_length_from_substrate(p=p)
    assert old.boundary_lattice_spacing(p)/lp==pytest.approx(np.sqrt(p/(6*np.pi)))
    implied_cutoff = HBAR*C/old.boundary_lattice_spacing(p)
    implied_G = old.newtons_constant_from_substrate(p,implied_cutoff)
    assert implied_G==pytest.approx(lp**2*C**3/HBAR,rel=2e-9)


@pytest.mark.parametrize('p',[5,11,104761])
def test_bridge_roundtrip_and_fallback_preserve_physical_anchor(monkeypatch,p):
    from bpr.bridges import cosmology_gravity as bridge
    r = bridge.planck_to_newton(p=p)
    expected = r['l_P_m']**2*C**3/HBAR
    assert r['G_derived_m3_kg_s2']==pytest.approx(expected,rel=2e-9)
    assert r['M_Pl_derived_kg']**2==pytest.approx(HBAR*C/expected,rel=2e-9,abs=0.)
    monkeypatch.setattr(bridge,'newtons_constant_from_substrate',None)
    fallback = bridge.planck_to_newton(p=p)
    assert fallback['G_derived_m3_kg_s2']==pytest.approx(expected,rel=2e-9)
    assert fallback['a_boundary_m']==pytest.approx(r['l_P_m']*np.sqrt(p/(6*np.pi)),abs=0.)


@pytest.mark.parametrize('p',[5,11,104761])
def test_existing_numeric_entropy_unchanged_and_matches_wald(p):
    from bpr import emergent_spacetime as old
    area = 1e-40
    lp = old.planck_length_from_substrate(p=p)
    entropy = old.HolographicEntropy(boundary_area=area,p=p).entropy
    assert entropy==pytest.approx(area/(4*lp**2),rel=2e-7)
    reduced = HBAR*C/(np.sqrt(8*np.pi)*lp)
    assert g.wald_entropy_si(area,reduced)==pytest.approx(entropy,rel=2e-7)


@pytest.mark.parametrize('p',[5,11,104761])
def test_graviton_default_anchor_matches_reduced_energy(p):
    from bpr import graviton_propagator as legacy
    from bpr import emergent_spacetime as space
    model = legacy.BoundaryGravitonPropagator(p=p)
    lp = space.planck_length_from_substrate(p=p)
    # Existing modules retain independently rounded physical length anchors.
    assert model.boundary_spacing_m==pytest.approx(legacy.L_PLANCK*np.sqrt(p/(6*np.pi)),rel=1e-13,abs=0.)
    assert model.boundary_spacing_m/space.boundary_lattice_spacing(p)==pytest.approx(legacy.L_PLANCK/lp,rel=1e-13)
    assert model.planck_energy_J==pytest.approx(HBAR*C/(np.sqrt(8*np.pi)*legacy.L_PLANCK),rel=2e-9)
    explicit = legacy.BoundaryGravitonPropagator(p=p,Lambda_b_J=2.)
    assert explicit.boundary_spacing_m==pytest.approx(HBAR*C/2,abs=0.,rel=2e-9)
    assert explicit.planck_energy_J==pytest.approx(2*np.sqrt(p/(48*np.pi**2)))


def test_pipeline_cutoff_roundtrip_without_production_evolution(monkeypatch):
    from bpr import pipelines
    class State:
        def __init__(self,**kwargs):
            pass
    class Evolution:
        def __init__(self,**kwargs):
            pass
        def evolve(self,state,steps):
            return [state]
    class Dimensions:
        total_dimensions = 4
        spatial_dimensions = 3
        time_dimensions = 1
        def __init__(self,**kwargs):
            pass
    class Inflation:
        spectral_index = .9
        tensor_to_scalar = .1
        def __init__(self,**kwargs):
            pass
    for name in ('_HAS_RPST','_HAS_SPACETIME','_HAS_COSMOLOGY'):
        monkeypatch.setattr(pipelines,name,True)
    for name,value in [('SubstrateState',State),('SymplecticEvolution',Evolution),('EmergentDimensions',Dimensions),('InflationaryParameters',Inflation),('gw_dispersion_correction',lambda frequency:C)]:
        monkeypatch.setattr(pipelines,name,value,raising=False)
    r = pipelines.pipeline_substrate_to_spacetime(p=5,n_sites=3,n_steps=1)
    assert r['G_derived_m3_kg_s2']==pytest.approx(r['G_measured_m3_kg_s2'],rel=2e-9)
    assert r['G_relative_error']<2e-9


@pytest.mark.parametrize('bad',[0.,-1.,np.inf,np.nan,True,1e-320])
def test_positive_domains(bad):
    for api,args in [(g.scalaron_mass_squared,(bad,1.)),(g.scalaron_mass_squared,(1.,bad)),(g.plateau_energy_density,(1.,bad)),(g.leading_scalar_amplitude,(bad,1.)),(g.leading_scalar_amplitude,(40.,bad)),(g.alpha_from_scalar_amplitude,(40.,bad)),(g.newton_constant_from_reduced_energy,(bad,)),(g.reduced_energy_from_newton_constant,(bad,)),(g.propagator_amplitude,(1.,1.,bad))]:
        with pytest.raises((ValueError,TypeError)):
            api(*args)


def test_scalar_multiplicity_has_no_artificial_allocation_cap():
    p = 2**63
    assert g.induced_reduced_coefficient(p,1.)==pytest.approx(p/(48*np.pi**2))
    assert g.induced_spacing_planck_ratio(p)==pytest.approx(np.sqrt(p/(6*np.pi)))


@pytest.mark.parametrize('bad',[0,-1,True,1.5,2**53+1])
def test_induced_multiplicity_validation(bad):
    with pytest.raises((ValueError,TypeError)):
        g.induced_reduced_coefficient(bad,1.)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_stdout_demo(tmp_path,flags):
    script = Path(__file__).resolve().parents[1]/'scripts/demo_gravity_consistency.py'
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    result = subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout,parse_constant=lambda value:pytest.fail(value))
    else:
        assert 'gravity' in result.stdout.lower()
