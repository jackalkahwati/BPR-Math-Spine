"""Independent conditional-model checks; synthetic data are not experiments."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from bpr import phason_response as p


BETA = 761 / 4200
TAU = 20 / 21


def test_symbolic_elimination_and_discriminant():
    C,K,D,rho,Gamma,q,omega = sp.symbols('C K D rho Gamma q omega', real=True)
    matrix = sp.Matrix([[C*q**2-rho*omega**2,D*q**2],[D*q**2,K*q**2-sp.I*Gamma*omega]])
    assert sp.expand(matrix.det()-((C*q**2-rho*omega**2)*(K*q**2-sp.I*Gamma*omega)-D**2*q**4)) == 0
    beta,u,r = sp.symbols('beta u r',real=True)
    polynomial = u*r**3+r**2+u*r+1-beta
    disc = sp.discriminant(polynomial,r)
    expected = -4*u**4+(-8+36*beta-27*beta**2)*u**2-4*(1-beta)
    assert sp.expand(disc-expected)==0
    A = -8+36*beta-27*beta**2
    assert sp.factor(A**2-64*(1-beta))==beta*(9*beta-8)**3
    assert sp.simplify(polynomial.subs({beta:sp.Rational(8,9),u:1/sp.sqrt(3)})-(r+1/sp.sqrt(3))**3/sp.sqrt(3))==0


@pytest.mark.parametrize('omega',[.1,1.,10.])
def test_causal_time_kernel_matches_frequency_response(omega):
    from scipy.integrate import quad
    rate=1/TAU
    # Retarded relaxation kernel is -beta*rate*exp(-rate*t), supported t>=0.
    real=quad(lambda t:BETA*rate*np.exp(-rate*t)*np.cos(omega*t),0,np.inf,epsabs=1e-11)[0]
    imag=quad(lambda t:BETA*rate*np.exp(-rate*t)*np.sin(omega*t),0,np.inf,epsabs=1e-11)[0]
    response=p.normalized_response(BETA,omega*TAU)
    np.testing.assert_allclose([response['X'],response['Y']],[real,imag],atol=2e-10)


def test_energy_balance_and_direct_phason_frequency_equation():
    U,V,W,C,K,D,rho,Gamma,q = sp.symbols('U V W C K D rho Gamma q',real=True)
    energy = rho*V**2/2+q**2*(C*U**2/2+K*W**2/2+D*U*W)
    acceleration = -q**2*(C*U+D*W)/rho
    Wdot = -q**2*(K*W+D*U)/Gamma
    derivative = sp.diff(energy,U)*V+sp.diff(energy,V)*acceleration+sp.diff(energy,W)*Wdot
    assert sp.simplify(derivative+Gamma*Wdot**2)==0
    for omega in [.1,1.,10.]:
        D = np.sqrt(BETA)
        phason_amplitude = -D/(1-1j*omega*TAU)
        assert abs(-1j*omega*TAU*phason_amplitude+phason_amplitude+D)<1e-14
        physical = p.constitutive_response(1.,1.,D,1.,TAU,1.,omega)
        modulus = 1+D*phason_amplitude
        np.testing.assert_allclose([physical['X'],physical['Y']],[1-modulus.real,-modulus.imag],atol=2e-15)


def test_exact_rational_resonance_oracle():
    r = sp.symbols('r')
    b,t = sp.Rational(761,4200),sp.Rational(20,21)
    polynomial = t*r**3+r**2+t*r+1-b
    factor = t*(r+sp.Rational(19,20))*((r+sp.Rational(1,20))**2+sp.Rational(19,20)**2)
    assert sp.expand(polynomial-factor)==0
    z = sp.Rational(19,20)-sp.I/20
    assert sp.simplify((1-z**2)*(1-sp.I*t*z)-b)==0


@pytest.mark.parametrize('beta,u',itertools.product([0.,.001,.02,.1,.25],[0.,1e-4,1.,1e4]))
def test_constitutive_circle_and_passivity(beta,u):
    r = p.normalized_response(beta,u)
    expected = 1-beta/(1-1j*u)
    assert r['X']==pytest.approx(1-expected.real,abs=2e-16)
    assert r['Y']==pytest.approx(-expected.imag,abs=2e-16)
    assert r['X']**2+r['Y']**2==pytest.approx(beta*r['X'],abs=2e-17)
    assert r['Y']>=0 and 1-r['X']>=1-beta
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('beta,u',itertools.product([0.,.001,.02,.1,.25],[.01,.1,1.,10.,100.]))
def test_pole_bounds_against_independent_real_cubic(beta,u):
    report = p.pole_report(beta,u)
    z = complex(*report['positive_pole'])
    roots = 1j*np.roots([u,1.,u,1-beta])
    oracle = roots[np.argmax(roots.real)]
    assert z==pytest.approx(oracle,rel=2e-10,abs=2e-12)
    assert abs((1-z*z)*(1-1j*u*z)-beta)<1e-10
    assert z.real>0 and z.imag<=0
    if beta:
        assert abs(z-1)<beta+2e-12
        assert z.imag<0
    else:
        assert z==1 and report['Q_inverse']==0
    assert abs(z-1+beta/(2*(1-1j*u)))<=13*beta**2/8+2e-12
    qinv = -2*z.imag/z.real
    assert report['Q_inverse']==pytest.approx(qinv,abs=2e-12)
    assert abs(qinv-beta*u/(1+u*u))<=5*beta**2+2e-12
    assert report['bounds']['pole_error_bound']==pytest.approx(13*beta**2/8,abs=1e-20)
    assert report['bounds']['Q_error_bound']==pytest.approx(5*beta**2,abs=1e-20)
    assert report['ancestry_available']
    json.dumps(report,allow_nan=False)


def test_pole_circle_and_constitutive_loss_tangent_are_not_exactly_same():
    r=p.pole_report(BETA,TAU)
    z=complex(*r['positive_pole'])
    x=-2*(z.real-1)
    y=r['Q_inverse']
    assert abs(x*x+y*y-BETA*x)>1e-4
    response=p.normalized_response(BETA,TAU)
    loss_tangent=response['Y']/(1-response['X'])
    assert abs(loss_tangent-y)>1e-4


def test_collision_and_overdamped_boundaries():
    beta=.95
    A=-8+36*beta-27*beta**2
    width=np.sqrt(beta*(9*beta-8)**3)
    lower,upper=(A-width)/8,(A+width)/8
    for u in [np.sqrt(lower),np.sqrt(upper),np.sqrt((lower+upper)/2)]:
        report=p.pole_report(beta,u)
        assert report['positive_pole'] is None
        assert report['Q_inverse'] is None
        json.dumps(report,allow_nan=False)
    middle=p.pole_report(beta,np.sqrt((lower+upper)/2))
    assert middle['overdamped'] and not middle['ancestry_available']
    report=p.pole_report(8/9,1/np.sqrt(3))
    assert report['collision'] or report['near_collision']
    assert report['positive_pole'] is None
    assert report['Q_inverse'] is None
    # Preserve all initial oscillatory controls; no claim of continuation through collisions.
    for u in [.2,.4,1.]:
        report=p.pole_report(beta,u)
        if report['positive_pole'] is not None:
            z=complex(*report['positive_pole'])
            assert z.real>0 and z.imag<0
            assert abs((1-z*z)*(1-1j*u*z)-beta)<1e-10


@pytest.mark.parametrize('beta,u0',[(.1,1e7),(1e-11,1.)])
def test_numeric_unavailability_preserves_analytic_theorem_status(beta,u0):
    report=p.pole_report(beta,u0)
    assert not report['numerical_available']
    assert report['positive_pole'] is None
    assert report['Q_inverse'] is None
    assert report['bounds'] is not None
    assert report['bounds']['certifies_roundoff'] is False
    json.dumps(report,allow_nan=False)


def test_static_high_frequency_and_noncommuting_limits():
    beta=.1
    static=p.normalized_response(beta,0.)
    assert static['X']==beta and static['Y']==0
    high=p.normalized_response(beta,1e4)
    assert high['X']<1.1e-9 and high['Y']<1.1e-5
    # q -> 0 at fixed Omega is the high-u, not static, limit.
    small_q=p.constitutive_response(1.,1.,np.sqrt(beta),1.,1.,.01,1.)
    assert small_q['X']==pytest.approx(high['X'])
    assert small_q['Y']==pytest.approx(high['Y'])


def test_synthetic_training_and_heldout_constitutive_predictions():
    X=Y=761/8400
    fitted = p.constitutive_inverse(X,Y,1.,21/20)
    assert fitted['beta']==pytest.approx(BETA)
    assert fitted['tau']==pytest.approx(TAU)
    for q,omega,expected in [(1.,21/10,(761/21000,761/10500)),(2.,21/5,(X,Y))]:
        r = p.normalized_response(fitted['beta'],omega*fitted['tau']/q**2)
        np.testing.assert_allclose([r['X'],r['Y']],expected,rtol=2e-14,atol=1e-15)


def test_exact_pole_inverse_and_linked_wave_number():
    z = .95-.05j
    fitted = p.pole_inverse(z,1.,1.)
    assert fitted['beta']==pytest.approx(BETA,rel=2e-13)
    assert fitted['tau']==pytest.approx(TAU,rel=2e-13)
    r = p.pole_report(BETA,TAU)
    assert complex(*r['positive_pole'])==pytest.approx(z,abs=2e-13)
    assert r['Q_inverse']==pytest.approx(2/19)
    # At fixed material parameters omega0=q, u0=tau/q, not an independent drive.
    for q in [1.,2.]:
        r = p.resonance_report(1.,1.,np.sqrt(BETA),1.,TAU,q)
        dimless = p.pole_report(BETA,TAU/q)
        assert r['omega0']==pytest.approx(q)
        assert r['u0']==pytest.approx(TAU/q)
        assert complex(*r['positive_pole'])==pytest.approx(complex(*dimless['positive_pole']))


@pytest.mark.parametrize('scale',[.25,4.])
@pytest.mark.parametrize('sign',[-1,1])
def test_physical_sign_and_scale_degeneracy(scale,sign):
    base = p.constitutive_response(1.,1.,np.sqrt(BETA),1.,TAU,1.,21/20)
    other = p.constitutive_response(1.,scale,sign*np.sqrt(scale*BETA),1.,scale*TAU,1.,21/20)
    for name in ['X','Y']:
        assert other[name]==pytest.approx(base[name],rel=2e-14)


def test_inverse_jacobians_from_independent_symbolic_differentiation():
    X,Y,q,w = sp.symbols('X Y q w',positive=True)
    inverse = sp.Matrix([X+Y**2/X,q**2*Y/(w*X)])
    oracle = np.array(inverse.jacobian([X,Y]).subs({X:761/8400,Y:761/8400,q:1,w:21/20})).astype(float)
    J = np.asarray(p.inverse_jacobian(761/8400,761/8400,1.,21/20))
    np.testing.assert_allclose(J,oracle,rtol=2e-14,atol=1e-14)
    beta,tau = sp.symbols('beta tau',positive=True)
    u=w*tau/q**2
    response=sp.Matrix([beta/(1+u**2),beta*u/(1+u**2)])
    oracle = np.array(response.jacobian([beta,tau]).subs({beta:BETA,tau:TAU,q:1,w:21/10})).astype(float)
    J = np.asarray(p.prediction_jacobian(BETA,TAU,1.,21/10))
    np.testing.assert_allclose(J,oracle,rtol=2e-14,atol=1e-14)


def test_covariance_propagation_and_training_roundtrip():
    sigma = np.diag([1e-10,4e-10])
    J = np.asarray(p.inverse_jacobian(761/8400,761/8400,1.,21/20))
    result = np.asarray(p.propagate_covariance(J,sigma))
    np.testing.assert_allclose(result,J@sigma@J.T,rtol=2e-14,atol=1e-22)
    prediction = np.asarray(p.prediction_jacobian(BETA,TAU,1.,21/20))
    np.testing.assert_allclose(prediction@J,np.eye(2),atol=2e-14)
    np.testing.assert_allclose(p.propagate_covariance(prediction,result),sigma,atol=1e-22)
    assert np.linalg.eigvalsh(result).min()>=0


def test_unknown_baseline_leaves_one_complex_pole_nonidentifiable():
    # The measured physical pole is fixed; alternative baseline frequencies
    # imply different allowed material combinations through the exact inverse.
    measured=.95-.05j
    alternatives=[]
    for omega0 in [1.,1.02]:
        fitted=p.pole_inverse(measured/omega0,1.,omega0)
        alternatives.append(fitted)
        z=measured/omega0
        u0=fitted['tau']*omega0
        assert abs((1-z*z)*(1-1j*u0*z)-fitted['beta'])<2e-14
    assert abs(alternatives[0]['beta']-alternatives[1]['beta'])>.001
    assert abs(alternatives[0]['tau']-alternatives[1]['tau'])>.001


def test_unknown_participation_confounds_beta_and_background_biases_ratio():
    X=Y=761/8400
    r = p.constitutive_inverse(.4*X,.4*Y,1.,21/20)
    assert r['beta']==pytest.approx(.4*BETA)
    assert r['tau']==pytest.approx(TAU)
    biased = p.constitutive_inverse(X,Y+.01,1.,21/20)
    assert abs(biased['tau']-TAU)>.01


def test_two_channel_training_cannot_predict_frozen_heldout_response():
    def mixture(w):
        value = .1/(1-.5j*w)+.05/(1-2j*w)
        return value.real,value.imag
    X,Y=mixture(1.)
    fitted=p.constitutive_inverse(X,Y,1.,1.)
    assert fitted['beta']==pytest.approx(13/100)
    assert fitted['tau']==pytest.approx(2/3)
    prediction=p.normalized_response(fitted['beta'],2*fitted['tau'])
    actual=np.array(mixture(2.))
    residual=actual-[prediction['X'],prediction['Y']]
    np.testing.assert_allclose(residual,np.array([261.,-27.])/42500,atol=1e-16)
    assert np.linalg.norm(residual)>1e-4


@pytest.mark.parametrize('beta',[-.1,1.,1.1,np.nan,np.inf,True,1e-320])
def test_invalid_coupling_domain(beta):
    with pytest.raises((ValueError,TypeError)):
        p.normalized_response(beta,1.)
    with pytest.raises((ValueError,TypeError)):
        p.pole_report(beta,1.)


@pytest.mark.parametrize('u',[-1.,np.nan,np.inf,True,1e-320])
def test_invalid_frequency_domain(u):
    with pytest.raises((ValueError,TypeError)):
        p.normalized_response(.1,u)


def test_uniform_zero_wave_number_is_explicitly_degenerate():
    for api,args in [(p.constitutive_response,(1.,1.,.1,1.,1.,0.,1.)),(p.resonance_report,(1.,1.,.1,1.,1.,0.)),(p.constitutive_inverse,(.1,.1,0.,1.)),(p.pole_inverse,(.95-.05j,0.,1.))]:
        with pytest.raises(ValueError):
            api(*args)


def test_inverse_zero_coupling_and_invalid_physical_domains():
    with pytest.raises((ValueError,TypeError)):
        p.constitutive_inverse(0.,0.,1.,1.)
    with pytest.raises((ValueError,TypeError)):
        p.pole_inverse(1.+0j,1.,1.)
    with pytest.raises((ValueError,TypeError)):
        p.pole_inverse(-.1j,1.,1.)
    with pytest.raises((ValueError,TypeError)):
        p.constitutive_response(1.,1.,.1,1.,-1.,1.,1.)
    with pytest.raises((ValueError,TypeError)):
        p.constitutive_response(1.,1.,1.,1.,1.,1.,1.)


@pytest.mark.parametrize('covariance',[[[1.,2.],[2.,1.]],[[1.,0.],[1.,1.]],[[np.nan,0.],[0.,1.]]])
def test_invalid_covariance(covariance):
    with pytest.raises((ValueError,TypeError)):
        p.propagate_covariance(np.eye(2),covariance)


@pytest.mark.parametrize('scale',[2.,7.,8.,10.])
def test_exact_stability_boundary_cannot_round_into_stable_domain(scale):
    for api,args in [(p.constitutive_response,(scale,scale,scale,1.,1.,1.,1.)),(p.resonance_report,(scale,scale,scale,1.,1.,1.))]:
        with pytest.raises(ValueError):
            api(*args)


def test_near_cusp_cannot_report_inverted_collision_interval():
    beta=np.nextafter(8/9,1.)
    try:
        report=p.pole_report(beta,1.)
    except p.NumericalUnavailable:
        return
    interval=report['collision_u0_interval']
    if interval is None:
        assert not report['numerical_available'] and report['reason']
    else:
        assert interval[0]<=interval[1]
        if interval[0]==interval[1]:
            assert not report['numerical_available']


def test_leading_bound_cannot_emit_subnormal_complex_component():
    with pytest.raises(p.NumericalUnavailable):
        p.pole_report(.1,3e-307)


def test_tiny_exact_real_cannot_silently_become_zero_coupling():
    from fractions import Fraction
    with pytest.raises((ValueError,TypeError)):
        p.normalized_response(Fraction(1,10**400),1.)


def test_covariance_cancellation_retains_normal_variance_or_rejects():
    from fractions import Fraction
    neighbor=np.nextafter(1.,2.)
    J=np.array([[1.,-neighbor],[0.,1.]])
    try:
        result=p.propagate_covariance(J,np.ones((2,2)))
    except p.NumericalUnavailable:
        return
    delta=Fraction(1)-Fraction.from_float(float(neighbor))
    expected=np.array([[float(delta**2),float(delta)],[float(delta),1.]])
    np.testing.assert_allclose(result,expected,rtol=1e-14,atol=0.)
    assert result[0,0]>0
    assert Fraction.from_float(float(result[0,0]))*Fraction.from_float(float(result[1,1]))>=Fraction.from_float(float(result[0,1]))**2


def test_covariance_zero_and_rank_one_are_valid_controls():
    J=np.array([[1.,2.],[3.,4.]])
    for sigma in [np.zeros((2,2)),np.ones((2,2))]:
        np.testing.assert_allclose(p.propagate_covariance(J,sigma),J@sigma@J.T,atol=1e-14)


def test_covariance_shape_cap_precedes_conversion(monkeypatch):
    oversized=np.eye(33)
    original=np.asarray
    def checked(value,*args,**kwargs):
        assert value is not oversized,'converted before shape cap'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(np,'asarray',checked)
    with pytest.raises((ValueError,TypeError)):
        p.propagate_covariance(np.eye(2),oversized)


@pytest.mark.parametrize('flags',[[],['--json']])
def test_demo_is_stdout_only_and_strict_json(tmp_path,flags):
    script=Path(__file__).resolve().parents[1]/'scripts/demo_phason_response.py'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    result=subprocess.run([sys.executable,str(script)]+flags,cwd=tmp_path,env=env,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stderr
    assert not result.stderr
    assert not list(tmp_path.iterdir())
    if flags:
        json.loads(result.stdout,parse_constant=lambda value:pytest.fail(value))
    else:
        assert 'phason' in result.stdout.lower()


def test_new_files_support_python38_grammar():
    import ast
    root=Path(__file__).resolve().parents[1]
    for name in ['bpr/phason_response.py','scripts/demo_phason_response.py']:
        ast.parse((root/name).read_text(),feature_version=(3,8))
