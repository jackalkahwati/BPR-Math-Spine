"""Conditional flavor model checks; analytic toy identities, no measured targets."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from bpr import chiral_flavor_prototype as f


def assert_close(actual, expected, atol=2e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=2e-12)


@pytest.mark.parametrize("q", [-5, -3, -1, 0, 1, 3, 5])
def test_signed_counts_and_dirac_negative_control(q):
    a = f.signed_mode_bookkeeping(q)
    assert a["internal_index"] == q
    assert a["su_twisted_component_index"] == 3*q
    assert a["net_left_families"] == q
    assert a["net_left_gauge_components"] == 3*q
    assert f.signed_mode_bookkeeping(q, parent_chirality=-1)["net_left_families"] == -q
    d = f.signed_mode_bookkeeping(q, parent="dirac")
    assert d["left_family_copies"] == d["right_family_copies"] == abs(q)
    assert d["net_left_families"] == 0


@pytest.mark.parametrize("q", [1, 2, 3, 6, 11])
@pytest.mark.parametrize("radius", [0.4, 1.0, 3.7])
def test_profiles_gram_radius_and_common_patch(q, radius):
    assert_close(f.overlap_matrix(1, q=q, R=radius), np.eye(q))
    field = lambda x, y, z: 1+x+z*z
    a = f.overlap_matrix(field, q=q, R=radius)
    b = f.overlap_matrix(field, q=q, R=radius, patch_phase=lambda x, y, z: 0.7*x+2*y-z)
    assert_close(a, b)
    assert_close(a, f.overlap_matrix(field, q=q, R=1))
    theta, phi, w = f.sphere_quadrature(radius)
    assert_close(w.sum(), 4*np.pi*radius**2)
    assert_close(f.monopole_profiles(theta, phi, q=q, R=radius)*radius,
                 f.monopole_profiles(theta, phi, q=q))


def test_profile_endpoints_and_north_ordering():
    p = f.monopole_profiles(np.array([0, np.pi]), np.array([0, 0]))
    assert_close(p[0], [np.sqrt(3/(4*np.pi)), 0, 0])
    assert_close(p[1], [0, 0, np.sqrt(3/(4*np.pi))])


@pytest.mark.parametrize("q", [0, -1, -3])
def test_nonpositive_profile_flux_rejected_not_counts(q):
    f.signed_mode_bookkeeping(q)
    with pytest.raises(ValueError):
        f.monopole_profiles(1, 2, q=q)


@pytest.mark.parametrize("q", [True, 3.0, "3"])
def test_flux_type_validation(q):
    with pytest.raises(TypeError):
        f.signed_mode_bookkeeping(q)
    with pytest.raises(TypeError):
        f.monopole_profiles(1, 2, q=q)


@pytest.mark.parametrize("kwargs", [{"R": 0}, {"R": -1}, {"R": np.nan},
    {"mu2": 0}, {"mu2": np.inf}, {"kappa": -1}, {"h0": np.nan}])
def test_scalar_parameter_validation(kwargs):
    for function, args in ((f.scalar_minimizer, (np.zeros(9),)),
                           (f.scalar_action, (np.zeros(9), np.zeros(9))),
                           (f.scalar_stationarity, (np.zeros(9), np.zeros(9)))):
        with pytest.raises(ValueError):
            function(*args, **kwargs)


def test_misc_input_validation():
    for kwargs in ({"gauge_rank": 1}, {"parent": "majorana"}, {"parent_chirality": 0}):
        with pytest.raises(ValueError):
            f.signed_mode_bookkeeping(3, **kwargs)
    for kwargs in ({"n_polar": 0}, {"n_azimuth": 0}, {"R": np.inf}):
        with pytest.raises(ValueError):
            f.sphere_quadrature(**kwargs)
    with pytest.raises(TypeError):
        f.sphere_quadrature(n_polar=2.0)
    for field in (np.zeros(8), np.full(9, np.nan), np.ones(9)*1j):
        with pytest.raises(ValueError):
            f.scalar_minimizer(field)
    with pytest.raises(ValueError):
        f.overlap_matrix(lambda x, y, z: x+1j*y)
    with pytest.raises(ValueError):
        f.monopole_profiles(-1, 0)
    with pytest.raises(ValueError):
        f.monopole_profiles(1, np.nan)
    with pytest.raises(ValueError):
        f.monopole_profiles(1, 0, patch_phase=1j)
    with pytest.raises(ValueError):
        f.mass_matrix(np.eye(3), 0)
    with pytest.raises(ValueError):
        f.svd_sector(np.eye(2))
    with pytest.raises(ValueError):
        f.svd_sector(np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        f.mixing_observables(np.eye(3), np.eye(3), degeneracy_rtol=-1)
    with pytest.raises(ValueError):
        f.sm_anomalies(-1)
    with pytest.raises(TypeError):
        f.sm_anomalies(1, include_neutral_neutrino=1)


def test_real_harmonic_normalization():
    theta, phi, weights = f.sphere_quadrature()
    Y = f.real_harmonics(theta, phi)
    assert Y.shape == (24*48, 9)
    assert_close(Y.T @ (weights[:, None]*Y), np.eye(9))


def test_exact_low_projections_and_linearity():
    assert_close(f.overlap_matrix(1), np.eye(3))
    assert_close(f.overlap_matrix(lambda x, y, z: z), np.diag([0.5, 0, -0.5]))
    assert_close(f.overlap_matrix(lambda x, y, z: (3*z*z-1)/2), np.diag([1, -2, 1])/10)
    a = f.overlap_matrix(lambda x, y, z: x+y*z)
    b = f.overlap_matrix(lambda x, y, z: z+x*y)
    assert_close(a, a.conj().T)
    assert_close(f.overlap_matrix(lambda x, y, z: 2*(x+y*z)-3*(z+x*y)), 2*a-3*b)


def test_nine_real_sources_span_hermitian_matrices():
    matrices = np.array([f.overlap_matrix(f.scalar_minimizer(row, h0=0)) for row in np.eye(9)])
    assert_close(matrices, matrices.conj().transpose(0, 2, 1))
    embedding = np.concatenate((matrices.real.reshape(9, -1), matrices.imag.reshape(9, -1)), axis=1)
    assert np.linalg.matrix_rank(embedding, tol=1e-11) == 9
    # Algebraic span identity, not an optimization or any empirical target fit.
    gram = np.einsum("aij,bij->ab", matrices.conj(), matrices).real
    assert np.linalg.eigvalsh(gram).min() > 0


@pytest.mark.parametrize("harmonic", [
    lambda x, y, z: (5*z**3-3*z)/2,                  # l=3,m=0
    lambda x, y, z: x*y*z,                         # l=3,m=2 real
    lambda x, y, z: x**3-3*x*y*y,                  # l=3,m=3 real
    lambda x, y, z: (35*z**4-30*z*z+3)/8,          # l=4,m=0
    lambda x, y, z: x**4-6*x*x*y*y+y**4,           # l=4,m=4 real
])
def test_callable_higher_harmonics_project_to_zero(harmonic):
    assert_close(f.overlap_matrix(harmonic), np.zeros((3, 3)))
    assert_close(f.source_coefficients(harmonic), np.zeros(9))


def test_quadrature_refinement_and_source_formulas():
    Ju = lambda x, y, z: z+(3*z*z-1)/4
    Jd = lambda x, y, z: x+z/3+x*y/2
    for direct, coeff in zip((Ju, Jd), f.fixed_toy_sources()):
        assert_close(f.source_coefficients(direct), coeff)
        assert_close(f.overlap_matrix(direct, n_polar=8, n_azimuth=16),
                     f.overlap_matrix(coeff, n_polar=32, n_azimuth=64))
    # A nonpolynomial callable checks actual quadrature convergence as well.
    field = lambda x, y, z: np.exp(0.4*x+0.2*y-0.3*z)
    coarse = f.overlap_matrix(field, n_polar=4, n_azimuth=8)
    medium = f.overlap_matrix(field, n_polar=12, n_azimuth=24)
    fine = f.overlap_matrix(field, n_polar=24, n_azimuth=48)
    assert np.linalg.norm(medium-fine) < np.linalg.norm(coarse-fine)/100
    assert_close(medium, fine)


@pytest.mark.parametrize("radius,kappa,mu2", [(1, 1, 1), (2.3, 0.7, 1.4), (0.4, 0, 2)])
def test_action_minimizer_hessian_and_perturbations(radius, kappa, mu2):
    source = np.arange(1, 10)/9
    params = {"R": radius, "kappa": kappa, "mu2": mu2, "h0": 1.7}
    h = f.scalar_minimizer(source, **params)
    assert_close(f.scalar_stationarity(h, source, **params), np.zeros(9))
    H = f.scalar_hessian(radius, kappa, mu2)
    assert np.linalg.eigvalsh(H).min() > 0
    ell = f.HARMONIC_DEGREES
    assert_close(np.diag(H), radius**2*mu2+kappa*ell*(ell+1))
    for delta in (np.arange(9)/90, np.ones(9)*0.1, -np.arange(9)/9):
        increase = f.scalar_action(h+delta, source, **params)-f.scalar_action(h, source, **params)
        assert increase > 0
        assert_close(increase, delta @ H @ delta / 2)
        assert_close(f.scalar_stationarity(h+delta, source, **params), H@delta)
    # Explicit quadrature of potential/source and exact sphere gradient norm.
    theta, phi, area = f.sphere_quadrature(radius)
    basis = f.real_harmonics(theta, phi)
    values, source_values = basis@h, basis@source
    direct = np.dot(area, mu2*(values-1.7)**2/2-source_values*values)
    direct += kappa/2 * np.dot(ell*(ell+1)*h, h)
    assert_close(f.scalar_action(h, source, **params), direct)
    assert f.scalar_action(h, source, **params) < 0  # Convex does not mean positive.


def test_action_matches_direct_angular_gradient_quadrature():
    radius, kappa, mu2, h0 = 2.4, 0.7, 1.3, 0.8
    h = np.arange(9)/8
    source = np.arange(9, 0, -1)/7
    theta, phi, area = f.sphere_quadrature(radius, 32, 64)
    step = 1e-5
    values = f.real_harmonics(theta, phi)@h
    dt = ((f.real_harmonics(theta+step, phi)-f.real_harmonics(theta-step, phi))@h)/(2*step)
    dp = ((f.real_harmonics(theta, phi+step)-f.real_harmonics(theta, phi-step))@h)/(2*step)
    gradient_squared = (dt**2+dp**2/np.sin(theta)**2)/radius**2
    direct = np.dot(area, kappa*gradient_squared/2 + mu2*(values-h0)**2/2
                    - (f.real_harmonics(theta, phi)@source)*values)
    assert_close(f.scalar_action(h, source, radius, kappa, mu2, h0), direct, atol=2e-8)


def test_common_proper_spatial_rotation_invariance():
    # Rotate unit-sphere coordinates using Rz(alpha) Ry(beta), determinant +1.
    alpha, beta = 0.71, 0.43
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    rotation = Rz@Ry
    assert_close(np.linalg.det(rotation), 1)
    sources = (lambda x, y, z: z+(3*z*z-1)/4,
               lambda x, y, z: x+z/3+x*y/2)
    base, rotated = [], []
    for source in sources:
        def transformed(x, y, z):
            return source(*(rotation @ np.stack((x, y, z))))
        base.append(f.yukawa_matrix(f.scalar_minimizer(f.source_coefficients(source))))
        rotated.append(f.yukawa_matrix(f.scalar_minimizer(f.source_coefficients(transformed))))
    before, after = f.mixing_observables(*base), f.mixing_observables(*rotated)
    for sector in ("up", "down"):
        assert_close(before[sector]["singular_values"], after[sector]["singular_values"])
    assert_close(before["abs_mixing"], after["abs_mixing"])
    assert_close(before["J"], after["J"])


def test_unforced_and_physical_mass_input():
    h = f.scalar_minimizer(np.zeros(9), h0=2)
    Y = f.yukawa_matrix(h, y_eff=0.7)
    assert_close(Y, 1.4*np.eye(3))
    assert_close(f.mass_matrix(Y, v=7), Y*7/np.sqrt(2))
    result = f.mixing_observables(Y, Y)
    assert result["mixing"] is result["abs_mixing"] is result["J"] is None
    assert "up, down" in result["mixing_undefined_reason"]


@pytest.mark.parametrize("order", [5, 8, 9, 12])
def test_exact_cyclic_and_proper_dihedral_symmetry(order):
    axial = lambda x, y, z: 2+0.3*z+0.2*(3*z*z-1)/2
    theta, phi, _ = f.sphere_quadrature()
    xyz = f._xyz(theta, phi)
    rotated = f._xyz(theta, phi+2*np.pi/order)
    assert_close(axial(*xyz), axial(*rotated))
    Y = f.overlap_matrix(axial)
    assert_close(Y, np.diag(np.diag(Y)))
    # Section representation of azimuthal rotations has weights k=0,1,2.
    U = np.diag(np.exp(2j*np.pi*np.arange(3)/order))
    assert_close(U.conj().T@Y@U, Y)
    control = f.mixing_observables(Y, f.overlap_matrix(lambda x, y, z: 2+0.2*z))
    assert_close(control["abs_mixing"], np.eye(3))
    assert abs(control["J"]) < 1e-12
    # Proper rotation by pi about x: (x,y,z)->(x,-y,-z), determinant +1.
    even = lambda x, y, z: 2+0.2*(3*z*z-1)/2
    x, y, z = xyz
    assert_close(even(x, y, z), even(x, -y, -z))
    D = f.overlap_matrix(even)
    assert_close(D[0, 0], D[2, 2])
    assert f.mixing_observables(Y, D)["J"] is None


def test_svd_signed_degeneracy_and_reconstruction():
    opposite = np.diag([-1., 1., 2.])
    result = f.mixing_observables(opposite, np.diag([1., 2., 3.]))
    assert result["abs_mixing"] is result["J"] is None
    assert result["mixing_undefined_reason"] == "degenerate singular spectrum in up"
    Y = np.array([[1+1j, 2, 0], [0.2j, -1, 3j], [2, 0.7j, 0.3]])
    result = f.svd_sector(Y)
    U, s, Vh = result["left"], result["singular_values"], result["right_adjoint"]
    assert np.all(np.diff(s) >= 0)
    assert_close((U*s)@Vh, Y)
    assert_close(U.conj().T@Y@Y.conj().T@U, np.diag(s*s))
    assert_close(U.conj().T@U, np.eye(3))


def test_frozen_toy_analytic_matrices_and_cp():
    up, down = f.fixed_toy_sources()
    Yu = f.yukawa_matrix(f.scalar_minimizer(up))
    Yd = f.yukawa_matrix(f.scalar_minimizer(down))
    neighbor = 1/(6*np.sqrt(2))
    expected_up = np.diag([913/420, 139/70, 773/420])
    expected_down = np.array([[37/18, neighbor, 1j/140],
                              [neighbor, 2, neighbor], [-1j/140, neighbor, 35/18]])
    assert_close(Yu, expected_up)
    assert_close(Yd, expected_down)
    result = f.mixing_observables(Yu, Yd)
    assert_close(result["mixing"].conj().T@result["mixing"], np.eye(3))
    assert result["left_squared_commutator_norm"] > 0
    assert abs(result["J"]) > 1e-4
    cp = f.mixing_observables(Yu, Yd.real)
    assert abs(cp["J"]) < 1e-12


def test_rephasing_right_rotations_and_common_basis_invariance():
    up, down = f.fixed_toy_sources()
    Yu, Yd = (f.yukawa_matrix(f.scalar_minimizer(s)) for s in (up, down))
    base = f.mixing_observables(Yu, Yd)
    # Deterministic unitary family basis and independent right-handed bases.
    raw = np.array([[1, 2j, 3], [2, 1, 1j], [1j, 3, 2]])
    W, _ = np.linalg.qr(raw)
    Ru = np.diag(np.exp(1j*np.array([0.3, 1.7, -0.2])))
    Rd = np.diag(np.exp(1j*np.array([-0.9, 0.2, 2.1])))
    changed = f.mixing_observables(W.conj().T@Yu@Ru, W.conj().T@Yd@Rd)
    assert_close(changed["abs_mixing"], base["abs_mixing"])
    assert_close(changed["J"], base["J"])
    V = Ru@base["mixing"]@Rd
    assert_close(np.abs(V), base["abs_mixing"])
    assert_close(np.imag(V[0, 0]*V[1, 1]*V[0, 1].conj()*V[1, 0].conj()), base["J"])


@pytest.mark.parametrize("families", [0, 1, 3, 4, 11])
@pytest.mark.parametrize("neutral", [False, True])
def test_exact_sm_anomalies_do_not_select_families(families, neutral):
    result = f.sm_anomalies(families, neutral)
    assert all(type(value) is Fraction and value == 0 for value in result["per_family"].values())
    assert all(value == 0 for value in result["totals"].values())
    assert result["weak_doublets_per_family"] == 4
    assert result["witten_mod2"] == 0
    assert not result["family_count_selected"]
    assert "unresolved" in result["higher_dimensional_status"]
    assert result["field_contributions_per_family"]["u^c"]["SU3_cubed"] == -1
    assert result["field_contributions_per_family"]["Q"]["SU2_squared_U1"] == Fraction(1, 4)


def test_assumptions_demo_and_dependency_boundary():
    report = f.demonstration()
    assert report["model_id"] == f.MODEL_ID
    assert report["inputs"]["v"] is None
    assert "dimensionless" in report["inputs"]["output_units"]
    assert report["unforced"]["J"] is None
    assert report["cyclic"]["proper_dihedral_control"]["J"] is None
    assert report["forced"]["J"] is not None
    ledger = report["assumption_ledger"]
    assert "any Hermitian 3x3" in ledger["source_freedom"]
    assert "not literal 6D or Spin(10)" in ledger["scope"]
    assert "not spontaneous" in ledger["selection_status"]
    # Whitelist imports instead of opening any empirical/benchmark modules.
    tree = ast.parse(Path(f.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) <= {"numpy", "fractions", "math", "numbers", "bpr.flavor_foundations"}
    assert "tilde(H)=i sigma_2 H*" in f.__doc__
    assert "bar(Q_L)^a_A" in f.__doc__


def test_stdout_demo_both_modes_from_other_directory(tmp_path):
    script = Path(f.__file__).resolve().parents[1]/"scripts"/"demo_conditional_gauge_flavor.py"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    result = subprocess.run([sys.executable, str(script), "--json"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True, timeout=60)
    def reject_nonstandard(value):
        raise ValueError(f"nonstandard JSON constant {value}")
    parsed = json.loads(result.stdout, parse_constant=reject_nonstandard)
    assert parsed["flavor"]["model_id"] == f.MODEL_ID
    assert parsed["gauge"]["specification"]["model_id"] != f.MODEL_ID
    assert parsed["flavor"]["unforced"]["mixing"] is None
    assert parsed["flavor"]["unforced"]["abs_mixing"] is None
    assert parsed["flavor"]["unforced"]["J"] is None
    assert set(parsed["flavor"]["forced"]["up"]["matrix"]) == {"real", "imag"}
    plain = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                           capture_output=True, text=True, check=True, timeout=60)
    assert "FLAVOR" in plain.stdout and "GAUGE" in plain.stdout
    assert list(tmp_path.iterdir()) == []
