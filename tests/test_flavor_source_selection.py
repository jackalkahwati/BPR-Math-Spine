"""Analytic occupation selection and limitations, without empirical targets."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import chiral_flavor_prototype as f
from bpr import flavor_source_selection as s


def close(actual, expected, atol=3e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=3e-12)


def pure(v):
    v = np.asarray(v, dtype=complex)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


@pytest.mark.parametrize("q", [1, 2, 3, 4])
@pytest.mark.parametrize("R", [0.4, 1.0, 2.7])
def test_addition_theorem_equal_occupation_and_radius(q, R):
    theta, phi, area = f.sphere_quadrature(R)
    psi = f.monopole_profiles(theta, phi, q=q, R=R)
    close(np.sum(abs(psi)**2, axis=1), q/(4*np.pi*R**2))
    rho = s.equal_occupation(q)
    density = s.occupation_density(rho, theta, phi, R)
    close(density, 1/(4*np.pi*R**2))
    close(np.dot(area, density), 1)
    close(s.occupation_density(rho, theta, phi, R, 0.3*np.sin(phi)+theta), density)


@pytest.mark.parametrize("theta0,phi0", [(0, 0), (0.7, 1.1), (1.9, -0.4), (np.pi, 2)])
def test_coherent_complex_convention_peaks_at_actual_axis(theta0, phi0):
    rho = s.coherent_occupation(theta0, phi0)
    direction = np.array([np.sin(theta0)*np.cos(phi0), np.sin(theta0)*np.sin(phi0), np.cos(theta0)])
    theta, phi, area = f.sphere_quadrature(1.7)
    xyz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    z = direction @ xyz
    expected = 3*(1+z)**2/(16*np.pi*1.7**2)
    density = s.occupation_density(rho, theta, phi, 1.7)
    close(density, expected)
    close(np.dot(area, density), 1)
    close(s.spin_vector(rho), direction)
    close(s.occupation_density(rho, theta0, phi0), 3/(4*np.pi))
    if theta0 == 0.7:
        assert s.occupation_density(rho, theta0, -phi0) < s.occupation_density(rho, theta0, phi0)
        assert np.max(abs(rho.imag)) > 0.1


def test_arbitrary_complex_density_trace_overlap_and_direct_harmonic_coefficients():
    rho = 0.7*pure([1, 2j, -0.3+0.8j]) + 0.3*s.equal_occupation()
    R = 2.3
    theta, phi, area = f.sphere_quadrature(R)
    n = s.occupation_density(rho, theta, phi, R)
    h = np.arange(9)/5
    assert np.min(n) >= 0
    close(np.dot(area, n), np.trace(rho))
    close(np.dot(area, (f.real_harmonics(theta, phi)@h)*n), np.trace(rho @ f.overlap_matrix(h, R=R)))
    coefficients = s.density_coefficients(rho, R)
    close(coefficients, f.real_harmonics(theta, phi).T @ (area*n)/R**2)
    close(f.real_harmonics(theta, phi) @ coefficients, n)
    close(coefficients*R**2, s.density_coefficients(rho))


def test_fixed_trace_source_map_has_eight_local_directions_not_all_signed_sources():
    tangent = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        real = np.zeros((3, 3), complex)
        real[i, j] = real[j, i] = 1
        imag = np.zeros((3, 3), complex)
        imag[i, j], imag[j, i] = 1j, -1j
        tangent.extend((real, imag))
    tangent.extend((np.diag([1, -1, 0]), np.diag([0, 1, -1])))
    equal = s.equal_occupation()
    base = s.density_coefficients(equal)
    differences = np.array([s.density_coefficients(equal+0.01*t)-base for t in tangent])
    assert np.linalg.matrix_rank(differences, tol=1e-12) == 8
    close(differences[:, 0], 0)
    assert base[0] > 0
    assert "do not cover all signed" in s.assumption_ledger()["source_freedom"]


@pytest.mark.parametrize("v", [[1, 0, 0], [0, 1, 0], [1, 1j, 2], [0.2j, 0.7, 0.3+0.5j]])
@pytest.mark.parametrize("R", [0.5, 1.0, 2.2])
def test_pure_spin_one_multipole_identity(v, R):
    rho = pure(v)
    spin = s.spin_vector(rho)
    spin2 = spin @ spin
    assert -1e-12 <= spin2 <= 1+1e-12
    expected = np.array([1/4, 3*spin2/16, (4-3*spin2)/80])/(np.pi*R**4)
    close(s.density_multipole_powers(rho, R), expected)


@pytest.mark.parametrize("rho", [np.eye(2), np.ones(3), np.eye(3), -np.eye(3)/3,
    np.diag([-0.01, 0.5, 0.51]), np.diag([np.nan, 0, 1]), np.diag([np.inf, 0, 1]),
    np.array([[0.3, 0.1j, 0], [0.1j, 0.3, 0], [0, 0, 0.4]])])
def test_invalid_density_rejected_without_repair(rho):
    with pytest.raises(ValueError):
        s.validated_density_matrix(rho)
    with pytest.raises(ValueError):
        s.density_coefficients(rho)


def test_density_validator_does_not_mutate_clip_or_renormalize():
    rho = np.diag([-1e-15, 0.5, 0.5+1e-15]).astype(complex)
    result = s.validated_density_matrix(rho)
    np.testing.assert_array_equal(result, rho)
    assert result[0, 0] < 0
    result[1, 1] = 0
    assert rho[1, 1] == 0.5
    with pytest.raises(ValueError, match="trace"):
        s.validated_density_matrix(np.eye(3)*(1+1e-9)/3)


@pytest.mark.parametrize("kwargs", [{"R": 0}, {"R": -1}, {"R": np.nan}, {"R": np.inf},
    {"mu2": 0}, {"mu2": 0.25}, {"mu2": 0.1}, {"eta": -0.1}, {"kappa": -1},
    {"h0": -1}, {"g_u": 0}, {"g_d": -1}, {"g_u": np.nan}, {"h0": np.inf}])
def test_invalid_model_domain(kwargs):
    with pytest.raises(ValueError):
        s.SelectionParameters(**kwargs)


@pytest.mark.parametrize("kwargs", [{"R": True}, {"g_d": "1"}, {"eta": 1j}])
def test_parameter_types(kwargs):
    with pytest.raises(TypeError):
        s.SelectionParameters(**kwargs)


@pytest.mark.parametrize("kwargs", [{"R": 1e-200}, {"R": 1e200},
    {"mu2": 1.0, "eta": np.nextafter(1.0, 0.0)}, {"kappa": 1e308}])
def test_unresolved_hessian_or_radius_rejected(kwargs):
    with pytest.raises(ValueError, match="unresolved"):
        s.SelectionParameters(**kwargs)


def test_extreme_scales_do_not_fabricate_resolved_spectra_or_response():
    rho = s.coherent_occupation()
    p = s.SelectionParameters(g_u=1e-100, g_d=1e-100)
    with pytest.raises(ValueError, match="unresolved"):
        s.coupled_scalar_response(rho, rho, p)
    with pytest.raises(ValueError, match="unresolved"):
        s.coherent_axis_profiles(p)
    p = s.SelectionParameters(mu2=1e200, eta=1e-200)
    with pytest.raises(ValueError, match="unresolved"):
        s.response_weights(p)
    p = s.SelectionParameters(mu2=1e100, g_u=1e155, g_d=1e155, h0=0)
    # Conservative intermediate-range rejection, not a claim of infinite exact energy.
    with pytest.raises(ValueError, match="unresolved"):
        s.analytic_energy_bound(p)
    with pytest.raises(ValueError, match="unresolved"):
        s.effective_energy(rho, rho, p)
    p = s.SelectionParameters(g_u=1e-100, g_d=1e-100)
    with pytest.raises(ValueError, match="unresolved"):
        s.analytic_energy_bound(p)
    with pytest.raises(ValueError, match="unresolved"):
        s.effective_energy(rho, rho, p)


@pytest.mark.parametrize("eta,angle,coupling", [(0, 0.9, 1e40), (0.25, 0, 1e80)])
def test_nonfinite_inherited_yukawa_diagnostics_rejected(eta, angle, coupling):
    p = s.SelectionParameters(eta=eta)
    pair = s.minimizing_family(down_theta=angle, down_phi=1.1, parameters=p)
    with pytest.raises(ValueError, match="unresolved.*diagnostics"):
        s.solution_diagnostics(*pair, p, y_eff_u=coupling, y_eff_d=coupling)


@pytest.mark.parametrize("p", [s.SelectionParameters(), s.SelectionParameters(R=2.3, kappa=0.7, mu2=1.8, eta=0.6, g_u=0.4, g_d=1.3, h0=0.5),
    s.SelectionParameters(R=0.4, kappa=0, mu2=2, eta=0), s.SelectionParameters(kappa=0, eta=0.9, h0=0)])
def test_block_solution_action_stationarity_and_hessian(p):
    rho_u = pure([1, 1j, -0.4])
    rho_d = 0.4*s.coherent_occupation(0.8, 1.4)+0.6*s.equal_occupation()
    h = s.coupled_scalar_response(rho_u, rho_d, p)
    j = s.occupation_sources(rho_u, rho_d, p)
    delta = h.copy()
    delta[:, 0] -= p.h0*np.sqrt(4*np.pi)
    a, b = s.response_weights(p)
    for ell in range(3):
        d = p.mu2+p.kappa*ell*(ell+1)/p.R**2
        close(np.array([[d, -p.eta], [-p.eta, d]]) @ np.array([[a[ell], b[ell]], [b[ell], a[ell]]]), np.eye(2))
    close(s.coupled_scalar_stationarity(h, rho_u, rho_d, p), np.zeros((2, 9)))
    Hessian = s.coupled_scalar_hessian(p)
    assert np.linalg.eigvalsh(Hessian).min() > 0
    close(Hessian @ delta.ravel(), p.R**2*j.ravel())
    energy = s.coupled_scalar_action(h, rho_u, rho_d, p)
    close(energy, s.effective_energy(rho_u, rho_d, p))
    variation = np.arange(18).reshape(2, 9)/45 - 0.2
    increase = s.coupled_scalar_action(h+variation, rho_u, rho_d, p)-energy
    close(increase, variation.ravel() @ Hessian @ variation.ravel()/2)
    assert increase > 0
    close(s.coupled_scalar_stationarity(h+variation, rho_u, rho_d, p).ravel(), Hessian @ variation.ravel())
    step = 1e-5
    numerical = (s.coupled_scalar_action(h+variation+step*variation, rho_u, rho_d, p)
                 - s.coupled_scalar_action(h+variation-step*variation, rho_u, rho_d, p))/(2*step)
    close(numerical, variation.ravel() @ Hessian @ variation.ravel(), atol=1e-8)
    # Independent area integral including direct finite-difference gradients.
    theta, phi, area = f.sphere_quadrature(p.R)
    basis = f.real_harmonics(theta, phi)
    values = basis @ h.T
    dt = ((f.real_harmonics(theta+step, phi)-f.real_harmonics(theta-step, phi)) @ h.T)/(2*step)
    dp = ((f.real_harmonics(theta, phi+step)-f.real_harmonics(theta, phi-step)) @ h.T)/(2*step)
    grad2 = (dt**2+dp**2/np.sin(theta[:, None])**2)/p.R**2
    direct = area @ (np.sum(p.kappa*grad2/2+p.mu2*(values-p.h0)**2/2-(basis @ j.T)*values, axis=1)
                     - p.eta*(values[:, 0]-p.h0)*(values[:, 1]-p.h0))
    close(energy, direct, atol=1e-8)


@pytest.mark.parametrize("eta,kappa", [(0, 0), (0, 1), (0.25, 0), (0.25, 1)])
def test_global_bound_saturation_and_competitors(eta, kappa):
    p = s.SelectionParameters(R=1.3, eta=eta, kappa=kappa, g_u=0.7, g_d=1.4)
    bound = s.analytic_energy_bound(p)
    rho = s.minimizing_family(0.8, 0.9, parameters=p)
    close(s.effective_energy(*rho, p), bound)
    # Supplementary seeded algebraic trials, not a search or proof of the bound.
    rng = np.random.default_rng(7201)
    for index in range(12):
        matrices = []
        for _ in range(2):
            raw = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
            trial = pure(raw[:, 0]) if index % 2 else raw @ raw.conj().T / np.trace(raw @ raw.conj().T)
            matrices.append(trial)
        assert s.effective_energy(*matrices, p) >= bound-1e-12
    noncoherent = pure([0, 1, 0])
    assert s.effective_energy(noncoherent, noncoherent, p) > bound
    mixed = s.equal_occupation()
    assert s.effective_energy(mixed, mixed, p) > bound
    # Noncoherent pure spin-zero state has a downhill coherent admixture.
    less_symmetric = pure([0.15, 1, 0.15])
    assert s.effective_energy(less_symmetric, less_symmetric, p) < s.effective_energy(noncoherent, noncoherent, p)
    a, b = s.response_weights(p)
    assert 5*a[1] > a[2]
    if eta:
        assert 5*b[1] > b[2]
    close(np.maximum(np.diff(a), 0), 0)
    close(np.maximum(np.diff(b), 0), 0)


@pytest.mark.parametrize("p", [s.SelectionParameters(), s.SelectionParameters(kappa=0, h0=0, g_u=0.8, g_d=1.4),
    s.SelectionParameters(R=2.1, kappa=2, mu2=3, eta=1, g_u=0.3, g_d=0.9)])
def test_analytic_profiles_positive_nondegenerate_overlaps_and_no_mixing(p):
    rho = s.coherent_occupation()
    h = s.coupled_scalar_response(rho, rho, p)
    abc = s.coherent_axis_profiles(p)
    A, B, C = abc.T
    assert np.all(B >= 3*C-1e-14)
    assert np.all(A-p.h0 >= 2*B/3-1e-14)
    expected = s.analytic_overlap_diagonal(p)
    assert np.all(expected[:, -1] >= p.h0+B/6+C/10-1e-14)
    assert np.all(np.diff(expected, axis=1) < 0)
    assert np.min(expected) > 0
    for sector in range(2):
        coefficients = np.zeros(9)
        coefficients[[0, 3, 6]] = abc[sector]/f.HARMONIC_NORMALIZATIONS[[0, 3, 6]]
        close(h[sector], coefficients)
        close(f.yukawa_matrix(h[sector], R=p.R), np.diag(expected[sector]))
    result = s.solution_diagnostics(rho, rho, p)
    close(result["observables"]["abs_mixing"], np.eye(3))
    close(result["observables"]["J"], 0)
    close(result["observables"]["up"]["singular_values"], expected[0, ::-1])


def test_alignment_strict_energy_and_uncoupled_relative_flatness():
    north, tilted = s.coherent_occupation(), s.coherent_occupation(1.1, 0.7)
    p = s.SelectionParameters()
    assert s.effective_energy(north, tilted, p) > s.effective_energy(north, north, p)
    with pytest.raises(ValueError, match="identical coherent"):
        s.minimizing_family(down_theta=1.1, down_phi=0.7, parameters=p)
    p0 = replace(p, eta=0)
    pair = s.minimizing_family(down_theta=1.1, down_phi=0.7, parameters=p0)
    close(pair, [north, tilted])
    close(s.effective_energy(*pair, p0), s.analytic_energy_bound(p0))
    aligned = s.solution_diagnostics(north, north, p0)["observables"]
    separate = s.solution_diagnostics(*pair, p0)["observables"]
    assert aligned["mixing"] is not None and separate["mixing"] is not None
    assert np.linalg.norm(aligned["abs_mixing"]-separate["abs_mixing"]) > 0.5
    # A common rotation is a flat symmetry direction, not an instability.
    both_tilted = s.solution_diagnostics(tilted, tilted, p)
    close(both_tilted["effective_energy"], s.analytic_energy_bound(p))
    close(both_tilted["observables"]["abs_mixing"], np.eye(3))


def test_common_basis_and_right_rephasing_diagnostics():
    p = s.SelectionParameters(eta=0)
    pair = s.minimizing_family(down_theta=0.9, down_phi=1.1, parameters=p)
    obs = s.solution_diagnostics(*pair, p)["observables"]
    U, _ = np.linalg.qr(np.array([[1, 2j, 3], [2, 1, 1j], [1j, 3, 2]]))
    ru = np.diag(np.exp(1j*np.array([0.1, 0.7, -0.6])))
    rd = np.diag(np.exp(1j*np.array([0.3, -0.2, 1.3])))
    rotated = f.mixing_observables(U.conj().T @ obs["up"]["matrix"] @ ru,
                                   U.conj().T @ obs["down"]["matrix"] @ rd)
    close(rotated["abs_mixing"], obs["abs_mixing"])
    close(rotated["J"], obs["J"])


def test_equal_control_is_stationary_in_occupation_but_not_minimum():
    p = s.SelectionParameters()
    equal = s.equal_occupation()
    h = s.coupled_scalar_response(equal, equal, p)
    close(h[:, 1:], 0)
    close(s.coupled_scalar_stationarity(h, equal, equal, p), 0)
    result = s.solution_diagnostics(equal, equal, p)
    assert result["bound_residual"] > 0
    assert result["observables"]["abs_mixing"] is None
    assert "degenerate" in result["observables"]["mixing_undefined_reason"]
    delta = 0.01*np.diag([1, 0, -1])
    e0 = s.effective_energy(equal, equal, p)
    plus = s.effective_energy(equal+delta, equal, p)
    minus = s.effective_energy(equal-delta, equal, p)
    close(plus, minus)
    assert plus < e0


def test_stable_bound_gap_excludes_common_monopole_and_reports_resolution():
    p = s.SelectionParameters(kappa=1e15)
    equal = s.equal_occupation()
    result = s.solution_diagnostics(equal, equal, p)
    a, b = s.response_weights(p)
    expected = (a[1]+b[1])*3/(16*np.pi)+(a[2]+b[2])/(80*np.pi)
    # The naive subtraction demonstrably loses the physically conditional gap.
    assert result["effective_energy"]-result["energy_bound"] == 0
    np.testing.assert_allclose(result["bound_residual"], expected, rtol=2e-12, atol=0)
    assert result["bound_residual"] > result["bound_residual_resolution"] > 0
    assert result["bound_residual_status"] == "positive angular energy gap"
    coherent = s.coherent_occupation(0.8, 1.1)
    gap = s.energy_bound_gap(coherent, coherent, p)
    assert abs(gap["gap"]) <= gap["resolution"]
    assert "not a numerical selection proof" in gap["status"]
    for p in (s.SelectionParameters(), s.SelectionParameters(R=2.3, eta=0)):
        rho = pure([1, 1j, 0.8])
        stable = s.energy_bound_gap(rho, equal, p)
        close(stable["gap"], s.effective_energy(rho, equal, p)-s.analytic_energy_bound(p))


def test_action_radius_scaling_at_fixed_gradient_response():
    p = s.SelectionParameters(h0=0)
    larger = replace(p, R=2, kappa=4)
    rho = s.coherent_occupation(0.6, 1.1)
    close(s.coupled_scalar_response(rho, rho, larger)*4, s.coupled_scalar_response(rho, rho, p))
    close(s.effective_energy(rho, rho, larger)*4, s.effective_energy(rho, rho, p))
    close(s.analytic_energy_bound(larger)*4, s.analytic_energy_bound(p))


def test_misc_validation():
    for q in (True, 3.0, "3"):
        with pytest.raises(TypeError):
            s.equal_occupation(q)
    with pytest.raises(ValueError):
        s.equal_occupation(0)
    with pytest.raises(ValueError):
        s.coherent_occupation(-1, 0)
    with pytest.raises(ValueError):
        s.coherent_occupation(1, np.nan)
    with pytest.raises(TypeError):
        s.analytic_energy_bound({})
    rho = s.equal_occupation()
    for h in (np.zeros(9), np.zeros((9, 2)), np.ones((2, 9))*1j, np.full((2, 9), np.nan)):
        with pytest.raises(ValueError):
            s.coupled_scalar_action(h, rho, rho)
    with pytest.raises(ValueError):
        s.solution_diagnostics(rho, rho, y_eff_u=0)


def test_frozen_demonstration_ledger_rotational_average_and_dependency_boundary():
    report = s.demonstration()
    assert report["model_id"] == s.MODEL_ID
    for name in ("R", "kappa", "mu2", "g_u", "g_d", "y_eff_u", "y_eff_d"):
        assert report["inputs"][name] == 1
    assert report["inputs"]["h0"] == 2 and report["inputs"]["eta"] == 0.25
    close(report["aligned"]["bound_residual"], 0)
    close(report["aligned"]["action"], report["aligned"]["energy_bound"])
    close(report["aligned"]["observables"]["up"]["singular_values"], report["aligned"]["observables"]["down"]["singular_values"])
    assert "relative mixing unselected" == report["uncoupled"]["selection_status"]
    assert all(report["uncoupled"][key] is None for key in ("mixing", "abs_mixing", "J"))
    for case in report["uncoupled"]["representatives"].values():
        close(case["bound_residual"], 0)
        assert case["observables"]["abs_mixing"] is not None
    average = report["rotational_average"]
    close(average["density_matrix"], s.equal_occupation())
    assert average["energy_of_average_density"] > average["averaged_orbit_energy"]
    assert "not an exact quantum ground state" in report["assumption_ledger"]["quantum_boundary"]
    tree = ast.parse(Path(s.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) <= {"dataclasses", "numbers", "numpy", "bpr.chiral_flavor_prototype"}


def test_stdout_demo_text_and_strict_json(tmp_path):
    script = Path(s.__file__).resolve().parents[1]/"scripts"/"demo_flavor_source_selection.py"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    outputs = []
    for args in (["--json"], []):
        result = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, env=env,
                                capture_output=True, text=True, check=True, timeout=60)
        outputs.append(result.stdout)
    def reject_nonstandard(value):
        raise ValueError(value)
    report = json.loads(outputs[0], parse_constant=reject_nonstandard)
    assert report["model_id"] == s.MODEL_ID
    assert report["uncoupled"]["abs_mixing"] is None
    assert report["equal_filled"]["observables"]["J"] is None
    assert set(report["aligned"]["observables"]["up"]["matrix"]) == {"real", "imag"}
    assert "relative mixing unselected" in outputs[1]
    assert "not a BPR-derived vacuum" in outputs[1]
    assert list(tmp_path.iterdir()) == []
