"""Independent fixed-ring integration checks, not physical benchmarks or fits."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad_vec, solve_ivp

from bpr import substrate_nonlinear_response as r
from bpr import substrate_triplet_projection as projection

Z = np.array([0.3+0.7j, -0.4+0.2j, 0.6-0.1j])
PS = (5, 6, 7, 11)


def u(p):
    return np.array([[np.exp(2j*np.pi*x*k/p)/np.sqrt(p) for k in (-1, 0, 1)] for x in range(p)])


def h0(p, C):
    identity = np.eye(p)
    return C*(2*identity-np.roll(identity, 1, axis=0)-np.roll(identity, -1, axis=0))


def independent_integral(z, p, C, t):
    """Site-space cubic forcing, with a dense real hopping eigensystem."""
    eigenvalues, eigenvectors = np.linalg.eigh(h0(p, C))
    def propagate(field, time):
        return eigenvectors @ (np.exp(-1j*eigenvalues*time)*(eigenvectors.T @ field))
    initial = u(p) @ z
    def integrand(s):
        field = propagate(initial, s)
        return -1j*propagate(abs(field)**2*field, t-s)
    return quad_vec(integrand, 0, t, epsabs=2e-13, epsrel=2e-13)[0]


def trajectories(z, p, C, g, t):
    """DOP853 tolerance is a numerical check, NOT a rigorous solver certificate."""
    U, H = u(p), h0(p, C)
    def full(_, field):
        return -1j*(H @ field+g*abs(field)**2*field)
    def projected(_, a):
        field = U @ a
        return U.conj().T @ full(0, field)
    kwargs = {"method": "DOP853", "rtol": 2e-12, "atol": 2e-14}
    full_sol = solve_ivp(full, (0, t), U @ z, **kwargs)
    gal_sol = solve_ivp(projected, (0, t), z, **kwargs)
    assert full_sol.success and gal_sol.success
    return full_sol.y[:, -1], U @ gal_sol.y[:, -1]


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("t", [0.0, 0.1, -0.27, 1.1])
def test_first_derivative_independent_site_space_quadrature(p, t):
    a = r.first_order_response(Z, p, C=1.3, t=t)
    expected = independent_integral(Z, p, 1.3, t)
    np.testing.assert_allclose(a["psi1"], expected, atol=3e-14, rtol=2e-13)
    U = u(p)
    np.testing.assert_allclose(a["retained_psi1"], U @ (U.conj().T @ expected), atol=3e-14)
    np.testing.assert_allclose(a["discarded_psi1"], expected-U @ (U.conj().T @ expected), atol=3e-14)
    np.testing.assert_allclose(a["psi1"], a["retained_psi1"]+a["discarded_psi1"], atol=2e-16)


@pytest.mark.parametrize("p", PS)
def test_all_27_channels_and_exact_identity_resonances(p):
    channels = r.cubic_channels(p, 1.7)
    assert len(channels) == 27
    seen = set()
    for channel in channels:
        i, j, ell = channel["input_modes"]
        k = i-j+ell
        seen.add((i, j, ell))
        assert channel["raw_output"] == k
        assert channel["residue"] == k % p
        assert channel["retained"] == (k in (-1, 0, 1))
        energies = lambda a: 4*1.7*np.sin(np.pi*a/p)**2
        expected = energies(i)-energies(j)+energies(ell)-energies(k)
        assert channel["detuning"] == pytest.approx(expected, abs=5e-15)
        if k in (-1, 0, 1) and i*i-j*j+ell*ell-k*k == 0:
            assert channel["detuning"] == 0
            assert channel["exact_resonance"]
        else:
            assert channel["detuning"] != 0
            assert not channel["exact_resonance"]
    assert len(seen) == 27


@pytest.mark.parametrize("p", [5, 6])
def test_aliases_integrated_before_aggregation(p):
    channels = r.cubic_channels(p)
    off = [c for c in channels if not c["retained"]]
    residues = {c["residue"] for c in off}
    assert len(residues) == (2 if p == 5 else 3)
    if p == 5:
        assert any(len({c["detuning"] for c in off if c["residue"] == residue}) > 1 for residue in residues)
    actual = r.first_order_response(Z, p, t=0.9)
    expected = independent_integral(Z, p, 1.0, 0.9)
    np.testing.assert_allclose(actual["discarded_psi1"], expected-u(p) @ (u(p).conj().T @ expected), atol=1e-14)


@pytest.mark.parametrize("omega", [0, 1e-15, -1e-15, 0.7, -2.3])
@pytest.mark.parametrize("t", [0, 1e-5, 0.6, -0.6])
def test_sinc_kernel_against_independent_quadrature(omega, t):
    expected = quad_vec(lambda x: np.exp(-1j*omega*x), 0, t, epsabs=1e-14)[0]
    assert r.duhamel_kernel(omega, t) == pytest.approx(expected, abs=4e-16)
    if omega == 0:
        assert r.duhamel_kernel(omega, t) == complex(t)


def test_near_zero_nonresonance_is_not_rounded_to_exact_resonance():
    channels = r.cubic_channels(10**10, 1.0)
    assert all(c["detuning"] != 0 for c in channels if not c["exact_resonance"])
    off = [c for c in channels if c["detuning_identity"] == "2W-epsilon2"]
    assert off[0]["detuning"] == pytest.approx(-8*np.pi**2/10**20, rel=1e-14)
    a = r.first_order_response(Z, 7, C=1e-12, t=0.2)
    np.testing.assert_allclose(a["psi1"], independent_integral(Z, 7, 1e-12, 0.2), atol=2e-15)


@pytest.mark.parametrize("p", PS)
def test_global_phase_translation_reflection_time_reversal_and_amplitude_scaling(p):
    a = r.first_order_response(Z, p, t=0.21)
    phase = np.exp(0.37j)
    b = r.first_order_response(phase*Z, p, t=0.21)
    translated = r.first_order_response(Z*np.exp(2j*np.pi*np.array([-1, 0, 1])/p), p, t=0.21)
    reflected = r.first_order_response(Z[::-1], p, t=0.21)
    reversed_time = r.first_order_response(Z[::-1].conj(), p, t=-0.21)
    scaled = r.first_order_response(1.4*Z, p, t=0.21)
    for key in ("free_field", "psi1", "retained_psi1", "discarded_psi1"):
        np.testing.assert_allclose(b[key], phase*a[key], atol=2e-15)
        np.testing.assert_allclose(translated[key], np.roll(a[key], -1), atol=2e-15)
        np.testing.assert_allclose(reflected[key], a[key][(-np.arange(p)) % p], atol=2e-15)
        np.testing.assert_allclose(reversed_time[key], a[key].conj(), atol=2e-15)
        power = 1 if key == "free_field" else 3
        np.testing.assert_allclose(scaled[key], 1.4**power*a[key], atol=2e-15)


@pytest.mark.parametrize("p", PS)
def test_short_time_existing_gradient_and_leakage(p):
    t = 1e-7
    a = r.first_order_response(Z, p, t=t)
    projected_cubic = projection.projected_gradient(Z, p, g=1)-projection.projected_gradient(Z, p, g=0)
    np.testing.assert_allclose(1j*a["retained_fourier"][[p-1, 0, 1]]/t, projected_cubic, atol=2e-7)
    expected = projection.offband_coefficients(Z, p, g=1)
    for residue, coefficient in expected.items():
        assert 1j*a["discarded_fourier"][residue]/t == pytest.approx(coefficient, abs=2e-7)


@pytest.mark.parametrize("p", [5, 7])
@pytest.mark.parametrize("t", [0.2, -0.2])
def test_dop853_full_and_galerkin_bounds_and_second_order_scaling(p, t):
    source = np.sin(4*np.pi*np.arange(p)/p)+0.2*np.cos(2*np.pi*np.arange(p)/p)
    response = r.first_order_response(Z, p, C=1.2, t=t)
    errors, gal_errors, observable_errors = [], [], []
    for g in (0.08, 0.04, 0.02):
        full, galerkin = trajectories(Z, p, 1.2, g, t)
        b = r.finite_time_bounds(float(np.vdot(Z, Z).real), g, t, max(abs(source)))
        error = np.linalg.norm(full-response["free_field"]-g*response["psi1"])
        gal_error = np.linalg.norm(galerkin-response["free_field"]-g*response["retained_psi1"])
        assert error <= b["state_remainder_bound"]
        assert gal_error <= b["state_remainder_bound"]
        assert np.linalg.norm(full-response["free_field"]) <= b["free_state_error_bound"]
        obs = r.observable_response(Z, source, p, C=1.2, g=g, t=t)
        actual = np.vdot(full, source*full).real
        actual_gal = np.vdot(galerkin, source*galerkin).real
        obs_error = abs(actual-obs["first_order_observable"])
        assert obs_error <= b["observable_remainder_bound"]
        assert abs(actual_gal-obs["galerkin_first_order_observable"]) <= b["observable_remainder_bound"]
        assert abs(actual-actual_gal-obs["discarded_correction"]) <= b["full_vs_galerkin_observable_remainder_bound"]
        assert np.vdot(full, full).real == pytest.approx(np.vdot(Z, Z).real, abs=2e-11)
        errors.append(error)
        gal_errors.append(gal_error)
        observable_errors.append(obs_error)
    for sequence in (errors, gal_errors, observable_errors):
        assert sequence[0]/sequence[1] == pytest.approx(4, rel=0.08)
        assert sequence[1]/sequence[2] == pytest.approx(4, rel=0.04)


@pytest.mark.parametrize("p", PS)
def test_real_adjacent_mode_analytic_witness_and_cubic_cancellation(p):
    N, C, t = 1.0, 1.0, 0.1
    z = np.array([0, np.sqrt(N/2), np.sqrt(N/2)])
    source = np.sin(4*np.pi*np.arange(p)/p)
    obs = r.observable_response(z, source, p, C=C, g=0.5, t=t)
    w, e2 = 4*C*np.sin(np.pi/p)**2, 4*C*np.sin(2*np.pi/p)**2
    d = 2*w-e2
    lp = -N*N/(4*p)*np.sin(2*w*t)/(2*w)
    lq = N*N/(4*p)*t*np.sinc(d*t/(2*np.pi))*np.cos((e2+2*w)*t/2)
    assert obs["retained_derivative"] == pytest.approx(lp, abs=2e-17)
    assert obs["discarded_derivative"] == pytest.approx(lq, abs=2e-17)
    assert obs["full_derivative"] == pytest.approx(lp+lq, abs=3e-17)
    assert lq != 0 and abs(lp+lq) < abs(lp)*0.1
    small_t = 0.001
    small = r.observable_response(z, source, p, C=C, t=small_t)
    cubic = -N*N*e2*(e2+2*w)/(24*p)
    assert small["full_derivative"]/small_t**3 == pytest.approx(cubic, rel=1e-5)
    assert small["retained_derivative"]/small_t == pytest.approx(-N*N/(4*p), rel=1e-5)
    assert small["discarded_derivative"]/small_t == pytest.approx(N*N/(4*p), rel=2e-5)


@pytest.mark.parametrize("p", PS)
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_pure_mode_exact_control(p, k):
    z = np.zeros(3, complex)
    z[k+1] = 0.8+0.3j
    response = r.first_order_response(z, p, t=0.3)
    np.testing.assert_array_equal(response["discarded_fourier"], np.zeros(p))
    n = np.vdot(z, z).real
    np.testing.assert_allclose(response["psi1"], -1j*0.3*n/p*response["free_field"], atol=3e-17)
    assert response["arithmetic"]["discarded_status"] == "structural_zero"
    obs = r.observable_response(z, np.arange(p), p, t=0.3)
    assert obs["full_derivative"] == obs["discarded_derivative"] == 0
    assert obs["arithmetic"]["full"]["status"] == "structural_zero"


@pytest.mark.parametrize("p", PS)
def test_total_norm_and_full_square_distinguished(p):
    a = r.first_order_response(Z, p, t=0.4)
    # Independently check cancellation before the API's structural norm branch.
    assert 2*np.vdot(a["free_field"], a["psi1"]).real == pytest.approx(0, abs=2e-16)
    obs = r.observable_response(Z, np.ones(p), p, g=0.5, t=0.4)
    assert obs["full_derivative"] == obs["retained_derivative"] == obs["discarded_derivative"] == 0
    assert obs["first_order_observable"] == pytest.approx(np.vdot(Z, Z).real)
    assert obs["squared_approximate_field_observable"] == pytest.approx(np.vdot(Z, Z).real+0.25*np.vdot(a["psi1"], a["psi1"]).real)
    assert obs["squared_approximate_field_observable"] > obs["first_order_observable"]
    assert not obs["renormalized"] and not obs["densities_clipped"]


def test_bounds_formula_scaling_and_structural_cases():
    N, g, t, f = 1.3, 0.7, -0.4, 2.3
    b = r.finite_time_bounds(N, g, t, f)
    assert b["free_state_error_bound"] == pytest.approx(g*N**1.5*abs(t))
    assert b["state_remainder_bound"] == pytest.approx(1.5*g*g*N**2.5*t*t)
    assert b["relative_state_remainder_bound"] == pytest.approx(1.5*(g*N*t)**2)
    assert r.finite_time_bounds(0, g, t)["relative_state_remainder_bound"] is None
    assert b["observable_remainder_bound"] == pytest.approx(4*f*g*g*N**3*t*t)
    assert b["full_vs_galerkin_observable_remainder_bound"] == 2*b["observable_remainder_bound"]
    assert not b["includes_arithmetic_error"] and not b["includes_solver_error"]
    assert r.finite_time_bounds(0, g, t)["status"] == "vacuum_exact"
    for args in ((N, 0, t), (N, g, 0), (0, g, t)):
        b = r.finite_time_bounds(*args)
        assert all(b[key] == 0 for key in ("free_state_error_bound", "state_remainder_bound", "observable_remainder_bound"))
    assert r.finite_time_bounds(N, g, t, 0)["observable_remainder_bound"] == 0
    for z, time in ((np.zeros(3), 0.4), (Z, 0.0)):
        a = r.first_order_response(z, 7, t=time)
        np.testing.assert_array_equal(a["psi1"], np.zeros(7))
    free = r.observable_response(Z, np.arange(7), 7, g=0)
    assert free["full_correction"] == 0 and free["bounds"]["observable_remainder_bound"] == 0


def test_quartic_observable_scaling_and_source_linearity():
    source = np.sin(4*np.pi*np.arange(7)/7)
    a = r.observable_response(Z, source, 7)
    scaled = r.observable_response(1.2*Z, 2*source, 7)
    for key in ("full_derivative", "retained_derivative", "discarded_derivative"):
        assert scaled[key] == pytest.approx(2*1.2**4*a[key], rel=1e-12)


@pytest.mark.parametrize("p", [4, 0, -1, 10**20])
def test_invalid_ring_values(p):
    with pytest.raises(ValueError):
        r.cubic_channels(p)


@pytest.mark.parametrize("p", [True, 7.0, "7"])
def test_invalid_ring_types(p):
    with pytest.raises(TypeError):
        r.cubic_channels(p)


@pytest.mark.parametrize("C", [0, -1, float("nan"), float("inf"), 1e-320, 1e308])
def test_invalid_or_unresolved_C(C):
    with pytest.raises(ValueError):
        r.first_order_response(Z, 7, C=C)


@pytest.mark.parametrize("z", [[1, 2], [[1, 2, 3]], [1, 2, np.inf], [1, 2, np.nan], [1e-200, 0, 0], [1e200, 0, 0], [1, 1e-200, 0], [1e-110, 0, 0]])
def test_invalid_amplitudes_and_unresolved_intermediates(z):
    with pytest.raises(ValueError):
        r.first_order_response(z, 7)


@pytest.mark.parametrize("g", [-1, np.inf, np.nan, 1e-320, 1e308])
def test_invalid_or_unresolved_g(g):
    with pytest.raises(ValueError):
        r.observable_response(Z, np.ones(7), 7, g=g)


@pytest.mark.parametrize("t", [np.inf, np.nan, 1e100, 1e-320])
def test_invalid_or_unresolved_time(t):
    with pytest.raises(ValueError):
        r.first_order_response(Z, 7, t=t)


def test_invalid_source_and_bound_inputs():
    for source in ([1, 2], [np.nan]*7, [np.inf]*7, [1e-320]*7):
        with pytest.raises(ValueError):
            r.observable_response(Z, source, 7)
    with pytest.raises(TypeError):
        r.observable_response(Z, np.ones(7, complex), 7)
    for args in ((-1, 1, 1), (1, -1, 1), (1, 1, np.nan), (1, 1, 1, -1), (1e-100, 1e-100, 1e-100)):
        with pytest.raises(ValueError):
            r.finite_time_bounds(*args)
    with pytest.raises(ValueError):
        r.duhamel_kernel(1e-200, 1e-200)
    with pytest.raises(ValueError):
        r.duhamel_kernel(1e100, 1e100)


def test_strict_frozen_json_arithmetic_and_analytic_status_separate():
    demo = r.demonstration()
    json.dumps(demo, allow_nan=False)
    assert demo == r.demonstration()
    assert demo["parameters"] == {"p": 7, "C": 1, "g": 0.5, "N": 1, "t": 0.1}
    assert demo["physical_masses"] is None and demo["physical_mixing"] is None
    witness = demo["witness"]
    for key in ("retained_derivative", "discarded_derivative", "full_derivative"):
        assert witness["response"][key] == pytest.approx(witness["analytic_coefficients"][key], abs=3e-17)
    response = witness["response"]
    assert response["arithmetic"]["discarded"]["status"] == "resolved_nonzero"
    assert response["arithmetic"]["full"]["status"] == "resolved_nonzero"
    assert not response["full_correction_exceeds_analytic_bound"]
    assert not response["omitted_correction_exceeds_two_trajectory_bound"]
    assert not response["arithmetic"]["rigorous_roundoff_certificate"]
    assert abs(response["full_correction"]) < abs(response["discarded_correction"])


def test_long_time_phase_sensitive_cancellation_is_not_resolved():
    # Independent review's 60-digit evaluation is -18.70894766214145, while
    # binary64 phase reduction can even flip this small secular difference's
    # sign. The public raw value is not a high-precision guarantee: its status
    # must warn rather than call the inaccurate residual resolved_nonzero.
    p, time = 7, 417198883.39213
    source = np.cos(2*np.pi*np.arange(p)/p)
    response = r.observable_response([1, 2, 3], source, p, g=0, t=time)
    arithmetic = response["arithmetic"]["full"]
    assert arithmetic["status"] == "unresolved_cancellation"
    assert arithmetic["resolved_value"] is None
    assert arithmetic["arithmetic_warning_scale"] > abs(arithmetic["raw_value"]-(-18.70894766214145))
    assert response["bounds"]["observable_remainder_bound"] == 0  # g=0, separate issue
    first = r.first_order_response([1, 2, 3], p, t=time)
    assert first["arithmetic"]["phase_sensitive"]
    assert first["arithmetic"]["maximum_phase"] > 1e8


def test_cli_json_from_unrelated_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1]/"scripts"/"demo_substrate_nonlinear_response.py"
    completed = subprocess.run([sys.executable, str(script), "--json"], cwd=tmp_path, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout) == json.loads(json.dumps(r.demonstration()))
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_cli_text_from_unrelated_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1]/"scripts"/"demo_substrate_nonlinear_response.py"
    completed = subprocess.run([sys.executable, str(script)], cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "conditional" in completed.stdout.lower()
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []
