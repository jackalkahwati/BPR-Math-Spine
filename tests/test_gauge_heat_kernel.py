"""Exact finite-matrix checks for the new central model, with no sampling."""
import ast
import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from bpr import gauge_heat_kernel as gh
from bpr.nonabelian_gauge_sector import (
    ALLOWED_CLASSES, character_table, conjugacy_classes, elements, inv, mul,
)


@pytest.fixture(params=ALLOWED_CLASSES)
def n(request):
    return request.param


def _conjugate(g, h, n):
    return mul(mul(g, h, n), inv(g, n), n)


def test_regular_actions_and_endpoint_conventions(n):
    els = elements(n)
    eye = np.eye(len(els))
    for g in els:
        left, right = gh.left_regular(n, g), gh.right_regular(n, g)
        assert_allclose(left.T @ left, eye, atol=1e-14)
        assert_allclose(right.T @ right, eye, atol=1e-14)
        for j, h in enumerate(els):
            assert np.argmax(left[:, j]) == els.index(mul(g, h, n))
            assert np.argmax(right[:, j]) == els.index(mul(h, inv(g, n), n))
        for h in ((1, 1), (0, -1)):
            assert_allclose(left @ gh.left_regular(n, h), gh.left_regular(n, mul(g, h, n)))
            assert_allclose(right @ gh.right_regular(n, h), gh.right_regular(n, mul(g, h, n)))
            assert_allclose(left @ gh.right_regular(n, h), gh.right_regular(n, h) @ left)


def test_original_negative_control_and_exact_conjugacy_twirl(n):
    els = elements(n)
    original = (3 * np.eye(2 * n) - gh.left_regular(n, (1, 1))
                - gh.left_regular(n, inv((1, 1), n)) - gh.left_regular(n, (0, -1)))
    rotation = gh.left_regular(n, (1, 1))
    assert np.linalg.norm(original @ rotation - rotation @ original) > 1
    twirl = sum(gh.left_regular(n, g) @ original @ gh.left_regular(n, g).T
                for g in els) / len(els)
    central = gh.central_laplacian(n)
    assert_allclose(central, twirl, atol=2e-14)
    for g in els:
        for endpoint in (gh.left_regular(n, g), gh.right_regular(n, g)):
            assert_allclose(central @ endpoint, endpoint @ central, atol=2e-14)
    cl = gh.reflection_class(n)
    assert len(cl) == (n if n % 2 else n // 2)
    assert set(cl) == {_conjugate(g, (0, -1), n) for g in els}
    if n % 2 == 0:
        assert all(g[0] % 2 == 0 for g in cl)
        wrong = (3 * np.eye(2 * n) - gh.left_regular(n, (1, 1))
                 - gh.left_regular(n, inv((1, 1), n))
                 - sum(gh.left_regular(n, (k, -1)) for k in range(n)) / n)
        assert np.linalg.norm(wrong - central) > 1


def test_psd_constant_mode_irrep_projectors_and_multiplicities(n):
    central = gh.central_laplacian(n)
    assert central.shape == (2 * n, 2 * n)
    assert central.shape[0] <= 24
    assert_allclose(central, central.T, atol=1e-14)
    values = np.linalg.eigvalsh(central)
    assert min(values) > -2e-14
    assert np.count_nonzero(np.abs(values) < 1e-12) == 1
    assert_allclose(central @ np.ones(2 * n), 0, atol=2e-14)
    spectrum = gh.electric_spectrum(n)
    assert tuple(spectrum) == tuple(character_table(n))
    expected, projectors = [], []
    for name, chi in character_table(n).items():
        data = spectrum[name]
        d = data["dimension"]
        projector = d / (2 * n) * sum(chi[inv(g, n)] * gh.left_regular(n, g)
                                      for g in elements(n))
        assert_allclose(projector @ projector, projector, atol=2e-14)
        assert_allclose(central @ projector, data["epsilon"] * projector, atol=2e-14)
        assert np.linalg.matrix_rank(projector, tol=1e-10) == d * d == data["multiplicity"]
        expected.extend([data["epsilon"]] * data["multiplicity"])
        projectors.append(projector)
    assert_allclose(sum(projectors), np.eye(2 * n), atol=2e-14)
    assert_allclose(values, sorted(expected), atol=2e-14)
    assert spectrum["A1_trivial"]["epsilon"] == 0
    assert spectrum["A2_sign"]["epsilon"] == 2
    if n % 2 == 0:
        assert spectrum["B1"]["epsilon"] == 4
        assert spectrum["B2"]["epsilon"] == 6


@pytest.mark.parametrize("dt", [0.0, 1e-7, 0.13, 1.0, 30.0])
def test_heat_character_matrix_agreement_and_probability(n, dt):
    lam = 1.3
    weights = gh.heat_kernel(n, lam, dt)
    expected = expm(-dt * gh.central_laplacian(n) / lam)
    assert_allclose(gh.heat_transfer(n, lam, dt), expected, atol=3e-14)
    assert_allclose(weights, expected[:, 0], atol=3e-14)
    assert_allclose(sum(weights), 1, atol=2e-14)
    assert min(weights) >= -2e-14
    els = elements(n)
    for g, value in zip(els, weights):
        assert_allclose(value, weights[els.index(inv(g, n))], atol=2e-14)
    for cl in conjugacy_classes(n):
        assert_allclose([weights[els.index(g)] for g in cl], weights[els.index(cl[0])], atol=2e-14)
    if dt == 0:
        assert_allclose(weights, np.eye(2 * n)[:, 0], atol=2e-14)


def test_heat_extreme_positive_lambda_and_zero_time(n):
    tiny = np.nextafter(0.0, 1.0)
    assert_allclose(gh.heat_kernel(n, tiny, 0), np.eye(2 * n)[:, 0], atol=0)
    assert_allclose(gh.heat_kernel(n, tiny, tiny), gh.heat_kernel(n, 1, 1), atol=2e-14)
    assert_allclose(gh.heat_kernel(n, tiny, 1), np.ones(2 * n) / (2 * n), atol=2e-14)


def test_heat_semigroup_as_group_convolution(n):
    lam, t1, t2 = 0.8, 0.17, 0.31
    els = elements(n)
    first, second = gh.heat_kernel(n, lam, t1), gh.heat_kernel(n, lam, t2)
    convolution = [sum(first[i] * second[els.index(mul(inv(h, n), g, n))]
                       for i, h in enumerate(els)) for g in els]
    combined = gh.heat_kernel(n, lam, t1 + t2)
    assert_allclose(convolution, combined, atol=2e-14)
    assert_allclose(gh.heat_transfer(n, lam, t1) @ gh.heat_transfer(n, lam, t2),
                    gh.heat_transfer(n, lam, t1 + t2), atol=2e-14)


def test_square_character_basis_four_link_factor_and_magnetic_operator(n):
    lam = 1.3
    square = gh.isolated_square(n, lam)
    names = square["irrep_names"]
    size = len(names)
    assert size <= 9
    assert_allclose(square["character_gram"], np.eye(size), atol=2e-14)
    epsilon = np.array([gh.electric_spectrum(n)[name]["epsilon"] for name in names])
    assert_allclose(square["electric"], 4 * np.diag(epsilon) / lam, atol=2e-14)
    table = character_table(n)
    chars = np.array([[table[name][g] for name in names] for g in elements(n)])
    potential = np.array([1 - table["E1"][g] / 2 for g in elements(n)])
    # Multiplication stays in the class-function subspace, not a projection loss.
    assert_allclose(chars @ square["magnetic"], potential[:, None] * chars, atol=2e-14)
    class_potentials = sorted(1 - table["E1"][cl[0]] / 2 for cl in conjugacy_classes(n))
    assert_allclose(np.linalg.eigvalsh(square["magnetic"]), class_potentials, atol=2e-14)
    assert min(np.linalg.eigvalsh(square["magnetic"])) > -2e-14
    assert_allclose(square["hamiltonian"], square["electric"] + lam * square["magnetic"])
    assert_allclose(square["hamiltonian"], square["hamiltonian"].T, atol=2e-14)
    assert min(square["energies"]) > 0


def test_four_link_gauss_reduction_by_direct_holonomy_evaluation(n):
    """Independent link-coordinate check without a four-link tensor matrix.

    Orient edges A->B, B->C, D->C, A->D. Holonomy based at A is
    g1 g2 g3^-1 g4^-1; impose every vertex action, not only global conjugation.
    """
    configurations = [
        ((0, 1), (0, 1), (0, 1), (0, 1)),
        ((1, 1), (2, -1), (3, 1), (0, -1)),
        ((n - 1, -1), (1, 1), (0, -1), (2, -1)),
        ((2, 1), (n - 1, 1), (1, 1), (3, 1)),
    ]
    endpoints = ((0, 1), (1, 2), (3, 2), (0, 3))

    def holonomy(links):
        return mul(mul(mul(links[0], links[1], n), inv(links[2], n), n),
                   inv(links[3], n), n)

    cl = gh.reflection_class(n)
    generators = [(1.0, (1, 1)), (1.0, inv((1, 1), n))]
    generators += [(1 / len(cl), g) for g in cl]
    spectrum = gh.electric_spectrum(n)
    for links in configurations:
        for chi_name, chi in character_table(n).items():
            base = chi[holonomy(links)]
            for vertex in range(4):
                for gauge in elements(n):
                    transformed = []
                    for link, (source, target) in zip(links, endpoints):
                        value = mul(gauge, link, n) if source == vertex else link
                        value = mul(value, inv(gauge, n), n) if target == vertex else value
                        transformed.append(value)
                    assert_allclose(chi[holonomy(transformed)], base, atol=2e-14)
            total = 0.0
            for link_index in range(4):
                shifted_sum = 0.0
                for weight, generator in generators:
                    shifted = list(links)
                    shifted[link_index] = mul(inv(generator, n), links[link_index], n)
                    shifted_sum += weight * chi[holonomy(shifted)]
                link_electric = 3 * base - shifted_sum
                assert_allclose(link_electric, spectrum[chi_name]["epsilon"] * base, atol=3e-14)
                total += link_electric
            assert_allclose(total, 4 * spectrum[chi_name]["epsilon"] * base, atol=1e-13)


def test_symmetric_square_transfer_and_second_order_log_convergence(n):
    lam, dt = 1.3, 0.08
    square = gh.isolated_square(n, lam)
    transfer = gh.isolated_square_transfer(n, lam, dt)
    half = expm(-dt * lam * square["magnetic"] / 2)
    assert_allclose(transfer, half @ expm(-dt * square["electric"]) @ half, atol=2e-14)
    assert_allclose(transfer, transfer.T, atol=2e-14)
    assert min(np.linalg.eigvalsh(transfer)) > 0
    assert_allclose(gh.isolated_square_transfer(n, lam, 0), np.eye(len(transfer)), atol=2e-14)
    diagnostics = gh.isolated_square_diagnostics(n, lam, dt)
    assert all(value > 0 for value in diagnostics["transfer_min_eigenvalues"])
    assert all(value < 1e-12 for value in diagnostics["transfer_hermiticity_errors"])
    assert diagnostics["hamiltonian_hermiticity_error"] < 1e-12
    assert all(1.95 < order < 2.05 for order in diagnostics["observed_orders"])
    errors = diagnostics["effective_hamiltonian_errors"]
    assert errors[2] < errors[1] < errors[0]
    assert_allclose(expm(-dt * gh.effective_hamiltonian(transfer, dt)), transfer, atol=2e-14)


def test_square_transfer_resolved_exponents_with_unrepresentable_hamiltonian(n):
    epsilon = np.array([data["epsilon"] for data in gh.electric_spectrum(n).values()])
    assert_allclose(gh.isolated_square_transfer(n, 1e-307, 0), np.eye(len(epsilon)), atol=0)
    assert_allclose(gh.isolated_square_transfer(n, 1e-307, 1e-307),
                    np.diag(np.exp(-4 * epsilon)), atol=2e-14)
    with pytest.raises(FloatingPointError, match="Hamiltonian exceeds floating-point range"):
        gh.isolated_square(n, 1e-307)


@pytest.mark.parametrize("lam", [1e9, 1e16, 1e18])
def test_large_lambda_rejects_unresolved_spectrum_and_preserves_transfer_contraction(n, lam):
    with pytest.raises(FloatingPointError, match="ground energy is numerically unresolved"):
        gh.isolated_square(n, lam)
    transfer = gh.isolated_square_transfer(n, lam, 1)
    assert np.linalg.norm(transfer, ord=2) <= 1 + 2e-14
    assert np.min(np.linalg.eigvalsh(transfer)) >= -2e-14
    # Long magnetic time projects onto identity holonomy. Its normalized
    # character coordinates are d_R/sqrt(|G|); the electric insertion persists.
    spectrum = gh.electric_spectrum(n)
    dimensions = np.array([data["dimension"] for data in spectrum.values()])
    epsilon = np.array([data["epsilon"] for data in spectrum.values()])
    identity_state = dimensions / np.sqrt(2 * n)
    projection = np.outer(identity_state, identity_state)
    expected = projection @ np.diag(np.exp(-4 * epsilon / lam)) @ projection
    assert_allclose(transfer, expected, atol=2e-14)


def test_wilson_transfer_is_normalized_but_not_heat_model(n):
    beta, lam, dt = 1.1, 1.3, 0.17
    comparison = gh.wilson_comparison(n, beta, lam, dt)
    wilson = comparison["wilson"]
    table, els = character_table(n), elements(n)
    direct = np.exp([beta * table["E1"][g] / 2 for g in els])
    assert_allclose(wilson["weights"], direct / sum(direct), atol=2e-14)
    assert_allclose(wilson["transfer"].sum(axis=0), 1, atol=2e-14)
    expected_spectrum = []
    for value, chi in zip(wilson["irrep_transfer_eigenvalues"], table.values()):
        expected_spectrum.extend([value] * int(chi[(0, 1)] ** 2))
    assert_allclose(np.linalg.eigvalsh(wilson["transfer"]), sorted(expected_spectrum), atol=2e-14)
    assert wilson["undefined_reason"] is None
    assert_allclose(np.exp(-dt * wilson["energies"]), wilson["irrep_transfer_eigenvalues"])
    assert comparison["transfer_difference_norm"] > 0.1
    assert not np.allclose(wilson["energies"], comparison["heat_energies"])
    # Even a common energy rescaling would not identify all irrep energies.
    ratios = wilson["energies"][1:] / comparison["heat_energies"][1:]
    assert np.ptp(ratios) > 0.1
    zero = gh.wilson_single_link(n, 0)
    assert_allclose(zero["weights"], np.ones(2 * n) / (2 * n))
    assert zero["energies"] is None
    assert "no logarithmic floor" in zero["undefined_reason"]


def test_wilson_comparison_resolves_transfer_when_energies_overflow(n):
    comparison = gh.wilson_comparison(n, 1.1, 1e-308, 1e-308)
    spectrum = gh.electric_spectrum(n)
    epsilon = np.array([data["epsilon"] for data in spectrum.values()])
    assert_allclose(comparison["heat_irrep_transfer_eigenvalues"], np.exp(-epsilon), atol=2e-14)
    assert comparison["heat_energies"] is None
    assert "exceed floating-point range" in comparison["heat_energies_undefined_reason"]
    assert comparison["wilson"]["energies"] is None
    assert "exceed floating-point range" in comparison["wilson"]["undefined_reason"]
    expected_regular = np.repeat(comparison["heat_irrep_transfer_eigenvalues"],
                                 [data["multiplicity"] for data in spectrum.values()])
    assert_allclose(np.linalg.eigvalsh(gh.heat_transfer(n, 1e-308, 1e-308)),
                    sorted(expected_regular), atol=2e-14)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf, 1j, True, "1"])
def test_invalid_positive_parameters(bad):
    for call in (
        lambda: gh.model_spec(5, bad), lambda: gh.heat_kernel(5, bad, 0.1),
        lambda: gh.isolated_square(5, bad), lambda: gh.isolated_square_transfer(5, bad, 0.1),
        lambda: gh.wilson_comparison(5, 1, bad, 0.1),
        lambda: gh.effective_hamiltonian(np.eye(2), bad),
        lambda: gh.isolated_square_diagnostics(5, 1, bad),
        lambda: gh.wilson_single_link(5, 1, bad),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, -np.inf, 1j, True, "1"])
def test_invalid_nonnegative_parameters(bad):
    for call in (lambda: gh.heat_kernel(5, 1, bad),
                 lambda: gh.isolated_square_transfer(5, 1, bad),
                 lambda: gh.wilson_single_link(5, bad)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [3, 4, 6, 5.0, True, "5", None])
def test_invalid_group(bad):
    for call in (lambda: gh.model_spec(bad, 1), lambda: gh.central_laplacian(bad),
                 lambda: gh.electric_spectrum(bad), lambda: gh.heat_kernel(bad, 1, 0.1),
                 lambda: gh.isolated_square(bad, 1), lambda: gh.wilson_single_link(bad, 1)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [(5, 1), (-1, 1), (0, 0), (0, True), (0.5, 1), [0, 1], None])
def test_invalid_regular_element(bad):
    with pytest.raises(ValueError):
        gh.left_regular(5, bad)
    with pytest.raises(ValueError):
        gh.right_regular(5, bad)


def test_log_diagnostics_do_not_hide_numerical_failure():
    with pytest.raises(FloatingPointError, match="exceed floating-point range"):
        gh.effective_hamiltonian(np.diag([1.0, 0.1]), 1e-308)
    for matrix in (np.diag([1.0, -0.1]), np.diag([1.0, 0.0]), np.diag([1.0, 1e-30])):
        with pytest.raises(FloatingPointError, match="nonpositive or numerically unresolved"):
            gh.effective_hamiltonian(matrix, 1)
    for matrix in (np.array([[1, 0.1], [0, 1]]), np.full((2, 2), np.nan), np.ones((2, 3)),
                   np.eye(25), np.empty((0, 0))):
        with pytest.raises(ValueError):
            gh.effective_hamiltonian(matrix, 1)
    # Underflow cannot be converted into invented finite energies.
    with pytest.raises(FloatingPointError):
        gh.isolated_square_diagnostics(5, 1, 1000)


def test_model_provenance_and_dependency_guards(n):
    spec = gh.model_spec(n, 1.3)
    assert spec["model_id"] == gh.MODEL_ID != gh.WILSON_MODEL_ID
    assert spec["units"] == "dimensionless"
    assert "E0 * H_dimensionless" in spec["physical_unit_restoration"]
    assert "dt = E0 * time" in spec["physical_unit_restoration"]
    assert "not predicted" in spec["energy_scale_status"]
    assert "different action" in spec["wilson_mc_relation"]
    assert "four distinct links" in spec["square_geometry"]
    assert "not assumed or calibrated" == spec["beta_lambda_mapping"]
    assert "isolated-square toy energies" in spec["energy_interpretation"]
    assert "No glueball masses" in " ".join(spec["limitations"])
    source = inspect.getsource(gh)
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert "nonabelian_gauge_sector" in imports
    for forbidden in ("glueball_benchmark", "experimental_data", "physical_lambda", "gauge_phase_mc",
                      "gauge_mc_fast", "gauge_dynamics_m2_m3"):
        assert forbidden not in imports
    assert "np.random" not in source
    assert "np.kron" not in source
