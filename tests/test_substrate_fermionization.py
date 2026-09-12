"""Independent finite-matrix identities, not physical benchmarks or fits."""
import itertools
import json
from math import comb
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bpr import substrate_fermionization as f
from bpr.zp_selection_principle import _small_hilbert


def tensor(operators):
    result = np.array([[1.0]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def tensor_bosons(L, d, C, g):
    """Full local tensor construction independent of occupation hopping loops."""
    a = np.diag(np.sqrt(np.arange(1, d)), k=1)
    operators = [tensor([a if site == x else np.eye(d) for site in range(L)]) for x in range(L)]
    numbers = [operator.T @ operator for operator in operators]
    dim = d ** L
    number = sum(numbers)
    H = sum(-C * (operators[x].T @ operators[(x + 1) % L]
                  + operators[(x + 1) % L].T @ operators[x]) for x in range(L))
    H += g / 2 * sum(n @ (n - np.eye(dim)) for n in numbers)
    return H, number


def tensor_index(state, d):
    return sum(n * d ** (len(state) - 1 - x) for x, n in enumerate(state))


def bit_index(state):
    return sum(n * 2 ** x for x, n in enumerate(state))


@pytest.mark.parametrize("N", [0, 1, 2])
@pytest.mark.parametrize("C,g", [(1.0, 0.7), (1.3, 40.0), (0.2, 0.0)])
def test_three_site_independent_tensor_and_old_oracle(N, C, g):
    model = f.fixed_number_model(3, N, C, g)
    expected, number = tensor_bosons(3, 3, C, g)
    old, old_number, _ = _small_hilbert(C, g)
    np.testing.assert_allclose(expected, old, atol=4e-14)
    np.testing.assert_allclose(number, old_number, atol=2e-15)
    indices = [tensor_index(state, 3) for state in model.basis]
    np.testing.assert_allclose(model.H, expected[np.ix_(indices, indices)], atol=4e-14)
    np.testing.assert_allclose(model.H, old[np.ix_(indices, indices)], atol=4e-14)
    np.testing.assert_allclose(expected @ number, number @ expected, atol=4e-14)
    assert np.allclose(np.diag(number)[indices], N)


@pytest.mark.parametrize("L,N", [(3, 0), (3, 1), (3, 2), (3, 3), (4, 2), (4, 4), (5, 2), (6, 3), (12, 0), (12, 1)])
def test_complete_basis_dimensions_blocks_and_boson_factors(L, N):
    model = f.fixed_number_model(L, N, C=1.7, g=2.3)
    assert len(model.basis) == comb(L + N - 1, N)
    assert len(set(model.basis)) == len(model.basis)
    assert model.basis == tuple(sorted(model.basis))
    assert all(sum(state) == N and min(state) >= 0 for state in model.basis)
    assert len(model.p_indices) == comb(L, N)
    assert len(model.q_indices) == len(model.basis) - comb(L, N)
    np.testing.assert_array_equal(model.H, model.H.T)
    np.testing.assert_array_equal(model.V, model.V.T)
    np.testing.assert_allclose(model.H, model.V + model.g * np.diag(model.D))
    np.testing.assert_array_equal(model.PHP, model.H[np.ix_(model.p_indices, model.p_indices)])
    np.testing.assert_array_equal(model.B, model.H[np.ix_(model.p_indices, model.q_indices)])
    np.testing.assert_array_equal(model.QHQ, model.H[np.ix_(model.q_indices, model.q_indices)])
    assert np.linalg.norm(model.V, 2) <= 2 * model.C * N + 2e-13
    for col, state in enumerate(model.basis):
        for x in range(L):
            y = (x + 1) % L
            if state[y]:
                moved = list(state)
                moved[y] -= 1
                moved[x] += 1
                row = model.basis.index(tuple(moved))
                assert model.H[row, col] == pytest.approx(-model.C * np.sqrt(state[y] * (state[x] + 1)))
    if N:
        assert (N,) + (0,) * (L - 1) in model.basis


def test_N3_full_basis_not_old_cutoff_toy():
    model = f.fixed_number_model(3, 3)
    assert len(model.basis) == 10
    assert all(state in model.basis for state in [(3, 0, 0), (0, 3, 0), (0, 0, 3)])
    _, old_number, _ = _small_hilbert()
    assert np.count_nonzero(np.isclose(np.diag(old_number), 3)) == 7
    expected, _ = tensor_bosons(3, 4, model.C, model.g)
    indices = [tensor_index(state, 4) for state in model.basis]
    np.testing.assert_allclose(model.H, expected[np.ix_(indices, indices)], atol=3e-15)


@pytest.mark.parametrize("L", [3, 4, 5])
def test_full_binary_tensor_JW_CAR_charge_and_nonlocality(L):
    ops = f.binary_operators(L)
    b, c, n = (ops[k] for k in ("b", "c", "n"))
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]])
    Z = np.diag([1.0, -1.0])
    identity = np.eye(2 ** L)
    for x in range(L):
        # Tensor factors reversed so site x has bit weight 2**x.
        bx = tensor([lowering if site == x else np.eye(2) for site in reversed(range(L))])
        cx = tensor([lowering if site == x else Z if site < x else np.eye(2) for site in reversed(range(L))])
        np.testing.assert_array_equal(b[x], bx)
        np.testing.assert_array_equal(c[x], cx)
        np.testing.assert_array_equal(c[x].T @ c[x], n[x])
        np.testing.assert_array_equal(ops["number"] @ c[x].T - c[x].T @ ops["number"], c[x].T)
        for y in range(L):
            np.testing.assert_array_equal(c[x] @ c[y] + c[y] @ c[x], np.zeros_like(identity))
            np.testing.assert_array_equal(c[x] @ c[y].T + c[y].T @ c[x], identity if x == y else np.zeros_like(identity))
            if x != y:
                np.testing.assert_array_equal(b[x] @ b[y], b[y] @ b[x])
            bilinear = c[x].T @ c[y]
            np.testing.assert_array_equal(ops["number"] @ bilinear, bilinear @ ops["number"])
    # c at the right end does not commute with an upstream local b creator.
    assert np.linalg.norm(c[-1] @ b[0].T - b[0].T @ c[-1], 2) == 2
    assert f.algebra_diagnostics(L) == {key: 0.0 for key in f.algebra_diagnostics(L)}
    # Naive fixed-number compression of annihilation loses CAR entirely.
    indices = [bit for bit in range(2 ** L) if bin(bit).count("1") == 1]
    compressed = c[0][np.ix_(indices, indices)]
    np.testing.assert_array_equal(compressed, np.zeros((L, L)))


@pytest.mark.parametrize("L,N", [(3, 0), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 2), (5, 3), (5, 5)])
def test_periodic_boundary_parity_tensor_matrix_and_spectrum(L, N):
    model = f.fixed_number_model(L, N, C=1.2, g=40)
    ops = f.binary_operators(L)
    c = ops["c"]
    twist = 1 if N % 2 else -1
    assert f.boundary_twist(N) == twist
    fermion = sum(-model.C * (c[x].T @ c[x + 1] + c[x + 1].T @ c[x]) for x in range(L - 1))
    fermion -= model.C * twist * (c[-1].T @ c[0] + c[0].T @ c[-1])
    indices = [bit_index(state) for state in model.hard_core_basis]
    tensor_fixed = fermion[np.ix_(indices, indices)]
    np.testing.assert_array_equal(tensor_fixed, model.PHP)
    np.testing.assert_array_equal(f.fermionic_hopping(L, N, C=model.C), tensor_fixed)
    # Independent single-particle momenta; antiperiodic momenta for even N.
    shift = 0 if twist == 1 else 0.5
    levels = [-2 * model.C * np.cos(2 * np.pi * (k + shift) / L) for k in range(L)]
    expected = sorted(sum(levels[k] for k in occupied) for occupied in itertools.combinations(range(L), N))
    np.testing.assert_allclose(np.linalg.eigvalsh(model.PHP), expected, atol=7e-15)


@pytest.mark.parametrize("L,N", [(3, 1), (3, 2), (4, 2), (5, 2), (5, 3)])
def test_wrong_twist_control_changes_spectrum(L, N):
    correct = f.fermionic_hopping(L, N)
    wrong = f.fermionic_hopping(L, N, twist=-f.boundary_twist(N))
    assert not np.allclose(correct, wrong)
    assert not np.allclose(np.linalg.eigvalsh(correct), np.linalg.eigvalsh(wrong))


@pytest.mark.parametrize("C,g", [(1.0, 40.0), (0.4, 7.0), (1.7, 33.0)])
def test_virtual_doublon_coefficients_derived_from_distinct_paths(C, g):
    model = f.fixed_number_model(5, 2, C, g)
    second = f.second_order_effective(model)
    delta = second["correction"]
    initial = model.hard_core_basis.index((1, 1, 0, 0, 0))
    final = model.hard_core_basis.index((0, 1, 1, 0, 0))
    assert delta[initial, initial] == pytest.approx(-4 * C ** 2 / g)
    assert delta[final, initial] == pytest.approx(-2 * C ** 2 / g)
    paths = [(model.basis[q], model.B[initial, j], model.B[final, j]) for j, q in enumerate(model.q_indices)]
    assert {state for state, a, b in paths if a != 0} == {(2, 0, 0, 0, 0), (0, 2, 0, 0, 0)}
    assert {state for state, a, b in paths if a != 0 and b != 0} == {(0, 2, 0, 0, 0)}
    for _, a, b in paths:
        if a != 0:
            assert a == pytest.approx(-np.sqrt(2) * C)
        if b != 0:
            assert b == pytest.approx(-np.sqrt(2) * C)
    full = f.fixed_number_model(3, 3, C, g)
    assert full.PHP.shape == (1, 1)
    assert full.PHP[0, 0] == 0
    reachable = [full.basis[q] for j, q in enumerate(full.q_indices) if full.B[0, j] != 0]
    assert len(reachable) == 6
    assert set(reachable) == set(itertools.permutations((2, 1, 0)))
    assert f.second_order_effective(full)["correction"][0, 0] == pytest.approx(-12 * C ** 2 / g)
    np.testing.assert_allclose(delta, -model.B @ np.diag(1 / (g * model.D[model.q_indices])) @ model.B.T)
    assert np.max(np.linalg.eigvalsh(delta)) < 2e-14
    assert np.linalg.norm(model.B) > 0  # Compression is not invariant dynamics.


@pytest.mark.parametrize("L,N,C,g", [(3, 2, 1.0, 40.0), (3, 3, 1.0, 40.0), (4, 2, 0.7, 20.0), (5, 3, 1.1, 55.0)])
def test_exact_Schur_eigenstate_elimination_leakage_and_bounds(L, N, C, g):
    model = f.fixed_number_model(L, N, C, g)
    cert = f.validity_certificate(model)
    assert cert["status"] == "certified_separated_cluster"
    assert cert["available"] and not cert["q_empty"]
    assert cert["roundoff_included"] is False
    B_norm = np.linalg.norm(model.B, 2)
    v = 2 * C * N
    assert cert["v"] == v
    assert cert["B_norm"] == pytest.approx(B_norm)
    assert cert["leakage_ratio_bound"] == pytest.approx(B_norm / (g - 2 * v))
    assert cert["schur_shift_bound"] == pytest.approx(B_norm ** 2 / (g - 2 * v))
    assert cert["second_order_remainder_bound"] == pytest.approx(2 * v * B_norm ** 2 / (g * (g - 2 * v)))
    values, vectors = np.linalg.eigh(model.H)
    d = len(model.p_indices)
    assert values[:d].min() >= -v - 2e-13
    assert values[:d].max() <= v + 2e-13
    assert values[d:].min() >= g - v - 2e-13
    H2 = f.second_order_effective(model)
    for j, E in enumerate(values[:d]):
        p = vectors[model.p_indices, j]
        q = vectors[model.q_indices, j]
        eliminated = -np.linalg.solve(model.QHQ - E * np.eye(len(q)), model.B.T @ p)
        np.testing.assert_allclose(q, eliminated, atol=2e-14)
        np.testing.assert_allclose(f.schur_operator(model, E) @ p, E * p, atol=8e-14)
        assert np.linalg.norm(q) <= cert["leakage_ratio_bound"] * np.linalg.norm(p) + 2e-14
    for E in (-v, -0.37 * v, 0.0, v):
        expected = -model.B @ np.linalg.solve(model.QHQ - E * np.eye(len(model.q_indices)), model.B.T)
        delta = f.schur_correction(model, E)
        np.testing.assert_allclose(delta, expected, atol=2e-15)
        np.testing.assert_allclose(f.schur_remainder(model, E), delta - H2["correction"], atol=2e-15)
        assert np.linalg.norm(delta, 2) <= cert["schur_shift_bound"] + 1e-14
        assert np.linalg.norm(delta - H2["correction"], 2) <= cert["second_order_remainder_bound"] + 1e-14
        # Schur factorization identity for arbitrary retained vector, not just eigenstates.
        p = np.arange(1, d + 1, dtype=float) / d
        q = -np.linalg.solve(model.QHQ - E * np.eye(len(model.q_indices)), model.B.T @ p)
        state = np.zeros(len(model.basis))
        state[model.p_indices], state[model.q_indices] = p, q
        residual = (model.H - E * np.eye(len(state))) @ state
        np.testing.assert_allclose(residual[model.q_indices], 0, atol=2e-14)
        np.testing.assert_allclose(residual[model.p_indices], (f.schur_operator(model, E) - E * np.eye(d)) @ p, atol=2e-14)


@pytest.mark.parametrize("N", [0, 1])
@pytest.mark.parametrize("g", [0.0, 0.7, 40.0, 1e200])
def test_structural_Qempty_no_placeholder_denominators(N, g):
    model = f.fixed_number_model(5, N, g=g)
    assert model.B.shape == (comb(5, N), 0)
    assert model.QHQ.shape == (0, 0)
    cert = f.validity_certificate(model)
    assert cert["status"] == "exact_q_empty"
    assert cert["available"] and cert["q_empty"]
    assert cert["margin"] is None
    assert cert["high_cluster_lower_bound"] is None
    for field in ("leakage_ratio_bound", "schur_shift_bound", "second_order_remainder_bound"):
        assert cert[field] == 0
    np.testing.assert_array_equal(f.schur_operator(model, 0.0), model.H)
    np.testing.assert_array_equal(f.schur_correction(model, -2.0), np.zeros_like(model.H))
    second = f.second_order_effective(model)
    np.testing.assert_array_equal(second["hamiltonian"], model.H)
    assert second["scale"] == 0
    assert second["total_addition_resolved"]


@pytest.mark.parametrize("g", [0.0, 0.7, 7.0, 8.0])
def test_weak_or_equal_repulsion_is_uncertified_not_disproved(g):
    model = f.fixed_number_model(5, 2, g=g)
    cert = f.validity_certificate(model)
    assert cert["status"] == "unavailable"
    assert not cert["available"]
    assert cert["leakage_ratio_bound"] is None
    assert cert["schur_shift_bound"] is None
    assert cert["second_order_remainder_bound"] is None
    if g == 0:
        with pytest.raises(ValueError, match="g>0"):
            f.second_order_effective(model)
    else:
        assert f.second_order_effective(model)["correction"].shape == model.PHP.shape


@pytest.mark.parametrize("L,q", [(3, 3), (5, 5), (5, 7), (6, 2), (7, 3), (8, 4), (4, 10 ** 50)])
def test_neutrality_counts_independent_binary_enumeration(L, q):
    result = f.neutrality_counts(L, q)
    states = list(itertools.product((0, 1), repeat=L))
    counts = {N: sum(sum(state) == N for state in states) for N in range(L + 1) if N % q == 0}
    assert result["sectors"] == [{"N": N, "dimension": d} for N, d in counts.items()]
    assert result["allowed_dimension"] == sum(sum(state) % q == 0 for state in states)
    assert result["assumed"] is True
    assert result["dynamical_confinement_derived"] is False
    if L == q:
        assert result["sectors"] == [{"N": 0, "dimension": 1}, {"N": L, "dimension": 1}]
    if q > L:
        assert result["sectors"] == [{"N": 0, "dimension": 1}]


@pytest.mark.parametrize("args,error", [
    ((2, 1), ValueError), ((13, 0), ValueError), ((100000000, 0), ValueError),
    ((5, -1), ValueError), ((5, 6), ValueError), ((7, 7), ValueError),
    ((True, 1), TypeError), ((5.0, 1), TypeError), ((5, 1.0), TypeError),
    ((5, False), TypeError), ((3, 1, 0), ValueError), ((3, 1, -1), ValueError),
    ((3, 1, 1, -1), ValueError), ((3, 1, np.inf), ValueError),
    ((3, 1, np.nan), ValueError), ((3, 1, 1, np.inf), ValueError),
    ((3, 1, 1, np.nan), ValueError), ((3, 1, True), TypeError),
    ((3, 1, 1, False), TypeError), ((3, 1, 1j), TypeError),
    ((3, 1, "1"), TypeError), ((3, 1, 1, "2"), TypeError),
    ((3, 1, 1e-320), ValueError), ((3, 1, 1, 1e-320), ValueError),
    ((3, 3, 1, 1e308), ValueError), ((3, 2, 1e308, 1), ValueError),
])
def test_invalid_and_unresolved_construction(args, error):
    with pytest.raises(error):
        f.fixed_number_model(*args)


def test_caps_before_enumeration_or_allocation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("allocation/enumeration attempted")
    monkeypatch.setattr(f, "_occupations", fail)
    monkeypatch.setattr(np, "zeros", fail)
    with pytest.raises(ValueError, match="dimension"):
        f.fixed_number_model(7, 7)
    with pytest.raises(ValueError, match="L"):
        f.fixed_number_model(10000000, 0)
    with pytest.raises(ValueError, match="binary dimension"):
        f.binary_operators(9)
    with pytest.raises(ValueError, match="L"):
        f.binary_operators(10000000)


@pytest.mark.parametrize("q,error", [(1, ValueError), (0, ValueError), (-2, ValueError), (2.0, TypeError), (True, TypeError)])
def test_invalid_neutrality_modulus(q, error):
    with pytest.raises(error):
        f.neutrality_counts(5, q)


@pytest.mark.parametrize("twist,error", [(0, ValueError), (2, ValueError), (1.0, TypeError), (True, TypeError)])
def test_invalid_twist(twist, error):
    with pytest.raises(error):
        f.fermionic_hopping(5, 2, twist=twist)


@pytest.mark.parametrize("E,error", [(np.inf, ValueError), (np.nan, ValueError), (1e-320, ValueError), (True, TypeError), (1j, TypeError)])
def test_invalid_Schur_energy(E, error):
    with pytest.raises(error):
        f.schur_operator(f.fixed_number_model(3, 2), E)


def test_singular_resolvent_rejected():
    model = f.fixed_number_model(5, 2, g=40)
    # N=2: QHQ = g I (no Q-to-Q hopping).
    with pytest.raises(ValueError, match="resolvent"):
        f.schur_operator(model, 40)


@pytest.mark.parametrize("factor", [np.nextafter(1.0, np.inf), 1 + 8 * np.finfo(float).eps])
def test_unresolved_positive_separation_rejected(factor):
    model = f.fixed_number_model(5, 2, g=8 * factor)
    with pytest.raises(ValueError, match="positive separation margin"):
        f.validity_certificate(model)


def test_well_resolved_small_positive_margin_allowed():
    cert = f.validity_certificate(f.fixed_number_model(5, 2, g=8 * (1 + 1e-10)))
    assert cert["available"]
    assert cert["leakage_ratio_bound"] > 1e9  # Sufficient does not imply useful.


@pytest.mark.parametrize("C,g", [(1e-150, 40e-150), (1e150, 40e150)])
def test_uniform_rescaling_does_not_square_extreme_energy_scales(C, g):
    model = f.fixed_number_model(3, 3, C=C, g=g)
    cert = f.validity_certificate(model)
    reference = f.validity_certificate(f.fixed_number_model(3, 3, C=1, g=40))
    assert cert["available"]
    assert cert["leakage_ratio_bound"] == pytest.approx(reference["leakage_ratio_bound"])
    assert cert["schur_shift_bound"] / C == pytest.approx(reference["schur_shift_bound"])
    second = f.second_order_effective(model)
    assert second["correction"][0, 0] / C == pytest.approx(-0.3)
    schur = f.schur_correction(model, 0)
    expected = f.schur_correction(f.fixed_number_model(3, 3, g=40), 0)
    np.testing.assert_allclose(schur / C, expected, atol=1e-15)


def test_unresolved_underflow_certificate_not_false_zero():
    model = f.fixed_number_model(3, 2, C=1e-200, g=1)
    with pytest.raises(ValueError, match="unresolved"):
        f.validity_certificate(model)
    with pytest.raises(ValueError, match="unresolved"):
        f.second_order_effective(model)
    with pytest.raises(ValueError, match="unresolved"):
        f.schur_operator(model, 0)


def test_separately_resolved_correction_can_be_lost_in_total():
    model = f.fixed_number_model(3, 2, C=1, g=1e18)
    second = f.second_order_effective(model)
    assert second["scale"] == pytest.approx(1e-18, abs=0)
    assert np.linalg.norm(second["correction"], 2) > 0
    assert second["total_addition_resolved"] is False
    assert np.any((second["correction"] != 0) & (second["hamiltonian"] == model.PHP))
    assert f.validity_certificate(model)["available"]


@pytest.mark.parametrize("g", [1e16, 1e18, 1e20])
def test_large_g_resolvent_remainder_survives_cancellation(g):
    model = f.fixed_number_model(3, 3, C=1, g=g)
    remainder = f.schur_remainder(model, 0)
    # Leading large-g coefficient independently from B D^-1 W D^-1 B^T.
    B = model.B
    Dinv = np.diag(1 / model.D[model.q_indices])
    W = model.V[np.ix_(model.q_indices, model.q_indices)]
    coefficient = (B @ Dinv @ W @ Dinv @ B.T)[0, 0]
    assert coefficient == pytest.approx(-36)
    assert remainder[0, 0] != 0
    assert remainder[0, 0] == pytest.approx(-36 / g / g, rel=1e-13, abs=0)
    assert f.validity_certificate(model)["available"]
    with pytest.raises(ValueError, match="report eigensystem"):
        f.case_report(3, 3, 1, g)


def test_structural_zero_remainder_and_Qempty_at_every_g():
    for g in (0, 40, 1e20):
        model = f.fixed_number_model(3, 1, g=g)
        np.testing.assert_array_equal(f.schur_remainder(model, 0), np.zeros_like(model.PHP))
    model = f.fixed_number_model(5, 2, g=1e20)
    np.testing.assert_array_equal(f.schur_remainder(model, 0), np.zeros_like(model.PHP))
    with pytest.raises(ValueError, match="g>0"):
        f.schur_remainder(f.fixed_number_model(3, 2, g=0), 0)


def test_numpy_integer_parameters_are_native_JSON():
    report = f.case_report(np.int64(3), np.int64(1), np.float64(1), np.float64(0), np.int64(3))
    json.dumps(report, allow_nan=False)
    assert type(report["parameters"]["q"]) is int
    assert type(report["neutrality"]["comparison_sector_neutral"]) is bool


def test_zero_g_and_Qempty_reports_are_strict_JSON():
    for N in (0, 1, 2):
        report = f.case_report(3, N, 1, 0)
        json.dumps(report, allow_nan=False)
        if N < 2:
            assert report["certificate"]["status"] == "exact_q_empty"
        else:
            assert report["certificate"]["status"] == "unavailable"
            assert report["correction"] is None
            assert report["spectra"]["second_order"] is None


def test_frozen_report_schema_witnesses_and_scientific_limits():
    report = f.demonstration_report()
    json.dumps(report, allow_nan=False)
    expected_controls = [{"L": 5, "N": 2, "C": 1.0, "g": 40.0, "q": 5},
                         {"L": 5, "N": 2, "C": 1.0, "g": 0.7, "q": 5},
                         {"L": 3, "N": 3, "C": 1.0, "g": 40.0, "q": 3}]
    assert report["frozen_controls"] == expected_controls
    assert [case["parameters"] for case in report["cases"]] == expected_controls
    assert report["controls_frozen_before_evaluation"] is True
    assert report["empirical_calibration"] is False
    assert report["physical_predictions"] == {"masses": None, "mixing": None}
    assert [case["certificate"]["status"] for case in report["cases"]] == ["certified_separated_cluster", "unavailable", "certified_separated_cluster"]
    assert [case["neutrality"]["comparison_sector_neutral"] for case in report["cases"]] == [False, False, True]
    assert report["cases"][2]["dimensions"] == {"full": 10, "hard_core": 1, "discarded": 9, "binary": 8}
    assert [len(w["paths"]) for w in report["cases"][0]["correction"]["virtual_witnesses"]] == [2, 1]
    assert len(report["cases"][2]["correction"]["virtual_witnesses"][0]["paths"]) == 6
    text = " ".join(report["limitations"])
    for term in ("not physical", "state-selection assumption", "energy dependent", "not an exact autonomous", "roundoff", "not exactly free", "charge one"):
        assert term in text


@pytest.mark.parametrize("json_mode", [False, True])
def test_demo_strict_CLI_from_empty_tmp_writes_no_files(tmp_path, json_mode):
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_substrate_fermionization.py"
    command = [sys.executable, "-W", "error", str(script)] + (["--json"] if json_mode else [])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=90)
    assert not result.stderr
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_constant(value):
            raise AssertionError(f"nonfinite JSON constant {value}")
        report = json.loads(result.stdout, parse_constant=reject_constant)
        assert report["physical_predictions"] == {"masses": None, "mixing": None}
        assert len(report["cases"]) == 3
        assert report["cases"][1]["certificate"]["available"] is False
    else:
        assert "not physical fermion emergence" in result.stdout
        assert "LIMITATIONS" in result.stdout
        assert "null" in result.stdout
