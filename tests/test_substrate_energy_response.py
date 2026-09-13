"""Independent complete-sector tests for the stipulated microscopic energy source.

The occupation, commutator, Lehmann and full-space resolvent oracles below do
not call the energy module's construction or contraction helpers. Spectral
fixtures are reused unchanged, including across the two local partitions.
"""
import ast
from dataclasses import replace
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from bpr import substrate_energy_response as energy
from bpr import substrate_current_response as current
from bpr.substrate_fermionization import fixed_number_model


CASES = tuple(itertools.product((3, 4, 5), (0.7, 40.0)))
PARTITIONS = ("symmetric", "improved")
FREQUENCIES = (0.5j, 1 + 0.5j, 4 + 1j)
ROOT = Path(__file__).resolve().parents[1]


def occupations(L, N):
    if L == 1:
        yield (N,)
    else:
        for n in range(N + 1):
            for rest in occupations(L - 1, N - n):
                yield (n,) + rest


def occupation_operators(model):
    """Build every onsite/bond matrix from occupation selection rules."""
    basis = tuple(occupations(model.L, model.N))
    assert tuple(model.basis) == basis
    dimension = math.comb(model.L + model.N - 1, model.N)
    assert len(basis) == dimension
    index = {state: row for row, state in enumerate(basis)}
    onsite, bonds, currents = [], [], []
    for x in range(model.L):
        y = (x + 1) % model.L
        transfer = np.zeros((dimension, dimension), dtype=complex)
        for col, ket in enumerate(basis):
            if ket[x]:
                bra = list(ket)
                bra[x] -= 1
                bra[y] += 1
                transfer[index[tuple(bra)], col] = math.sqrt(ket[x] * (ket[y] + 1))
        onsite.append(np.diag([model.g * n[x] * (n[x] - 1) / 2 for n in basis]))
        bonds.append(-model.C * (transfer + transfer.conj().T))
        currents.append(1j * model.C * (transfer - transfer.conj().T))
    return tuple(onsite), tuple(bonds), tuple(currents)


def comm(A, B):
    # Sparse multiplication keeps the complete 462-state support check bounded.
    a, b = sparse.csr_matrix(A), sparse.csr_matrix(B)
    return (a @ b - b @ a).toarray()


def local_oracle(onsite, bonds, partition):
    a, b = (0.5, 0.5) if partition == "symmetric" else (0.25, 0.75)
    return tuple(u + a * bonds[x - 1] + b * bonds[x] for x, u in enumerate(onsite))


def fourier(operators, m):
    L = len(operators)
    k = 2 * np.pi * (m % L) / L
    return sum(np.exp(-1j * k * x) * a for x, a in enumerate(operators)) / np.sqrt(L)


def expectation(ground, A):
    return np.vdot(ground, A @ ground)


def lehmann(eigenvalues, vectors, A, B, z):
    """Both mixed numerators with actual independent eigenvalue gaps."""
    ground = vectors[:, 0]
    excited = vectors[:, 1:]
    gaps = eigenvalues[1:] - eigenvalues[0]
    first = (ground.conj() @ A @ excited) * (excited.conj().T @ B @ ground)
    second = (ground.conj() @ B @ excited) * (excited.conj().T @ A @ ground)
    return sum(first / (z - gaps) - second / (z + gaps))


def resolvent(H, eigenvalue, ground, A, B, z):
    eye = np.eye(len(H))
    excitation = H - eigenvalue * eye
    positive = np.linalg.solve(z * eye - excitation, B @ ground)
    negative = np.linalg.solve(z * eye + excitation, A @ ground)
    return np.vdot(ground, A @ positive) - np.vdot(ground, B @ negative)


def assert_operator(actual, expected, tolerance=2e-10):
    assert actual.shape == expected.shape
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "L%d-g%g" % case)
def frozen(request):
    L, g = request.param
    model = fixed_number_model(L, L, 1.0, g)
    operators = occupation_operators(model)
    system = current.spectral_system(model.H)
    eigenvalues, vectors = np.linalg.eigh(model.H)
    return {"model": model, "operators": operators, "system": system,
            "eigenvalues": eigenvalues, "vectors": vectors, "responses": {}}


def response(frozen, partition, m, z):
    key = (partition, m, z)
    if key not in frozen["responses"]:
        frozen["responses"][key] = energy.energy_response(
            frozen["system"], frozen["model"], m, z, partition)
    return frozen["responses"][key]


@pytest.mark.parametrize("partition", PARTITIONS)
def test_complete_local_partition_and_all_pair_continuity(frozen, partition):
    model = frozen["model"]
    onsite, bonds, _ = frozen["operators"]
    expected = local_oracle(onsite, bonds, partition)
    actual = energy.local_energies(model, partition)
    transfers = energy.energy_transfers(model, partition)
    assert isinstance(actual, tuple) and len(actual) == model.L
    assert isinstance(transfers, tuple) and all(isinstance(row, tuple) for row in transfers)
    assert all(len(row) == model.L for row in transfers)
    assert_operator(model.H, sum(onsite) + sum(bonds))
    assert_operator(sum(actual), model.H)
    for x in range(model.L):
        assert_operator(actual[x], expected[x], 2e-13)
        assert_operator(actual[x], actual[x].conj().T, 2e-13)
        for y in range(model.L):
            direct = 1j * comm(expected[x], expected[y])
            assert_operator(transfers[x][y], direct, 2e-13)
            assert_operator(transfers[x][y], transfers[x][y].conj().T, 2e-13)
            assert_operator(transfers[x][y], -transfers[y][x], 2e-13)
        assert_operator(1j * comm(model.H, expected[x]), -sum(transfers[x]))
    assert_operator(sum(sum(row) for row in transfers), np.zeros_like(model.H))
    # Frozen L3 and L4 are included: both modular coincidences must be added.
    assert np.max(np.abs(transfers[0][2])) > 0.01


@pytest.mark.parametrize("partition", PARTITIONS)
def test_independent_A_Q_and_fourier_continuity(frozen, partition):
    model = frozen["model"]
    onsite, bonds, currents = frozen["operators"]
    local = local_oracle(onsite, bonds, partition)
    L = model.L
    Q = tuple(1j * comm(bonds[x], bonds[(x + 1) % L]) for x in range(L))
    A = []
    for x in range(L):
        nx = np.diag([state[x] for state in model.basis])
        ny = np.diag([state[(x + 1) % L] for state in model.basis])
        if partition == "symmetric":
            coefficient = nx + ny - np.eye(len(model.H))
            A.append(model.g * coefficient @ currents[x] / 2 + (Q[x - 1] + Q[x]) / 4)
        else:
            coefficient = nx + 3 * ny - 2 * np.eye(len(model.H))
            anticommutator = coefficient @ currents[x] + currents[x] @ coefficient
            A.append(model.g * anticommutator / 8 + (Q[x - 1] + 9 * Q[x]) / 16)
    ab = 0.25 if partition == "symmetric" else 3 / 16
    for m in range(L):
        obs = energy.fourier_observables(model, m, partition)
        E = fourier(local, m)
        R = 1j * comm(model.H, E)
        q = np.exp(-2j * np.pi * m / L) - 1
        q2 = np.exp(-4j * np.pi * m / L) - 1
        assert obs["m"] == m and obs["partition"] == partition
        assert obs["k"] == pytest.approx(2 * np.pi * m / L)
        assert obs["q"] == pytest.approx(q, abs=1e-15)
        assert obs["q2"] == pytest.approx(q2, abs=1e-15)
        assert obs["d"] == pytest.approx(-q, abs=1e-15)
        for key, expected in (("h", E), ("R", R), ("A", fourier(A, m)),
                              ("Q", fourier(Q, m)), ("V", fourier(bonds, m)),
                              ("dotV", 1j * comm(model.H, fourier(bonds, m)))):
            assert_operator(obs[key], expected)
        assert_operator(R, q * fourier(A, m) + ab * q2 * fourier(Q, m))
    assert_operator(energy.fourier_observables(model, 0, partition)["h"], model.H / np.sqrt(L))


@pytest.mark.parametrize("N", (3, 6))
def test_unwrapped_six_site_support_and_complete_sector_cap(N):
    model = fixed_number_model(6, N, 1.0, 0.7)
    onsite, bonds, _ = occupation_operators(model)
    local = local_oracle(onsite, bonds, "symmetric")
    assert len(model.basis) == math.comb(5 + N, N)
    assert len(model.basis) <= 512
    actual = energy.local_energies(model)
    assert_operator(sum(actual), model.H)
    # N=6 explicitly constructs the complete 462-state model, without eigh.
    if N == 6:
        assert len(model.basis) == 462
        assert any(max(ket) == 6 for ket in model.basis)
        return
    transfers = energy.energy_transfers(model)
    for x in range(6):
        for y in range(6):
            assert_operator(transfers[x][y], 1j * comm(local[x], local[y]))
            distance = min((x - y) % 6, (y - x) % 6)
            if distance == 2:
                assert np.max(np.abs(transfers[x][y])) > 0.01
            elif distance == 3 or x == y:
                np.testing.assert_array_equal(transfers[x][y], np.zeros_like(model.H))


def test_full_partition_change_not_false_pair_equality(frozen):
    model = frozen["model"]
    onsite, bonds, _ = frozen["operators"]
    local = local_oracle(onsite, bonds, "symmetric")
    delta = tuple((bonds[x] - bonds[x - 1]) / 4 for x in range(model.L))
    original = energy.energy_transfers(model, "symmetric")
    changed = energy.energy_transfers(model, "improved")
    witness = 0.0
    for x in range(model.L):
        for y in range(model.L):
            correction = 1j * (comm(local[x], delta[y]) + comm(delta[x], local[y])
                                + comm(delta[x], delta[y]))
            assert_operator(changed[x][y], original[x][y] + correction)
            witness = max(witness, float(np.max(np.abs(correction))))
    assert witness > 0.01
    for m in range(model.L):
        old = energy.fourier_observables(model, m, "symmetric")
        new = energy.fourier_observables(model, m, "improved")
        d = 1 - np.exp(-2j * np.pi * m / model.L)
        Vk = fourier(bonds, m)
        assert_operator(new["h"], old["h"] + d * Vk / 4)
        assert_operator(new["R"], old["R"] + d * 1j * comm(model.H, Vk) / 4)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_both_ward_signs_contacts_and_both_adjoint_moments(frozen, partition):
    model = frozen["model"]
    eigenvalues, vectors = frozen["eigenvalues"], frozen["vectors"]
    ground = vectors[:, 0]
    gaps = eigenvalues[1:] - eigenvalues[0]
    local = local_oracle(*frozen["operators"][:2], partition)
    for m in range(model.L):
        E = fourier(local, m)
        R = 1j * comm(model.H, E)
        C = expectation(ground, comm(E, E.conj().T))
        D = expectation(ground, comm(R, E.conj().T))
        plus = np.abs(vectors[:, 1:].conj().T @ E @ ground) ** 2
        minus = np.abs(ground.conj() @ E @ vectors[:, 1:]) ** 2
        Mplus, Mminus = np.dot(gaps, plus), np.dot(gaps, minus)
        double = expectation(ground, comm(E, comm(model.H, E.conj().T)))
        scale = max(1.0, abs(double))
        assert double == pytest.approx(Mplus + Mminus, rel=2e-10, abs=2e-10)
        assert D == pytest.approx(-1j * (Mplus + Mminus), rel=2e-10, abs=2e-10)
        for z in FREQUENCIES:
            result = response(frozen, partition, m, z)
            assert result["energy_expectation"] == pytest.approx(expectation(ground, E), rel=2e-10, abs=2e-10)
            assert result["rate_expectation"] == pytest.approx(expectation(ground, R), abs=2e-10 * scale)
            assert result["Mplus"] == pytest.approx(Mplus, rel=2e-10, abs=2e-10)
            assert result["Mminus"] == pytest.approx(Mminus, rel=2e-10, abs=2e-10)
            assert result["double_commutator"] == pytest.approx(double, rel=2e-10, abs=2e-10)
            assert result["contact_C"] == pytest.approx(C, abs=2e-10 * scale)
            assert result["contact_D"] == pytest.approx(D, rel=2e-10, abs=2e-10)
            np.testing.assert_allclose(result["gaps"], frozen["system"]["gaps"][1:], rtol=0, atol=0)
            # Compare weights using the passed system's eigenvectors. Independent
            # eigensolvers may rotate a degenerate excited eigenspace arbitrarily.
            U = frozen["system"]["vectors"]
            G = frozen["system"]["ground"]
            wp = np.abs(U[:, 1:].conj().T @ E @ G) ** 2
            wm = np.abs(G.conj() @ E @ U[:, 1:]) ** 2
            np.testing.assert_allclose(result["transition_weights_plus"], wp, rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(result["transition_weights_minus"], wm, rtol=2e-10, atol=2e-10)
            for key, A, B in (("chi_EE", E, E.conj().T), ("chi_RE", R, E.conj().T),
                              ("chi_ER", E, R.conj().T), ("chi_RR", R, R.conj().T)):
                expected = lehmann(eigenvalues, vectors, A, B, z)
                assert result[key] == pytest.approx(expected, rel=2e-10, abs=2e-10 * scale)
            left = z * result["chi_EE"]
            assert left == pytest.approx(C + 1j * result["chi_RE"], abs=2e-10 * scale)
            assert left == pytest.approx(C - 1j * result["chi_ER"], abs=2e-10 * scale)
            assert z * left == pytest.approx(z * C + 1j * D + result["chi_RR"], abs=2e-10 * scale)
            for key in ("fsum_residual", "ward_first_residual", "ward_second_residual"):
                assert np.isfinite(result[key]) and 0 <= result[key] <= 2e-10 * scale


@pytest.mark.parametrize("partition", PARTITIONS)
def test_full_space_resolvent_oracle(frozen, partition):
    if frozen["model"].L != 3:
        return
    model = frozen["model"]
    local = local_oracle(*frozen["operators"][:2], partition)
    E = fourier(local, 1)
    R = 1j * comm(model.H, E)
    ground = frozen["vectors"][:, 0]
    for z in FREQUENCIES:
        result = response(frozen, partition, 1, z)
        for key, A, B in (("chi_EE", E, E.conj().T), ("chi_RE", R, E.conj().T),
                          ("chi_ER", E, R.conj().T), ("chi_RR", R, R.conj().T)):
            expected = resolvent(model.H, frozen["eigenvalues"][0], ground, A, B, z)
            assert result[key] == pytest.approx(expected, rel=2e-10, abs=2e-10)


def test_reflection_partition_response_not_invariance(frozen):
    model = frozen["model"]
    bonds = frozen["operators"][1]
    for m in range(model.L):
        Vk = fourier(bonds, m)
        coefficient = abs(1 - np.exp(-2j * np.pi * m / model.L)) ** 2 / 16
        for z in FREQUENCIES:
            original = response(frozen, "symmetric", m, z)["chi_EE"]
            improved = response(frozen, "improved", m, z)["chi_EE"]
            vv = lehmann(frozen["eigenvalues"], frozen["vectors"], Vk, Vk.conj().T, z)
            # Only a summed self response, never individual degenerate weights
            # or an assumption that either mixed susceptibility vanishes.
            assert improved - original == pytest.approx(coefficient * vv, rel=2e-10, abs=2e-10)
    if model.L == 3:
        difference = (response(frozen, "improved", 1, 0.5j)["chi_EE"]
                      - response(frozen, "symmetric", 1, 0.5j)["chi_EE"])
        assert difference.real < 0
        assert abs(difference.imag) < 2e-10


@pytest.mark.parametrize("partition", PARTITIONS)
def test_uniform_energy_semantic_zero_not_transition_threshold(frozen, partition):
    model = frozen["model"]
    for m in (0,):
        for z in FREQUENCIES:
            result = response(frozen, partition, m, z)
            for key in ("chi_EE", "chi_RE", "chi_ER", "chi_RR", "contact_C", "contact_D",
                        "Mplus", "Mminus", "double_commutator"):
                assert result[key] == 0
            np.testing.assert_array_equal(result["transition_weights_plus"], 0)
            np.testing.assert_array_equal(result["transition_weights_minus"], 0)


@pytest.mark.parametrize("L", (3, 4, 5))
def test_free_ring_control_from_condensate_selection_rule(L):
    model = fixed_number_model(L, L, 1.0, 0.0)
    system = current.spectral_system(model.H)
    onsite, bonds, _ = occupation_operators(model)
    local = local_oracle(onsite, bonds, "symmetric")
    # Exact N-boson zero-momentum condensate in the complete occupation basis.
    ground = np.array([math.sqrt(math.factorial(L) / math.prod(math.factorial(n) for n in ket))
                       / L ** (L / 2) for ket in model.basis])
    assert_operator(model.H @ ground, -2 * L * ground)
    for m in range(1, L):
        k = 2 * np.pi * m / L
        gap = 2 * (1 - np.cos(k))
        weight = (L / L) * (1 + np.cos(k)) ** 2
        E = fourier(local, m)
        source = E @ ground
        assert np.vdot(source, source).real == pytest.approx(weight, abs=2e-13)
        assert_operator((model.H + 2 * L * np.eye(len(model.H))) @ source, gap * source)
        for z in FREQUENCIES:
            result = energy.energy_response(system, model, m, z)
            expected = weight * (1 / (z - gap) - 1 / (z + gap))
            assert result["chi_EE"] == pytest.approx(expected, rel=2e-10, abs=2e-12)
            assert result["Mplus"] == pytest.approx(gap * weight, rel=2e-10, abs=2e-12)


def test_nonhermitian_oracle_requires_both_adjoint_weights():
    # A deliberately non-reflection-symmetric source guards the test oracle
    # against silently replacing two spectral measures by twice one measure.
    H = np.diag([0.0, 2.0, 5.0])
    E = np.array([[0, 1j, 2], [3, 0, 0], [4j, 0, 0]], dtype=complex)
    ground = np.array([1.0, 0.0, 0.0])
    R = 1j * comm(H, E)
    plus = 2 * 9 + 5 * 16
    minus = 2 * 1 + 5 * 4
    assert plus != minus
    double = expectation(ground, comm(E, comm(H, E.conj().T)))
    assert double == plus + minus
    C = expectation(ground, comm(E, E.conj().T))
    D = expectation(ground, comm(R, E.conj().T))
    assert C != 0
    assert D == -1j * (plus + minus)
    system = current.spectral_system(H)
    for z in FREQUENCIES:
        chi = lehmann(np.diag(H), np.eye(3), E, E.conj().T, z)
        assert chi == pytest.approx(resolvent(H, 0, ground, E, E.conj().T, z), abs=2e-13)
        assert chi == pytest.approx(current.retarded_response(system, E, E.conj().T, z), abs=2e-13)
        re = lehmann(np.diag(H), np.eye(3), R, E.conj().T, z)
        er = lehmann(np.diag(H), np.eye(3), E, R.conj().T, z)
        rr = lehmann(np.diag(H), np.eye(3), R, R.conj().T, z)
        assert z * chi == pytest.approx(C + 1j * re, abs=2e-13)
        assert z * chi == pytest.approx(C - 1j * er, abs=2e-13)
        assert z * z * chi == pytest.approx(z * C + 1j * D + rr, abs=2e-13)


def test_identity_origin_shift_actual_matrices_and_gaps():
    model = fixed_number_model(3, 3, 1.0, 0.7)
    onsite, bonds, _ = occupation_operators(model)
    local = local_oracle(onsite, bonds, "symmetric")
    original = current.spectral_system(model.H)
    eye = np.eye(len(model.H))
    for kappa in (-3.0, 0.0, 2.5):
        shifted_H = model.H + kappa * eye
        shifted_local = tuple(h + kappa * eye / model.L for h in local)
        shifted = original if kappa == 0 else current.spectral_system(shifted_H)
        assert_operator(sum(shifted_local), shifted_H)
        np.testing.assert_allclose(shifted["gaps"], original["gaps"], rtol=2e-12, atol=2e-12)
        for x in range(model.L):
            assert_operator(1j * comm(shifted_H, shifted_local[x]), 1j * comm(model.H, local[x]))
        for m in range(1, model.L):
            E, shifted_E = fourier(local, m), fourier(shifted_local, m)
            assert_operator(shifted_E, E)
            for z in FREQUENCIES:
                expected = current.retarded_response(original, E, E.conj().T, z)
                actual = current.retarded_response(shifted, shifted_E, shifted_E.conj().T, z)
                assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)
        # Exact H recognition with normalization outside the generic evaluator.
        assert current.retarded_response(shifted, shifted_H, shifted_H, 0.5j) / model.L == 0
        assert current.retarded_response(shifted, eye, shifted_H, 0.5j) == 0


@pytest.fixture(scope="module")
def small():
    model = fixed_number_model(3, 3, 1.0, 0.7)
    return model, current.spectral_system(model.H)


@pytest.mark.parametrize("z", (True, np.bool_(False), 0, 1, -0.5j, complex(np.nan, 1),
                              complex(1, np.inf), 1e-320j, "0.5j", [0.5j]))
def test_frequency_validation_precedes_uniform_zero(small, z):
    model, system = small
    with pytest.raises(ValueError):
        energy.energy_response(system, model, 0, z)


@pytest.mark.parametrize("m", (True, np.bool_(True), 1.0, 1 + 0j, "1", None, [1]))
def test_invalid_momenta(small, m):
    model, system = small
    with pytest.raises(ValueError):
        energy.fourier_observables(model, m)
    with pytest.raises(ValueError):
        energy.energy_response(system, model, m, 0.5j)


@pytest.mark.parametrize("partition", ("", "Symmetric", "prime", None, True, ["symmetric"]))
def test_invalid_partition_all_public_operator_apis(small, partition):
    model, system = small
    for function, args in ((energy.local_energies, (model,)),
                           (energy.energy_transfers, (model,)),
                           (energy.fourier_observables, (model, 0)),
                           (energy.energy_response, (system, model, 0, 0.5j))):
        with pytest.raises(ValueError):
            function(*args, partition=partition)


def test_signed_aliases_and_numpy_integer_momenta(small):
    model, system = small
    reference = energy.fourier_observables(model, 1)
    for m in (-5, -2, 4, np.int64(7), 10 ** 40):
        assert m % model.L == 1
        actual = energy.fourier_observables(model, m)
        assert actual["m"] == 1
        assert_operator(actual["h"], reference["h"], 2e-13)
        result = energy.energy_response(system, model, m, 0.5j)
        expected = energy.energy_response(system, model, 1, 0.5j)
        assert result["chi_EE"] == expected["chi_EE"]
    for m in (-6, 3, np.int64(6)):
        assert energy.energy_response(system, model, m, 0.5j)["chi_EE"] == 0


@pytest.mark.parametrize("field,value", (("L", True), ("N", False), ("C", True), ("g", True),
                                         ("C", np.inf), ("g", np.nan), ("C", 1e-320),
                                         ("g", -1.0), ("C", 0.0), ("C", 1 + 1j),
                                         ("C", Fraction(1, 10 ** 400))))
def test_model_scalar_validation(small, field, value):
    model, _ = small
    with pytest.raises(ValueError):
        energy.local_energies(replace(model, **{field: value}))


def test_malformed_models_and_system_mismatch_rejected(small):
    model, system = small
    bad_models = (replace(model, H=np.zeros((2, 3))), replace(model, H=model.H[:-1, :-1]),
                  replace(model, H=model.H + np.eye(len(model.H))),
                  replace(model, basis=model.basis[:-1]), replace(model, D=model.D + 1),
                  replace(model, V=model.V + np.eye(len(model.H))))
    for bad in bad_models:
        with pytest.raises(ValueError):
            energy.local_energies(bad)
        with pytest.raises(ValueError):
            energy.energy_response(system, bad, 0, 0.5j)
    other = fixed_number_model(3, 3, 1.0, 40.0)
    other_system = current.spectral_system(other.H)
    with pytest.raises(ValueError):
        energy.energy_response(other_system, model, 0, 0.5j)
    with pytest.raises(ValueError):
        energy.energy_response({}, model, 0, 0.5j)


def test_complete_sector_cap_checked_before_reconstruction(small, monkeypatch):
    model, _ = small
    oversized = replace(model, L=7, N=7)
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation or reconstruction occurred before cap validation")
    monkeypatch.setattr(energy, "fixed_number_model", forbidden)
    with pytest.raises(ValueError):
        energy.local_energies(oversized)


@pytest.mark.parametrize("kwargs", ({"L": True}, {"L": 2}, {"L": 6}, {"L": 10 ** 20},
                                   {"L": 3.0}, {"L": 3, "C": False},
                                   {"L": 3, "C": 0}, {"L": 3, "g": -1},
                                   {"L": 3, "g": np.nan}))
def test_report_validation_before_model_allocation(kwargs, monkeypatch):
    def forbidden(*args, **kw):
        raise AssertionError("model allocated before invalid report arguments rejected")
    monkeypatch.setattr(energy, "fixed_number_model", forbidden)
    with pytest.raises(ValueError):
        energy.case_report(**kwargs)


@pytest.mark.parametrize("C,g", ((2.0 ** -250, 2.0 ** 250), (2.0 ** 250, 2.0 ** -250)))
def test_severe_scale_separation_explicitly_unavailable(C, g):
    model = fixed_number_model(3, 3, C, g)
    for function in (energy.local_energies, energy.energy_transfers):
        with pytest.raises(current.NumericalUnavailable):
            function(model)


def exact_real_commutator(A, B):
    """Exact binary-input reference, independent of dyadic production helpers."""
    result = np.zeros(A.shape, dtype=float)
    for i in range(len(A)):
        for j in range(len(A)):
            value = sum((Fraction(float(A[i, k].real)) * Fraction(float(B[k, j].real))
                         - Fraction(float(B[i, k].real)) * Fraction(float(A[k, j].real))
                         for k in range(len(A))), Fraction(0))
            result[i, j] = float(value)
    return result


@pytest.mark.parametrize("scale", (2.0 ** -400, 2.0 ** 400))
def test_extreme_balanced_transfer_components_or_explicit_unavailability(scale):
    model = fixed_number_model(3, 3, scale, 0.7 * scale)
    onsite, bonds, _ = occupation_operators(model)
    local = local_oracle(onsite, bonds, "symmetric")
    try:
        transfers = energy.energy_transfers(model)
    except current.NumericalUnavailable:
        return
    for x in range(3):
        for y in range(3):
            expected = exact_real_commutator(local[x], local[y])
            # atol=0 prevents enormous or tiny unrelated channels hiding erasure.
            np.testing.assert_allclose(transfers[x][y].imag, expected, rtol=2e-12, atol=0)
            np.testing.assert_array_equal(transfers[x][y].real, 0)


def test_returned_arrays_do_not_mutate_reusable_transition_cache(small):
    model, system = small
    before = energy.energy_response(system, model, 1, 0.5j)
    expected = {key: np.array(before[key], copy=True) for key in
                ("gaps", "transition_weights_plus", "transition_weights_minus")}
    baseline = before["chi_EE"]
    for key in expected:
        try:
            before[key][...] = 12345
        except ValueError:  # Read-only detached views are also acceptable.
            pass
    after = energy.energy_response(system, model, 1, 0.5j)
    assert after["chi_EE"] == baseline
    for key, value in expected.items():
        np.testing.assert_array_equal(after[key], value)
    assert not system["H"].flags.writeable
    assert not system["vectors"].flags.writeable


def test_impostor_eigensystem_rejected_even_with_forged_resolution():
    model = fixed_number_model(3, 3, 1.0, 0.7)
    for resolution in (None, 1e100):
        bad = current.spectral_system(model.H)
        bad["vectors"] = np.eye(len(model.H), dtype=complex)
        bad["ground"] = bad["vectors"][:, 0]
        if resolution is not None:
            bad["resolution"] = resolution
        for m in (0, 1):
            with pytest.raises(ValueError):
                energy.energy_response(bad, model, m, 0.5j)


def test_adversarial_stale_copy_rejected_without_mutating_original():
    # Deliberate invalid-input test, not a scientific partition comparison.
    model = fixed_number_model(3, 3, 1.0, 0.7)
    system = current.spectral_system(model.H)
    E = fourier(local_oracle(*occupation_operators(model)[:2], "symmetric"), 1)
    baseline = current.retarded_response(system, E, E.conj().T, 0.5j)
    snapshot = {key: tuple(a.copy() for a in value)
                for key, value in system["_transition_cache"].items()}
    original_vectors = system["vectors"].copy()
    bad = dict(system)
    bad["vectors"] = np.eye(len(model.H), dtype=complex)
    bad["ground"] = bad["vectors"][:, 0]
    with pytest.raises(ValueError):
        energy.energy_response(bad, model, 1, 0.5j)
    np.testing.assert_array_equal(system["vectors"], original_vectors)
    for key, values in snapshot.items():
        for actual, expected in zip(system["_transition_cache"][key], values):
            np.testing.assert_array_equal(actual, expected)
    result = energy.energy_response(system, model, 1, 0.5j)
    assert result["chi_EE"] == pytest.approx(baseline, rel=2e-10, abs=2e-12)


def test_external_poisoned_transition_cache_is_ignored_and_preserved():
    model = fixed_number_model(4, 2, 1.0, 0.7)
    system = current.spectral_system(model.H)
    E = energy.fourier_observables(model, 1, "improved")["h"]
    independent_E = fourier(local_oracle(*occupation_operators(model)[:2], "improved"), 1)
    assert_operator(E, independent_E, 2e-13)
    baseline = current.retarded_response(system, E, E.conj().T, 1 + 0.5j)
    oracle = resolvent(model.H, system["energies"][0], system["ground"], E, E.conj().T, 1 + 0.5j)
    assert baseline == pytest.approx(oracle, rel=2e-12, abs=2e-12)
    cache = system["_transition_cache"]
    assert cache
    for values in cache.values():
        for array in values:
            array[...] = 0
    result = energy.energy_response(system, model, 1, 1 + 0.5j, "improved")
    # Both components are materially nonzero: a loose absolute tolerance cannot
    # let the poisoned zero response pass.
    assert result["chi_EE"].real == pytest.approx(baseline.real, rel=2e-10, abs=0)
    assert result["chi_EE"].imag == pytest.approx(baseline.imag, rel=2e-10, abs=0)
    assert system["_transition_cache"] is cache
    for values in cache.values():
        for array in values:
            np.testing.assert_array_equal(array, 0)


def test_computed_response_overflow_reports_numerical_unavailability():
    scale = 1e150
    model = fixed_number_model(3, 3, scale, 0.7 * scale)
    system = current.spectral_system(model.H)
    with pytest.raises(current.NumericalUnavailable):
        energy.energy_response(system, model, 1, 0.5j * scale)


def test_tiny_frequency_component_exact_oracle_or_explicit_unavailability():
    model = fixed_number_model(4, 2, 1.0, 0.7)
    system = current.spectral_system(model.H)
    local = local_oracle(*occupation_operators(model)[:2], "symmetric")
    # Exact real pi phases and sqrt(4)=2 avoid a spurious floating sine source.
    E = sum((-1) ** x * h for x, h in enumerate(local)).real / 2
    ground = system["ground"]
    vectors = system["vectors"]
    np.testing.assert_array_equal(ground.imag, 0)
    np.testing.assert_array_equal(vectors.imag, 0)
    matrix = [[Fraction(float(value)) for value in row] for row in E]
    g = [Fraction(float(value)) for value in ground.real]
    right = [sum((matrix[i][j] * g[j] for j in range(len(g))), Fraction(0))
             for i in range(len(g))]
    norm = sum((value * value for value in g), Fraction(0))
    mean = sum((g[i] * right[i] for i in range(len(g))), Fraction(0)) / norm
    x, eta = Fraction(1, 2 ** 500), Fraction(1, 2)
    real, imag = Fraction(0), Fraction(0)
    for n in range(1, len(g)):
        excited = [Fraction(float(value)) for value in vectors[:, n].real]
        transition = sum((excited[i] * (right[i] - mean * g[i])
                          for i in range(len(g))), Fraction(0))
        weight = transition * transition
        gap = Fraction(float(system["gaps"][n]))
        denominator_minus = (x - gap) ** 2 + eta ** 2
        denominator_plus = (x + gap) ** 2 + eta ** 2
        real += weight * ((x - gap) / denominator_minus - (x + gap) / denominator_plus)
        imag += weight * (-eta / denominator_minus + eta / denominator_plus)
    expected_real, expected_imag = float(real), float(imag)
    assert expected_real != 0 and expected_imag != 0
    try:
        result = energy.energy_response(system, model, 2, complex(float(x), float(eta)))
    except current.NumericalUnavailable:
        return
    assert result["chi_EE"].real == pytest.approx(expected_real, rel=2e-10, abs=0)
    assert result["chi_EE"].imag == pytest.approx(expected_imag, rel=2e-10, abs=0)


def test_python38_grammar_for_new_deliverables():
    paths = (ROOT / "bpr/substrate_energy_response.py",
             ROOT / "scripts/demo_substrate_energy_response.py", Path(__file__))
    for path in paths:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


def test_isolated_text_and_strict_json_demos_no_files_or_stderr(tmp_path):
    script = ROOT / "scripts/demo_substrate_energy_response.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    for flags in ((), ("--json",)):
        before = set(tmp_path.iterdir())
        run = subprocess.run([sys.executable, "-B", "-W", "error", str(script)] + list(flags),
                             cwd=str(tmp_path), env=environment, capture_output=True,
                             text=True, timeout=600, check=False)
        assert run.returncode == 0, run.stderr
        assert run.stderr == ""
        assert set(tmp_path.iterdir()) == before
        assert run.stdout.strip()
        if flags:
            def invalid_constant(value):
                raise AssertionError("non-strict JSON constant " + value)
            payload = json.loads(run.stdout, parse_constant=invalid_constant)
            assert set(("model_id", "conventions", "numerical_policy", "limitations", "cases")) <= set(payload)
            assert payload["limitations"] and payload["numerical_policy"]
            assert len(payload["cases"]) == 6
            seen = set()
            for case in payload["cases"]:
                params = case["parameters"]
                L = params["L"]
                seen.add((L, params["g"]))
                assert params["N"] == L and params["C"] == 1
                assert params["dimension"] == math.comb(2 * L - 1, L)
                assert {p["partition"] for p in case["partitions"]} == set(PARTITIONS)
                for part in case["partitions"]:
                    assert len(part["momenta"]) == L
                    assert {row["m"] for row in part["momenta"]} == set(range(L))
                    assert part["diagnostics"]["range2_witness_norm"] > 0
                    for row in part["momenta"]:
                        assert len(row["responses"]) == 3
                        for result in row["responses"]:
                            assert len(result["z"]) == 2
                            assert len(result["chi_EE"]) == 2
                            if row["m"] == 0:
                                assert result["chi_EE"] == [0, 0]
                change = case["partition_change"]
                assert len(change["momenta"]) == L
                witness = change["strict_L3_m1_imaginary_z_difference"]
                assert witness < 0 if L == 3 else witness is None
                if L == 3:
                    moment = next(row for row in change["momenta"] if row["m"] == 1)
                    control = next(row for row in moment["responses"] if row["z"] == [0, 0.5])
                    subtraction = complex(*control["chi_improved"]) - complex(*control["chi_symmetric"])
                    assert witness == pytest.approx(subtraction.real, rel=2e-12, abs=0)
                    assert complex(*control["difference"]) == pytest.approx(subtraction, rel=2e-12, abs=0)
                    assert abs(subtraction.imag) < 2e-10
                for row in change["momenta"]:
                    for result in row["responses"]:
                        actual = complex(*result["chi_improved"])
                        expected = complex(*result["predicted_improved"])
                        assert actual == pytest.approx(expected, rel=2e-10, abs=2e-10)
                shifts = case["identity_shifts"]
                assert {row["kappa"] for row in shifts} == {-3, 0, 2.5}
                for row in shifts:
                    assert row["ground_energy_shift"] == pytest.approx(row["kappa"], abs=2e-10)
                    assert row["uniform_structural_zero"]
                    assert row["uniform_raw_fourier_not_used"]
                    for key in ("gap_max_residual", "local_sum_max_residual", "response_max_residual"):
                        assert 0 <= row[key] < 2e-9
            assert seen == set(CASES)
        else:
            assert "energy" in run.stdout.lower()
