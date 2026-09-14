"""Independent tests of the frozen 2026-09-13 bounded-local-symmetry contract.

Only ``physical_cases`` executes the six actual constructions. All other
orchestration, fault, allocation and CLI probes use mocks or retained arrays.
Occupation entries below are derived directly from the displayed Hamiltonian,
not from production operators, basis builders or ladder helpers. Boundary
matrix-unit probes are finite recurrence controls, not infinite-Fock proofs.
"""
import copy
import importlib
import itertools
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bpr import substrate_local_symmetry as subject


PAIRS = tuple((L, g) for L in (3, 4, 5) for g in (0, 1))
NAMES = (
    "identity", "local_flip", "local_vacuum_projector",
    "total_number_parity", "reflection",
)
CLASSES = (
    "empty_support", "singleton", "singleton", "full_support",
    "one_exterior_counterexample",
)
CASE_KEYS = {
    "L", "g", "C", "cutoff", "dimension", "basis_order", "status",
    "operators", "scope",
}
OPERATOR_KEYS = {
    "name", "support_class", "exact_conservation", "commutator",
    "finite_frobenius_norm", "conservation_diagnostic", "witness",
}
EPS = 2.0 ** -52


def _sector(L, N):
    return tuple(state for state in itertools.product(range(N + 1), repeat=L)
                 if sum(state) == N)


def _basis(L, cutoff=2):
    return tuple(state for N in range(cutoff + 1) for state in _sector(L, N))


def _hamiltonian(basis, g):
    """H[r,c] from occupation deltas, with C=1 and every ring bond once."""
    L = len(basis[0])
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for row, final in enumerate(basis):
        for column, initial in enumerate(basis):
            if final == initial:
                matrix[row, column] += g * sum(n * (n - 1) // 2 for n in initial)
            for source in range(L):
                for target in ((source - 1) % L, (source + 1) % L):
                    if initial[source] == 0:
                        continue
                    moved = list(initial)
                    moved[source] -= 1
                    moved[target] += 1
                    if tuple(moved) == final:
                        matrix[row, column] -= math.sqrt(
                            initial[source] * (initial[target] + 1))
    return matrix


def _fixed_operators(basis):
    """Closed occupation-entry formulas, including cross-number compression."""
    L = len(basis[0])
    names = NAMES + (("opposite_site_swap", "dark_vacuum_projector") if L == 4 else ())
    result = {name: np.zeros((len(basis), len(basis)), dtype=np.complex128)
              for name in names}
    for row, final in enumerate(basis):
        for column, initial in enumerate(basis):
            if final == initial:
                result["identity"][row, column] = 1
                result["local_vacuum_projector"][row, column] = int(initial[0] == 0)
                result["total_number_parity"][row, column] = (-1) ** sum(initial)
            if final[1:] == initial[1:] and (final[0], initial[0]) in ((0, 1), (1, 0)):
                result["local_flip"][row, column] = 1
            reflected = tuple(initial[(2 - x) % L] for x in range(L))
            result["reflection"][row, column] = int(final == reflected)
            if L == 4:
                swapped = (initial[2], initial[1], initial[0], initial[3])
                result["opposite_site_swap"][row, column] = int(final == swapped)
                n = initial[0] + initial[2]
                if (final[1] == initial[1] and final[3] == initial[3]
                        and final[0] + final[2] == n):
                    result["dark_vacuum_projector"][row, column] = (
                        math.sqrt(math.comb(n, initial[0]) * math.comb(n, final[0]))
                        / (2 ** n))
    return result


def _ladder(L, N):
    lower, upper = _sector(L, N - 1), _sector(L, N)
    result = np.zeros((len(lower), len(upper)), dtype=np.complex128)
    for row, final in enumerate(lower):
        for column, initial in enumerate(upper):
            if initial[0] and final == (initial[0] - 1,) + initial[1:]:
                result[row, column] = math.sqrt(initial[0])
    return result


def _unit(basis, state):
    vector = np.zeros(len(basis), dtype=np.complex128)
    vector[basis.index(tuple(state))] = 1
    return vector


def _witness_vectors(basis, name, g):
    L = len(basis[0])
    vacuum = [0] * L
    at_zero, at_one = vacuum.copy(), vacuum.copy()
    at_zero[0], at_one[1] = 1, 1
    if name == "local_flip":
        return _unit(basis, at_one), _unit(basis, vacuum), -1.0
    if name == "local_vacuum_projector":
        return _unit(basis, at_one), _unit(basis, at_zero), 1.0
    if name == "dark_vacuum_projector":
        row = np.zeros(len(basis), dtype=np.complex128)
        column = row.copy()
        for state, bright, dark in (
            ((2, 0, 0, 0), 0.5, 0.5),
            ((1, 0, 1, 0), 1 / math.sqrt(2), -1 / math.sqrt(2)),
            ((0, 0, 2, 0), 0.5, 0.5),
        ):
            row[basis.index(state)] = dark
            column[basis.index(state)] = bright
        return row, column, g / 2.0
    return None


def _decode_matrix(value):
    assert set(value) == {"shape", "real", "imag"}
    real, imag = np.asarray(value["real"]), np.asarray(value["imag"])
    assert list(real.shape) == value["shape"] == list(imag.shape)
    return real + 1j * imag


def _decode_scalar(value):
    assert set(value) == {"real", "imag"}
    return complex(value["real"], value["imag"])


def _component_close(actual, expected, allowance):
    delta = actual - expected
    assert np.all(np.abs(np.real(delta)) <= allowance)
    assert np.all(np.abs(np.imag(delta)) <= allowance)


def _strict_roundtrip(value):
    encoded = json.dumps(value, allow_nan=False)
    def reject_constant(token):
        raise AssertionError("non-strict JSON constant " + token)
    assert json.loads(encoded, parse_constant=reject_constant) == value


def _forbidden(*args, **kwargs):
    raise AssertionError("unexpected inherited construction or numerical solver")


@pytest.fixture(scope="module")
def physical_cases():
    """The sole real demo: exactly six cases, eighteen sectors, twelve maps."""
    owned = {}
    real_owned = subject._owned_case
    real_model = subject.all_number_model
    real_ladder = subject.local_annihilation_map
    def capture(L, g):
        assert (L, g) not in owned, "case construction retried"
        result = real_owned(L, g)
        owned[L, g] = result
        return result
    with patch.object(subject, "_owned_case", side_effect=capture), \
            patch.object(subject, "all_number_model", wraps=real_model) as models, \
            patch.object(subject, "local_annihilation_map", wraps=real_ladder) as ladders, \
            patch.object(np.linalg, "eigh", side_effect=_forbidden), \
            patch.object(np.linalg, "eig", side_effect=_forbidden), \
            patch.object(np.linalg, "eigvalsh", side_effect=_forbidden), \
            patch.object(np.linalg, "svd", side_effect=_forbidden), \
            patch.object(np.linalg, "matrix_rank", side_effect=_forbidden), \
            patch.object(np.linalg, "solve", side_effect=_forbidden):
        demo = subject.demonstration_report()
    return SimpleNamespace(demo=demo, owned=owned,
                           model_calls=models.call_args_list,
                           ladder_calls=ladders.call_args_list)


@pytest.fixture
def fake_inherited(monkeypatch):
    """Independent, preallocated inherited results; never call a real builder."""
    models = {(L, g, N): SimpleNamespace(
        L=L, N=N, C=1.0, g=g, basis=_sector(L, N),
        H=_hamiltonian(_sector(L, N), g))
        for L, g in PAIRS for N in (0, 1, 2)}
    ladders = {(L, N): _ladder(L, N) for L in (3, 4, 5) for N in (1, 2)}
    calls = []
    def model(L, N, C=1.0, g=0.7):
        calls.append(("model", L, N, C, g))
        assert C == 1
        return copy.deepcopy(models[L, g, N])
    def ladder(L, N, x):
        calls.append(("map", L, N, x))
        assert x == 0
        return ladders[L, N].copy()
    monkeypatch.setattr(subject, "all_number_model", model)
    monkeypatch.setattr(subject, "local_annihilation_map", ladder)
    return SimpleNamespace(models=models, ladders=ladders, calls=calls,
                           model=model, ladder=ladder)


def test_six_physical_cases_built_once(physical_cases):
    assert list(physical_cases.owned) == list(PAIRS)
    assert len(physical_cases.model_calls) == 18
    assert len(physical_cases.ladder_calls) == 12
    actual_models = []
    for call in physical_cases.model_calls:
        args, kwargs = call.args, call.kwargs
        actual_models.append((args[0] if args else kwargs["L"],
                              args[1] if len(args) > 1 else kwargs["N"]))
    assert actual_models == [(L, N) for L, g in PAIRS for N in (0, 1, 2)]
    actual_maps = []
    for call in physical_cases.ladder_calls:
        args, kwargs = call.args, call.kwargs
        actual_maps.append(tuple(args[i] if len(args) > i else kwargs[key]
                                 for i, key in enumerate(("L", "N", "x"))))
    assert actual_maps == [(L, N, 0) for L, g in PAIRS for N in (1, 2)]


@pytest.mark.parametrize("L,g", PAIRS)
def test_independent_occupation_matrices_and_report(physical_cases, L, g):
    owned = physical_cases.owned[L, g]
    slot = physical_cases.demo["cases"][PAIRS.index((L, g))]
    assert set(slot) == {"L", "g", "status", "reason", "report"}
    assert (slot["L"], slot["g"], slot["status"], slot["reason"]) == (
        L, g, "available_diagnostic", None)
    report = slot["report"]
    assert set(report) == CASE_KEYS
    basis = _basis(L)
    assert report["basis_order"] == [list(state) for state in basis]
    assert (report["L"], report["g"], report["C"], report["cutoff"], report["dimension"]) == (
        L, g, 1, 2, math.comb(L + 2, 2))
    assert report["status"] == "available_diagnostic"
    H, operators = _hamiltonian(basis, g), _fixed_operators(basis)
    np.testing.assert_allclose(owned["H"], H, rtol=4 * EPS, atol=0)
    owned_ops = {operator["name"]: operator for operator in owned["operators"]}
    assert [record["name"] for record in report["operators"]] == list(operators)
    for number, record in enumerate(report["operators"]):
        assert set(record) == OPERATOR_KEYS
        name, A = record["name"], operators[record["name"]]
        expected_class = CLASSES[number] if number < len(CLASSES) else "disconnected_opposite_sites"
        assert record["support_class"] == expected_class
        conserved = name not in ("local_flip", "local_vacuum_projector") and not (
            name == "dark_vacuum_projector" and g == 1)
        assert record["exact_conservation"] is conserved
        np.testing.assert_allclose(owned_ops[name]["matrix"], A, rtol=4 * EPS, atol=0)
        commutator = H @ A - A @ H
        scale = np.abs(H) @ np.abs(A) + np.abs(A) @ np.abs(H)
        allowance = 256 * EPS * len(basis) * scale
        raw = _decode_matrix(record["commutator"])
        _component_close(raw, commutator, allowance)
        expected_norm = math.sqrt(sum(abs(z) ** 2 for z in raw.flat))
        assert record["finite_frobenius_norm"] == pytest.approx(expected_norm, rel=16 * EPS, abs=0)
        if conserved:
            passes = np.all(np.abs(raw.real) <= allowance) and np.all(np.abs(raw.imag) <= allowance)
            assert record["conservation_diagnostic"] == (
                "within_heuristic_allowance" if passes else "outside_heuristic_allowance")
            assert passes
        else:
            assert record["conservation_diagnostic"] == "not_applicable_nonconserved"
        vectors = _witness_vectors(basis, name, g)
        if vectors is None:
            assert record["witness"] is None
            continue
        row, column, expected = vectors
        witness = record["witness"]
        assert set(witness) == {"value", "expected", "allowance", "status",
                                "nonzero_sign_required", "nonzero_sign_satisfied"}
        measured = np.vdot(row, raw @ column)
        witness_allowance = 256 * EPS * len(basis) * float(np.abs(row) @ scale @ np.abs(column))
        _component_close(_decode_scalar(witness["value"]), measured, witness_allowance)
        assert _decode_scalar(witness["expected"]) == expected
        assert witness["allowance"] == pytest.approx(witness_allowance, rel=16 * EPS, abs=0)
        assert witness["status"] == "within_heuristic_allowance"
        assert witness["nonzero_sign_required"] is (expected != 0)
        assert witness["nonzero_sign_satisfied"] is True
        _component_close(measured, expected, witness_allowance)
        if expected:
            assert measured.real * expected > 0


@pytest.mark.parametrize("L,g", PAIRS)
def test_cross_number_flip_and_exact_upper_boundary(physical_cases, L, g):
    owned = physical_cases.owned[L, g]
    basis = _basis(L)
    operators = {item["name"]: item["matrix"] for item in owned["operators"]}
    flip = operators["local_flip"]
    for N in (0, 1, 2):
        indices = [i for i, state in enumerate(basis) if sum(state) == N]
        assert not np.any(flip[np.ix_(indices, indices)])
    vacuum = (0,) * L
    at_zero, at_one, twice_zero = [0] * L, [0] * L, [0] * L
    at_zero[0], at_one[1], twice_zero[0] = 1, 1, 2
    assert flip[basis.index(tuple(at_zero)), basis.index(vacuum)] == 1
    assert not np.any(flip[:, basis.index(tuple(twice_zero))])
    H = _hamiltonian(basis, g)
    assert (H @ flip - flip @ H)[basis.index(tuple(at_one)), basis.index(vacuum)] == -1
    parity = operators["total_number_parity"]
    assert set(np.diag(parity).real) == {-1, 1}
    for N in (0, 1, 2):
        indices = [i for i, state in enumerate(basis) if sum(state) == N]
        np.testing.assert_array_equal(parity[np.ix_(indices, indices)], (-1) ** N * np.eye(len(indices)))


def test_swap_survives_interaction_but_dark_projector_does_not(physical_cases):
    for g in (0, 1):
        records = {item["name"]: item for item in physical_cases.demo["cases"][PAIRS.index((4, g))]["report"]["operators"]}
        swap, dark = records["opposite_site_swap"], records["dark_vacuum_projector"]
        assert swap["exact_conservation"] is True
        assert swap["conservation_diagnostic"] == "within_heuristic_allowance"
        assert dark["exact_conservation"] is (g == 0)
        value = _decode_scalar(dark["witness"]["value"])
        _component_close(value, g / 2, dark["witness"]["allowance"])
        if g:
            assert value.real > 0
            assert dark["finite_frobenius_norm"] > 0
        owned_ops = {item["name"]: item["matrix"] for item in physical_cases.owned[4, g]["operators"]}
        basis = _basis(4)
        # Exterior particles do not destroy dark-mode vacuum membership.
        assert owned_ops["dark_vacuum_projector"][basis.index((0, 1, 0, 0)), basis.index((0, 1, 0, 0))] == 1
        np.testing.assert_allclose(owned_ops["dark_vacuum_projector"] @ owned_ops["dark_vacuum_projector"],
                                   owned_ops["dark_vacuum_projector"], rtol=8 * EPS, atol=0)


def test_interval_hypotheses_and_original_ring_peeling():
    for L in (3, 4, 5, 64):
        for start in range(L):
            for length in range(1, L + 1):
                result = subject.interval_support_report(L, start, length)
                support = [(start + j) % L for j in range(length)]
                assert set(result) == {"L", "start", "length", "support", "complement_size", "status", "peel_steps", "scope"}
                assert (result["L"], result["start"], result["length"], result["support"], result["complement_size"]) == (
                    L, start, length, support, L - length)
                if length <= L - 2:
                    assert result["status"] == "theorem_applies"
                    assert result["peel_steps"] == [
                        {"removed_site": site, "exterior_neighbor": (site - 1) % L,
                         "remaining_support": support[j + 1:]}
                        for j, site in enumerate(support)]
                else:
                    assert result["peel_steps"] == []
                    assert result["status"] == ("outside_theorem_full_support" if length == L else "outside_theorem_one_exterior")
                scope = result["scope"]
                assert scope["checks_support_hypotheses_only"] is True
                assert scope["low_energy_gauge_emergence_excluded"] is False
                conditions = " ".join(scope["conditional_on"]).lower()
                assert "bound" in conditions and "weak" in conditions
                assert "c>0" in conditions.replace(" ", "")
                assert "rank" not in result


@pytest.mark.parametrize("args", [(2, 0, 1), (65, 0, 1), (3, -1, 1), (3, 3, 1),
                                  (3, 0, 0), (3, 0, 4), (True, 0, 1), (3, False, 1),
                                  (3, 0, True), (3.0, 0, 1), (3, 0.0, 1), (3, 0, 1.0),
                                  ("3", 0, 1), (3, None, 1), (np.int64(3), 0, 1),
                                  (3, np.int64(0), 1), (3, 0, np.int64(1))])
def test_interval_invalid_inputs_precede_work(monkeypatch, args):
    monkeypatch.setattr(subject, "_preflight", _forbidden)
    monkeypatch.setattr(subject, "all_number_model", _forbidden)
    with pytest.raises(ValueError):
        subject.interval_support_report(*args)


@pytest.mark.parametrize("L,g", [(2, 0), (6, 0), (3, -1), (3, 2), (True, 0), (3, False),
                                  (3.0, 0), (3, 0.0), ("3", 0), (3, None),
                                  (np.int64(3), 0), (3, np.int64(0))])
def test_case_invalid_inputs_are_programmer_errors(monkeypatch, L, g):
    monkeypatch.setattr(subject, "_owned_case", _forbidden)
    with pytest.raises(ValueError):
        subject.case_report(L, g)


def test_demo_strict_json_and_scope(physical_cases):
    demo = physical_cases.demo
    assert set(demo) == {"module", "cases", "interval_controls", "limitations"}
    assert demo["module"] == "conditional-substrate-local-symmetry-v1"
    assert [(slot["L"], slot["g"]) for slot in demo["cases"]] == list(PAIRS)
    assert [(record["L"], record["start"], record["length"]) for record in demo["interval_controls"]] == (
        [(L, 0, length) for L in (3, 4, 5) for length in range(1, L + 1)]
        + [(3, 2, 1), (4, 3, 2), (5, 4, 3)])
    assert demo["limitations"]
    for slot in demo["cases"]:
        scope = slot["report"]["scope"]
        assert scope["finite_compression_only"] is True
        for key in ("numerical_error_certified", "empirical_validation", "infinite_theorem_from_numerics"):
            assert scope[key] is False
    _strict_roundtrip(demo)


def test_public_reports_detach_owned_arrays(physical_cases):
    owned = copy.deepcopy(physical_cases.owned[3, 0])
    report = subject._case_report(owned)
    snapshot = copy.deepcopy(report)
    owned["H"][:] = 123
    for operator in owned["operators"]:
        operator["matrix"][:] = 456
        if operator["witness"] is not None:
            operator["witness"]["row"][:] = 0
            operator["witness"]["column"][:] = 0
    assert report == snapshot
    report["operators"][0]["commutator"]["real"][0][0] = 789
    assert snapshot["operators"][0]["commutator"]["real"][0][0] != 789
    _strict_roundtrip(snapshot)


def test_case_report_does_not_cache_mutable_results(monkeypatch, physical_cases):
    calls = []
    def make(L, g):
        calls.append((L, g))
        return copy.deepcopy(physical_cases.owned[L, g])
    monkeypatch.setattr(subject, "_owned_case", make)
    first = subject.case_report(3, 0)
    original = copy.deepcopy(first)
    first["basis_order"][0][0] = 999
    first["operators"][0]["commutator"]["real"][0][0] = 999
    assert subject.case_report(3, 0) == original
    assert calls == [(3, 0), (3, 0)]


def test_demo_retains_failure_slots_without_retries(monkeypatch):
    calls = []
    sentinels = {(L, g): {"status": "available_diagnostic", "marker": [L, g],
                           "operators": [{"conservation_diagnostic": "outside_heuristic_allowance"}]}
                 for L, g in PAIRS}
    def fake(L, g):
        calls.append((L, g))
        if (L, g) in ((3, 1), (5, 0)):
            raise subject.NumericalUnavailable("controlled arithmetic failure")
        return sentinels[L, g]
    monkeypatch.setattr(subject, "case_report", fake)
    monkeypatch.setattr(subject, "all_number_model", _forbidden)
    result = subject.demonstration_report()
    assert calls == list(PAIRS)
    assert len(result["cases"]) == 6 and len(result["interval_controls"]) == 15
    for slot, pair in zip(result["cases"], PAIRS):
        assert set(slot) == {"L", "g", "status", "reason", "report"}
        assert (slot["L"], slot["g"]) == pair
        if pair in ((3, 1), (5, 0)):
            assert slot["status"] == "numerical_unavailable"
            assert slot["reason"] and slot["report"] is None
        else:
            assert slot["status"] == "available_diagnostic" and slot["reason"] is None
            assert slot["report"] == sentinels[pair]
    _strict_roundtrip(result)


@pytest.mark.parametrize("error", [ValueError("integration failure"), TypeError("programmer failure"),
                                   RuntimeError("unexpected failure"), FloatingPointError("outside boundary")])
def test_demo_only_catches_numerical_unavailable(monkeypatch, error):
    calls = []
    def fail(L, g):
        calls.append((L, g))
        raise error
    monkeypatch.setattr(subject, "case_report", fail)
    with pytest.raises(type(error), match=str(error)):
        subject.demonstration_report()
    assert calls == [(3, 0)]


@pytest.mark.parametrize("L,cap", [(3, 9), (4, 14), (5, 20)])
def test_initial_cap_blocks_all_builders_maps_and_dense_allocations(monkeypatch, L, cap):
    monkeypatch.setattr(subject, "MAX_DIMENSION", cap)
    monkeypatch.setattr(subject, "all_number_model", _forbidden)
    monkeypatch.setattr(subject, "local_annihilation_map", _forbidden)
    for name in ("zeros", "empty", "ones", "eye", "full", "array"):
        monkeypatch.setattr(np, name, _forbidden)
    with pytest.raises(ValueError):
        subject.case_report(L, 0)


@pytest.mark.parametrize("sector_N", [0, 1, 2])
def test_each_sector_preflight_precedes_builders_and_allocations(monkeypatch, sector_N):
    target = (3 + sector_N - 1, sector_N)
    def capped(n, k, cap=None):
        assert cap == subject.MAX_DIMENSION
        return cap + 1 if (n, k) == target else math.comb(n, k)
    monkeypatch.setattr(subject, "capped_binomial", capped)
    monkeypatch.setattr(subject, "all_number_model", _forbidden)
    monkeypatch.setattr(subject, "local_annihilation_map", _forbidden)
    monkeypatch.setattr(np, "zeros", _forbidden)
    with pytest.raises(ValueError):
        subject.case_report(3, 0)


def test_every_binomial_preflight_receives_live_cap(monkeypatch, fake_inherited):
    calls = []
    def capped(n, k, cap=None):
        assert cap == subject.MAX_DIMENSION == 21
        calls.append((n, k, cap))
        return min(math.comb(n, k), cap + 1)
    monkeypatch.setattr(subject, "MAX_DIMENSION", 21)
    monkeypatch.setattr(subject, "capped_binomial", capped)
    assert subject.case_report(5, 0)["dimension"] == 21
    for n, k in ((7, 2), (4, 0), (5, 1), (6, 2)):
        assert (n, k, 21) in calls
    assert len(fake_inherited.calls) == 5


@pytest.mark.parametrize("stop_after", range(1, 6))
def test_cap_rechecked_after_every_inherited_boundary(monkeypatch, fake_inherited, stop_after):
    def model(*args, **kwargs):
        result = fake_inherited.model(*args, **kwargs)
        if len(fake_inherited.calls) == stop_after:
            monkeypatch.setattr(subject, "MAX_DIMENSION", 1)
        return result
    def ladder(*args, **kwargs):
        result = fake_inherited.ladder(*args, **kwargs)
        if len(fake_inherited.calls) == stop_after:
            monkeypatch.setattr(subject, "MAX_DIMENSION", 1)
        return result
    monkeypatch.setattr(subject, "all_number_model", model)
    monkeypatch.setattr(subject, "local_annihilation_map", ladder)
    with pytest.raises(ValueError):
        subject.case_report(3, 0)
    assert len(fake_inherited.calls) == stop_after


def test_owned_allocation_rechecks_live_cap(monkeypatch):
    subject._preflight(3)
    monkeypatch.setattr(subject, "MAX_DIMENSION", 9)
    monkeypatch.setattr(np, "zeros", _forbidden)
    with pytest.raises(ValueError):
        subject._zeros(3, (10, 10))


def test_reporting_preflight_checks_live_cap_before_allocations(monkeypatch, physical_cases):
    owned = copy.deepcopy(physical_cases.owned[3, 0])
    monkeypatch.setattr(subject, "MAX_DIMENSION", 9)
    monkeypatch.setattr(np, "zeros", _forbidden)
    with pytest.raises(ValueError):
        subject._case_report(owned)


def test_cap_shrink_during_owned_allocation_stops_next_work(monkeypatch, fake_inherited):
    real_zeros = np.zeros
    allocated = []
    def shrinking_zeros(*args, **kwargs):
        assert subject.MAX_DIMENSION >= 10, "allocation after live cap became insufficient"
        result = real_zeros(*args, **kwargs)
        allocated.append(True)
        monkeypatch.setattr(subject, "MAX_DIMENSION", 9)
        return result
    def guarded_model(*args, **kwargs):
        assert subject.MAX_DIMENSION >= 10, "builder after live cap became insufficient"
        return fake_inherited.model(*args, **kwargs)
    def guarded_map(*args, **kwargs):
        assert subject.MAX_DIMENSION >= 10, "map after live cap became insufficient"
        return fake_inherited.ladder(*args, **kwargs)
    monkeypatch.setattr(np, "zeros", shrinking_zeros)
    monkeypatch.setattr(subject, "all_number_model", guarded_model)
    monkeypatch.setattr(subject, "local_annihilation_map", guarded_map)
    with pytest.raises(ValueError):
        subject.case_report(3, 0)
    assert allocated == [True]


ALLOWLIST_LABELS = (
    "C", "g", "hopping scale", "interaction scale", "hopping matrix",
    "interaction diagonal", "annihilation map real component",
    "annihilation map imaginary component",
)


@pytest.mark.parametrize("boundary", ["all_number_model", "local_annihilation_map"])
@pytest.mark.parametrize("label", ALLOWLIST_LABELS)
@pytest.mark.parametrize("reason", ["nonfinite", "subnormal", "underflow"])
def test_exact_inherited_numerical_allowlist(monkeypatch, fake_inherited, boundary, label, reason):
    message = "numerically unresolved " + label + ": " + reason
    def fail(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(subject, boundary, fail)
    with pytest.raises(subject.NumericalUnavailable) as caught:
        subject.case_report(3, 0)
    assert str(caught.value)


@pytest.mark.parametrize("boundary", ["all_number_model", "local_annihilation_map"])
@pytest.mark.parametrize("cause_type", [FloatingPointError, OverflowError, np.linalg.LinAlgError])
def test_inherited_guard_arithmetic_cause_allowlist(monkeypatch, fake_inherited, boundary, cause_type):
    def fail(*args, **kwargs):
        raise ValueError("numerically unresolved arithmetic or singular solve") from cause_type("controlled")
    monkeypatch.setattr(subject, boundary, fail)
    with pytest.raises(subject.NumericalUnavailable):
        subject.case_report(3, 0)


@pytest.mark.parametrize("boundary", ["all_number_model", "local_annihilation_map"])
@pytest.mark.parametrize("message,cause", [
    ("numerically unresolved arithmetic or singular solve", None),
    ("numerically unresolved arithmetic or singular solve", RuntimeError("wrong cause")),
    ("numerically unresolved arithmetic or singular solve ", OverflowError("suffix")),
    ("numerically unresolved Hamiltonian: nonfinite", None),
    ("numerically unresolved hopping matrix: overflow", None),
    ("numerically unresolved hopping matrix: nonfinite extra", None),
    ("prefix numerically unresolved C: nonfinite", None),
    ("fixed-number dimension exceeds cap 512", None),
    ("missing basis", None),
])
def test_near_match_and_programmer_valueerrors_propagate(monkeypatch, fake_inherited, boundary, message, cause):
    error = ValueError(message)
    def fail(*args, **kwargs):
        raise error from cause
    monkeypatch.setattr(subject, boundary, fail)
    with pytest.raises(ValueError) as caught:
        subject.case_report(3, 0)
    assert caught.value is error


@pytest.mark.parametrize("boundary", ["all_number_model", "local_annihilation_map"])
@pytest.mark.parametrize("error", [TypeError("bad API"), KeyError("missing"),
                                   FloatingPointError("unguarded inherited error"),
                                   OverflowError("unguarded inherited error")])
def test_other_inherited_failures_not_translated(monkeypatch, fake_inherited, boundary, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(subject, boundary, fail)
    with pytest.raises(type(error)) as caught:
        subject.case_report(3, 0)
    assert caught.value is error


@pytest.mark.parametrize("error", [FloatingPointError("owned"), OverflowError("owned")])
def test_owned_arithmetic_translated(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(np, "zeros", fail)
    with pytest.raises(subject.NumericalUnavailable):
        subject._zeros(3, (10, 10))


@pytest.mark.parametrize("error", [ValueError("bad shape"), TypeError("bad dtype"), np.linalg.LinAlgError("no solve")])
def test_owned_programmer_errors_not_translated(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(np, "zeros", fail)
    with pytest.raises(type(error)) as caught:
        subject._zeros(3, (10, 10))
    assert caught.value is error


@pytest.mark.parametrize("defect", ["missing_L", "missing_N", "missing_C", "missing_g", "missing_basis", "missing_H",
                                    "L", "N", "C", "g", "reversed_basis", "duplicate_basis", "incomplete_basis",
                                    "wrong_occupation", "wrong_length", "fractional_occupation", "shape", "object_H"])
def test_structural_model_invalidity_is_not_unavailability(monkeypatch, fake_inherited, defect):
    bad = fake_inherited.models[3, 0, 1]
    if defect.startswith("missing_"):
        delattr(bad, defect[len("missing_"):])
    elif defect in ("L", "N", "C", "g"):
        setattr(bad, defect, getattr(bad, defect) + 1)
    elif defect == "reversed_basis":
        bad.basis = tuple(reversed(bad.basis))
    elif defect == "duplicate_basis":
        bad.basis = (bad.basis[0],) * len(bad.basis)
    elif defect == "incomplete_basis":
        bad.basis = bad.basis[:-1]
    elif defect == "wrong_occupation":
        bad.basis = ((-1, 1, 1),) + bad.basis[1:]
    elif defect == "wrong_length":
        bad.basis = ((0, 0, 0, 1),) + bad.basis[1:]
    elif defect == "fractional_occupation":
        bad.basis = ((0, 0.5, 0.5),) + bad.basis[1:]
    elif defect == "shape":
        bad.H = np.zeros((3, 2))
    else:
        bad.H = np.full((3, 3), "not a number", dtype=object)
    with pytest.raises(ValueError):
        subject.case_report(3, 0)
    assert len([call for call in fake_inherited.calls if call[0] == "model"]) == 2


@pytest.mark.parametrize("shape", [(3, 1), (1, 2), (3,), (1, 3, 1)])
def test_ladder_rectangular_shape_checked(monkeypatch, fake_inherited, shape):
    fake_inherited.ladders[3, 1] = np.zeros(shape)
    with pytest.raises(ValueError):
        subject.case_report(3, 0)
    assert len([call for call in fake_inherited.calls if call[0] == "map"]) == 1


@pytest.mark.parametrize("location", ["H_real", "H_imag", "map_real", "map_imag"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inherited_components_unavailable(monkeypatch, fake_inherited, location, bad):
    target = fake_inherited.models[3, 0, 1].H if location.startswith("H") else fake_inherited.ladders[3, 1]
    if location.endswith("real"):
        target.real.flat[0] = bad
    else:
        target.imag.flat[0] = bad
    with pytest.raises(subject.NumericalUnavailable):
        subject.case_report(3, 0)


def _synthetic_owned(H, A, conserved=False, witness=None):
    return {"L": 3, "g": 0, "dimension": 10, "basis": _basis(3), "H": H,
            "operators": [{"name": "local_flip", "support_class": "singleton",
                           "exact_conservation": conserved, "matrix": A, "witness": witness}]}


def test_raw_tiny_complex_residual_not_clipped_and_disagreement_available():
    H = np.zeros((10, 10), dtype=np.complex128)
    A = H.copy()
    H[1, 1] = 2.0 ** -40
    A[1, 0] = 1 + 2j
    report = subject._case_report(_synthetic_owned(H, A, conserved=True))
    assert report["status"] == "available_diagnostic"
    operator = report["operators"][0]
    raw = _decode_matrix(operator["commutator"])
    np.testing.assert_array_equal(raw, H @ A - A @ H)
    assert raw[1, 0] == 2.0 ** -40 * (1 + 2j)
    assert operator["exact_conservation"] is True
    assert operator["conservation_diagnostic"] == "outside_heuristic_allowance"
    assert operator["finite_frobenius_norm"] > 0
    _strict_roundtrip(report)


def test_within_allowance_keeps_precancellation_residual():
    H = np.zeros((10, 10), dtype=np.complex128)
    A = H.copy()
    H[0, 0], H[1, 1], A[1, 0] = 1, 1 + EPS, 1 + 1j
    report = subject._case_report(_synthetic_owned(H, A, conserved=True))
    operator = report["operators"][0]
    raw = _decode_matrix(operator["commutator"])
    assert raw[1, 0] == EPS * (1 + 1j)
    assert operator["conservation_diagnostic"] == "within_heuristic_allowance"
    assert operator["finite_frobenius_norm"] > 0


@pytest.mark.parametrize("imaginary", [False, True])
def test_conserved_allowance_checks_each_component_without_floor(imaginary):
    H = np.zeros((10, 10), dtype=np.complex128)
    A = H.copy()
    H[1, 1] = 2.0 ** -100
    A[1, 0] = 1j if imaginary else 1
    report = subject._case_report(_synthetic_owned(H, A, conserved=True))
    assert report["operators"][0]["conservation_diagnostic"] == "outside_heuristic_allowance"
    assert _decode_matrix(report["operators"][0]["commutator"])[1, 0] != 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("location", ["comm_real", "comm_imag", "comm_unselected", "scale", "scale_unselected", "row_real", "row_imag", "column_real", "column_imag", "expected"])
def test_witness_finiteness_all_components(location, bad):
    commutator = np.zeros((10, 10), dtype=np.complex128)
    scale = np.ones((10, 10))
    row, column = np.zeros(10, dtype=np.complex128), np.zeros(10, dtype=np.complex128)
    row[1], column[0], expected = 1, 1, 0.0
    if location == "comm_real":
        commutator.real[1, 0] = bad
    elif location == "comm_imag":
        commutator.imag[1, 0] = bad
    elif location == "comm_unselected":
        commutator.imag[9, 9] = bad
    elif location == "scale":
        scale[1, 0] = bad
    elif location == "scale_unselected":
        scale[9, 9] = bad
    elif location.startswith("row"):
        (row.real if location.endswith("real") else row.imag)[1] = bad
    elif location.startswith("column"):
        (column.real if location.endswith("real") else column.imag)[0] = bad
    else:
        expected = bad
    with pytest.raises(subject.NumericalUnavailable):
        subject._witness_report(3, commutator, scale, row, column, expected)


@pytest.mark.parametrize("value,expected,scale_entry,passes,sign_required,sign_satisfied", [
    (1 + 1e-3j, 1.0, 1.0, False, True, True),
    (1.001 + 0j, 1.0, 1.0, False, True, True),
    (1 + 0j, 1.0, 1.0, True, True, True),
    (-1 + 0j, 1.0, 1e20, False, True, False),
    (0j, 1.0, 1e20, False, True, False),
    (0j, 0.0, 0.0, True, False, True),
    (1e-30j, 0.0, 0.0, False, False, True),
    (1e-30 + 0j, 0.0, 1e-30, False, False, True),
    (-1 + 0j, -1.0, 1.0, True, True, True),
])
def test_witness_componentwise_error_and_sign_conjunction(value, expected, scale_entry, passes, sign_required, sign_satisfied):
    commutator = np.zeros((10, 10), dtype=np.complex128)
    scale = np.zeros((10, 10))
    row, column = np.zeros(10, dtype=np.complex128), np.zeros(10, dtype=np.complex128)
    row[1], column[0] = 1, 1
    commutator[1, 0], scale[1, 0] = value, scale_entry
    result = subject._witness_report(3, commutator, scale, row, column, expected)
    assert _decode_scalar(result["value"]) == value
    assert _decode_scalar(result["expected"]) == expected
    assert result["allowance"] == 256 * EPS * 10 * scale_entry
    assert result["nonzero_sign_required"] is sign_required
    assert result["nonzero_sign_satisfied"] is sign_satisfied
    assert result["status"] == ("within_heuristic_allowance" if passes else "outside_heuristic_allowance")


def test_witness_uses_absolute_precancellation_scale_and_conjugate_row():
    commutator = np.zeros((10, 10), dtype=np.complex128)
    scale = np.zeros((10, 10))
    row, column = np.zeros(10, dtype=np.complex128), np.zeros(10, dtype=np.complex128)
    row[:2], column[:2] = [1j, -1], [1, 1j]
    commutator[:2, :2] = [[1, 2j], [3j, 4]]
    scale[:2, :2] = [[2, 3], [5, 7]]
    expected = float((np.vdot(row, commutator @ column)).real)
    result = subject._witness_report(3, commutator, scale, row, column, expected)
    assert _decode_scalar(result["value"]) == np.vdot(row, commutator @ column)
    assert result["allowance"] == 256 * EPS * 10 * 17


@pytest.mark.parametrize("location", ["commutator", "scale", "norm", "witness"])
def test_derived_overflow_is_unavailable(location):
    H = np.zeros((10, 10), dtype=np.complex128)
    A = H.copy()
    witness = None
    if location == "commutator":
        H[1, 1], A[1, 0] = 1e308, 4
    elif location == "scale":
        H[0, 0], A[0, 0] = 1e308, 4
    elif location == "norm":
        H[1, 1] = 1e308
        A[1, 0] = A[1, 2] = A[1, 3] = A[1, 4] = 1
    else:
        H[1, 1], A[1, 0] = 1, 1
        row, column = np.zeros(10, dtype=np.complex128), np.zeros(10, dtype=np.complex128)
        row[1], column[0] = 1e308, 4
        witness = {"row": row, "column": column, "expected": 0.0}
    with pytest.raises(subject.NumericalUnavailable):
        subject._case_report(_synthetic_owned(H, A, witness=witness))


@pytest.mark.parametrize("p,q", [(0, 1), (1, 0), (1, 2), (2, 1)])
@pytest.mark.parametrize("orientation", ["exterior_bra", "exterior_ket"])
def test_nonhermitian_boundary_orientations_from_entry_oracle(p, q, orientation):
    """Both weak recurrences independently, never by taking B's adjoint."""
    L, g, coefficient = 3, 1, 1 + 2j
    basis = _basis(L)
    H = _hamiltonian(basis, g)
    B = np.zeros((10, 10), dtype=np.complex128)
    for r, final in enumerate(basis):
        for c, initial in enumerate(basis):
            if final[1:] == initial[1:] and final[0] == p and initial[0] == q:
                B[r, c] = coefficient
    assert not np.array_equal(B, B.conj().T)
    report = subject._case_report(_synthetic_owned(H, B))
    commutator = _decode_matrix(report["operators"][0]["commutator"])
    np.testing.assert_allclose(commutator, H @ B - B @ H, rtol=16 * EPS, atol=0)
    def entry(m, n):
        return coefficient if (m, n) == (p, q) else 0j
    checked_nonzero = False
    for m in range(3):
        for n in range(3):
            if orientation == "exterior_bra":
                bra, ket = (m, 1, 0), (n, 0, 0)
                expected = -(math.sqrt(m + 1) * entry(m + 1, n)
                             - math.sqrt(n) * entry(m, n - 1))
            else:
                bra, ket = (m, 0, 0), (n, 1, 0)
                expected = -(math.sqrt(m) * entry(m - 1, n)
                             - math.sqrt(n + 1) * entry(m, n + 1))
            if bra not in basis or ket not in basis:
                continue
            actual = commutator[basis.index(bra), basis.index(ket)]
            _component_close(actual, expected, 32 * EPS * abs(expected))
            checked_nonzero = checked_nonzero or expected != 0
    assert checked_nonzero


def test_invisible_projector_does_not_prove_full_fock_conservation():
    """A finite zero is compatible with a nonzero next-sector matrix element."""
    low_basis = _basis(3)
    assert all(state[0] < 3 for state in low_basis)
    H = _hamiltonian(low_basis, 1)
    invisible = np.diag([int(state[0] == 3) for state in low_basis])
    assert not np.any(H @ invisible - invisible @ H)
    # <2,1,0|[H,P_(n0=3)]|3,0,0> = -sqrt(3) in the complete N=3 block.
    next_sector = _sector(3, 3)
    H3 = _hamiltonian(next_sector, 1)
    P3 = np.diag([int(state[0] == 3) for state in next_sector])
    assert (H3 @ P3 - P3 @ H3)[next_sector.index((2, 1, 0)), next_sector.index((3, 0, 0))] == -math.sqrt(3)


@pytest.mark.parametrize("json_mode", [False, True])
def test_cli_main_uses_one_mock_report_without_real_demo(monkeypatch, capsys, tmp_path, physical_cases, json_mode):
    cli = importlib.import_module("scripts.demo_substrate_local_symmetry")
    calls = []
    report = copy.deepcopy(physical_cases.demo)
    def fake_demo():
        calls.append(True)
        return report
    monkeypatch.setattr(cli, "demonstration_report", fake_demo)
    monkeypatch.setattr(subject, "all_number_model", _forbidden)
    monkeypatch.setattr(subject, "local_annihilation_map", _forbidden)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    returned = cli.main(["--json"] if json_mode else [])
    assert returned in (None, 0)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert calls == [True]
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_constant(token):
            raise AssertionError("non-strict CLI JSON constant " + token)
        assert json.loads(captured.out, parse_constant=reject_constant) == report
    else:
        text = captured.out.lower()
        assert "conditional" in text and "finite" in text
        assert "theorem" in text and ("diagnostic" in text or "compression" in text)
