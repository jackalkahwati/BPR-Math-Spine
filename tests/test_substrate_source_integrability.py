"""Independent tests of the frozen 2026-09-13 external-source contract.

Authored from the complete derivation, including final-review/author-interface
clarifications, and OLD source/tests only. The implementation and demo were not
read, imported, collected or executed during authorship. All tolerances below
are heuristic screens. In particular tiny fixed-step matrix identities are NOT
certificates for energy finite differences or eigensystem forward error.
"""
from collections import Counter
import copy
from fractions import Fraction
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

from bpr import substrate_fermionization as occupation
from bpr import substrate_gauge_encoding as inherited
from bpr import substrate_source_integrability as api


ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("symmetric", "improved")
BACKGROUNDS = ("uniform", "modulated")
COUPLINGS = (0.0, 0.7, 40.0)
STEPS = (2.0 ** -6, 2.0 ** -7, 2.0 ** -8)
GRID = tuple(itertools.product((3, 4, 5), COUPLINGS, PARTITIONS, BACKGROUNDS))
AVAILABLE = "available_heuristic"
UNAVAILABLE = "numerical_unavailable"
UNRESOLVED = "numerical_unresolved"
MISMATCH = "diagnostic_mismatch"
DIAGNOSTIC = "diagnostic_only"
NA = "not_applicable"
STATUSES = (AVAILABLE, UNAVAILABLE, UNRESOLVED, MISMATCH, DIAGNOSTIC, NA)
PRIORITY = {AVAILABLE: 0, UNRESOLVED: 1, UNAVAILABLE: 2, MISMATCH: 3}
EPS = np.finfo(float).eps
TOL = 2e-10
CASE_KEYS = {"model_id", "L", "N", "g", "C", "partition", "background",
             "dimension", "source_order", "sources", "status", "reason", "scope",
             "operator_checks", "ground", "hessian", "ward_checks", "free_oracle",
             "missing_contact_control", "finite_differences"}
HESSIAN_ARRAYS = ("contact", "spectral", "connected", "hessian",
                  "contact_absolute_scale", "spectral_absolute_scale",
                  "arithmetic_proxy", "arithmetic_resolved", "cancellation_ratio")
FD_FIELDS = {"energies", "resolutions", "weights", "denominator", "energy_proxy",
             "subtraction_proxy", "numerator", "value", "numerator_proxy",
             "derivative_proxy", "target_proxy", "combined_proxy", "target",
             "error", "tolerance", "step", "status", "reason", "agreement"}
GROUND_FIELDS = {"status", "reason", "energy", "ground_gap", "resolution",
                 "orthogonality_residual", "eigenpair_residual", "scale"}


def same(actual, expected, atol=TOL, rtol=TOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def fro(matrix):
    # Independent scaled Frobenius norm, not an eigensolve or API helper.
    flat = np.asarray(matrix).ravel()
    return math.hypot(*(float(x) for z in flat for x in (z.real, z.imag)))


def native(value):
    if type(value) is dict:
        assert all(type(k) is str for k in value)
        for child in value.values():
            native(child)
    elif type(value) is list:
        for child in value:
            native(child)
    else:
        assert value is None or type(value) in (str, bool, int, float)
        if type(value) is float:
            assert math.isfinite(value)


def strict_json(value):
    native(value)
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def statused(record):
    assert record["status"] in STATUSES
    if record["status"] == AVAILABLE:
        assert record["reason"] is None
    elif record["status"] not in (DIAGNOSTIC, NA):
        assert type(record["reason"]) is str and record["reason"]


def aggregate(records):
    statuses = [r["status"] for r in records if r["status"] in PRIORITY]
    return max(statuses, key=PRIORITY.get) if statuses else AVAILABLE


def comparison(record, error=None, scale=None):
    assert {"error", "scale", "tolerance", "consistent", "status", "reason"} <= set(record)
    statused(record)
    if record["error"] is None:
        assert record["consistent"] is None
        return
    assert record["error"] >= 0 and record["scale"] >= 0
    if error is not None:
        same(record["error"], error)
    if scale is not None:
        same(record["scale"], scale)
    tau = TOL + TOL * record["scale"]
    same(record["tolerance"], tau, atol=1e-25, rtol=5e-14)
    assert record["consistent"] is (record["error"] <= tau)
    assert record["status"] == (AVAILABLE if record["consistent"] else MISMATCH)


def complex_scalar(record):
    assert {"real", "imag"} <= set(record)
    return complex(record["real"], record["imag"])


def sources(L, background="modulated"):
    angle = 2 * np.pi * np.arange(L) / L
    f = np.ones(L) if background == "uniform" else 1 + 0.05 * np.cos(angle)
    u = np.zeros(L) if background == "uniform" else 0.05 * np.sin(angle)
    return f, np.full(L, 0.3 / L), u


def incidence(L):
    B = np.zeros((L, L))
    for x in range(L):
        B[x, x] = -1
        B[x, (x + 1) % L] = 1
    return B


def directions(L):
    p, r = np.zeros(3 * L), np.zeros(3 * L)
    p[0], r[L] = 1, 1
    d = np.concatenate((np.cos(2 * np.pi * np.arange(L) / L),
                        np.eye(L)[0], np.sin(2 * np.pi * np.arange(L) / L)))
    return p, r, d


def occupation_oracle(L, N, g, partition, f, A, u, C=1.0):
    """Stars-and-bars plus occupation moves; no inherited/new matrix helpers."""
    length = L + N - 1
    states = []
    for bars in itertools.combinations(range(length), L - 1):
        endpoints = (-1,) + bars + (length,)
        states.append(tuple(endpoints[i + 1] - endpoints[i] - 1 for i in range(L)))
    basis = tuple(sorted(states))
    assert len(basis) == math.comb(length, N)
    assert basis[0] == (0,) * (L - 1) + (N,)
    index = {state: i for i, state in enumerate(basis)}
    dim = len(basis)
    a, b = (0.5, 0.5) if partition == "symmetric" else (0.25, 0.75)
    density, onsite, bonds, currents = [], [], [], []
    for x in range(L):
        density.append(np.diag([state[x] for state in basis]).astype(float))
        onsite.append(np.diag([g * state[x] * (state[x] - 1) / (2 * C)
                               for state in basis]))
        T = np.zeros((dim, dim))
        for col, state in enumerate(basis):
            if state[x]:
                moved = list(state)
                moved[x] -= 1
                moved[(x + 1) % L] += 1
                T[index[tuple(moved)], col] = math.sqrt(
                    state[x] * (state[(x + 1) % L] + 1))
        phase = complex(math.cos(float(A[x])), math.sin(float(A[x])))
        bonds.append(-(phase * T + phase.conjugate() * T.T))
        currents.append(1j * (phase * T - phase.conjugate() * T.T))
    w = b * f + a * np.roll(f, -1)
    H = sum(f[x] * onsite[x] + w[x] * bonds[x] + u[x] * density[x]
            for x in range(L))
    first = ([onsite[y] + a * bonds[y - 1] + b * bonds[y] for y in range(L)]
             + [-w[x] * currents[x] for x in range(L)] + density)

    def second(i, j):
        if i >= 2 * L or j >= 2 * L or (i < L and j < L):
            return np.zeros((dim, dim), dtype=complex)
        if L <= i < 2 * L and L <= j < 2 * L:
            return -w[i - L] * bonds[i - L] if i == j else np.zeros((dim, dim), complex)
        y, x = (i, j - L) if i < L else (j, i - L)
        return -(b * (y == x) + a * (y == (x + 1) % L)) * currents[x]

    return {"basis": basis, "H": H, "density": density, "onsite": onsite,
            "bonds": bonds, "currents": currents, "w": w, "first": first,
            "second": second, "a": a, "b": b, "L": L, "N": N}


def make_system(C=1.0, partition="improved", background="modulated"):
    f, A, u = sources(3, background)
    return api._source_system(3, 3, 0.7, partition, f, A, u, C=C)


def contact_oracle(oracle, ground):
    """Contact evidence remains meaningful when later spectral arithmetic fails."""
    v = ground["ground"]
    p = len(oracle["first"])
    contact, imaginary, scale = (np.empty((p, p)) for _ in range(3))
    for i, j in itertools.product(range(p), repeat=2):
        matrix = oracle["second"](i, j)
        value = np.vdot(v, matrix @ v)
        contact[i, j], imaginary[i, j] = value.real, value.imag
        scale[i, j] = math.fsum(float(abs(v[row]) * abs(matrix[row, col]) * abs(v[col]))
                               for row in range(len(v)) for col in range(len(v)))
    return {"contact": contact, "contact_absolute_scale": scale, "imaginary": imaginary}


def check_unavailable_hessian(record, expected):
    assert record["status"] == UNAVAILABLE and record["reason"]
    if record["contact"] is None:
        # Before contact completion nothing can be advertised as calculated.
        assert all(record[key] is None for key in HESSIAN_ARRAYS)
        return
    # Later spectral failure must not discard completed contact evidence.
    assert expected is not None
    for key in ("contact", "contact_absolute_scale"):
        same(record[key], expected[key])
    realness = record["contact_realness"]
    same(realness["imaginary"], expected["imaginary"])
    comparison(realness, fro(expected["imaginary"]), fro(expected["contact_absolute_scale"]))
    for key in ("spectral", "connected", "hessian"):
        assert record[key] is None
    for key in ("spectral_absolute_scale", "arithmetic_proxy", "arithmetic_resolved", "cancellation_ratio"):
        if record[key] is not None:
            assert key in expected
            same(record[key], expected[key])


def hessian_oracle(oracle, ground):
    """Raw columns and both contact orders; no centering/normalization repair."""
    v = ground["ground"]
    V = ground["vectors"]
    gaps = ground["values"][1:] - ground["values"][0]
    actions = np.column_stack([matrix @ v for matrix in oracle["first"]])
    T = V[:, 1:].conj().T @ actions
    m, p = T.shape
    contact = np.empty((p, p))
    imag = np.empty((p, p))
    Qc = np.empty((p, p))
    spectral = np.empty((p, p))
    Qs = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            matrix = oracle["second"](i, j)
            value = np.vdot(v, matrix @ v)
            contact[i, j], imag[i, j] = value.real, value.imag
            Qc[i, j] = math.fsum(float(abs(v[row]) * abs(matrix[row, col]) * abs(v[col]))
                                  for row in range(len(v)) for col in range(len(v)))
            spectral[i, j] = 2 * math.fsum(float((T[n, i].conjugate() * T[n, j]).real / gaps[n])
                                          for n in range(m))
            Qs[i, j] = 2 * math.fsum(float((abs(T[n, i].real * T[n, j].real)
                                           + abs(T[n, i].imag * T[n, j].imag)) / gaps[n])
                                    for n in range(m))
    K = contact - spectral
    proxy = 64 * EPS * (2 * m + 2) * (Qc + Qs)
    return {"contact": contact, "spectral": spectral, "connected": -spectral,
            "hessian": K, "contact_absolute_scale": Qc, "spectral_absolute_scale": Qs,
            "arithmetic_proxy": proxy, "arithmetic_resolved": np.abs(K) > proxy,
            "imaginary": imag, "transitions": T}


def check_hessian(record, expected):
    assert {"status", "reason", "symmetry", "contact_realness", "arithmetic_scope"} <= set(record)
    assert set(HESSIAN_ARRAYS) <= set(record)
    statused(record)
    if record["hessian"] is None:
        check_unavailable_hessian(record, expected)
        return
    for key in HESSIAN_ARRAYS:
        if key not in ("arithmetic_resolved", "cancellation_ratio"):
            same(record[key], expected[key])
    K = np.array(record["hessian"])
    proxy = np.array(record["arithmetic_proxy"])
    np.testing.assert_array_equal(record["arithmetic_resolved"], np.abs(K) > proxy)
    scale = np.array(record["contact_absolute_scale"]) + np.array(record["spectral_absolute_scale"])
    ratio = record["cancellation_ratio"]
    for i, j in itertools.product(range(len(K)), repeat=2):
        if scale[i, j] == 0:
            assert ratio[i][j] is None
        else:
            same(ratio[i][j], abs(K[i, j]) / scale[i, j], atol=1e-24)
    comparison(record["symmetry"])
    realness = record["contact_realness"]
    same(realness["imaginary"], expected["imaginary"])
    comparison(realness, fro(realness["imaginary"]), fro(record["contact_absolute_scale"]))
    scope = str(record["arithmetic_scope"]).lower()
    for term in ("source", "eigensystem", "action", "projection"):
        assert term in scope


def patch_builder(patch, replacement):
    original = occupation.fixed_number_model
    for module in (api, occupation):
        for name, value in tuple(vars(module).items()):
            if value is original:
                patch.setattr(module, name, replacement)


def forbidden(*args, **kwargs):
    raise AssertionError("operation must not occur at this boundary")


@pytest.fixture(scope="module")
def frozen_demo():
    """One full grid, instrumented. Oracles reuse screened spectra, not solves.

    Dense oracle workspace is released after each case; retained expectations
    have at most 15 by 15 entries. A diagnostic mismatch is retained, not hidden
    by an assertion that all selected finite steps must pass in anticipation.
    """
    observed = {"screen_calls": 0, "eigh_calls": 0, "source_calls": 0,
                "case_keys": [], "derivative_peak": 0, "bond_peak": 0}
    expected = {}
    raw_base = [None]
    current = [None]
    screen_in_case = [0]
    endpoint_screens = []
    source_refs, derivative_refs, bond_refs, eigen_refs = [], [], [], []
    real_case, real_source = api.case_report, api._source_system
    real_screen, real_eigh = api._screened_ground, np.linalg.eigh
    real_first, real_second, real_bond = api._first_derivative, api._second_derivative, api._bond

    def source(*args, **kwargs):
        system = real_source(*args, **kwargs)
        observed["source_calls"] += 1
        source_refs.append(weakref.ref(system["H"]))
        source_refs.append(weakref.ref(system["model"]))
        assert system["H"].ndim == 2
        return system

    def screen(H):
        observed["screen_calls"] += 1
        is_base = screen_in_case[0] == 0
        screen_in_case[0] += 1
        assert sum(ref() is not None for ref in eigen_refs) <= 1
        if is_base:
            L, g, partition, background = current[0]
            f, A, u = sources(L, background)
            independent = occupation_oracle(L, L, g, partition, f, A, u)
            same(H, independent["H"])
            del independent
        result = real_screen(H)
        eigen_refs.append(weakref.ref(result["vectors"]))
        if is_base:
            raw_base[0] = {k: v.copy() if type(v) is np.ndarray else v
                           for k, v in result.items()}
        else:
            L, g, partition, background = current[0]
            endpoint_screens.append((float(result["values"][0]), float(result["resolution"])))
        return result

    def eigh(H, *args, **kwargs):
        observed["eigh_calls"] += 1
        assert H.ndim == 2 and 2 <= len(H) <= 512
        return real_eigh(H, *args, **kwargs)

    def dense_call(function, refs, peak):
        def wrapped(*args, **kwargs):
            live = sum(ref() is not None for ref in refs)
            observed[peak] = max(observed[peak], live)
            # A streamed implementation may retain a few work matrices, not 3L.
            assert live < 9
            result = function(*args, **kwargs)
            for matrix in result if type(result) is tuple else (result,):
                assert type(matrix) is np.ndarray and matrix.ndim == 2
                refs.append(weakref.ref(matrix))
            return result
        return wrapped

    def case(L, N, g, partition="symmetric", background="uniform"):
        # No dense case/model cache can survive into the next labeled case.
        assert all(ref() is None for ref in source_refs + eigen_refs)
        raw_base[0] = None
        screen_in_case[0] = 0
        endpoint_screens.clear()
        key = (L, g, partition, background)
        current[0] = key
        observed["case_keys"].append(key)
        result = real_case(L, N, g, partition=partition, background=background)
        f, A, u = sources(L, background)
        oracle = occupation_oracle(L, N, g, partition, f, A, u)
        item = {"work": work_oracle(oracle), "energy": None, "hessian": None}
        if result["ground"]["energy"] is not None:
            assert raw_base[0] is not None
            item["energy"] = float(raw_base[0]["values"][0])
            item["hessian"] = contact_oracle(oracle, raw_base[0])
            if result["hessian"]["hessian"] is not None:
                item["hessian"] = hessian_oracle(oracle, raw_base[0])
                item["hessian"].pop("transitions")
        retained_endpoints = [e for t in result["finite_differences"]["stencils"]
                              for s in t["steps"] for e in s["endpoints"]
                              if any(e["offsets"]) and e["status"] == AVAILABLE]
        assert len(retained_endpoints) == len(endpoint_screens)
        for endpoint in retained_endpoints:
            pair = (endpoint["energy"], endpoint["resolution"])
            assert pair in endpoint_screens
            endpoint_screens.remove(pair)
        expected[key] = item
        raw_base[0] = None
        return result

    def allocation_guard(function):
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            if type(result) is np.ndarray and result.ndim >= 3:
                assert not (result.shape[-1] >= 10 and result.shape[-2] >= 10), (
                    "dense derivative/contact/eigensystem stacks are forbidden", result.shape)
            return result
        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "case_report", case)
        patch.setattr(api, "_source_system", source)
        patch.setattr(api, "_screened_ground", screen)
        patch.setattr(api, "_first_derivative", dense_call(real_first, derivative_refs, "derivative_peak"))
        patch.setattr(api, "_second_derivative", dense_call(real_second, derivative_refs, "derivative_peak"))
        patch.setattr(api, "_bond", dense_call(real_bond, bond_refs, "bond_peak"))
        patch.setattr(np.linalg, "eigh", eigh)
        patch.setattr(np.linalg, "eigvalsh", forbidden)
        patch.setattr(np.linalg, "eig", forbidden)
        for name in ("zeros", "empty", "ones", "full", "array", "stack"):
            patch.setattr(np, name, allocation_guard(getattr(np, name)))
        report = api.demonstration_report()
    assert all(ref() is None for ref in source_refs)
    return report, expected, observed


def work_oracle(oracle):
    L = oracle["L"]
    _, _, rates = directions(L)
    df, dA, du = np.split(rates, 3)
    chain = sum(df[x] * oracle["onsite"][x]
                + (oracle["b"] * df[x] + oracle["a"] * df[(x + 1) % L]) * oracle["bonds"][x]
                - dA[x] * oracle["w"][x] * oracle["currents"][x]
                + du[x] * oracle["density"][x] for x in range(L))
    k = np.arange(len(chain)) % 3
    psi = (1 + 1j * k) / math.sqrt(sum(1 + k * k))
    return {"value": np.vdot(psi, chain @ psi), "matrix_scale": 2 * fro(chain)}


def test_public_and_documented_private_signatures_and_exception_identity():
    expected = {"case_report": ("L", "N", "g", "partition", "background"),
                "demonstration_report": (),
                "_source_system": ("L", "N", "g", "partition", "f", "A", "u", "C"),
                "_bond": ("system", "x"), "_first_derivative": ("system", "index"),
                "_second_derivative": ("system", "i", "j"), "_screened_ground": ("H",),
                "_hessian_from_transitions": ("transitions", "gaps", "contact"),
                "_finite_difference_record": ("energies", "resolutions", "weights", "denominator",
                                              "target", "target_proxy", "step", "diagnostic_only")}
    for name, parameters in expected.items():
        assert tuple(inspect.signature(getattr(api, name)).parameters) == parameters
    assert api.NumericalUnavailable is inherited.NumericalUnavailable
    assert api._screened_eigensystem is inherited._screened_eigensystem
    sig = inspect.signature(api.case_report)
    assert sig.parameters["partition"].default == "symmetric"
    assert sig.parameters["background"].default == "uniform"


def test_full_grid_shape_order_counts_statuses_and_solve_budget(frozen_demo):
    demo, _, calls = frozen_demo
    assert {"model_id", "status", "reason", "scope", "controls", "cases", "summary"} <= set(demo)
    assert demo["model_id"] == "substrate_source_integrability"
    assert type(demo["scope"]) is str
    expected_controls = {"sectors": [[3, 3], [4, 4], [5, 5]], "couplings": list(COUPLINGS),
                         "C": 1.0, "partitions": list(PARTITIONS), "backgrounds": list(BACKGROUNDS),
                         "total_flux": 0.3, "source_order": "f,A,u", "energy_normalization": "H/C",
                         "gauge_amplitude": 0.2, "fd_steps": list(STEPS),
                         "fd_anchor": {"L": 3, "N": 3, "g": 0.7},
                         "fd_directions": ["link", "mixed", "joint"],
                         "fd_tolerance": 0.0001, "max_ground_solves": 132}
    assert demo["controls"] == expected_controls
    cases = demo["cases"]
    assert [(r["L"], r["g"], r["partition"], r["background"]) for r in cases] == list(GRID)
    assert calls["case_keys"] == list(GRID)
    assert calls["eigh_calls"] == calls["screen_calls"] <= 132
    assert calls["screen_calls"] >= 36
    steps = [s for c in cases for t in c["finite_differences"]["stencils"] for s in t["steps"]]
    assert len(steps) == 36
    assert sum(len(s["endpoints"]) for s in steps) == 120
    endpoints = [e for s in steps for e in s["endpoints"]]
    assert sum(not any(e["offsets"]) for e in endpoints) == 24
    assert sum(any(e["offsets"]) for e in endpoints) == 96
    if all(c["ground"]["status"] == AVAILABLE and c["hessian"]["hessian"] is not None for c in cases):
        assert calls["screen_calls"] == calls["eigh_calls"] == 132
    summary = demo["summary"]
    assert summary["case_count"] == 36
    assert summary["ground_available_count"] == sum(c["ground"]["status"] == AVAILABLE for c in cases)
    assert summary["fd_anchor_count"] == 4
    assert summary["fd_stencil_count"] == 12
    assert summary["fd_step_count"] == 36
    assert summary["free_oracle_count"] == 6
    assert summary["missing_contact_control_count"] == 2
    for name, records in (("case_status_counts", cases), ("fd_step_status_counts", steps)):
        counts = Counter(r["status"] for r in records)
        assert summary[name] == {s: counts[s] for s in STATUSES}
    assert demo["status"] == aggregate(cases)
    strict_json(demo)


@pytest.mark.parametrize("key", GRID)
def test_case_schema_raw_hessian_work_and_entire_ward_subspaces(frozen_demo, key):
    demo, expected, _ = frozen_demo
    case = demo["cases"][GRID.index(key)]
    L, g, partition, background = key
    assert CASE_KEYS <= set(case)
    assert case["model_id"] == "substrate_source_integrability"
    assert case["L"] == case["N"] == L
    assert case["dimension"] == math.comb(2 * L - 1, L)
    assert case["C"] == 1
    assert case["source_order"] == ["%s%d" % (s, x) for s in ("f", "A", "u") for x in range(L)]
    for name, value in zip(("f", "A", "u"), sources(L, background)):
        same(case["sources"][name], value)
    assert type(case["scope"]) is str and "metric" in case["scope"].lower()
    assert GROUND_FIELDS <= set(case["ground"])
    for field in GROUND_FIELDS - {"status", "reason"}:
        assert case["ground"][field] is None or type(case["ground"][field]) in (float, int)
    statused(case)
    statused(case["ground"])
    checks = case["operator_checks"]
    assert {"gauge_covariance", "continuity", "density_sum", "work_identity"} <= set(checks)
    for name in ("gauge_covariance", "continuity", "density_sum"):
        comparison(checks[name])
    work = checks["work_identity"]
    for name in ("expected", "observed"):
        same(complex_scalar(work[name]), expected[key]["work"]["value"])
    same(work["matrix_scale"], expected[key]["work"]["matrix_scale"])
    same(work["matrix_tolerance"], TOL + TOL * work["matrix_scale"])
    assert work["matrix_consistent"] is (work["matrix_error"] <= work["matrix_tolerance"])
    assert work["consistent"] is (work["error"] <= work["tolerance"] and work["matrix_consistent"])
    if case["ground"]["energy"] is not None:
        same(case["ground"]["energy"], expected[key]["energy"])
    hessian = case["hessian"]
    if hessian["hessian"] is None:
        check_hessian(hessian, expected[key]["hessian"])
        for record in case["ward_checks"].values():
            assert record["status"] == UNAVAILABLE and record["reason"]
            assert record["completed_contraction"] is None
        return
    check_hessian(hessian, expected[key]["hessian"])
    C, S, K = (np.asarray(hessian[name]) for name in ("contact", "spectral", "hessian"))
    B = incidence(L)
    rows = {"fA": slice(0, L), "AA": slice(L, 2 * L), "uA": slice(2 * L, 3 * L)}
    for name, row in rows.items():
        record = case["ward_checks"][name]
        c, s, k = C[row, L:2 * L] @ B, S[row, L:2 * L] @ B, K[row, L:2 * L] @ B
        same(record["contact_contraction"], c)
        same(record["spectral_contraction"], s)
        same(record["completed_contraction"], k)
        comparison(record, fro(k), fro(c) + fro(s))
        assert record["analytic"]
    record = case["ward_checks"]["constant_density"]
    c, s, k = (matrix[:, 2 * L:] @ np.ones(L) for matrix in (C, S, K))
    same(record["contact_contraction"], c)
    same(record["spectral_contraction"], s)
    same(record["completed_contraction"], k)
    comparison(record, fro(k), fro(c) + fro(s))


@pytest.mark.parametrize("C", (0.5, 1.0, 2.0))
@pytest.mark.parametrize("partition", PARTITIONS)
def test_occupation_move_source_assembly_units_contacts_and_partition(C, partition):
    system = make_system(C, partition)
    oracle = occupation_oracle(3, 3, 0.7, partition, system["f"], system["A"], system["u"], C)
    assert {"L", "N", "g", "C", "partition", "basis", "occupations", "f", "A", "u", "w", "H", "model"} <= set(system)
    assert tuple(tuple(s) for s in system["basis"]) == oracle["basis"]
    same(system["occupations"], oracle["basis"])
    same(system["H"], oracle["H"])
    same(system["w"], oracle["w"])
    assert system["model"].C == C and system["model"].g == 0.7
    unsourced = occupation_oracle(3, 3, 0.7, partition, np.ones(3), np.zeros(3), np.zeros(3), C)
    same(system["model"].H, C * unsourced["H"])
    assert fro(system["H"] - system["model"].H / C) > 0.01
    for x in range(3):
        V, j = api._bond(system, x)
        same(V, oracle["bonds"][x])
        same(j, oracle["currents"][x])
        same(V, V.conj().T, atol=0, rtol=0)
        same(j, j.conj().T, atol=0, rtol=0)
    for i in range(9):
        first = api._first_derivative(system, i)
        same(first, oracle["first"][i])
        for j in range(9):
            contact = api._second_derivative(system, i, j)
            same(contact, oracle["second"](i, j))
            if (i < 3 and j < 3) or i >= 6 or j >= 6 or (3 <= i < 6 and 3 <= j < 6 and i != j):
                assert np.count_nonzero(contact) == 0
    for y in range(3):
        n = oracle["density"][y]
        lhs = 1j * (system["H"] @ n - n @ system["H"])
        rhs = oracle["w"][y - 1] * oracle["currents"][y - 1] - oracle["w"][y] * oracle["currents"][y]
        same(lhs, rhs)
    assert np.max(np.abs(system["H"].imag)) > 0


@pytest.mark.parametrize("h", (2.0 ** -8, 2.0 ** -9))
@pytest.mark.parametrize("partition", PARTITIONS)
def test_tiny_fixed_step_matrix_identities_use_sine_factors_not_limits(h, partition):
    system = make_system(partition=partition)
    f, A, u = (system[name].copy() for name in ("f", "A", "u"))
    oracle = occupation_oracle(3, 3, 0.7, partition, f, A, u)
    base = oracle["H"]
    dim = len(base)

    def matrix(df=0.0, dA=0.0, du=0.0):
        ff, AA, uu = f.copy(), A.copy(), u.copy()
        ff[0] += df
        AA[0] += dA
        uu[0] += du
        # Independent displacement assembly rather than the helper under test.
        return occupation_oracle(3, 3, 0.7, partition, ff, AA, uu)["H"]

    def stencil(matrices, coefficients, reference):
        value = sum(c * H for c, H in zip(coefficients, matrices))
        scale = sum(abs(c) * fro(H) for c, H in zip(coefficients, matrices)) + fro(reference)
        assert fro(value - reference) <= 128 * EPS * dim * scale

    sine = math.sin(h) / h
    firstA = api._first_derivative(system, 3)
    secondAA = api._second_derivative(system, 3, 3)
    mixed = api._second_derivative(system, 0, 3)
    stencil([matrix(dA=h), matrix(dA=-h)], [1 / (2 * h), -1 / (2 * h)], sine * firstA)
    stencil([matrix(dA=h), base, matrix(dA=-h)], [1 / h**2, -2 / h**2, 1 / h**2],
            4 * math.sin(h / 2)**2 / h**2 * secondAA)
    stencil([matrix(df=h, dA=h), matrix(df=h, dA=-h), matrix(df=-h, dA=h), matrix(df=-h, dA=-h)],
            [1 / (4 * h**2), -1 / (4 * h**2), -1 / (4 * h**2), 1 / (4 * h**2)], sine * mixed)
    for coordinate, keyword in ((0, "df"), (6, "du")):
        stencil([matrix(**{keyword: h}), matrix(**{keyword: -h})],
                [1 / (2 * h), -1 / (2 * h)], api._first_derivative(system, coordinate))
    assert abs(sine - 1) > 1000 * EPS


def test_gauge_sign_partition_shift_and_raw_constant_density_without_solve():
    f, A, u = sources(3)
    systems = {p: api._source_system(3, 3, 0.7, p, f, A, u) for p in PARTITIONS}
    oracle = occupation_oracle(3, 3, 0.7, "symmetric", f, A, u)
    delta = sum(0.25 * (f[x] - f[(x + 1) % 3]) * oracle["bonds"][x] for x in range(3))
    same(systems["improved"]["H"] - systems["symmetric"]["H"], delta)
    theta = 0.2 * np.cos(2 * np.pi * np.arange(3) / 3)
    G = np.array(oracle["basis"]) @ theta
    phase = np.exp(1j * G)
    for partition, system in systems.items():
        shifted = api._source_system(3, 3, 0.7, partition, f, A + incidence(3) @ theta, u)
        same(shifted["H"], phase[:, None] * system["H"] * phase.conj()[None, :])
        density = sum(api._first_derivative(system, 6 + x) for x in range(3))
        np.testing.assert_array_equal(density, 3 * np.eye(10))
        shift = api._source_system(3, 3, 0.7, partition, f, A, u + 0.125)
        same(shift["H"] - system["H"], 0.375 * np.eye(10))
        along = sum((incidence(3) @ theta)[x] * api._first_derivative(system, 3 + x) for x in range(3))
        same(along, 1j * (G[:, None] * system["H"] - system["H"] * G[None, :]))
        uniform = api._source_system(3, 3, 0.7, partition, np.ones(3), A, np.zeros(3))
        same(sum(api._first_derivative(uniform, x) for x in range(3)), uniform["H"])
        assert fro(sum(api._second_derivative(uniform, y, 3) for y in range(3))) > 0


def test_all_free_closed_forms_and_missing_contact_negative_controls(frozen_demo):
    demo, _, _ = frozen_demo
    for case in demo["cases"]:
        L, g = case["L"], case["g"]
        free = case["free_oracle"]
        assert {"status", "reason", "expected_energy", "energy_comparison", "expected_AA", "AA_comparison"} <= set(free)
        if g != 0 or case["background"] != "uniform":
            assert free["status"] == NA
            assert case["missing_contact_control"]["status"] == NA
            continue
        E = -2 * L * math.cos(0.3 / L)
        AA = np.full((L, L), 2 * L * math.cos(0.3 / L) / L**2)
        same(free["expected_energy"], E)
        same(free["expected_AA"], AA)
        if case["ground"]["energy"] is not None:
            same(case["ground"]["energy"], E)
            comparison(free["energy_comparison"])
        if case["hessian"]["hessian"] is not None:
            same(np.array(case["hessian"]["hessian"])[L:2 * L, L:2 * L], AA)
            comparison(free["AA_comparison"])
        negative = case["missing_contact_control"]
        if L != 3:
            assert negative["status"] == NA
            continue
        assert {"status", "reason", "d", "expected_AA", "observed_AA", "AA_comparison",
                "expected_fA", "observed_fA", "fA_comparison"} <= set(negative)
        d = np.array([-1.0, 0.0, 1.0])
        a, b = (0.5, 0.5) if case["partition"] == "symmetric" else (0.25, 0.75)
        expected_AA = -2 * math.cos(0.1) * d
        expected_fA = -2 * math.sin(0.1) * np.array([a - b, -a, b])
        same(negative["d"], d)
        same(negative["expected_AA"], expected_AA)
        same(negative["expected_fA"], expected_fA)
        assert fro(expected_AA) > 1 and fro(expected_fA) > 0.05
        if case["hessian"]["hessian"] is not None:
            S = np.array(case["hessian"]["spectral"])
            same(negative["observed_AA"], -S[3:6, 3:6] @ d)
            same(negative["observed_fA"], -S[:3, 3:6] @ d)
            same(negative["observed_AA"], expected_AA)
            same(negative["observed_fA"], expected_fA)
            comparison(negative["AA_comparison"])
            comparison(negative["fA_comparison"])
            assert negative["status"] == AVAILABLE


class ConversionTrap:
    def __float__(self):
        raise AssertionError("custom scalar conversion must never run")

    def __array__(self, *args, **kwargs):
        raise AssertionError("custom array conversion must never run")


class ArraySubclass(np.ndarray):
    pass


class FloatSubclass(float):
    pass


class IntSubclass(int):
    pass


class StringSubclass(str):
    pass


@pytest.mark.parametrize("replace", [
    {"L": True}, {"N": np.bool_(True)}, {"L": 3.0}, {"N": "3"},
    {"L": IntSubclass(3)}, {"L": ConversionTrap()}, {"L": 2}, {"L": 6},
    {"L": 4, "N": 3}, {"g": True}, {"g": np.bool_(False)}, {"g": "0.7"},
    {"g": 0.7 + 0j}, {"g": np.float32(0.7)}, {"g": FloatSubclass(0.7)},
    {"g": ConversionTrap()}, {"g": 0.7000000000000001}, {"g": float("nan")},
    {"g": float("inf")}, {"g": -1}, {"g": 10**400},
    {"partition": StringSubclass("symmetric")}, {"partition": "SYMMETRIC"},
    {"background": StringSubclass("uniform")}, {"background": "other"},
])
def test_public_original_scalar_validation_precedes_source_allocation(replace, monkeypatch):
    monkeypatch.setattr(api, "_source_system", forbidden)
    patch_builder(monkeypatch, forbidden)
    arguments = dict(L=3, N=3, g=0.7, partition="symmetric", background="uniform")
    arguments.update(replace)
    with pytest.raises(ValueError):
        api.case_report(**arguments)


@pytest.mark.parametrize("extra", ("C", "model", "cache", "eigensystem", "steps", "directions"))
def test_public_has_no_external_cache_or_control_override(extra):
    with pytest.raises(TypeError):
        api.case_report(3, 3, 0.7, **{extra: None})
    with pytest.raises(TypeError):
        api.demonstration_report(**{extra: None})


@pytest.mark.parametrize("L,N,g", [(3, 3, 0), (np.int64(3), np.int32(3), np.float64(0.7)),
                                    (np.uint32(4), np.int64(4), np.float32(40)),
                                    (5, np.int64(5), np.longdouble(0.7))])
def test_exact_original_scalar_members_reach_construction(L, N, g, monkeypatch):
    class Reached(Exception):
        pass

    def stop(*args, **kwargs):
        raise Reached()

    monkeypatch.setattr(api, "_source_system", stop)
    with pytest.raises(Reached):
        api.case_report(L, N, g)


def test_extended_precision_membership_never_snaps_to_stored_coupling(monkeypatch):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble has no extra significand precision")
    g = np.nextafter(np.longdouble(0.7), np.longdouble(1))
    assert g != np.longdouble(0.7) and float(g) == 0.7
    monkeypatch.setattr(api, "_source_system", forbidden)
    with pytest.raises(ValueError):
        api.case_report(3, 3, g)


@pytest.mark.parametrize("name,bad", [
    ("f", [1, 1, 1]), ("f", ConversionTrap()), ("f", np.ones((1, 3))),
    ("f", np.ones((3, 1))), ("f", np.ones(4)), ("f", np.ones(3, dtype=bool)),
    ("f", np.ones(3, dtype=object)), ("f", np.ones(3, dtype=complex)),
    ("f", np.ones(3).view(ArraySubclass)), ("f", np.array([0.49, 1, 1])),
    ("f", np.array([1.51, 1, 1])), ("A", np.array([2.01, 0, 0])),
    ("A", np.array([-2.01, 0, 0])), ("A", np.array([2 * np.pi, 0, 0])),
    ("u", np.array([1.01, 0, 0])), ("u", np.zeros(0)),
    ("C", True), ("C", "1"), ("C", FloatSubclass(1)), ("C", ConversionTrap()),
    ("C", 0.49), ("C", 2.01), ("C", 1j),
])
def test_private_source_shape_type_domain_before_inherited_builder(name, bad, monkeypatch):
    patch_builder(monkeypatch, forbidden)
    f, A, u = sources(3)
    arguments = dict(L=3, N=3, g=0.7, partition="symmetric", f=f, A=A, u=u, C=1.0)
    arguments[name] = bad
    with pytest.raises(ValueError):
        api._source_system(**arguments)


@pytest.mark.parametrize("name,bad", [
    ("f", np.array([np.nan, 1, 1])), ("A", np.array([np.inf, 0, 0])),
    ("u", np.array([np.nextafter(0.0, 1.0), 0, 0])),
    ("A", np.array([np.finfo(float).tiny / 2, 0, 0])),
])
def test_source_nonfinite_or_subnormal_original_components_are_unavailable(name, bad, monkeypatch):
    patch_builder(monkeypatch, forbidden)
    f, A, u = sources(3)
    arguments = dict(L=3, N=3, g=0.7, partition="symmetric", f=f, A=A, u=u)
    arguments[name] = bad
    with pytest.raises(api.NumericalUnavailable):
        api._source_system(**arguments)


def test_sources_reject_value_changing_extended_conversion_before_allocation(monkeypatch):
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble has no extra significand precision")
    f, A, u = sources(3)
    extended = f.astype(np.longdouble)
    extended[0] = np.nextafter(np.longdouble(1), np.longdouble(2))
    assert extended[0] != np.longdouble(float(extended[0]))
    patch_builder(monkeypatch, forbidden)
    with pytest.raises(api.NumericalUnavailable):
        api._source_system(3, 3, 0.7, "symmetric", extended, A, u)


def test_out_of_grid_oversized_sector_rejected_before_basis_or_dense_allocation(monkeypatch):
    # Public sectors already imply dimensions <=126. Test the oversized input
    # boundary without inventing an undocumented mutable MAX_DIMENSION seam.
    patch_builder(monkeypatch, forbidden)
    monkeypatch.setattr(occupation, "_occupations", forbidden)
    f, A, u = np.ones(12), np.zeros(12), np.zeros(12)
    with pytest.raises(ValueError):
        api._source_system(12, 12, 0.7, "symmetric", f, A, u)


@pytest.mark.parametrize("helper", ("_bond", "_first_derivative", "_second_derivative"))
@pytest.mark.parametrize("bad", (True, np.bool_(True), 0.0, "0", IntSubclass(0), -1, 100, ConversionTrap()))
def test_coordinate_indices_are_original_nonbool_integers(helper, bad):
    system = make_system()
    arguments = (system, bad, 0) if helper == "_second_derivative" else (system, bad)
    with pytest.raises(ValueError):
        getattr(api, helper)(*arguments)
    if helper == "_second_derivative":
        with pytest.raises(ValueError):
            api._second_derivative(system, 0, bad)


def test_source_and_derivative_arrays_are_owned_without_cross_call_cache():
    f, A, u = sources(3)
    first = api._source_system(3, 3, 0.7, "improved", f, A, u)
    second = api._source_system(3, 3, 0.7, "improved", f, A, u)
    original = {key: first[key].copy() for key in ("f", "A", "u", "w", "H", "occupations")}
    for key in original:
        assert type(first[key]) is np.ndarray and first[key].flags.owndata
        assert not np.shares_memory(first[key], second[key])
    for name, caller in (("f", f), ("A", A), ("u", u)):
        assert not np.shares_memory(first[name], caller)
        caller[:] = 0
        np.testing.assert_array_equal(first[name], original[name])
    for key in original:
        first[key].flat[0] = 123
        np.testing.assert_array_equal(second[key], original[key])
    for helper, arguments in ((api._bond, (second, 0)),
                              (api._first_derivative, (second, np.int64(0))),
                              (api._second_derivative, (second, np.int32(0), np.int64(3)))):
        value = helper(*arguments)
        arrays = value if type(value) is tuple else (value,)
        expected = [matrix.copy() for matrix in arrays]
        for matrix in arrays:
            assert matrix.flags.owndata
            matrix[:] = 99
        repeat = helper(*arguments)
        repeats = repeat if type(repeat) is tuple else (repeat,)
        for matrix, reference in zip(repeats, expected):
            same(matrix, reference)


@pytest.mark.parametrize("H", [np.array([[2.0, 1j], [-1j, 2.0]]),
                               np.diag([0.0, 2.0, 2.0]),
                               np.diag([-3.0, -1.0, 4.0])])
def test_screened_ground_complex_hermitian_and_excited_degeneracy(H, monkeypatch):
    original = H.copy()
    calls = []
    real_eigh = np.linalg.eigh

    def eigh(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return real_eigh(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", eigh)
    result = api._screened_ground(H)
    assert len(calls) == 1
    required = {"values", "vectors", "ground", "gaps", "ground_gap", "resolution",
                "orthogonality_residual", "eigenpair_residual", "scale"}
    assert required <= set(result)
    d = len(H)
    values, vectors = result["values"], result["vectors"]
    same(H @ vectors, vectors * values)
    same(vectors.conj().T @ vectors, np.eye(d))
    np.testing.assert_array_equal(result["ground"], vectors[:, 0])
    np.testing.assert_array_equal(result["gaps"], values - values[0])
    assert result["gaps"].shape == (d,) and result["gaps"][0] == 0
    assert np.all(np.diff(values) >= 0)
    assert np.all(result["gaps"][1:] > 0)
    scale = fro(H)
    residual = fro(H @ vectors - vectors * values)
    same(result["scale"], scale)
    same(result["resolution"], max(64 * EPS * d * scale, 8 * residual), atol=1e-25)
    assert result["ground_gap"] == result["gaps"][1] > result["resolution"]
    if H.shape == (2, 2):
        np.testing.assert_allclose(values, [1., 3.], atol=1e-14, rtol=1e-14)
        same(result["ground"][1] / result["ground"][0], 1j)
    for name in ("values", "vectors", "ground", "gaps"):
        assert type(result[name]) is np.ndarray and result[name].flags.owndata
        assert not np.shares_memory(result[name], H)
    assert not np.shares_memory(result["ground"], result["vectors"])
    H[:] = 0
    same(original @ vectors, vectors * values)


@pytest.mark.parametrize("H", [np.diag([0.0, 0.0, 1.0]), np.zeros((2, 2)),
                               np.diag([1.0, 1.0 + 2**-50])])
def test_degenerate_zero_scale_or_unresolved_ground_is_unavailable(H):
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(H)


@pytest.mark.parametrize("H", [np.eye(2).tolist(), np.eye(2).view(ArraySubclass),
                               np.ones((1, 1)), np.ones((513, 513)), np.ones((2, 3)),
                               np.eye(2, dtype=bool), np.eye(2, dtype=object), ConversionTrap(),
                               np.array([[1.0, 1e-30], [0.0, 2.0]]),
                               np.array([[1 + 1e-30j, 0], [0, 2]])])
def test_screened_ground_strict_input_shape_type_exact_hermiticity_before_solver(H, monkeypatch):
    monkeypatch.setattr(api, "_screened_eigensystem", forbidden)
    with pytest.raises(ValueError):
        api._screened_ground(H)


@pytest.mark.parametrize("H", [np.diag([np.nan, 1.0]), np.diag([np.inf, 1.0]),
                               np.diag([np.nextafter(0.0, 1.0), 1.0]),
                               np.diag(np.array([2**53 + 1, 2**53 + 3], dtype=np.int64))])
def test_screened_ground_nonfinite_subnormal_and_lossy_integer_conversion(H, monkeypatch):
    monkeypatch.setattr(api, "_screened_eigensystem", forbidden)
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(H)


@pytest.mark.parametrize("values,vectors", [
    ([0.0, 1.0], np.eye(2)), (np.zeros((2, 1)), np.eye(2)),
    (np.zeros(3), np.eye(2)), (np.array([0j, 1j]), np.eye(2)),
    (np.array([0.0, np.nan]), np.eye(2)), (np.array([0.0, 1.0]), [1, 0]),
    (np.array([0.0, 1.0]), np.ones((2, 1))),
    (np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.0, np.inf]])),
    (np.array([0.0, 1.0]), np.ones((2, 2))),
])
def test_malformed_solver_outputs_are_screened_before_broadcasting(values, vectors, monkeypatch):
    monkeypatch.setattr(np.linalg, "eigh", lambda *args, **kwargs: (values, vectors))
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(np.diag([0.0, 1.0]))


def test_sorted_output_required_even_when_all_eigenpairs_are_correct(monkeypatch):
    H = np.diag([0.0, 1.0, 2.0])
    # The inherited scaled solver sees H / sqrt(5). Every permuted eigenpair
    # remains exact, but ordering [0,2,1] violates the new wrapper contract.
    def shuffled(matrix, *args, **kwargs):
        return np.diag(matrix)[[0, 2, 1]].copy(), np.eye(3)[:, [0, 2, 1]].copy()

    monkeypatch.setattr(np.linalg, "eigh", shuffled)
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(H)


def test_actual_tiny_norm_screen_has_no_unit_floor_and_allows_solver_subnormals(monkeypatch):
    tiny = 1e-150
    H = np.diag([0.0, tiny])
    result = api._screened_ground(H)
    assert result["resolution"] < tiny * 1e-10
    same(result["scale"], tiny, atol=0, rtol=1e-14)
    assert result["ground_gap"] > result["resolution"]

    def raw_columns(matrix, *args, **kwargs):
        return np.diag(matrix).copy(), np.array([[1.0, 0.0], [1e-310, 1.0]])

    monkeypatch.setattr(np.linalg, "eigh", raw_columns)
    result = api._screened_ground(np.diag([0.0, 1.0]))
    assert result["ground"][1] == 1e-310


def test_ground_tighter_orthogonality_and_residual_screens_than_inherited(monkeypatch):
    H = np.diag([0.0, 1.0])

    def almost(matrix, *args, **kwargs):
        return np.diag(matrix).copy(), np.array([[1 + 100 * EPS, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(np.linalg, "eigh", almost)
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(H)


def test_ground_eigenpair_screen_uses_actual_norm_not_inherited_unit_floor(monkeypatch):
    H = np.diag([0.0, 1e-150])
    monkeypatch.setattr(np.linalg, "eigh", lambda matrix: (np.array([0.0, 2e-150]), np.eye(2)))
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(H)


def fixture_hessian_expected(T, gaps, contact):
    """Exact rational arithmetic on the supplied binary input components."""
    m, p = T.shape
    F = Fraction.from_float
    S, Q = np.empty((p, p)), np.empty((p, p))
    for i in range(p):
        for j in range(p):
            real_products, absolute_products = Fraction(0), Fraction(0)
            for n in range(m):
                rr = F(float(T[n, i].real)) * F(float(T[n, j].real))
                ii = F(float(T[n, i].imag)) * F(float(T[n, j].imag))
                gap = F(float(gaps[n]))
                real_products += 2 * (rr + ii) / gap
                absolute_products += 2 * (abs(rr) + abs(ii)) / gap
            S[i, j], Q[i, j] = float(real_products), float(absolute_products)
    K = contact - S
    return S, Q, K, 64 * EPS * (2 * m + 2) * (np.abs(contact) + Q)


@pytest.mark.parametrize("T,gaps,contact", [
    (np.array([[1 + 2j, -3 + 4j], [2 - 1j, 1 + 3j]]), np.array([2.0, 2.0]), np.array([[3., 2.], [2., -1.]])),
    (np.array([[1j, 1], [1, 1j]], complex), np.array([1., 2.]), np.zeros((2, 2))),
    (np.array([[1., 2.]]), np.array([2.]), np.array([[1., 0.], [0., 0.]])),
    (np.zeros((2, 3), complex), np.array([1., 1.]), np.zeros((3, 3))),
])
def test_transition_hessian_complex_phase_all_excited_states_and_exact_proxy(T, gaps, contact):
    before = (T.copy(), gaps.copy(), contact.copy())
    result = api._hessian_from_transitions(T, gaps, contact)
    S, Q, K, proxy = fixture_hessian_expected(T, gaps, contact)
    for key, expected in (("contact", contact), ("spectral", S), ("connected", -S),
                          ("hessian", K), ("spectral_absolute_scale", Q),
                          ("contact_absolute_scale", np.abs(contact)), ("arithmetic_proxy", proxy)):
        same(result[key], expected, atol=1e-25)
    np.testing.assert_array_equal(result["arithmetic_resolved"], np.abs(K) > proxy)
    for i, j in itertools.product(range(len(contact)), repeat=2):
        scale = abs(contact[i, j]) + Q[i, j]
        ratio = result["cancellation_ratio"][i][j]
        if scale == 0:
            assert ratio is None
        else:
            same(ratio, abs(K[i, j]) / scale, atol=1e-25)
    for source, old in zip((T, gaps, contact), before):
        np.testing.assert_array_equal(source, old)
    for key in HESSIAN_ARRAYS:
        if type(result[key]) is np.ndarray:
            assert result[key].flags.owndata
            assert not any(np.shares_memory(result[key], source) for source in (T, gaps, contact))
    # S is a real Gram matrix by construction; no extra eigensolve is needed.
    for vector in (np.ones(len(contact)), np.arange(len(contact), dtype=float)):
        assert vector @ S @ vector >= -1e-12


@pytest.mark.parametrize("bad,which", [
    ([[1.0]], "transitions"), (ConversionTrap(), "transitions"),
    (np.ones((1, 1)).view(ArraySubclass), "transitions"),
    (np.ones((0, 1)), "transitions"), (np.ones((512, 1)), "transitions"),
    (np.ones((1, 16)), "transitions"), (np.ones((1, 1, 1)), "transitions"),
    (np.ones((1, 1), dtype=bool), "transitions"), (np.ones((1, 1), dtype=object), "transitions"),
    (np.ones((1, 1)), "gaps"), (np.ones(2), "gaps"), (np.ones(1, complex), "gaps"),
    (np.array([0.]), "gaps"), (np.array([-1.]), "gaps"),
    (np.zeros(1), "contact"), (np.zeros((2, 2)), "contact"),
    (np.zeros((1, 1), complex), "contact"), (np.zeros((1, 1)).view(ArraySubclass), "contact"),
])
def test_transition_fixture_strict_shapes_types_and_domains(bad, which):
    arguments = dict(transitions=np.ones((1, 1)), gaps=np.ones(1), contact=np.zeros((1, 1)))
    arguments[which] = bad
    with pytest.raises(ValueError):
        api._hessian_from_transitions(**arguments)


def test_fixture_rejects_raw_nonsymmetric_contact_instead_of_repair():
    with pytest.raises(ValueError):
        api._hessian_from_transitions(np.ones((1, 2)), np.ones(1), np.array([[0., 1e-30], [0., 0.]]))


@pytest.mark.parametrize("T,gaps,contact", [
    (np.array([[np.nan]]), np.ones(1), np.zeros((1, 1))),
    (np.ones((1, 1)), np.array([np.inf]), np.zeros((1, 1))),
    (np.ones((1, 1)), np.ones(1), np.array([[np.inf]])),
    (np.array([[1e-310]]), np.ones(1), np.zeros((1, 1))),
    (np.array([[1e-200]]), np.ones(1), np.zeros((1, 1))),
    (np.array([[1e200]]), np.ones(1), np.zeros((1, 1))),
    (np.ones((1, 1)), np.array([1e-308]), np.zeros((1, 1))),
    (np.array([[10.]]), np.array([1e-307]), np.zeros((1, 1))),
    (np.array([[2**53 + 1]], np.int64), np.ones(1), np.zeros((1, 1))),
])
def test_hessian_nonfinite_conversion_loss_and_erased_products_unavailable(T, gaps, contact):
    with pytest.raises(api.NumericalUnavailable):
        api._hessian_from_transitions(T, gaps, contact)


def test_tiny_normal_hessian_arithmetic_is_not_zeroed_by_unit_floor():
    result = api._hessian_from_transitions(np.array([[1e-100]]), np.ones(1), np.zeros((1, 1)))
    expected = 2e-200
    same(result["spectral"][0, 0], expected, atol=0, rtol=1e-14)
    assert result["hessian"][0, 0] < 0
    assert result["arithmetic_resolved"][0, 0]
    assert result["arithmetic_proxy"][0, 0] < expected * 1e-10


def test_exact_contact_spectral_cancellation_is_valid_but_not_arithmetic_resolved():
    result = api._hessian_from_transitions(np.array([[1.]]), np.array([2.]), np.array([[1.]]))
    assert result["hessian"][0, 0] == 0
    assert result["arithmetic_proxy"][0, 0] > 0
    assert not result["arithmetic_resolved"][0, 0]
    assert result["cancellation_ratio"][0][0] == 0


def test_transition_fixture_cross_call_ownership_including_zero_contacts():
    T = np.array([[1 + 2j, 3 - 1j], [2j, 1.]])
    gaps, contact = np.array([1., 2.]), np.zeros((2, 2))
    first = api._hessian_from_transitions(T, gaps, contact)
    second = api._hessian_from_transitions(T, gaps, contact)
    snapshot = copy.deepcopy(second)
    for key in HESSIAN_ARRAYS:
        if type(first[key]) is np.ndarray:
            assert not np.shares_memory(first[key], second[key])
            if first[key].dtype.kind != "O":
                first[key].flat[0] = 0
        else:
            first[key][0][0] = None
    T[:] = 0
    gaps[:] = 99
    contact[:] = 99
    for key in HESSIAN_ARRAYS:
        np.testing.assert_array_equal(second[key], snapshot[key])


def test_hessian_extended_component_loss_is_unavailable_before_contraction():
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble has no extra significand precision")
    T = np.array([[np.nextafter(np.longdouble(1), np.longdouble(2))]])
    assert T[0, 0] != np.longdouble(float(T[0, 0]))
    with pytest.raises(api.NumericalUnavailable):
        api._hessian_from_transitions(T, np.ones(1), np.zeros((1, 1)))


def fd_arguments(step=STEPS[1], target=2.0, curvature=2.0, mixed=False):
    denominator = (4 if mixed else 1) * step**2
    weights = np.array([1., -1., -1., 1.]) if mixed else np.array([1., -2., 1.])
    energies = (np.array([curvature * step**2, -curvature * step**2,
                          -curvature * step**2, curvature * step**2]) if mixed else
                np.array([curvature * step**2 / 2, 0., curvature * step**2 / 2]))
    return dict(energies=energies, resolutions=np.zeros(len(weights)), weights=weights,
                denominator=denominator, target=target, target_proxy=0., step=step,
                diagnostic_only=step == STEPS[0])


def fd_expected(arguments):
    energies, weights = arguments["energies"], arguments["weights"]
    numerator = math.fsum(float(w * e) for w, e in zip(weights, energies))
    energy_proxy = math.fsum(float(abs(w) * r) for w, r in zip(weights, arguments["resolutions"]))
    subtraction_proxy = 16 * EPS * math.fsum(float(abs(w * e)) for w, e in zip(weights, energies))
    numerator_proxy = energy_proxy + subtraction_proxy
    value = numerator / arguments["denominator"]
    derivative_proxy = numerator_proxy / arguments["denominator"]
    combined_proxy = derivative_proxy + arguments["target_proxy"]
    error = abs(value - arguments["target"])
    tolerance = 1e-4 * max(1., abs(arguments["target"]))
    if arguments["diagnostic_only"]:
        status, agreement = DIAGNOSTIC, None
    elif abs(numerator) <= numerator_proxy or combined_proxy > tolerance / 10:
        status, agreement = UNRESOLVED, None
    elif error <= tolerance:
        status, agreement = AVAILABLE, True
    else:
        status, agreement = MISMATCH, False
    return {"numerator": numerator, "value": value, "energy_proxy": energy_proxy,
            "subtraction_proxy": subtraction_proxy, "numerator_proxy": numerator_proxy,
            "derivative_proxy": derivative_proxy, "combined_proxy": combined_proxy,
            "error": error, "tolerance": tolerance, "status": status, "agreement": agreement}


def assert_fd(record, arguments):
    assert FD_FIELDS <= set(record)
    for key in ("energies", "resolutions", "weights", "denominator", "target", "target_proxy", "step"):
        np.testing.assert_array_equal(record[key], arguments[key])
    expected = fd_expected(arguments)
    for key, value in expected.items():
        if key in ("status", "agreement"):
            assert record[key] == value
        else:
            same(record[key], value, atol=1e-25, rtol=5e-14)
    statused(record)
    strict_json(record)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("mixed", (False, True))
@pytest.mark.parametrize("target,curvature", [(2., 2.), (0., 0.), (0., 2e-4), (0., 1e-9), (2., -2.)])
def test_fixed_fd_coefficients_raw_energy_proxies_and_status_gates(step, mixed, target, curvature):
    arguments = fd_arguments(step, target, curvature, mixed)
    record = api._finite_difference_record(**arguments)
    assert_fd(record, arguments)
    if target == curvature == 0 and step != STEPS[0]:
        assert record["status"] == UNRESOLVED and record["agreement"] is None


@pytest.mark.parametrize("resolution,target_proxy", [(1e-4, 0.), (0., 1e-3), (1e-16, 0.)])
def test_fd_endpoint_and_target_proxy_gates_are_separate(resolution, target_proxy):
    arguments = fd_arguments()
    arguments["resolutions"][:] = resolution
    arguments["target_proxy"] = target_proxy
    assert_fd(api._finite_difference_record(**arguments), arguments)


def test_fd_uses_math_fsum_without_rounding_or_high_precision_recovery():
    arguments = fd_arguments(mixed=True, target=0.)
    arguments["energies"] = np.array([1e16, 1., 1e16, 2.])
    record = api._finite_difference_record(**arguments)
    assert record["numerator"] == 1.0
    assert record["status"] == UNRESOLVED
    assert_fd(record, arguments)


def test_fd_tolerance_never_inflated_by_refinement_drift_and_tiny_sign_not_certified():
    records = []
    for step, curvature in zip(STEPS, (0.004, 0.0004, 0.0002)):
        arguments = fd_arguments(step, target=0., curvature=curvature)
        records.append(api._finite_difference_record(**arguments))
    assert records[1]["status"] == records[2]["status"] == MISMATCH
    assert all(r["tolerance"] == 1e-4 for r in records)
    for curvature in (1e-12, -1e-12):
        record = api._finite_difference_record(**fd_arguments(target=0., curvature=curvature))
        assert record["status"] == AVAILABLE and record["agreement"] is True
        assert record["tolerance"] > abs(record["value"])


@pytest.mark.parametrize("key,bad", [
    ("energies", [0., 0., 0.]), ("energies", np.zeros((3, 1))),
    ("energies", np.zeros(3, complex)), ("energies", np.zeros(3, dtype=bool)),
    ("energies", np.zeros(3).view(ArraySubclass)), ("energies", ConversionTrap()),
    ("energies", np.zeros(2)), ("energies", np.zeros(5)),
    ("resolutions", np.zeros(4)), ("resolutions", np.array([0., -1., 0.])),
    ("weights", np.array([-1., 2., -1.])), ("weights", np.array([1., -2., 1. + 2**-51])),
    ("denominator", 0.), ("denominator", -1.), ("denominator", STEPS[1]**2 * 4),
    ("denominator", np.nextafter(STEPS[1]**2, 1.)),
    ("target", True), ("target", 1j), ("target", "2"), ("target", ConversionTrap()),
    ("target", FloatSubclass(2)), ("target_proxy", -1.),
    ("step", 2**-9), ("step", True), ("diagnostic_only", np.bool_(False)),
    ("diagnostic_only", 0), ("diagnostic_only", True),
])
def test_fd_strict_shapes_original_values_and_coefficient_denominator_consistency(key, bad):
    arguments = fd_arguments()
    arguments[key] = bad
    with pytest.raises(ValueError):
        api._finite_difference_record(**arguments)


@pytest.mark.parametrize("key,bad", [
    ("energies", np.array([np.inf, 0., 0.])),
    ("energies", np.array([1e-310, 0., 0.])),
    ("energies", np.array([2**53 + 1, 0, 0], dtype=np.int64)),
    ("resolutions", np.array([0., np.nan, 0.])), ("target", np.inf),
    ("target_proxy", np.nan), ("target", 1e-310),
])
def test_fd_input_numerical_validation_is_unavailable_exception(key, bad):
    arguments = fd_arguments()
    arguments[key] = bad
    with pytest.raises(api.NumericalUnavailable):
        api._finite_difference_record(**arguments)


def test_fd_valid_input_arithmetic_failure_returns_null_record_not_fake_zero():
    arguments = fd_arguments()
    arguments["energies"] = np.array([1e308, -1e308, 1e308])
    record = api._finite_difference_record(**arguments)
    assert FD_FIELDS <= set(record)
    assert record["status"] == UNAVAILABLE and record["reason"]
    assert record["value"] is None and record["agreement"] is None
    assert record["step"] == arguments["step"]
    assert record["denominator"] == arguments["denominator"]
    assert record["weights"] == arguments["weights"].tolist()
    strict_json(record)


def expected_offsets(L, stencil_index, step):
    p, r, d = directions(L)
    if stencil_index == 1:
        return [step * (p + r), step * (p - r), step * (-p + r), step * (-p - r)]
    vector = r if stencil_index == 0 else d
    return [-step * vector, np.zeros(3 * L), step * vector]


def test_all_fd_anchor_directions_endpoint_order_targets_proxies_and_drift(frozen_demo):
    demo, _, _ = frozen_demo
    for case in demo["cases"]:
        finite = case["finite_differences"]
        assert {"status", "reason", "stencils"} <= set(finite)
        if case["L"] != 3 or case["g"] != 0.7:
            assert finite["status"] == NA and finite["stencils"] == []
            continue
        assert len(finite["stencils"]) == 3
        p, r, d = directions(3)
        for index, (stencil, vectors) in enumerate(zip(finite["stencils"], ([r], [p, r], [d]))):
            assert {"directions", "target", "target_proxy", "status", "reason", "steps"} <= set(stencil)
            same(stencil["directions"], vectors)
            if case["hessian"]["hessian"] is not None:
                K = np.asarray(case["hessian"]["hessian"])
                P = np.asarray(case["hessian"]["arithmetic_proxy"])
                left, right = vectors[0], vectors[-1]
                same(stencil["target"], left @ K @ right)
                same(stencil["target_proxy"], np.abs(left) @ P @ np.abs(right), atol=1e-25)
            assert [s["step"] for s in stencil["steps"]] == list(STEPS)
            previous = None
            for step, h in zip(stencil["steps"], STEPS):
                assert FD_FIELDS | {"endpoints", "drift"} <= set(step)
                endpoints = step["endpoints"]
                offsets = expected_offsets(3, index, h)
                assert len(endpoints) == len(offsets)
                for endpoint, offset in zip(endpoints, offsets):
                    assert {"offsets", "energy", "resolution", "status", "reason"} <= set(endpoint)
                    same(endpoint["offsets"], offset, atol=0, rtol=0)
                    statused(endpoint)
                    if not np.any(offset) and case["ground"]["energy"] is not None:
                        assert endpoint["energy"] == case["ground"]["energy"]
                        assert endpoint["resolution"] == case["ground"]["resolution"]
                if step["value"] is not None and step["status"] != UNAVAILABLE:
                    arguments = {name: step[name] for name in
                                 ("energies", "resolutions", "weights", "denominator", "target", "target_proxy", "step")}
                    for name in ("energies", "resolutions", "weights"):
                        arguments[name] = np.array(arguments[name])
                    arguments["diagnostic_only"] = h == STEPS[0]
                    assert_fd({key: step[key] for key in FD_FIELDS}, arguments)
                    assert step["energies"] == [e["energy"] for e in endpoints]
                    assert step["resolutions"] == [e["resolution"] for e in endpoints]
                    if previous is None:
                        assert step["drift"] is None
                    else:
                        # The contract says diagnostic drift, not signed versus
                        # absolute serialization; magnitude is unambiguous.
                        same(abs(step["drift"]), abs(step["value"] - previous))
                    previous = step["value"]
                else:
                    assert step["status"] == UNAVAILABLE
                    assert step["agreement"] is None
            assert stencil["status"] == aggregate(stencil["steps"])
        assert finite["status"] == aggregate(finite["stencils"])


def synthetic_ground(H, raw=False):
    """Small instrumentation substitute, not an additional physical control.

    Used ONLY after the documented ground-screen seam. Raw mode intentionally
    exposes a non-unit ground norm and an excited ground-component residue so
    hidden centering/renormalization changes observable contractions.
    """
    d = len(H)
    vectors = np.eye(d, dtype=complex)
    if raw:
        vectors[0, 0] = 1.125
        vectors[1, 0] = 0.25j
        vectors[0, 1] = 0.125j
    values = np.arange(d, dtype=float)
    return {"values": values, "vectors": vectors, "ground": vectors[:, 0].copy(),
            "gaps": values.copy(), "ground_gap": 1., "resolution": 1e-12,
            "orthogonality_residual": 0., "eigenpair_residual": 0., "scale": fro(H)}


def test_owning_hessian_keeps_raw_ground_columns_no_centering_and_coordinate_contact_proxy(monkeypatch):
    captured = []

    def screen(H):
        result = synthetic_ground(H, raw=True)
        captured.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(api, "_screened_ground", screen)
    case = api.case_report(3, 3, 0., partition="improved", background="modulated")
    f, A, u = sources(3)
    oracle = occupation_oracle(3, 3, 0., "improved", f, A, u)
    expected = hessian_oracle(oracle, captured[0])
    check_hessian(case["hessian"], expected)
    # Uniform raw density transition is deliberately nonzero at this seam.
    T = expected["transitions"]
    assert fro(T[:, 6:] @ np.ones(3)) > 0.1
    assert fro(np.array(case["hessian"]["spectral"])[6:, 6:]) > 0.01
    Qc = expected["contact_absolute_scale"]
    assert np.any(Qc > np.abs(expected["contact"]) + 1e-6)


@pytest.mark.parametrize("kind", ("ordered", "imaginary"))
def test_raw_contact_orders_and_imaginary_expectations_retained_with_mismatch(monkeypatch, kind):
    original = api._second_derivative
    counts = Counter()

    def second(system, i, j):
        counts[i, j] += 1
        result = original(system, i, j)
        if (i, j) == (0, 3):
            result = result + (0.125 if kind == "ordered" else 0.125j) * np.eye(len(result))
        return result

    monkeypatch.setattr(api, "_second_derivative", second)
    monkeypatch.setattr(api, "_screened_ground", synthetic_ground)
    case = api.case_report(3, 3, 0., background="modulated")
    assert counts[0, 3] and counts[3, 0]
    hessian = case["hessian"]
    assert hessian["hessian"] is not None
    assert case["status"] == MISMATCH
    if kind == "ordered":
        C = np.array(hessian["contact"])
        assert abs(C[0, 3] - C[3, 0]) > 0.1
        assert hessian["symmetry"]["status"] == MISMATCH
    else:
        raw = np.array(hessian["contact_realness"]["imaginary"])
        assert raw[0, 3] == pytest.approx(0.125)
        assert hessian["contact_realness"]["status"] == MISMATCH
    assert case["ward_checks"]["AA"]["status"] != UNAVAILABLE


def test_full_matrix_work_mismatch_cannot_hide_in_zero_expectation_or_skip_fd(monkeypatch):
    original = api._first_derivative
    screens = []

    def first(system, index):
        matrix = original(system, index)
        if index == 0:
            matrix = matrix.copy()
            matrix[1, 1] += 0.125
            matrix[4, 4] -= 0.125
        return matrix

    def screen(H):
        screens.append(H.shape)
        return synthetic_ground(H)

    monkeypatch.setattr(api, "_first_derivative", first)
    monkeypatch.setattr(api, "_screened_ground", screen)
    case = api.case_report(3, 3, 0.7)
    work = case["operator_checks"]["work_identity"]
    assert work["error"] <= work["tolerance"]
    assert work["matrix_error"] > work["matrix_tolerance"]
    assert work["matrix_consistent"] is False and work["consistent"] is False
    assert work["status"] == case["status"] == MISMATCH
    assert case["hessian"]["hessian"] is not None
    assert len(screens) == 25
    assert len(case["finite_differences"]["stencils"]) == 3


def assert_failure_placeholders(case):
    strict_json(case)
    assert CASE_KEYS <= set(case)
    hessian = case["hessian"]
    assert set(HESSIAN_ARRAYS) <= set(hessian)
    assert all(hessian[name] is None for name in HESSIAN_ARRAYS)
    for ward in case["ward_checks"].values():
        assert ward["status"] == UNAVAILABLE and ward["reason"]
        for name in ("contact_contraction", "spectral_contraction", "completed_contraction"):
            assert ward[name] is None
    stencils = case["finite_differences"]["stencils"]
    assert len(stencils) == 3
    for index, stencil in enumerate(stencils):
        assert len(stencil["steps"]) == 3
        for step, h in zip(stencil["steps"], STEPS):
            assert FD_FIELDS | {"endpoints", "drift"} <= set(step)
            assert step["status"] == UNAVAILABLE and step["reason"]
            assert step["value"] is None and step["agreement"] is None
            same([e["offsets"] for e in step["endpoints"]], expected_offsets(3, index, h), atol=0, rtol=0)
            assert all(e["energy"] is None for e in step["endpoints"] if any(e["offsets"]))


@pytest.mark.parametrize("error_type", (inherited.NumericalUnavailable, FloatingPointError,
                                        OverflowError, np.linalg.LinAlgError))
def test_ground_failure_keeps_operator_checks_and_all_fd_placeholders(error_type, monkeypatch):
    calls = []

    def failure(H):
        calls.append(H.shape)
        raise error_type("injected ground numerical failure")

    monkeypatch.setattr(api, "_screened_ground", failure)
    case = api.case_report(3, 3, 0.7)
    assert calls == [(10, 10)]
    assert case["ground"]["status"] == UNAVAILABLE
    assert case["status"] == UNAVAILABLE
    for name in ("gauge_covariance", "continuity", "density_sum", "work_identity"):
        assert case["operator_checks"][name]["status"] == AVAILABLE
    assert_failure_placeholders(case)


def test_unavailable_hessian_schema_retains_completed_contact_evidence():
    # Schema-helper regression only: the fixture helper is not assumed to be
    # the owning implementation's contraction path or an injection seam.
    f, A, u = sources(3)
    oracle = occupation_oracle(3, 3, 0.7, "improved", f, A, u)
    expected = contact_oracle(oracle, synthetic_ground(oracle["H"], raw=True))
    scale = fro(expected["contact_absolute_scale"])
    error = fro(expected["imaginary"])
    record = {key: None for key in HESSIAN_ARRAYS}
    record.update(status=UNAVAILABLE, reason="later spectral contraction unavailable",
                  contact=expected["contact"].tolist(),
                  contact_absolute_scale=expected["contact_absolute_scale"].tolist(),
                  symmetry={"status": UNAVAILABLE, "reason": "completed Hessian unavailable"},
                  arithmetic_scope="source eigensystem action projection errors excluded",
                  contact_realness={"imaginary": expected["imaginary"].tolist(),
                                    "error": error, "scale": scale,
                                    "tolerance": TOL + TOL * scale, "consistent": True,
                                    "status": AVAILABLE, "reason": None})
    check_hessian(record, expected)
    strict_json(record)
    wrong_contact = copy.deepcopy(record)
    wrong_contact["contact"][0][0] += 1.
    with pytest.raises(AssertionError):
        check_hessian(wrong_contact, expected)
    wrong_scale = copy.deepcopy(record)
    wrong_scale["contact_absolute_scale"][0][0] += 1.
    with pytest.raises(AssertionError):
        check_hessian(wrong_scale, expected)
    wrong_spectral = copy.deepcopy(record)
    wrong_spectral["spectral"] = np.zeros((9, 9)).tolist()
    with pytest.raises(AssertionError):
        check_hessian(wrong_spectral, expected)
    early = copy.deepcopy(record)
    for key in HESSIAN_ARRAYS:
        early[key] = None
    check_hessian(early, None)
    early["contact_absolute_scale"] = expected["contact_absolute_scale"].tolist()
    with pytest.raises(AssertionError):
        check_hessian(early, None)


def test_hessian_failure_keeps_ground_and_free_energy_oracle(monkeypatch):
    monkeypatch.setattr(api, "_screened_ground", synthetic_ground)

    def failure(*args, **kwargs):
        raise api.NumericalUnavailable("injected contact arithmetic failure")

    monkeypatch.setattr(api, "_second_derivative", failure)
    case = api.case_report(3, 3, 0.)
    assert case["ground"]["status"] == AVAILABLE and case["ground"]["energy"] == 0
    assert case["hessian"]["status"] == UNAVAILABLE
    free = case["free_oracle"]
    assert free["energy_comparison"]["error"] is not None
    assert free["energy_comparison"]["status"] == MISMATCH
    assert free["AA_comparison"]["status"] == UNAVAILABLE
    assert case["status"] == MISMATCH


def test_failed_first_fd_endpoint_preserves_full_fixed_grid_and_is_not_ignored(monkeypatch):
    calls = []

    def screen(H):
        calls.append(H.copy())
        if len(calls) == 2:
            raise api.NumericalUnavailable("first displaced endpoint unavailable")
        return synthetic_ground(H)

    monkeypatch.setattr(api, "_screened_ground", screen)
    case = api.case_report(3, 3, 0.7)
    assert len(calls) == 25
    finite = case["finite_differences"]
    first = finite["stencils"][0]["steps"][0]
    assert first["status"] == UNAVAILABLE and first["agreement"] is None
    assert first["endpoints"][0]["status"] == UNAVAILABLE
    assert first["endpoints"][0]["energy"] is None
    assert first["endpoints"][2]["status"] == AVAILABLE
    assert len([e for t in finite["stencils"] for s in t["steps"] for e in s["endpoints"]]) == 30
    for t in finite["stencils"]:
        assert t["status"] == aggregate(t["steps"])
    assert finite["status"] == aggregate(finite["stencils"])
    # Mismatch, if generated by the synthetic data, outranks unavailable.
    assert finite["status"] in (UNAVAILABLE, MISMATCH)
    strict_json(case)


@pytest.mark.parametrize("boundary", ("_source_system", "_screened_ground", "_first_derivative",
                                      "_second_derivative", "_finite_difference_record"))
@pytest.mark.parametrize("error_type", (TypeError, ValueError, AssertionError, RuntimeError, ZeroDivisionError))
def test_programming_errors_propagate_at_every_documented_numerical_boundary(boundary, error_type, monkeypatch):
    def broken(*args, **kwargs):
        raise error_type("unexpected programming error")

    monkeypatch.setattr(api, "_screened_ground", synthetic_ground)
    monkeypatch.setattr(api, boundary, broken)
    with pytest.raises(error_type, match="unexpected programming error"):
        api.case_report(3, 3, 0.7)


@pytest.mark.parametrize("error_type", (TypeError, ValueError, AssertionError, RuntimeError, ZeroDivisionError))
def test_unexpected_inherited_eigensystem_exception_not_laundered(error_type, monkeypatch):
    def broken(H):
        raise error_type("unrelated injected helper bug")

    monkeypatch.setattr(api, "_screened_eigensystem", broken)
    with pytest.raises(error_type, match="unrelated injected helper bug"):
        api._screened_ground(np.diag([0., 1.]))


def test_case_reports_are_detached_and_repeated_calls_have_no_mutable_cache(monkeypatch):
    calls = []

    def screen(H):
        calls.append(1)
        return synthetic_ground(H)

    monkeypatch.setattr(api, "_screened_ground", screen)
    first = api.case_report(3, 3, 0., background="modulated")
    snapshot = copy.deepcopy(first)
    second = api.case_report(3, 3, 0., background="modulated")
    assert len(calls) == 2
    assert first == second
    first["sources"]["f"][0] = -99
    first["hessian"]["contact"][0][0] = -99
    first["operator_checks"]["gauge_covariance"]["status"] = "mutated"
    assert second == snapshot
    strict_json(second)


def test_demo_aggregation_derives_status_counts_and_detaches_injected_reports(frozen_demo, monkeypatch):
    demo, _, _ = frozen_demo
    supplied = [copy.deepcopy(c) for c in demo["cases"]]
    assigned = (UNRESOLVED, UNAVAILABLE, MISMATCH, AVAILABLE)
    for i, case in enumerate(supplied):
        case["status"] = assigned[i % 4]
        case["reason"] = None if case["status"] == AVAILABLE else "injected retained status"
    seen = []

    def case(L, N, g, partition="symmetric", background="uniform"):
        key = (L, g, partition, background)
        seen.append(key)
        return supplied[GRID.index(key)]

    monkeypatch.setattr(api, "case_report", case)
    report = api.demonstration_report()
    assert seen == list(GRID)
    assert report["status"] == MISMATCH
    counts = Counter(c["status"] for c in supplied)
    assert report["summary"]["case_status_counts"] == {s: counts[s] for s in STATUSES}
    snapshot = copy.deepcopy(supplied)
    report["cases"][0]["sources"]["f"][0] = -99
    report["cases"][0]["hessian"]["status"] = "mutated"
    assert supplied == snapshot


def test_partial_unavailable_fd_value_is_not_a_drift_history_sample(monkeypatch):
    original = api._finite_difference_record
    calls = []

    def fd(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        step = arguments["step"]
        # Deliberately preserve a partial numeric value in an unavailable
        # arithmetic record. It must not become the next drift reference.
        value = {STEPS[0]: 0.004, STEPS[1]: 100., STEPS[2]: 0.0002}[step]
        inputs = fd_arguments(step, target=0., curvature=value,
                              mixed=len(arguments["energies"]) == 4)
        result = original(**inputs)
        if step == STEPS[1]:
            result["status"] = UNAVAILABLE
            result["reason"] = "injected arithmetic failure after partial value"
            result["agreement"] = None
        calls.append(step)
        return result

    monkeypatch.setattr(api, "_screened_ground", synthetic_ground)
    monkeypatch.setattr(api, "_finite_difference_record", fd)
    case = api.case_report(3, 3, 0.7)
    assert len(calls) == 9
    for stencil in case["finite_differences"]["stencils"]:
        first, failed, last = stencil["steps"]
        assert failed["status"] == UNAVAILABLE and failed["agreement"] is None
        assert failed["value"] == 100.
        assert last["value"] == pytest.approx(0.0002)
        same(abs(last["drift"]), abs(last["value"] - first["value"]))
        assert last["tolerance"] == 1e-4 and last["status"] == MISMATCH
    strict_json(case)


def test_fd_drift_overflow_makes_step_unavailable_and_clears_agreement(monkeypatch):
    original = api._finite_difference_record

    def fd(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        result = original(**fd_arguments(arguments["step"], mixed=len(arguments["energies"]) == 4))
        result["value"] = -1e308 if arguments["step"] == STEPS[0] else 1e308
        return result

    monkeypatch.setattr(api, "_screened_ground", synthetic_ground)
    monkeypatch.setattr(api, "_finite_difference_record", fd)
    case = api.case_report(3, 3, 0.7)
    for stencil in case["finite_differences"]["stencils"]:
        for record in stencil["steps"][1:]:
            assert record["status"] == UNAVAILABLE
            assert record["agreement"] is None and record["reason"]
            assert record["drift"] is None
    strict_json(case)


def test_fd_late_arithmetic_failure_clears_agreement_on_unavailable_record():
    arguments = fd_arguments(target=-1e308, curvature=0.)
    arguments["energies"] = np.array([3e303, 0., 3e303])
    record = api._finite_difference_record(**arguments)
    # The derivative is representable, but abs(value-target) is not.
    assert record["status"] == UNAVAILABLE
    assert record["agreement"] is None and record["reason"]
    assert FD_FIELDS <= set(record)
    strict_json(record)


def test_solver_lost_nonzero_extended_output_rejected_without_blanket_subnormal_ban(monkeypatch):
    if np.finfo(np.longdouble).minexp >= np.finfo(float).minexp:
        pytest.skip("platform longdouble has no wider exponent range")
    values = np.array([np.longdouble("1e-4000"), np.longdouble(1)])
    assert values[0] != 0 and float(values[0]) == 0
    monkeypatch.setattr(np.linalg, "eigh", lambda matrix: (values.copy(), np.eye(2)))
    with pytest.raises(api.NumericalUnavailable):
        api._screened_ground(np.diag([0., 1.]))


@pytest.mark.parametrize("json_mode", (False, True), ids=("text", "strict-json"))
def test_absolute_demo_from_empty_cwd_stdout_only_no_files_or_stderr(tmp_path, json_mode):
    script = ROOT / "scripts" / "demo_substrate_source_integrability.py"
    assert script.is_file()
    assert list(tmp_path.iterdir()) == []
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONWARNINGS"] = "error"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    command = [sys.executable, "-B", str(script.resolve())]
    if json_mode:
        command.append("--json")
    completed = subprocess.run(command, cwd=str(tmp_path), env=env, text=True,
                               capture_output=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.strip()
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_constant(value):
            raise AssertionError("non-JSON numeric constant " + value)

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                assert key not in result, "duplicate JSON key"
                result[key] = value
            return result

        report = json.loads(completed.stdout, parse_constant=reject_constant,
                            object_pairs_hook=unique_object)
        strict_json(report)
        assert report["model_id"] == "substrate_source_integrability"
        assert len(report["cases"]) == 36
        assert report["summary"]["case_count"] == 36
        assert report["status"] == aggregate(report["cases"])
    else:
        assert "integrability" in completed.stdout.lower()
