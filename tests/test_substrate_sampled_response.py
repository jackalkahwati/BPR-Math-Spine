"""Independent Module6 tests, frozen before any scientific execution.

Oracle provenance: entire substrate_sampled_response_2026-09-13.md, SHA256
 ef22910d2fb42200dd213fa283d9db88dfbb472ea149fd9228ceb2a0300df108;
Clarification A appended freeze SHA256
 e6ea4a8bf3c24963c13d2c4f021fc17ec8cf5566de1d967b67dbe5a6a6d0fe10.
Original blind version read no new production/demo source; the explicitly
labeled post-authorship supplement below records its narrow authorized read.
Exact schemas follow section10/revisions
2/3 and the final numerical policy, not historical pending shorthand.

FIXTURE LAYOUT AND INVOCATION BUDGET (not a collected test-node count)
* shared_physical_demo: ONE complete demonstration, nine keyed _owned_case
  slots, at most one joint_system/eigh attempt per key, at most nine total.
  Fully available means exactly nine. Capture owned systems; no repeat solves.
  Consumers: slot/budget/schema; normalized independent matrices/metadata;
  free analytic identities; physical sparse Taylor (both partitions together);
  post-authorship paired synthetic-noise differences (existing records only).
* Taylor: ONE existing (L3,g=.7,C1) reference, two _time_trace invocations,
  times(0,2^-10,2^-8), degree32, at most256 matrix-vector products total.
* scalar_acquisitions: four named small spectral fixtures, each two schedules
  and one fixed nu=1.25, G=ones, epsilon0: eight _acquire calls, no solves.
  Shared by modal/geometric/budget/degeneracy consumers; no hidden grid.
* Other valid tiny acquisitions (one invocation per listed item, before any
  execution): Pauli signed/plain gain2; degenerate original/rotation2; isolated
  noise schedules2; tiny phase2; zero gain1; ownership2; phase/exp failure3;
  unrepresentable grid1; underflowing-Gram rational kernel1; natural proxy
  underflow1; injected trace
  failure1; injected raw-complex trace1. All have ZERO model/eigh calls.
* Direct tiny _time_trace: exact zero-column1; readonly ownership1; validation
  lists below (one call per bad-input parameter), no solves. Ratio fixtures:
  named RATIO_FIXTURES13, final conversion failures5, Pauli ratio1, signed-gain
  ratio6; validation lists below one invocation per parameter, no solves.
* Synthetic-joint ownership test: two _owned_case calls, each independent
  test-only ten-state diagonal/source arrays, at most two maximum traces per
  call, ZERO model/eigh. Tiny-g boundary2 mocked joint attempts; tiny-g3 zero
  joint attempts; failed demos2*9 mocked slot/joint attempts; standalone failed
  synthetic2 mocked reference attempts. No physical Hamiltonian is evaluated.
* Failure demos replace _owned_case, joint_system or other owned boundaries;
  no repeated physical construction. Fresh-report repeats use only independent
  synthetic arrays/mocked owned systems, never physical solves.
* POST-AUTHORSHIP supplement (review-authorized, original blind8556aebc SHA
  preserved externally): two dimension2 joint stubs, one call per gap-state;
  real _normalize_gaps; partition normalization injected unavailable; reuse
  each same owned object in _synthetic_owned; zero traces/physical solves.
  Inspected only production lines643..816 to establish these seam signatures
  and necessary wrapper fields; no analytical oracle derives from that read.
* CLI text/json are separate singleton subprocess tests with a mocked full
  report and exactly one demo entry; ZERO model/joint/eigh. The coordinator's
  later unwrapped isolated CLI gate is separate, not covered by these mocks.

All transcendental scalar references use fixed100 decimal digits. Tol(F,S)
=256*2^-52*F*S+8*F*2^-1074 is compared as exact Fraction arithmetic against
raw complex component residuals. No max(1,S), adaptive tolerance/precision,
PSD eigensolve, SVD, new physical grid, or tiny-to-zero substitution.
"""
from collections import Counter
import copy
from decimal import Decimal
from fractions import Fraction
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import mpmath as mp
import numpy as np
import pytest

from bpr import substrate_current_response as current
from bpr import substrate_energy_response as energy
from bpr import substrate_fermionization as occupation
from bpr import substrate_joint_source_kernel as joint
from bpr import substrate_prediction_contract as prediction
from bpr import substrate_sampled_response as api

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_substrate_sampled_response.py"
GRID = tuple((L, g, 1.0) for L in (3, 4, 5) for g in (0.0, 0.7, 40.0))
PARTITIONS = ("symmetric", "improved")
SCHEDULES = ((4.0, 128), (8.0, 256))
ETAS = (0.5, 1.0, 2.0)
LEVELS = (0.0, 1e-8)
GAINS = ((10.0, 14.0), (-15.0, -21.0))
U = Fraction(1, 2**52)
Q0 = Fraction(1, 2**1074)
AVAILABLE = "available_conditional"
PARTIAL = "partially_unavailable"
UNAVAILABLE = "numerical_unavailable"
SCOPE = dict(empirical_status="empirical_test_unavailable", empirical_validation=False,
             numerical_error_certified=False,
             analytic_bounds="exact-model conditional formulas; floating evaluation not enclosed",
             assembly_proxy="heuristic, excludes eigensystem certification",
             contacts="not reconstructed", clock="model C units only")
REASONS = set("base_system_unavailable tiny_g_unavailable ground_gap_unresolved "
              "normalized_data_unavailable trace_unavailable sample_assembly_unavailable "
              "reference_unavailable estimate_unavailable envelope_unavailable "
              "arithmetic_proxy_unavailable denominator_center_zero "
              "denominator_disk_contains_zero ratio_arithmetic_unavailable "
              "dependency_unavailable nonfinite_arithmetic nonzero_underflow "
              "negative_computed_bound precision_exhausted".split())
RATIO_KEYS = set("status value radius reason detail denominator_margin_status conditional "
                 "numerical_error_certified exact_denominator_gate center_evaluation "
                 "center_convention error_radius_evaluation".split())
RECORD_KEYS = set("partition schedule eta nu error_level gains status reason detail "
                  "finite_reference infinite_reference estimate errors error_status "
                  "error_reason error_detail arithmetic_proxy proxy_status proxy_reason "
                  "proxy_detail diagnostics ratio numerical_error_certified".split())
CASE_KEYS = set("L N g C m dimension status reason detail ground_gap normalized_ground_gap "
                "normalized_resolution partitions scope".split())
PARTITION_KEYS = set("partition status reason detail normalized_total_weight source_metadata records".split())
SYNTHETIC_KEYS = set("reference readout_gains source_gains error_levels perturbation_definition "
                     "status reason detail records scope".split())
DIAGNOSTIC_KEYS = set("status entry_status residual analytic_limit proxy reason detail".split())
INTERNAL_KEYS = set("times noiseless_samples samples sample_status sample_reason sample_detail record".split())
MODULES = (api, joint, prediction, current, energy, occupation, np, np.linalg, math)


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


class ArraySubclass(np.ndarray):
    pass


class ProtocolBomb:
    def __float__(self):
        raise AssertionError("unexpected float protocol")

    def __array__(self, *args, **kwargs):
        raise AssertionError("unexpected array protocol")

    def __index__(self):
        raise AssertionError("unexpected index protocol")


def deny(*args, **kwargs):
    raise AssertionError("forbidden extra physical construction or solver")


def patch_aliases(patch, original, replacement):
    found = []
    for module in MODULES:
        for name, value in tuple(vars(module).items()):
            if value is original:
                patch.setattr(module, name, replacement)
                found.append((module.__name__, name))
    assert found
    return found


def block_physics(patch):
    for target in (joint.joint_system, joint.joint_report, prediction.case_report,
                   occupation.fixed_number_model, occupation._occupations,
                   np.linalg.eigh, np.linalg.eigvalsh, np.linalg.eig,
                   np.linalg.eigvals, np.linalg.svd):
        patch_aliases(patch, target, deny)


def native(tree):
    if type(tree) is dict:
        assert all(type(key) is str for key in tree)
        for value in tree.values():
            native(value)
    elif type(tree) is list:
        for value in tree:
            native(value)
    else:
        assert tree is None or type(tree) in (str, float, int, bool)
        if type(tree) is float:
            assert math.isfinite(tree)


def reason_pair(record, available, reason="reason", detail="detail"):
    if available:
        assert record[reason] is None and record[detail] is None
    else:
        assert record[reason] in REASONS
        assert type(record[detail]) is str and record[detail].strip()


def real_matrix(value):
    assert type(value) is list and len(value) == 2
    assert all(type(row) is list and len(row) == 2 for row in value)
    assert all(type(x) is float and math.isfinite(x) for row in value for x in row)
    return np.array(value, dtype=np.float64)


def complex_matrix(value):
    assert set(value) == {"shape", "real", "imag"} and value["shape"] == [2, 2]
    return real_matrix(value["real"]) + 1j * real_matrix(value["imag"])


def complex_scalar(value):
    assert set(value) == {"real", "imag"}
    assert all(type(x) is float and math.isfinite(x) for x in value.values())
    return complex(value["real"], value["imag"])


def ratio_schema(row):
    assert set(row) == RATIO_KEYS
    assert row["conditional"] is True and row["numerical_error_certified"] is False
    assert row["center_evaluation"] == "rounded_non_enclosed"
    assert row["center_convention"] == "exact_quotient_radius_rounded_display"
    assert row["error_radius_evaluation"] == "non_enclosed"
    assert type(row["exact_denominator_gate"]) is bool
    statuses = {"conditional_available": "excludes_zero", "zero_denominator": "center_zero",
                "unresolved_denominator": "contains_zero", "dependency_unavailable": "not_evaluated"}
    if row["status"] in statuses:
        assert row["denominator_margin_status"] == statuses[row["status"]]
    else:
        assert row["status"] == UNAVAILABLE
        assert row["denominator_margin_status"] in ("excludes_zero", "not_evaluated")
    reason_pair(row, row["status"] == "conditional_available")
    if row["status"] in ("zero_denominator", "unresolved_denominator", "dependency_unavailable"):
        assert row["value"] is row["radius"] is None
    if row["value"] is not None:
        complex_scalar(row["value"])
    if row["radius"] is not None:
        assert type(row["radius"]) is float and math.isfinite(row["radius"]) and row["radius"] >= 0
    if row["status"] == "conditional_available":
        assert row["value"] is not None and row["radius"] is not None
    if row["denominator_margin_status"] != "not_evaluated":
        assert row["exact_denominator_gate"] is True
    else:
        assert row["exact_denominator_gate"] is False


def acquisition_schema(row, partition=None, eta=None, theta=None, intervals=None, level=None, gains=None):
    assert set(row) == RECORD_KEYS
    assert row["partition"] == partition and row["eta"] == eta
    schedule = row["schedule"]
    assert set(schedule) == {"theta", "intervals", "sample_count", "step"}
    if theta is not None:
        exact_step = f(theta) / intervals
        displayed_step = float(exact_step)
        expected_step = displayed_step if displayed_step != 0 else None
        assert schedule == dict(theta=theta, intervals=intervals, sample_count=intervals + 1, step=expected_step)
    assert type(schedule["intervals"]) is int and 1 <= schedule["intervals"] <= 256
    assert schedule["sample_count"] == schedule["intervals"] + 1
    if schedule["step"] is None:
        assert f(schedule["theta"]) / schedule["intervals"] > 0
        assert float(f(schedule["theta"]) / schedule["intervals"]) == 0
    else:
        assert schedule["step"] == float(f(schedule["theta"]) / schedule["intervals"])
    if level is not None:
        assert row["error_level"] == level
    actual_gains = real_matrix(row["gains"])
    if gains is not None:
        np.testing.assert_array_equal(actual_gains, gains)
    assert row["status"] in (AVAILABLE, PARTIAL, UNAVAILABLE)
    reason_pair(row, row["status"] == AVAILABLE)
    assert row["numerical_error_certified"] is False
    for name in ("finite_reference", "infinite_reference", "estimate"):
        if row[name] is not None:
            complex_matrix(row[name])
    assert set(row["errors"]) == {"tail", "quadrature", "observation", "total"}
    complete = []
    for name, value in row["errors"].items():
        if value is not None:
            assert np.all(real_matrix(value) >= 0)
            complete.append(name)
    expected = AVAILABLE if len(complete) == 4 else PARTIAL if complete else UNAVAILABLE
    assert row["error_status"] == expected
    reason_pair(row, expected == AVAILABLE, "error_reason", "error_detail")
    if row["errors"]["total"] is not None:
        assert all(row["errors"][key] is not None for key in ("tail", "quadrature", "observation"))
    if row["arithmetic_proxy"] is not None:
        assert np.all(real_matrix(row["arithmetic_proxy"]) >= 0)
        assert row["proxy_status"] == "available_heuristic"
    else:
        assert row["proxy_status"] in (UNAVAILABLE, "dependency_unavailable")
    reason_pair(row, row["proxy_status"] == "available_heuristic", "proxy_reason", "proxy_detail")
    assert set(row["diagnostics"]) == {"window", "quadrature", "total"}
    for diagnostic in row["diagnostics"].values():
        assert set(diagnostic) == DIAGNOSTIC_KEYS
        statuses = ("within_diagnostic_envelope", "outside_diagnostic_envelope", "inconclusive")
        assert diagnostic["status"] in statuses
        entries = diagnostic["entry_status"]
        assert len(entries) == 2 and all(len(r) == 2 for r in entries)
        assert all(x in statuses for r in entries for x in r)
        for name in ("residual", "analytic_limit", "proxy"):
            if diagnostic[name] is not None:
                assert np.all(real_matrix(diagnostic[name]) >= 0)
        if any(diagnostic[name] is None for name in ("residual", "analytic_limit", "proxy")):
            assert entries == [["inconclusive"] * 2] * 2 and diagnostic["status"] == "inconclusive"
            assert diagnostic["reason"] == "dependency_unavailable"
        else:
            residual, limit, proxy = (real_matrix(diagnostic[name]) for name in ("residual", "analytic_limit", "proxy"))
            expected_entries = [["within_diagnostic_envelope" if f(residual[a, b]) <= f(limit[a, b]) + f(proxy[a, b])
                                 else "outside_diagnostic_envelope" for b in range(2)] for a in range(2)]
            assert entries == expected_entries
            aggregate = "outside_diagnostic_envelope" if any("outside_diagnostic_envelope" in r for r in entries) else "within_diagnostic_envelope"
            assert diagnostic["status"] == aggregate
            assert diagnostic["reason"] is diagnostic["detail"] is None
    ratio_schema(row["ratio"])
    if row["status"] == AVAILABLE:
        assert all(row[name] is not None for name in ("finite_reference", "infinite_reference", "estimate"))
        assert row["error_status"] == AVAILABLE and row["arithmetic_proxy"] is not None
        assert row["ratio"]["status"] in ("conditional_available", "zero_denominator", "unresolved_denominator")
    if row["status"] == UNAVAILABLE:
        assert all(row[name] is None for name in ("finite_reference", "infinite_reference", "estimate"))
    native(row)


def case_schema(case, key):
    assert set(case) == CASE_KEYS
    L, g, C = key
    assert (case["L"], case["N"], case["g"], case["C"], case["m"]) == (L, L, g, C, 1)
    assert case["scope"] == SCOPE
    assert case["status"] in (AVAILABLE, PARTIAL, UNAVAILABLE)
    reason_pair(case, case["status"] == AVAILABLE)
    assert case["dimension"] in (None, math.comb(2 * L - 1, L))
    assert [p["partition"] for p in case["partitions"]] == list(PARTITIONS)
    for partition in case["partitions"]:
        assert set(partition) == PARTITION_KEYS
        reason_pair(partition, partition["status"] == AVAILABLE)
        assert set(partition["source_metadata"]) == {"rho", "h"}
        for data in partition["source_metadata"].values():
            assert set(data) == {"normalized_operator_frobenius_norm", "normalized_transition_norm"}
            for value in data.values():
                assert value is None or (type(value) is float and math.isfinite(value) and value >= 0)
        if partition["normalized_total_weight"] is not None:
            complex_matrix(partition["normalized_total_weight"])
        assert len(partition["records"]) == 6
        for row, (theta, n, eta) in zip(partition["records"],
                                       ((theta, n, eta) for theta, n in SCHEDULES for eta in ETAS)):
            acquisition_schema(row, partition["partition"], eta, theta, n, 0.0, [[1.0] * 2] * 2)
            if case["normalized_ground_gap"] is not None and row["nu"] is not None:
                assert row["nu"] == eta * case["normalized_ground_gap"]
    native(case)


def synthetic_schema(report):
    assert set(report) == SYNTHETIC_KEYS
    assert report["reference"] == dict(L=3, N=3, C=1.0, g=0.7, m=1, partition="symmetric")
    assert report["readout_gains"] == [2, -3] and report["source_gains"] == [5, 7]
    assert report["error_levels"] == list(LEVELS)
    assert report["perturbation_definition"] == "ideal postgain delta[j,a,b]=epsilon*(-1)^(j+a+b); common j across nested schedules"
    assert report["scope"] == SCOPE
    assert report["status"] in (AVAILABLE, PARTIAL, UNAVAILABLE)
    reason_pair(report, report["status"] == AVAILABLE)
    assert len(report["records"]) == 12
    for row, (level, theta, n, eta) in zip(report["records"],
         ((level, theta, n, eta) for level in LEVELS for theta, n in SCHEDULES for eta in ETAS)):
        acquisition_schema(row, "symmetric", eta, theta, n, level, GAINS)
    native(report)


def demo_schema(report):
    assert set(report) == {"module", "physical_cases", "synthetic", "counts", "limitations", "scope"}
    assert report["module"] == "substrate_sampled_response"
    assert report["counts"] == dict(hamiltonian_slots=9, partition_slots=18, physical_records=108,
                                    synthetic_records=12, max_samples=257)
    assert report["scope"] == SCOPE
    assert type(report["limitations"]) is list and report["limitations"]
    assert all(type(item) is str and item.strip() for item in report["limitations"])
    assert len(report["physical_cases"]) == 9
    for case, key in zip(report["physical_cases"], GRID):
        case_schema(case, key)
    synthetic_schema(report["synthetic"])
    native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def f(value):
    if isinstance(value, Fraction):
        return value
    return Fraction.from_float(float(value))


def pair(value):
    z = complex(value)
    return f(z.real), f(z.imag)


def cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cdiv(a, b):
    denominator = b[0]**2 + b[1]**2
    return ((a[0] * b[0] + a[1] * b[1]) / denominator,
            (a[1] * b[0] - a[0] * b[1]) / denominator)


def conjugate(z):
    return z[0], -z[1]


def exact_grams(transitions):
    return [[[cmul(conjugate(pair(row[a])), pair(row[b])) for b in range(2)] for a in range(2)]
            for row in transitions]


def exact_kernel(gaps, transitions, nu, gains):
    grams = exact_grams(transitions)
    result = []
    for a in range(2):
        row = []
        for b in range(2):
            value = (Fraction(0), Fraction(0))
            for delta, gram in zip(gaps, grams):
                positive = cdiv(gram[a][b], (-f(delta), f(nu)))
                negative = cdiv(gram[b][a], (f(delta), f(nu)))
                value = cadd(value, (positive[0] - negative[0], positive[1] - negative[1]))
            row.append((value[0] * f(gains[a][b]), value[1] * f(gains[a][b])))
        result.append(row)
    return result


def mp_real(value):
    value = f(value)
    return mp.mpf(value.numerator) / value.denominator


def mp_complex(z):
    real, imag = pair(z)
    return mp.mpc(mp_real(real), mp_real(imag))


def allowance(F, S):
    return 256 * U * f(F) * f(S) + 8 * f(F) * Q0


def compare_component(actual, expected, tolerance):
    ar, ai = pair(actual)
    er, ei = pair(expected)
    assert (ar - er)**2 + (ai - ei)**2 <= tolerance**2


def mp_oracle(gaps, transitions, nu, theta, intervals, gains, epsilon):
    """Fixed100-digit independent modal F/J factors and absolute-weight budgets."""
    with mp.workdps(100):
        deltas = [mp_real(x) for x in gaps]
        v = [[mp_complex(z) for z in row] for row in transitions]
        W = [[[mp.conj(row[a]) * row[b] for b in range(2)] for a in range(2)] for row in v]
        nu, theta, epsilon = mp_real(nu), mp_real(theta), mp_real(epsilon)
        h = theta / intervals
        q = mp.exp(-nu * h)
        ac = -mp.expm1(-nu * theta) / nu
        ah = h / 2 * (-mp.expm1(-nu * theta)) * (1 + q) / (-mp.expm1(-nu * h))
        result = {key: [[None] * 2 for _ in range(2)] for key in
                  ("finite", "infinite", "estimate", "tail", "quadrature", "observation", "total", "proxy", "B", "infinite_scale")}
        times = [h * j for j in range(intervals + 1)]
        trace = []
        for t in times:
            trace.append([[complex(-1j * mp_real(gains[a][b]) * sum(
                W[n][a][b] * mp.exp(-1j * delta * t) - W[n][b][a] * mp.exp(1j * delta * t)
                for n, delta in enumerate(deltas))) for b in range(2)] for a in range(2)])
        for a in range(2):
            for b in range(2):
                gain = mp_real(gains[a][b])
                B = sum(abs(w[a][b]) + abs(w[b][a]) for w in W)
                finite = mp.mpc(0)
                estimate = mp.mpc(0)
                infinite = mp.mpc(0)
                for n, delta in enumerate(deltas):
                    plus, minus = nu + 1j * delta, nu - 1j * delta
                    Fp, Fm = -mp.expm1(-plus * theta) / plus, -mp.expm1(-minus * theta) / minus
                    Jp = h / 2 * (-mp.expm1(-plus * theta)) * (1 + mp.exp(-plus * h)) / (-mp.expm1(-plus * h))
                    Jm = h / 2 * (-mp.expm1(-minus * theta)) * (1 + mp.exp(-minus * h)) / (-mp.expm1(-minus * h))
                    finite += -1j * (W[n][a][b] * Fp - W[n][b][a] * Fm)
                    estimate += -1j * (W[n][a][b] * Jp - W[n][b][a] * Jm)
                    infinite += W[n][a][b] / (1j * nu - delta) - W[n][b][a] / (1j * nu + delta)
                shift = epsilon * (-1)**(a + b) * h / 2 * (-mp.expm1(-nu * theta)) * (-mp.expm1(-nu * h)) / (1 + q)
                tail = abs(gain) * 2 * mp.exp(-nu * theta) * mp.sqrt(
                    sum(w[a][a].real for w in W) * sum(w[b][b].real for w in W)) / nu
                quadrature = abs(gain) * h*h*theta / 6 * sum(abs(w[a][b]) * (nu + delta)**2 for w, delta in zip(W, deltas))
                observation = epsilon * ah
                proxy = 256 * mp_real(U) * (len(gaps) + intervals + 2) * (1 + theta * (nu + max(deltas))) * ah * (abs(gain) * B + epsilon)
                values = dict(finite=gain * finite, infinite=gain * infinite,
                              estimate=gain * estimate + shift, tail=tail, quadrature=quadrature,
                              observation=observation, total=tail + quadrature + observation,
                              proxy=proxy, B=B,
                              infinite_scale=abs(gain) * sum((abs(w[a][b]) + abs(w[b][a])) /
                                  mp.sqrt(nu*nu + delta*delta) for w, delta in zip(W, deltas)))
                for name, value in values.items():
                    result[name][a][b] = complex(value) if name in ("finite", "infinite", "estimate") else float(value)
        result.update(trace=np.array(trace), A_h=float(ah), A_c=float(ac))
        return result


def assert_modal_acquisition(data, gaps, transitions, nu, theta, n, gains, epsilon):
    assert set(data) == INTERNAL_KEYS
    acquisition_schema(data["record"], theta=theta, intervals=n, level=epsilon, gains=gains)
    assert data["sample_status"] == "available"
    assert data["sample_reason"] is data["sample_detail"] is None
    assert data["record"]["status"] == AVAILABLE
    np.testing.assert_array_equal(data["times"], [float(f(theta) * j / n) for j in range(n + 1)])
    assert data["times"][0] == 0.0 and data["times"][-1] == theta
    assert all(right > left for left, right in zip(data["times"], data["times"][1:]))
    for name, shape, dtype in (("times", (n + 1,), np.float64),
                               ("noiseless_samples", (n + 1, 2, 2), np.complex128),
                               ("samples", (n + 1, 2, 2), np.complex128)):
        assert type(data[name]) is np.ndarray and data[name].shape == shape
        assert data[name].dtype == np.dtype(dtype)
    expected = mp_oracle(gaps, transitions, nu, theta, n, gains, epsilon)
    K, M, dmax = len(gaps), n + 1, max(gaps)
    row = data["record"]
    for a in range(2):
        for b in range(2):
            B, G = expected["B"][a][b], abs(gains[a][b])
            for j, tau in enumerate(data["times"]):
                F = (K + 1) * (1 + dmax * tau)
                compare_component(data["noiseless_samples"][j, a, b], expected["trace"][j, a, b], allowance(F, G * B))
            F = (K + M + 1) * (1 + theta * (nu + dmax))
            comparisons = (("finite_reference", "finite", (K + 1) * (1 + theta * (nu + dmax)), G * B * expected["A_c"]),
                           ("infinite_reference", "infinite", K + 1, expected["infinite_scale"][a][b]),
                           ("estimate", "estimate", F, expected["A_h"] * (G * B + epsilon)))
            for public, name, factor, scale in comparisons:
                compare_component(complex_matrix(row[public])[a, b], expected[name][a][b], allowance(factor, scale))
            compare_component(real_matrix(row["errors"]["observation"])[a, b],
                              expected["observation"][a][b],
                              allowance((M + 1) * (1 + nu * theta), epsilon * expected["A_h"]))
            # No standalone approximate-equality tolerance for tail/quadrature/
            # proxy was frozen. Check positive budgets and diagnostic envelope
            # inequalities, plus separately exact real-weight fixtures below.
            for diagnostic in row["diagnostics"].values():
                assert diagnostic["status"] != "outside_diagnostic_envelope", diagnostic
    return expected


SMALL_FIXTURES = {
    "one_pole": ([1.0], [[1.0, 2.0]]),
    "pauli": ([1.0], [[1.0, 1j]]),
    "unequal_complex": ([1.0, 3.0], [[1.0, 1.0 + 1j], [2.0, -1j]]),
    "unequal_cancellation": ([1.0, 3.0], [[1.0, 1.0], [1.0, -1.0]]),
}


@pytest.fixture(scope="module")
def scalar_acquisitions():
    results = {}
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        for name, (gaps, transitions) in SMALL_FIXTURES.items():
            for theta, n in SCHEDULES:
                results[name, theta] = api._acquire(gaps, transitions, 1.25, theta, n, [[1.0] * 2] * 2, 0.0)
    return results


def test_fixed_scalar_modal_geometric_oracles_and_exact_kernel(scalar_acquisitions):
    for name, (gaps, transitions) in SMALL_FIXTURES.items():
        for theta, n in SCHEDULES:
            data = scalar_acquisitions[name, theta]
            assert_modal_acquisition(data, gaps, transitions, 1.25, theta, n, [[1.0] * 2] * 2, 0.0)
            exact = exact_kernel(gaps, transitions, 1.25, [[1.0] * 2] * 2)
            actual = complex_matrix(data["record"]["infinite_reference"])
            for a in range(2):
                for b in range(2):
                    assert actual[a, b] == complex(float(exact[a][b][0]), float(exact[a][b][1]))


def test_longer_window_is_not_finer_quadrature_or_monotone_total(scalar_acquisitions):
    for name in SMALL_FIXTURES:
        short, long = (scalar_acquisitions[name, theta] for theta in (4.0, 8.0))
        np.testing.assert_array_equal(short["times"], long["times"][:129])
        np.testing.assert_array_equal(short["noiseless_samples"], long["noiseless_samples"][:129])
        assert short["record"]["schedule"]["step"] == long["record"]["schedule"]["step"] == 1 / 32
        short_quad, long_quad = (real_matrix(data["record"]["errors"]["quadrature"]) for data in (short, long))
        np.testing.assert_array_equal(long_quad, 2 * short_quad)
        assert np.all(real_matrix(long["record"]["errors"]["tail"]) <= real_matrix(short["record"]["errors"]["tail"]))


def test_pauli_sign_endpoint_nonsymmetry_and_negative_ratio(scalar_acquisitions):
    data = scalar_acquisitions["pauli", 4.0]
    np.testing.assert_array_equal(data["noiseless_samples"][0], [[0, 2], [-2, 0]])
    K = complex_matrix(data["record"]["infinite_reference"])
    np.testing.assert_array_equal(K, [[-2 / (1 + 1.25**2), 2 * 1.25 / (1 + 1.25**2)],
                                     [-2 * 1.25 / (1 + 1.25**2), -2 / (1 + 1.25**2)]])
    result = api._ratio_disk(K, [[0.0] * 2] * 2)
    ratio_schema(result)
    assert complex_scalar(result["value"]).real < 0
    exact = cdiv(cmul(pair(K[0, 1]), pair(K[1, 0])), cmul(pair(K[0, 0]), pair(K[1, 1])))
    assert complex_scalar(result["value"]) == complex(float(exact[0]), float(exact[1]))
    assert result["radius"] == 0.0


def test_unequal_gap_cancellation_keeps_per_state_absolute_weight(scalar_acquisitions):
    data = scalar_acquisitions["unequal_cancellation", 4.0]
    assert sum(gram[0][1][0] for gram in exact_grams(SMALL_FIXTURES["unequal_cancellation"][1])) == 0
    assert complex_matrix(data["record"]["infinite_reference"])[0, 1] != 0
    assert real_matrix(data["record"]["errors"]["quadrature"])[0, 1] > 0
    assert np.any(data["noiseless_samples"][:, 0, 1] != 0)


def test_signed_gains_apply_after_completed_pauli_response_and_preserve_quotient():
    params = ([1.0], [[1.0, 1j]], 1.25, 4.0, 128)
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        plain = api._acquire(*params, [[1.0] * 2] * 2, 0.0)
        gained = api._acquire(*params, GAINS, 0.0)
    assert_modal_acquisition(plain, *params, [[1.0] * 2] * 2, 0.0)
    assert_modal_acquisition(gained, *params, GAINS, 0.0)
    np.testing.assert_array_equal(gained["noiseless_samples"][0], [[0, 28], [30, 0]])
    for key in ("finite_reference", "infinite_reference", "estimate"):
        left, right = complex_matrix(plain["record"][key]), complex_matrix(gained["record"][key])
        for a in range(2):
            for b in range(2):
                # Both acquisitions have already been compared independently
                # against the gain-aware modal oracle with frozen table scales.
                assert np.isfinite(right[a, b].real) and np.isfinite(right[a, b].imag)
        p = api._ratio_disk(left, [[0.0] * 2] * 2)
        q = api._ratio_disk(right, [[0.0] * 2] * 2)
        ratio_schema(p)
        ratio_schema(q)
        # Gain cancellation is exact algebra before independent entry rounding;
        # each emitted quotient is separately checked against its own inputs.
        for values, report in ((left, p), (right, q)):
            quotient = cdiv(cmul(pair(values[0, 1]), pair(values[1, 0])),
                            cmul(pair(values[0, 0]), pair(values[1, 1])))
            assert complex_scalar(report["value"]) == complex(float(quotient[0]), float(quotient[1]))
        assert complex_scalar(p["value"]).real < 0 and complex_scalar(q["value"]).real < 0
    assert gained["record"]["gains"] == [list(row) for row in GAINS]


def test_exact_degenerate_rotation_response_not_absolute_envelope_invariant():
    # Canonical exact degenerate rotation: (1,1),(1,-1) -> (sqrt2,0),(0,sqrt2).
    # The latter stored sqrt2 is rounded; use the sum of each basis's own frozen
    # comparison allowance rather than asserting exact floating Gram equality.
    original = [[1.0, 1.0], [1.0, -1.0]]
    rotated = [[math.sqrt(2.0), 0.0], [0.0, math.sqrt(2.0)]]
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        first = api._acquire([1.0, 1.0], original, 1.25, 4.0, 128, [[1.0] * 2] * 2, 0.0)
        second = api._acquire([1.0, 1.0], rotated, 1.25, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    left = assert_modal_acquisition(first, [1.0, 1.0], original, 1.25, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    right = assert_modal_acquisition(second, [1.0, 1.0], rotated, 1.25, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    assert real_matrix(first["record"]["errors"]["quadrature"])[0, 1] > 0
    assert real_matrix(second["record"]["errors"]["quadrature"])[0, 1] == 0.0
    assert complex_matrix(first["record"]["infinite_reference"])[0, 1] == 0j
    assert complex_matrix(second["record"]["infinite_reference"])[0, 1] == 0j
    for a in range(2):
        for b in range(2):
            for j, tau in enumerate(first["times"]):
                F = 3 * (1 + tau)
                tol = allowance(F, left["B"][a][b]) + allowance(F, right["B"][a][b])
                compare_component(first["noiseless_samples"][j, a, b], second["noiseless_samples"][j, a, b], tol)
    # Exact ideal rotated rows have |W01| sum0 vs2, despite total Gram2I.
    assert sum(abs(complex(float(w[0][1][0]), float(w[0][1][1]))) for w in exact_grams(original)) == 2
    assert sum(abs(complex(float(w[0][1][0]), float(w[0][1][1]))) for w in exact_grams(rotated)) == 0


def test_isolated_zero_background_noise_envelope_and_nested_prefix():
    results = []
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        for theta, n in SCHEDULES:
            results.append(api._acquire([1.0], [[0.0, 0.0]], 2.0**-20, theta, n, GAINS, 1e-8))
    for data, (theta, n) in zip(results, SCHEDULES):
        oracle = assert_modal_acquisition(data, [1.0], [[0.0, 0.0]], 2.0**-20, theta, n, GAINS, 1e-8)
        np.testing.assert_array_equal(data["noiseless_samples"], np.zeros_like(data["noiseless_samples"]))
        for j in range(n + 1):
            for a in range(2):
                for b in range(2):
                    assert data["samples"][j, a, b] == 1e-8 * (-1)**(j + a + b)
        estimate = complex_matrix(data["record"]["estimate"])
        assert estimate[0, 0].real > 0 and estimate[0, 1].real < 0
        factor = (n + 2) * (1 + (2.0**-20) * theta)
        for a in range(2):
            for b in range(2):
                compare_component(estimate[a, b], oracle["estimate"][a][b],
                                  allowance(factor, 1e-8 * oracle["A_h"]))
        np.testing.assert_array_equal(real_matrix(data["record"]["errors"]["observation"]),
                                      [[real_matrix(data["record"]["errors"]["observation"])[0, 0]] * 2] * 2)
    np.testing.assert_array_equal(results[0]["samples"], results[1]["samples"][:129])
    assert real_matrix(results[1]["record"]["errors"]["observation"])[0, 0] > real_matrix(results[0]["record"]["errors"]["observation"])[0, 0]


@pytest.mark.parametrize("theta,nu", [(2.0**-40, 1.0), (2.0**-100, 2.0**-100)])
def test_tiny_phase_finite_factor_is_nonzero_not_swallowed_by_one_minus_exp(theta, nu):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        data = api._acquire([1.0], [[1.0, 1j]], nu, theta, 2, [[1.0] * 2] * 2, 0.0)
    assert_modal_acquisition(data, [1.0], [[1.0, 1j]], nu, theta, 2, [[1.0] * 2] * 2, 0.0)
    finite = complex_matrix(data["record"]["finite_reference"])
    assert finite[0, 1].real > 0 and finite[1, 0].real < 0
    assert finite[0, 0].real < 0 and finite[0, 0] != 0


def test_exact_zero_gains_columns_and_endpoint_are_not_tolerance_claims():
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        zero_gain = api._acquire([1.0], [[1.0, 1j]], 1.0, 4.0, 128, [[0.0] * 2] * 2, 0.0)
        zero_column = api._time_trace([1.0], [[1.0, 0.0]], [0.0, 0.25])
    for key in ("finite_reference", "infinite_reference", "estimate"):
        np.testing.assert_array_equal(complex_matrix(zero_gain["record"][key]), np.zeros((2, 2)))
    for value in zero_gain["record"]["errors"].values():
        np.testing.assert_array_equal(real_matrix(value), np.zeros((2, 2)))
    np.testing.assert_array_equal(real_matrix(zero_gain["record"]["arithmetic_proxy"]), np.zeros((2, 2)))
    assert zero_gain["record"]["ratio"]["status"] == "zero_denominator"
    assert np.all(zero_column[:, 0, 1] == 0) and np.all(zero_column[:, 1, :] == 0)
    assert zero_column[0, 0, 0] == 0j


def exact_ratio_oracle(kernel, errors):
    """Independent exact product gate for rational real fixture diagonals."""
    k = [[pair(z) for z in row] for row in kernel]
    numerator = cmul(k[0][1], k[1][0])
    denominator = cmul(k[0][0], k[1][1])
    if denominator == (0, 0):
        return "zero_denominator", None, None
    # Exact real or Pythagorean diagonal moduli, independent of production's
    # radical-free gate; general irrational numerator moduli use100digits below.
    diagonal_moduli = []
    for z in (k[0][0], k[1][1]):
        squared = z[0]**2 + z[1]**2
        nr, dr = math.isqrt(squared.numerator), math.isqrt(squared.denominator)
        assert nr*nr == squared.numerator and dr*dr == squared.denominator
        diagonal_moduli.append(Fraction(nr, dr))
    a, b = diagonal_moduli
    u = [[f(z) for z in row] for row in errors]
    ed = a * u[1][1] + b * u[0][0] + u[0][0] * u[1][1]
    if a * b <= ed:
        return "unresolved_denominator", None, None
    quotient = cdiv(numerator, denominator)
    with mp.workdps(100):
        en = abs(mp_complex(kernel[0][1])) * mp_real(u[1][0]) + abs(mp_complex(kernel[1][0])) * mp_real(u[0][1]) + mp_real(u[0][1] * u[1][0])
        radius = (en + abs(mp.mpc(mp_real(quotient[0]), mp_real(quotient[1]))) * mp_real(ed)) / mp_real(a * b - ed)
        return "conditional_available", quotient, float(radius)


RATIO_FIXTURES = (
    ("center_zero", [[0.0, 1.0], [2.0, 1.0]], [[0.0] * 2] * 2),
    ("equality", [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]),
    ("just_excludes", [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 1.0 - 2.0**-53]]),
    ("just_contains", [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 1.0 + 2.0**-52]]),
    ("conservative", [[1.0, 0.5], [0.5, 1.0]], [[0.5, 0.0], [0.0, 0.5]]),
    ("zero_numerator", [[1.0, 0.0], [2.0, 1.0]], [[0.0, 0.25], [0.25, 0.0]]),
    ("complex_cross", [[2.0, 1.0 + 2j], [3.0 - 1j, 4.0]], [[0.0] * 2] * 2),
    ("subnormal", [[1.0, 2.0**-537], [2.0**-537, 1.0]], [[0.0] * 2] * 2),
    ("intermediate_underflow", [[2.0**-600, 2.0**-600], [2.0**-600, 2.0**-600]], [[0.0] * 2] * 2),
    ("intermediate_overflow", [[2.0**600, 2.0**600], [2.0**600, 2.0**600]], [[0.0] * 2] * 2),
    ("negative_ratio", [[1.0, 1j], [1j, 1.0]], [[0.0] * 2] * 2),
    ("complex_diagonal_equality", [[3.0 + 4j, 1.0], [1.0, 1j]], [[0.0, 0.0], [0.0, 1.0]]),
    ("complex_diagonal_excludes", [[3.0 + 4j, 1.0], [1.0, 1j]], [[0.0, 0.0], [0.0, 0.5]]),
)


@pytest.mark.parametrize("name,kernel,errors", RATIO_FIXTURES)
def test_exact_ratio_gate_complex_product_and_extreme_intermediates(name, kernel, errors):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._ratio_disk(kernel, errors)
    ratio_schema(result)
    status, quotient, radius = exact_ratio_oracle(kernel, errors)
    assert result["status"] == status
    if quotient is not None:
        assert complex_scalar(result["value"]) == complex(float(quotient[0]), float(quotient[1]))
        assert result["radius"] == radius
        if name == "subnormal":
            assert result["value"]["real"] == float(Q0) > 0
        if name == "zero_numerator":
            assert result["value"] == {"real": 0.0, "imag": 0.0} and result["radius"] > 0
        if name == "negative_ratio":
            assert result["value"]["real"] == -1
    if name == "conservative":
        assert errors[0][0] < abs(kernel[0][0]) and errors[1][1] < abs(kernel[1][1])


@pytest.mark.parametrize("name,kernel,errors,value_present", [
    ("quotient_overflow", [[2.0**-600, 1.0], [1.0, 2.0**-600]], [[0.0] * 2] * 2, False),
    ("quotient_underflow", [[1.0, 2.0**-600], [2.0**-600, 1.0]], [[0.0] * 2] * 2, False),
    ("imaginary_underflow", [[1.0, complex(1.0, 2.0**-600)], [2.0**-600, 1.0]], [[0.0] * 2] * 2, False),
    ("radius_overflow", [[1.0, 1.0], [1.0, 1.0]], [[0.0, 2.0**600], [2.0**600, 0.0]], True),
    ("radius_underflow", [[1.0, 0.0], [0.0, 1.0]], [[0.0, 2.0**-600], [2.0**-600, 0.0]], True),
])
def test_ratio_final_conversion_failure_retains_gate_and_quotient(name, kernel, errors, value_present):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._ratio_disk(kernel, errors)
    ratio_schema(result)
    assert result["status"] == UNAVAILABLE and result["reason"] == "ratio_arithmetic_unavailable"
    assert result["denominator_margin_status"] == "excludes_zero" and result["exact_denominator_gate"] is True
    assert (result["value"] is not None) is value_present
    assert result["radius"] is None


BAD_PUBLIC = (True, np.bool_(True), np.int64(3), np.float64(1), IntSubclass(3),
              FloatSubclass(1.0), Fraction(1), Decimal("1"), 1 + 0j, "1", ProtocolBomb(), None)


@pytest.mark.parametrize("name", ("L", "g", "C"))
@pytest.mark.parametrize("bad", BAD_PUBLIC)
def test_public_rejects_nonbase_scalars_before_owned_call(name, bad):
    args = dict(L=3, g=0.7, C=1.0)
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api.case_report(**args)
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("name,bad", [("L", 2), ("L", 6), ("L", 3.0), ("C", 0.49),
                                      ("C", 2.01), ("g", -1.0), ("g", 41.0),
                                      ("C", float("nan")), ("g", float("inf")),
                                      ("C", 2**53 + 1), ("g", 10**400)])
def test_public_domain_validation_no_physics(name, bad):
    args = dict(L=3, g=0.7, C=1.0)
    args[name] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api.case_report(**args)
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("g,C", [(float(Q0), 1.0), (2.0**-41, 1.0), (2.0**-40, 2.0)])
def test_positive_tiny_g_is_preserved_unavailable_without_joint_call(g, C):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        owned = api._owned_case(3, g, C)
    assert set(owned) == {"system", "normalized", "report"}
    assert owned["system"] is owned["normalized"] is None
    case_schema(owned["report"], (3, g, C))
    assert owned["report"]["reason"] == "tiny_g_unavailable"
    assert owned["report"]["g"] == g and g != 0
    for partition in owned["report"]["partitions"]:
        for row in partition["records"]:
            assert row["nu"] is None
            assert row["schedule"]["step"] == 1 / 32


def test_exact_tiny_g_boundary_and_free_are_not_preclassified_tiny():
    calls = []
    def unavailable(*args, **kwargs):
        calls.append((args, kwargs))
        raise current.NumericalUnavailable("boundary model intentionally unavailable")
    with pytest.MonkeyPatch.context() as patch:
        patch_aliases(patch, joint.joint_system, unavailable)
        for g in (0.0, 2.0**-40):
            owned = api._owned_case(3, g, 1.0)
            assert owned["report"]["reason"] == "base_system_unavailable"
    assert len(calls) == 2


def private_inputs():
    return dict(gaps=[1.0], transitions=[[1.0, 1j]], times=[0.0, 0.25],
                nu=1.0, theta=4.0, intervals=128, gains=[[1.0] * 2] * 2, error_level=0.0)


@pytest.mark.parametrize("field,bad", [
    ("gaps", []), ("gaps", [0.0]), ("gaps", [-1.0]), ("gaps", [True]),
    ("gaps", [1 + 0j]), ("gaps", [float("nan")]), ("gaps", [float("inf")]),
    ("gaps", [2**53 + 1]), ("gaps", [1.0] * 512),
    ("transitions", [[True, 1.0]]), ("transitions", [[1.0], [2.0]]),
    ("transitions", [[1.0, 2.0], [3.0]]), ("transitions", [["1", 2.0]]),
    ("transitions", [[2**53 + 1, 1.0]]), ("transitions", [[float("inf"), 1.0]]),
    ("times", []), ("times", [-0.1]), ("times", [0.0, 0.0]),
    ("times", [1.0, 0.0]), ("times", [0.0] * 258), ("times", [0j]),
    ("times", [False]), ("times", [float("nan")]),
])
def test_time_trace_rejects_malformed_leaves_shapes_domains(field, bad):
    args = private_inputs()
    args[field] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api._time_trace(args["gaps"], args["transitions"], args["times"])
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("container", [ProtocolBomb(), ListSubclass([1.0]), TupleSubclass((1.0,))])
def test_private_container_subclasses_and_protocol_rejected(container):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api._time_trace(container, [[1.0, 1.0]], [0.0])
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("kind", ("array_subclass", "object", "bool", "longdouble", "clongdouble"))
def test_private_array_storage_validation_before_conversion(kind):
    if kind == "array_subclass":
        gaps = np.array([1.0]).view(ArraySubclass)
    elif kind == "object":
        gaps = np.array([1.0], dtype=object)
    elif kind == "bool":
        gaps = np.array([True], dtype=bool)
    else:
        dtype = np.longdouble if kind == "longdouble" else np.clongdouble
        if np.dtype(dtype).itemsize <= (8 if kind == "longdouble" else 16):
            pytest.skip("platform has no wider precision dtype to reject")
        gaps = np.array([1.0], dtype=dtype)
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api._time_trace(gaps, [[1.0, 1.0]], [0.0])
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("field,bad", [
    ("nu", 0.0), ("nu", -1.0), ("nu", 1j), ("nu", True), ("nu", float("inf")),
    ("theta", 0.0), ("theta", float("nan")), ("theta", ProtocolBomb()),
    ("intervals", 0), ("intervals", 257), ("intervals", 2.0), ("intervals", True),
    ("intervals", np.bool_(True)), ("gains", [[1 + 0j] * 2] * 2),
    ("gains", [[1.0, True], [1.0, 1.0]]), ("gains", [[1.0, 2.0]]),
    ("gains", [[float("nan")] * 2] * 2), ("error_level", -1e-8),
    ("error_level", 0j), ("error_level", True),
])
def test_acquire_strict_scalar_and_real_gain_validation(field, bad):
    args = private_inputs()
    args.pop("times")
    args[field] = bad
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api._acquire(**args)
    assert not isinstance(exc.value, current.NumericalUnavailable)


@pytest.mark.parametrize("kernel,errors", [
    ([[1, 2], [3, 4]], [[-0.1, 0], [0, 0]]),
    ([[1, 2], [3, 4]], [[0j, 0j], [0j, 0j]]),
    ([[1, True], [3, 4]], [[0, 0], [0, 0]]),
    ([[1, 2], [3]], [[0, 0], [0, 0]]),
    ([[1, 2], [3, 4]], [[0, False], [0, 0]]),
    ([[2**53 + 1, 2], [3, 4]], [[0, 0], [0, 0]]),
    ([[float("nan"), 2], [3, 4]], [[0, 0], [0, 0]]),
    (ProtocolBomb(), [[0, 0], [0, 0]]),
])
def test_ratio_malformed_inputs_raise_not_unavailable(kernel, errors):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        with pytest.raises(ValueError) as exc:
            api._ratio_disk(kernel, errors)
    assert not isinstance(exc.value, current.NumericalUnavailable)


def test_private_arrays_owned_readonly_and_independent_repeated_acquisitions():
    gaps = np.array([1.0])
    transitions = np.array([[1.0, 1j]])
    times = np.array([0.0, 0.25])
    for value in (gaps, transitions, times):
        value.flags.writeable = False
    before = [value.copy() for value in (gaps, transitions, times)]
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        trace = api._time_trace(gaps, transitions, times)
        first = api._acquire(gaps, transitions, 1.0, 4.0, np.int64(128), [[1.0] * 2] * 2, 0.0)
        second = api._acquire(gaps, transitions, 1.0, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    for value, old in zip((gaps, transitions, times), before):
        np.testing.assert_array_equal(value, old)
        assert not value.flags.writeable and not np.shares_memory(trace, value)
    for name in ("times", "samples", "noiseless_samples"):
        assert not np.shares_memory(first[name], second[name])
        for source in (gaps, transitions, times):
            assert not np.shares_memory(first[name], source)
    saved = copy.deepcopy(second)
    first["samples"][:] = 100
    first["record"]["estimate"]["imag"][0][0] = 99.0
    np.testing.assert_array_equal(second["samples"], saved["samples"])
    assert second["record"] == saved["record"]


@pytest.mark.parametrize("delta,nu,theta,detail", [(2.0**21, 1.0, 1.0, "phase"),
                                                    (1.0, 1000.0, 1.0, "exp"),
                                                    (2.0**-600, 1.0, 2.0**-600, "phase")])
def test_phase_or_exponential_failure_preserves_rational_infinite_reference(delta, nu, theta, detail):
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._acquire([delta], [[1.0, 1j]], nu, theta, 2, [[1.0] * 2] * 2, 0.0)
    assert set(result) == INTERNAL_KEYS
    row = result["record"]
    acquisition_schema(row, theta=theta, intervals=2, level=0.0, gains=[[1.0] * 2] * 2)
    assert row["status"] == PARTIAL
    assert row["infinite_reference"] is not None
    exact = exact_kernel([delta], [[1.0, 1j]], nu, [[1.0] * 2] * 2)
    for a in range(2):
        for b in range(2):
            assert complex_matrix(row["infinite_reference"])[a, b] == complex(float(exact[a][b][0]), float(exact[a][b][1]))
    assert any(detail in str(value).lower() for key, value in row.items() if "detail" in key)


def test_valid_unrepresentable_grid_step_retains_metadata_and_independent_reference():
    theta = float(Q0)
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._acquire([1.0], [[1.0, 1j]], 1.0, theta, 2, [[1.0] * 2] * 2, 0.0)
    assert result["times"] is result["samples"] is result["noiseless_samples"] is None
    assert result["sample_status"] == UNAVAILABLE and result["sample_reason"] == "trace_unavailable"
    assert result["sample_detail"]
    row = result["record"]
    acquisition_schema(row, theta=theta, intervals=2, level=0.0, gains=[[1.0] * 2] * 2)
    assert row["schedule"] == dict(theta=theta, intervals=2, sample_count=3, step=None)
    assert row["estimate"] is None and row["infinite_reference"] is not None
    assert row["status"] == PARTIAL


def test_subnormal_final_kernel_survives_underflowing_gram_intermediate():
    # W~2^-1200 cannot be materialized; dividing by delta=nu=2^-130
    # gives a representable subnormal kernel~2^-1070. This is a rational reference
    # check even if another required acquisition stage is unavailable.
    gaps, transitions, nu = [2.0**-130], [[2.0**-600, 2.0**-600]], 2.0**-130
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._acquire(gaps, transitions, nu, 1.0, 2, [[1.0] * 2] * 2, 0.0)
    row = result["record"]
    assert row["infinite_reference"] is not None
    expected = -2.0**-1070
    np.testing.assert_array_equal(complex_matrix(row["infinite_reference"]), [[expected] * 2] * 2)
    assert expected != 0


def test_natural_proxy_underflow_preserves_subnormal_estimate_budgets_and_ratio():
    transitions = [[2.0**-530, 2.0**-530]]
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        result = api._acquire([1.0], transitions, 1.0, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    row = result["record"]
    acquisition_schema(row, theta=4.0, intervals=128, level=0.0, gains=[[1.0] * 2] * 2)
    assert row["status"] == PARTIAL
    assert row["arithmetic_proxy"] is None and row["proxy_status"] == UNAVAILABLE
    assert row["proxy_reason"] == "arithmetic_proxy_unavailable"
    assert row["estimate"] is not None and row["error_status"] == AVAILABLE
    # Pre-execution scale derivation: W=2^-1060; tail=2exp(-4)W,
    # quadrature=W/384, observation=0. Each positive budget is representable
    # (quadrature exceeds42 minsubnormals), while prescribed P rounds to zero.
    for name in ("tail", "quadrature", "total"):
        budget = real_matrix(row["errors"][name])
        assert np.all(budget > 0) and np.all(budget < sys.float_info.min)
    np.testing.assert_array_equal(real_matrix(row["errors"]["observation"]), np.zeros((2, 2)))
    assert complex_matrix(row["estimate"])[0, 0].real < 0
    assert abs(complex_matrix(row["estimate"])[0, 0].real) < sys.float_info.min
    expected_status, quotient, radius = exact_ratio_oracle(
        complex_matrix(row["estimate"]), real_matrix(row["errors"]["total"]))
    assert row["ratio"]["status"] == expected_status
    assert row["ratio"]["exact_denominator_gate"] is True
    if quotient is not None:
        assert complex_scalar(row["ratio"]["value"]) == complex(float(quotient[0]), float(quotient[1]))
    for diagnostic in row["diagnostics"].values():
        assert diagnostic["status"] == "inconclusive"
        assert diagnostic["residual"] is not None and diagnostic["analytic_limit"] is not None
        assert diagnostic["proxy"] is None


def test_exact_real_weight_quadrature_formula_without_transcendental_tolerance(scalar_acquisitions):
    for name in ("one_pole", "unequal_cancellation"):
        gaps, transitions = SMALL_FIXTURES[name]
        grams = exact_grams(transitions)
        for theta, n in SCHEDULES:
            actual = real_matrix(scalar_acquisitions[name, theta]["record"]["errors"]["quadrature"])
            for a in range(2):
                for b in range(2):
                    exact = (f(theta) / n)**2 * f(theta) / 6 * sum(
                        abs(w[a][b][0]) * (f(1.25) + f(delta))**2 for w, delta in zip(grams, gaps))
                    assert actual[a, b] == float(exact)


@pytest.fixture(scope="module")
def shared_physical_demo():
    calls = {"owned": [], "joint": [], "eigh": [], "trace": []}
    owned_objects = {}
    active = {"owned": None, "joint": None}
    originals = api._owned_case, joint.joint_system, np.linalg.eigh, api._time_trace

    def owned(L, g, C):
        key = (L, g, C)
        calls["owned"].append(key)
        assert len(calls["owned"]) <= 9 and key == GRID[len(calls["owned"]) - 1]
        active["owned"] = key
        try:
            result = originals[0](L, g, C)
        finally:
            active["owned"] = None
        assert set(result) == {"system", "normalized", "report"}
        owned_objects[key] = result
        return result

    def system(*args, **kwargs):
        bound = inspect.signature(originals[1]).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "g", "C"))
        assert key == active["owned"] and key not in calls["joint"]
        assert bound.arguments["m"] == 1
        calls["joint"].append(key)
        active["joint"] = key
        try:
            return originals[1](*args, **kwargs)
        finally:
            active["joint"] = None

    def eigh(H, *args, **kwargs):
        key = active["joint"]
        assert key in GRID and key not in [item[0] for item in calls["eigh"]]
        assert H.shape == ({3: 10, 4: 35, 5: 126}[key[0]],) * 2
        calls["eigh"].append((key, H.shape))
        return originals[2](H, *args, **kwargs)

    def trace(gaps, transitions, times):
        key = active["owned"]
        assert key in GRID
        calls["trace"].append((key, len(times)))
        assert len(times) == 257
        assert sum(old == key for old, _ in calls["trace"]) <= 2
        return originals[3](gaps, transitions, times)

    with pytest.MonkeyPatch.context() as patch:
        for original, replacement in zip(originals, (owned, system, eigh, trace)):
            patch_aliases(patch, original, replacement)
        for target in (joint.joint_report, prediction.case_report, np.linalg.eigvalsh,
                       np.linalg.eig, np.linalg.eigvals, np.linalg.svd):
            patch_aliases(patch, target, deny)
        report = api.demonstration_report()
    return report, calls, owned_objects


def assert_no_diagnostic_mismatch(report):
    cases = [row for case in report["physical_cases"] for partition in case["partitions"] for row in partition["records"]]
    cases.extend(report["synthetic"]["records"])
    for row in cases:
        for diagnostic in row["diagnostics"].values():
            assert diagnostic["status"] != "outside_diagnostic_envelope", diagnostic


def test_physical_demo_complete_slots_metadata_and_attempt_budget(shared_physical_demo):
    report, calls, owned = shared_physical_demo
    demo_schema(report)
    assert_no_diagnostic_mismatch(report)
    assert calls["owned"] == list(GRID)
    assert set(calls["joint"]) <= set(GRID) and len(calls["joint"]) == len(set(calls["joint"])) <= 9
    eigh_keys = [key for key, _ in calls["eigh"]]
    assert len(eigh_keys) == len(set(eigh_keys)) <= 9 and set(eigh_keys) <= set(calls["joint"])
    for key, case in zip(GRID, report["physical_cases"]):
        if case["status"] == AVAILABLE:
            assert key in owned and owned[key]["system"] is not None
            assert key in calls["joint"] and key in eigh_keys
        else:
            assert case["reason"] in REASONS and case["detail"]
        if key in owned and owned[key]["system"] is None:
            assert owned[key]["normalized"] is None
            assert case["dimension"] is None
            assert case["status"] == UNAVAILABLE
    if all(case["status"] == AVAILABLE for case in report["physical_cases"]):
        assert len(calls["joint"]) == len(calls["eigh"]) == 9
        assert len(owned) == 9
    assert len({id(value["system"]) for value in owned.values() if value["system"] is not None}) == sum(value["system"] is not None for value in owned.values())
    assert api.NumericalUnavailable is current.NumericalUnavailable


def independent_bose_sources(L, g, C):
    basis = tuple(sorted(tuple(sites.count(x) for x in range(L))
                         for sites in itertools.combinations_with_replacement(range(L), L)))
    index = {state: j for j, state in enumerate(basis)}
    d = len(basis)
    onsite, bonds = [], []
    for x in range(L):
        onsite.append(np.diag([g * state[x] * (state[x] - 1) / 2 for state in basis]))
        bond = np.zeros((d, d))
        for col, state in enumerate(basis):
            for source, target in ((x, (x + 1) % L), ((x + 1) % L, x)):
                if state[source]:
                    moved = list(state)
                    moved[source] -= 1
                    moved[target] += 1
                    bond[index[tuple(moved)], col] -= C * math.sqrt(state[source] * (state[target] + 1))
        bonds.append(bond)
    weights = []
    for x in range(L):
        residue = x % L
        cosine = (1.0 if residue == 0 else -1.0 if 2 * residue == L else
                  0.0 if 4 * residue in (L, 3 * L) else math.cos(2 * math.pi * residue / L))
        weights.append(math.sqrt(2.0 / L) * cosine)
    rho = np.diag([sum(weights[x] * (state[x] - 1) for x in range(L)) for state in basis])
    sources = {}
    for name, a, b in (("symmetric", 0.5, 0.5), ("improved", 0.25, 0.75)):
        local = [onsite[x] + a * bonds[(x - 1) % L] + b * bonds[x] for x in range(L)]
        sources[name] = np.array([rho, sum(weights[x] * local[x] for x in range(L)) / C], dtype=complex)
    return basis, sum(onsite) + sum(bonds), sources


@pytest.mark.parametrize("key", GRID)
def test_owned_normalization_metadata_and_independent_matrix_structure(shared_physical_demo, key):
    report, _, owned = shared_physical_demo
    case = next(case for case in report["physical_cases"] if (case["L"], case["g"], case["C"]) == key)
    if key not in owned or owned[key]["normalized"] is None:
        assert case["status"] != AVAILABLE and case["detail"]
        pytest.skip("computationally unavailable owned normalization {!r}: {}".format(key, case["detail"]))
    data = owned[key]
    system, normalized = data["system"], data["normalized"]
    assert set(normalized) == {"gaps", "ground_gap", "resolution", "partitions"}
    assert set(normalized["partitions"]) == set(PARTITIONS)
    L, g, C = key
    basis, H, sources = independent_bose_sources(L, g, C)
    assert tuple(system["basis"]) == basis
    assert len(basis) == math.comb(2 * L - 1, L)
    # Occupation/connectivity and dyadic onsite counts are exact checks; generic
    # sqrt/libm matrix comparisons have no invented absolute tolerance here.
    assert np.array_equal(system["H"] != 0, H != 0)
    for j, state in enumerate(basis):
        exact_diagonal = f(g) * sum(n * (n - 1) // 2 for n in state)
        assert system["H"][j, j] == float(exact_diagonal)
    assert normalized["gaps"].shape == (len(basis) - 1,)
    assert normalized["gaps"].dtype == np.float64
    np.testing.assert_array_equal(normalized["gaps"], system["gaps"][1:] / C)
    assert normalized["ground_gap"] == system["gaps"][1] / C
    assert normalized["resolution"] == system["resolution"] / C
    for name, public in zip(PARTITIONS, case["partitions"]):
        p = normalized["partitions"][name]
        if p is None:
            assert public["status"] != AVAILABLE and public["detail"]
            continue
        assert set(p) == {"sources", "transitions", "total_weight", "maximum_times", "maximum_trace",
                           "trace_status", "trace_reason", "trace_detail"}
        assert p["sources"].shape == (2, len(basis), len(basis))
        assert p["transitions"].shape == (len(basis) - 1, 2)
        assert p["total_weight"].shape == (2, 2)
        # General source cancellation patterns depend on separately rounded
        # trigonometric weights; no zero-pattern or invented tolerance assertion.
        for a, degree in enumerate((0, 1)):
            np.testing.assert_array_equal(p["sources"][a], system["partitions"][name]["sources"][a] / C**degree)
            np.testing.assert_array_equal(p["transitions"][:, a], system["partitions"][name]["transitions"][:, a] / C**degree)
        for array in (normalized["gaps"], p["sources"], p["transitions"], p["total_weight"]):
            assert type(array) is np.ndarray
        if p["maximum_trace"] is not None:
            assert p["maximum_trace"].shape == (257, 2, 2) and p["maximum_trace"].dtype == np.complex128
            assert p["maximum_times"].shape == (257,)
            np.testing.assert_array_equal(p["maximum_times"], [j / 32 for j in range(257)])
            assert p["trace_status"] == "available"
        else:
            assert p["trace_status"] == UNAVAILABLE and p["trace_detail"]


def test_one_owned_reference_both_partition_sparse_taylor_actions(shared_physical_demo):
    report, _, owned = shared_physical_demo
    key = (3, 0.7, 1.0)
    if key not in owned or owned[key]["normalized"] is None:
        case = report["physical_cases"][1]
        assert case["status"] != AVAILABLE
        pytest.skip("computationally unavailable Taylor reference: " + str(case["detail"]))
    system, normalized = owned[key]["system"], owned[key]["normalized"]
    _, H, sources = independent_bose_sources(*key)
    ground = system["ground"]
    E0 = float(system["energies"][0])
    A = (H - E0 * np.eye(len(H))) / key[2]
    times = (0.0, 2.0**-10, 2.0**-8)
    products = 0
    calls = 0
    skipped = []
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        for name in PARTITIONS:
            p = normalized["partitions"][name]
            public = next(partition for partition in owned[key]["report"]["partitions"]
                          if partition["partition"] == name)
            if p is None:
                assert public["status"] != AVAILABLE
                assert public["reason"] in REASONS and public["detail"]
                skipped.append(name + ": normalization unavailable: " + public["detail"])
                continue
            if p["maximum_trace"] is None or p["trace_status"] != "available":
                assert p["maximum_trace"] is None
                assert p["trace_status"] == UNAVAILABLE
                assert p["trace_reason"] == "trace_unavailable"
                assert type(p["trace_detail"]) is str and p["trace_detail"]
                assert public["status"] != AVAILABLE
                skipped.append(name + ": shared maximum trace unavailable: " + p["trace_detail"])
                continue  # No sparse retry after an unavailable shared trace.
            assert p["trace_reason"] is p["trace_detail"] is None
            vectors = []
            norm = np.vdot(ground, ground)
            for O in sources[name]:
                direct = O @ ground
                vectors.append(direct - ground * (np.vdot(ground, direct) / norm))
            calls += 1  # Attempt, including legitimate numerical failure.
            try:
                trace = api._time_trace(normalized["gaps"], p["transitions"], times)
            except current.NumericalUnavailable as exc:
                skipped.append(name + ": sparse helper numerically unavailable: " + str(exc))
                continue
            for j, tau in enumerate(times):
                evolved = []
                for v in vectors:
                    term, total = v.copy(), v.copy()
                    if tau:
                        for k in range(1, 33):
                            term = (-1j * tau / k) * (A @ term)
                            products += 1
                            total = total + term
                    evolved.append(total)
                x = tau * ((key[1] / key[2]) * math.comb(3, 2) + 4 * 3)
                for a in range(2):
                    for b in range(2):
                        S = 2 * math.sqrt(float(np.vdot(vectors[a], vectors[a]).real)) * math.sqrt(float(np.vdot(vectors[b], vectors[b]).real))
                        with mp.workdps(100):
                            remainder = 0.0 if tau == 0 else float(mp_real(S) * mp.exp(mp_real(x)) * mp_real(x)**33 / mp.factorial(33))
                        F = (33 * (len(H) + 1) + len(normalized["gaps"]) + 1) * (1 + x) * math.exp(x)
                        expected = 2 * np.vdot(vectors[a], evolved[b]).imag
                        compare_component(trace[j, a, b], expected, f(remainder) + allowance(F, S))
    assert calls <= 2 and products <= 256
    if skipped:
        pytest.skip("COMPUTATIONAL unavailable Taylor dependencies (not platform skips): " + "; ".join(skipped))
    assert calls == 2 and products == 256


@pytest.mark.parametrize("L", (3, 4, 5))
def test_free_owned_modal_identity_without_new_solve(shared_physical_demo, L):
    report, _, owned = shared_physical_demo
    key = (L, 0.0, 1.0)
    if key not in owned or owned[key]["normalized"] is None:
        pytest.skip("computationally unavailable free normalization for L={}".format(L))
    normalized = owned[key]["normalized"]
    k = 2 * math.pi / L
    delta = 4 * math.sin(k / 2)**2
    alpha = -(1 + math.cos(k))
    unavailable_partitions = []
    for name in PARTITIONS:
        p = normalized["partitions"][name]
        if p is None or p["maximum_trace"] is None:
            unavailable_partitions.append(name)
            continue
        beta = 0.0 if name == "symmetric" else math.sin(k) / 2
        gram = [[1.0, alpha], [alpha, alpha*alpha + beta*beta]]
        for j, tau in enumerate(p["maximum_times"]):
            for a in range(2):
                for b in range(2):
                    with mp.workdps(100):
                        expected = float(-2 * mp_real(gram[a][b]) * mp.sin(mp_real(delta) * mp_real(tau)))
                    # Free analytic pole is a one-mode comparison, but the
                    # computed complete excited measure retains all tiny modes.
                    computed_B = sum(abs(complex(float(w[a][b][0]), float(w[a][b][1]))) +
                                     abs(complex(float(w[b][a][0]), float(w[b][a][1])))
                                     for w in exact_grams(p["transitions"]))
                    F = (len(normalized["gaps"]) + 1) * (1 + max(normalized["gaps"]) * tau)
                    compare_component(p["maximum_trace"][j, a, b], expected,
                                      allowance(F, computed_B) + allowance(2 * (1 + delta * tau), 2 * abs(gram[a][b])))
        Q = alpha*alpha / (alpha*alpha + beta*beta)
        assert Q == 1.0 if name == "symmetric" else 0 < Q < 1
        # POST-AUTHORSHIP review addition: emitted ratio is the quotient of
        # THAT finite sampled estimate, not the ideal infinite free Q above.
        case = next(case for case in report["physical_cases"]
                    if (case["L"], case["g"], case["C"]) == key)
        public = next(partition for partition in case["partitions"] if partition["partition"] == name)
        for row in public["records"]:
            ratio_schema(row["ratio"])
            if row["ratio"]["status"] == "conditional_available":
                assert row["estimate"] is not None
                estimate = complex_matrix(row["estimate"])
                quotient = cdiv(cmul(pair(estimate[0, 1]), pair(estimate[1, 0])),
                                cmul(pair(estimate[0, 0]), pair(estimate[1, 1])))
                assert complex_scalar(row["ratio"]["value"]) == complex(float(quotient[0]), float(quotient[1]))
            else:
                assert row["ratio"]["status"] in ("zero_denominator", "unresolved_denominator",
                                                   UNAVAILABLE, "dependency_unavailable")
    if unavailable_partitions:
        pytest.skip("computationally unavailable free traces: " + ",".join(unavailable_partitions))


def test_post_authorship_paired_synthetic_noise_uses_two_full_allowances(shared_physical_demo):
    """Reuse all six existing pairs; no acquisition, trace or solver invocation."""
    report, _, owned = shared_physical_demo
    key = (3, 0.7, 1.0)
    if key not in owned or owned[key]["normalized"] is None:
        case = next(case for case in report["physical_cases"]
                    if (case["L"], case["g"], case["C"]) == key)
        assert case["status"] != AVAILABLE and case["detail"]
        pytest.skip("COMPUTATIONAL unavailable synthetic noise reference: " + case["detail"])
    normalized = owned[key]["normalized"]
    p = normalized["partitions"]["symmetric"]
    if p is None:
        pytest.skip("COMPUTATIONAL unavailable symmetric normalization for paired noise")
    grams = exact_grams(p["transitions"])
    with mp.workdps(100):
        B = [[float(sum(mp.sqrt(mp_real(w[a][b][0])**2 + mp_real(w[a][b][1])**2)
                        + mp.sqrt(mp_real(w[b][a][0])**2 + mp_real(w[b][a][1])**2)
                        for w in grams)) for b in range(2)] for a in range(2)]
    rows = {(row["error_level"], row["schedule"]["theta"], row["eta"]): row
            for row in report["synthetic"]["records"]}
    assert len(rows) == 12
    unavailable = []
    for theta, n in SCHEDULES:
        for eta in ETAS:
            plain, noisy = rows[0.0, theta, eta], rows[1e-8, theta, eta]
            if plain["estimate"] is None or noisy["estimate"] is None:
                assert plain["status"] != AVAILABLE or noisy["status"] != AVAILABLE
                unavailable.append((theta, eta, plain["detail"], noisy["detail"]))
                continue
            assert plain["nu"] == noisy["nu"] and plain["nu"] is not None
            nu = plain["nu"]
            with mp.workdps(100):
                mh, mt, mn = mp_real(theta) / n, mp_real(theta), mp_real(nu)
                q = mp.exp(-mn * mh)
                one_minus_qn = -mp.expm1(-mn * mt)
                one_minus_q = -mp.expm1(-mn * mh)
                Ah = float(mh / 2 * one_minus_qn * (1 + q) / one_minus_q)
                shift = float(mp_real(1e-8) * mh / 2 * one_minus_qn * one_minus_q / (1 + q))
            X, Y = complex_matrix(plain["estimate"]), complex_matrix(noisy["estimate"])
            K, M = len(normalized["gaps"]), n + 1
            F = (K + M + 1) * (1 + theta * (nu + max(normalized["gaps"])))
            for a in range(2):
                for b in range(2):
                    # Exact subtraction of emitted components avoids adding a
                    # third rounding error to the two independently rounded
                    # estimate allowances. Scale is pre-cancellation, gain-aware.
                    yr, yi = pair(Y[a, b])
                    xr, xi = pair(X[a, b])
                    residual_real = yr - xr - f(shift) * (-1)**(a + b)
                    residual_imag = yi - xi
                    S0 = f(Ah) * f(abs(GAINS[a][b])) * f(B[a][b])
                    S1 = f(Ah) * (f(abs(GAINS[a][b])) * f(B[a][b]) + f(1e-8))
                    tol = allowance(F, S0) + allowance(F, S1)
                    assert residual_real**2 + residual_imag**2 <= tol**2
    if unavailable:
        pytest.skip("COMPUTATIONAL unavailable paired synthetic estimates: " + repr(unavailable))


def failure(exception):
    def raise_it(*args, **kwargs):
        raise exception
    return raise_it


@pytest.mark.parametrize("stage", ("owned", "joint"))
def test_failed_demo_preserves_all_slots_and_reference_without_retries(stage):
    attempts = []
    exception = current.NumericalUnavailable("frozen early failure sentinel")
    original = api._owned_case if stage == "owned" else joint.joint_system

    def unavailable(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments[name] for name in ("L", "g", "C"))
        attempts.append(key)
        assert len(attempts) <= 9 and key == GRID[len(attempts) - 1]
        raise exception

    with pytest.MonkeyPatch.context() as patch:
        # Patch the designated owned seam; all physics below it is forbidden.
        for target in (occupation.fixed_number_model, occupation._occupations, np.linalg.eigh,
                       np.linalg.eigvalsh, np.linalg.svd):
            patch_aliases(patch, target, deny)
        patch_aliases(patch, original, unavailable)
        report = api.demonstration_report()
    assert attempts == list(GRID)
    demo_schema(report)
    for case in report["physical_cases"]:
        assert case["status"] == UNAVAILABLE and case["reason"] == "base_system_unavailable"
        assert "frozen early failure sentinel" in case["detail"]
        assert case["dimension"] is case["ground_gap"] is case["normalized_ground_gap"] is None
        for partition in case["partitions"]:
            for row in partition["records"]:
                assert row["nu"] is None and row["status"] == UNAVAILABLE
                assert row["ratio"]["status"] == "dependency_unavailable"
    assert report["synthetic"]["status"] == UNAVAILABLE
    assert all(row["nu"] is None for row in report["synthetic"]["records"])


@pytest.mark.parametrize("exception_type", (ValueError, RuntimeError, TypeError))
def test_unexpected_programming_errors_not_converted_to_unavailable(exception_type):
    exception = exception_type("unexpected programming sentinel")
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        patch_aliases(patch, api._owned_case, failure(exception))
        with pytest.raises(exception_type) as caught:
            api.demonstration_report()
    assert caught.value is exception


def test_standalone_synthetic_calls_owned_reference_once_even_after_previous_failure():
    calls = []
    def unavailable(L, g, C):
        calls.append((L, g, C))
        raise current.NumericalUnavailable("standalone reference sentinel")
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        patch_aliases(patch, api._owned_case, unavailable)
        first = api.synthetic_report()
        second = api.synthetic_report()
    assert calls == [(3, 0.7, 1.0)] * 2
    synthetic_schema(first)
    synthetic_schema(second)
    assert first == second and first is not second
    first["records"][0]["schedule"]["theta"] = -1.0
    assert second["records"][0]["schedule"]["theta"] == 4.0


def test_trace_failure_keeps_independent_references_and_budgets():
    exception = current.NumericalUnavailable("trace deliberately unavailable")
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        patch_aliases(patch, api._time_trace, failure(exception))
        result = api._acquire([1.0], [[1.0, 1j]], 1.0, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    assert result["sample_status"] == UNAVAILABLE
    assert result["sample_reason"] == "trace_unavailable"
    assert result["noiseless_samples"] is result["samples"] is None
    row = result["record"]
    acquisition_schema(row, theta=4.0, intervals=128, level=0.0, gains=[[1.0] * 2] * 2)
    assert row["status"] == PARTIAL and row["estimate"] is None
    assert row["finite_reference"] is not None and row["infinite_reference"] is not None
    assert row["error_status"] == AVAILABLE
    assert row["ratio"]["status"] == "dependency_unavailable"


def test_raw_complex_trace_residual_is_retained_not_real_projected():
    def complex_trace(gaps, transitions, times):
        # Deliberate raw floating residual: fixture does not claim this is an
        # exact Hermitian-source identity. Serialization must retain it.
        return np.full((len(times), 2, 2), complex(0.0, 2.0**-500), dtype=np.complex128)
    with pytest.MonkeyPatch.context() as patch:
        block_physics(patch)
        patch_aliases(patch, api._time_trace, complex_trace)
        result = api._acquire([1.0], [[1.0, 1j]], 1.0, 4.0, 128, [[1.0] * 2] * 2, 0.0)
    assert np.all(result["samples"].imag == 2.0**-500)
    estimate = complex_matrix(result["record"]["estimate"])
    assert np.all(estimate.imag > 0)
    assert result["record"]["diagnostics"]["quadrature"]["status"] == "outside_diagnostic_envelope"
    # A discrepancy is retained as such, not relabeled numerical unavailability.
    assert result["record"]["status"] == AVAILABLE


def synthetic_joint_system(L, g, C=1.0, m=1):
    """Independent small arrays only; NEVER a solved physical Hamiltonian."""
    d = math.comb(2 * L - 1, L)
    H = np.diag(np.arange(d, dtype=float))
    ground = np.eye(d)[:, 0].astype(complex)
    sources = np.zeros((2, d, d), dtype=complex)
    sources[0, 1, 0] = sources[0, 0, 1] = 1
    sources[1, 1, 0] = sources[1, 0, 1] = 2 * C
    transitions = np.zeros((d - 1, 2), dtype=complex)
    transitions[0] = [1, 2 * C]
    partitions = {}
    for name in PARTITIONS:
        u = transitions.copy()
        grams = np.array([np.outer(row.conj(), row) for row in u])
        partitions[name] = dict(sources=sources.copy(), transitions=u,
                                connected=np.column_stack((sources[0] @ ground, sources[1] @ ground)),
                                grams=grams)
    return dict(L=L, N=L, g=g, C=C, m=m, k=2 * math.pi / L, dimension=d,
                basis=tuple((L,) + (0,) * (L - 1) for _ in range(d)), H=H,
                energies=np.arange(d, dtype=float), vectors=np.eye(d, dtype=complex),
                ground=ground, gaps=np.arange(d, dtype=float), resolution=1e-12,
                partitions=partitions)


def test_owned_reports_detached_normalization_with_mocked_independent_arrays():
    originals = []
    def system(*args, **kwargs):
        result = synthetic_joint_system(*args, **kwargs)
        originals.append(result)
        return result
    with pytest.MonkeyPatch.context() as patch:
        for target in (occupation.fixed_number_model, occupation._occupations, np.linalg.eigh,
                       np.linalg.eigvalsh, np.linalg.svd):
            patch_aliases(patch, target, deny)
        patch_aliases(patch, joint.joint_system, system)
        first = api._owned_case(3, 0.7, 2.0)
        second = api._owned_case(3, 0.7, 2.0)
    assert len(originals) == 2
    for owned in (first, second):
        assert set(owned) == {"system", "normalized", "report"}
        case_schema(owned["report"], (3, 0.7, 2.0))
        normalized = owned["normalized"]
        np.testing.assert_array_equal(normalized["gaps"], np.arange(1, 10) / 2)
        for name in PARTITIONS:
            p = normalized["partitions"][name]
            np.testing.assert_array_equal(p["transitions"][0], [1, 2])
            np.testing.assert_array_equal(p["total_weight"], [[1, 2], [2, 4]])
    first["report"]["partitions"][0]["records"][0]["schedule"]["theta"] = -1.0
    assert second["report"]["partitions"][0]["records"][0]["schedule"]["theta"] == 4.0
    for name in PARTITIONS:
        assert not np.shares_memory(first["normalized"]["partitions"][name]["transitions"],
                                   second["normalized"]["partitions"][name]["transitions"])


@pytest.mark.parametrize("gap_available", (True, False))
def test_post_authorship_partition_failure_preserves_known_nu_and_synthetic_reuse(gap_available):
    """Reviewed implementation-specific two-state mock, not blind oracle work."""
    joint_attempts, partition_attempts = [], []
    sentinel_data = {name: object() for name in PARTITIONS}
    system = dict(dimension=2, gaps=np.array([0.0, 2.0]),
                  resolution=0.125 if gap_available else 2.0,
                  partitions=sentinel_data)

    def mock_joint(L, g, C=1.0, m=1):
        joint_attempts.append((L, g, C, m))
        assert joint_attempts == [(3, 0.7, 1.0, 1)]
        return system

    def unavailable_partition(data, C):
        assert C == 1.0
        partition_attempts.append(data)
        raise current.NumericalUnavailable("post-authorship partition sentinel")

    with pytest.MonkeyPatch.context() as patch:
        original_joint = joint.joint_system
        alias_names = [name for name, value in vars(api).items() if value is original_joint]
        block_physics(patch)
        patch.setattr(joint, "joint_system", mock_joint)
        for name in alias_names:
            patch.setattr(api, name, mock_joint)
        patch.setattr(api, "_normalize_partition", unavailable_partition)
        patch_aliases(patch, api._time_trace, deny)
        owned = api._owned_case(3, 0.7, 1.0)
        synthetic = api._synthetic_owned(owned)
    assert joint_attempts == [(3, 0.7, 1.0, 1)]
    assert owned["system"] is system
    report = owned["report"]
    assert set(report) == CASE_KEYS and report["dimension"] == 2
    assert report["ground_gap"] == 2.0 and report["status"] == PARTIAL
    expected = [1.0, 2.0, 4.0] * 2 if gap_available else [None] * 6
    if gap_available:
        assert partition_attempts == [sentinel_data[name] for name in PARTITIONS]
        assert owned["normalized"]["ground_gap"] == report["normalized_ground_gap"] == 2.0
        assert owned["normalized"]["resolution"] == 0.125
        assert owned["normalized"]["partitions"] == dict.fromkeys(PARTITIONS)
    else:
        assert not partition_attempts
        assert owned["normalized"] is None and report["normalized_ground_gap"] is None
        assert report["reason"] == "ground_gap_unresolved"
    for name, partition in zip(PARTITIONS, report["partitions"]):
        assert set(partition) == PARTITION_KEYS
        assert partition["partition"] == name
        assert [row["nu"] for row in partition["records"]] == expected
        for row, (theta, n, eta) in zip(partition["records"],
                                      ((theta, n, eta) for theta, n in SCHEDULES for eta in ETAS)):
            acquisition_schema(row, name, eta, theta, n, 0.0, [[1.0] * 2] * 2)
            assert row["status"] == UNAVAILABLE
            assert row["estimate"] is row["finite_reference"] is row["infinite_reference"] is None
    synthetic_schema(synthetic)
    assert [row["nu"] for row in synthetic["records"]] == expected * 2
    assert synthetic["status"] == UNAVAILABLE


def test_contact_nonidentifiability_is_algebra_not_new_time_sample(scalar_acquisitions):
    data = scalar_acquisitions["pauli", 4.0]
    connected = complex_matrix(data["record"]["infinite_reference"])
    trace = data["noiseless_samples"].copy()
    diagonal_contact = np.diag([2.0, -3.0])
    cross_contact = np.array([[0.0, 5.0], [5.0, 0.0]])
    full_diagonal = connected + diagonal_contact
    full_cross = connected + cross_contact
    assert not np.array_equal(full_diagonal, connected)
    assert not np.array_equal(full_cross, connected)
    np.testing.assert_array_equal(trace, data["noiseless_samples"])
    assert SCOPE["contacts"] == "not reconstructed"


CLI_BOOTSTRAP = r'''
import copy
import json
from pathlib import Path
import runpy
import sys
script = Path(sys.argv[1]).resolve()
mode = sys.argv[2:]
assert script.is_absolute() and not list(Path.cwd().iterdir())
sys.path.insert(0, str(script.parents[1]))
import numpy as np
from bpr import substrate_sampled_response as api
from bpr import substrate_joint_source_kernel as joint
from bpr import substrate_fermionization as occupation
from bpr import substrate_current_response as current
modules = (api, joint, occupation, current, np, np.linalg)
def patch(original, replacement):
    count = 0
    for module in modules:
        for name, value in tuple(vars(module).items()):
            if value is original:
                setattr(module, name, replacement)
                count += 1
    assert count

def forbidden(*args, **kwargs):
    raise AssertionError('CLI mock must not execute physical work')
for function in (joint.joint_system, occupation.fixed_number_model, occupation._occupations,
                 np.linalg.eigh, np.linalg.eigvalsh, np.linalg.svd):
    patch(function, forbidden)
original_demo = api.demonstration_report
original_owned = api._owned_case
slots = []
def unavailable(L, g, C):
    slots.append((L, g, C))
    raise current.NumericalUnavailable('CLI mock reference unavailable')
patch(original_owned, unavailable)
# Obtain a complete schema-valid mock via failure slots, never a scientific run.
mock_report = original_demo()
assert slots == [(L, g, 1.0) for L in (3,4,5) for g in (0.0,0.7,40.0)]
assert len(mock_report['physical_cases']) == 9 and len(mock_report['synthetic']['records']) == 12
calls = []
def demonstration():
    calls.append(1)
    assert calls == [1]
    return copy.deepcopy(mock_report)
patch(original_demo, demonstration)
sys.argv = [str(script)] + mode
try:
    runpy.run_path(str(script), run_name='__main__')
except SystemExit as exc:
    assert exc.code in (None, 0)
assert calls == [1]
assert len(slots) == 9
assert not list(Path.cwd().iterdir())
'''


@pytest.mark.parametrize("mode", ("text", "json"))
def test_cli_singleton_mock_stdout_only_empty_cwd_without_physical_grid(tmp_path, mode):
    assert DEMO.is_absolute() and DEMO.is_file()
    cwd = tmp_path / mode
    cwd.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-c", CLI_BOOTSTRAP, str(DEMO)]
    if mode == "json":
        command.append("--json")
    result = subprocess.run(command, cwd=str(cwd), env=env, text=True,
                            capture_output=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == "" and result.stdout.strip()
    assert not list(cwd.iterdir())
    if mode == "json":
        report = json.loads(result.stdout)
        demo_schema(report)
        assert all(case["status"] == UNAVAILABLE for case in report["physical_cases"])
    else:
        assert "response" in result.stdout.lower()
        assert not result.stdout.lstrip().startswith("{")
