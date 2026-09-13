"""Independent exact tests frozen before implementation access or execution.

Oracle source: doc/derivations/chiral_content_constraints_2026-09-12.md.
The author did not read/import the implementation or demo before freezing these
initial tests. Representation traces, a Leibniz determinant, spinor character,
and formal substitution supply expectations, not implementation round trips.
SymPy is a test-only independent exact algebra oracle. No numerical tolerance,
floating rank, representation search, or scientific parameter sweep is used.
"""

import ast
import copy
from fractions import Fraction as F
from functools import lru_cache
from itertools import permutations, product
import json
from math import factorial, gcd
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from bpr import chiral_content_constraints as subject


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "scripts" / "demo_chiral_content_constraints.py"
COLUMNS = ("Q", "uc", "dc", "L", "ec")
ROWS = (
    "SU3_cubic", "SU3_squared_Y", "SU2_squared_Y", "Y_cubic",
    "gravity_squared_Y",
)
DOMAIN = (
    "S4", "S2_squared", "S2_X_squared", "X_fourth", "p1_S2",
    "p1_X_squared", "p1_squared", "p2",
)
CODOMAIN = ("S2_Xb", "Xb_cubed", "p1_Xb")
# color dimension, weak dimension, hypercharge, color cubic, color index,
# weak index. All entries describe left-handed species, not fitted anomalies.
REPRESENTATIONS = (
    (3, 2, F(1, 6), 1, F(1, 2), F(1, 2)),
    (3, 1, F(-2, 3), -1, F(1, 2), F(0)),
    (3, 1, F(1, 3), -1, F(1, 2), F(0)),
    (1, 2, F(-1, 2), 0, F(0), F(1, 2)),
    (1, 1, F(1), 0, F(0), F(0)),
)
LOCAL_CASES = (
    (0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2),
    (3, 3, 3, 3, 3),
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1),
    (32, 1, 0, 0, 0),
)
SINGLET_CASES = (
    (1, 1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1, 2),
    (0, 0, 0, 0, 0, 1),
)


def expected_anomaly_matrix(singlet=False):
    columns = []
    for dc, dw, y, cubic, tc, tw in REPRESENTATIONS:
        columns.append((dw * cubic, dw * tc * y, dc * tw * y,
                        dc * dw * y ** 3, dc * dw * y))
    if singlet:
        columns.append((0,) * 5)
    return tuple(tuple(F(column[row]) for column in columns)
                 for row in range(5))


def unit(size, index):
    return tuple(F(int(i == index)) for i in range(size))


def as_fraction(value):
    return F(int(sp.numer(value)), int(sp.denom(value)))


def as_rows(matrix):
    return tuple(tuple(as_fraction(matrix[i, j]) for j in range(matrix.cols))
                 for i in range(matrix.rows))


def determinant_leibniz(matrix):
    result = F(0)
    for permutation in permutations(range(len(matrix))):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(len(matrix))
                         for j in range(i + 1, len(matrix)))
        term = F((-1) ** inversions)
        for row, column in enumerate(permutation):
            term *= matrix[row][column]
        result += term
    return result


def rational_strings(values):
    return [str(F(value)) for value in values]


def encode(value):
    """Only serialize primitives; do not use production encoding helpers."""
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def assert_json_native(value):
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_json_native(item)
    elif type(value) is list:
        for item in value:
            assert_json_native(item)
    else:
        assert value is None or type(value) in (str, int, bool)


def assert_fraction_vector(vector, length):
    assert type(vector) is tuple
    assert len(vector) == length
    assert all(type(value) is F for value in vector)


def assert_fraction_rows(matrix, rows, columns):
    assert type(matrix) is tuple
    assert len(matrix) == rows
    for row in matrix:
        assert_fraction_vector(row, columns)


def assert_canonical_strings(values):
    assert type(values) is list
    assert all(type(value) is str and str(F(value)) == value for value in values)


@lru_cache(maxsize=1)
def independent_parent():
    """Eight-form coefficient of Ahat(T) sum_w exp(t*(w.F + X))."""
    h = sp.symbols("h0:5")
    x, p1, p2 = sp.symbols("X p1 p2")
    weights = [tuple(sp.Rational(sign, 2) for sign in signs)
               for signs in product((-1, 1), repeat=5)
               if sum(sign < 0 for sign in signs) % 2 == 0]
    assert len(weights) == 16
    eigenvalues = [sum(wi * hi for wi, hi in zip(weight, h))
                   for weight in weights]
    s2 = sum(hi ** 2 for hi in h)
    s4 = sum(hi ** 4 for hi in h)
    assert sp.expand(sum(z ** 2 for z in eigenvalues) - 4 * s2) == 0
    assert sp.expand(sum(z ** 4 for z in eigenvalues)
                     - (3 * s2 ** 2 - 2 * s4)) == 0
    ch2 = sum((z + x) ** 2 / factorial(2) for z in eigenvalues)
    ch4 = sum((z + x) ** 4 / factorial(4) for z in eigenvalues)
    # Ahat_4 = -p1/24; Ahat_8 = (7*p1**2 - 4*p2)/5760.
    polynomial = sp.Poly(sp.expand(ch4 - p1 * ch2 / 24
                                  + 16 * (7 * p1 ** 2 - 4 * p2) / 5760),
                         *h, x, p1, p2)
    second = polynomial.coeff_monomial(h[0] ** 2 * h[1] ** 2) / 2
    coefficients = (
        polynomial.coeff_monomial(h[0] ** 4) - second,
        second,
        polynomial.coeff_monomial(h[0] ** 2 * x ** 2),
        polynomial.coeff_monomial(x ** 4),
        polynomial.coeff_monomial(p1 * h[0] ** 2),
        polynomial.coeff_monomial(p1 * x ** 2),
        polynomial.coeff_monomial(p1 ** 2),
        polynomial.coeff_monomial(p2),
    )
    basis = (s4, s2 ** 2, s2 * x ** 2, x ** 4,
             p1 * s2, p1 * x ** 2, p1 ** 2, p2)
    reconstructed = sum(c * b for c, b in zip(coefficients, basis))
    assert sp.expand(polynomial.as_expr() - reconstructed) == 0
    return tuple(as_fraction(c) for c in coefficients)


def formal_pushforward(coefficients, flux):
    """Integrate by substituting X=Xb+m*y and taking coefficient y^1.

    S2,S4,p1,p2 are pulled back; terms y**2 and above vanish on the sphere.
    This uses symbolic polynomial expansion, not the production sparse map.
    """
    s2, s4, x, b, p1, p2, y = sp.symbols("S2 S4 X Xb p1 p2 y")
    domain = (s4, s2 ** 2, s2 * x ** 2, x ** 4,
              p1 * s2, p1 * x ** 2, p1 ** 2, p2)
    expression = sum(sp.Rational(c.numerator, c.denominator) * monomial
                     for c, monomial in zip(map(F, coefficients), domain))
    expanded = sp.expand(expression.subs(x, b + int(flux) * y))
    integrated = sp.expand(expanded).coeff(y, 1)
    target = (s2 * b, b ** 3, p1 * b)
    polynomial = sp.Poly(integrated, s2, s4, b, p1, p2)
    result = tuple(polynomial.coeff_monomial(monomial) for monomial in target)
    assert sp.expand(integrated - sum(c * m for c, m in zip(result, target))) == 0
    return tuple(as_fraction(c) for c in result)


@lru_cache(maxsize=None)
def expected_reduction(flux):
    columns = [formal_pushforward(unit(8, index), flux) for index in range(8)]
    return tuple(tuple(column[row] for column in columns) for row in range(3))


def check_multiplicity_report(report, multiplicities, singlet=False):
    integers = [int(value) for value in multiplicities]
    expected = expected_anomaly_matrix(singlet)
    anomalies = [sum(c * n for c, n in zip(row, integers)) for row in expected]
    equal = len(set(integers[:5])) == 1
    count = 3 * integers[0] + integers[3]
    assert set(report) == {
        "columns", "multiplicities", "anomalies", "local_anomaly_free",
        "witten_doublet_count", "witten_parity", "witten_anomaly_free",
        "fully_consistent_with_tested_constraints", "charged_multiplicities_equal",
        "family_count", "singlet_count", "scope", "exact_arithmetic",
    }
    assert report["columns"] == list(COLUMNS + (("nc",) if singlet else ()))
    assert report["multiplicities"] == integers
    assert all(type(value) is int for value in report["multiplicities"])
    assert report["anomalies"] == dict(zip(ROWS, rational_strings(anomalies)))
    assert list(report["anomalies"]) == list(ROWS)
    assert report["local_anomaly_free"] is all(value == 0 for value in anomalies)
    assert report["witten_doublet_count"] == count
    assert type(report["witten_doublet_count"]) is int
    assert report["witten_parity"] == count % 2
    assert type(report["witten_parity"]) is int
    assert report["witten_anomaly_free"] is (count % 2 == 0)
    assert report["fully_consistent_with_tested_constraints"] is (
        all(value == 0 for value in anomalies) and count % 2 == 0)
    assert report["charged_multiplicities_equal"] is equal
    assert report["family_count"] == (integers[0] if equal else None)
    assert report["singlet_count"] == (integers[5] if singlet else None)
    for key in ("family_count", "singlet_count"):
        assert report[key] is None or type(report[key]) is int
    assert type(report["scope"]) is str and report["scope"].strip()
    assert report["exact_arithmetic"] is True
    assert_json_native(report)


@pytest.mark.parametrize("singlet", [False, True])
def test_all_anomaly_coefficients_and_exact_rank_witness(singlet):
    system = subject.anomaly_system(include_singlet=singlet)
    expected = expected_anomaly_matrix(singlet)
    size = 6 if singlet else 5
    assert set(system) == {
        "columns", "row_labels", "matrix", "rank", "nullity", "rref",
        "pivot_columns", "kernel_basis", "primitive_charged_generator",
        "rank_minor", "rank_minor_determinant", "row_dependence_coefficients",
        "exact_arithmetic",
    }
    assert system["columns"] == COLUMNS + (("nc",) if singlet else ())
    assert system["row_labels"] == ROWS
    assert_fraction_rows(system["matrix"], 5, size)
    # Every coefficient is necessary: rank 4 alone cannot detect a dropped row.
    assert system["matrix"] == expected
    oracle = sp.Matrix(expected)
    reduced, pivots = oracle.rref()
    assert system["rref"] == as_rows(reduced)
    assert_fraction_rows(system["rref"], 5, size)
    assert system["pivot_columns"] == pivots
    assert type(system["pivot_columns"]) is tuple
    assert all(type(index) is int for index in system["pivot_columns"])
    assert system["rank"] == oracle.rank() == 4
    assert system["nullity"] == size - oracle.rank()
    assert type(system["rank"]) is type(system["nullity"]) is int
    minor_rows, minor_columns = (0, 1, 2, 4), (1, 2, 3, 4)
    minor = tuple(tuple(expected[i][j] for j in minor_columns) for i in minor_rows)
    assert system["rank_minor"] == {
        "rows": minor_rows, "columns": minor_columns, "matrix": minor}
    assert_fraction_rows(system["rank_minor"]["matrix"], 4, 4)
    determinant = determinant_leibniz(minor)
    assert determinant == F(1, 8)
    assert system["rank_minor_determinant"] == determinant
    assert type(system["rank_minor_determinant"]) is F
    relation = (F(2, 9), F(-4), F(-3), F(1))
    assert_fraction_vector(system["row_dependence_coefficients"], 4)
    assert system["row_dependence_coefficients"] == relation
    assert tuple(sum(weight * expected[row][j]
                     for weight, row in zip(relation, minor_rows))
                 for j in range(size)) == expected[3]
    for omitted in range(5):
        assert sp.Matrix([row for i, row in enumerate(expected) if i != omitted]).rank() == 4
    charged = (F(1),) * 5 + ((F(0),) if singlet else ())
    expected_kernel = (charged,) + ((unit(6, 5),) if singlet else ())
    assert system["kernel_basis"] == expected_kernel
    assert type(system["kernel_basis"]) is tuple
    assert system["primitive_charged_generator"] == charged
    assert_fraction_vector(system["primitive_charged_generator"], size)
    for vector in system["kernel_basis"]:
        assert_fraction_vector(vector, size)
        assert oracle * sp.Matrix(vector) == sp.zeros(5, 1)
        divisor = 0
        for coordinate in vector:
            assert coordinate.denominator == 1
            divisor = gcd(divisor, coordinate.numerator)
        assert divisor == 1
    assert sp.Matrix.hstack(*oracle.nullspace()).columnspace() == list(
        sp.Matrix(expected_kernel).T.columnspace())
    assert system["exact_arithmetic"] is True


@pytest.mark.parametrize("values", LOCAL_CASES)
def test_all_fixed_charged_controls(values):
    check_multiplicity_report(subject.multiplicity_report(values), values)


@pytest.mark.parametrize("values", SINGLET_CASES)
def test_all_fixed_neutral_controls(values):
    check_multiplicity_report(subject.multiplicity_report(values, include_singlet=True),
                              values, True)


def test_cubic_blind_control_does_not_stand_in_for_full_local_equations():
    report = subject.multiplicity_report((32, 1, 0, 0, 0))
    assert report["anomalies"]["Y_cubic"] == "0"
    assert all(report["anomalies"][row] != "0" for row in ROWS if row != "Y_cubic")
    assert report["local_anomaly_free"] is False
    assert report["witten_anomaly_free"] is True
    assert report["fully_consistent_with_tested_constraints"] is False


def test_rational_kernel_is_not_physical_multiplicity_or_witten_domain():
    matrix = sp.Matrix(expected_anomaly_matrix())
    assert matrix * sp.Matrix([sp.Rational(1, 2)] * 5) == sp.zeros(5, 1)
    assert matrix * sp.Matrix([-1] * 5) == sp.zeros(5, 1)
    for nonphysical in ((F(1, 2),) * 5, (-1,) * 5):
        with pytest.raises(ValueError):
            subject.multiplicity_report(nonphysical)
    # Witten is a separate modulo-2 test, not an extra rational matrix row.
    for values in ((2, 0, 0, 0, 0), (1, 0, 0, 1, 0)):
        report = subject.multiplicity_report(values)
        check_multiplicity_report(report, values)
        assert report["witten_anomaly_free"] is True
        assert report["local_anomaly_free"] is False
    for index in (0, 3):
        values = [int(value) for value in unit(5, index)]
        assert subject.multiplicity_report(values)["witten_anomaly_free"] is False


@pytest.mark.parametrize("flux", [0, 1, -1, 16, -16])
def test_reduction_from_formal_substitution_rational_and_integer_images(flux):
    system = subject.reduction_system(flux)
    expected = expected_reduction(flux)
    oracle = sp.Matrix(expected)
    reduced, pivots = oracle.rref()
    assert set(system) == {
        "flux", "domain_basis", "codomain_basis", "matrix", "rank", "nullity",
        "rref", "pivot_columns", "kernel_basis", "image_basis",
        "integer_image_steps", "integer_image_index", "scope", "exact_arithmetic",
    }
    assert system["flux"] == flux and type(system["flux"]) is int
    assert system["domain_basis"] == DOMAIN
    assert system["codomain_basis"] == CODOMAIN
    assert_fraction_rows(system["matrix"], 3, 8)
    assert system["matrix"] == expected
    assert system["rref"] == as_rows(reduced)
    assert_fraction_rows(system["rref"], 3, 8)
    assert system["pivot_columns"] == pivots
    assert type(system["pivot_columns"]) is tuple
    assert all(type(index) is int for index in system["pivot_columns"])
    assert system["rank"] == oracle.rank() == (3 if flux else 0)
    assert system["nullity"] == 8 - oracle.rank()
    assert type(system["rank"]) is type(system["nullity"]) is int
    expected_kernel = tuple(tuple(as_fraction(c) for c in v) for v in oracle.nullspace())
    assert system["kernel_basis"] == expected_kernel
    assert type(system["kernel_basis"]) is tuple
    assert expected_kernel == tuple(unit(8, i) for i in ((0, 1, 4, 6, 7) if flux else range(8)))
    for vector in system["kernel_basis"]:
        assert_fraction_vector(vector, 8)
    image = []
    for column in oracle.columnspace():
        first = next(value for value in column if value)
        image.append(tuple(as_fraction(value / first) for value in column))
    assert system["image_basis"] == tuple(image)
    assert type(system["image_basis"]) is tuple
    for vector in system["image_basis"]:
        assert_fraction_vector(vector, 3)
    # A gcd in each independent target coordinate determines this diagonal lattice.
    steps = []
    for row in expected:
        divisor = 0
        for value in row:
            assert value.denominator == 1
            divisor = gcd(divisor, abs(value.numerator))
        steps.append(divisor)
    assert system["integer_image_steps"] == tuple(steps)
    assert type(system["integer_image_steps"]) is tuple
    assert all(type(value) is int for value in system["integer_image_steps"])
    index = steps[0] * steps[1] * steps[2] if flux else None
    assert system["integer_image_index"] == index
    assert index == (16 * abs(flux) ** 3 if flux else None)
    assert system["integer_image_index"] is None or type(system["integer_image_index"]) is int
    if flux:
        assert index > 1  # Full rational image does not imply all of Z**3.
    assert type(system["scope"]) is str and system["scope"].strip()
    assert system["exact_arithmetic"] is True


@pytest.mark.parametrize("flux", [0, 1, -1])
def test_each_domain_direction_and_nontrivial_rational_combination(flux):
    for index in range(8):
        coefficients = unit(8, index)
        actual = subject.pushforward(coefficients, flux)
        assert_fraction_vector(actual, 3)
        assert actual == formal_pushforward(coefficients, flux)
    coefficients = (F(-2, 7), F(3, 11), F(-5, 13), F(7, 17),
                    F(-11, 19), F(13, 23), F(-17, 29), F(19, 31))
    actual = subject.pushforward(coefficients, flux)
    assert_fraction_vector(actual, 3)
    assert actual == formal_pushforward(coefficients, flux)
    invisible = tuple(c if i in (0, 1, 4, 6, 7) else F(0)
                      for i, c in enumerate(coefficients))
    assert subject.pushforward(invisible, flux) == (F(0),) * 3
    assert subject.pushforward(unit(8, 7), flux) == (F(0),) * 3
    if flux:
        assert subject.pushforward(coefficients, -flux) == tuple(-c for c in actual)


@pytest.mark.parametrize("flux", [0, 1, -1, 16, -16])
def test_parent_from_sixteen_spinor_weights_and_ahat_character(flux):
    parent = independent_parent()
    assert parent == (F(-1, 12), F(1, 8), F(1), F(2, 3),
                      F(-1, 12), F(-1, 3), F(7, 360), F(-1, 90))
    reduced = formal_pushforward(parent, flux)
    visible = tuple(c if i in (2, 3, 5) else F(0) for i, c in enumerate(parent))
    invisible = tuple(c - v for c, v in zip(parent, visible))
    report = subject.parent_report(flux)
    assert set(report) == {
        "flux", "charge", "chirality", "domain_basis", "codomain_basis",
        "parent_coefficients", "reduced_coefficients", "visible_coefficients",
        "invisible_coefficients", "invisible_pushforward", "parent_nonzero",
        "reduced_nonzero", "p2_coefficient", "split_convention",
        "restricted_reduction_certifies_parent", "exact_arithmetic",
    }
    assert report["flux"] == flux
    assert report["charge"] == report["chirality"] == 1
    assert all(type(report[key]) is int for key in ("flux", "charge", "chirality"))
    assert report["domain_basis"] == list(DOMAIN)
    assert report["codomain_basis"] == list(CODOMAIN)
    for key, values in (
        ("parent_coefficients", parent), ("reduced_coefficients", reduced),
        ("visible_coefficients", visible), ("invisible_coefficients", invisible),
        ("invisible_pushforward", formal_pushforward(invisible, flux)),
    ):
        assert report[key] == rational_strings(values)
        assert_canonical_strings(report[key])
    assert report["parent_nonzero"] is any(parent)
    assert report["reduced_nonzero"] is any(reduced)
    assert report["p2_coefficient"] == str(parent[7]) == "-1/90"
    assert report["invisible_pushforward"] == ["0"] * 3
    assert report["split_convention"] == "nonzero_flux_coordinate_split"
    assert report["restricted_reduction_certifies_parent"] is False
    assert report["exact_arithmetic"] is True
    assert formal_pushforward(visible, flux) == reduced
    if flux == 0:
        assert any(visible)  # Fixed split, NOT complement of zero-flux kernel.
        assert report["reduced_nonzero"] is False
        assert report["parent_nonzero"] is True
    assert_json_native(report)


class CoercionTrap:
    def __int__(self):
        raise AssertionError("custom integer conversion must never be attempted")

    def __index__(self):
        raise AssertionError("custom index conversion must never be attempted")

    def __float__(self):
        raise AssertionError("custom float conversion must never be attempted")


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


BAD_SCALARS = (
    True, False, np.bool_(True), np.bool_(False), 1.0, np.float64(1),
    float("nan"), float("inf"), complex(1, 0), "1", None, CoercionTrap(),
)
BAD_CONTAINERS = (
    "00000", {0: 0}, {0}, range(5), np.zeros(5, dtype=int), CoercionTrap(), None,
)


def test_declared_computational_caps():
    assert type(subject.MAX_MULTIPLICITY) is int
    assert type(subject.MAX_COEFFICIENT_COMPONENT) is int
    assert type(subject.MAX_ABS_FLUX) is int
    assert subject.MAX_MULTIPLICITY == 1_000_000
    assert subject.MAX_COEFFICIENT_COMPONENT == 1_000_000
    assert subject.MAX_ABS_FLUX == 16


@pytest.mark.parametrize("flag", [0, 1, np.int64(1), np.bool_(True), None, "False", 0.0, CoercionTrap()])
def test_include_singlet_requires_builtin_bool(flag):
    with pytest.raises(ValueError):
        subject.anomaly_system(include_singlet=flag)
    with pytest.raises(ValueError):
        subject.multiplicity_report([0] * 5, include_singlet=flag)


@pytest.mark.parametrize("bad", BAD_SCALARS + (F(1), -17, 17, 10 ** 100, -(10 ** 100), np.uint64(2 ** 64 - 1)))
def test_flux_validation_on_every_flux_api(bad):
    for function, args in (
        (subject.reduction_system, (bad,)),
        (subject.parent_report, (bad,)),
        (subject.pushforward, ((0,) * 8, bad)),
    ):
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize("bad", BAD_SCALARS + (F(1), F(1, 2), -1, 1_000_001, 10 ** 100, np.uint64(2 ** 64 - 1)))
@pytest.mark.parametrize("singlet", [False, True])
def test_multiplicity_rejects_invalid_values_in_every_position(bad, singlet):
    size = 6 if singlet else 5
    for index in range(size):
        values = [0] * size
        values[index] = bad
        with pytest.raises(ValueError):
            subject.multiplicity_report(values, include_singlet=singlet)


@pytest.mark.parametrize("singlet", [False, True])
def test_multiplicity_length_container_and_no_coercion_preflight(singlet):
    size = 6 if singlet else 5
    bad_vectors = list(BAD_CONTAINERS) + [iter([0] * size), (0 for _ in range(size))]
    bad_vectors += [ListSubclass([0] * size), TupleSubclass([0] * size),
                    [], [0] * (size - 1), [0] * (size + 1),
                    [CoercionTrap()] * (size + 1), [CoercionTrap()] * (size - 1)]
    for values in bad_vectors:
        with pytest.raises(ValueError):
            subject.multiplicity_report(values, include_singlet=singlet)


@pytest.mark.parametrize("bad", BAD_SCALARS + (
    1_000_001, -1_000_001, F(1, 1_000_001), F(-1, 1_000_001),
    F(1_000_001, 2), F(-1_000_001, 2), 10 ** 100, np.uint64(2 ** 64 - 1),
    sp.Rational(1, 2), sp.Integer(1),
))
@pytest.mark.parametrize("flux", [0, 1])
def test_coefficient_validation_including_zero_flux_every_position(bad, flux):
    for index in range(8):
        values = [0] * 8
        values[index] = bad
        with pytest.raises(ValueError):
            subject.pushforward(values, flux)


@pytest.mark.parametrize("flux", [0, 1])
def test_pushforward_length_container_and_no_coercion_preflight(flux):
    bad_vectors = list(BAD_CONTAINERS) + [ListSubclass([0] * 8), TupleSubclass([0] * 8),
                                        np.zeros(8, dtype=int), iter([0] * 8),
                                        (0 for _ in range(8)), [], [0] * 7, [0] * 9,
                                        [CoercionTrap()] * 7, [CoercionTrap()] * 9]
    for values in bad_vectors:
        with pytest.raises(ValueError):
            subject.pushforward(values, flux)


@pytest.mark.parametrize("integer_type", [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64])
def test_builtin_and_numpy_integer_scalars_are_safely_accepted(integer_type):
    values = [integer_type(1)] * 5
    check_multiplicity_report(subject.multiplicity_report(values), values)
    coefficients = [integer_type(1)] * 8
    flux = integer_type(1)
    assert subject.pushforward(coefficients, flux) == formal_pushforward(coefficients, 1)
    assert subject.reduction_system(flux)["flux"] == 1
    assert type(subject.reduction_system(flux)["flux"]) is int
    assert subject.parent_report(flux)["flux"] == 1
    assert type(subject.parent_report(flux)["flux"]) is int


@pytest.mark.parametrize("container", [list, tuple])
def test_multiplicity_cap_and_unconstrained_singlet_boundaries(container):
    for singlet, values in (
        (False, (1_000_000,) * 5),
        (True, (0, 0, 0, 0, 0, 1_000_000)),
        (True, (1_000_000,) * 5 + (0,)),
        (True, (1_000_000,) * 6),
    ):
        report = subject.multiplicity_report(container(values), include_singlet=singlet)
        check_multiplicity_report(report, values, singlet)
        assert report["fully_consistent_with_tested_constraints"] is True


@pytest.mark.parametrize("flux", [0, 16, -16])
@pytest.mark.parametrize("container", [list, tuple])
def test_coefficient_canonical_fraction_bounds_and_output_not_input_capped(flux, container):
    values = (F(1, 1_000_000), -1_000_000, F(1_000_000, 999_983),
              1_000_000, F(-1, 1_000_000), F(-1_000_000, 999_983),
              F(1_000_001, 1_000_001), np.int64(1_000_000))
    actual = subject.pushforward(container(values), flux)
    assert_fraction_vector(actual, 3)
    assert actual == formal_pushforward(values, flux)
    if flux:
        assert abs(actual[1]) > 1_000_000
    # Fractions are validated after canonical reduction, not on construction history.
    canonical = [F(2_000_000, 2_000_000)] * 8
    assert subject.pushforward(canonical, flux) == formal_pushforward(canonical, flux)


@pytest.mark.parametrize("boundary", [
    1_000_000, -1_000_000, F(1, 1_000_000), F(-1, 1_000_000),
    F(1_000_000, 999_983), F(-1_000_000, 999_983),
])
def test_each_coefficient_coordinate_accepts_inclusive_component_caps(boundary):
    for index in range(8):
        values = [F(0)] * 8
        values[index] = boundary
        actual = subject.pushforward(values, -16)
        assert_fraction_vector(actual, 3)
        assert actual == formal_pushforward(values, -16)


def test_system_and_report_objects_are_detached_and_inputs_unchanged():
    first = subject.anomaly_system(True)
    pristine = copy.deepcopy(first)
    first["rank_minor"]["matrix"] = ()
    first["rank"] = -1
    assert subject.anomaly_system(True) == pristine
    reduction = subject.reduction_system(1)
    pristine_reduction = copy.deepcopy(reduction)
    reduction["matrix"] = ()
    assert subject.reduction_system(1) == pristine_reduction
    values = [1, 1, 1, 1, 1, 2]
    report = subject.multiplicity_report(values, True)
    saved = copy.deepcopy(report)
    assert values == [1, 1, 1, 1, 1, 2]
    values[0] = 99
    assert report == saved
    report["multiplicities"][0] = 100
    report["anomalies"][ROWS[0]] = "999"
    report["columns"][0] = "not-Q"
    assert subject.multiplicity_report([1, 1, 1, 1, 1, 2], True) == saved
    parent = subject.parent_report(0)
    saved_parent = copy.deepcopy(parent)
    parent["visible_coefficients"][2] = "999"
    parent["parent_coefficients"][0] = "999"
    assert subject.parent_report(0) == saved_parent
    coefficients = [F(1, 7)] * 8
    before = list(coefficients)
    subject.pushforward(coefficients, 1)
    assert coefficients == before


def test_complete_fixed_demo_grid_and_exact_recursive_encoding():
    report = subject.demonstration_report()
    assert set(report) == {
        "anomaly_systems", "multiplicity_cases", "reduction_cases", "parent_cases",
        "limitations", "arithmetic_domain",
    }
    assert set(report["anomaly_systems"]) == {"without_singlet", "with_singlet"}
    for name, singlet in (("without_singlet", False), ("with_singlet", True)):
        assert report["anomaly_systems"][name] == encode(subject.anomaly_system(singlet))
        assert report["anomaly_systems"][name]["matrix"] == [
            rational_strings(row) for row in expected_anomaly_matrix(singlet)]
    assert len(report["multiplicity_cases"]) == 13
    expected_cases = [(values, False) for values in LOCAL_CASES]
    expected_cases += [(values, True) for values in SINGLET_CASES]
    for actual, (values, singlet) in zip(report["multiplicity_cases"], expected_cases):
        check_multiplicity_report(actual, values, singlet)
    assert len(report["reduction_cases"]) == len(report["parent_cases"]) == 3
    for actual, parent, flux in zip(report["reduction_cases"], report["parent_cases"], (0, 1, -1)):
        assert {"system", "basis_images", "kernel_images"} == set(actual)
        assert actual["system"] == encode(subject.reduction_system(flux))
        assert actual["system"]["matrix"] == [rational_strings(row) for row in expected_reduction(flux)]
        assert actual["basis_images"] == [
            rational_strings(formal_pushforward(unit(8, i), flux)) for i in range(8)]
        indices = (0, 1, 4, 6, 7) if flux else range(8)
        assert actual["kernel_images"] == [
            rational_strings(formal_pushforward(unit(8, i), flux)) for i in indices]
        assert parent == subject.parent_report(flux)
        assert parent["parent_coefficients"] == rational_strings(independent_parent())
        assert parent["reduced_coefficients"] == rational_strings(formal_pushforward(independent_parent(), flux))
    assert type(report["limitations"]) is list and report["limitations"]
    assert all(type(item) is str and item.strip() for item in report["limitations"])
    assert report["arithmetic_domain"] == {
        "kind": "exact_rational", "max_multiplicity": 1_000_000,
        "max_coefficient_component": 1_000_000, "max_abs_flux": 16,
        "rational_encoding": "canonical_fraction_string",
    }
    assert_json_native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    saved = copy.deepcopy(report)
    report["multiplicity_cases"][0]["anomalies"][ROWS[0]] = "999"
    report["anomaly_systems"]["without_singlet"]["matrix"][0][0] = "999"
    assert subject.demonstration_report() == saved


@pytest.mark.parametrize("json_mode", [False, True])
def test_demo_stdout_only_from_empty_directory(tmp_path, json_mode):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-W", "error", str(DEMO)]
    if json_mode:
        command.append("--json")
    result = subprocess.run(command, cwd=str(tmp_path), env=environment,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip()
    assert list(tmp_path.iterdir()) == []
    if json_mode:
        def reject_nonfinite(value):
            raise AssertionError("nonfinite JSON constant: " + value)
        parsed = json.loads(result.stdout, parse_constant=reject_nonfinite)
        assert_json_native(parsed)
        assert parsed == subject.demonstration_report()
    else:
        assert "anomal" in result.stdout.lower()


def test_new_source_demo_and_tests_parse_as_python38():
    for path in (Path(subject.__file__), DEMO, Path(__file__)):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path), feature_version=(3, 8))
