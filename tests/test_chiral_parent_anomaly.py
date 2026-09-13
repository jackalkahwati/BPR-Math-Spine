"""Independent exact product/weight oracles for the supplied chiral parent.

These tests do not use production trace coefficients to build expected answers.
Polynomial dictionaries use half the differential-form degree as their grade.
"""

import re
import json
import math
import os
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import subprocess
import sys

import pytest
import sympy as sp

from bpr import chiral_parent_anomaly as anomaly


# Five independent Cartan variables, an external U(1) root, p1 and p2.
_ORACLE_SYMBOLS = sp.symbols("a b c d e u t1 t2")
_GRADES = (1, 1, 1, 1, 1, 1, 2, 4)
_ZERO = (0,) * 8


def _add(*polynomials):
    answer = {}
    for polynomial in polynomials:
        for powers, coefficient in polynomial.items():
            answer[powers] = answer.get(powers, Fraction(0)) + coefficient
    return {powers: coefficient for powers, coefficient in answer.items() if coefficient}


def _scale(polynomial, factor):
    return {powers: coefficient * factor for powers, coefficient in polynomial.items()
            if coefficient * factor}


def _multiply(left, right, maximum=4):
    answer = {}
    for left_powers, left_coefficient in left.items():
        for right_powers, right_coefficient in right.items():
            powers = tuple(a + b for a, b in zip(left_powers, right_powers))
            if sum(a * b for a, b in zip(powers, _GRADES)) > maximum:
                continue
            answer[powers] = (answer.get(powers, Fraction(0))
                              + left_coefficient * right_coefficient)
    return {powers: coefficient for powers, coefficient in answer.items() if coefficient}


def _generator(index, coefficient=Fraction(1)):
    powers = list(_ZERO)
    powers[index] = 1
    return {tuple(powers): Fraction(coefficient)}


def _exp_series(linear, maximum=4):
    # Repeated polynomial multiplication, rather than a character coefficient table.
    answer = {_ZERO: Fraction(1)}
    power = {_ZERO: Fraction(1)}
    for degree in range(1, maximum + 1):
        power = _multiply(power, linear, maximum)
        answer = _add(answer, _scale(power, Fraction(1, math.factorial(degree))))
    return answer


def _weights(parity):
    # Construct by bit masks, independently of the production enumerator.
    result = []
    for mask in range(32):
        signs = tuple(-1 if mask & (1 << index) else 1 for index in range(5))
        if (-1) ** bin(mask).count("1") == parity:
            result.append(tuple(Fraction(sign, 2) for sign in signs))
    return tuple(result)


@lru_cache(maxsize=None)
def _character(parity=1):
    result = {}
    for weight in _weights(parity):
        root = _add(*(_generator(index, value) for index, value in enumerate(weight)))
        result = _add(result, _exp_series(root))
    return result


@lru_cache(maxsize=None)
def _product_oracle(charge, degree, parity=1):
    tangent = _add(
        {_ZERO: Fraction(1)},
        _generator(6, Fraction(-1, 24)),
        _scale(_multiply(_generator(6), _generator(6)), Fraction(7, 5760)),
        _generator(7, Fraction(-4, 5760)),
    )
    product = _multiply(_multiply(tangent, _character(parity)),
                        _exp_series(_generator(5, charge)))
    return {powers: coefficient for powers, coefficient in product.items()
            if sum(a * b for a, b in zip(powers, _GRADES)) == degree}


def _expression(polynomial, symbols=_ORACLE_SYMBOLS):
    return sp.Add(*(sp.Rational(coefficient.numerator, coefficient.denominator)
                    * sp.Mul(*(symbol ** power for symbol, power in zip(symbols, powers)))
                    for powers, coefficient in polynomial.items()))


def _cartan_form(expression, charge=None):
    a, b, c, d, e, u, t1, t2 = _ORACLE_SYMBOLS
    substitutions = {
        anomaly.S2: sum(root ** 2 for root in (a, b, c, d, e)),
        anomaly.S4: sum(root ** 4 for root in (a, b, c, d, e)),
        anomaly.X: u,
        anomaly.X4: u,
        anomaly.p1: t1,
        anomaly.p2: t2,
    }
    if charge is not None:
        substitutions[anomaly.Q] = sp.Integer(charge)
    return sp.expand(expression.subs(substitutions, simultaneous=True))


def _sphere_oracle(charge, flux):
    # Explicit X = u + flux*y, retaining y^0 and y^1 because y^2 = 0.
    # This starts with the degree-eight product, not q times the lower answer.
    y = sp.Symbol("internal_y")
    dual_polynomial = {}
    for powers, coefficient in _product_oracle(charge, 4).items():
        external_power = powers[5]
        for internal_power in range(min(external_power, 1) + 1):
            new_powers = list(powers)
            new_powers[5] -= internal_power
            key = tuple(new_powers) + (internal_power,)
            value = coefficient * math.comb(external_power, internal_power) * flux ** internal_power
            dual_polynomial[key] = dual_polynomial.get(key, Fraction(0)) + value
    expanded = _expression(dual_polynomial, _ORACLE_SYMBOLS + (y,))
    assert sp.Poly(expanded, y).degree() <= 1
    return sp.expand(expanded).coeff(y, 1)


@pytest.mark.parametrize("parity", [1, -1])
def test_weights_are_the_sixteen_exact_half_integral_weights(parity):
    actual = anomaly.spinor_weights(parity)
    expected = _weights(parity)
    assert isinstance(actual, tuple)
    assert len(actual) == len(set(actual)) == 16
    assert all(isinstance(weight, tuple) and len(weight) == 5 for weight in actual)
    assert set(actual) == {tuple(sp.Rational(value.numerator, value.denominator)
                                for value in weight) for weight in expected}
    assert all(value in (sp.Rational(-1, 2), sp.Rational(1, 2))
               for weight in actual for value in weight)


@pytest.mark.parametrize("parity", [1, -1])
@pytest.mark.parametrize("power", range(5))
def test_traces_against_direct_weight_powers(parity, power):
    expected = sp.Add(*(sum(sp.Rational(value.numerator, value.denominator) * root
                            for value, root in zip(weight, anomaly.CARTAN_SYMBOLS)) ** power
                        for weight in _weights(parity)))
    actual = anomaly.trace_polynomial(power, parity)
    assert sp.expand(actual - expected) == 0
    assert not actual.atoms(sp.Float)


def test_conjugate_spinor_parity_does_not_reverse_spacetime_chirality():
    for power in range(5):
        assert anomaly.trace_polynomial(power, 1) == anomaly.trace_polynomial(power, -1)
    assert _character(1) == _character(-1)
    assert anomaly.parent_polynomial(1) != anomaly.parent_polynomial(-1)


@pytest.mark.parametrize("chirality", [1, -1])
@pytest.mark.parametrize("charge", [0, 1, -1, 3])
def test_parent_and_lower_against_independent_truncated_product(chirality, charge):
    assert _cartan_form(anomaly.parent_polynomial(chirality), charge) == (
        chirality * _expression(_product_oracle(charge, 4)))
    assert _cartan_form(anomaly.lower_polynomial(chirality), charge) == (
        chirality * _expression(_product_oracle(charge, 3)))


@pytest.mark.parametrize("charge,flux", [(1, 0), (1, 1), (1, -1), (1, 3),
                                         (3, 1), (0, 1), (-1, -3), (-3, 1)])
@pytest.mark.parametrize("chirality", [1, -1])
def test_sphere_pushforward_against_explicit_internal_generator(charge, flux, chirality):
    expected = chirality * _sphere_oracle(charge, flux)
    actual = _cartan_form(anomaly.sphere_pushforward(charge, flux, chirality))
    lower = chirality * charge * flux * _expression(_product_oracle(charge, 3))
    assert actual == expected
    assert sp.expand(expected - lower) == 0


def test_sphere_tangent_pontryagin_has_no_vertical_two_form():
    y, tangent_root = sp.symbols("internal_y tangent_root")
    parent = _expression(_product_oracle(3, 4))
    external = _ORACLE_SYMBOLS[5]
    t1, t2 = _ORACLE_SYMBOLS[6:]
    # Whitney product for T4 plus a rank-two tangent root tangent_root*y.
    product_tangent = parent.subs({
        external: external + y,
        t1: t1 + tangent_root ** 2 * y ** 2,
        t2: t2 + t1 * tangent_root ** 2 * y ** 2,
    }, simultaneous=True)
    unchanged_tangent = parent.subs(external, external + y)
    assert sp.expand(product_tangent - unchanged_tangent).coeff(y, 1) == 0
    assert sp.expand(product_tangent).coeff(y, 1) == _sphere_oracle(3, 1)


def test_equal_signed_index_retains_charge_dependent_mixed_terms():
    first = _cartan_form(anomaly.sphere_pushforward(1, 3))
    second = _cartan_form(anomaly.sphere_pushforward(3, 1))
    assert first != second
    assert sp.expand(first - second) == sp.expand(_sphere_oracle(1, 3) - _sphere_oracle(3, 1))
    # Equal q does not absorb Q in external gauge or gravity mixed terms.
    u, t1, _ = _ORACLE_SYMBOLS[5:]
    a = _ORACLE_SYMBOLS[0]
    assert first.coeff(a ** 2 * u) == 6
    assert second.coeff(a ** 2 * u) == 18
    assert first.coeff(u, 3) == 8
    assert second.coeff(u, 3) == 216
    assert first.coeff(t1 * u) == -2
    assert second.coeff(t1 * u) == -6


def test_full_gauge_conjugation_preserves_parent_not_chirality_sign():
    for chirality in (1, -1):
        parent = anomaly.parent_polynomial(chirality)
        assert sp.expand(parent.subs(anomaly.Q, -anomaly.Q) - parent) == 0
        assert _character(-1) == _character(1)
        assert parent != 0
        assert anomaly.sphere_pushforward(3, 1, chirality).subs(anomaly.X4, 0) == 0
        assert parent.subs(anomaly.X, 0) != 0


def test_character_hyperbolic_product_and_first_parity_difference():
    # This private oracle may inspect degree five without widening the public cap.
    z = sp.Symbol("character_grade")
    roots = (sp.Integer(1), sp.Integer(2), sp.Integer(3), sp.Integer(4), sp.Integer(5))
    for parity in (1, -1):
        weight_sum = sum(sum(sp.Rational(value.numerator, value.denominator) * root
                             for value, root in zip(weight, roots)) ** 5
                         / sp.factorial(5) for weight in _weights(parity))
        sinh_product = 16 * parity * sp.prod(root / 2 for root in roots)
        assert weight_sum == sinh_product
        cosine_product = sp.Integer(16)
        for root in roots:
            cosine_product *= 1 + (root * z / 2) ** 2 / 2 + (root * z / 2) ** 4 / 24
        direct = sum(sum((z * sum(sp.Rational(value.numerator, value.denominator) * root
                                   for value, root in zip(weight, roots))) ** degree
                         / sp.factorial(degree) for degree in range(5))
                     for weight in _weights(parity))
        for degree in range(5):
            assert sp.expand(cosine_product - direct).coeff(z, degree) == 0


def test_vectorlike_pair_and_signed_flux_controls():
    assert sp.expand(anomaly.parent_polynomial(1) + anomaly.parent_polynomial(-1)) == 0
    assert sp.expand(anomaly.lower_polynomial(1) + anomaly.lower_polynomial(-1)) == 0
    for charge, flux in [(1, 1), (1, -1), (1, 3), (3, 1), (0, 1)]:
        positive = anomaly.sphere_pushforward(charge, flux, 1)
        assert sp.expand(positive + anomaly.sphere_pushforward(charge, flux, -1)) == 0
        assert sp.expand(positive + anomaly.sphere_pushforward(charge, -flux, 1)) == 0


@pytest.mark.parametrize("chirality", [1, -1])
def test_irreducible_gravity_and_neutral_restrictions(chirality):
    parent = anomaly.parent_polynomial(chirality)
    assert sp.expand(parent).coeff(anomaly.p2) == -sp.Rational(chirality * 16, 1440)
    gravity = parent.subs({anomaly.S2: 0, anomaly.S4: 0, anomaly.X: 0})
    assert sp.expand(gravity - chirality * 16 * (
        7 * anomaly.p1 ** 2 - 4 * anomaly.p2) / 5760) == 0
    neutral = parent.subs(anomaly.Q, 0)
    assert not neutral.has(anomaly.X)
    assert neutral != 0
    assert anomaly.sphere_pushforward(0, 1, chirality) == 0
    assert anomaly.sphere_pushforward(1, 0, chirality) == 0


@pytest.mark.parametrize("expression,degree", [("parent", 8), ("lower", 6), ("sphere", 6)])
def test_exact_rational_coefficients_and_form_homogeneity(expression, degree):
    if expression == "parent":
        polynomial = anomaly.parent_polynomial()
    elif expression == "lower":
        polynomial = anomaly.lower_polynomial()
    else:
        polynomial = anomaly.sphere_pushforward(3, 1)
    grades = {anomaly.S2: 4, anomaly.S4: 8, anomaly.X: 2, anomaly.X4: 2,
              anomaly.p1: 4, anomaly.p2: 8, anomaly.Q: 0}
    generators = tuple(grades)
    assert not polynomial.atoms(sp.Float)
    assert polynomial.free_symbols <= set(generators)
    for powers, coefficient in sp.Poly(polynomial, *generators).terms():
        assert coefficient.is_Rational
        assert sum(power * grades[generator] for power, generator in zip(powers, generators)) == degree


def _forbid_expansion(*args, **kwargs):
    raise AssertionError("invalid public input reached symbolic expansion")


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", None, sp.Rational(1, 2),
                                 sp.Symbol("untrusted"), sp.Symbol("untrusted") + 1])
def test_invalid_scalar_domains_rejected_before_expansion(monkeypatch, bad):
    monkeypatch.setattr(anomaly.sp, "expand", _forbid_expansion)
    calls = [lambda: anomaly.spinor_weights(bad),
             lambda: anomaly.trace_polynomial(bad),
             lambda: anomaly.trace_polynomial(2, bad),
             lambda: anomaly.parent_polynomial(bad),
             lambda: anomaly.lower_polynomial(bad),
             lambda: anomaly.sphere_pushforward(bad, 1),
             lambda: anomaly.sphere_pushforward(1, bad),
             lambda: anomaly.sphere_pushforward(1, 1, bad),
             lambda: anomaly.case_report(bad, 1),
             lambda: anomaly.case_report(1, bad),
             lambda: anomaly.case_report(1, 1, bad)]
    for call in calls:
        with pytest.raises(TypeError):
            call()


@pytest.mark.parametrize("bad", [0, 2, -2, 10 ** 100])
def test_invalid_signs_rejected_before_expansion(monkeypatch, bad):
    monkeypatch.setattr(anomaly.sp, "expand", _forbid_expansion)
    calls = [lambda: anomaly.spinor_weights(bad),
             lambda: anomaly.trace_polynomial(2, bad),
             lambda: anomaly.parent_polynomial(bad),
             lambda: anomaly.lower_polynomial(bad),
             lambda: anomaly.sphere_pushforward(1, 1, bad),
             lambda: anomaly.case_report(1, 1, bad)]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [-1, 5, 10 ** 100])
def test_trace_resource_bound_precedes_expansion(monkeypatch, bad):
    monkeypatch.setattr(anomaly.sp, "expand", _forbid_expansion)
    with pytest.raises(ValueError):
        anomaly.trace_polynomial(bad)


@pytest.mark.parametrize("bad", [10 ** 6 + 1, -(10 ** 6 + 1), 10 ** 100])
def test_charge_and_flux_caps_precede_expansion(monkeypatch, bad):
    assert anomaly.MAX_ABS_INTEGER == 10 ** 6
    monkeypatch.setattr(anomaly.sp, "expand", _forbid_expansion)
    for call in [lambda: anomaly.sphere_pushforward(bad, 1),
                 lambda: anomaly.sphere_pushforward(1, bad),
                 lambda: anomaly.case_report(bad, 1),
                 lambda: anomaly.case_report(1, bad)]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("charge,flux", [(10 ** 6, 10 ** 6), (-10 ** 6, 10 ** 6),
                                         (10 ** 6, -10 ** 6)])
def test_endpoint_inputs_allow_large_index_without_mode_allocation(charge, flux):
    expected = _sphere_oracle(charge, flux)
    assert _cartan_form(anomaly.sphere_pushforward(charge, flux)) == expected


_CONTRIBUTIONS = {
    "pure_spin10", "mixed_spin10_u1", "pure_u1", "mixed_gravity_spin10",
    "mixed_gravity_u1", "pure_gravity",
}


def _decode_rational(record):
    assert set(record) == {"numerator", "denominator"}
    numerator, denominator = record["numerator"], record["denominator"]
    assert isinstance(numerator, str) and re.fullmatch(r"0|-?[1-9][0-9]*", numerator)
    assert isinstance(denominator, str) and re.fullmatch(r"[1-9][0-9]*", denominator)
    assert math.gcd(int(numerator), int(denominator)) == 1
    return sp.Rational(int(numerator), int(denominator))


def _decode_polynomial(record):
    # Do not parse the display expression: structured terms are the exact contract.
    assert set(record) == {"expression", "generators", "terms"}
    assert isinstance(record["expression"], str)
    assert isinstance(record["generators"], list)
    names = {str(symbol): symbol for symbol in (
        *anomaly.CARTAN_SYMBOLS, anomaly.S2, anomaly.S4, anomaly.p1,
        anomaly.p2, anomaly.X, anomaly.X4, anomaly.Q,
    )}
    assert len(set(record["generators"])) == len(record["generators"])
    assert all(name in names for name in record["generators"])
    generators = [names[name] for name in record["generators"]]
    assert isinstance(record["terms"], list)
    monomials = set()
    result = sp.Integer(0)
    for term in record["terms"]:
        assert set(term) == {"powers", "coefficient"}
        assert len(term["powers"]) == len(generators)
        assert all(type(power) is int and power >= 0 for power in term["powers"])
        powers = tuple(term["powers"])
        assert powers not in monomials
        monomials.add(powers)
        coefficient = _decode_rational(term["coefficient"])
        assert coefficient != 0
        result += coefficient * sp.prod(symbol ** power for symbol, power in zip(generators, powers))
    return sp.expand(result)


def _assert_json_native(value):
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_json_native(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_native(item)
    else:
        assert type(value) in (str, int, bool, type(None))


def _reject_json_constant(value):
    raise AssertionError("nonstandard JSON constant: " + value)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result
        result[key] = value
    return result


@pytest.mark.parametrize("charge,flux", [(1, 0), (1, 1), (1, -1), (1, 3),
                                         (3, 1), (0, 1), (10 ** 6, 10 ** 6)])
@pytest.mark.parametrize("chirality", [1, -1])
def test_case_report_exact_roundtrip_and_contributions(charge, flux, chirality):
    report = anomaly.case_report(charge, flux, chirality)
    _assert_json_native(report)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report["parameters"] == {"charge": charge, "flux": flux,
                                    "q": charge * flux, "chirality": chirality}
    q = charge * flux
    bookkeeping = report["bookkeeping"]
    expected_indices = {
        "flux": q, "internal_positive": max(q, 0), "internal_negative": max(-q, 0),
        "internal_index": q, "su_twisted_component_index": 16 * q,
        "parent": "weyl", "parent_chirality": chirality,
        "left_family_copies": max(chirality * q, 0),
        "right_family_copies": max(-chirality * q, 0),
        "net_left_families": chirality * q, "net_left_gauge_components": 16 * chirality * q,
    }
    assert {key: bookkeeping[key] for key in expected_indices} == expected_indices
    assert bookkeeping == anomaly.signed_mode_bookkeeping(
        q, gauge_rank=16, parent="weyl", parent_chirality=chirality)
    parent = _decode_polynomial(report["parent"])
    lower = _decode_polynomial(report["lower"])
    pushed = _decode_polynomial(report["pushforward"])
    assert not any(polynomial.has(anomaly.Q) for polynomial in (parent, lower, pushed))
    assert _cartan_form(parent) == chirality * _expression(_product_oracle(charge, 4))
    assert _cartan_form(lower) == chirality * _expression(_product_oracle(charge, 3))
    assert _cartan_form(pushed) == chirality * _sphere_oracle(charge, flux)
    assert pushed == _decode_polynomial(report["expected_index_times_lower"])
    assert set(report["contributions"]) == _CONTRIBUTIONS
    contributions = {key: _decode_polynomial(record)
                     for key, record in report["contributions"].items()}
    assert sp.expand(sum(contributions.values()) - parent) == 0
    expected_contributions = {key: {} for key in _CONTRIBUTIONS}
    for powers, coefficient in _product_oracle(charge, 4).items():
        has_spin10 = any(powers[:5])
        has_u1 = powers[5] != 0
        has_gravity = any(powers[6:])
        if has_gravity:
            key = ("mixed_gravity_spin10" if has_spin10 else
                   "mixed_gravity_u1" if has_u1 else "pure_gravity")
        else:
            key = ("mixed_spin10_u1" if has_spin10 and has_u1 else
                   "pure_spin10" if has_spin10 else "pure_u1")
        expected_contributions[key][powers] = chirality * coefficient
    for key in _CONTRIBUTIONS:
        assert _cartan_form(contributions[key]) == _expression(expected_contributions[key])
    assert _decode_rational(report["p2_coefficient"]) == -sp.Rational(chirality, 90)
    assert report["controls"] and all(value is True for value in report["controls"].values())
    if charge * flux == 0:
        assert report["pushforward"]["terms"] == []
        assert report["expected_index_times_lower"]["terms"] == []
        assert parent != 0


def test_demonstration_schema_and_symbolic_exact_roundtrip():
    report = anomaly.demonstration_report()
    _assert_json_native(report)
    assert report["schema_version"] == 1
    assert [(case["parameters"]["charge"], case["parameters"]["flux"])
            for case in report["cases"]] == [(1, 0), (1, 1), (1, -1), (1, 3), (3, 1), (0, 1)]
    symbolic = report["symbolic"]
    parent = _decode_polynomial(symbolic["parent"])
    lower = _decode_polynomial(symbolic["lower"])
    assert parent.has(anomaly.Q) and lower.has(anomaly.Q)
    for charge in (0, 1, -1, 3):
        assert _cartan_form(parent, charge) == _expression(_product_oracle(charge, 4))
        assert _cartan_form(lower, charge) == _expression(_product_oracle(charge, 3))
    assert set(symbolic["contributions"]) == _CONTRIBUTIONS
    assert sp.expand(sum(_decode_polynomial(record)
                         for record in symbolic["contributions"].values()) - parent) == 0
    assert _decode_rational(symbolic["p2_coefficient"]) == -sp.Rational(1, 90)
    assert _decode_polynomial(symbolic["pure_gravity"]) == parent.subs({
        anomaly.S2: 0, anomaly.S4: 0, anomaly.X: 0})
    assert _decode_polynomial(symbolic["neutral_u1"]) == parent.subs(anomaly.Q, 0)
    assert set(symbolic["traces"]) == {str(power) for power in range(5)}
    for power in range(5):
        assert _decode_polynomial(symbolic["traces"][str(power)]) == anomaly.trace_polynomial(power)
    assert report["controls"] and all(value is True for value in report["controls"].values())
    policy = report["resource_policy"]
    assert policy["rank"] == 5 and policy["weight_count"] == 16
    assert policy["max_trace_power"] == 4 and policy["max_form_degree"] == 8
    assert policy["max_abs_charge_or_flux"] == 10 ** 6
    assert policy["max_abs_q"] == 10 ** 12
    assert policy["cap_is_physical"] is False
    expected_grades = {**{root: 2 for root in anomaly.CARTAN_SYMBOLS},
                       anomaly.S2: 4, anomaly.S4: 8, anomaly.p1: 4, anomaly.p2: 8,
                       anomaly.X: 2, anomaly.X4: 2, anomaly.Q: 0}
    assert anomaly.FORM_DEGREES == expected_grades
    assert report["conventions"]["form_degrees"] == {
        str(symbol): degree for symbol, degree in expected_grades.items()}


def test_form_degree_metadata_is_immutable():
    original = dict(anomaly.FORM_DEGREES)
    with pytest.raises(TypeError):
        anomaly.FORM_DEGREES[anomaly.X] = 99
    with pytest.raises(TypeError):
        del anomaly.FORM_DEGREES[anomaly.p2]
    assert dict(anomaly.FORM_DEGREES) == original
    assert anomaly.FORM_DEGREES[anomaly.X] == 2
    assert anomaly.FORM_DEGREES[anomaly.p2] == 8


@pytest.mark.parametrize("arguments", [[], ["--json"]])
def test_demo_stdout_only_in_empty_working_directory(tmp_path, arguments):
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_chiral_parent_anomaly.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run([sys.executable, "-B", str(script)] + arguments,
                             cwd=str(tmp_path), env=environment, text=True,
                             capture_output=True, timeout=60, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stderr == ""
    assert process.stdout.strip()
    assert list(tmp_path.iterdir()) == []
    if arguments:
        report = json.loads(process.stdout, parse_constant=_reject_json_constant,
                            object_pairs_hook=_reject_duplicate_keys)
        _assert_json_native(report)
        assert report == anomaly.demonstration_report()
    else:
        assert "p2" in process.stdout
        assert "Q" in process.stdout and "q" in process.stdout


def test_demo_rejects_unsupported_grammar_without_files(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_chiral_parent_anomaly.py"
    process = subprocess.run([sys.executable, "-B", str(script), "--charge", "1"],
                             cwd=str(tmp_path), text=True, capture_output=True,
                             timeout=60, check=False)
    assert process.returncode != 0
    assert process.stdout == ""
    assert process.stderr
    assert list(tmp_path.iterdir()) == []


def test_sympy_integers_are_accepted_without_floating_conversion():
    assert anomaly.spinor_weights(sp.Integer(-1)) == anomaly.spinor_weights(-1)
    assert anomaly.trace_polynomial(sp.Integer(4)) == anomaly.trace_polynomial(4)
    assert anomaly.parent_polynomial(sp.Integer(-1)) == anomaly.parent_polynomial(-1)
    assert anomaly.sphere_pushforward(sp.Integer(3), sp.Integer(1), sp.Integer(-1)) == (
        anomaly.sphere_pushforward(3, 1, -1))
