"""Independent exact checks for doc/derivations/chiral_parent_completion_2026-09-25.md.

Oracles rebuild Ahat * ch_R * exp(QX) from Cartan weights and factor with
SymPy, independently of the module's coefficient formulas. These are exact
algebraic checks of a supplied field content, not physics validation.
"""

import itertools
import json
from fractions import Fraction

import pytest
import sympy as sp

import bpr.chiral_parent_completion as c
from bpr.chiral_parent_anomaly import parent_polynomial, trace_polynomial, Q as PARENT_Q


xs = sp.symbols("y1:6")
S2, S4, X, p1, p2, X4 = c.S2, c.S4, c.X, c.p1, c.p2, c.X4
t = sp.Symbol("t")  # form-degree bookkeeping: curvature ~ t, p1 ~ t^2, p2 ~ t^4


def _weights(rep):
    if rep == "1":
        return [(0,) * 5]
    if rep == "10":
        return [tuple(sign if j == i else 0 for j in range(5)) for i in range(5) for sign in (1, -1)]
    parity = 1 if rep == "16" else -1
    return [tuple(sp.Rational(s, 2) for s in signs) for signs in itertools.product((-1, 1), repeat=5)
            if (-1) ** signs.count(-1) == parity]


def _oracle_polynomial(fields):
    """Degree-8 part of sum s * Ahat * ch_R * exp(QX), in Cartan variables."""
    total = 0
    ahat = 1 - p1 * t ** 2 / 24 + (7 * p1 ** 2 - 4 * p2) * t ** 4 / 5760
    for s, rep, Q in fields:
        ch = sum(sp.series(sp.exp(t * sum(w * x for w, x in zip(weight, xs))), t, 0, 5).removeO()
                 for weight in _weights(rep))
        ex = sp.series(sp.exp(t * Q * X), t, 0, 5).removeO()
        total += s * sp.expand(ahat * ch * ex).coeff(t, 4)
    return sp.expand(total)


def _to_cartan(expr):
    return sp.expand(expr.subs({S2: sum(x ** 2 for x in xs), S4: sum(x ** 4 for x in xs)}))


def _factorizable_by_sympy(expr):
    u, v, w = sp.symbols("u v w")
    reduced = sp.expand(expr.subs({S2: u, p1: w}))
    reduced = sp.expand(reduced.subs(X ** 4, v ** 2).subs(X ** 2, v))
    if reduced.has(X) or reduced.has(S4) or reduced.has(p2):
        return False
    if reduced == 0:
        return True
    _, factors = sp.factor_list(reduced, u, v, w, domain=sp.QQ)
    degree = sum(sp.Poly(f, u, v, w).total_degree() * k for f, k in factors)
    return degree == 2 and not (len(factors) == 1 and factors[0][1] == 1)


# ---------------------------------------------------------------------------
# Representation data and per-field polynomials
# ---------------------------------------------------------------------------

def test_traces_match_parent_module_for_16():
    dim, _, t2, _, a4, b4 = c.representation_traces("16")
    assert dim == 16
    y = sp.symbols("x1:6")
    s2 = sum(v ** 2 for v in y)
    s4 = sum(v ** 4 for v in y)
    assert sp.expand(trace_polynomial(2) - sp.Rational(t2.numerator, t2.denominator) * s2) == 0
    assert sp.expand(trace_polynomial(4) - sp.Rational(a4.numerator, a4.denominator) * s2 ** 2
                     - sp.Rational(b4.numerator, b4.denominator) * s4) == 0


@pytest.mark.parametrize("rep,expected", [("1", (1, 0, 0, 0)), ("10", (10, 2, 0, 2)),
                                          ("16", (16, 4, 3, -2)), ("16bar", (16, 4, 3, -2))])
def test_representation_table(rep, expected):
    dim, _, t2, _, a4, b4 = c.representation_traces(rep)
    assert (dim, t2, a4, b4) == tuple(Fraction(v) for v in expected)


@pytest.mark.parametrize("s", [1, -1])
@pytest.mark.parametrize("charge", [0, 1, -2, 3])
def test_16_field_polynomial_matches_parent_module(s, charge):
    assert sp.expand(c.field_polynomial((s, "16", charge))
                     - parent_polynomial(s).subs(PARENT_Q, charge)) == 0


@pytest.mark.parametrize("field", [(1, "10", 0), (-1, "10", 2), (1, "1", 3), (-1, "16bar", 1)])
def test_field_polynomials_match_characteristic_class_oracle(field):
    assert _to_cartan(c.field_polynomial(field)) == _oracle_polynomial([field])


# ---------------------------------------------------------------------------
# The minimal completion
# ---------------------------------------------------------------------------

def test_minimal_completion_polynomial_and_factorization():
    fields = c.minimal_completion()
    total = c.total_polynomial(fields)
    target = sp.Rational(1, 8) * S2 * (S2 + 8 * X ** 2 - p1)
    assert sp.expand(total - target) == 0
    assert _to_cartan(total) == _oracle_polynomial(fields)
    report = c.completion_report(fields)
    assert report["status"] == "factorizable"
    assert report["factor_product_matches_I8"] is True
    assert report["pushforward"] == str(6 * S2 * X4)
    assert report["pushforward_matches_zero_modes"] is True


def test_minimal_completion_zero_modes_and_ledger():
    report = c.completion_report(c.minimal_completion(), flux=3)
    modes = {(m["representation"], m["charge"]): m["multiplicity"] for m in report["zero_modes"]}
    assert modes == {("16", 1): 3, ("1", -1): 48}
    ledger = report["four_d_ledger"]
    assert ledger["spin10_cubic"] == 0
    assert ledger["u1_cubic"] == 0 and ledger["gravity_u1"] == 0
    assert ledger["spin10_squared_u1"] == {"numerator": 12, "denominator": 1}
    assert ledger["su2_doublets"] == 12 and ledger["witten_parity_even"]
    assert ledger["z16_spin10_charged_count_mod16"] == 0
    assert ledger["z16_full_spectrum_applicable"] is False


def test_flux_sign_reverses_four_d_chirality():
    plus = c.zero_modes(c.minimal_completion(), 3)
    minus = c.zero_modes(c.minimal_completion(), -3)
    assert {(m["representation"], m["charge"]) for m in minus} == {("16bar", -1), ("1", 1)}
    assert sum(m["multiplicity"] for m in plus) == sum(m["multiplicity"] for m in minus)


@pytest.mark.parametrize("flux", [1, 2, 3, -4])
def test_pushforward_equals_index_theorem_content(flux):
    fields = c.minimal_completion()
    assert sp.expand(c.pushforward(fields, flux)
                     - c.four_d_polynomial(c.zero_modes(fields, flux))) == 0


def test_partial_completions_keep_their_obstructions():
    parent = [c.PARENT]
    assert c.factorization_status(c.anomaly_coefficients(parent))["status"] == "irreducible_S4"
    with_ten = parent + [(1, "10", 0)]
    assert c.factorization_status(c.anomaly_coefficients(with_ten))["status"] == "irreducible_p2"
    neutral = with_ten + [(-1, "1", 0)] * 26
    status = c.factorization_status(c.anomaly_coefficients(neutral))
    assert status["status"] == "not_factorizable_rank3_or_irrational"
    assert not _factorizable_by_sympy(c.total_polynomial(neutral))


# ---------------------------------------------------------------------------
# Exhaustive search against a SymPy oracle
# ---------------------------------------------------------------------------

def _oracle_search(q_max):
    found = set()
    for counts in itertools.product(range(27), repeat=q_max + 1):
        if sum(counts) != 26:
            continue
        fields = [c.PARENT, (1, "10", 0)]
        for q, k in enumerate(counts):
            fields += [(-1, "1", q)] * k
        if _factorizable_by_sympy(c.total_polynomial(fields)):
            found.add(tuple(counts))
    return found


@pytest.mark.parametrize("q_max", [1, 2])
def test_search_matches_sympy_oracle(q_max):
    result = c.completion_search(q_max)
    module = set()
    for solution in result["solutions"]:
        counts = solution["negative_singlet_charge_counts"]
        module.add(tuple(counts.get(str(q), 0) for q in range(q_max + 1)))
        assert solution["tens_plus"] == 1 and solution["tens_minus"] == 0
        assert solution["positive_singlet_charge_counts"] == {}
    assert module == _oracle_search(q_max)


def test_q_max_one_is_unique_and_q_max_two_is_not():
    one = c.completion_search(1)["solutions"]
    assert len(one) == 1
    assert one[0]["negative_singlet_charge_counts"] == {"0": 10, "1": 16}
    two = c.completion_search(2)["solutions"]
    assert {tuple(sorted(s["negative_singlet_charge_counts"].items())) for s in two} == {
        (("0", 10), ("1", 16)), (("0", 10), ("1", 12), ("2", 4)), (("0", 16), ("2", 10))}
    assert len(c.completion_search(3)["solutions"]) == 8


def test_below_minimal_budget_has_no_candidates():
    for budget in range(0, 36):
        result = c.completion_search(2, budget)
        assert result["examined"] == 0 and result["solutions"] == []


def test_every_solution_factorizes_and_pushes_forward_consistently():
    for budget in (36, 38):
        for solution in c.completion_search(2, budget)["solutions"]:
            fields = [tuple(f) for f in solution["fields"]]
            total = c.total_polynomial(fields)
            assert sp.expand(total - c.factor_expression(solution["factors"])) == 0
            assert _factorizable_by_sympy(total)
            assert sp.expand(c.pushforward(fields, 3)
                             - c.four_d_polynomial(c.zero_modes(fields, 3))) == 0


def test_abelian_anomaly_freedom_selects_the_minimal_completion():
    survivors = []
    for solution in c.completion_search(3)["solutions"]:
        fields = [tuple(f) for f in solution["fields"]]
        for flux in (1, 3, -2):
            ledger = c.four_d_ledger(c.zero_modes(fields, flux))
            free = ledger["u1_cubic"] == 0 and ledger["gravity_u1"] == 0
            if flux == 3 and free:
                survivors.append(solution["negative_singlet_charge_counts"])
    assert survivors == [{"0": 10, "1": 16}]


# ---------------------------------------------------------------------------
# Standard Model Z16 control and branching data
# ---------------------------------------------------------------------------

def test_standard_model_charges_are_one_mod_four():
    # (B-L, Y) for Q, u^c, d^c, L, e^c, nu^c.
    for bl, y in [(Fraction(1, 3), Fraction(1, 6)), (Fraction(-1, 3), Fraction(-2, 3)),
                  (Fraction(-1, 3), Fraction(1, 3)), (Fraction(-1), Fraction(-1, 2)),
                  (Fraction(1), Fraction(1)), (Fraction(1), Fraction(0))]:
        x = 5 * bl - 4 * y
        assert x.denominator == 1 and x.numerator % 4 == 1


@pytest.mark.parametrize("generations", range(0, 6))
@pytest.mark.parametrize("neutrinos", range(0, 21))
def test_standard_model_z16_requires_neutrinos_equal_generations_mod_16(generations, neutrinos):
    result = c.standard_model_z16(generations, neutrinos)
    assert result["anomaly_free"] == ((neutrinos - generations) % 16 == 0)


def test_su2_doublets_from_su5_branching():
    # 16 = 10 + 5bar + 1; 10 = (3,2)+(3bar,1)+(1,1); 5bar = (3bar,1)+(1,2).
    assert c.SU2_DOUBLETS["16"] == 3 + 1
    # 10 of SO(10) = 5 + 5bar: one doublet each.
    assert c.SU2_DOUBLETS["10"] == 2


# ---------------------------------------------------------------------------
# Report contract
# ---------------------------------------------------------------------------

def test_demonstration_report_strict_json():
    report = c.demonstration_report()
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report
    assert report["empirical_validation"] is False
    assert report["derived_from_bpr"] is False
    assert report["parent_only"]["status"] == "irreducible_S4"
    assert report["limitations"] == c.LIMITATIONS
    abelian_free = [s for search in report["searches"] for s in search["solutions"]
                    if s["four_d_abelian_anomaly_free"]]
    assert all(s["negative_singlet_charge_counts"] == {"0": 10, "1": 16} for s in abelian_free)


@pytest.mark.parametrize("bad", [(1, "16", 1.0), (0, "16", 1), (1, "45", 0), (True, "1", 0), [1, "1", 0]])
def test_bad_fields_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        c.field_polynomial(bad)


def test_search_caps():
    with pytest.raises(ValueError):
        c.completion_search(5)
    with pytest.raises(ValueError):
        c.completion_search(1, 41)
