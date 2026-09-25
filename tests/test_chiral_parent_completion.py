"""Exact checks for doc/derivations/chiral_parent_completion_2026-09-25.md.

The oracle rebuilds s*[Ahat * ch_R * exp(QX)]_8 from Cartan weights by series
expansion, recovers invariant coefficients by solving a linear system, and
factors with SymPy. It does not use the module's trace table or coefficient
formulas. These are exact algebraic checks of supplied field content, not
physics validation.
"""

import itertools
import json
from fractions import Fraction
from functools import lru_cache

import pytest
import sympy as sp

import bpr.chiral_parent_completion as c
from bpr.chiral_parent_anomaly import parent_polynomial, trace_polynomial, Q as PARENT_Q


ys = sp.symbols("y1:6")
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
        ch = sum(sp.series(sp.exp(t * sum(w * y for w, y in zip(weight, ys))), t, 0, 5).removeO()
                 for weight in _weights(rep))
        ex = sp.series(sp.exp(t * Q * X), t, 0, 5).removeO()
        total += s * sp.expand(ahat * ch * ex).coeff(t, 4)
    return sp.expand(total)


@lru_cache(maxsize=None)
def _oracle_invariant(field):
    """Invariant-symbol form of one field's I8, by solving for coefficients."""
    target = _oracle_polynomial([field])
    s2 = sum(y ** 2 for y in ys)
    s4 = sum(y ** 4 for y in ys)
    basis_inv = [S2 ** 2, S4, S2 * X ** 2, X ** 4, S2 * p1, X ** 2 * p1, p1 ** 2, p2]
    basis_cartan = [s2 ** 2, s4, s2 * X ** 2, X ** 4, s2 * p1, X ** 2 * p1, p1 ** 2, p2]
    coeffs = sp.symbols("k0:8")
    residual = sp.expand(target - sum(k * b for k, b in zip(coeffs, basis_cartan)))
    equations = sp.Poly(residual, *ys, X, p1, p2).coeffs()
    solution = sp.solve(equations, coeffs, dict=True)
    assert len(solution) == 1
    return sp.expand(sum(solution[0][k] * b for k, b in zip(coeffs, basis_inv)))


def _oracle_total(fields):
    counts = {}
    for field in fields:
        counts[tuple(field)] = counts.get(tuple(field), 0) + 1
    return sp.expand(sum(k * _oracle_invariant(f) for f, k in counts.items()))


def _to_cartan(expr):
    return sp.expand(expr.subs({S2: sum(y ** 2 for y in ys), S4: sum(y ** 4 for y in ys)}))


def _factorizable_by_sympy(expr):
    u, v, w = sp.symbols("u v w")
    reduced = sp.expand(expr.subs({S2: u, p1: w}))
    reduced = sp.expand(reduced.subs(X ** 4, v ** 2).subs(X ** 2, v))
    if reduced.has(X) or reduced.has(S4) or reduced.has(p2):
        return False
    if reduced == 0:
        return True
    _, factors = sp.factor_list(reduced, u, v, w, domain=sp.QQ)
    return not (len(factors) == 1 and factors[0][1] == 1)


# ---------------------------------------------------------------------------
# Representation data and per-field polynomials
# ---------------------------------------------------------------------------

def test_traces_match_parent_module_for_16():
    dim, _, t2, _, a4, b4 = c.representation_traces("16")
    assert dim == 16
    x = sp.symbols("x1:6")
    s2 = sum(v ** 2 for v in x)
    s4 = sum(v ** 4 for v in x)
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


@pytest.mark.parametrize("field", [(1, "10", 0), (-1, "10", 2), (1, "1", 3), (-1, "16bar", 1),
                                   (-1, "16", 0), (1, "16", 1)])
def test_field_polynomials_match_characteristic_class_oracle(field):
    assert _to_cartan(c.field_polynomial(field)) == _oracle_polynomial([field])
    assert sp.expand(c.field_polynomial(field) - _oracle_invariant(field)) == 0


def test_conjugation_preserves_six_dimensional_field():
    # (s, R, Q) and (s, Rbar, -Q) are the same 6D field.
    for s in (1, -1):
        for Q in (0, 1, 2):
            assert sp.expand(c.field_polynomial((s, "16", Q)) - c.field_polynomial((s, "16bar", -Q))) == 0


# ---------------------------------------------------------------------------
# Theorem 3: the minimal completion in class C
# ---------------------------------------------------------------------------

def test_minimal_completion_polynomial_and_factorization():
    fields = c.minimal_completion()
    assert fields == [c.PARENT, (-1, "16", 0)]
    total = c.total_polynomial(fields)
    target = sp.Rational(1, 3) * X ** 2 * (3 * S2 + 2 * X ** 2 - p1)
    assert sp.expand(total - target) == 0
    assert sp.expand(_oracle_total(fields) - target) == 0
    coefficients = c.anomaly_coefficients(fields)
    assert coefficients["delta"] == 0 and coefficients["epsilon"] != 0  # delta=0 branch
    report = c.completion_report(fields)
    assert report["status"] == "factorizable"
    assert report["factor_product_matches_I8"] is True
    assert sp.expand(sp.sympify(report["pushforward"]) - (6 * S2 * X4 + 8 * X4 ** 3 - 2 * p1 * X4)) == 0
    assert report["pushforward_matches_zero_modes"] is True


def test_minimal_completion_zero_modes_and_ledger():
    report = c.completion_report(c.minimal_completion(), flux=3)
    modes = {(m["representation"], m["charge"]): m["multiplicity"] for m in report["zero_modes"]}
    assert modes == {("16", 1): 3}
    ledger = report["four_d_ledger"]
    assert ledger["spin10_cubic"] == 0
    assert ledger["u1_cubic"] == 48 and ledger["gravity_u1"] == 48
    assert ledger["spin10_squared_u1_trace_sum"] == {"numerator": 12, "denominator": 1}
    assert ledger["spin10_squared_u1_I6_coefficient"] == {"numerator": 6, "denominator": 1}
    assert ledger["su2_doublets"] == 12 and ledger["witten_parity_even"]
    assert ledger["z16_odd_center_count_mod16"] == 0
    assert ledger["z16_all_massless_modes_odd_center"] is True


def test_four_d_anomaly_is_proportional_to_x4():
    # Every pushforward term carries X4, so the descended axion can cancel it.
    for fields in (c.minimal_completion(), c.restricted_minimal_completion()):
        pushed = c.pushforward(fields, 3)
        assert sp.expand(pushed - X4 * sp.cancel(pushed / X4)) == 0
        assert sp.Poly(sp.cancel(pushed / X4), X4, S2, p1).total_degree() <= 2


def test_flux_sign_reverses_four_d_chirality():
    minus = c.zero_modes(c.minimal_completion(), -3)
    assert minus == [{"representation": "16bar", "charge": -1, "multiplicity": 3}]


@pytest.mark.parametrize("flux", [1, 2, 3, -4])
def test_pushforward_equals_index_theorem_content(flux):
    for fields in (c.minimal_completion(), c.restricted_minimal_completion()):
        assert sp.expand(c.pushforward(fields, flux)
                         - c.four_d_polynomial(c.zero_modes(fields, flux))) == 0


def test_class_c_budget_scan():
    for budget in range(0, 16):
        result = c.completion_search(4, budget, True)
        assert result["examined"] == 0 and result["solutions"] == []
    sixteen = c.completion_search(4, 16, True)["solutions"]
    assert len(sixteen) == 1
    assert sixteen[0]["spinors_minus"] == 1 and sixteen[0]["tens_plus"] == 0
    assert c.completion_search(4, 17, True)["solutions"] == []
    eighteen = c.completion_search(4, 18, True)["solutions"]
    assert len(eighteen) == 25
    for solution in eighteen:
        assert solution["spinors_minus"] == 1
        fields = [tuple(f) for f in solution["fields"]]
        total = c.total_polynomial(fields)
        # Vectorlike Spin(10) content: every term carries X^2.
        assert sp.expand(total - X ** 2 * sp.cancel(total / X ** 2)) == 0
        assert _factorizable_by_sympy(total)


def test_reviewer_eighteen_component_example():
    fields = c.minimal_completion() + [(1, "1", 0), (-1, "1", 4)]
    assert sp.expand(c.total_polynomial(fields) - X ** 2 * (S2 - 10 * X ** 2)) == 0
    modes = {(m["representation"], m["charge"]): m["multiplicity"] for m in c.zero_modes(fields, 3)}
    assert modes == {("16", 1): 3, ("1", -4): 12}


# ---------------------------------------------------------------------------
# Theorem 4: the narrower class C0 without added spinors
# ---------------------------------------------------------------------------

def test_restricted_minimal_completion():
    fields = c.restricted_minimal_completion()
    total = c.total_polynomial(fields)
    target = sp.Rational(1, 8) * S2 * (S2 + 8 * X ** 2 - p1)
    assert sp.expand(total - target) == 0
    assert sp.expand(_oracle_total(fields) - target) == 0
    report = c.completion_report(fields, flux=3)
    modes = {(m["representation"], m["charge"]): m["multiplicity"] for m in report["zero_modes"]}
    assert modes == {("16", 1): 3, ("1", -1): 48}
    ledger = report["four_d_ledger"]
    assert ledger["u1_cubic"] == 0 and ledger["gravity_u1"] == 0
    assert ledger["z16_all_massless_modes_odd_center"] is False


def test_partial_completions_keep_their_obstructions():
    parent = [c.PARENT]
    assert c.factorization_status(c.anomaly_coefficients(parent))["status"] == "irreducible_S4"
    with_ten = parent + [(1, "10", 0)]
    assert c.factorization_status(c.anomaly_coefficients(with_ten))["status"] == "irreducible_p2"
    neutral = with_ten + [(-1, "1", 0)] * 26
    status = c.factorization_status(c.anomaly_coefficients(neutral))
    assert status["status"] == "not_factorizable_rank3_or_irrational"
    assert not _factorizable_by_sympy(c.total_polynomial(neutral))


def test_two_two_forms_suffice_for_the_neutral_singlet_case():
    neutral = [c.PARENT, (1, "10", 0)] + [(-1, "1", 0)] * 26
    identity = (sp.Rational(1, 24) * (3 * S2 + 8 * X ** 2) * (S2 + sp.Rational(16, 3) * X ** 2 - p1)
                - sp.Rational(10, 9) * X ** 4)
    assert sp.expand(c.total_polynomial(neutral) - identity) == 0


def _oracle_search(q_max):
    found = set()
    for counts in itertools.product(range(27), repeat=q_max + 1):
        if sum(counts) != 26:
            continue
        fields = [c.PARENT, (1, "10", 0)]
        for q, k in enumerate(counts):
            fields += [(-1, "1", q)] * k
        if _factorizable_by_sympy(_oracle_total(fields)):
            found.add(tuple(counts))
    return found


@pytest.mark.parametrize("q_max", [1, 2])
def test_restricted_search_matches_independent_oracle(q_max):
    result = c.completion_search(q_max, 36, False)
    module = set()
    for solution in result["solutions"]:
        counts = solution["negative_singlet_charge_counts"]
        module.add(tuple(counts.get(str(q), 0) for q in range(q_max + 1)))
        assert solution["tens_plus"] == 1 and solution["tens_minus"] == 0
        assert solution["spinors_plus"] == 0 and solution["spinors_minus"] == 0
        assert solution["positive_singlet_charge_counts"] == {}
    assert module == _oracle_search(q_max)


def test_restricted_solution_counts():
    one = c.completion_search(1, 36, False)["solutions"]
    assert len(one) == 1
    assert one[0]["negative_singlet_charge_counts"] == {"0": 10, "1": 16}
    two = c.completion_search(2, 36, False)["solutions"]
    assert {tuple(sorted(s["negative_singlet_charge_counts"].items())) for s in two} == {
        (("0", 10), ("1", 16)), (("0", 10), ("1", 12), ("2", 4)), (("0", 16), ("2", 10))}
    assert len(c.completion_search(3, 36, False)["solutions"]) == 8
    four = c.completion_search(4, 36, False)["solutions"]
    assert len(four) == 26
    for solution in four:
        counts = {int(q): k for q, k in solution["negative_singlet_charge_counts"].items()}
        P2 = sum(q * q * k for q, k in counts.items())
        k_shift = sp.Rational(P2 - 16, 6)
        form = sp.Rational(1, 8) * (S2 - k_shift * X ** 2) * (S2 + (k_shift + 8) * X ** 2 - p1)
        fields = [tuple(f) for f in solution["fields"]]
        assert sp.expand(c.total_polynomial(fields) - form) == 0


def test_below_minimal_restricted_budget_has_no_candidates():
    for budget in range(0, 36):
        result = c.completion_search(2, budget, False)
        assert result["examined"] == 0 and result["solutions"] == []


def test_every_restricted_solution_factorizes_and_pushes_forward():
    cases = [(2, 36), (2, 38), (3, 36)]
    for q_max, budget in cases:
        for solution in c.completion_search(q_max, budget, False)["solutions"]:
            fields = [tuple(f) for f in solution["fields"]]
            total = c.total_polynomial(fields)
            assert sp.expand(total - c.factor_expression(solution["factors"])) == 0
            assert _factorizable_by_sympy(total)
            assert sp.expand(c.pushforward(fields, 3)
                             - c.four_d_polynomial(c.zero_modes(fields, 3))) == 0


def test_abelian_anomaly_freedom_selects_the_restricted_minimal_completion():
    survivors = []
    for solution in c.completion_search(3, 36, False)["solutions"]:
        fields = [tuple(f) for f in solution["fields"]]
        ledger = c.four_d_ledger(c.zero_modes(fields, 3))
        if ledger["u1_cubic"] == 0 and ledger["gravity_u1"] == 0:
            survivors.append(solution["negative_singlet_charge_counts"])
    assert survivors == [{"0": 10, "1": 16}]


# ---------------------------------------------------------------------------
# Factorization criterion: every branch
# ---------------------------------------------------------------------------

def _coefficients(**values):
    base = {key: Fraction(0) for key in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "S4", "p2")}
    base.update({k: Fraction(v) for k, v in values.items()})
    return base


def _form(coefficients):
    u, v, w = S2, X ** 2, p1
    cc = {k: sp.Rational(val.numerator, val.denominator) for k, val in coefficients.items()}
    return sp.expand(cc["alpha"] * u * u + cc["beta"] * u * v + cc["gamma"] * v * v
                     + cc["delta"] * u * w + cc["epsilon"] * v * w)


@pytest.mark.parametrize("values,factorizable", [
    ({"alpha": 1, "gamma": -1}, True),                   # delta=eps=0, square discriminant
    ({"alpha": 1, "gamma": -2}, False),                  # delta=eps=0, irrational
    ({"alpha": 1, "gamma": 1}, False),                   # delta=eps=0, definite
    ({"beta": 3}, True),                                 # delta=eps=0, alpha=gamma=0
    ({"gamma": 5}, True),                                # delta=eps=0, perfect square
    ({"alpha": 2, "beta": 1, "delta": 4}, True),         # epsilon=0 branch
    ({"beta": 1, "gamma": 2, "epsilon": -1}, True),      # delta=0 branch
    ({"alpha": 1, "beta": 1, "gamma": 1, "delta": 1, "epsilon": 1}, False),
])
def test_factorization_status_branches(values, factorizable):
    coefficients = _coefficients(**values)
    status = c.factorization_status(coefficients)
    assert (status["status"] == "factorizable") == factorizable
    assert _factorizable_by_sympy(_form(coefficients)) == factorizable
    if factorizable:
        assert sp.expand(c.factor_expression(status["factors"]) - _form(coefficients)) == 0


def test_delta_epsilon_zero_branch_from_fields():
    fields = c.minimal_completion() + [(-1, "1", 1)] * 16 + [(1, "1", 0)] * 16
    coefficients = c.anomaly_coefficients(fields)
    assert coefficients["delta"] == 0 and coefficients["epsilon"] == 0
    status = c.factorization_status(coefficients)
    assert status["status"] == "factorizable"
    assert sp.expand(c.factor_expression(status["factors"]) - c.total_polynomial(fields)) == 0
    assert sp.expand(c.total_polynomial(fields) - S2 * X ** 2) == 0


# ---------------------------------------------------------------------------
# Standard Model Z16 control and branching data
# ---------------------------------------------------------------------------

def test_standard_model_charges_are_one_mod_four():
    # (B-L, Y) for Q, u^c, d^c, L, e^c, nu^c with X_SM = 5(B-L) - 4Y.
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
    assert report["minimal_completion"]["status"] == "factorizable"
    assert report["limitations"] == c.LIMITATIONS
    scan = {row["extra_components"]: row["solutions"] for row in report["class_C_budget_scan_q_max_4"]}
    assert [b for b, n in scan.items() if n] == [16, 18]
    assert scan[16] == 1


@pytest.mark.parametrize("bad", [(1, "16", 1.0), (0, "16", 1), (1, "45", 0), (True, "1", 0), [1, "1", 0]])
def test_bad_fields_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        c.field_polynomial(bad)


def test_search_caps():
    with pytest.raises(ValueError):
        c.completion_search(5)
    with pytest.raises(ValueError):
        c.completion_search(1, 41)
    with pytest.raises(TypeError):
        c.completion_search(1, 16, 1)
    with pytest.raises(ValueError):
        c.completion_search(4, 40, True)  # candidate cap
