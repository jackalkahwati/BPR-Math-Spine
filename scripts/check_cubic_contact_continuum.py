"""Exact finite contact identities, not an interacting continuum simulator."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json

MAX_MOMENTA = 343
MAX_ALIAS_QUADRUPLES = 81


def _admit_case(n, g):
    if type(n) is not int or n not in (3, 5, 7):
        raise ValueError("n must be a built-in integer in {3, 5, 7}")
    if type(g) is not int or g not in (0, 1, n ** 3):
        raise ValueError("g must be a built-in integer in {0, 1, n**3}")
    if n ** 3 > MAX_MOMENTA or 81 > MAX_ALIAS_QUADRUPLES:
        raise ValueError("control exceeds live enumeration cap")


def _counts(n):
    _admit_case(n, 0)
    band = {0, 1, n - 1}
    retained = excluded = self_inverse = 0
    retained_unordered = excluded_unordered = 0
    for p in product(range(n), repeat=3):
        inverse = tuple((-component) % n for component in p)
        inside = all(component in band for component in p + inverse)
        if inside:
            retained += 1
        else:
            excluded += 1
        self_inverse += p == inverse
        if p <= inverse:
            if inside:
                retained_unordered += 1
            else:
                excluded_unordered += 1
    integer_collisions = modular_collisions = 0
    for p, q, r, s in product((-1, 0, 1), repeat=4):
        difference = p + q - r - s
        integer_collisions += difference == 0
        modular_collisions += difference % n == 0
    fibers = Counter((p + q) % n for p, q in product((-1, 0, 1), repeat=2))
    return {
        "inverse_pairs": retained + excluded,
        "retained_pairs": retained,
        "excluded_pairs": excluded,
        "self_inverse_pairs": self_inverse,
        "retained_unordered_pairs": retained_unordered,
        "excluded_unordered_pairs": excluded_unordered,
        "integer_collisions_1d": integer_collisions,
        "modular_collisions_1d": modular_collisions,
        "modular_collisions_3d": modular_collisions ** 3,
        "max_fiber_3d": max(fibers.values()) ** 3,
    }


def _ordered_contact_element(n, g, p, q, r, s):
    _admit_case(n, g)
    for mode in (p, q, r, s):
        if (type(mode) is not tuple or len(mode) != 3
                or any(type(component) is not int or component not in (-1, 0, 1)
                       for component in mode)):
            raise ValueError("momentum must be a tuple of three admitted built-in integers")
    allowed = all((p[j] + q[j] - r[j] - s[j]) % n == 0 for j in range(3))
    return Fraction(g if allowed else 0, n ** 3)


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("rational encoding requires a built-in integer or Fraction")
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _case_report(n, g, label):
    _admit_case(n, g)
    mapping = {"zero": 0, "fixed": 1, "formal-contact": n ** 3}
    if type(label) is not str or label not in mapping or mapping[label] != g:
        raise ValueError("coupling label does not match the frozen control")
    M, K = n ** 3, 27
    return {
        "n": n, "M": M, "J": 1, "K": K, "g": g, "coupling_label": label,
        "band_saturated": M == K, "no_alias_condition_met": n > 4,
        "counts": _counts(n),
        "expectation": _rational(Fraction(g, M)),
        "contact_norm_squared": _rational(Fraction(g * g, M)),
        "projected_norm_squared": _rational(Fraction(g * g * K, M * M)),
        "leakage_norm_squared": _rational(Fraction(g * g * (M - K), M * M)),
        "band_contact_norm_squared": _rational(Fraction(K, M)),
        "interaction_duhamel_coefficient_squared": _rational(Fraction(g * g * K, M)),
    }


def demonstration_report():
    """Fresh native JSON data for nine fixed exact-arithmetic controls."""
    cases = []
    for n in (3, 5, 7):
        for label, g in (("zero", 0), ("fixed", 1), ("formal-contact", n ** 3)):
            cases.append(_case_report(n, g, label))
    return {
        "schema_version": 1,
        "status": "conditional_contact_scaling_diagnostics",
        "empirical_validation": False,
        "controls": {"n": [3, 5, 7], "J": 1, "ell": 1,
                     "coupling_labels": ["zero", "fixed", "formal-contact"]},
        "cases": cases,
        "limitations": [
            "Geometry, quantization and coupling scalings are supplied assumptions.",
            "The proved continuum comparison is free, fixed-band and two-particle only.",
            "Formal contact matching does not define an interacting continuum Hamiltonian.",
            "Generator leakage is not a lower bound on finite-time evolution error.",
            "Exact finite identities are not empirical validation or matter/gravity completion.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit strict JSON only")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Cubic contact continuum: exact finite identities")
    for case in report["cases"]:
        values = []
        for key in ("expectation", "contact_norm_squared", "projected_norm_squared",
                    "leakage_norm_squared"):
            value = case[key]
            values.append("{}/{}".format(value["numerator"], value["denominator"]))
        print("n={} g={} {} mean={} total_norm_squared={} projected_norm_squared={} leakage_norm_squared={}".format(
            case["n"], case["g"], case["coupling_label"], *values))
    for limitation in report["limitations"]:
        print(limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
