"""Exact abstract rank-one resolvent checks, not numerical lattice scattering."""

import argparse
from fractions import Fraction
import json

MAX_DIMENSION = 2


def _admit_dimension():
    if MAX_DIMENSION < 2:
        raise ValueError("fixture exceeds live dimension cap")


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("rational encoding requires a built-in integer or Fraction")
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _free_resolvent(z):
    if type(z) is not int or z not in (-2, -1, 1, 2):
        raise ValueError("unsupported built-in integer frequency")
    if z in (-1, 1):
        raise ValueError("free pole: free resolvent is undefined")
    return Fraction(z, z * z - 1)


def _case_report(z, g):
    _admit_dimension()
    if type(z) is not int or z not in (-2, 2):
        raise ValueError("unsupported case frequency")
    regular = type(g) is int and g in (-1, 0, 1)
    pole_control = type(g) is Fraction and g == Fraction(3, 2) and z == 2
    if not (regular or pole_control):
        raise ValueError("unsupported case coupling")
    free = _free_resolvent(z)
    denominator = 1 - g * free
    determinant = z * (z - g) - 1
    if denominator == 0:
        status = "expected_interacting_pole"
        tau = source = None
    else:
        status = "regular"
        tau = _rational(g / denominator)
        source = _rational(free / denominator)
    return {
        "z": z, "g": _rational(g), "status": status,
        "free_source": _rational(free), "denominator": _rational(denominator),
        "interacting_determinant": _rational(determinant),
        "tau": tau, "interacting_source": source,
    }


def demonstration_report():
    """Return six regular cases and one explicitly singular algebraic control."""
    _admit_dimension()
    regular = [_case_report(z, g) for z in (-2, 2) for g in (-1, 0, 1)]
    pole = _case_report(2, Fraction(3, 2))
    _admit_dimension()
    fixture = {"A": [[0, 1], [1, 0]], "e": [1, 0], "dimension": 2}
    return {
        "schema_version": 1,
        "status": "abstract_rank_one_resolvent_diagnostics",
        "empirical_validation": False,
        "fixture": fixture,
        "regular_cases": regular,
        "pole_case": pole,
        "limitations": [
            "This fixture checks finite rank-one algebra, not a cubic lattice approximation.",
            "The scattering limit rests on the reviewed analytic boundary proof, not these tests.",
            "Nonnegative onsite couplings have vanishing continuum scattering in the stated model.",
            "The attractive two-body tuning is not an adopted stable many-body model.",
            "No empirical validation, matter/gravity completion or TOE claim is made.",
        ],
    }


def _format_rational(value):
    if value is None:
        return "undefined"
    return "{}/{}".format(value["numerator"], value["denominator"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit strict JSON only")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Cubic two-body scattering: finite resolvent algebra")
    for case in report["regular_cases"] + [report["pole_case"]]:
        print("z={} g={} status={} determinant={} tau={} source={}".format(
            case["z"], _format_rational(case["g"]), case["status"],
            _format_rational(case["interacting_determinant"]),
            _format_rational(case["tau"]), _format_rational(case["interacting_source"])))
    for limitation in report["limitations"]:
        print(limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
