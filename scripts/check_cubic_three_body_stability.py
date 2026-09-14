"""Exact scalar stability diagnostics; no lattice or number-sector simulation."""

import argparse
from fractions import Fraction
import json

MAX_OCCUPATION = 6
MAX_BLOCK_SIDE = 128
MAX_LOCAL_ENTRIES = 3


def _admit():
    if MAX_OCCUPATION < 6 or MAX_BLOCK_SIDE < 128 or MAX_LOCAL_ENTRIES < 3:
        raise ValueError("fixed controls exceed live scalar admission caps")


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("rational encoding requires a built-in integer or Fraction")
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _onsite_case(n, w):
    _admit()
    if type(n) is not int or n not in range(7):
        raise ValueError("n must be a built-in integer in 0..6")
    if type(w) is not int or w not in (1, 2):
        raise ValueError("w must be a built-in integer in {1, 2}")
    pairs = n * (n - 1) // 2
    triples = n * (n - 1) * (n - 2) // 6
    energy = -pairs + w * triples
    bound = Fraction((3 + w) ** 2, 24 * w)
    return {"n": n, "w": w, "pair_count": pairs, "triple_count": triples,
            "energy": _rational(energy), "bound_coefficient": _rational(bound),
            "square_remainder": _rational(energy + n * bound)}


def _local_moments():
    _admit()
    x = Fraction(1, 4)
    Z = 1 + x + x * x / 2
    mean = (x + x * x) / Z
    alpha = Fraction(1, 2) * (1 + x) / Z
    pairs = x * x / (2 * Z)
    return {
        "x": _rational(x), "normalization": _rational(Z),
        "mean_number": _rational(mean), "annihilation": _rational(alpha),
        "pair_mean": _rational(pairs), "triple_mean": _rational(0),
        "kinetic_difference": _rational(mean - alpha * alpha),
        "probabilities": [_rational(1 / Z), _rational(x / Z), _rational(pairs)],
    }


def _block_case(b):
    _admit()
    if type(b) is not int or b not in (1, 2, 128):
        raise ValueError("b must be a built-in integer in {1, 2, 128}")
    # These are scalar counts, never loop bounds or allocated graph dimensions.
    sites = b ** 3
    edges = 3 * b * b * (b - 1)
    crossing = 6 * b * b
    mean = Fraction(10, 41)
    alpha = Fraction(20, 41)
    kinetic = 6 * sites * mean - 2 * edges * alpha * alpha
    pair = Fraction(-2 * sites, 41)
    return {
        "b": b, "sites": sites, "internal_edges": edges,
        "crossing_edges": crossing, "max_number_support": 2 * sites,
        "mean_number": _rational(sites * mean), "shifted_kinetic": _rational(kinetic),
        "pair_energy": _rational(pair), "triple_energy": _rational(0),
        "reference_energy": _rational(kinetic + pair),
    }


def demonstration_report():
    """Fresh JSON-native records for frozen local and scalar-block controls."""
    _admit()
    return {
        "schema_version": 1,
        "status": "conditional_onsite_three_body_stability_diagnostics",
        "empirical_validation": False,
        "onsite_cases": [_onsite_case(n, w) for w in (1, 2) for n in range(7)],
        "local_state": _local_moments(),
        "block_cases": [_block_case(b) for b in (1, 2, 128)],
        "limitations": [
            "The onsite three-body penalty is a supplied assumption and leaves N<=2 unchanged.",
            "Fixed-spacing finite-volume boundedness is not uniform continuum stability.",
            "The continuum obstruction is proved by a trial state with zero triple occupancy.",
            "The fixed particle number exists analytically; no large block or sector is constructed.",
            "Exact scalar checks are not empirical validation or a theory of everything.",
        ],
    }


def _format(value):
    return "{}/{}".format(value["numerator"], value["denominator"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="emit strict JSON only")
    args = parser.parse_args(argv)
    report = demonstration_report()
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    print("Cubic three-body stability: exact scalar checks")
    for case in report["onsite_cases"]:
        print("n={} w={} energy={} bound={} square_remainder={}".format(
            case["n"], case["w"], _format(case["energy"]),
            _format(case["bound_coefficient"]), _format(case["square_remainder"])))
    for case in report["block_cases"]:
        print("scalar_block_b={} reference_energy={} triple_energy={}".format(
            case["b"], _format(case["reference_energy"]), _format(case["triple_energy"])))
    for limitation in report["limitations"]:
        print(limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
