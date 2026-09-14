"""Fixed exact geometry controls; not a continuum stability simulation."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from math import comb

MAX_PARTICLES = 6
MAX_PAIRS = 15
MAX_TRIPLES = 20


def _admit(positions, spacing):
    if type(positions) is not tuple:
        raise ValueError("positions must be a built-in tuple")
    count = len(positions)
    if count > 6 or count > MAX_PARTICLES:
        raise ValueError("particle count exceeds fixed or live cap")
    if comb(count, 2) > MAX_PAIRS or comb(count, 3) > MAX_TRIPLES:
        raise ValueError("geometry exceeds live pair or triple cap")
    if type(spacing) is not Fraction or spacing not in (Fraction(1), Fraction(1, 2)):
        raise ValueError("spacing must be an exact admitted Fraction")
    for point in positions:
        if type(point) is not tuple or len(point) != 3:
            raise ValueError("each position must be a tuple of three coordinates")
        for coordinate in point:
            if type(coordinate) is not int or not -4 <= coordinate <= 4:
                raise ValueError("coordinates must be built-in integers in [-4, 4]")


def _squared_distance(left, right, spacing):
    return spacing * spacing * sum((x - y) ** 2 for x, y in zip(left, right))


def _count_triples(positions, spacing):
    _admit(positions, spacing)
    distances = {
        (i, j): _squared_distance(positions[i], positions[j], spacing)
        for i, j in combinations(range(len(positions)), 2)
    }
    return sum(
        distances[i, j] <= 1 and distances[i, k] <= 1 and distances[j, k] <= 1
        for i, j, k in combinations(range(len(positions)), 3)
    )


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("rational encoding requires a built-in integer or Fraction")
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fixtures():
    O, X, Y = (0, 0, 0), (1, 0, 0), (0, 1, 0)
    one, half = Fraction(1), Fraction(1, 2)
    return (
        ("empty", (), one),
        ("single", (O,), one),
        ("coincident_pair", (O, O), one),
        ("separated_pair", (O, (2, 0, 0)), one),
        ("coincident_triple", (O, O, O), one),
        ("doubled_near", (O, O, X), half),
        ("doubled_boundary", (O, O, X), one),
        ("doubled_far", (O, O, (2, 0, 0)), one),
        ("distinct_near", (O, X, Y), half),
        ("diagonal_far", (O, X, Y), one),
        ("anchor_trap", ((-1, 0, 0), O, X), one),
        ("six_near", (O, O, O, X, X, Y), half),
        ("split_clusters", (O, O, O, (3, 0, 0), (3, 0, 0), (3, 0, 0)), one),
    )


def _case_report(label, positions, spacing):
    _admit(positions, spacing)
    if type(label) is not str or label not in tuple(item[0] for item in _fixtures()):
        raise ValueError("unknown frozen fixture label")
    triples = _count_triples(positions, spacing)
    onsite = sum(comb(count, 3) for count in Counter(positions).values())
    return {
        "id": label, "positions": [list(point) for point in positions],
        "spacing": _rational(spacing), "particle_count": len(positions),
        "triple_count": triples, "onsite_triple_count": onsite,
        "triple_upper_bound": comb(len(positions), 3),
        "repulsion_energy": _rational(triples),
    }


def demonstration_report():
    """Return fresh native JSON data for exactly thirteen fixed configurations."""
    return {
        "schema_version": 1,
        "status": "conditional_fixed_range_three_body_diagnostics",
        "empirical_validation": False,
        "range": _rational(1), "strength": _rational(1),
        "cases": [_case_report(*fixture) for fixture in _fixtures()],
        "limitations": [
            "Range and strength are supplied fixed physical parameters, not fitted predictions.",
            "Triples count particle labels, including coincident positions, using all three distances.",
            "Two-particle scattering is unchanged, but bounded strength does not cure continuum collapse.",
            "Multiplier convergence is not convergence of the tuned many-body dynamics.",
            "Exact geometry checks are not empirical validation or a theory of everything.",
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
    print("Finite-range three-body repulsion: exact geometry checks")
    for case in report["cases"]:
        energy = case["repulsion_energy"]
        print("{} particles={} triples={} onsite={} energy={}/{}".format(
            case["id"], case["particle_count"], case["triple_count"],
            case["onsite_triple_count"], energy["numerator"], energy["denominator"]))
    for limitation in report["limitations"]:
        print(limitation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
