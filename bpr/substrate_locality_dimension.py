"""Exact ball-count diagnostics for bounded-fiber maps to a simple cycle.

The source in growth_report is an integer lattice neighborhood, not an
inferred physical geometry. Passing a count does not establish a map.
See doc/derivations/substrate_locality_dimension_2026-09-13.md.
"""

from itertools import product
from math import comb


MAX_INTEGER = 10**6
MAX_BOX_SITES = 4096


def _validate_integer(name, value, minimum, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            "{} must be a built-in int in {}..{}".format(name, minimum, maximum)
        )


def lattice_ball_count(d, r):
    """Count a radius-r Manhattan ball in the d-dimensional integer lattice."""
    _validate_integer("d", d, 1, 3)
    _validate_integer("r", r, 0)
    return sum(2**j * comb(d, j) * comb(r, j) for j in range(min(d, r) + 1))


def ring_ball_count(L, r):
    """Count a radius-r ball in the simple cycle with L vertices."""
    _validate_integer("L", L, 3)
    _validate_integer("r", r, 0)
    return min(L, 2 * r + 1)


def box_ball_count(shape, center, r):
    """Count the actual ball in a capped, open nearest-neighbor box."""
    if type(shape) not in (list, tuple) or type(center) not in (list, tuple):
        raise ValueError("shape and center must be built-in lists or tuples")
    if not 1 <= len(shape) <= 3 or len(center) != len(shape):
        raise ValueError("shape and center must have equal length in 1..3")
    _validate_integer("r", r, 0)

    sites = 1
    for side in shape:
        _validate_integer("side", side, 1)
        sites *= side
        if sites > MAX_BOX_SITES:
            raise ValueError("box must contain at most 4096 sites")
    for side, coordinate in zip(shape, center):
        _validate_integer("center coordinate", coordinate, 0, side - 1)

    # All shape, center, radius and product checks precede Cartesian iteration.
    return sum(
        1
        for point in product(*(range(side) for side in shape))
        if sum(abs(coordinate - origin) for coordinate, origin in zip(point, center))
        <= r
    )


def growth_report(d, r, L, K_num=1, K_den=1, multiplicity=1):
    """Report a necessary counting condition, never map existence."""
    _validate_integer("d", d, 1, 3)
    _validate_integer("r", r, 0)
    _validate_integer("L", L, 3)
    _validate_integer("K_num", K_num, 0)
    _validate_integer("K_den", K_den, 1)
    _validate_integer("multiplicity", multiplicity, 1)

    k = K_num // K_den
    source_count = lattice_ball_count(d, r)
    # k*r is an exact intermediate, not a ceiling-limited public radius input.
    target_count = min(L, 2 * k * r + 1)
    capacity = multiplicity * target_count
    signed_excess = source_count - capacity
    status = (
        "excluded_by_ball_count"
        if signed_excess > 0
        else "not_excluded_by_ball_count"
    )
    return {
        "source_kind": "integer_lattice_ball",
        "dimension": d,
        "radius": r,
        "target_sites": L,
        "dilation": {
            "numerator": K_num,
            "denominator": K_den,
            "effective_integer": k,
        },
        "multiplicity": multiplicity,
        "source_count": source_count,
        "target_ball_count": target_count,
        "capacity": capacity,
        "signed_excess": signed_excess,
        "status": status,
        "scope": {
            "map_existence_established": False,
            "physical_dimension_derived": False,
            "empirical_validation": False,
            "bounded_fiber_assumption": (
                "Each target vertex has at most multiplicity source preimages. "
                "This is an explicit graph-site encoding assumption."
            ),
        },
    }


def demonstration_report():
    """Return only the frozen 16 growth and 12 finite-box controls."""
    growth_cases = [
        growth_report(d, r, 31)
        for d in (1, 2, 3)
        for r in (0, 1, 2, 3)
    ]
    for case in (
        (3, 2, 31, 3, 2, 1),
        (3, 1, 31, 1, 2, 1),
        (1, 3, 4, 1, 1, 1),
        (1, 2, 9, 0, 1, 5),
    ):
        growth_cases.append(growth_report(*case))

    finite_box_controls = []
    for shape, center in (
        ((3, 3, 3), (1, 1, 1)),
        ((1, 1, 5), (0, 0, 2)),
        ((2, 2), (0, 0)),
    ):
        for r in (0, 1, 2, 3):
            finite_box_controls.append(
                {
                    "shape": list(shape),
                    "center": list(center),
                    "radius": r,
                    "count": box_ball_count(shape, center, r),
                }
            )

    return {
        "module": "substrate_locality_dimension",
        "growth_cases": growth_cases,
        "finite_box_controls": finite_box_controls,
        "limitations": [
            "Ball counting is necessary, not sufficient for a graph-site map. "
            "A nonexcluded case does not establish map existence or physical "
            "compatibility.",
            "Bounded fibers are an explicit encoding assumption, not a "
            "consequence of unrestricted Bose on-site Hilbert spaces. "
            "This is not a universal dimensional no-go theorem.",
            "Growth cases use integer lattice neighborhoods. Finite boxes "
            "require actual boundary-truncated counts unless the entire "
            "lattice ball lies inside the box.",
            "Global capacity |V(G)| <= multiplicity * L is independently "
            "necessary. A passing local ball count does not certify it.",
            "Physical dimension, Lorentz symmetry and time emergence are "
            "not derived. These fixed controls are counting diagnostics, "
            "not empirical validation or a replacement for proof.",
        ],
    }
