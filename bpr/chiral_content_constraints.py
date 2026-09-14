"""Exact fixed-content anomaly constraints and a restricted sphere reduction.

Conventions are frozen in
``doc/derivations/chiral_content_constraints_2026-09-12.md``.  All species
are left-handed.  Quadratic fundamental indices are 1/2; the SU(3)
fundamental/antifundamental cubic coefficients are +1/-1.  Spectator
representation dimensions multiply traces.  SU(2) has no local cubic
invariant; traceless non-Abelian generators kill the omitted mixed traces.
There is no pure perturbative gravitational anomaly in four dimensions.

The reduction treats Spin(10) and tangent characteristic classes as pulled
back from a product base, while X = X_b + m eta, eta**2 = 0 and integral eta
= 1.  Its codomain consists of formal six-form anomaly-polynomial terms,
not literal nonzero six-forms on a physical four-dimensional base.  Nothing
here derives the supplied representations, a chirality mechanism, or a
positive number of families, nor supplies an anomaly-canceling parent.
"""

from fractions import Fraction
from math import gcd

try:
    import numpy as _np
except ImportError:
    _NUMPY_INTEGER_TYPES = ()
else:
    # Exact concrete scalar types only: do not invoke arbitrary __int__ hooks.
    _NUMPY_INTEGER_TYPES = tuple({
        _np.int8, _np.int16, _np.int32, _np.int64,
        _np.uint8, _np.uint16, _np.uint32, _np.uint64,
        _np.intp, _np.uintp, _np.longlong, _np.ulonglong,
    })


MAX_MULTIPLICITY = 1_000_000
MAX_COEFFICIENT_COMPONENT = 1_000_000
MAX_ABS_FLUX = 16

_COLUMNS = ("Q", "uc", "dc", "L", "ec")
_ROW_LABELS = (
    "SU3_cubic", "SU3_squared_Y", "SU2_squared_Y", "Y_cubic",
    "gravity_squared_Y",
)
_DOMAIN_BASIS = (
    "S4", "S2_squared", "S2_X_squared", "X_fourth", "p1_S2",
    "p1_X_squared", "p1_squared", "p2",
)
_CODOMAIN_BASIS = ("S2_Xb", "Xb_cubed", "p1_Xb")
_VISIBLE_COLUMNS = (2, 3, 5)

_MULTIPLICITY_SCOPE = (
    "Only the five displayed local anomalies of the supplied left-handed "
    "representations and integer SU2 Witten parity are tested. Full consistency "
    "here does not mean parent or all-global consistency. No positive family "
    "count, in particular three, is selected."
)
_REDUCTION_SCOPE = (
    "Restricted product-background sphere pushforward of formal anomaly "
    "polynomials, with Spin10 invariants and tangent classes pulled back "
    "and X carrying vertical flux. Integer-image steps and index describe "
    "the declared basis-dependent formal coefficient lattice, not a physical "
    "anomaly-quantization law. Flux bounds are computational only."
)

__all__ = [
    "MAX_MULTIPLICITY", "MAX_COEFFICIENT_COMPONENT", "MAX_ABS_FLUX",
    "anomaly_system", "multiplicity_report", "reduction_system",
    "pushforward", "parent_report", "demonstration_report",
]


def _integer(value, name):
    if type(value) is int or type(value) in _NUMPY_INTEGER_TYPES:
        return int(value)
    raise ValueError(name + " must be a builtin or NumPy integer, not bool")


def _singlet_flag(value):
    if type(value) is not bool:
        raise ValueError("include_singlet must be a builtin bool")
    return value


def _fixed_sequence(value, length, name):
    # Shape is checked before inspecting or coercing any element.
    if type(value) not in (list, tuple) or len(value) != length:
        raise ValueError(name + " must be a builtin list or tuple of length "
                         + str(length))
    return value


def _flux(value):
    result = _integer(value, "flux")
    if abs(result) > MAX_ABS_FLUX:
        raise ValueError("absolute flux must be at most " + str(MAX_ABS_FLUX))
    return result


def _coefficient(value):
    if type(value) is Fraction:
        result = value
    else:
        result = Fraction(_integer(value, "coefficient"))
    if (abs(result.numerator) > MAX_COEFFICIENT_COMPONENT
            or result.denominator > MAX_COEFFICIENT_COMPONENT):
        raise ValueError("coefficient numerator and denominator exceed bounds")
    return result


def _unit(size, position):
    return tuple(Fraction(int(i == position)) for i in range(size))


def _rref(matrix):
    """Bounded exact elimination on internal matrices of at most 8 by 8."""
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if not (1 <= rows <= 8 and 1 <= columns <= 8):
        raise ValueError("internal elimination shape must be within 8 by 8")
    if any(len(row) != columns for row in matrix):
        raise ValueError("internal elimination matrix must be rectangular")
    work = [[Fraction(entry) for entry in row] for row in matrix]
    pivots = []
    pivot_row = 0
    for column in range(columns):
        selected = next((i for i in range(pivot_row, rows)
                         if work[i][column]), None)
        if selected is None:
            continue
        work[pivot_row], work[selected] = work[selected], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [entry / scale for entry in work[pivot_row]]
        for i in range(rows):
            if i == pivot_row:
                continue
            factor = work[i][column]
            if factor:
                work[i] = [entry - factor * pivot
                           for entry, pivot in zip(work[i], work[pivot_row])]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return tuple(tuple(row) for row in work), tuple(pivots)


def _kernel(rref, pivot_columns):
    columns = len(rref[0])
    basis = []
    for free in range(columns):
        if free in pivot_columns:
            continue
        vector = list(_unit(columns, free))
        for row, pivot in enumerate(pivot_columns):
            vector[pivot] = -rref[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def _positive_primitive(vector):
    """Normalize an exact kernel vector to primitive integral coordinates."""
    denominator = 1
    for entry in vector:
        denominator = denominator * entry.denominator // gcd(
            denominator, entry.denominator)
    integers = [entry.numerator * (denominator // entry.denominator)
                for entry in vector]
    divisor = 0
    for entry in integers:
        divisor = gcd(divisor, abs(entry))
    if not divisor:
        raise ValueError("a primitive kernel generator cannot be zero")
    if next(entry for entry in integers if entry) < 0:
        divisor = -divisor
    return tuple(Fraction(entry // divisor) for entry in integers)


def _determinant(matrix):
    size = len(matrix)
    if not 1 <= size <= 8 or any(len(row) != size for row in matrix):
        raise ValueError("internal determinant requires a square matrix <= 8")
    work = [[Fraction(entry) for entry in row] for row in matrix]
    determinant = Fraction(1)
    for column in range(size):
        selected = next((i for i in range(column, size)
                         if work[i][column]), None)
        if selected is None:
            return Fraction(0)
        if selected != column:
            work[column], work[selected] = work[selected], work[column]
            determinant = -determinant
        pivot = work[column][column]
        determinant *= pivot
        for i in range(column + 1, size):
            factor = work[i][column] / pivot
            for j in range(column, size):
                work[i][j] -= factor * work[column][j]
    return determinant


def _anomaly_matrix(include_singlet):
    # Fixed (label, color dimension, weak dimension, color cubic trace,
    # color quadratic index, weak quadratic index, hypercharge).
    # Antifundamentals change the cubic sign but not the quadratic index.
    half = Fraction(1, 2)
    zero = Fraction(0)
    representations = (
        ("Q", 3, 2, Fraction(1), half, half, Fraction(1, 6)),
        ("uc", 3, 1, Fraction(-1), half, zero, Fraction(-2, 3)),
        ("dc", 3, 1, Fraction(-1), half, zero, Fraction(1, 3)),
        ("L", 1, 2, zero, zero, half, Fraction(-1, 2)),
        ("ec", 1, 1, zero, zero, zero, Fraction(1)),
    )
    if include_singlet:
        representations += (("nc", 1, 1, zero, zero, zero, zero),)
    columns = []
    for label, d3, d2, cubic3, index3, index2, charge in representations:
        columns.append((
            d2 * cubic3,
            d2 * index3 * charge,
            d3 * index2 * charge,
            d3 * d2 * charge ** 3,
            d3 * d2 * charge,
        ))
    return tuple(tuple(column[i] for column in columns)
                 for i in range(len(_ROW_LABELS)))


def anomaly_system(include_singlet=False):
    """Return immutable exact linear primitives for the full five local rows.

    The rational kernel and its nonnegative-integer interpretation are distinct:
    a charged primitive generator admits any nonnegative integer multiplier;
    nc, when included, has a separate unconstrained nonnegative multiplicity.
    Witten parity is not a rational anomaly row.
    """
    include_singlet = _singlet_flag(include_singlet)
    columns = _COLUMNS + (("nc",) if include_singlet else ())
    matrix = _anomaly_matrix(include_singlet)
    rref, pivots = _rref(matrix)
    kernel = tuple(_positive_primitive(vector)
                   for vector in _kernel(rref, pivots))
    minor_rows = (0, 1, 2, 4)
    minor_columns = (1, 2, 3, 4)
    minor = tuple(tuple(matrix[i][j] for j in minor_columns)
                  for i in minor_rows)
    return {
        "columns": columns,
        "row_labels": _ROW_LABELS,
        "matrix": matrix,
        "rank": len(pivots),
        "nullity": len(columns) - len(pivots),
        "rref": rref,
        "pivot_columns": pivots,
        "kernel_basis": kernel,
        "primitive_charged_generator": kernel[0],
        "rank_minor": {
            "rows": minor_rows, "columns": minor_columns, "matrix": minor,
        },
        "rank_minor_determinant": _determinant(minor),
        "row_dependence_coefficients": (
            Fraction(2, 9), Fraction(-4), Fraction(-3), Fraction(1),
        ),
        "exact_arithmetic": True,
    }


def multiplicity_report(multiplicities, include_singlet=False):
    """Evaluate local anomalies and integer Witten parity, without fitting."""
    include_singlet = _singlet_flag(include_singlet)
    length = 6 if include_singlet else 5
    supplied = _fixed_sequence(multiplicities, length, "multiplicities")
    counts = tuple(_integer(value, "multiplicity") for value in supplied)
    if any(value < 0 or value > MAX_MULTIPLICITY for value in counts):
        raise ValueError("multiplicities must be within 0.." + str(MAX_MULTIPLICITY))
    system = anomaly_system(include_singlet)
    anomalies = tuple(sum((entry * count for entry, count in zip(row, counts)),
                          Fraction(0)) for row in system["matrix"])
    local_free = not any(anomalies)
    doublet_count = 3 * counts[0] + counts[3]
    parity = doublet_count % 2
    witten_free = parity == 0
    equal = all(count == counts[0] for count in counts[:5])
    return {
        "columns": list(system["columns"]),
        "multiplicities": list(counts),
        "anomalies": dict(zip(_ROW_LABELS, map(str, anomalies))),
        "local_anomaly_free": local_free,
        "witten_doublet_count": doublet_count,
        "witten_parity": parity,
        "witten_anomaly_free": witten_free,
        "fully_consistent_with_tested_constraints": local_free and witten_free,
        "charged_multiplicities_equal": equal,
        "family_count": counts[0] if equal else None,
        "singlet_count": counts[5] if include_singlet else None,
        "scope": _MULTIPLICITY_SCOPE,
        "exact_arithmetic": True,
    }


def _reduction_matrix(flux):
    # Only one eta survives: X**2 -> 2m X_b; X**4 -> 4m X_b**3.
    return tuple(tuple(Fraction(multiplier * flux if column == visible else 0)
                       for column in range(8))
                 for visible, multiplier in zip(_VISIBLE_COLUMNS, (2, 4, 2)))


def reduction_system(flux):
    """Return rational image/kernel and the separate formal integer lattice."""
    flux = _flux(flux)
    matrix = _reduction_matrix(flux)
    rref, pivots = _rref(matrix)
    kernel = _kernel(rref, pivots)
    image = []
    for column in pivots:
        vector = tuple(row[column] for row in matrix)
        scale = next(entry for entry in vector if entry)
        image.append(tuple(entry / scale for entry in vector))
    steps = (abs(2 * flux), abs(4 * flux), abs(2 * flux))
    return {
        "flux": flux,
        "domain_basis": _DOMAIN_BASIS,
        "codomain_basis": _CODOMAIN_BASIS,
        "matrix": matrix,
        "rank": len(pivots),
        "nullity": len(_DOMAIN_BASIS) - len(pivots),
        "rref": rref,
        "pivot_columns": pivots,
        "kernel_basis": kernel,
        "image_basis": tuple(image),
        "integer_image_steps": steps,
        "integer_image_index": steps[0] * steps[1] * steps[2] if flux else None,
        "scope": _REDUCTION_SCOPE,
        "exact_arithmetic": True,
    }


def pushforward(coefficients, flux):
    """Map a bounded eight-component exact vector, validating also at m=0."""
    supplied = _fixed_sequence(coefficients, 8, "coefficients")
    vector = tuple(_coefficient(value) for value in supplied)
    flux = _flux(flux)
    matrix = _reduction_matrix(flux)
    return tuple(sum((entry * value for entry, value in zip(row, vector)),
                     Fraction(0)) for row in matrix)


def parent_report(flux):
    """Report the supplied Q=1, s=+1 parent and a fixed coordinate split.

    The split uses the nonzero-flux map at every flux. At zero flux its
    visible coordinates are NOT a complement to the enlarged kernel.
    The p2 coefficient remains a formal parent obstruction although p2 has
    no vertical degree-two contribution in the restricted product geometry.
    """
    flux = _flux(flux)
    charge, chirality = 1, 1
    parent = tuple(chirality * entry for entry in (
        Fraction(-1, 12), Fraction(1, 8), Fraction(charge ** 2),
        Fraction(2 * charge ** 4, 3), Fraction(-1, 12),
        Fraction(-charge ** 2, 3), Fraction(7, 360), Fraction(-1, 90),
    ))
    visible = tuple(entry if i in _VISIBLE_COLUMNS else Fraction(0)
                    for i, entry in enumerate(parent))
    invisible = tuple(entry - shown for entry, shown in zip(parent, visible))
    reduced = pushforward(parent, flux)
    invisible_image = pushforward(invisible, flux)
    return {
        "flux": flux,
        "charge": charge,
        "chirality": chirality,
        "domain_basis": list(_DOMAIN_BASIS),
        "codomain_basis": list(_CODOMAIN_BASIS),
        "parent_coefficients": list(map(str, parent)),
        "reduced_coefficients": list(map(str, reduced)),
        "visible_coefficients": list(map(str, visible)),
        "invisible_coefficients": list(map(str, invisible)),
        "invisible_pushforward": list(map(str, invisible_image)),
        "parent_nonzero": any(parent),
        "reduced_nonzero": any(reduced),
        "p2_coefficient": str(parent[7]),
        "split_convention": "nonzero_flux_coordinate_split",
        "restricted_reduction_certifies_parent": False,
        "exact_arithmetic": True,
    }


def _json_native(value):
    """Detach immutable exact system primitives without rational float casts."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def demonstration_report():
    """Return only the thirteen multiplicity and three fixed flux controls."""
    charged_cases = [(0, 0, 0, 0, 0)]
    charged_cases.extend((t, t, t, t, t) for t in (1, 2, 3))
    charged_cases.extend(tuple(int(i == j) for i in range(5)) for j in range(5))
    charged_cases.append((32, 1, 0, 0, 0))
    singlet_cases = (
        (1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 1, 2), (0, 0, 0, 0, 0, 1),
    )
    multiplicity_cases = [multiplicity_report(case) for case in charged_cases]
    multiplicity_cases.extend(multiplicity_report(case, True)
                              for case in singlet_cases)
    reductions = []
    for flux in (0, 1, -1):
        system = reduction_system(flux)
        reductions.append({
            "system": _json_native(system),
            "basis_images": [list(map(str, pushforward(_unit(8, i), flux)))
                             for i in range(8)],
            "kernel_images": [list(map(str, pushforward(vector, flux)))
                              for vector in system["kernel_basis"]],
        })
    return {
        "anomaly_systems": {
            "without_singlet": _json_native(anomaly_system()),
            "with_singlet": _json_native(anomaly_system(True)),
        },
        "multiplicity_cases": multiplicity_cases,
        "reduction_cases": reductions,
        "parent_cases": [parent_report(flux) for flux in (0, 1, -1)],
        "limitations": [
            _MULTIPLICITY_SCOPE,
            _REDUCTION_SCOPE,
            "The supplied representations, hypercharges, and physical "
            "spacetime chirality are not derived from the Bose ring.",
            "A neutral singlet has an independent multiplicity; it does not "
            "select or alter the charged family count.",
            "A vanishing restricted reduction cannot certify the lone "
            "nonzero parent. Invisible directions include p2. No completion "
            "mechanism or additional U1/global consistency is supplied.",
            "The zero-flux parent split retains the nonzero-flux coordinate "
            "convention and is not a complement to the zero-flux kernel.",
            "Inputs use fixed-length builtin lists or tuples. Integer scalars "
            "are builtin int or NumPy integer, excluding booleans; coefficients "
            "also accept Fraction. include_singlet is builtin bool only. "
            "Bounds are computational limits, not physical predictions.",
        ],
        "arithmetic_domain": {
            "kind": "exact_rational",
            "max_multiplicity": MAX_MULTIPLICITY,
            "max_coefficient_component": MAX_COEFFICIENT_COMPONENT,
            "max_abs_flux": MAX_ABS_FLUX,
            "rational_encoding": "canonical_fraction_string",
        },
    }
