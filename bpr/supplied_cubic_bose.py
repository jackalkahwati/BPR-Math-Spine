"""Fixed supplied-cubic Bose witness, not a derived geometry or physical theory.

See doc/derivations/supplied_cubic_bose_2026-09-13.md for the conditional
proof, comparison phase, frozen controls and limitations. No ring APIs imported.
"""

from itertools import product
from math import comb, cos, isfinite, pi, sin, sqrt

import numpy as np

MAX_DIMENSION = 378


def _admit(dimension):
    if type(dimension) is not int or dimension < 1 or dimension > MAX_DIMENSION:
        raise ValueError("dimension exceeds the live allocation cap")


def _occupations(sites, number):
    if sites == 1:
        yield (number,)
    else:
        for first in range(number + 1):
            for rest in _occupations(sites - 1, number - first):
                yield (first,) + rest


def _graph():
    sites = tuple(product(range(3), repeat=3))
    indices = {site: i for i, site in enumerate(sites)}
    edges = set()
    for i, site in enumerate(sites):
        for axis in range(3):
            neighbor = list(site)
            neighbor[axis] = (neighbor[axis] + 1) % 3
            j = indices[tuple(neighbor)]
            edges.add((min(i, j), max(i, j)))
    return sites, tuple(sorted(edges))


def _finite(value):
    """Validate native JSON output without hiding structural programming errors."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("numerical failure: nonfinite computed output")
        return
    if type(value) is list:
        for item in value:
            _finite(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("report keys must be strings")
            _finite(item)
        return
    raise TypeError("report contains a non-JSON-native value")


def cubic_sector(N, g):
    """Fresh complete normalized sector on the fixed 27-site, C=1 graph."""
    if type(N) is not int or N not in (0, 1, 2):
        raise ValueError("N must be a built-in integer in {0, 1, 2}")
    if type(g) is not int or g not in (0, 1):
        raise ValueError("g must be a built-in integer in {0, 1}")
    dimension = comb(27 + N - 1, N)
    _admit(dimension)
    basis = list(_occupations(27, N))
    indices = {state: i for i, state in enumerate(basis)}
    _, edges = _graph()
    _admit(dimension)
    H = np.zeros((dimension, dimension), dtype=np.float64)
    for column, state in enumerate(basis):
        H[column, column] = g * sum(n * (n - 1) // 2 for n in state)
        for x, y in edges:
            for target, source in ((x, y), (y, x)):
                if not state[source]:
                    continue
                moved = list(state)
                moved[source] -= 1
                moved[target] += 1
                row = indices[tuple(moved)]
                H[row, column] -= sqrt(state[source] * (state[target] + 1))
    _admit(dimension)
    if not np.isfinite(H).all():
        raise ValueError("numerical failure: nonfinite sector matrix")
    return basis, H


def _fourier_diagnostics(basis, H):
    sites, _ = _graph()
    modes = tuple(product((-1, 0, 1), repeat=3))
    rows = {state: i for i, state in enumerate(basis)}
    _admit(27)
    F = np.empty((27, 27), dtype=np.complex128)
    for x, site in enumerate(sites):
        state = tuple(int(j == x) for j in range(27))
        row = rows[state]
        for column, mode in enumerate(modes):
            angle = 2 * pi * sum(m * s for m, s in zip(mode, site)) / 3
            F[row, column] = complex(cos(angle), sin(angle)) / sqrt(27)
    energies = [4 * sum(sin(pi * m / 3) ** 2 for m in mode) for mode in modes]
    _admit(27)
    K = H.copy()
    for i in range(27):
        K[i, i] += 6
    _admit(27)
    orthogonality = float(np.max(np.abs(F.conj().T @ F - np.eye(27))))
    _admit(27)
    eigenvectors = float(np.max(np.abs(K @ F - F * np.asarray(energies))))
    records = []
    for mode, epsilon in zip(modes, energies):
        q = [2 * pi * m / 3 for m in mode]
        continuum = sum(component ** 2 for component in q)
        bound = sum(component ** 4 for component in q) / 12
        error = continuum - epsilon
        phases = []
        for time in (0, 0.1):
            lattice_phase = complex(cos(time * epsilon), -sin(time * epsilon))
            continuum_phase = complex(cos(time * continuum), -sin(time * continuum))
            phase_error = abs(lattice_phase - continuum_phase)
            phase_bound = min(2, abs(time) * bound)
            phases.append({"time": time, "error": phase_error, "bound": phase_bound,
                           "signed_excess": phase_error - phase_bound})
        records.append({"m": list(mode), "lattice_energy": epsilon,
                        "continuum_energy": continuum, "generator_error": error,
                        "generator_bound": bound, "generator_lower_excess": -error,
                        "generator_upper_excess": error - bound, "phases": phases})
    result = {"orthogonality_residual": orthogonality,
              "eigenvector_residual": eigenvectors, "modes": records}
    _finite(result)
    return result


def demonstration_report():
    """Return detached raw diagnostics for exactly six predetermined controls."""
    sites, edges = _graph()
    degrees = [0] * len(sites)
    for x, y in edges:
        degrees[x] += 1
        degrees[y] += 1
    sectors = []
    fourier = None
    for N in (0, 1, 2):
        for g in (0, 1):
            basis, H = cubic_sector(N, g)
            dimension = len(basis)
            _admit(dimension)
            hermiticity = float(np.max(np.abs(H - H.T)))
            commutator = None
            if N == 2:
                x = sites.index((0, 0, 0))
                y = sites.index((1, 0, 0))
                doublon = tuple(2 * int(j == x) for j in range(27))
                split = tuple(int(j == x) + int(j == y) for j in range(27))
                d, s = basis.index(doublon), basis.index(split)
                commutator = float((H[s, s] - H[d, d]) * H[s, d])
            sectors.append({"N": N, "g": g, "dimension": dimension,
                            "hermiticity_residual": hermiticity,
                            "interaction_trace": float(np.trace(H)),
                            "doublons": sum(2 in state for state in basis),
                            "commutator_sd": commutator})
            if N == 1 and g == 0:
                fourier = _fourier_diagnostics(basis, H)
            del basis, H
    report = {
        "schema_version": 1,
        "status": "supplied_geometry_nonrelativistic_demonstrator",
        "empirical_validation": False,
        "derived_dimension": False,
        "controls": {"n": 3, "C": 1, "populations": [0, 1, 2], "couplings": [0, 1],
                     "ell": 3, "kappa": 1, "J": 1, "times": [0, 0.1],
                     "absolute_allowance": 1e-11},
        "graph": {"sites": len(sites), "edges": len(edges), "degrees": degrees},
        "sectors": sectors,
        "fourier": fourier,
        "limitations": [
            "Three-dimensional geometry is supplied, not derived.",
            "The continuum theorem is nonrelativistic and one-particle only.",
            "The empty vacuum is not the ground state of the unshifted model.",
            "No physical fermions, matter, gravity, calibration or empirical distinction is established.",
            "Finite floating checks are not certified roundoff bounds or empirical validation.",
        ],
    }
    _finite(report)
    return report
