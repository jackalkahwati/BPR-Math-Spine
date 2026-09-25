"""Common light cone for several emergent species: Bogoliubov-level criteria.

See doc/derivations/common_light_cone_2026-09-25.md. Two boson species on the
supplied cubic lattice (a proposed extension) condense uniformly; their
phonon speeds are the square roots of the eigenvalues of 2 D^(1/2) G D^(1/2),
D = diag(kappa_i), G_ij = mu_ij. This module checks that formula against
exact Bogoliubov-de Gennes blocks and records which symmetries do or do not
force a single cone. It is Bogoliubov-level mathematics, not a
thermodynamic-limit theorem or empirical validation.
"""

from itertools import product
from math import cos, isfinite, pi, sqrt

import numpy as np

MODEL_ID = "two-species-common-cone-v1"
TOLERANCE = 1e-10

LIMITATIONS = [
    "The second boson species is a proposed extension of the supplied cubic lattice.",
    "Results are at Bogoliubov (Hartree plus quadratic fluctuation) level.",
    "Radiative Lorentz-violation transmission is cited from the literature, not derived here.",
    "No mechanism forcing a universal limiting speed across species is supplied.",
]


def _finite(value):
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


def _matrix(G):
    G = np.asarray(G, dtype=float)
    if G.shape != (2, 2) or not np.allclose(G, G.T):
        raise ValueError("G must be a symmetric 2x2 matrix")
    return G


def miscible(G):
    """Uniform mixture is the Hartree minimizer (by convexity) when G is positive semidefinite."""
    G = _matrix(G)
    return bool(G[0, 0] >= 0 and G[1, 1] >= 0 and G[0, 1] ** 2 <= G[0, 0] * G[1, 1] + TOLERANCE)


def speed_matrix(kappa, G):
    """S = 2 D^(1/2) G D^(1/2); phonon speeds squared are its eigenvalues."""
    kappa = np.asarray(kappa, dtype=float)
    if kappa.shape != (2,) or np.any(kappa <= 0):
        raise ValueError("kappa must be two positive numbers")
    root = np.diag(np.sqrt(kappa))
    return 2 * root @ _matrix(G) @ root


def phonon_speeds(kappa, G):
    values = np.linalg.eigvalsh(speed_matrix(kappa, G))
    return [float(np.sqrt(v)) if v > 0 else 0.0 for v in values]


def common_cone(kappa, G):
    """True iff both speeds coincide, i.e. S is a multiple of the identity."""
    S = speed_matrix(kappa, G)
    scale = max(1.0, float(np.max(np.abs(S))))
    return bool(abs(S[0, 1]) < TOLERANCE * scale and abs(S[0, 0] - S[1, 1]) < TOLERANCE * scale)


def dispersion(k, C):
    return 2 * C * sum(1 - cos(kj) for kj in k)


def bdg_frequencies(k, C_pair, G):
    """Exact positive BdG frequencies at lattice momentum k (4x4 block, both species)."""
    G = _matrix(G)
    E = np.diag([dispersion(k, C_pair[0]), dispersion(k, C_pair[1])])
    A = E + G
    L = np.block([[A, G], [G, A]])
    sigma = np.diag([1.0, 1.0, -1.0, -1.0])
    w = np.linalg.eigvals(sigma @ L)
    if np.max(np.abs(w.imag)) > 1e-9:
        return None
    return sorted(float(x) for x in w.real if x > 1e-12)


def formula_frequencies(k, C_pair, G):
    """omega^2 = eig(E (E + 2G)) for real symmetric blocks A=E+G, B=G."""
    E = np.diag([dispersion(k, C_pair[0]), dispersion(k, C_pair[1])])
    values = np.linalg.eigvals(E @ (E + 2 * _matrix(G)))
    if np.max(np.abs(values.imag)) > 1e-9 or np.min(values.real) < -1e-12:
        return None
    return sorted(float(np.sqrt(max(v, 0.0))) for v in values.real)


def long_wave_speeds(C_pair, G, step=1e-4):
    """Numerical speeds from the exact BdG block along a generic small k direction."""
    direction = np.array([0.6, 0.48, 0.64])
    direction = direction / np.linalg.norm(direction)
    k = step * direction
    freqs = bdg_frequencies(k, C_pair, G)
    if freqs is None:
        return None
    return [f / step for f in freqs]


def lattice_scan(n, C_pair, G):
    """Compare exact BdG and formula frequencies at every nonzero mode of the n^3 lattice."""
    worst = 0.0
    unstable = 0
    for m in product(range(n), repeat=3):
        if not any(m):
            continue
        k = [2 * pi * x / n for x in m]
        exact = bdg_frequencies(k, C_pair, G)
        formula = formula_frequencies(k, C_pair, G)
        if exact is None or formula is None:
            unstable += 1
            continue
        worst = max(worst, max(abs(a - b) for a, b in zip(exact, formula)))
    return {"n": n, "max_formula_error": worst, "unstable_modes": unstable}


def z2_point(kappa, mu, mu_ab):
    """Symmetric point: speeds sqrt(2 kappa (mu +- mu_ab)); split at first order in mu_ab."""
    return sorted([sqrt(2 * kappa * (mu - mu_ab)) if mu >= mu_ab else 0.0,
                   sqrt(2 * kappa * (mu + mu_ab))])


def demonstration_report():
    cases = []
    specs = [
        ("decoupled_identical", (1.0, 1.0), [[1.0, 0.0], [0.0, 1.0]]),
        ("decoupled_tuned_unequal_hopping", (1.0, 2.0), [[2.0, 0.0], [0.0, 1.0]]),
        ("z2_symmetric_coupled", (1.0, 1.0), [[1.0, 0.3], [0.3, 1.0]]),
        ("su2_symmetric", (1.0, 1.0), [[1.0, 1.0], [1.0, 1.0]]),
        ("generic", (1.0, 1.3), [[1.0, 0.2], [0.2, 0.7]]),
        ("immiscible", (1.0, 1.0), [[1.0, 1.4], [1.4, 1.0]]),
    ]
    for name, C_pair, G in specs:
        kappa = list(C_pair)  # lattice units a=1
        entry = {"name": name, "C": list(C_pair), "G": [list(map(float, row)) for row in G],
                 "miscible": miscible(G), "common_cone": common_cone(kappa, G)}
        if miscible(G):
            entry["speeds_formula"] = phonon_speeds(kappa, G)
            numeric = long_wave_speeds(C_pair, G)
            entry["speeds_long_wave_numeric"] = numeric
        entry["lattice_scan_n8"] = lattice_scan(8, C_pair, G)
        cases.append(entry)
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "bogoliubov_level_common_cone_criteria",
        "empirical_validation": False,
        "cases": cases,
        "z2_splitting_example": {"kappa": 1.0, "mu": 1.0, "mu_ab": 0.1,
                                 "speeds": z2_point(1.0, 1.0, 0.1)},
        "collider_speed_difference_bound_order": 1e-11,
        "limitations": list(LIMITATIONS),
    }
    _finite(report)
    return report
