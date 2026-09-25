"""Controlled condensate regime of the supplied cubic Bose lattice.

Bounded witnesses for doc/derivations/cubic_condensate_regime_2026-09-25.md:
exact sector-vacuum properties (Perron-Frobenius), mean-field Bogoliubov
comparisons, exact acoustic-window inequalities, and current-carrying
condensate stability with its long-wave acoustic metric. The unchanged
lattice model with C>0, g>=0 is used; no field, constraint, chemical
potential or continuum limit is added. Finite floating checks are not
certified bounds, a thermodynamic limit, or empirical validation.
"""

from itertools import combinations_with_replacement, product
from math import comb, cos, factorial, isfinite, lgamma, log, exp, pi, sin, sqrt

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

MODEL_ID = "cubic-lattice-condensate-regime-v1"
# Computational allocation caps, not physical bounds.
MAX_SECTOR_DIMENSION = 7000
MAX_POSITION_DIMENSION = 4000
MAX_BDG_SITES = 512
DENSE_LIMIT = 600
# Heuristic floating allowance for classification; not a certified error bound.
TOLERANCE = 1e-10

LIMITATIONS = [
    "The cubic graph, couplings and particle number are supplied, not derived.",
    "Bogoliubov convergence is a mean-field limit at fixed lattice size, not a thermodynamic limit.",
    "The phonon is one scalar mode; no photons, chiral fermions or metric dynamics are supplied.",
    "The acoustic metric is kinematic and covers uniform flows only.",
    "The attractive tuned continuum model and its equilibrium gate are not addressed.",
    "Finite floating checks are not certified roundoff bounds or empirical validation.",
]


# ---------------------------------------------------------------------------
# Input validation and JSON hygiene
# ---------------------------------------------------------------------------

def _integer(value, name, lower, upper=None):
    if type(value) is not int:
        raise TypeError(name + " must be a builtin int")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(name + " outside supported range")
    return value


def _real(value, name, lower=None, strict=False):
    if type(value) not in (int, float):
        raise TypeError(name + " must be a builtin int or float")
    value = float(value)
    if not isfinite(value):
        raise ValueError(name + " must be finite")
    if lower is not None and (value < lower or (strict and value == lower)):
        raise ValueError(name + " outside supported range")
    return value


def _finite(value):
    """Reject nonfinite floats and non-JSON-native containers."""
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


# ---------------------------------------------------------------------------
# Lattice and one-body data
# ---------------------------------------------------------------------------

def _modes(n):
    return tuple(product(range(n), repeat=3))


def _wavevector(m, n):
    """Representative lattice momentum with components in (-pi, pi]."""
    out = []
    for mj in m:
        r = mj % n
        if 2 * r > n:
            r -= n
        out.append(2 * pi * r / n)
    return out


def dispersion(n, C=1.0):
    """epsilon_m = 2C sum_j (1 - cos k_j) for all modes, lexicographic order."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    return [2 * C * sum(1 - cos(2 * pi * mj / n) for mj in m) for m in _modes(n)]


def gap_constant(n, C=1.0):
    """epsilon_* = 4C sin^2(pi/n), the smallest nonzero one-body energy."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    return 4 * C * sin(pi / n) ** 2


def _tables(n):
    modes = _modes(n)
    index = {m: i for i, m in enumerate(modes)}
    M = len(modes)
    add = np.empty((M, M), dtype=np.int64)
    for i, a in enumerate(modes):
        for j, b in enumerate(modes):
            add[i, j] = index[tuple((a[t] + b[t]) % n for t in range(3))]
    neg = np.array([index[tuple((-a[t]) % n for t in range(3))] for a in modes], dtype=np.int64)
    return modes, index, add, neg


# ---------------------------------------------------------------------------
# Hartree functional and Bogoliubov data (Lemma 2, Theorem 4)
# ---------------------------------------------------------------------------

def _adjacency_apply(u, n):
    grid = np.asarray(u, dtype=np.complex128).reshape(n, n, n)
    out = np.zeros_like(grid)
    for axis in range(3):
        out += np.roll(grid, 1, axis=axis) + np.roll(grid, -1, axis=axis)
    return out.reshape(-1)


def hartree_energy(u, n, C=1.0, lam=0.0):
    """E_H[u] = <u,-CAu> + (lam/2) sum |u_x|^4 for a normalized site vector u."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    lam = _real(lam, "lam", 0.0)
    u = np.asarray(u, dtype=np.complex128)
    if u.shape != (n ** 3,):
        raise ValueError("u must have one entry per site")
    norm = float(np.vdot(u, u).real)
    if not norm > 0:
        raise ValueError("u must be nonzero")
    u = u / sqrt(norm)
    kinetic = -C * float(np.vdot(u, _adjacency_apply(u, n)).real)
    return kinetic + 0.5 * lam * float(np.sum(np.abs(u) ** 4))


def hartree_minimum(n, C=1.0, lam=0.0):
    """Lemma 2: e_H = -6C + lam/(2M), attained only by constant vectors."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    lam = _real(lam, "lam", 0.0)
    return -6 * C + lam / (2 * n ** 3)


def bogoliubov_modes(n, C=1.0, lam=0.0):
    """Exact Bogoliubov data e_m = sqrt(eps(eps+2mu)), mu = lam/M, for m != 0."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    lam = _real(lam, "lam", 0.0)
    M = n ** 3
    mu = lam / M
    eps = dispersion(n, C)
    records = []
    energy = 0.0
    depletion = 0.0
    for m, epsilon in zip(_modes(n), eps):
        if not any(m):
            continue
        e = sqrt(epsilon * (epsilon + 2 * mu))
        u2 = (epsilon + mu + e) / (2 * e)
        v2 = (epsilon + mu - e) / (2 * e)
        energy += -0.5 * (epsilon + mu - e)
        depletion += v2
        records.append({"m": list(m), "epsilon": epsilon, "e": e, "u2": u2, "v2": v2,
                        "residue_per_particle": epsilon / e})
    return {"n": n, "C": C, "lam": lam, "mu": mu, "E_B": energy,
            "bogoliubov_depletion": depletion, "modes": records}


# ---------------------------------------------------------------------------
# Position-basis sectors (Theorem 1)
# ---------------------------------------------------------------------------

def _occupations(sites, number):
    if sites == 1:
        yield (number,)
        return
    for first in range(number, -1, -1):
        for rest in _occupations(sites - 1, number - first):
            yield (first,) + rest


def _edges(n):
    sites = tuple(product(range(n), repeat=3))
    index = {s: i for i, s in enumerate(sites)}
    edges = set()
    for i, s in enumerate(sites):
        for axis in range(3):
            t = list(s)
            t[axis] = (t[axis] + 1) % n
            j = index[tuple(t)]
            edges.add((min(i, j), max(i, j)))
    return sites, index, tuple(sorted(edges))


def position_sector(n, N, g, C=1.0):
    """Complete N-boson occupation sector as a sparse symmetric matrix."""
    n = _integer(n, "n", 3)
    N = _integer(N, "N", 1)
    g = _real(g, "g", 0.0)
    C = _real(C, "C", 0.0, strict=True)
    M = n ** 3
    dimension = comb(M + N - 1, N)
    if dimension > MAX_POSITION_DIMENSION:
        raise ValueError("dimension exceeds the position-basis allocation cap")
    basis = list(_occupations(M, N))
    rows_of = {state: i for i, state in enumerate(basis)}
    _, _, edges = _edges(n)
    rows, cols, vals = [], [], []
    for col, state in enumerate(basis):
        diagonal = 0.5 * g * sum(v * (v - 1) for v in state)
        rows.append(col)
        cols.append(col)
        vals.append(diagonal)
        for x, y in edges:
            for target, source in ((x, y), (y, x)):
                if not state[source]:
                    continue
                moved = list(state)
                moved[source] -= 1
                moved[target] += 1
                rows.append(rows_of[tuple(moved)])
                cols.append(col)
                vals.append(-C * sqrt(state[source] * (state[target] + 1)))
    H = sparse.csr_matrix((vals, (rows, cols)), shape=(dimension, dimension))
    return basis, H


def _lowest(H, count):
    dimension = H.shape[0]
    if dimension <= DENSE_LIMIT:
        w, v = np.linalg.eigh(H.toarray())
        return w[:count], v[:, :count]
    count = min(count, dimension - 1)
    w, v = sparse_linalg.eigsh(H, k=count, which="SA", tol=1e-13, maxiter=100000)
    order = np.argsort(w)
    return w[order], v[:, order]


def _site_permutation(n, mapping):
    sites = tuple(product(range(n), repeat=3))
    index = {s: i for i, s in enumerate(sites)}
    return [index[tuple(c % n for c in mapping(s))] for s in sites]


def _state_permutation(basis, site_map):
    rows_of = {state: i for i, state in enumerate(basis)}
    out = np.empty(len(basis), dtype=np.int64)
    for i, state in enumerate(basis):
        image = [0] * len(state)
        for x, v in enumerate(state):
            image[site_map[x]] = v
        out[i] = rows_of[tuple(image)]
    return out


def sector_vacuum_report(n, N, g, C=1.0):
    """Theorem 1 witnesses: simplicity, positivity, symmetry, condensate overlap."""
    basis, H = position_sector(n, N, g, C)
    w, v = _lowest(H, 2)
    ground = v[:, 0].real.copy()
    if ground.sum() < 0:
        ground = -ground
    M = n ** 3
    maps = {
        "translation_e1": lambda s: (s[0] + 1, s[1], s[2]),
        "inversion": lambda s: (-s[0], -s[1], -s[2]),
        "swap_12": lambda s: (s[1], s[0], s[2]),
        "reflection_3": lambda s: (s[0], s[1], -s[2]),
    }
    residuals = {}
    for name, mapping in maps.items():
        perm = _state_permutation(basis, _site_permutation(n, mapping))
        image = np.empty_like(ground)
        image[perm] = ground
        residuals[name] = float(np.max(np.abs(image - ground)))
    # Pure condensate coefficients sqrt(N!/prod nu!) M^(-N/2), via logs.
    log_norm = lgamma(N + 1) - N * log(M)
    condensate = np.array([exp(0.5 * (log_norm - sum(lgamma(k + 1) for k in state)))
                           for state in basis])
    overlap = float(condensate @ ground)
    product_energy = -6 * C * N + 0.5 * g * N * (N - 1) / M
    result = {
        "n": n, "N": N, "g": g, "C": C, "dimension": len(basis),
        "ground_energy": float(w[0]), "first_gap": float(w[1] - w[0]),
        "min_component": float(ground.min()), "max_component": float(ground.max()),
        "symmetry_residuals": residuals, "condensate_overlap": overlap,
        "product_state_energy": product_energy,
        "product_state_upper_bound_margin": product_energy - float(w[0]),
    }
    _finite(result)
    return result


# ---------------------------------------------------------------------------
# Momentum-basis sectors (Theorems 3-4, Corollary 1)
# ---------------------------------------------------------------------------

def momentum_sector(n, N, g, K, C=1.0):
    """Total-momentum-K block of H_N in the plane-wave occupation basis.

    States are sorted tuples of mode indices (lexicographic mode order).
    """
    n = _integer(n, "n", 3)
    N = _integer(N, "N", 1)
    g = _real(g, "g", 0.0)
    C = _real(C, "C", 0.0, strict=True)
    modes, index, add, neg = _tables(n)
    M = len(modes)
    if type(K) is tuple:
        K = index[tuple(k % n for k in K)]
    K = _integer(K, "K", 0, M - 1)
    total = comb(M + N - 1, N)
    if total > 40 * MAX_SECTOR_DIMENSION * M:
        raise ValueError("enumeration exceeds the allocation cap")
    basis = []
    for combo in combinations_with_replacement(range(M), N):
        momentum = 0
        for c in combo:
            momentum = add[momentum, c]
        if momentum == K:
            basis.append(combo)
    if len(basis) > MAX_SECTOR_DIMENSION:
        raise ValueError("dimension exceeds the momentum-sector allocation cap")
    eps = dispersion(n, C)
    rows_of = {s: i for i, s in enumerate(basis)}
    rows, cols, vals = [], [], []
    coupling = g / (2 * M)
    for col, state in enumerate(basis):
        occ = {}
        for c in state:
            occ[c] = occ.get(c, 0) + 1
        rows.append(col)
        cols.append(col)
        vals.append(sum(eps[c] - 6 * C for c in state))
        if coupling == 0:
            continue
        keys = sorted(occ)
        for i1, k1 in enumerate(keys):
            for k2 in keys[i1:]:
                if k1 == k2:
                    if occ[k1] < 2:
                        continue
                    annihilate = sqrt(occ[k1] * (occ[k1] - 1))
                    multiplicity = 1.0
                else:
                    annihilate = sqrt(occ[k1] * occ[k2])
                    multiplicity = 2.0
                rest = dict(occ)
                rest[k1] -= 1
                rest[k2] -= 1
                for q in range(M):
                    p1 = int(add[k1, q])
                    p2 = int(add[k2, neg[q]])
                    created = dict(rest)
                    created[p2] = created.get(p2, 0) + 1
                    amplitude = sqrt(created[p2])
                    created[p1] = created.get(p1, 0) + 1
                    amplitude *= sqrt(created[p1])
                    target = tuple(sorted(c for c, m in created.items() for _ in range(m)))
                    rows.append(rows_of[target])
                    cols.append(col)
                    vals.append(coupling * multiplicity * annihilate * amplitude)
    H = sparse.csr_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)))
    return basis, H


def _raise_density(basis_from, basis_to, vector, q, add):
    """Apply rho_q^dagger = sum_k b_{k+q}^dagger b_k between momentum sectors."""
    rows_of = {s: i for i, s in enumerate(basis_to)}
    out = np.zeros(len(basis_to))
    for col, state in enumerate(basis_from):
        if vector[col] == 0:
            continue
        occ = {}
        for c in state:
            occ[c] = occ.get(c, 0) + 1
        for k, count in occ.items():
            moved = dict(occ)
            moved[k] -= 1
            target_mode = int(add[k, q])
            moved[target_mode] = moved.get(target_mode, 0) + 1
            amplitude = sqrt(count * moved[target_mode])
            target = tuple(sorted(c for c, m in moved.items() for _ in range(m)))
            out[rows_of[target]] += amplitude * vector[col]
    return out


def mean_field_case(N, lam, n=3, C=1.0, mode=(1, 0, 0)):
    """ED of H_N with g=lam/(N-1) against Bogoliubov predictions at one mode."""
    N = _integer(N, "N", 2)
    lam = _real(lam, "lam", 0.0)
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    modes, index, add, _ = _tables(n)
    M = len(modes)
    q = index[tuple(m % n for m in mode)]
    if q == 0:
        raise ValueError("mode must be nonzero")
    g = lam / (N - 1)
    mu = lam / M
    eps = dispersion(n, C)
    bog = bogoliubov_modes(n, C, lam)
    e_q = sqrt(eps[q] * (eps[q] + 2 * mu))
    e_H = hartree_minimum(n, C, lam)
    eps_star = gap_constant(n, C)

    basis0, H0 = momentum_sector(n, N, g, 0, C)
    w0, v0 = _lowest(H0, 1)
    omega = v0[:, 0].real.copy()
    E0 = float(w0[0])
    basis1, H1 = momentum_sector(n, N, g, q, C)
    w1, v1 = _lowest(H1, 1)
    phonon = v1[:, 0].real.copy()
    E1 = float(w1[0])

    occupations = np.zeros(M)
    for amplitude, state in zip(omega, basis0):
        weight = amplitude * amplitude
        for c in state:
            occupations[c] += weight
    depletion = float(N - occupations[0])

    raised = _raise_density(basis0, basis1, omega, q, add)
    residue = float(phonon @ raised) ** 2
    fsum_spectral = float(raised @ (H1 @ raised) - E0 * (raised @ raised))
    k_q = [2 * pi * m / n for m in modes[q]]
    fsum_kinetic = 0.0
    for j in range(3):
        kinetic_j = sum(2 * C * cos(2 * pi * modes[c][j] / n) * occupations[c] for c in range(M))
        fsum_kinetic += (1 - cos(k_q[j])) * float(kinetic_j)

    # Coercivity (Theorem 3) in the K=0 sector: G_N - eps_* N_+ + bound >= 0.
    plus = np.array([sum(1 for c in state if c != 0) for state in basis0], dtype=float)
    exact_bound = mu * N * (M - 1) / (2 * (N - 1))
    shifted = H0 - sparse.diags(N * e_H + eps_star * plus - exact_bound)
    coercive_min = float(_lowest(shifted.tocsr(), 1)[0][0])

    result = {
        "N": N, "lam": lam, "g": g, "n": n, "C": C, "mode": list(modes[q]),
        "sector_dimensions": [len(basis0), len(basis1)],
        "E0_minus_N_eH": E0 - N * e_H, "E_B": bog["E_B"],
        "energy_error": E0 - N * e_H - bog["E_B"],
        "gap": E1 - E0, "e_q": e_q, "gap_error": E1 - E0 - e_q,
        "depletion": depletion, "depletion_bound": mu * (M - 1) / eps_star,
        "bogoliubov_depletion": bog["bogoliubov_depletion"],
        "residue_per_particle": residue / N, "bogoliubov_residue": eps[q] / e_q,
        "fsum_spectral": fsum_spectral, "fsum_kinetic": fsum_kinetic,
        "fsum_residual": fsum_spectral - fsum_kinetic,
        "coercivity_min_eigenvalue": coercive_min,
    }
    _finite(result)
    return result


# ---------------------------------------------------------------------------
# Acoustic window (Theorem 5)
# ---------------------------------------------------------------------------

def acoustic_window(n, C=1.0, mu=1.0, a=1.0):
    """Check inequalities (4)-(6) for every nonzero mode of the n^3 lattice."""
    n = _integer(n, "n", 3)
    C = _real(C, "C", 0.0, strict=True)
    mu = _real(mu, "mu", 0.0, strict=True)
    a = _real(a, "a", 0.0, strict=True)
    kappa = C * a * a
    c2 = 2 * mu * kappa
    xi2 = kappa / (2 * mu)
    worst = {"eq4_lower": float("inf"), "eq4_upper": float("inf"),
             "eq5_lower": float("inf"), "eq5_upper": float("inf"), "eq6": float("inf")}
    max_deviation = 0.0
    lowest = None
    for m in _modes(n):
        if not any(m):
            continue
        k = _wavevector(m, n)
        q = [kj / a for kj in k]
        q2 = sum(x * x for x in q)
        q4 = sum(x ** 4 for x in q)
        epsilon = 4 * C * sum(sin(kj / 2) ** 2 for kj in k)
        e2 = epsilon * (epsilon + 2 * mu)
        b = kappa * a * a * q4 / 12
        diff = e2 - c2 * q2
        worst["eq4_lower"] = min(worst["eq4_lower"], diff + 2 * mu * b)
        worst["eq4_upper"] = min(worst["eq4_upper"], kappa * kappa * q2 * q2 - diff)
        ratio = e2 / (c2 * q2) - 1
        worst["eq5_lower"] = min(worst["eq5_lower"], ratio + a * a * q4 / (12 * q2))
        worst["eq5_upper"] = min(worst["eq5_upper"], xi2 * q2 - ratio)
        deviation = abs(sqrt(e2) / sqrt(c2 * q2) - 1)
        max_deviation = max(max_deviation, deviation)
        if m == (1, 0, 0):
            lowest = deviation
        worst["eq6"] = min(worst["eq6"], max(a * a / 12, xi2) * q2 - deviation)
    result = {"n": n, "C": C, "mu": mu, "a": a, "c_s": sqrt(c2), "healing_length": sqrt(xi2),
              "minimum_margins": worst, "max_relative_deviation": max_deviation,
              "lowest_mode_relative_deviation": lowest}
    _finite(result)
    return result


def lorentz_window_requirements(delta, C=1.0):
    """Smallest n and the mu(n) threshold making bound (6) <= delta at the lowest mode."""
    delta = _real(delta, "delta", 0.0, strict=True)
    C = _real(C, "C", 0.0, strict=True)
    n_min = int(np.ceil(pi / sqrt(3 * delta) - 1e-12))
    n_min = max(n_min, 3)
    return {"delta": delta, "n_min": n_min, "mu_min_at_n_min": 2 * pi * pi * C / (delta * n_min ** 2)}


# ---------------------------------------------------------------------------
# Current-carrying condensates (Theorems 6-7)
# ---------------------------------------------------------------------------

def _classify(value, scale):
    if value > TOLERANCE * scale:
        return 1
    if value < -TOLERANCE * scale:
        return -1
    return 0


def moving_condensate(n, m_p, mu, C=1.0):
    """Exact BdG frequencies (7), stability criteria and long-wave metric."""
    n = _integer(n, "n", 3)
    mu = _real(mu, "mu", 0.0)
    C = _real(C, "C", 0.0, strict=True)
    if type(m_p) not in (tuple, list) or len(m_p) != 3:
        raise TypeError("m_p must be a length-3 tuple or list of ints")
    m_p = tuple(_integer(x, "m_p component", -10 ** 6, 10 ** 6) % n for x in m_p)
    p = [2 * pi * x / n for x in m_p]
    scale = 1.0 + 12 * C + 2 * mu
    frequencies = []
    dynamical = "stable"
    energetic = "stable"
    for m in _modes(n):
        if not any(m):
            continue
        k = [2 * pi * x / n for x in m]
        S = 2 * C * sum(cos(p[j]) * (1 - cos(k[j])) for j in range(3))
        D = 2 * C * sum(sin(p[j]) * sin(k[j]) for j in range(3))
        disc = S * (S + 2 * mu)
        sign = _classify(disc, scale * scale)
        if sign < 0:
            dynamical = "unstable"
            root = 1j * sqrt(-disc)
        else:
            if sign == 0 and dynamical == "stable":
                dynamical = "marginal_unresolved"
            root = sqrt(max(disc, 0.0))
        frequencies.append((D + root, D - root))
        s_sign = _classify(S, scale)
        margin = _classify(disc - D * D, scale * scale)
        if s_sign < 0 or margin < 0:
            energetic = "unstable"
        elif (s_sign == 0 or margin == 0) and energetic == "stable":
            energetic = "marginal_unresolved"
    v = [2 * C * sin(x) for x in p]
    W = [2 * mu * C * cos(x) for x in p]
    w_signs = [_classify(x, scale) for x in W]
    if all(s > 0 for s in w_signs):
        signature = "lorentzian"
        s_value = sum(v[j] ** 2 / W[j] for j in range(3))
        g_lower = {"g00": -(1 - s_value), "g0i": [-v[j] / W[j] for j in range(3)],
                   "gij_diagonal": [1 / W[j] for j in range(3)]}
        ergoregion = "absent" if _classify(1 - s_value, 1.0) > 0 else (
            "present" if _classify(1 - s_value, 1.0) < 0 else "marginal_unresolved")
    elif any(s < 0 for s in w_signs):
        signature = "non_lorentzian"
        s_value = None
        g_lower = None
        ergoregion = "undefined"
    else:
        signature = "degenerate_unresolved"
        s_value = None
        g_lower = None
        ergoregion = "undefined"
    g_upper = [[-1.0] + [-x for x in v]]
    for i in range(3):
        g_upper.append([-v[i]] + [(W[i] if i == j else 0.0) - v[i] * v[j] for j in range(3)])
    real_parts = [[float(np.real(a)), float(np.real(b))] for a, b in frequencies]
    imag_parts = [[float(np.imag(a)), float(np.imag(b))] for a, b in frequencies]
    result = {
        "n": n, "m_p": list(m_p), "mu": mu, "C": C,
        "dynamical_stability": dynamical, "energetic_stability": energetic,
        "velocity": v, "W_diagonal": W, "signature": signature,
        "v_Winv_v": s_value, "ergoregion": ergoregion,
        "inverse_metric": g_upper, "metric": g_lower,
        "frequency_real_parts": real_parts, "frequency_imaginary_parts": imag_parts,
    }
    _finite(result)
    return result


def bdg_matrix(n, m_p, mu, C=1.0):
    """Full 2M x 2M site-basis BdG generator sigma_z L about the plane wave m_p."""
    n = _integer(n, "n", 3)
    mu = _real(mu, "mu", 0.0)
    C = _real(C, "C", 0.0, strict=True)
    M = n ** 3
    if M > MAX_BDG_SITES:
        raise ValueError("lattice exceeds the BdG allocation cap")
    sites = tuple(product(range(n), repeat=3))
    phase = np.array([np.exp(2j * pi * sum(a * b for a, b in zip(m_p, s)) / n) for s in sites])
    phi = phase / sqrt(M)
    A = np.zeros((M, M))
    _, _, edges = _edges(n)
    for x, y in edges:
        A[x, y] += 1
        A[y, x] += 1
    lam = mu * M
    eps_p = 2 * C * sum(1 - cos(2 * pi * x / n) for x in m_p)
    nu = eps_p - 6 * C + mu
    diag_block = -C * A + np.diag(2 * lam * np.abs(phi) ** 2 - nu)
    off = np.diag(lam * phi ** 2)
    L = np.block([[diag_block, off], [off.conj(), diag_block.conj()]])
    sigma = np.diag(np.concatenate([np.ones(M), -np.ones(M)]))
    return sigma @ L


# ---------------------------------------------------------------------------
# Fixed demonstration
# ---------------------------------------------------------------------------

def demonstration_report(populations=(2, 3, 4)):
    """Fixed controls; no fitted inputs. Populations may be extended to 5."""
    if type(populations) is not tuple or not populations:
        raise TypeError("populations must be a nonempty tuple of ints")
    for N in populations:
        _integer(N, "population", 2, 5)
    vacuum = [sector_vacuum_report(3, N, g) for N in (1, 2, 3) for g in (0, 1, 5)]
    ladder = [mean_field_case(N, lam) for lam in (5.0, 20.0) for N in populations]
    windows = [acoustic_window(n, 1.0, mu) for n in (3, 8, 16) for mu in (0.5, 4.0)]
    currents = []
    for n in (3, 5, 8):
        for m_p in dict.fromkeys(((0, 0, 0), (1, 0, 0), (1, 1, 0), (n // 2, 0, 0))):
            for mu in (0.1, 1.0, 4.0):
                case = moving_condensate(n, m_p, mu)
                currents.append({key: case[key] for key in (
                    "n", "m_p", "mu", "dynamical_stability", "energetic_stability",
                    "signature", "ergoregion", "v_Winv_v")})
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "mean_field_condensate_regime_demonstrator",
        "empirical_validation": False,
        "thermodynamic_limit": False,
        "controls": {"vacuum_lattice": 3, "vacuum_populations": [1, 2, 3],
                     "vacuum_couplings": [0, 1, 5], "ladder_lambdas": [5.0, 20.0],
                     "ladder_populations": list(populations), "ladder_mode": [1, 0, 0],
                     "window_lattices": [3, 8, 16], "window_mus": [0.5, 4.0],
                     "current_lattices": [3, 5, 8], "current_mus": [0.1, 1.0, 4.0],
                     "tolerance": TOLERANCE},
        "lorentz_window_example": lorentz_window_requirements(0.01),
        "sector_vacuum": vacuum,
        "mean_field_ladder": ladder,
        "acoustic_windows": windows,
        "moving_condensates": currents,
        "limitations": list(LIMITATIONS),
    }
    _finite(report)
    return report
