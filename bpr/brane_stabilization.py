"""Phase 1d: fixing the brane positions of BPR-6D-M with a vortex condensate.

See doc/derivations/brane_positions_2026-09-27.md. Tension-only codimension-2 branes on the flux sphere are classical
moduli (spherical metrics with cone points exist for any positions); bulk scalar exchange between like-sign brane
couplings attracts (collapse would make the Yukawas rank 1); one-loop Casimir forces are not computed. A dominant,
computable mechanism is needed.

Mechanism: a bulk SO(10)-singlet scalar chi of U(1)_F charge -4. In unit flux its lowest level is a spin-2 multiplet
(spin weight s = 2, m^2 r^2 = 2, next level 8), and every lowest-level configuration is (1 + |zeta|^2)^-2 times a
quartic polynomial in zeta, up to a gauge phase: exactly four zeros (vortices), forced by the flux topology. With a
bulk mass between -8/r^2 and -2/r^2, chi condenses in this level. For a type-II quartic (lambda above the critical,
BPS value of order q_chi^2, q_chi = (4/3) g4) the condensate minimizes int |chi|^4 at fixed int |chi|^2: the
tetrahedral state, whose zeros form a regular tetrahedron (at critical coupling the zeros are exact moduli, and below
it they coalesce). Brane couplings kappa |chi(z_a)|^2 with kappa > 0 pin branes to zeros.

The pinning is metastable, not established as a vacuum: kappa |chi|^2 vanishes for any assignment of branes to zeros,
including several branes on one zero (which attractive scalar exchange favours and which makes the Yukawas rank
deficient); the barrier to leave a zero is only |chi|^2 at an edge midpoint, 5/(16 pi) in normalized units; and
one-loop forces from the Higgs brane terms may be comparable.

Checks: the lowest level and the zero count; the global minimizer (exact energies 25/(84 pi), 5/(14 pi), 25/(36 pi));
its Hessian (three rotations; 10/(21 pi) twice, 100/(63 pi) three times); the pinning stiffness 5/(3 pi) and barrier
5/(16 pi); stacking; the residual A4 x Z4 of the chi-plus-geometry sector; the Yukawa consequence (fixed tetrahedral
positions reach only a proper subset of Yukawa pairs modulo U(3): a constraint); and the scale window.
"""

from itertools import combinations

import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize

try:
    from .minimal_model import spin_matrices, brane_matrix, _vec, _U3
    from .sphere_family_structure import swsh, eth, _is_zero, theta, phi
except ImportError:  # loaded as a top-level module by the demo script
    from minimal_model import spin_matrices, brane_matrix, _vec, _U3
    from sphere_family_structure import swsh, eth, _is_zero, theta, phi

MODEL_ID = "bpr6d-brane-stabilization-v2"
CHI_F_CHARGE = -4
SPIN = 2  # s = -F/2 in unit flux (fermions: F = 3 gives -3/2 from the monopole)

LIMITATIONS = [
    "chi is an added field, chosen because U(1)_F charge 4 forces exactly four vortices; it is not derived.",
    "The one-brane-per-vortex configuration is metastable at best: stacking several branes on one zero costs no "
    "pinning energy, the barrier is small (5/(16 pi) kappa (v r)^2 / r^4), and one-loop forces from the Higgs brane "
    "terms (estimated up to ~0.3 / r^4) are not computed; kill check 11 is conditional, not passed.",
    "The tetrahedral minimum needs a type-II quartic (lambda above the critical coupling); at critical coupling the "
    "vortex positions are exact moduli (Bradlow; Baptista-Manton 2003), below it they coalesce.",
    "The lowest-level (Abrikosov-like) description needs lambda (v r)^2 below the level gap and a nearly uniform flux; "
    "outside that regime vortex cores localize the flux and the zero-mode profiles change.",
    "One-loop Casimir forces between the branes are estimated only by dimensional analysis, not computed.",
    "Unequal brane tensions distort the round sphere at O(deficit), moving the vortices off the regular tetrahedron at "
    "that order.",
    "Before chi condenses (early universe) the positions are free; relaxation and possible domain walls between "
    "mirror configurations are not analysed.",
]


# ---------------------------------------------------------------------------
# 1. The lowest level of chi and its four zeros
# ---------------------------------------------------------------------------

def lowest_level_checks(s=SPIN):
    """eth annihilates the l = s harmonics; the level is m^2 r^2 = l (l+1) - s^2 = s with 2 s + 1 states."""
    ok = all(_is_zero(eth(swsh(sp.Integer(s), sp.Integer(s), sp.Integer(m)), s)) for m in range(-s, s + 1))
    return {"eth_annihilates": bool(ok), "level": s * (s + 1) - s * s, "states": 2 * s + 1}


_GRID = None
_HARMONICS = None


def _harmonics():
    """Lambdified lowest-level harmonics sY_{s,m}, basis order m = s, ..., -s (cached)."""
    global _HARMONICS
    if _HARMONICS is None:
        _HARMONICS = [sp.lambdify((theta, phi), swsh(sp.Integer(SPIN), sp.Integer(SPIN), sp.Integer(m)), "numpy")
                      for m in range(SPIN, -SPIN - 1, -1)]
    return _HARMONICS


def _quadrature():
    """Gauss-Legendre x uniform grid (16 x 32), exact for the polynomial degrees used here (cached)."""
    global _GRID
    if _GRID is None:
        n_theta, n_phi = 16, 32
        xs, ws = np.polynomial.legendre.leggauss(n_theta)
        th = np.arccos(xs)
        ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(th, ph, indexing="ij")
        W = np.outer(ws, np.full(n_phi, 2 * np.pi / n_phi))
        Y = np.array([np.asarray(f(T, P), dtype=complex) * np.ones_like(T) for f in _harmonics()])
        _GRID = (T, P, W, Y)
    return _GRID


def profile(c, t, p):
    """chi(theta, phi) = sum_m c_m sY_{s,m} (lowest level)."""
    return sum(ck * np.asarray(f(t, p), dtype=complex) for ck, f in zip(c, _harmonics()))


def quartic_energy(c):
    """int |chi|^4 dOmega for int |chi|^2 dOmega = 1 (exact Gauss quadrature for these polynomial degrees)."""
    c = np.asarray(c, complex)
    c = c / np.linalg.norm(c)
    _, _, W, Y = _quadrature()
    chi = np.tensordot(c, Y, axes=1)
    return float(np.sum(W * abs(chi) ** 4))


def minimize_quartic(starts=60, seed=0):
    """Global minimization over the lowest level; returns the best state and all local minimum values found."""
    rng = np.random.default_rng(seed)
    n = 2 * SPIN + 1
    best, values = None, []
    for _ in range(starts):
        r = minimize(lambda x: quartic_energy(x[:n] + 1j * x[n:]), rng.normal(size=2 * n), method="BFGS",
                     options={"gtol": 1e-12})
        values.append(r.fun)
        if best is None or r.fun < best.fun:
            best = r
    c = best.x[:n] + 1j * best.x[n:]
    return c / np.linalg.norm(c), values


def _polynomial_coefficients(c, t0=0.9, p0=0.4):
    """Up to a gauge phase and a positive factor, chi is a polynomial in zeta = tan(theta/2) e^{i phi}:
    sY_{s,m} = N_m cos^{2s}(theta/2) zeta^{s+m} e^{-i s phi}. N_m is read off at one sample point."""
    zeta = np.tan(t0 / 2) * np.exp(1j * p0)
    coeffs = np.zeros(2 * SPIN + 1, complex)  # ascending powers of zeta
    for k, (ck, f) in enumerate(zip(c, _harmonics())):
        m = SPIN - k
        N = complex(f(t0, p0)) / (np.cos(t0 / 2) ** (2 * SPIN) * zeta ** (SPIN + m) * np.exp(-1j * SPIN * p0))
        coeffs[SPIN + m] += ck * N
    return coeffs


def zeros(c):
    """The four zeros of the lowest-level profile, as unit vectors (roots of the quartic in zeta; a missing
    top coefficient means a zero at the south pole)."""
    coeffs = _polynomial_coefficients(c)
    roots = np.roots(coeffs[::-1])
    pts = []
    for z in roots:
        th = 2 * np.arctan(abs(z))
        ph = np.angle(z)
        pts.append(np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    while len(pts) < 2 * SPIN:
        pts.append(np.array([0.0, 0.0, -1.0]))
    return pts


def normalized_density(c, point):
    """|chi|^2 at a unit vector for the profile normalized to int |chi|^2 dOmega = 1."""
    c = np.asarray(c, complex)
    _, _, W, Y = _quadrature()
    norm = np.sum(W * abs(np.tensordot(c, Y, axes=1)) ** 2)
    v = np.asarray(point, float) / np.linalg.norm(point)
    return float(abs(profile(c, np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0]))) ** 2 / norm)


def pinning_barrier(c):
    """Energy (per kappa, normalized) at an edge midpoint between two zeros: the saddle a pinned brane must cross to
    move to a neighbouring vortex. Exact value for the tetrahedral state: 5/(16 pi)."""
    pts = zeros(c)
    return normalized_density(c, pts[0] + pts[1])


def stacking_pinning_energy(c):
    """Pinning energy (per kappa) of four branes with two stacked on one zero and one zero left empty: zero, the same
    as one brane per zero. The pinning term does not prevent stacking."""
    pts = zeros(c)
    stacked = [pts[0], pts[0], pts[1], pts[2]]
    return float(sum(normalized_density(c, p) for p in stacked))


def tetrahedron_test(points):
    """Largest deviation of the pairwise dot products from -1/3 (regular tetrahedron)."""
    return float(max(abs(a @ b + 1 / 3) for a, b in combinations(points, 2)))


def alternative_configurations():
    """Quartic energy of other four-zero configurations, for comparison: all four zeros at one point (a
    coherent state), a square on the equator, and two double zeros at the poles."""
    n = 2 * SPIN + 1
    coherent = np.zeros(n, complex)
    coherent[0] = 1.0
    square = np.zeros(n, complex)
    square[0], square[-1] = 1.0, 1.0  # zeta^4 + const: four zeros on a circle
    poles = np.zeros(n, complex)
    poles[SPIN] = 1.0  # m = 0: double zeros at both poles
    return {"coherent (one quadruple zero)": quartic_energy(coherent), "equatorial square": quartic_energy(square),
            "double zeros at both poles": quartic_energy(poles)}


# ---------------------------------------------------------------------------
# 2. Stability
# ---------------------------------------------------------------------------

def quartic_hessian(c, h=1e-4):
    """Hessian of the normalized quartic energy on CP^4 at c (8 real directions orthogonal to c and i c).
    Expected: three zero modes (rotations, eaten by the family gauge bosons) and five positive modes."""
    c = np.asarray(c, complex) / np.linalg.norm(c)
    basis = np.linalg.svd(c.reshape(1, -1).conj())[2][1:].conj()

    def f(x):
        return quartic_energy(c + (x[:4] + 1j * x[4:]) @ basis)

    n = 8
    H = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            ea, eb = np.eye(n)[a] * h, np.eye(n)[b] * h
            H[a, b] = H[b, a] = (f(ea + eb) - f(ea - eb) - f(-ea + eb) + f(-ea - eb)) / (4 * h * h)
    return np.linalg.eigvalsh(H)


def pinning_stiffness(c, point, h=1e-4):
    """Eigenvalues of the Hessian of |chi|^2 (normalized chi) at a zero, in local orthonormal tangent coordinates."""
    c = np.asarray(c, complex) / np.linalg.norm(c)
    e1 = np.cross(point, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(point, e1)

    def f(d):
        v = point + d[0] * e1 + d[1] * e2
        v /= np.linalg.norm(v)
        return abs(profile(c, np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0]))) ** 2

    H = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            ea, eb = np.eye(2)[a] * h, np.eye(2)[b] * h
            H[a, b] = (f(ea + eb) - f(ea - eb) - f(-ea + eb) + f(-ea - eb)) / (4 * h * h)
    return np.linalg.eigvalsh(H)


# ---------------------------------------------------------------------------
# 3. Residual family symmetry
# ---------------------------------------------------------------------------

def tetrahedral_rotations(points):
    """The 12 rotations permuting the four points (identity, 8 three-fold, 3 two-fold)."""
    rots = [np.eye(3)]
    for v in points:
        for ang in (2 * np.pi / 3, 4 * np.pi / 3):
            rots.append(_rotation_matrix(v, ang))
    for a, b in [(0, 1), (0, 2), (0, 3)]:
        axis = points[a] + points[b]
        rots.append(_rotation_matrix(axis / np.linalg.norm(axis), np.pi))
    return rots


def _rotation_matrix(axis, ang):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return expm(ang * K)


def _spin_rep(j, R):
    """D^j(R) from the rotation vector of R."""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(R).as_rotvec()
    Jx, Jy, Jz = spin_matrices(j)
    return expm(-1j * (rv[0] * Jx + rv[1] * Jy + rv[2] * Jz))


def residual_family_group(c):
    """Elements (R, alpha) of SO(3)_iso x U(1)_F leaving chi invariant: D^2(R) c = e^{-i F_chi alpha} c.
    Their action on the families (spin 1, F = 3) is D^1(R) e^{3 i alpha}. Returns the order of that matrix group, the
    phases of the lifted rotations, and whether its quotient by central phases is A4 (order 12)."""
    c = np.asarray(c, complex) / np.linalg.norm(c)
    pts = zeros(c)
    mats, phases = [], []
    for R in tetrahedral_rotations(pts):
        w = _spin_rep(SPIN, R) @ c
        ph = np.vdot(c, w)
        if abs(abs(ph) - 1) > 1e-6:
            raise ArithmeticError("a tetrahedral rotation does not preserve the condensate up to phase")
        phases.append(float(np.angle(ph) / (2 * np.pi)))
        for k in range(abs(CHI_F_CHARGE)):
            alpha = (np.angle(ph) + 2 * np.pi * k) / (-CHI_F_CHARGE)  # e^{-i F alpha} ph = 1 with F = -4
            mats.append(_spin_rep(1, R) * np.exp(3j * alpha))
    unique = []
    for M in mats:
        if not any(np.allclose(M, U, atol=1e-8) for U in unique):
            unique.append(M)
    # quotient by central phases: distinct up to scalar multiples
    projective = []
    for M in unique:
        if not any(abs(abs(np.vdot(M.ravel(), U.ravel())) - 3) < 1e-8 for U in projective):
            projective.append(M)
    return {"order_on_families": len(unique), "projective_order": len(projective), "rotation_phases": phases,
            "families_irreducible": _irreducible(projective)}


def _irreducible(mats):
    """Burnside test: the matrices act irreducibly iff sum |tr|^2 / |G| = 1 (for the projective group, use |tr|)."""
    return bool(abs(sum(abs(np.trace(M)) ** 2 for M in mats) / len(mats) - 1) < 1e-8)


# ---------------------------------------------------------------------------
# 4. Yukawa consequence and scales
# ---------------------------------------------------------------------------

def tetrahedral_yukawa_rank(points, trials=3, seed=0):
    """Branes fixed at the four points; couplings c_a (10H) and d_a (126barH) free, plus U(3). Real rank of the map
    to (Y10, Y126) (target 24) and the span of the four brane matrices. Full rank only means the image contains an
    open set; it is a proper subset (reachability_cost), so fixed positions do constrain the Yukawas."""
    zs = [np.tan(np.arccos(np.clip(v[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(v[1], v[0])) for v in points]
    span = int(np.linalg.matrix_rank(np.array([_vec(brane_matrix(z)) for z in zs]), tol=1e-10))

    def G(x):
        c, d = x[0:4] + 1j * x[4:8], x[8:12] + 1j * x[12:16]
        Y1 = sum(c[a] * brane_matrix(zs[a]) for a in range(4))
        Y2 = sum(d[a] * brane_matrix(zs[a]) for a in range(4))
        U = expm(sum(t * g for t, g in zip(x[16:], _U3)))
        v = np.concatenate([_vec(U.T @ Y1 @ U), _vec(U.T @ Y2 @ U)])
        return np.concatenate([v.real, v.imag])

    rng = np.random.default_rng(seed)
    ranks = []
    for _ in range(trials):
        x = np.concatenate([rng.normal(size=16), np.zeros(9)])
        f0 = G(x)
        J = np.array([(G(x + 1e-6 * e) - f0) / 1e-6 for e in np.eye(25)]).T
        s = np.linalg.svd(J, compute_uv=False)
        ranks.append(int(np.sum(s > 1e-6 * s[0])))
    return {"span": span, "rank_mod_U3": max(ranks)}


def reachability_cost(Y10, Y126, points, starts=40, seed=0):
    """How far a target pair (Y10, Y126) is from the Yukawas of branes fixed at the four points, modulo U(3).

    The four brane matrices span W, of complex codimension 2 in the symmetric matrices, so a pair is reachable iff some
    U in U(3) puts both U^T Y U in W: 8 real conditions on U(3)/U(1). Returns the smallest normalized squared distance
    found (reachable pairs reach ~0; the minimum over U is a lower bound for any search, so a large value is robust).
    """
    from scipy.optimize import least_squares
    zs = [np.tan(np.arccos(np.clip(v[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(v[1], v[0])) for v in points]
    Q, _ = np.linalg.qr(np.array([brane_matrix(z).ravel() for z in zs]).T)

    def perp(M):
        v = M.ravel()
        return v - Q @ (Q.conj().T @ v)

    n1, n2 = np.linalg.norm(Y10), np.linalg.norm(Y126)

    def res(t):
        U = expm(sum(x * g for x, g in zip(t, _U3)))
        r = np.concatenate([perp(U.T @ Y10 @ U) / n1, perp(U.T @ Y126 @ U) / n2])
        return np.concatenate([r.real, r.imag])

    rng = np.random.default_rng(seed)
    return float(min(2 * least_squares(res, rng.normal(size=9) * 2).cost for _ in range(starts)))


def reachability_fraction(points, n_pairs=20, starts=40, seed=0, tol=1e-10):
    """Fraction of random complex-symmetric pairs reachable with branes fixed at the points."""
    rng = np.random.default_rng(seed)
    hits = 0
    for k in range(n_pairs):
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        hits += reachability_cost(A + A.T, B + B.T, points, starts=starts, seed=seed + k) < tol
    return hits / n_pairs


def type_ii(lam=1.0, g4=0.03):
    """Type-II (repulsive-vortex) condition lambda > lambda_c ~ q_chi^2 / 2, q_chi = (4/3) g4 (order of magnitude)."""
    q = 4 / 3 * g4
    return {"lambda_critical_estimate": q * q / 2, "type_II": bool(lam > q * q / 2)}


def scale_window(kappa=1.0, deficit=0.1, rM=3.5, g4=0.03, lam=1.0, n_dof=100, higgs_brane_force=0.3):
    """Dimensionless conditions for v r (chi vev times radius); lam is the quartic of the lowest-mode amplitude:
    - pinning beats one-loop forces with a factor-10 margin: kappa (v r)^2 > 10 x (one-loop force r^4), where the
      one-loop force is the Casimir estimate n_dof deficit^2 / (16 pi^2) alone, or that plus the Higgs brane-term
      estimate higgs_brane_force (~0.3, a crude estimate of the derivative brane terms);
    - flux nearly uniform: the U(1)_F mass sqrt(2) (4/3) g4 v below 1/r;
    - lowest-level regime: lam (v r)^2 below the level gap (next level m^2 r^2 = 8 vs 2, gap 6).
    Also the brane-modulus mass m r ~ sqrt(kappa x 5/(3 pi)) (v r) / (sqrt(deficit) (r M)^2) (order of magnitude)."""
    casimir = n_dof * deficit ** 2 / (16 * np.pi ** 2)
    v_max = min(1 / (np.sqrt(2) * g4 * 4 / 3), np.sqrt(6 / lam))
    v_min_casimir = np.sqrt(10 * casimir / kappa)
    v_min_all = np.sqrt(10 * (casimir + higgs_brane_force) / kappa)
    stiffness = 5 / (3 * np.pi)
    return {"vr_min_casimir_only": float(v_min_casimir), "vr_min_with_higgs_brane_terms": float(v_min_all),
            "vr_max": float(v_max), "window_open_casimir_only": bool(v_min_casimir < v_max),
            "window_open_with_higgs_brane_terms": bool(v_min_all < v_max),
            "flux_bound_vr": float(1 / (np.sqrt(2) * g4 * 4 / 3)),
            "modulus_mass_times_r_at_vr_1": float(np.sqrt(kappa * stiffness) / (np.sqrt(deficit) * rM ** 2))}


def _phase1_example_cost(points, starts=30):
    """The hierarchical pair of the Phase 1 note (Takagi values (1e-5, 3e-3, 1) and a generic 0.02 Y126, seed 5):
    realizable with free brane positions, but not with the positions fixed at the tetrahedron."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return reachability_cost(np.diag([1e-5, 3e-3, 1.0]).astype(complex), 0.02 * (A + A.T), points, starts=starts)


def demonstration_report():
    c, values = minimize_quartic()
    pts = zeros(c)
    group = residual_family_group(c)
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "status": "tetrahedral_vortex_pinning_metastable",
        "empirical_validation": False,
        "lowest_level": lowest_level_checks(),
        "quartic_minimum": min(values), "spread_of_local_minima": float(max(values) - min(values)),
        "alternatives": alternative_configurations(),
        "tetrahedron_deviation": tetrahedron_test(pts),
        "quartic_hessian": [float(x) for x in quartic_hessian(c)],
        "pinning_stiffness": [float(x) for x in pinning_stiffness(c, pts[0])],
        "residual_family_group": {k: v for k, v in group.items() if k != "rotation_phases"},
        "rotation_phases": sorted(set(round(x, 6) for x in group["rotation_phases"])),
        "pinning_barrier": pinning_barrier(c), "stacking_pinning_energy": stacking_pinning_energy(c),
        "tetrahedral_yukawa": tetrahedral_yukawa_rank(pts),
        "phase1_example_reachability_cost": _phase1_example_cost(pts),
        "type_II": type_ii(),
        "scale_window": scale_window(),
        "limitations": list(LIMITATIONS),
    }
