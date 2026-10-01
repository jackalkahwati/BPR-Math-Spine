"""Phase 2b: next-to-leading corrections to the pinned flavour structure of BPR-6D.

See doc/derivations/kinetic_corrections_2026-10-01.md. Phase 2 found that with the branes pinned at the Phase 1d
tetrahedron the minimal 10 + 126bar model fails the charged-fermion data at leading order (best chi^2 found 123.5
with neutrinos, 73.26 for the charged sector alone; m_s about 3.6 times too large). This module asks which
next-to-leading corrections can repair that, and at what size.

The families are the holomorphic zero modes f_m = sqrt(3/(4 pi r^2)) (1, sqrt2 z, z^2)/(1 + |z|^2) of the flux sphere;
a brane at z_a sees psi(z_a) = sqrt(3/(4 pi r^2)) u_a^T psi with u_a = coherent_state(z_a). Two classes of correction
appear at relative order 1/(rM)^2.
1. Normalization corrections. Brane-localized fermion kinetic terms (BKTs, with or without derivatives) and the
   metric/flux distortion from unequal brane tensions change only the family kinetic matrix, psi^dag K psi. After
   canonical normalization every Yukawa matrix (H, F, hence M_u, M_d, M_e, M_D, M_R) transforms as Y -> A Y A^T with
   A = Kbar^{-1/2}. A BKT kappa_a/M^2 at brane a gives Kbar = 1 + sum_a eps_a u_a u_a^dag, eps_a = 3 kappa_a/(4 pi (rM)^2).
   The tetrahedral states form a tight frame, sum_a u_a u_a^dag = (4/3) 1, and any A4-symmetric K is a multiple of 1
   (Schur), so only brane-to-brane differences matter. An O(1) kinetic matrix erases the pinning constraint
   altogether (four points in CP^2 are projectively equivalent; kinetic_map_to_tetrahedron).
2. Vertex corrections. By the J_z rule (yukawa_mechanisms, Lemma 0) a fermion bilinear with one eth-derivative
   couples to ethbar^2 Phi: brane a then contributes c_a [P_a + delta_a (u_a w_a^T + w_a u_a^T)], w_a = D(R_a) e_0,
   with delta_a independent for 10_H and 126bar_H and of natural size ~ (1/(rM)^2) times O(1) eth factors.
   S_a = u_a w_a^T + w_a u_a^T is pure J = 2 like P_a but is not annihilated by the condensate, so the vertex relaxes
   the condensate condition of W; unbounded, P_a and S_a span all of J = 2, the free-position model of Phase 1.
   Brane displacements (from the condensate deformed by unequal tensions) are a third, geometric correction.

For each correction the module fits the pinned model with the correction bounded in size (homotopy over the bound,
random restarts, a tight polish), giving upper bounds on chi^2_min as a function of the bound.
"""

import numpy as np

try:
    from . import fermion_fit as ff
    from .minimal_model import coherent_state
except ImportError:  # pragma: no cover - script use
    import fermion_fit as ff
    from minimal_model import coherent_state

MODEL_ID = "bpr6d-kinetic-corrections-v1"

LIMITATIONS = [
    "Corrections included: BKTs, a general family kinetic matrix (which also covers derivative BKTs and the tension "
    "distortion), rigid brane displacements, and the one-derivative brane vertex. SO(10)-breaking (non-universal) "
    "brane terms, two-derivative vertices and loop corrections are not.",
    "Natural sizes are estimates with unit coefficients at the cutoff: eps_a = 3 kappa_a/(4 pi (rM)^2) with kappa ~ 1, "
    "vertex delta ~ (1/(rM)^2) x (1 to 2.2), brane displacements of order the deficit spread (~0.1). Strong-coupling "
    "(NDA) coefficients could be up to ~8 pi larger. The 6D background with four unequal branes is not solved.",
    "The scans are bounded local optimizations (homotopy over the bound, random restarts, tight polish, re-seeding); "
    "every value is an upper bound on the true chi^2_min at that bound, and some with-neutrino rows are rugged.",
    "Inputs, running and errors are those of Phase 2 (one-loop SM running, assumed GUT-scale errors).",
]


# ---------------------------------------------------------------------------
# 1. Zero modes, brane states and the kinetic matrix
# ---------------------------------------------------------------------------

def brane_states():
    """The four spin-1 coherent states u_a at the tetrahedral zeros (P_a = u_a u_a^T)."""
    _, pts = ff.tetrahedral_brane_matrices()
    zs = [np.tan(np.arccos(np.clip(v[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(v[1], v[0])) for v in pts]
    return [coherent_state(z) for z in zs]


def zero_mode_gram(n_theta=401, n_phi=400):
    """Gram matrix of (1, sqrt2 z, z^2)/(1 + |z|^2) with the measure 3/(4 pi) dA on the unit sphere (expected: 1),
    and the largest deviation of sum_m |f_m|^2 from 3/(4 pi) (expected: 0, i.e. the value at any point is u(z))."""
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    Z = np.tan(T / 2) * np.exp(1j * Ph)
    F = np.array([np.ones_like(Z), np.sqrt(2) * Z, Z ** 2]) / (1 + abs(Z) ** 2)
    w = np.sin(T) * (th[1] - th[0]) * (ph[1] - ph[0]) * 3 / (4 * np.pi)
    G = np.einsum("mij,nij,ij->mn", F.conj(), F, w)
    density = np.sum(abs(F) ** 2, axis=0)
    return G, float(np.max(abs(density - 1)))


def epsilon_from_bkt(kappa, rM):
    """A BKT (kappa/M^2) psibar i gamma.d psi at a brane gives eps = kappa * (3/(4 pi r^2)) / M^2."""
    return 3 * kappa / (4 * np.pi * rM ** 2)


def natural_epsilon_range(kappa=1.0):
    """eps for kappa over the Phase 1 control window rM in [3, rM_max(M_GUT)]."""
    try:
        from . import model_scales as ms
    except ImportError:  # pragma: no cover
        import model_scales as ms
    rM = ms.control_window(ms.unify_3221()["M_GUT"])["rM_range"]
    return {"rM_range": (float(rM[0]), float(rM[1])), "eps_range": (epsilon_from_bkt(kappa, rM[1]),
                                                                     epsilon_from_bkt(kappa, rM[0]))}


def bkt_kinetic_matrix(eps, u=None):
    """Kbar = conj(K) = 1 + sum_a eps_a u_a u_a^dag (the BKT kinetic matrix, complex conjugated)."""
    u = brane_states() if u is None else u
    return np.eye(3) + sum(e * np.outer(a, a.conj()) for e, a in zip(eps, u))


def _inverse_sqrt(K):
    w, V = np.linalg.eigh(K)
    return V @ np.diag(w ** -0.5) @ V.conj().T


def gell_mann():
    """Orthonormal basis (tr L_a L_b = delta_ab) of traceless Hermitian 3x3 matrices."""
    L = []
    for i in range(3):
        for j in range(i + 1, 3):
            E = np.zeros((3, 3), complex)
            E[i, j] = E[j, i] = 1
            L.append(E / np.sqrt(2))
            E = np.zeros((3, 3), complex)
            E[i, j], E[j, i] = -1j, 1j
            L.append(E / np.sqrt(2))
    L.append(np.diag([1, -1, 0]).astype(complex) / np.sqrt(2))
    L.append(np.diag([1, 1, -2]).astype(complex) / np.sqrt(6))
    return L


_GM = gell_mann()


def general_kinetic_matrix(t):
    """Kbar = exp(h), h = sum_k t_k L_k traceless Hermitian (the overall scale is absorbed by the couplings)."""
    h = sum(tk * L for tk, L in zip(t, _GM))
    w, V = np.linalg.eigh(h)
    return V @ np.diag(np.exp(w)) @ V.conj().T


def normalized(m, Kbar):
    """Canonically normalized mass matrices: H -> A H A^T, F -> A F A^T with A = Kbar^{-1/2}."""
    A = _inverse_sqrt(Kbar)
    H, F, r, s = A @ m["H"] @ A.T, A @ m["F"] @ A.T, m["r"], m["s"]
    return {"Mu": r * (H + s * F), "Md": H + F, "Me": H - 3 * F, "MD": r * (H - 3 * s * F), "F": F, "H": H,
            "r": r, "s": s}


def a4_average(K):
    """Average of rho(g) K rho(g)^dag over the A4 rotations of the tetrahedron acting on the family triplet."""
    try:
        from .brane_stabilization import tetrahedral_rotations
        from .minimal_model import spin_matrices
    except ImportError:  # pragma: no cover
        from brane_stabilization import tetrahedral_rotations
        from minimal_model import spin_matrices
    _, pts = ff.tetrahedral_brane_matrices()
    mats = _spin1_rotations(tetrahedral_rotations(pts), spin_matrices)
    return sum(D @ K @ D.conj().T for D in mats) / len(mats)


def _spin1_rotations(rotations, spin_matrices):
    """Spin-1 matrices (basis m = 1, 0, -1) of the given SO(3) rotations, from their axis-angle form."""
    from scipy.linalg import expm
    Jx, Jy, Jz = spin_matrices(1)
    out = []
    for R in rotations:
        R = np.asarray(R, float)
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-12:
            out.append(np.eye(3, dtype=complex))
            continue
        w, V = np.linalg.eig(R)  # the axis is the eigenvector with eigenvalue 1 (robust also at angle pi)
        axis = np.real(V[:, np.argmin(abs(w - 1))])
        axis /= np.linalg.norm(axis)
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rodrigues = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
        if not np.allclose(rodrigues, R, atol=1e-9):
            axis = -axis
        out.append(expm(-1j * angle * (axis[0] * Jx + axis[1] * Jy + axis[2] * Jz)))
    return out


# ---------------------------------------------------------------------------
# 2. Brane displacements
# ---------------------------------------------------------------------------

def _tangent_frames(pts):
    out = []
    for n in pts:
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = a - (a @ n) * n
        e1 /= np.linalg.norm(e1)
        out.append((n, e1, np.cross(n, e1)))
    return out


def displaced_brane_matrices(t):
    """Brane matrices with brane a moved along the geodesic of tangent vector (t[2a], t[2a+1]) (radians)."""
    _, pts = ff.tetrahedral_brane_matrices()
    P = []
    for k, (n, e1, e2) in enumerate(_tangent_frames(pts)):
        v = t[2 * k] * e1 + t[2 * k + 1] * e2
        ang = np.linalg.norm(v)
        m = n if ang < 1e-15 else np.cos(ang) * n + np.sin(ang) * v / ang
        z = np.tan(np.arccos(np.clip(m[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(m[1], m[0]))
        u = coherent_state(z)
        P.append(np.outer(u, u))
    return P


def _matrices_from(x, P):
    c = x[0:4] + 1j * x[4:8]
    d = x[8:12] + 1j * x[12:16]
    r, s = x[16] + 1j * x[17], x[18] + 1j * x[19]
    H = sum(ci * Pi for ci, Pi in zip(c, P))
    F = sum(di * Pi for di, Pi in zip(d, P))
    return {"Mu": r * (H + s * F), "Md": H + F, "Me": H - 3 * F, "MD": r * (H - 3 * s * F), "F": F, "H": H,
            "r": r, "s": s}


def derivative_states():
    """w_a = D(R_a) e_0: the zero-mode direction that is nonzero at brane a after one eth-derivative (m = 0 in the
    brane's own frame), with u_a = D(R_a) e_1 up to a phase. R_a carries the north pole to brane a."""
    try:
        from .minimal_model import wigner, basis_vector
    except ImportError:  # pragma: no cover
        from minimal_model import wigner, basis_vector
    _, pts = ff.tetrahedral_brane_matrices()
    out = []
    for n in pts:
        D = wigner(1, np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0]))
        out.append(D @ basis_vector(1, 0))
    return out


_VERTEX_BASIS = None


def _vertex_basis():
    """Cached brane matrices P_a and vertex matrices S_a = u_a w_a^T + w_a u_a^T."""
    global _VERTEX_BASIS
    if _VERTEX_BASIS is None:
        P, _ = ff.tetrahedral_brane_matrices()
        S = [np.outer(a, b) + np.outer(b, a) for a, b in zip(brane_states(), derivative_states())]
        _VERTEX_BASIS = (P, S)
    return _VERTEX_BASIS


def vertex_matrices(delta_H, delta_F, x):
    """Pinned model with the one-derivative brane vertex 16 (eth 16) (ethbar^2 Phi): brane a contributes
    c_a [P_a + delta^H_a S_a] to H and d_a [P_a + delta^F_a S_a] to F, S_a = u_a w_a^T + w_a u_a^T (rank 2)."""
    P, S = _vertex_basis()
    c = x[0:4] + 1j * x[4:8]
    d = x[8:12] + 1j * x[12:16]
    r, s_ = x[16] + 1j * x[17], x[18] + 1j * x[19]
    H = sum(ci * (Pi + di_ * Si) for ci, Pi, Si, di_ in zip(c, P, S, delta_H))
    F = sum(di * (Pi + dj * Si) for di, Pi, Si, dj in zip(d, P, S, delta_F))
    return {"Mu": r * (H + s_ * F), "Md": H + F, "Me": H - 3 * F, "MD": r * (H - 3 * s_ * F), "F": F, "H": H,
            "r": r, "s": s_}


def vertex_epsilon_estimate(rM, harmonic_factor=1.0):
    """Natural size of the vertex ratio delta: two extra derivatives at the cutoff, (1/(rM))^2, times an O(1) factor
    from the eth eigenvalues on the zero modes and on the l = 3 Higgs level (about sqrt2 sqrt10 / 2 = 2.24)."""
    return harmonic_factor / rM ** 2


# ---------------------------------------------------------------------------
# 3. Corrected pinned models and bounded fits
# ---------------------------------------------------------------------------

KINDS = {"bkt": 4, "general_K": 8, "displacement": 8, "combined": 16, "vertex": 16}  # parameters after the 20
# "vertex": (|delta^H_a|, arg delta^H_a, |delta^F_a|, arg delta^F_a), each modulus bounded by the bound.
# "combined": a general kinetic matrix and brane displacements together; its bound is a multiple of the natural
# sizes, |t_k| <= 0.05 * bound and displacement <= 0.1 * bound.
COMBINED_UNITS = (0.05, 0.1)


def corrected_matrices(kind, xf):
    """Mass matrices of the pinned model (xf[:20], layout of fermion_fit.pinned_matrices) with a correction."""
    if kind == "bkt":
        return normalized(ff.pinned_matrices(xf[:20]), bkt_kinetic_matrix(xf[20:24]))
    if kind == "general_K":
        return normalized(ff.pinned_matrices(xf[:20]), general_kinetic_matrix(xf[20:28]))
    if kind == "displacement":
        return _matrices_from(xf[:20], displaced_brane_matrices(xf[20:28]))
    if kind == "combined":
        return normalized(_matrices_from(xf[:20], displaced_brane_matrices(xf[28:36])), general_kinetic_matrix(xf[20:28]))
    if kind == "vertex":
        q = np.asarray(xf[20:36])
        return vertex_matrices(q[0:4] * np.exp(1j * q[4:8]), q[8:12] * np.exp(1j * q[12:16]), xf[:20])
    raise ValueError(kind)


def correction_size(kind, xf):
    """The size the bound constrains: bkt max |eps_a|; general_K max |t_k|; vertex max |delta_a|; displacement the
    largest box component times sqrt2 (an upper bound on the geodesic displacement); combined in natural units."""
    q = np.asarray(xf[20:20 + KINDS[kind]])
    if kind == "bkt":
        return float(np.max(abs(q)))
    if kind == "general_K":
        return float(np.max(abs(q)))
    if kind == "vertex":
        return float(max(np.max(abs(q[0:4])), np.max(abs(q[8:12]))))
    if kind == "combined":
        return max(float(np.max(abs(q[:8]))) / COMBINED_UNITS[0],
                   max(np.hypot(q[8 + 2 * k], q[9 + 2 * k]) for k in range(4)) / COMBINED_UNITS[1])
    return float(max(np.hypot(q[2 * k], q[2 * k + 1]) for k in range(4)))


def corrected_residuals(xf, kind, tg, with_nu=True):
    try:
        r = ff.pulls(ff.matrix_observables(corrected_matrices(kind, xf)), tg, with_nu=with_nu)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        r = np.full(17 if with_nu else 13, 1e3)
    return r if np.all(np.isfinite(r)) else np.full(len(r), 1e3)


def corrected_chi2(xf, kind, tg, with_nu=True):
    return float(np.sum(corrected_residuals(xf, kind, tg, with_nu) ** 2))


def _bounds(kind, bound):
    n = KINDS[kind]
    if kind == "vertex":
        lo = np.r_[np.full(20, -np.inf), np.zeros(4), np.full(4, -np.inf), np.zeros(4), np.full(4, -np.inf)]
        hi = np.r_[np.full(20, np.inf), np.full(4, bound), np.full(4, np.inf), np.full(4, bound), np.full(4, np.inf)]
        return lo, hi
    if kind == "combined":
        lim = np.r_[np.full(8, COMBINED_UNITS[0] * bound), np.full(8, COMBINED_UNITS[1] * bound / np.sqrt(2))]
    else:
        lim = np.full(n, bound / np.sqrt(2) if kind == "displacement" else bound)  # box per tangent component
    lo = np.r_[np.full(20, -np.inf), -lim]
    hi = np.r_[np.full(20, np.inf), lim]
    return lo, hi


def _random_correction(lo, hi, rng):
    """Uniform in the box; unbounded components are phases, drawn in [0, 2 pi)."""
    finite = np.isfinite(lo) & np.isfinite(hi)
    return np.where(finite, rng.uniform(np.where(finite, lo, 0), np.where(finite, hi, 1)), rng.uniform(0, 2 * np.pi, len(lo)))


def bounded_fit(kind, bound, tg, start, with_nu=True, restarts=10, seed=0, max_nfev=1500, polish=True):
    """chi^2_min with the correction parameters bounded, from `start` plus random restarts, then a tight-tolerance
    polish of the best point. Returns (chi^2, xf). The value is an upper bound on the true minimum at that bound."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(kind, bound)
    n = KINDS[kind]
    start = np.asarray(start, float)
    cands = [start] + [np.r_[start[:20] * np.exp(rng.normal(size=20) * rng.choice([0.05, 0.2, 0.5])),
                             _random_correction(lo[20:], hi[20:], rng)] for _ in range(restarts)]
    best = None
    for c0 in cands:
        c0 = np.clip(c0, lo + 1e-15 * (hi > lo), hi - 1e-15 * (hi > lo)) if bound > 0 else np.r_[c0[:20], np.zeros(n)]
        try:
            if bound == 0:
                sol = least_squares(lambda y: corrected_residuals(np.r_[y, np.zeros(n)], kind, tg, with_nu), c0[:20],
                                    max_nfev=max_nfev, x_scale="jac")
                x = np.r_[sol.x, np.zeros(n)]
            else:
                sol = least_squares(corrected_residuals, c0, args=(kind, tg, with_nu), bounds=(lo, hi),
                                    max_nfev=max_nfev, x_scale="jac")
                x = sol.x
        except (np.linalg.LinAlgError, ValueError):  # a degenerate start; skip it
            continue
        if best is None or 2 * sol.cost < best[0]:
            best = (2 * sol.cost, x)
    if polish and best is not None and bound > 0:
        try:
            sol = least_squares(corrected_residuals, np.clip(best[1], lo, hi), args=(kind, tg, with_nu),
                                bounds=(lo, hi), max_nfev=5000, x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if 2 * sol.cost < best[0]:
                best = (2 * sol.cost, sol.x)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return best


def within_bounds(kind, bound, xf, tol=1e-9):
    lo, hi = _bounds(kind, bound)
    q = np.asarray(xf)[20:]
    return bool(np.all(q >= lo[20:] - tol) and np.all(q <= hi[20:] + tol))


def homotopy_scan(kind, bounds, tg, start, with_nu=True, seed=0, restarts=10, callback=None):
    """chi^2_min along increasing bounds, each step started from the previous optimum (plus random restarts)."""
    x = np.r_[np.asarray(start, float), np.zeros(KINDS[kind])]
    rows = []
    for k, bound in enumerate(bounds):
        c, x = bounded_fit(kind, bound, tg, x, with_nu=with_nu, restarts=restarts, seed=seed + 1000 * k)
        rows.append((float(bound), float(c), x.tolist()))
        if callback is not None:
            callback(rows)
    return rows


# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------

# Scan results: upper bounds on chi^2_min at each bound, over all runs (homotopy seeds, re-seeded, reverse and
# polished fits), counting points found at smaller bounds; stored optima are recomputed by the tests.
SCANS = {
    "bkt": {
        "charged": [(0.0, 73.2621), (0.005, 72.511), (0.01, 71.7646), (0.02, 70.2861), (0.03, 68.8267), (0.05, 65.9635), (0.075, 62.4841), (0.1, 59.1047), (0.15, 52.6302), (0.2, 46.4933), (0.3, 35.0279), (0.5, 15.1156)],
        "nu": [(0.0, 120.4514), (0.005, 120.4514), (0.01, 120.4514), (0.02, 119.882), (0.03, 117.9042), (0.05, 116.3316), (0.075, 109.4784), (0.1, 106.1766), (0.15, 97.7554), (0.2, 93.461), (0.3, 78.4012), (0.5, 67.3633)],
    },
    "combined": {
        "charged": [(0.0, 73.2621), (0.5, 54.9018), (1.0, 40.3579), (2.0, 22.0692), (3.0, 12.484), (5.0, 8.0162)],
        "nu": [(0.0, 120.4514), (0.5, 110.3282), (1.0, 91.2521), (2.0, 78.4676), (3.0, 78.4676), (5.0, 21.1946)],
    },
    "displacement": {
        "charged": [(0.0, 73.2621), (0.01, 70.944), (0.02, 68.662), (0.05, 62.1177), (0.1, 51.9016), (0.15, 42.9816), (0.2, 35.1175), (0.3, 22.6706), (0.5, 12.3653)],
        "nu": [(0.0, 120.4514), (0.01, 120.4514), (0.02, 120.4514), (0.05, 114.8522), (0.1, 106.3285), (0.15, 102.0834), (0.2, 96.9878), (0.3, 94.2832), (0.5, 88.8448)],
    },
    "general_K": {
        "charged": [(0.0, 73.2621), (0.02, 66.0531), (0.05, 56.6713), (0.1, 43.8687), (0.2, 27.5257), (0.3, 15.9242), (0.5, 7.2477), (0.75, 1.5389), (1.0, 0.0), (1.0016, 0.0), (1.5, 0.0)],
        "nu": [(0.0, 120.4514), (0.02, 113.8311), (0.05, 105.4), (0.1, 93.0239), (0.2, 78.4443), (0.3, 61.4887), (0.5, 15.6294), (0.75, 2.9259), (1.0, 2.9256), (1.0016, 2.9256), (1.5, 2.9256)],
    },
    "vertex": {
        "charged": [(0.0, 73.2621), (0.01, 50.7421), (0.03, 27.6415), (0.06, 17.8172), (0.1, 11.6657), (0.15, 10.1479), (0.2, 0.5477), (0.3, 0.1147)],
        "nu": [(0.0, 120.4243), (0.01, 114.5411), (0.03, 109.8872), (0.045, 106.1342), (0.06, 106.1342), (0.08, 83.2806), (0.1, 50.7565), (0.15, 50.7565), (0.2, 32.1017), (0.3, 4.4649)],
    },
}
STORED_POINTS = {
    ("bkt", "charged", 0.03): (68.82665998168541, [-0.4111570077723412, 9.952210517563491e-05, -0.02673379733737749, 0.0001545738633569719, -0.8548418599255951, -0.0043539134319896195, -0.029203433884910618, 0.004717824253998768, 0.017608013481461703, 0.0010452535664981326, -0.02649952950600319, -0.0008308796221148289, 0.25180873373701734, -0.004094310306477165, -0.04206645351466271, 0.004375343062863933, 47.10038684927109, -44.49144954408653, -0.869162789200179, 0.134848322327487, -0.029999999999999995, 0.029999999999999995, -0.029999999999999995, 0.029999999999999995]),
    ("bkt", "charged", 0.5): (15.11560609169055, [-0.2923683545937464, 0.0004974848982074795, -0.004111797397267031, -0.00019992164451250458, -0.3897767120987457, -0.0022164078772262835, -0.0034221689240191817, 0.002427289404842134, 0.03644836522246013, 0.0018361929430984554, -0.02422757608568635, -0.0009149554060468972, 0.08860074372405416, -0.006347642121877499, -0.016852021766248403, 0.006573501370851851, 42.89179754617545, -48.178366023853556, -0.2932860824470201, -0.0454818591306889, -0.5000000000009999, 0.5000000000009999, -0.5000000000009999, 0.5000000000009999]),
    ("bkt", "nu", 0.03): (117.9042459581734, [-0.0032118400349680225, -0.6500798026163908, -0.09307466650305454, 0.0038499047300588393, -0.008292890164082177, -0.28917413511823403, -0.0017124357309743418, 0.008399091739852482, 0.001742210505993328, -0.09172358731075114, -0.0012587298893407281, -0.0016616577721383302, -0.0005652828273644312, 0.44694393946653554, -0.023271567882379933, 0.0006747412937006075, -19.114825845920432, 20.587610217425944, -0.28048435917914016, 4.348184527115681, 0.02999999999978811, -0.02999999999777973, -0.013455003324536096, 0.029999999999589886]),
    ("bkt", "nu", 0.5): (67.36332168261181, [-0.008680455699937376, -0.4492621546367509, -0.07875769572239658, 0.008786247379740665, -0.00724871099632366, -0.1311867341994408, 0.0029675444932254578, 0.006860768543154923, 0.0022412804911419427, -0.03391758417517658, 0.01284529957001453, -0.0020212680117699337, -0.0006943820036830581, 0.2606507717103496, -0.016586409558941813, 0.000747147602960386, 18.140912529630018, -20.776429256751765, 2.287768340342555, 3.3895962748462387, 0.5000000000009999, -0.5000000000009999, -0.08654043423458112, 0.5000000000009999]),
    ("combined", "charged", 1.0): (40.357903933525385, [-0.8975346909259801, 9.698380073207913e-05, -0.027077788832435116, -0.0005398796104808799, -0.4286602295325429, -0.003661833214762398, -0.005510505364735037, 0.0035298999674432015, 0.13611387126956967, 0.0016690964412091878, -0.044779091602841194, -0.002010833782398981, 0.21313074753633895, -0.004519880844093285, -0.025084267766536685, 0.004160722730664629, 43.22058402831927, -49.21865043257903, -0.6063277457067413, 0.14369777093200028, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, 0.07071067811865474, -0.0707106781186547, -0.07071067811865453, 0.07071067811865471, -0.07071067811865474, -0.07071067811865474, -0.07071067811865467, -0.07071067811865471]),
    ("combined", "charged", 5.0): (8.016249072296691, [-0.7220464122330943, -0.0002382688273233828, -0.013831537631439475, -0.00019992352960121097, -0.4065812859188168, -0.002229489901852554, -0.004616015211543634, 0.002257388492357375, 0.09168752645751382, 0.0024546994363400195, -0.003395017033661212, -0.003121493437986576, 0.09716572099491558, -0.00521793273274361, -0.06865871123178469, 0.00466574091237588, 24.406320827920297, -53.75283275574712, -0.18318919518323315, 0.1252249982057744, 0.24999005729801974, 0.24649428945318308, 0.24999928953363684, -0.2294699709740839, -0.17375269975887758, 0.24999801123696896, 0.2182148347766433, 0.24999997892402473, 0.288521306745082, -0.23787876279801132, -0.35355302424595625, -0.0364726587613763, -0.19402930486501765, -0.3535530837447339, -0.35355251809153226, -0.35333874543496485]),
    ("combined", "nu", 1.0): (91.25211163262858, [-0.002947480260094734, -0.2711830812174838, -0.08450308530905792, 0.003793408412360562, -0.008747763784116716, -0.6202201557455254, -0.018172791659924165, 0.008558283600118934, 0.0022128455586624797, -0.4600779753242804, 0.00937175228701011, -0.002119396964032594, -0.00012613125042916057, 0.22362069841991664, -0.02358964865614348, 0.00028207215128933237, -24.706722152311816, -12.378024166805453, 0.60845367442924, 3.759961238674794, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, 0.049999999999999996, -0.04468923130266562, 0.00022224907833262602, -0.004489423755980028, 0.07071067415523181, 0.027608561557079934, 0.07071063632228133, -0.021618282428786526, 0.009642231124687413, 0.0065112149160276795]),
    ("combined", "nu", 5.0): (21.19456843810803, [-0.0041098705828143305, 0.0012796083046746323, -0.0019621220881005257, 0.0621188621648231, -0.0086209954857295, 0.0015168977451669708, -0.0008067957708926988, -0.35213159853215803, 0.0181054566666702, -0.005542516541248954, 0.009898717508438003, 0.37548346671063504, 0.08750272266181403, -0.009295342029071108, 0.005628898902753262, -0.06528630170232555, 81.92022622979883, 99.9672966326184, 0.1684923569333967, -0.002455009625900602, 0.24999880808650435, 0.24999999999999997, 0.24999999999774197, -0.24999868987972912, -0.24999874176424103, -0.06775677955992963, -0.24999999999999997, -0.24999986779330072, 0.0763620463590076, 0.3317510134434963, 0.06907307344779624, 0.3534706670252442, 0.18226473031855928, -0.3535532311945578, -0.3535533888186222, -0.3535533890399105]),
    ("displacement", "charged", 0.1): (51.90156479496853, [-0.843735124885782, 9.762927094600584e-05, -0.03201335087937307, -0.0005952028141039204, -0.5325339561921182, -0.0039417731888747405, -0.011577001511258856, 0.003847884932772657, 0.11982055847672093, 0.0015580966243127757, -0.04083053616607977, -0.001904798393165224, 0.23674249484014862, -0.004088117909014304, -0.0313339309027608, 0.003796264136049016, 42.41292788878898, -49.51028037872532, -0.7198272410882822, 0.18654284232270588, 0.07071067811865474, -0.07071067811865474, -0.07071067811865472, 0.07071067811865472, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474]),
    ("displacement", "charged", 0.5): (12.365294143381476, [-1.1032148748652308, 0.000342852367444588, -0.018675546237891964, -0.0006683234951749976, 0.6061226516577811, -0.0026589175080968717, -0.008412645700636389, 0.0020400387135286265, 0.18844338471181396, 0.0027095706472562466, -0.035830075669673785, -0.0026719573230112844, -0.04336455616657421, -0.005056906384006506, -0.07022813556181534, 0.003678025272836093, 47.0428255987123, -33.29874798236517, -0.30043092075616695, 0.0960372430761852, 0.3534768111207712, -0.35354399732183694, -0.3283202463590397, 0.12785183955186014, -0.27026944338790454, -0.28321775335578075, -0.35354488319803556, -0.3535530226282814]),
    ("displacement", "nu", 0.1): (106.3285371096671, [-0.0004062257637194361, -0.1991062883691121, -0.07472590896990249, 0.0008393033893322877, -0.008366392337320352, -0.5746098406552382, -0.04039876296955601, 0.008746411503246019, 0.0022919692239314815, -0.49766003984220025, 0.012975255966943579, -0.002270272835786525, 0.00020962098192186922, 0.18834362658721826, -0.02481350332874917, -0.00012802182135438248, 15.077275128179211, -25.20259270222754, 0.029505953196590724, 3.363447582917489, 0.012749821429435187, 0.03283165560153021, 0.07071067637883645, 0.039741144169447386, 0.06348332628135948, -0.04571878650336852, 0.015084714670996925, 0.041540041383525864]),
    ("displacement", "nu", 0.5): (88.84475777330084, [-0.001203218185366159, -0.26082088568534834, -0.08235797634325784, 0.0016411303112662937, -0.00876093612568783, -0.5714636305547671, -0.02988485286362382, 0.009085769062350655, 0.002292242622414246, -0.5257877524692752, 0.012978820827413622, -0.0022705607611454705, 0.00020322145964670545, 0.18020151366825427, -0.02481962415804366, -0.00012649958896080346, 13.78526972621705, -24.44955284555854, 0.45426395157227767, 3.4794118562218874, 4.3832267939521796e-09, 0.034600735139312735, 0.14783345225595193, 0.09038307943748405, 0.13792081945973148, -0.08206935791729564, 0.014892078991677606, 0.09098728678176207]),
    ("general_K", "charged", 0.05): (56.67132371892599, [0.8502305841361933, -0.005590208458052485, 0.005965952411973635, 0.02196789753125177, 0.30651516732275624, 0.0022372332901594, -0.0024369536538127023, 0.052483276278680824, -0.2856404246586399, -0.0007716887163801877, 0.0007820517943776723, 0.040050862186929104, 0.23244652206860897, 0.0038399631049598182, -0.004024951483960151, -0.000526108411682181, 38.303248344175195, -37.781793170671364, -0.651604968202035, -1.3739837396084635, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996]),
    ("general_K", "charged", 1.5): (1.5251171483724885e-21, [-0.31925332029944503, -0.003641828375677405, 0.007386903860786715, -0.025593160191193055, 0.32865422762082974, -0.015571226218063967, 0.009655305629513186, -0.015544798870976541, -0.019588117051220953, 0.003450884810702152, -0.011484011976194032, 0.041898778879069405, -0.06761272887072817, 0.03484460279989336, -0.023086990938717265, 0.04193682564972771, 40.46944443944174, 60.62176903777471, 0.5076680482258259, -0.06582747597505403, 0.9999999935763471, -0.9999998358071781, -0.9999999994126632, -0.5547342593452214, 0.9999999987651536, -0.4209497500310352, 0.38768612421840937, -0.9999999418272851]),
    ("general_K", "nu", 0.05): (105.40001685306432, [-0.0025093615118190267, -0.20750479260600185, -0.07934390785237865, 0.0033635770475474096, -0.008294159514793465, -0.632911198705536, -0.025185529254439842, 0.00816348856506018, 0.0022125680050903147, -0.427593846804155, 0.009370732414563465, -0.0021193392670595225, -0.0001264110444636601, 0.23065899241381418, -0.023595977344689482, 0.0002819807010980456, -25.31636672516068, -14.140500188920138, 0.3318491186507578, 3.6264597492128168, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, 0.049999999999999996, -0.028417678065622582]),
    ("general_K", "nu", 1.5): (2.9256186299353946, [-0.19945548618956044, -0.003309692059455318, 0.004453326628690393, -0.01674102321761442, 0.19950960200485285, -0.0070626676411898025, 0.004791278082774428, 0.0007988501784575635, -0.0688797419368567, 0.00707203877968041, -0.01083033645818292, 0.04521953177438651, -0.0816704143125402, 0.020824354865663596, -0.014135225480152454, 0.023813708003973578, 65.82787702005069, 11.38142144938168, 0.32293437018713106, -0.01668903913023402, 0.9808405939362795, 0.9922042460334618, 0.7642173758468478, -0.9968811548858021, 0.36991087705894476, 0.22251347461183316, -0.9988669577191199, -0.7718459163461228]),
    ("vertex", "charged", 0.1): (11.665717882448831, [1.1442340355823606, -0.0016417500884799406, 0.007055825269322147, -0.00352827562802839, 0.04340500981832285, -0.006087371852472851, -0.014991803842205432, 0.0021695118564716014, -0.19208344869418673, -0.0011028846927827723, 0.039987904085408726, -0.001421850100635096, -0.005480452802794178, -0.007973441070609467, 0.0011751327044905982, 0.006278977024462436, 50.68271104555427, 41.26481921480346, -0.12224931531421288, 0.5304092840884, 0.09999998024952945, 0.09641923698414138, 0.08280129758217615, 0.0968009190348907, -0.7064500276959049, -1.2396571123976745, -0.0694409411701965, -2.4707352163995644, 0.051797120721126105, 0.09990973933575896, 0.09999999999999976, 0.09833092151747352, 1.1532231396126278, 2.21260860861899, -1.0191267695587667, 2.449788395956366]),
    ("vertex", "charged", 0.2): (0.547705023400488, [1.0392414998779607, -0.00029913685879607373, -0.006767845965908728, -0.005647443107483905, -0.4499355645492666, -0.006875266255715149, 0.0015002754952679227, -0.0013300966292777612, -0.2093693263184182, -0.001544011990312291, 0.020355118043327083, -0.000774152783335022, -0.1752973682032787, -0.0067353841729520724, -0.007117547880306822, 0.0072698115919840696, 26.36639646194061, -59.55378912520848, 0.5914694952739374, 0.4935922198011214, 0.1405487835464572, 0.1999998896803558, 0.19999923397188316, 0.18075363853678128, 2.5438371396299, 3.043811656029747, -1.4260694844304638, 0.5683913260397575, 0.19999999999999948, 0.1750421365593462, 0.14089635745496495, 0.19496384556388524, 0.32612372634354314, -1.882265760336007, -1.9309594680922821, -0.16540518954004946]),
    ("vertex", "charged", 0.3): (0.11470705823616156, [1.062189634464915, -0.0011883841465084564, -0.006082497448549085, -0.006124212838203023, -0.4667445360687688, -0.00814681057181653, -0.00046352606438690945, -0.0011603948239577151, -0.1975619227465378, -0.0009427539738708378, 0.022316736421916197, -0.0008116502750204843, -0.10613967762466937, -0.006528092529778947, -0.004693935646982354, 0.007287634531538772, 48.88256288153071, -46.14113468298951, 0.5934197808676899, 0.4216191925988835, 0.13566674125421788, 0.299971420059767, 0.2999757010156015, 0.2655000874358477, 2.4084408188245514, 2.9542100929360324, -0.8678741165370982, 0.38748275599661247, 0.2999999998129108, 0.23463132588348956, 0.1886160614299583, 0.2928448053616477, 0.3007803422709447, -2.1081686919873275, -1.9153679373278971, -0.15401168839296714]),
    ("vertex", "nu", 0.1): (50.75646824835134, [1.1272497482673582, -0.0009698226167540587, 0.017701744006349922, -0.007098558067118218, 0.09424789290080533, -0.00835944022822278, -0.005614129992164798, 0.004762310510322675, -0.1880520746344168, -0.0011044923543114006, 0.0427970719655518, -0.0014078140421440657, -0.08756702161397682, -0.007738414050430832, 0.006202145995710929, 0.005854818815432261, 47.135721423924224, 37.07334118124534, -0.09377474757974277, 0.31565714940705586, 0.09999934590498197, 0.09991089993935774, 0.09759187217807695, 0.06733486298198578, -0.7335128533475309, -0.1679096029391904, -0.6509556068108323, -1.2430075325230476, 0.04136760108014608, 0.0999999999233999, 0.08720862248927727, 0.09999999999999992, 1.2395256763791034, 2.281713348565094, -1.0590250821214904, 2.5291406717449627]),
    ("vertex", "nu", 0.2): (32.101717572115625, [1.0476105812978578, -0.0017342814860727516, 0.00018171586778368708, 0.0008645047374607045, 0.15749144498800716, -0.0021208952470356014, -0.007041797786509162, 0.0020950875036733424, -0.15243426851506686, -0.0006786879760776021, 0.0283183102723215, -0.005247427807333994, -0.21667607866335736, -0.013926878334266523, 0.008034955507863824, 0.011563082723689, 38.965330122264085, 43.06544496110089, 0.1682146357862709, 0.24133374947965866, 0.01843359422417398, 0.15656828421850455, 0.19999687618472953, 0.19998850496928544, -30.23553109899805, 5.839252075548809, 4.365301130075522, 2.534105239048292, 0.18592237839391995, 0.0639686632939452, 0.031656450216218726, 0.03121897696992197, 58.63701694694932, 2.820534056040499, 0.2503528472774298, 5.94079757230999]),
    ("vertex", "nu", 0.3): (4.464878082339092, [1.1989587577431502, 0.008303246007013428, -0.0063211904266661815, -0.008937385058384862, -0.04160202937517156, -0.0024445682319169915, 0.008444342075485238, -0.006776335462281175, -0.15718523109463092, -0.0006206916890547481, 0.023967902369818106, -0.001259634027605805, -0.10738025846463567, -0.004404828184337788, -0.004314522534258412, 0.009220257899448988, 37.47151936692172, -54.146945321240935, 0.3273137469134739, -0.005273753970695939, 0.08773483046013618, 0.2999999999592624, 0.2897154207489504, 0.29999999369100366, 2.9031658001832747, -2.272097143150981, -0.38280437061795103, -1.17506892945636, 0.29999999827595025, 0.24243172910792418, 0.21044510959874846, 0.18393510131314456, 0.6241195524724009, -1.6901681345386206, -1.9193817212137596, -0.10829198005951521]),
}

NATURAL_BOUND = {"bkt": 0.03, "general_K": 0.05, "displacement": 0.1, "combined": 1.0, "vertex": 0.1}
BOUND_MEANING = {"bkt": "max |eps_a|", "general_K": "max |t_k| of h = log Kbar",
                 "displacement": "max brane displacement (rad)",
                 "combined": "multiple of the natural sizes (|t_k| <= 0.05 x, displacement <= 0.1 x)",
                 "vertex": "max |delta_a| (one-derivative brane vertex, independent for 10_H and 126bar_H)"}


def bound_needed(kind, setting):
    """Smallest scanned bound whose chi^2 is at most the number of fitted observables (13 or 17), or None."""
    n_obs = 17 if setting == "nu" else 13
    for b, c in sorted(SCANS[kind][setting]):
        if c <= n_obs:
            return b
    return None


def demonstration_report():
    G, dens = zero_mode_gram()
    u = brane_states()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    K = X @ X.conj().T
    nat = natural_epsilon_range()
    out = {"schema_version": 1, "model_id": MODEL_ID, "status": "normalization_corrections_fail_vertex_correction_rescues",
           "empirical_validation": False,
           "zero_mode_gram_deviation": float(np.max(abs(G - np.eye(3)))), "density_deviation": dens,
           "tight_frame": bool(np.allclose(sum(np.outer(a, a.conj()) for a in u), 4 / 3 * np.eye(3))),
           "schur": bool(np.allclose(a4_average(K), np.trace(K).real / 3 * np.eye(3))),
           "natural_sizes": {"bkt_eps_range": nat["eps_range"], "rM_range": nat["rM_range"],
                             "vertex_delta_range": (vertex_epsilon_estimate(nat["rM_range"][1]),
                                                    vertex_epsilon_estimate(nat["rM_range"][0], 2.24)),
                             "bounds_used": NATURAL_BOUND},
           "bound_meaning": BOUND_MEANING,
           "scans": {k: {s: sorted(rows) for s, rows in v.items()} for k, v in SCANS.items()},
           "chi2_at_natural_size": {k: {s: min(c for b, c in rows if b <= NATURAL_BOUND[k] + 1e-12)
                                        for s, rows in v.items()} for k, v in SCANS.items()},
           "bound_needed": {k: {s: bound_needed(k, s) for s in v} for k, v in SCANS.items()},
           "limitations": list(LIMITATIONS)}
    return out


def kinetic_map_to_tetrahedron(m, seed=0):
    """Exact pinned-model point reproducing the mass matrices m (from any four-brane realization) with a kinetic matrix.

    Four points in general position in CP^2 are projectively equivalent: some G in GL(3) maps the tetrahedral brane
    vectors u(t_a) to multiples of the realized ones u(p_a), so M = G Y G^T with Y in W. Write G = s V A (V unitary,
    A = Kbar^{-1/2} positive with det 1); V is a family rotation, so an O(1) kinetic matrix erases the pinning
    constraint altogether. Returns (xf for kind 'general_K', max |t_k|), using the brane-to-vertex assignment with
    the smallest max |t_k|.
    """
    from itertools import permutations
    from scipy.linalg import logm
    try:
        from .minimal_model import realize_yukawa_pair
    except ImportError:  # pragma: no cover
        from minimal_model import realize_yukawa_pair
    rz = realize_yukawa_pair(m["H"], m["F"], seed=seed)
    U = rz["U"]
    Ms = [U.T @ m["H"] @ U, U.T @ m["F"] @ U]
    up = [coherent_state(complex(z)) for z in rz["z"]]
    ut = brane_states()
    P, _ = ff.tetrahedral_brane_matrices()
    basis = np.array([Pi.ravel() for Pi in P]).T
    best = None
    for perm in permutations(range(4)):
        rows = []
        for i in range(4):  # G u_t[perm[i]] = lam_i u_p[i]: 12 linear equations for (G, lam), one null vector
            for k in range(3):
                row = np.zeros(13, complex)
                row[3 * k:3 * k + 3] = ut[perm[i]]
                row[9 + i] = -up[i][k]
                rows.append(row)
        G = np.linalg.svd(np.array(rows))[2][-1].conj()[:9].reshape(3, 3)
        w, V = np.linalg.eigh(G.conj().T @ G)
        A = V @ np.diag(np.sqrt(w)) @ V.conj().T  # polar part of G
        A = A / abs(np.linalg.det(A)) ** (1 / 3)
        h = logm(np.linalg.inv(A @ A))  # Kbar = A^{-2}, traceless because det A = 1
        h = (h + h.conj().T) / 2
        t = np.array([np.trace(h @ L).real for L in _GM])
        if best is None or np.max(abs(t)) < best[0]:
            best = (float(np.max(abs(t))), t, G)
    size, t, G = best
    A = _inverse_sqrt(general_kinetic_matrix(t))  # exactly the A the corrected model uses
    sV = G @ np.linalg.inv(A)
    V = sV / abs(np.linalg.det(sV)) ** (1 / 3)  # unitary up to a phase
    Ai = np.linalg.inv(A)
    coef = []
    for M in Ms:  # A Y A^T = V^dag M V^*, i.e. Y = A^{-1} V^dag M V^* A^{-T}, which lies in W
        Y = Ai @ V.conj().T @ M @ V.conj() @ Ai.T
        cvec, *_ = np.linalg.lstsq(basis, Y.ravel(), rcond=None)
        coef.append(cvec)
    c, d = coef
    x = np.r_[c.real, c.imag, d.real, d.imag, m["r"].real, m["r"].imag, m["s"].real, m["s"].imag, t]
    return x, size
