"""Phase 2b: leading corrections to the pinned flavour structure of BPR-6D.

See doc/derivations/kinetic_corrections_2026-10-01.md. Phase 2 found that with the branes pinned at the Phase 1d
tetrahedron the minimal 10 + 126bar model fails the charged-fermion data at leading order (best chi^2 found 123.5
with neutrinos, 73.26 for the charged sector alone; m_s about 3.6 times too large). This module asks whether the
leading corrections can repair that, and at what size.

Structure. The three families are the holomorphic zero modes f_m = sqrt(3/(4 pi r^2)) (1, sqrt2 z, z^2)/(1 + |z|^2)
of the flux sphere, and a brane at z_a couples to their value there, psi(z_a) = sqrt(3/(4 pi r^2)) u_a^T psi with
u_a = coherent_state(z_a). Two kinds of correction follow.
1. Normalization corrections: brane-localized fermion kinetic terms (BKTs) and the metric/flux distortion from
   unequal brane tensions change the family kinetic matrix to psi^dag K psi, K Hermitian, while the brane Yukawas stay
   rank 1 (the value of a holomorphic section at z_a does not depend on the metric). After canonical normalization
   every Yukawa matrix (H, F, hence M_u, M_d, M_e, M_D, M_R) transforms as Y -> A Y A^T with A = Kbar^{-1/2}.
   A BKT kappa_a/M^2 at brane a gives K = 1 + sum_a eps_a conj(u_a) u_a^T with eps_a = 3 kappa_a / (4 pi (rM)^2).
   Since sum_a u_a u_a^dag = (4/3) 1 (the tetrahedral coherent states form a tight frame), equal BKTs do nothing;
   more generally any A4-symmetric K is proportional to 1 by Schur's lemma (the families are an A4 triplet). Only
   brane-to-brane differences matter.
2. Position corrections: unequal tensions also deform the condensate chi and move its zeros, hence the branes, off
   the regular tetrahedron.

For each correction the module fits the pinned model with the correction bounded in size, giving chi^2_min as a
function of the bound, and compares the size needed with the natural size.
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
    "Only the leading corrections are included: BKTs without derivatives, a general family kinetic matrix, and rigid "
    "brane displacements. Derivative BKTs, SO(10)-breaking (non-universal) brane terms and loop corrections to the "
    "brane Yukawas are not.",
    "The natural sizes are estimates: eps_a = 3 kappa_a / (4 pi (rM)^2) with kappa_a ~ 1 at the cutoff, and brane "
    "displacements of order the deficit spread (deficit ~ 0.1). The 6D background with four unequal branes is not "
    "solved.",
    "The scans are bounded local optimizations started from the Phase 2 pinned best points with random restarts; a "
    "better minimum at a given bound may exist.",
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


# ---------------------------------------------------------------------------
# 3. Corrected pinned models and bounded fits
# ---------------------------------------------------------------------------

KINDS = {"bkt": 4, "general_K": 8, "displacement": 8, "combined": 16}  # correction parameters after the 20 pinned
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
    raise ValueError(kind)


def correction_size(kind, xf):
    """bkt: max |eps_a|; general_K: max |eigenvalue of h| (the largest fractional change of a normalization is
    about half that); displacement: max geodesic displacement (radians)."""
    q = np.asarray(xf[20:20 + KINDS[kind]])
    if kind == "bkt":
        return float(np.max(abs(q)))
    if kind == "general_K":
        return float(np.max(abs(np.linalg.eigvalsh(sum(tk * L for tk, L in zip(q, _GM))))))
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
    if kind == "combined":
        lim = np.r_[np.full(8, COMBINED_UNITS[0] * bound), np.full(8, COMBINED_UNITS[1] * bound / np.sqrt(2))]
    else:
        lim = np.full(n, bound / np.sqrt(2) if kind == "displacement" else bound)  # box per tangent component
    lo = np.r_[np.full(20, -np.inf), -lim]
    hi = np.r_[np.full(20, np.inf), lim]
    return lo, hi


def bounded_fit(kind, bound, tg, start, with_nu=True, restarts=10, seed=0, max_nfev=1500):
    """chi^2_min with the correction parameters bounded (box), from `start` plus random restarts. Returns (chi^2, xf)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(kind, bound)
    n = KINDS[kind]
    start = np.asarray(start, float)
    cands = [start] + [np.r_[start[:20] * np.exp(rng.normal(size=20) * rng.choice([0.05, 0.2, 0.5])),
                             rng.uniform(lo[20:], hi[20:], n)] for _ in range(restarts)]
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

# Scan results: chi^2_min at each bound over all runs (two homotopy seeds per kind, plus seeded combined fits),
# counting points found at smaller bounds; stored optima are recomputed by the tests.
SCANS = {
    "bkt": {
        "charged": [(0.0, 73.2621), (0.005, 72.511), (0.01, 71.7646), (0.02, 70.2861), (0.03, 68.8268), (0.05, 65.9635), (0.075, 62.4841), (0.1, 59.1047), (0.15, 52.6302), (0.2, 46.4933), (0.3, 35.0279), (0.5, 15.1156)],
        "nu": [(0.0, 123.495), (0.005, 122.8585), (0.01, 121.2531), (0.02, 119.882), (0.03, 118.0705), (0.05, 116.3316), (0.075, 109.4784), (0.1, 106.1766), (0.15, 97.7554), (0.2, 93.461), (0.3, 78.4012), (0.5, 67.3633)],
    },
    "combined": {
        "charged": [(0.0, 73.2621), (0.5, 54.9018), (1.0, 40.5564), (2.0, 22.0692), (3.0, 12.484), (5.0, 8.0162)],
        "nu": [(0.0, 123.4949), (0.5, 110.3282), (1.0, 94.1712), (2.0, 78.4676), (3.0, 78.4676), (5.0, 67.1933)],
    },
    "displacement": {
        "charged": [(0.0, 73.2621), (0.01, 70.944), (0.02, 68.662), (0.05, 62.1177), (0.1, 52.0898), (0.15, 42.9816), (0.2, 35.1175), (0.3, 22.6706), (0.5, 12.3653)],
        "nu": [(0.0, 123.4949), (0.01, 122.1127), (0.02, 120.9223), (0.05, 114.8522), (0.1, 108.4067), (0.15, 102.0834), (0.2, 96.9878), (0.3, 94.2832), (0.5, 88.8448)],
    },
    "general_K": {
        "charged": [(0.0, 73.2621), (0.02, 66.2674), (0.05, 57.1497), (0.1, 44.6656), (0.2, 27.5257), (0.3, 15.9242), (0.5, 7.2477), (0.75, 2.182), (1.0, 0.5885), (1.5, 0.0)],
        "nu": [(0.0, 123.4949), (0.02, 113.8311), (0.05, 105.4), (0.1, 93.0239), (0.2, 78.4443), (0.3, 61.4887), (0.5, 48.5432), (0.75, 38.1265), (1.0, 21.119), (1.5, 18.643)],
    },
}
STORED_POINTS = {
    ("bkt", "charged", 0.03): (68.82675512962295, [-0.41076310999986215, 9.89356996678722e-05, -0.026698351637736283, 0.00015656984357959383, -0.8549598340206827, -0.004351676425890681, -0.029206405896738206, 0.004715324023860676, 0.017890776100267347, 0.0010469103965797184, -0.026469472458889563, -0.0008313761804397634, 0.2517265985110848, -0.004093516438891641, -0.04207745808172075, 0.004374663189803661, 47.11208612318644, -44.48775994515345, -0.8688704885968607, 0.13492928244276556, -0.030000000000999994, 0.030000000000999994, -0.030000000000999994, 0.030000000000999994]),
    ("bkt", "charged", 0.5): (15.11560609169055, [-0.2923683545937464, 0.0004974848982074795, -0.004111797397267031, -0.00019992164451250458, -0.3897767120987457, -0.0022164078772262835, -0.0034221689240191817, 0.002427289404842134, 0.03644836522246013, 0.0018361929430984554, -0.02422757608568635, -0.0009149554060468972, 0.08860074372405416, -0.006347642121877499, -0.016852021766248403, 0.006573501370851851, 42.89179754617545, -48.178366023853556, -0.2932860824470201, -0.0454818591306889, -0.5000000000009999, 0.5000000000009999, -0.5000000000009999, 0.5000000000009999]),
    ("bkt", "nu", 0.03): (118.07054796137018, [-0.003208595673835598, -0.653915544944523, -0.0932497481127667, 0.0038491725609894854, -0.008304449751254061, -0.2926346201159495, -0.0017983131888002328, 0.008409831701563958, 0.0017421919691720151, -0.08933619841530932, -0.001258456376315219, -0.0016616455036128652, -0.000565347836046414, 0.444901796705661, -0.023273887393493693, 0.0006747696793262293, -18.799871641729943, 20.926880664710126, -0.28425791317867005, 4.3548708241270555, 0.030000000000999994, -0.030000000000999994, -0.012352083945246865, 0.030000000000999994]),
    ("bkt", "nu", 0.5): (67.36332168261181, [-0.008680455699937376, -0.4492621546367509, -0.07875769572239658, 0.008786247379740665, -0.00724871099632366, -0.1311867341994408, 0.0029675444932254578, 0.006860768543154923, 0.0022412804911419427, -0.03391758417517658, 0.01284529957001453, -0.0020212680117699337, -0.0006943820036830581, 0.2606507717103496, -0.016586409558941813, 0.000747147602960386, 18.140912529630018, -20.776429256751765, 2.287768340342555, 3.3895962748462387, 0.5000000000009999, -0.5000000000009999, -0.08654043423458112, 0.5000000000009999]),
    ("combined", "charged", 1.0): (40.55641157979328, [-0.8288455358005026, 6.92654457517872e-05, -0.026439342399378205, -0.00046921951264526944, -0.5520099310519795, -0.003697773248251597, -0.009159213242295123, 0.0035839996848244836, 0.10363335737773456, 0.001677239554232352, -0.04080850917152881, -0.001969355095526262, 0.2305956653887447, -0.004523581005500166, -0.031171564676834292, 0.004181109959126549, 43.072914541291354, -49.08326496109867, -0.6118994386090414, 0.15157264685685487, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, 0.07071067811865474, -0.07071067811865474, -0.07071067811865472, 0.07071067811865472, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474]),
    ("combined", "charged", 5.0): (8.016249072296691, [-0.7220464122330943, -0.0002382688273233828, -0.013831537631439475, -0.00019992352960121097, -0.4065812859188168, -0.002229489901852554, -0.004616015211543634, 0.002257388492357375, 0.09168752645751382, 0.0024546994363400195, -0.003395017033661212, -0.003121493437986576, 0.09716572099491558, -0.00521793273274361, -0.06865871123178469, 0.00466574091237588, 24.406320827920297, -53.75283275574712, -0.18318919518323315, 0.1252249982057744, 0.24999005729801974, 0.24649428945318308, 0.24999928953363684, -0.2294699709740839, -0.17375269975887758, 0.24999801123696896, 0.2182148347766433, 0.24999997892402473, 0.288521306745082, -0.23787876279801132, -0.35355302424595625, -0.0364726587613763, -0.19402930486501765, -0.3535530837447339, -0.35355251809153226, -0.35333874543496485]),
    ("combined", "nu", 1.0): (94.17120286920229, [-0.002818678294181596, -0.25739207722122753, -0.08347079194846695, 0.0036719164360606397, -0.008675190431929446, -0.622645712291846, -0.02011357264435794, 0.008496213253556933, 0.0022127342046958107, -0.45392466267729836, 0.009371469915284732, -0.0021193130339628814, -0.00012636925185234552, 0.22463669789661872, -0.023590164637458725, 0.00028212331285512826, -24.565199079913857, -13.214220273278304, 0.5319734768609291, 3.7408834264153707, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, 0.049999999999999996, -0.049996630850179635, -0.012362155946189324, -0.005834610936272634, 0.07071048737528728, 0.015649895005434934, 0.03859909014900335, -0.01866620983456785, 0.01175198186628763, 0.00046249167722675524]),
    ("combined", "nu", 5.0): (67.19333261841777, [-0.0019742006638987313, -0.2599533161842981, -0.0945783414597156, 0.0019479667205719547, -0.009899820805558705, -0.2915795295177881, -0.010263559642724148, 0.010954511265341263, 0.002399094330444571, -0.4463535155491845, 0.01340007829032819, -0.0024328845661859932, 0.00023541040375426152, -0.0009735069319769913, -0.028076347227094872, -0.0002711078398362412, -18.3180590664067, -18.805936344287243, 0.9304821506396772, 3.3251368500995397, 0.15396592311917554, -0.24999999997195196, 0.24999999999999997, 0.24429584397120363, -0.24999999999999997, -0.24999999999999997, 0.24999999999999997, 0.24999999999999997, -0.14405813051306474, -0.29408418250450497, 0.2917136977196847, 0.1460942391290844, 0.2616957879266504, 0.13191559590317417, -0.29547128300965325, 0.29303631385800066]),
    ("displacement", "charged", 0.1): (52.08976154151817, [-0.694161215908074, 4.339994741517134e-05, -0.02914523631656096, -0.0004447834340556926, -0.702143521606691, -0.0039152087623058884, -0.017207946196020903, 0.0038556354320996534, 0.05029786854840213, 0.0015495629921861963, -0.032991086516690374, -0.0018107619565681141, 0.26897369734415794, -0.004067147968230284, -0.03920848106399736, 0.0038138402129191625, 43.19654862642091, -49.28733791241208, -0.7125280591535165, 0.20350107035648654, 0.07071067811865474, -0.07071067811865474, -0.07071067811865474, 0.06688291407282043, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474, -0.07071067811865474]),
    ("displacement", "charged", 0.5): (12.365294143381476, [-1.1032148748652308, 0.000342852367444588, -0.018675546237891964, -0.0006683234951749976, 0.6061226516577811, -0.0026589175080968717, -0.008412645700636389, 0.0020400387135286265, 0.18844338471181396, 0.0027095706472562466, -0.035830075669673785, -0.0026719573230112844, -0.04336455616657421, -0.005056906384006506, -0.07022813556181534, 0.003678025272836093, 47.0428255987123, -33.29874798236517, -0.30043092075616695, 0.0960372430761852, 0.3534768111207712, -0.35354399732183694, -0.3283202463590397, 0.12785183955186014, -0.27026944338790454, -0.28321775335578075, -0.35354488319803556, -0.3535530226282814]),
    ("displacement", "nu", 0.1): (108.40670809875725, [-0.0002988755413869982, -0.19022364666892516, -0.07377592070303089, 0.0007394142880926287, -0.008306642377223046, -0.5744823166469131, -0.041730056478660404, 0.00868907273266585, 0.0022917951987964096, -0.4955119174591211, 0.012975848203072737, -0.0022701179537177782, 0.00021018450102125075, 0.18734164108211243, -0.024820138598402968, -0.00012814763982610863, 15.314291039457217, -25.3147943446481, -0.024524845450005953, 3.348349791009507, 0.010165184475695714, 0.028434217479046925, 0.07071067216202934, 0.024381627609817318, 0.04834791450392448, -0.04592611716303312, 0.015460137762932594, 0.032587467541099965]),
    ("displacement", "nu", 0.5): (88.84475777330084, [-0.001203218185366159, -0.26082088568534834, -0.08235797634325784, 0.0016411303112662937, -0.00876093612568783, -0.5714636305547671, -0.02988485286362382, 0.009085769062350655, 0.002292242622414246, -0.5257877524692752, 0.012978820827413622, -0.0022705607611454705, 0.00020322145964670545, 0.18020151366825427, -0.02481962415804366, -0.00012649958896080346, 13.78526972621705, -24.44955284555854, 0.45426395157227767, 3.4794118562218874, 4.3832267939521796e-09, 0.034600735139312735, 0.14783345225595193, 0.09038307943748405, 0.13792081945973148, -0.08206935791729564, 0.014892078991677606, 0.09098728678176207]),
    ("general_K", "charged", 0.05): (57.149665243002445, [-0.6588095862728509, -2.2396039871835953e-05, -0.032117650418756104, 0.00017393802922601614, -0.7183306302655933, -0.004236683875734246, -0.0144840960642818, 0.00456211389174933, 0.06842646354756854, 0.0011734484330524128, -0.03942919102310408, -0.001015217007994498, 0.24470068449836038, -0.004407341197693818, -0.030665680008438966, 0.004691884848535688, 43.7537241947628, -47.821309439830195, -0.7766564633772562, 0.1527310644612374, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996]),
    ("general_K", "charged", 1.5): (9.666608193787628e-16, [-0.2635892496355022, 0.0014639632179448926, 0.021944231196692317, -0.0008222794347626113, 0.14704705204063392, 0.007056194252039909, -0.009806627348510481, -0.005963466689622013, -0.04842482304407706, -0.0031884087586302175, -0.053110242600552623, 0.0019782653269842716, 0.00989724086613781, -0.018898777738392485, 0.009336522810643871, 0.015512573920065022, -56.62918194565293, -40.851601924357304, 0.46948720654070714, -0.01810394906686415, -1.4999937720263228, -0.7654562416161467, -0.9477955781835979, 1.3348703661215462, 1.2950542766796027, -1.4999969286255397, 1.499986206130148, -1.499997389847362]),
    ("general_K", "nu", 0.05): (105.40001685306432, [-0.0025093615118190267, -0.20750479260600185, -0.07934390785237865, 0.0033635770475474096, -0.008294159514793465, -0.632911198705536, -0.025185529254439842, 0.00816348856506018, 0.0022125680050903147, -0.427593846804155, 0.009370732414563465, -0.0021193392670595225, -0.0001264110444636601, 0.23065899241381418, -0.023595977344689482, 0.0002819807010980456, -25.31636672516068, -14.140500188920138, 0.3318491186507578, 3.6264597492128168, 0.049999999999999996, 0.049999999999999996, 0.049999999999999996, -0.049999999999999996, -0.049999999999999996, -0.049999999999999996, 0.049999999999999996, -0.028417678065622582]),
    ("general_K", "nu", 1.5): (18.642956137110165, [-0.011010935235960343, -0.1244929435019019, -0.08771016720237536, 0.012520629521597309, -0.0062550144739800355, 0.07873122435700829, 0.05727799701318179, 0.00572678318497258, 0.002924874095712034, -0.08721122979017129, 0.01664852713295543, -0.0027160666888952044, -0.0006718216393466276, 0.15087373861525205, -0.02693038445381939, 0.0007873050720333954, -14.82599376493364, -23.064793716627104, 2.655702855240522, 1.5907051489306339, -1.0224098379377389, -0.5231453805835583, 1.3069451530002532, 0.3061727988251715, 0.12928916719853759, -1.4828696264783137, -1.0430476627130414, -0.8002588793531226]),
}

NATURAL_BOUND = {"bkt": 0.03, "general_K": 0.05, "displacement": 0.1, "combined": 1.0}
BOUND_MEANING = {"bkt": "max |eps_a|", "general_K": "max |t_k| of h = log Kbar",
                 "displacement": "max brane displacement (rad)",
                 "combined": "multiple of the natural sizes (|t_k| <= 0.05 x, displacement <= 0.1 x)"}


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
    out = {"schema_version": 1, "model_id": MODEL_ID, "status": "pinned_model_not_rescued_by_natural_corrections",
           "empirical_validation": False,
           "zero_mode_gram_deviation": float(np.max(abs(G - np.eye(3)))), "density_deviation": dens,
           "tight_frame": bool(np.allclose(sum(np.outer(a, a.conj()) for a in u), 4 / 3 * np.eye(3))),
           "schur": bool(np.allclose(a4_average(K), np.trace(K).real / 3 * np.eye(3))),
           "natural_sizes": {"bkt_eps_range": nat["eps_range"], "rM_range": nat["rM_range"], "tension_estimate": 0.1},
           "bound_meaning": BOUND_MEANING,
           "scans": {k: {s: sorted(rows) for s, rows in v.items()} for k, v in SCANS.items()},
           "chi2_at_natural_size": {k: {s: min(c for b, c in rows if b <= NATURAL_BOUND[k] + 1e-12)
                                        for s, rows in v.items()} for k, v in SCANS.items()},
           "bound_needed": {k: {s: bound_needed(k, s) for s in v} for k, v in SCANS.items()},
           "limitations": list(LIMITATIONS)}
    return out
