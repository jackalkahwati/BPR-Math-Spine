"""Phase 2: confronting the minimal BPR-6D model with fermion data.

See doc/derivations/phase2_fermion_fit_2026-09-27.md. The model (Phase 1 and 1d) gives, after symmetry breaking,
the non-supersymmetric SO(10) relations of a complex 10_H and a 126bar_H with a type-I seesaw:
    M_d = H + F,  M_e = H - 3 F,  M_u = r (H + s F),  M_D = r (H - 3 s F),  M_R = w F,
    m_nu = - M_D M_R^{-1} M_D^T,
where H and F are complex symmetric (H = Y10 v_d^10, F = Y126 v_d^126) and r, s, w are constants. Modulo U(3) family
rotations and the unobservable phase of r, the model has 18 real parameters (plus the scale w), against 17 fitted
observables. With free brane positions every pair (H, F) is reachable (Phase 1); with the branes pinned at a regular
tetrahedron (Phase 1d) H and F must lie, in some family basis, in the span W of the four tetrahedral brane matrices.

This module provides:
1. one-loop Standard-Model running of the gauge and Yukawa couplings from M_Z to the unification scale (inputs:
   running masses at M_Z from Xing-Zhang-Zhou 2008, CKM from PDG; anchors: b-tau non-unification, V_cb growth);
2. the identical-brane selection rule: with A4-symmetric brane couplings the unique light Higgs is the A4 singlet,
   whose eth-bar value vanishes at every vertex (3-fold selection rule), so identical branes give no Yukawas;
3. observables and chi^2 computed directly from the mass matrices, for any parametrization;
4. the generic fit: an exact construction of the charged sector in the basis where M_d is diagonal (18 parameters,
   locally complete: tested rank 18);
5. the pinned fit: H = sum_a c_a P_a and F = sum_a d_a P_a with P_a the tetrahedral brane matrices, so the pinned
   constraint holds exactly;
6. derived quantities: neutrino masses, m_betabeta, M_R and the implied lower bound on the B-L scale, and the
   stored best-fit points.
"""

import numpy as np
from scipy.integrate import solve_ivp

MODEL_ID = "bpr6d-fermion-fit-v2"
MZ = 91.1876
V_HIGGS = 174.10  # v / sqrt(2), GeV

LIMITATIONS = [
    "Running is one-loop Standard Model from M_Z to the unification scale; the 3221 stage between M_I and M_GUT, "
    "two-loop terms and thresholds are neglected. They are represented only by an 8% theory error on the quark masses "
    "(one- versus two-loop m_b/m_tau differs by 7.6%).",
    "Neutrino observables are taken at low energy; their running in the SM is small and neglected.",
    "The errors are assumed (light quarks 30%, m_s 15%, heavy quarks 5%, each with the 8% theory error in quadrature; "
    "charged leptons 1%; CKM angles 1-5% and delta 0.03 rad; neutrinos 3-10%). The chi^2 is a goodness-of-fit guide, "
    "not a likelihood, and the generic verdict depends on the m_d error.",
    "The fits are multi-start local optimizations; better minima may exist. The pinned verdict rests on searches, "
    "not on a proof of the global minimum.",
    "Type-II seesaw is neglected (v_L ~ 1e-11 eV in this chain).",
    "The pinned constraint is leading order: brane kinetic terms and the tension distortion add J = 0 pieces.",
]

# Inputs at M_Z (MS-bar running masses in GeV; Xing-Zhang-Zhou, PRD 77 (2008) 113016, as recalled; CKM from PDG).
MASSES_MZ = {"up": [1.27e-3, 0.619, 171.7], "down": [2.90e-3, 0.055, 2.89],
             "lepton": [0.486570e-3, 0.1027181, 1.74624]}
CKM_MZ = {"s12": 0.22500, "s23": 0.04182, "s13": 0.00369, "delta": 1.144}
GAUGE_MZ = {"alpha_em_inv": 127.951, "sin2w": 0.23122, "alpha_s": 0.1180}
# NuFIT-like global-fit values (recalled). Normal ordering: dm31 > 0; inverted ordering: dm32 < 0.
NEUTRINO_DATA = {"dm21": 7.41e-5, "dm31": 2.511e-3, "s12sq": 0.303, "s23sq": 0.451, "s13sq": 0.02225}
NEUTRINO_DATA_IO = {"dm21": 7.41e-5, "dm32": -2.498e-3, "s12sq": 0.303, "s23sq": 0.569, "s13sq": 0.02223}


def ckm_matrix(s12, s23, s13, delta):
    c12, c23, c13 = (np.sqrt(1 - x * x) for x in (s12, s23, s13))
    e = np.exp(1j * delta)
    return np.array([[c12 * c13, s12 * c13, s13 / e],
                     [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                     [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])


def ckm_parameters(V):
    """Standard-parametrization angles and phase of a unitary matrix, independent of rephasing.

    s13 = |V_ub|, s12 = |V_us|/c13, s23 = |V_cb|/c13; sin(delta) from the Jarlskog invariant and cos(delta) from |V_td|.
    """
    s13 = abs(V[0, 2])
    c13 = np.sqrt(1 - s13 ** 2)
    s12, s23 = abs(V[0, 1]) / c13, abs(V[1, 2]) / c13
    c12, c23 = np.sqrt(1 - s12 ** 2), np.sqrt(1 - s23 ** 2)
    J = np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1]))
    norm = c12 * s12 * c23 * s23 * s13
    sd = J / (norm * c13 ** 2)
    cd = (s12 ** 2 * s23 ** 2 + c12 ** 2 * c23 ** 2 * s13 ** 2 - abs(V[2, 0]) ** 2) / (2 * norm)
    return np.array([s12, s23, s13, np.arctan2(sd, cd)]), float(J)


def _left(M):
    """Left singular vectors and singular values of M, in ascending order (SVD keeps small masses accurate)."""
    U, s, _ = np.linalg.svd(M)
    return U[:, ::-1], s[::-1]


# ---------------------------------------------------------------------------
# 1. One-loop Standard-Model running
# ---------------------------------------------------------------------------

B_SM = np.array([41 / 10, -19 / 6, -7])


def _pack(g, Ys):
    return np.concatenate([g] + [np.concatenate([Y.ravel().real, Y.ravel().imag]) for Y in Ys])


def _unpack(x):
    g = x[:3]
    Ys = [(x[3 + 18 * k:3 + 18 * k + 9] + 1j * x[3 + 18 * k + 9:3 + 18 * (k + 1)]).reshape(3, 3) for k in range(3)]
    return g, Ys


def _rhs(t, x):
    """One-loop SM RGEs for L = -Qbar Y_u u H~ - Qbar Y_d d H - Lbar Y_e e H (doublet index on the left)."""
    g, (Yu, Yd, Ye) = _unpack(x)
    k = 1 / (16 * np.pi ** 2)
    Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
    T = np.trace(3 * Hu + 3 * Hd + He).real
    g1s, g2s, g3s = g ** 2
    I = np.eye(3)
    dYu = k * ((1.5 * (Hu - Hd) + (T - (17 / 20 * g1s + 9 / 4 * g2s + 8 * g3s)) * I) @ Yu)
    dYd = k * ((1.5 * (Hd - Hu) + (T - (1 / 4 * g1s + 9 / 4 * g2s + 8 * g3s)) * I) @ Yd)
    dYe = k * ((1.5 * He + (T - (9 / 4 * g1s + 9 / 4 * g2s)) * I) @ Ye)
    return _pack(k * B_SM * g ** 3, [dYu, dYd, dYe])


def initial_conditions():
    aem = 1 / GAUGE_MZ["alpha_em_inv"]
    sw2 = GAUGE_MZ["sin2w"]
    g1 = np.sqrt(4 * np.pi * aem / (1 - sw2) * 5 / 3)
    g2 = np.sqrt(4 * np.pi * aem / sw2)
    g3 = np.sqrt(4 * np.pi * GAUGE_MZ["alpha_s"])
    V = ckm_matrix(**CKM_MZ)
    Yu = np.diag(MASSES_MZ["up"]) / V_HIGGS
    Yd = V @ np.diag(MASSES_MZ["down"]) / V_HIGGS
    Ye = np.diag(MASSES_MZ["lepton"]) / V_HIGGS
    return np.array([g1, g2, g3]), [Yu.astype(complex), Yd.astype(complex), Ye.astype(complex)]


def run_to(mu=2e16):
    """Masses (GeV), CKM parameters and gauge couplings at scale mu (one loop)."""
    g0, Ys0 = initial_conditions()
    sol = solve_ivp(_rhs, [0, np.log(mu / MZ)], _pack(g0, Ys0), rtol=1e-10, atol=1e-13)
    g, (Yu, Yd, Ye) = _unpack(sol.y[:, -1])
    Uu, mu_ = _left(Yu)
    Ud, md = _left(Yd)
    _, me = _left(Ye)
    V = Uu.conj().T @ Ud
    ckm, J = ckm_parameters(V)
    return {"mu": mu, "up": (mu_ * V_HIGGS).tolist(), "down": (md * V_HIGGS).tolist(), "lepton": (me * V_HIGGS).tolist(),
            "Vus": float(abs(V[0, 1])), "Vcb": float(abs(V[1, 2])), "Vub": float(abs(V[0, 2])), "J": abs(J),
            "ckm": ckm.tolist(), "delta": float(ckm[3]), "gauge": g.tolist(), "mb_over_mtau": float(md[2] / me[2])}


# ---------------------------------------------------------------------------
# 2. Identical branes give no Yukawas
# ---------------------------------------------------------------------------

def identical_brane_selection_rule(kappa=-1.0, lam=-0.5):
    """Identical (A4-symmetric) brane mass terms on the l = 3 multiplet of the bulk Higgs: the nondegenerate
    eigenvectors are A4 singlets, and their eth-bar values at the tetrahedron vertices vanish.

    Proof: for the 3-fold rotation g_a about vertex a, D(g_a) v_a = e^{4 pi i/3} v_a with v_a = D(R_a) e_{-2}, while
    D(g_a) phi = phi for a singlet, so <v_a, phi> = e^{4 pi i/3} <v_a, phi> = 0. Returns the largest |c_a| over the
    nondegenerate levels and the degeneracy pattern of the spectrum (1 + 3 + 3 expected).
    """
    try:
        from .brane_stabilization import minimize_quartic, zeros
        from .minimal_model import wigner, basis_vector
    except ImportError:
        from brane_stabilization import minimize_quartic, zeros
        from minimal_model import wigner, basis_vector
    c, _ = minimize_quartic(starts=10)
    pts = zeros(c)
    Ds = [wigner(3, np.arccos(np.clip(p[2], -1, 1)), np.arctan2(p[1], p[0])) for p in pts]
    w = [D @ basis_vector(3, -3) for D in Ds]
    v = [D @ basis_vector(3, -2) for D in Ds]
    K = sum(kappa * np.outer(x, x.conj()) for x in w) + sum(lam * np.outer(x, x.conj()) for x in v)
    ev, U = np.linalg.eigh(K)
    groups, i = [], 0
    while i < 7:
        j = i
        while j + 1 < 7 and abs(ev[j + 1] - ev[i]) < 1e-6:
            j += 1
        groups.append(j - i + 1)
        i = j + 1
    worst = 0.0
    i = 0
    for size in groups:
        if size == 1:
            phi = U[:, i]
            worst = max(worst, max(abs(np.vdot(x, phi)) for x in v))
        i += size
    return {"degeneracies": groups, "max_singlet_coupling": float(worst)}


# ---------------------------------------------------------------------------
# 3. Targets, errors, and observables from the mass matrices
# ---------------------------------------------------------------------------

THEORY_ERROR = 0.08  # relative, on quark masses: one- vs two-loop m_b/m_tau differs by 7.6% (review)
_EXP = {"up": np.array([0.3, 0.05, 0.05]), "down": np.array([0.3, 0.15, 0.05])}
SIGMA = {"up": np.hypot(_EXP["up"], THEORY_ERROR), "down": np.hypot(_EXP["down"], THEORY_ERROR),
         "lepton": np.array([0.01, 0.01, 0.01]),
         "ckm": np.array([0.01, 0.03, 0.05, 0.03]),  # relative on s12, s23, s13; absolute (rad) on delta
         "nu": np.array([0.03, 0.04, 0.10, 0.03])}  # relative on dm21/dm3l, s12^2, s23^2, s13^2

NAMES = ["m_u", "m_c", "m_t", "m_d", "m_s", "m_b", "s12", "s23", "s13", "delta", "m_e", "m_mu", "m_tau",
         "dm21/dm3l", "s12^2 nu", "s23^2 nu", "s13^2 nu"]


def targets(mu=2e16, ordering="NO"):
    run = run_to(mu)
    if ordering == "NO":
        nu = NEUTRINO_DATA
        nu_t = [nu["dm21"] / nu["dm31"], nu["s12sq"], nu["s23sq"], nu["s13sq"]]
    else:
        nu = NEUTRINO_DATA_IO
        nu_t = [nu["dm21"] / abs(nu["dm32"]), nu["s12sq"], nu["s23sq"], nu["s13sq"]]
    return {"up": np.array(run["up"]), "down": np.array(run["down"]), "lepton": np.array(run["lepton"]),
            "ckm": np.array(run["ckm"]), "nu": np.array(nu_t), "ordering": ordering}


def light_neutrino_matrix(m, w=1.0):
    """m_nu = - M_D M_R^{-1} M_D^T with M_R = w F."""
    return -m["MD"] @ np.linalg.solve(w * m["F"], m["MD"].T)


def matrix_observables(m, ordering="NO"):
    """All fitted observables from the mass matrices (any family basis): masses, CKM (s12, s23, s13, delta),
    charged-lepton masses and the four neutrino observables, with the PMNS matrix and m_nu (for w = 1)."""
    Uu, mu = _left(m["Mu"])
    Ud, md = _left(m["Md"])
    Ue, me = _left(m["Me"])
    ckm, _ = ckm_parameters(Uu.conj().T @ Ud)
    out = {"up": mu, "down": md, "lepton": me, "ckm": ckm}
    if "MD" in m:
        Un, mn = _left(light_neutrino_matrix(m))
        if ordering == "IO":  # ascending masses are (m3, m1, m2)
            Un, mn = Un[:, [1, 2, 0]], mn[[1, 2, 0]]
            ratio = (mn[1] ** 2 - mn[0] ** 2) / (mn[1] ** 2 - mn[2] ** 2)
        else:
            ratio = (mn[1] ** 2 - mn[0] ** 2) / (mn[2] ** 2 - mn[0] ** 2)
        P = Ue.conj().T @ Un
        s13 = abs(P[0, 2]) ** 2
        out.update({"nu": np.array([ratio, abs(P[0, 1]) ** 2 / (1 - s13), abs(P[1, 2]) ** 2 / (1 - s13), s13]),
                    "mnu_unit": mn, "PMNS": P, "Ue": Ue})
    return out


def pulls(obs, tg, sigma=None, with_nu=True):
    """Residual vector (17 entries with neutrinos, 13 without), in the order of NAMES."""
    sg = SIGMA if sigma is None else sigma
    d = obs["ckm"][3] - tg["ckm"][3]
    d = (d + np.pi) % (2 * np.pi) - np.pi
    r = [np.log(obs["up"] / tg["up"]) / sg["up"], np.log(obs["down"] / tg["down"]) / sg["down"],
         (obs["ckm"][:3] / tg["ckm"][:3] - 1) / sg["ckm"][:3], [d / sg["ckm"][3]],
         np.log(obs["lepton"] / tg["lepton"]) / sg["lepton"]]
    if with_nu:
        r.append((obs["nu"] - tg["nu"]) / (sg["nu"] * tg["nu"]))
    return np.concatenate(r)


def matrix_chi2(m, tg, sigma=None, with_nu=True):
    return float(np.sum(pulls(matrix_observables(m, tg.get("ordering", "NO")), tg, sigma, with_nu) ** 2))


# ---------------------------------------------------------------------------
# 4. The generic fit (free brane positions)
# ---------------------------------------------------------------------------

N_GENERIC = 18


def build(p, tg, sigma=None):
    """Exact construction of the charged sector (18 parameters, the full model modulo U(3) and arg r).

    p[0:3], p[3:6]: pulls of the up and down masses; p[6:10]: CKM pulls (s12, s23, s13, delta); p[10:12]: two phases
    of M_u; p[12:14]: a (complex); p[14]: arg b; p[15]: pull of m_tau; p[16:18]: two phases of M_d.
    In the basis M_d = diag(m_d e^{i beta}) (a U(3) choice), M_u = V^dag diag(m_u e^{i alpha}) V^* reproduces the up
    masses and the CKM matrix exactly. The SO(10) relation M_u = a M_d + b M_e (a = r (3+s)/4, b = r (1-s)/4) then
    gives M_e = (M_u - a M_d)/b, with |b| fixed by m_tau; m_e and m_mu are the remaining charged-sector conditions.
    Then r = a + b, s = 4a/r - 3, F = (M_d - M_e)/4, H = (3 M_d + M_e)/4, M_D = r (H - 3 s F). The overall phase of
    M_u (arg r) is unobservable and fixed.
    """
    sg = SIGMA if sigma is None else sigma
    Du = tg["up"] * np.exp(p[0:3] * sg["up"])
    Dd = tg["down"] * np.exp(p[3:6] * sg["down"])
    s12, s23, s13, delta = tg["ckm"]
    V = ckm_matrix(s12 * (1 + sg["ckm"][0] * p[6]), s23 * (1 + sg["ckm"][1] * p[7]),
                   s13 * (1 + sg["ckm"][2] * p[8]), delta + sg["ckm"][3] * p[9])
    Mu = V.conj().T @ np.diag(Du * np.exp(1j * np.array([0.0, p[10], p[11]]))) @ V.conj()
    Md = np.diag(Dd * np.exp(1j * np.array([0.0, p[16], p[17]])))
    a = p[12] + 1j * p[13]
    X = Mu - a * Md
    b = np.linalg.svd(X, compute_uv=False)[0] / (tg["lepton"][2] * np.exp(p[15] * sg["lepton"][2])) * np.exp(1j * p[14])
    Me = X / b
    r = a + b
    s = 4 * a / r - 3
    F = (Md - Me) / 4
    H = (3 * Md + Me) / 4
    return {"Mu": Mu, "Md": Md, "Me": Me, "MD": r * (H - 3 * s * F), "F": F, "H": H, "r": r, "s": s}


def residuals(p, tg, with_nu=True, sigma=None):
    m = build(p, tg, sigma)
    return pulls(matrix_observables(m, tg.get("ordering", "NO")), tg, sigma, with_nu)


def chi2(p, tg, with_nu=True, sigma=None):
    return float(np.sum(residuals(p, tg, with_nu, sigma) ** 2))


def model_vector(m):
    """(H, F, r, s) as a real vector: the model data before the U(3) and arg r quotient."""
    v = np.concatenate([m["H"].ravel(), m["F"].ravel(), [m["r"], m["s"]]])
    return np.concatenate([v.real, v.imag])


def parametrization_rank(p, tg, frozen=(), eps=1e-6, tol=1e-9):
    """Rank of the construction modulo U(3) and arg r: rank[J_p, J_gauge] - rank[J_gauge].

    The full model has 28 - 9 - 1 = 18 real parameters, so a locally complete parametrization has rank 18. `frozen`
    lists parameter indices held fixed (e.g. the two M_d phases, 16 and 17: the first version of this module)."""
    try:
        from .minimal_model import _U3
    except ImportError:
        from minimal_model import _U3
    free = [k for k in range(len(p)) if k not in frozen]
    cols = []
    for k in free:
        dp = np.zeros(len(p))
        dp[k] = eps
        cols.append((model_vector(build(p + dp, tg)) - model_vector(build(p - dp, tg))) / (2 * eps))
    m = build(p, tg)
    gauge = []
    for g in _U3:
        gauge.append(model_vector({"H": g.T @ m["H"] + m["H"] @ g, "F": g.T @ m["F"] + m["F"] @ g, "r": 0, "s": 0}))
    gauge.append(model_vector({"H": 0 * m["H"], "F": 0 * m["F"], "r": 1j * m["r"], "s": 0}))
    G = np.array(gauge).T
    A = np.column_stack(cols + gauge)

    def rank(M):
        sv = np.linalg.svd(M / np.linalg.norm(M, axis=0), compute_uv=False)
        return int(np.sum(sv > tol * sv[0]))

    return rank(A) - rank(G)


def _random_start(rng):
    return np.concatenate([np.zeros(10), rng.uniform(0, 2 * np.pi, 2), rng.normal(size=2) * np.exp(rng.uniform(0, 5)),
                           [rng.uniform(0, 2 * np.pi)], [0.0], rng.uniform(0, 2 * np.pi, 2)])


def fit(tg, with_nu=True, start=None, seed=0, n_starts=20, max_nfev=2000, sigma=None):
    """Multi-start least squares with basin hopping around the best point. Returns (chi^2, p)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_starts):
        if best is not None and k % 2 == 1:
            p0 = best.x + rng.normal(size=N_GENERIC) * rng.choice([0.1, 0.3, 1.0])
        elif start is not None and k % 3 == 0:
            p0 = np.asarray(start) + rng.normal(size=N_GENERIC) * 0.1
        else:
            p0 = _random_start(rng)
        try:
            sol = least_squares(residuals, p0, args=(tg, with_nu, sigma), max_nfev=max_nfev)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    return 2 * best.cost, best.x


# ---------------------------------------------------------------------------
# 5. The pinned model: branes fixed at the tetrahedron (Phase 1d)
# ---------------------------------------------------------------------------

_P_TET = None


def tetrahedral_brane_matrices():
    """The four brane matrices u(z_a) u(z_a)^T at the zeros of the tetrahedral condensate (|2,2> + sqrt2 |2,-1>)/sqrt3.

    Any rotation of the tetrahedron acts on them as a U(3) family rotation (spin 1), so the orientation is immaterial.
    """
    global _P_TET
    if _P_TET is None:
        try:
            from .brane_stabilization import zeros
            from .minimal_model import brane_matrix
        except ImportError:
            from brane_stabilization import zeros
            from minimal_model import brane_matrix
        tet = np.array([1, 0, 0, np.sqrt(2), 0], complex) / np.sqrt(3)
        pts = zeros(tet)
        zs = [np.tan(np.arccos(np.clip(v[2], -1, 1)) / 2) * np.exp(1j * np.arctan2(v[1], v[0])) for v in pts]
        _P_TET = ([brane_matrix(z) for z in zs], pts)
    return _P_TET


def w_conditions(Y, c=None):
    """The two linear conditions that define W: the J = 0 part of Y, and its pairing with the condensate.

    Write Y in spin-2 components <2,M|Y> (spin-1 index m = 1, 0, -1). The pairing is fixed by <chi, u(z) u(z)^T> =
    the chi profile polynomial at z, so a brane at a zero of chi gives a matrix annihilated by chi. The four
    tetrahedral brane matrices are independent, so W is exactly the common kernel: the Yukawa matrices must be pure
    J = 2 and orthogonal to the same condensate that pins the branes.
    """
    try:
        from .brane_stabilization import _polynomial_coefficients
        from .minimal_model import j0_component
    except ImportError:
        from brane_stabilization import _polynomial_coefficients
        from minimal_model import j0_component
    c = np.array([1, 0, 0, np.sqrt(2), 0], complex) / np.sqrt(3) if c is None else c
    coef = _polynomial_coefficients(c)  # ascending powers zeta^k, k = 2 - M
    s2 = np.array([Y[0, 0], np.sqrt(2) * Y[0, 1], (2 * Y[0, 2] + 2 * Y[1, 1]) / np.sqrt(6), np.sqrt(2) * Y[1, 2], Y[2, 2]])
    return complex(j0_component(Y)), complex(np.sum(coef * s2 / np.sqrt([1, 4, 6, 4, 1])))


def pinned_matrices(x):
    """x = (Re c, Im c, Re d, Im d, Re r, Im r, Re s, Im s): H = sum c_a P_a and F = sum d_a P_a exactly in W."""
    P, _ = tetrahedral_brane_matrices()
    c = x[0:4] + 1j * x[4:8]
    d = x[8:12] + 1j * x[12:16]
    r, s = x[16] + 1j * x[17], x[18] + 1j * x[19]
    H = sum(ci * Pi for ci, Pi in zip(c, P))
    F = sum(di * Pi for di, Pi in zip(d, P))
    return {"Mu": r * (H + s * F), "Md": H + F, "Me": H - 3 * F, "MD": r * (H - 3 * s * F), "F": F, "H": H,
            "r": r, "s": s}


def pinned_residuals(x, tg, with_nu=True, sigma=None):
    m = pinned_matrices(x)
    try:
        r = pulls(matrix_observables(m, tg.get("ordering", "NO")), tg, sigma, with_nu)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        r = np.full(17 if with_nu else 13, 1e3)
    return r if np.all(np.isfinite(r)) else np.full(len(r), 1e3)


def pinned_chi2(x, tg, with_nu=True, sigma=None):
    return float(np.sum(pinned_residuals(x, tg, with_nu, sigma) ** 2))


def pinned_fit(tg, starts, seed=0, hops=10, max_nfev=3000, with_nu=True, sigma=None):
    """Least squares from each start, then basin hopping around the best point. Returns (chi^2, x)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    best = None
    for x0 in list(starts) + [None] * hops:
        if x0 is None:
            x0 = best[1] * np.exp(rng.normal(size=20) * rng.choice([0.05, 0.2, 0.5]))
        sol = least_squares(pinned_residuals, np.asarray(x0, float), args=(tg, with_nu, sigma), max_nfev=max_nfev,
                            x_scale="jac")
        if best is None or 2 * sol.cost < best[0]:
            best = (2 * sol.cost, sol.x)
    return best


def projected_onto_W(m):
    """Project H and F onto W (the span of the tetrahedral brane matrices) and rebuild all mass matrices."""
    P, _ = tetrahedral_brane_matrices()
    Q, _ = np.linalg.qr(np.array([Pi.ravel() for Pi in P]).T)
    proj = {k: (Q @ (Q.conj().T @ m[k].ravel())).reshape(3, 3) for k in ("H", "F")}
    H, F, r, s = proj["H"], proj["F"], m["r"], m["s"]
    return {"Mu": r * (H + s * F), "Md": H + F, "Me": H - 3 * F, "MD": r * (H - 3 * s * F), "F": F, "H": H,
            "r": r, "s": s}


# ---------------------------------------------------------------------------
# 6. Derived quantities and stored points
# ---------------------------------------------------------------------------

def neutrino_sector(m):
    """Scale w from Delta m^2_31 (normal ordering); light masses (eV), sum, m_betabeta, sin(delta_CP), and M_R
    eigenvalues (GeV) for M_R = w F, with the perturbativity bound v_R >= M_R,max / sqrt(4 pi)."""
    o = matrix_observables(m)
    mn = o["mnu_unit"]  # GeV, for w = 1
    w = np.sqrt((mn[2] ** 2 - mn[0] ** 2) / (NEUTRINO_DATA["dm31"] * 1e-18))
    light_ev = mn / w * 1e9
    P = o["PMNS"]
    MR = np.sort(np.linalg.svd(m["F"], compute_uv=False)) * w
    # m_betabeta in the charged-lepton mass basis. With Weyl fields (e^c^T M_e e_L, nu^T m_nu nu) and M_e symmetric,
    # the e_L rotation is the complex conjugate of the M_e M_e^dag eigenvectors, so m' = Ue^dag m_nu Ue^*.
    Ue = o["Ue"]
    m_bb = abs((Ue.conj().T @ light_neutrino_matrix(m, w) @ Ue.conj())[0, 0]) * 1e9
    s13 = abs(P[0, 2]) ** 2
    s12 = abs(P[0, 1]) ** 2 / (1 - s13)
    s23 = abs(P[1, 2]) ** 2 / (1 - s13)
    J = np.imag(P[0, 0] * P[1, 1] * np.conj(P[0, 1]) * np.conj(P[1, 0]))
    jmax = np.sqrt(s12 * (1 - s12) * s23 * (1 - s23) * s13) * (1 - s13)
    return {"w": float(w), "light_masses_ev": light_ev.tolist(), "sum_ev": float(light_ev.sum()),
            "m_betabeta_ev": float(m_bb), "jarlskog_lepton": float(J), "sin_delta_cp": float(J / jmax),
            "M_R_gev": MR.tolist(), "vR_lower_bound_gev": float(MR[-1] / np.sqrt(4 * np.pi))}


def one_loop_intermediate_scale():
    try:
        from .model_scales import unify_3221
    except ImportError:
        from model_scales import unify_3221
    return unify_3221()["M_I"]


def seesaw_consistency(m):
    """Compare the B-L scale the fit needs (v_R >= M_R,max / sqrt(4 pi)) with the one-loop intermediate scale."""
    nu = neutrino_sector(m)
    M_I = one_loop_intermediate_scale()
    return {"vR_lower_bound_gev": nu["vR_lower_bound_gev"], "M_I_one_loop_gev": M_I,
            "ratio": nu["vR_lower_bound_gev"] / M_I}


# Stored points, found by the searches described in the note; the tests recompute every chi^2 from scratch.
# Best generic fit (18 parameters, layout of build()); several independent multi-start searches reach it.
BEST_FIT_GENERIC = np.array([0.013462943952291643, 0.32867658943228595, -0.3335978909126514, -1.4721030880486072, 0.33534028360725243, 0.33265898005700895, 0.09797795172728226, 0.09398554530124709, -0.031910501845245014, 0.0661308429095271, 5.324008304759084, 1.660141618936364, 54.86564177289982, 8.265155852210505, 12.745492339955025, 0.0018195961011996482, -2.07527559440493, 1.6522709757861418])
# Representatives of the other generic minima within Delta chi^2 = 4 (chi^2 = 3.66 and 6.82).
GENERIC_NEAR_BEST = [np.array([0.01832440781688592, 0.4185441459014871, -0.4244102868484063, -1.7153243706398176, 0.46000655482410746, 0.19616976277074527, 0.11436963390754595, 0.0453498679924652, -0.019811442578275426, 0.07050651764174813, -0.8422123460549931, -4.486007951751287, 53.95028907517044, 9.866600391797878, 0.1836530343147331, 0.026549584865380484, 4.252490974824401, 8.027931438494553]),
                     np.array([0.01896423811273633, 0.5046752795006666, -0.5057680432427731, -2.4790392075220007, 0.31488435033378126, 0.03889998515485146, 0.10428406155336802, -0.02357812111093027, 0.051975367575752995, 0.06517923142147024, 6.568081296105201, 2.810319135846926, 53.74951178961406, 7.62602062582343, 12.720631207568628, 0.059217545371605826, 1.2308018100769091, 2.808358466863613])]
# Sensitivity: best generic fit with a 10% experimental m_d error (plus the theory error).
BEST_FIT_GENERIC_MD10 = np.array([0.040821588829361814, 0.8326271307744328, -0.8444175362384427, -1.7618971804810835, 0.39042205526451085, 0.654597005456226, 0.23564059058975576, 0.22821609050852762, -0.12396185694699746, 0.14390706133706233, 5.235638849154857, 1.5112037633300661, 51.72326353722566, 4.41335782072363, 12.657639763859127, 0.009009249893068382, -2.096617999292038, 1.5383726352349607])
# Best generic fit to inverted-ordering data (targets(ordering='IO')).
BEST_FIT_GENERIC_IO = np.array([0.490517585087434, -0.4450973873051285, -5.494259960034613, 9.942862561607019, 0.8512660145548092, -1.0153486197351205, 0.6040056585135641, -0.5495962961437111, -0.5966127980097901, 1.5332725550981712, 9.350998814356252, -0.06697374272795102, -10.395579668687505, -1.7796764745363427, 3.97520976448825, -1.3233960460882812, 6.016295884050969, 2.81383443151232])
# Best pinned points found (layout of pinned_matrices(); exactly in W): with neutrinos, and charged sector only.
BEST_FIT_PINNED = np.array([0.00010320340176006442, -0.11154127587423927, -0.06885885193572311, 0.0006167832370166705, -0.008046615387991687, -0.6245926437215529, -0.05403113317421591, 0.008263652179074939, 0.002081017284907687, -0.4612111191328607, 0.014052638592808617, -0.0020190801759831954, 0.00018379058725229397, 0.17792123104599586, -0.022777150476107164, -4.4194444583473535e-05, 13.148725055041833, -26.880093923341782, -0.2853209813443338, 3.5562496866877673])
BEST_FIT_PINNED_CHARGED = np.array([-0.4408334860050985, 7.512900459747062e-05, -0.02783732527723586, 0.00017910589984580826, -0.864237789283735, -0.004332242771775071, -0.032214593827929454, 0.004710516101915942, 0.03783544707028953, 0.0008438311153690518, -0.028148946095659627, -0.0006356489904410534, 0.2549893487960821, -0.003970129286250685, -0.04296648800271806, 0.004249559677299583, 47.9557814766127, -43.46840535917099, -0.9054831440669806, 0.10313038632547167])
STORED_CHI2 = {"generic": 2.9256, "generic_md10": 8.1188, "generic_IO": 43723.5167, "pinned": 123.4969, "pinned_charged_only": 73.2621}


def prediction_ranges(tg=None):
    """Spread of the neutrino-sector predictions over the stored generic minima (best plus those within 4)."""
    pts = [BEST_FIT_GENERIC] + list(GENERIC_NEAR_BEST)
    tg = targets(2e16) if tg is None else tg
    rows = [neutrino_sector(build(p, tg)) for p in pts]
    keys = ["sum_ev", "m_betabeta_ev", "sin_delta_cp", "vR_lower_bound_gev"]
    out = {k: (min(r[k] for r in rows), max(r[k] for r in rows)) for k in keys}
    for i in range(3):
        out["m%d_ev" % (i + 1)] = (min(r["light_masses_ev"][i] for r in rows), max(r["light_masses_ev"][i] for r in rows))
        out["M_R%d_gev" % (i + 1)] = (min(r["M_R_gev"][i] for r in rows), max(r["M_R_gev"][i] for r in rows))
    out["chi2"] = [chi2(p, tg) for p in pts]
    return out


def demonstration_report():
    tg = targets(2e16)
    run = run_to(2e16)
    out = {"schema_version": 2, "model_id": MODEL_ID, "status": "generic_fits_pinned_fails_charged_sector",
           "empirical_validation": False, "gut_scale_inputs": run,
           "identical_brane_rule": identical_brane_selection_rule()}
    cases = (("generic", build(BEST_FIT_GENERIC, tg), True), ("pinned", pinned_matrices(BEST_FIT_PINNED), True),
             ("pinned_charged_only", pinned_matrices(BEST_FIT_PINNED_CHARGED), False))
    for label, m, with_nu in cases:
        r = pulls(matrix_observables(m), tg, with_nu=with_nu)
        out[label] = {"chi2": float(np.sum(r ** 2)), "pulls": dict(zip(NAMES, [round(float(v), 2) for v in r]))}
        if with_nu:
            out[label]["neutrino_sector"] = neutrino_sector(m)
    out["generic"]["parametrization_rank"] = parametrization_rank(BEST_FIT_GENERIC, tg)
    out["prediction_ranges"] = prediction_ranges(tg)
    out["seesaw"] = seesaw_consistency(build(BEST_FIT_GENERIC, tg))
    out["limitations"] = list(LIMITATIONS)
    return out
