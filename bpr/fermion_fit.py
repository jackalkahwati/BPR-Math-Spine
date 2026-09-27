"""Phase 2: confronting the minimal BPR-6D model with fermion data.

See doc/derivations/phase2_fermion_fit_2026-09-27.md. The model (Phase 1 and 1d) gives, after symmetry breaking,
the non-supersymmetric SO(10) relations of a complex 10_H and a 126bar_H with a type-I seesaw:
    M_d = H + F,  M_e = H - 3 F,  M_u = r (H + s F),  M_D = r (H - 3 s F),  M_R = w F,
    m_nu = - M_D M_R^{-1} M_D^T,
where H and F are complex symmetric (H = Y10 v_d^10, F = Y126 v_d^126) and r, s, w are constants. With free brane
positions every pair (H, F) is reachable modulo U(3) (Phase 1); with the branes pinned at a regular tetrahedron
(Phase 1d) only a proper subset is (brane_stabilization.reachability_cost), so the generic fit here is a necessary
condition, and the pinned model adds the reachability of the fitted (H, F) as a further condition.

This module provides:
1. one-loop Standard-Model running of the gauge and Yukawa couplings from M_Z to the unification scale (inputs:
   running masses at M_Z from Xing-Zhang-Zhou 2008, CKM from PDG; anchors: b-tau non-unification, V_cb growth);
2. the identical-brane selection rule: with A4-symmetric brane couplings the unique light Higgs is the A4 singlet,
   whose eth-bar value vanishes at every vertex (3-fold selection rule), so identical branes give no Yukawas;
3. the fit: an exact construction of the charged sector in the basis where M_d is diagonal (only the two lepton mass
   ratios remain as conditions), neutrino observables from the type-I seesaw, and the stored best fits;
4. derived quantities: neutrino masses, M_R and the implied lower bound on the B-L scale.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

MODEL_ID = "bpr6d-fermion-fit-v1"
MZ = 91.1876
V_HIGGS = 174.10  # v / sqrt(2), GeV

LIMITATIONS = [
    "Running is one-loop Standard Model from M_Z to the unification scale; the 3221 stage between M_I and M_GUT, "
    "two-loop terms and thresholds are neglected (differences of order 5-10% in the GUT-scale masses).",
    "Neutrino observables are taken at low energy; their running in the SM is small and neglected.",
    "The fit uses assumed GUT-scale errors (light quarks 30%, heavy quarks 5%, charged leptons 1%, CKM 1-10%, "
    "neutrinos 3-10%); the chi^2 is a goodness-of-fit guide, not a likelihood.",
    "The fit is a multi-start local optimization; a better minimum may exist.",
    "Type-II seesaw is neglected (v_L ~ 1e-11 eV in this chain).",
]

# Inputs at M_Z (MS-bar running masses in GeV; Xing-Zhang-Zhou, PRD 77 (2008) 113016, as recalled; CKM from PDG).
MASSES_MZ = {"up": [1.27e-3, 0.619, 171.7], "down": [2.90e-3, 0.055, 2.89],
             "lepton": [0.486570e-3, 0.1027181, 1.74624]}
CKM_MZ = {"s12": 0.22500, "s23": 0.04182, "s13": 0.00369, "delta": 1.144}
GAUGE_MZ = {"alpha_em_inv": 127.951, "sin2w": 0.23122, "alpha_s": 0.1180}
NEUTRINO_DATA = {"dm21": 7.41e-5, "dm31": 2.511e-3, "s12sq": 0.303, "s23sq": 0.451, "s13sq": 0.02225}  # NuFIT-like, NO


def ckm_matrix(s12, s23, s13, delta):
    c12, c23, c13 = (np.sqrt(1 - x * x) for x in (s12, s23, s13))
    e = np.exp(1j * delta)
    return np.array([[c12 * c13, s12 * c13, s13 / e],
                     [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                     [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])


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


def _left(M):
    w, U = np.linalg.eigh(M @ M.conj().T)
    return U, np.sqrt(np.maximum(w, 0))


def run_to(mu=2e16):
    """Masses (GeV), |V_us|, |V_cb|, |V_ub|, J and gauge couplings at scale mu (one loop)."""
    g0, Ys0 = initial_conditions()
    sol = solve_ivp(_rhs, [0, np.log(mu / MZ)], _pack(g0, Ys0), rtol=1e-10, atol=1e-13)
    g, (Yu, Yd, Ye) = _unpack(sol.y[:, -1])
    Uu, mu_ = _left(Yu)
    Ud, md = _left(Yd)
    _, me = _left(Ye)
    V = Uu.conj().T @ Ud
    J = float(np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])))
    return {"mu": mu, "up": (mu_ * V_HIGGS).tolist(), "down": (md * V_HIGGS).tolist(), "lepton": (me * V_HIGGS).tolist(),
            "Vus": float(abs(V[0, 1])), "Vcb": float(abs(V[1, 2])), "Vub": float(abs(V[0, 2])), "J": abs(J),
            "gauge": g.tolist(), "mb_over_mtau": float(md[2] / me[2])}


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
# 3. The fit
# ---------------------------------------------------------------------------

SIGMA = {"up": np.array([0.3, 0.05, 0.05]), "down": np.array([0.3, 0.15, 0.05]), "lepton": np.array([0.01, 0.01, 0.01]),
         "ckm": np.array([0.01, 0.03, 0.05, 0.10]), "nu": np.array([0.03, 0.04, 0.10, 0.03])}


def targets(mu=2e16):
    run = run_to(mu)
    nu = NEUTRINO_DATA
    return {"up": np.array(run["up"]), "down": np.array(run["down"]), "lepton": np.array(run["lepton"]),
            "ckm": np.array([run["Vus"], run["Vcb"], run["Vub"], run["J"]]),
            "nu": np.array([nu["dm21"] / nu["dm31"], nu["s12sq"], nu["s23sq"], nu["s13sq"]])}


CKM_SIGMA = np.array([0.01, 0.03, 0.05, 0.10])  # relative, on s12, s23, s13, delta
DELTA_GUT = 1.2  # CKM phase at the unification scale (the SM running of delta is small)


def _ckm_from(pulls, tg):
    s12, s23, s13, _ = tg["ckm"]
    return ckm_matrix(s12 * (1 + CKM_SIGMA[0] * pulls[0]), s23 * (1 + CKM_SIGMA[1] * pulls[1]),
                      s13 * (1 + CKM_SIGMA[2] * pulls[2]), DELTA_GUT * (1 + CKM_SIGMA[3] * pulls[3]))


def build(p, tg):
    """Exact construction of the charged sector (18 parameters).

    p[0:3], p[3:6], p[15:18]: pulls of the up, down and lepton masses; p[6:10]: CKM pulls; p[10:12]: two Majorana-like
    phases of M_u; p[12:14]: a (complex); p[14]: arg b.
    In the basis M_d = diag(m_d) (a U(3) choice), M_u = V^dag diag(m_u e^{i alpha}) V^* reproduces the up masses and
    the CKM matrix exactly. The SO(10) relation M_u = a M_d + b M_e (a = r (3+s)/4, b = r (1-s)/4) then gives
    M_e = (M_u - a M_d)/b, with |b| fixed by m_tau; the only charged-sector conditions left are the two lepton mass
    ratios. Then r = a + b, s = 4a/r - 3, F = (M_d - M_e)/4, H = (3 M_d + M_e)/4, M_D = r (H - 3 s F).
    """
    Du = tg["up"] * np.exp(p[0:3] * SIGMA["up"])
    Dd = tg["down"] * np.exp(p[3:6] * SIGMA["down"])
    De = tg["lepton"] * np.exp(p[15:18] * SIGMA["lepton"])
    V = _ckm_from(p[6:10], tg)
    Mu = V.conj().T @ np.diag(Du * np.exp(1j * np.array([0.0, p[10], p[11]]))) @ V.conj()
    Md = np.diag(Dd).astype(complex)
    a = p[12] + 1j * p[13]
    X = Mu - a * Md
    sX = np.sort(np.linalg.svd(X, compute_uv=False))
    b = sX[2] / De[2] * np.exp(1j * p[14])
    Me = X / b
    r = a + b
    s = 4 * a / r - 3
    F = (Md - Me) / 4
    H = (3 * Md + Me) / 4
    return {"Mu": Mu, "Md": Md, "Me": Me, "MD": r * (H - 3 * s * F), "F": F, "H": H, "r": r, "s": s,
            "lepton_ratios": (sX[0] / sX[2], sX[1] / sX[2]), "lepton_targets": (De[0] / De[2], De[1] / De[2])}


def observables(p, tg):
    m = build(p, tg)
    Ue, _ = _left(m["Me"])
    mnu_unit = -m["MD"] @ np.linalg.solve(m["F"], m["MD"].T)  # m_nu for M_R = F (w = 1)
    Un, mn = _left(mnu_unit)
    P = Ue.conj().T @ Un
    s13 = abs(P[0, 2]) ** 2
    nu = np.array([(mn[1] ** 2 - mn[0] ** 2) / (mn[2] ** 2 - mn[0] ** 2), abs(P[0, 1]) ** 2 / (1 - s13),
                   abs(P[1, 2]) ** 2 / (1 - s13), s13])
    return {"nu": nu, "mnu_unit": mn, "PMNS": P, "F": m["F"], "matrices": m}


def residuals(p, tg, with_nu=True):
    o = observables(p, tg)
    m = o["matrices"]
    lep = [np.log(m["lepton_ratios"][k] / m["lepton_targets"][k]) / 0.01 for k in range(2)]
    r = [p[0:10], p[15:18], lep]
    if with_nu:
        r.append((o["nu"] - tg["nu"]) / (SIGMA["nu"] * tg["nu"]))
    return np.concatenate([np.ravel(x) for x in r])


def chi2(p, tg, with_nu=True):
    return float(np.sum(residuals(p, tg, with_nu) ** 2))


def _random_start(rng):
    return np.concatenate([np.zeros(10), rng.uniform(0, 2 * np.pi, 2), rng.normal(size=2) * np.exp(rng.uniform(0, 5)),
                           [rng.uniform(0, 2 * np.pi)], np.zeros(3)])


def fit(tg, with_nu=True, start=None, seed=0, n_starts=20, max_nfev=2000):
    """Multi-start least squares with basin hopping around the best point. Returns (chi^2, p)."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_starts):
        if best is not None and k % 2 == 1:
            p0 = best.x + rng.normal(size=18) * rng.choice([0.1, 0.3, 1.0])
        elif start is not None and k % 3 == 0:
            p0 = np.asarray(start) + rng.normal(size=18) * 0.1
        else:
            p0 = _random_start(rng)
        try:
            sol = least_squares(residuals, p0, args=(tg, with_nu), max_nfev=max_nfev)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    return 2 * best.cost, best.x


# ---------------------------------------------------------------------------
# 4. Derived quantities
# ---------------------------------------------------------------------------

def neutrino_sector(x, tg):
    """Scale w from Delta m^2_31; light masses (eV), sum, m_betabeta, and M_R eigenvalues (GeV) for M_R = w F."""
    o = observables(x, tg)
    mn = o["mnu_unit"]  # GeV, for w = 1
    dm31_gev2 = NEUTRINO_DATA["dm31"] * 1e-18
    w = np.sqrt((mn[2] ** 2 - mn[0] ** 2) / dm31_gev2)
    light_ev = mn / w * 1e9
    P = o["PMNS"]
    MR = np.sort(np.linalg.svd(o["F"], compute_uv=False)) * w
    return {"w": float(w), "light_masses_ev": light_ev.tolist(), "sum_ev": float(light_ev.sum()),
            "M_R_gev": MR.tolist(),
            "vR_lower_bound_gev": float(MR[-1] / np.sqrt(4 * np.pi)),  # |Y126| <= sqrt(4 pi)
            "abs_Ue": np.abs(P[0]).tolist()}
