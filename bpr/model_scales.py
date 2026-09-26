"""Phase 1b: the scales of the minimal BPR-6D model and its quantitative kill checks.

See doc/derivations/minimal_model_2026-09-26.md. One-loop gauge running for the chain
Spin(10) -> SU(3) x SU(2)_L x SU(2)_R x U(1)_B-L (at M_GUT, by the 45H) -> SM (at M_I, by the 126barH), with beta
coefficients computed from the field content (extended survival hypothesis), then:
- the proton lifetime against the Super-Kamiokande bound tau(p -> e+ pi0) > 2.4e34 yr;
- the compactification window: M_GUT <= 1/r (a 4D description of the breaking) and r M large enough for classical
  control, with 1/r = 2 g_F M_Pl / 3 and 1/(r M) = 2 pi^(1/4) sqrt(g_F / 3) (flux note, parent flux 3);
- the type-I seesaw: the Dirac neutrino Yukawa needed for m_nu ~ 0.05 eV at M_R ~ M_I.
Inputs at M_Z: alpha_em^-1 = 127.951, sin^2 theta_W = 0.23122 (MS-bar), alpha_s = 0.1180.
"""

from fractions import Fraction

import numpy as np

MODEL_ID = "bpr6d-model-scales-v1"
MZ = 91.1876
ALPHA_EM_INV = 127.951
SIN2W = 0.23122
ALPHA_S = 0.1180
REDUCED_PLANCK_GEV = 2.435e18
SUPER_K_TAU_YR = 2.4e34
GEV_INV_TO_YR = 6.582e-25 / 3.156e7
HIGGS_VEV_GEV = 174.1  # v / sqrt(2)

LIMITATIONS = [
    "One-loop running without threshold corrections; thresholds and two-loop terms can move M_I by orders of "
    "magnitude and M_GUT by a factor of a few.",
    "Only the 3221 chain is computed; the 421 chain of the same Higgs content is not.",
    "The proton lifetime uses the naive estimate M_GUT^4 / (alpha_G^2 m_p^5), uncertain by about an order of magnitude.",
    "Kaluza-Klein thresholds at 1/r, close to M_GUT, are not included.",
    "The classical-control criterion r M >= c is a convention; c = 1 and c = 3 are both reported.",
]

F = Fraction


def low_energy_couplings():
    """GUT-normalized 1/alpha_1, 1/alpha_2, 1/alpha_3 at M_Z."""
    inv1 = F(3, 5) * (1 - F(SIN2W).limit_denominator(10 ** 8)) * F(ALPHA_EM_INV).limit_denominator(10 ** 8)
    inv2 = F(SIN2W).limit_denominator(10 ** 8) * F(ALPHA_EM_INV).limit_denominator(10 ** 8)
    return [float(inv1), float(inv2), 1 / ALPHA_S]


# ---------------------------------------------------------------------------
# Beta coefficients from field content
# ---------------------------------------------------------------------------

class Charge:
    """A U(1) charge known through its square (one-loop coefficients use only q^2), e.g. sqrt(3/5) Y."""

    def __init__(self, square):
        self.square = F(square)


def beta_coefficients(groups, fields):
    """One-loop b_i = -11/3 C2(G_i) + sum_f k_f S_i(R_f) prod_{j != i} dim_j(R_f).

    groups: list of ("SU", N) or ("U1", None). fields: list of (kind, reps), kind in {"weyl", "complex", "real"}
    (k = 2/3, 1/3, 1/6); reps has one entry per group: (dim, index S) for SU(N), or a Charge for U(1) (S = q^2).
    """
    k = {"weyl": F(2, 3), "complex": F(1, 3), "real": F(1, 6)}
    b = []
    for i, (gtype, N) in enumerate(groups):
        total = F(-11, 3) * N if gtype == "SU" else F(0)
        for kind, reps in fields:
            S = reps[i][1] if gtype == "SU" else reps[i].square
            mult = 1
            for j, (gt, _) in enumerate(groups):
                if j != i and gt == "SU":
                    mult *= reps[j][0]
            total += k[kind] * S * mult
        b.append(total)
    return b


SM_GROUPS = [("U1", None), ("SU", 2), ("SU", 3)]


def _gut_q(Y):
    return Charge(F(3, 5) * F(Y) ** 2)


def _sm_fermions(families=3):
    rows = []
    for Y, d2, d3 in [(F(1, 6), 2, 3), (F(-2, 3), 1, 3), (F(1, 3), 1, 3), (F(-1, 2), 2, 1), (F(1), 1, 1)]:
        rows.append(("weyl", [_gut_q(Y), (d2, F(1, 2) if d2 == 2 else 0), (d3, F(1, 2) if d3 == 3 else 0)]))
    return rows * families


def sm_beta(higgs_doublets=1):
    h = [("complex", [_gut_q(F(1, 2)), (2, F(1, 2)), (1, 0)])] * higgs_doublets
    return beta_coefficients(SM_GROUPS, _sm_fermions() + h)


def mssm_beta():
    """Anchor: MSSM b = (33/5, 1, -3) from gauginos, fermions + sfermions and two Higgs doublets + higgsinos."""
    fields = _sm_fermions()
    fields += [("complex", reps) for _, reps in _sm_fermions()]  # sfermions
    for Y in (F(1, 2), F(-1, 2)):
        reps = [_gut_q(Y), (2, F(1, 2)), (1, 0)]
        fields += [("complex", reps), ("weyl", reps)]
    fields += [("weyl", [_gut_q(0), (1, 0), (1, 0)]),  # bino (no contribution)
               ("weyl", [_gut_q(0), (3, F(2)), (1, 0)]),  # wino: adjoint Weyl
               ("weyl", [_gut_q(0), (1, 0), (8, F(3))])]  # gluino
    return beta_coefficients(SM_GROUPS, fields)


G3221 = [("SU", 3), ("SU", 2), ("SU", 2), ("U1", None)]  # SU(3)c, SU(2)L, SU(2)R, U(1)_B-L


def _bl(BL):
    # GUT-normalized B-L charge: q = sqrt(3/8) (B - L).
    return Charge(F(3, 8) * BL * BL)


def beta_3221():
    """Three 16s: Q (3,2,1,1/3), Q^c (3b,1,2,-1/3), L (1,2,1,-1), L^c (1,1,2,+1); scalars: one complex bidoublet
    (1,2,2,0) from the complex 10H and Delta_R (1,1,3,-2) from the 126barH (extended survival hypothesis)."""
    d = lambda n: (n, F(1, 2) if n in (2, 3) else 0)
    fam = [("weyl", [d(3), d(2), (1, 0), _bl(F(1, 3))]), ("weyl", [d(3), (1, 0), d(2), _bl(F(-1, 3))]),
           ("weyl", [(1, 0), d(2), (1, 0), _bl(F(-1))]), ("weyl", [(1, 0), (1, 0), d(2), _bl(F(1))])]
    scalars = [("complex", [(1, 0), d(2), d(2), _bl(F(0))]), ("complex", [(1, 0), (1, 0), (3, F(2)), _bl(F(-2))])]
    return beta_coefficients(G3221, fam * 3 + scalars)


# ---------------------------------------------------------------------------
# Two-step unification
# ---------------------------------------------------------------------------

def unify_3221(higgs_doublets_below=1):
    """Solve 1/alpha_G, ln(M_I/M_Z), ln(M_G/M_Z) and the SU(2)_R, B-L couplings at M_I (linear one-loop system).

    Matching at M_I: 1/alpha_1 = (3/5)/alpha_2R + (2/5)/alpha_BL. Unification: alpha_3 = alpha_2L = alpha_2R = alpha_BL.
    """
    inv1, inv2, inv3 = low_energy_couplings()
    b1, b2, b3 = (float(x) for x in sm_beta(higgs_doublets_below))
    c3, c2L, c2R, cBL = (float(x) for x in beta_3221())
    tp = 2 * np.pi
    A = [[1, (b3 - c3) / tp, c3 / tp, 0, 0], [1, (b2 - c2L) / tp, c2L / tp, 0, 0],
         [1, -c2R / tp, c2R / tp, -1, 0], [1, -cBL / tp, cBL / tp, 0, -1], [0, b1 / tp, 0, 3 / 5, 2 / 5]]
    y = [inv3, inv2, 0, 0, inv1]
    x = np.linalg.solve(np.array(A), np.array(y))
    return {"alpha_G_inverse": float(x[0]), "M_I": float(MZ * np.exp(x[1])), "M_GUT": float(MZ * np.exp(x[2])),
            "alpha_2R_inverse_at_MI": float(x[3]), "alpha_BL_inverse_at_MI": float(x[4]),
            "ordered": bool(0 < x[1] < x[2])}


def mssm_unification():
    """Anchor: MSSM one-loop alpha_1 = alpha_2 near 2e16 GeV with alpha_3 close by."""
    inv = low_energy_couplings()
    b = [float(x) for x in mssm_beta()]
    t = (inv[0] - inv[1]) / ((b[0] - b[1]) / (2 * np.pi))
    return {"M_GUT": float(MZ * np.exp(t)), "alpha_G_inverse": inv[0] - b[0] * t / (2 * np.pi),
            "alpha_3_inverse_there": inv[2] - b[2] * t / (2 * np.pi)}


def proton_lifetime_years(M_GUT, alpha_G_inverse, m_p=0.938):
    """Naive tau(p -> e+ pi0) ~ M_GUT^4 / (alpha_G^2 m_p^5), uncertain by about an order of magnitude."""
    return M_GUT ** 4 * alpha_G_inverse ** 2 / m_p ** 5 * GEV_INV_TO_YR


def minimum_gut_scale(alpha_G_inverse, bound_years=SUPER_K_TAU_YR, m_p=0.938):
    return (bound_years / GEV_INV_TO_YR * m_p ** 5 / alpha_G_inverse ** 2) ** 0.25


# ---------------------------------------------------------------------------
# Compactification window
# ---------------------------------------------------------------------------

def inverse_radius(g_F, flux=3):
    return 2 * g_F * REDUCED_PLANCK_GEV / flux


def r_times_M(g_F, flux=3):
    """r M = 1 / (2 pi^(1/4) sqrt(g_F / flux)) (flux note, four_d_relations)."""
    return 1 / (2 * np.pi ** 0.25 * np.sqrt(g_F / flux))


def control_window(M_GUT, control=3.0, flux=3):
    """g_F range with M_GUT <= 1/r (lower end) and r M >= control (upper end); empty if lower > upper."""
    g_low = M_GUT * flux / (2 * REDUCED_PLANCK_GEV)
    g_high = flux / (4 * np.sqrt(np.pi) * control ** 2)
    return {"g_F_min": g_low, "g_F_max": g_high, "nonempty": bool(g_low <= g_high),
            "inverse_radius_range": (inverse_radius(g_low, flux), inverse_radius(g_high, flux)),
            "rM_range": (r_times_M(g_high, flux), r_times_M(g_low, flux))}


# ---------------------------------------------------------------------------
# Seesaw
# ---------------------------------------------------------------------------

def seesaw_dirac_yukawa(M_R, m_nu_ev=0.05):
    """Type-I seesaw m_nu = y_D^2 v^2 / M_R: the y_D needed (v = 174 GeV)."""
    return float(np.sqrt(m_nu_ev * 1e-9 * M_R) / HIGGS_VEV_GEV)


def demonstration_report():
    out = {"schema_version": 1, "model_id": MODEL_ID, "status": "one_loop_scales_and_kill_checks",
           "empirical_validation": False,
           "beta": {"SM": [str(x) for x in sm_beta(1)], "2HDM": [str(x) for x in sm_beta(2)],
                    "MSSM": [str(x) for x in mssm_beta()], "3221": [str(x) for x in beta_3221()]},
           "mssm_anchor": mssm_unification()}
    for label, nd in (("SM below M_I", 1), ("2HDM below M_I", 2)):
        u = unify_3221(nd)
        tau = proton_lifetime_years(u["M_GUT"], u["alpha_G_inverse"])
        out[label] = {**u, "proton_lifetime_yr": tau, "super_k_ok": bool(tau > SUPER_K_TAU_YR),
                      "min_M_GUT_for_super_k": minimum_gut_scale(u["alpha_G_inverse"]),
                      "window_rM3": control_window(u["M_GUT"], 3.0), "window_rM1": control_window(u["M_GUT"], 1.0),
                      "seesaw_yD_needed_at_M_I": seesaw_dirac_yukawa(u["M_I"])}
    out["limitations"] = list(LIMITATIONS)
    return out
