"""Checks for doc/derivations/minimal_model_2026-09-26.md (Phase 1b).

Oracles: the textbook one-loop coefficients (SM 41/10, -19/6, -7; 2HDM 21/5, -3, -7; MSSM 33/5, 1, -3), a
hand computation of the 3221 coefficients, MSSM unification near 2e16 GeV, forward re-running of the solved
couplings, and the flux module's own r M formula.
"""

import json
from fractions import Fraction as F

import numpy as np
import pytest
import sympy as sp

import bpr.model_scales as ms


def test_textbook_beta_coefficients():
    assert ms.sm_beta(1) == [F(41, 10), F(-19, 6), F(-7)]
    assert ms.sm_beta(2) == [F(21, 5), F(-3), F(-7)]
    assert ms.mssm_beta() == [F(33, 5), F(1), F(-3)]


def test_3221_coefficients_by_hand():
    # SU(3): -11 + (2/3)(2 per family)(3) = -7. SU(2)_L,R: -22/3 + (2/3)(2)(3) + bidoublet 1/3 (+ Delta_R 2/3 for R).
    # B-L with q = sqrt(3/8)(B-L): fermions (2/3)(16/3 * 3/8)(3) = 4; Delta_R (1/3)(3 * 4 * 3/8) = 3/2.
    assert ms.beta_3221() == [F(-7), F(-3), F(-7, 3), F(11, 2)]


def test_mssm_anchor():
    anchor = ms.mssm_unification()
    assert 1.5e16 < anchor["M_GUT"] < 2.5e16
    assert anchor["alpha_3_inverse_there"] == pytest.approx(anchor["alpha_G_inverse"], rel=0.01)


@pytest.mark.parametrize("doublets", [1, 2])
def test_3221_unification_by_forward_running(doublets):
    u = ms.unify_3221(doublets)
    assert u["ordered"] and 1e8 < u["M_I"] < 1e11 and 1e16 < u["M_GUT"] < 1e17
    inv1, inv2, inv3 = ms.low_energy_couplings()
    b1, b2, b3 = (float(x) for x in ms.sm_beta(doublets))
    c3, c2L, c2R, cBL = (float(x) for x in ms.beta_3221())
    tI, tG = np.log(u["M_I"] / ms.MZ), np.log(u["M_GUT"] / ms.MZ)
    tp = 2 * np.pi
    # Independent forward run: SM to M_I, match, 3221 to M_GUT.
    a3 = inv3 - b3 * tI / tp - c3 * (tG - tI) / tp
    a2L = inv2 - b2 * tI / tp - c2L * (tG - tI) / tp
    a1_I = inv1 - b1 * tI / tp
    a2R = u["alpha_2R_inverse_at_MI"] - c2R * (tG - tI) / tp
    aBL = u["alpha_BL_inverse_at_MI"] - cBL * (tG - tI) / tp
    assert 0.6 * u["alpha_2R_inverse_at_MI"] + 0.4 * u["alpha_BL_inverse_at_MI"] == pytest.approx(a1_I)
    for a in (a3, a2L, a2R, aBL):
        assert a == pytest.approx(u["alpha_G_inverse"], rel=1e-10)


def test_proton_lifetime_passes_super_k():
    for doublets in (1, 2):
        u = ms.unify_3221(doublets)
        tau = ms.proton_lifetime_years(u["M_GUT"], u["alpha_G_inverse"])
        assert tau > 100 * ms.SUPER_K_TAU_YR
        m_min = ms.minimum_gut_scale(u["alpha_G_inverse"])
        assert ms.proton_lifetime_years(m_min, u["alpha_G_inverse"]) == pytest.approx(ms.SUPER_K_TAU_YR, rel=1e-9)


def test_proton_lifetime_anchor():
    # M_GUT = 1e16 GeV, 1/alpha_G = 40, m_p = 0.938 GeV: 1e64 * 1600 / 0.7275 GeV^-1 = 2.2e67 GeV^-1 = 4.6e35 yr.
    assert ms.proton_lifetime_years(1e16, 40) == pytest.approx(4.6e35, rel=0.02)


def test_brane_copy_content_has_no_consistent_intermediate_scale():
    rows = ms.multiplicity_scan()
    for r in rows:
        if r["delta_R"] == 3:  # three condensing Delta_R, as brane copies need for a rank-3 Majorana matrix
            assert not (r["viable_M_I"] and r["super_k_ok"])
        if r["delta_R"] == 1 and r["bidoublets"] == 1:
            assert r["viable_M_I"] and r["super_k_ok"]
    # With two doublets below M_I, alpha_2 and alpha_3 run with the same coefficients above and below M_I
    # (b3 = c3 = -7, b2 = c2L = -3), so M_GUT does not depend on the Delta_R multiplicity.
    MG = {r["delta_R"]: r["M_GUT"] for r in rows if r["bidoublets"] == 1 and r["doublets_below"] == 2}
    assert MG[1] == pytest.approx(MG[2]) == pytest.approx(MG[3])
    # One extra bidoublet moves the one-doublet M_GUT from 4.5e16 to 1.4e16: the window is not robust.
    extra = [r for r in rows if r["delta_R"] == 1 and r["bidoublets"] == 2 and r["doublets_below"] == 1][0]
    assert 1.2e16 < extra["M_GUT"] < 1.6e16


def test_r_times_M_matches_the_flux_module():
    import bpr.six_dim_flux_vacuum as v
    rel = v.four_d_relations()
    g = sp.Symbol("g", positive=True)
    expr = 2 * sp.pi ** sp.Rational(1, 4) * sp.sqrt(g / v.m)  # 1/(rM) in terms of g4
    assert sp.simplify(rel["inverse_radius_over_M_in_terms_of_g4"]) == 0
    for gv in (0.02, 0.1, 0.4):
        assert 1 / ms.r_times_M(gv) == pytest.approx(float(expr.subs({g: gv, v.m: 3})), rel=1e-12)
        assert ms.inverse_radius(gv) == pytest.approx(2 * gv * ms.REDUCED_PLANCK_GEV / 3)


def test_control_window_is_narrow_but_nonempty():
    for doublets in (1, 2):
        u = ms.unify_3221(doublets)
        w3 = ms.control_window(u["M_GUT"], 3.0)
        assert w3["nonempty"] and w3["rM_range"][0] == pytest.approx(3.0)
        assert w3["inverse_radius_range"][0] == pytest.approx(u["M_GUT"])
        # M_GUT <= 1/r caps r M at r_times_M(g_F_min): about 3.9 (one light doublet) or 5.5 (two).
        cap = ms.r_times_M(w3["g_F_min"])
        assert cap == pytest.approx(w3["rM_range"][1]) and 3.5 < cap < 6
        assert not ms.control_window(u["M_GUT"], cap * 1.01)["nonempty"]
        assert ms.control_window(u["M_GUT"], cap * 0.99)["nonempty"]


def test_seesaw_needs_a_small_dirac_yukawa_at_M_I():
    u = ms.unify_3221(1)
    y = ms.seesaw_dirac_yukawa(u["M_I"])
    assert 1e-4 < y < 1e-2  # far below a top-like Dirac coupling: the known minimal-SO(10) seesaw tension
    assert ms.seesaw_dirac_yukawa(6e14) == pytest.approx(1.0, rel=0.05)


def test_report():
    report = ms.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == ms.LIMITATIONS
