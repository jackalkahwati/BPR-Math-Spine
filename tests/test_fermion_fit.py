"""Checks for doc/derivations/phase2_fermion_fit_2026-09-27.md (Phase 2).

Oracles: the SM gauge couplings from model_scales' one-loop running, known qualitative SM running (b-tau
non-unification, |V_cb| growth, |V_us| stability), the algebraic identities of the parametrization, the exact
3-fold selection rule, and recomputation of the stored best-fit points from scratch.
"""

import numpy as np
import pytest

import bpr.fermion_fit as ff
import bpr.model_scales as ms


@pytest.fixture(scope="module")
def run():
    return ff.run_to(2e16)


def test_gauge_running_matches_the_scales_module(run):
    inv = ms.low_energy_couplings()
    t = np.log(2e16 / ms.MZ)
    b = [float(x) for x in ms.sm_beta(1)]
    for g, i0, bi in zip(run["gauge"], inv, b):
        assert 4 * np.pi / g ** 2 == pytest.approx(i0 - bi * t / (2 * np.pi), rel=1e-6)


def test_known_features_of_sm_yukawa_running(run):
    assert 0.55 < run["mb_over_mtau"] < 0.7  # no b-tau unification in the SM
    assert run["Vcb"] > 1.1 * ff.CKM_MZ["s23"]  # V_cb grows
    assert run["Vus"] == pytest.approx(ff.CKM_MZ["s12"], rel=0.01)  # V_us barely runs
    assert 60 < run["up"][2] < 90 and 0.2 < run["up"][1] < 0.3  # m_t, m_c at 2e16 GeV (literature: ~70-80, ~0.22-0.25)
    assert run["lepton"][2] == pytest.approx(1.7, rel=0.05)


def test_charged_sector_construction_is_exact(run):
    tg = ff.targets(2e16)
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.normal(size=10) * 0.3, rng.uniform(0, 6, 2), [30.0, -20.0], [0.7], rng.normal(size=3) * 0.3])
    m = ff.build(p, tg)
    # Up masses and |V_CKM| are reproduced exactly (independent recomputation from the matrices).
    Uu, mu = ff._left(m["Mu"])
    Ud, md = ff._left(m["Md"])
    assert np.allclose(mu, tg["up"] * np.exp(p[0:3] * ff.SIGMA["up"]), rtol=1e-9)
    assert np.allclose(md, tg["down"] * np.exp(p[3:6] * ff.SIGMA["down"]), rtol=1e-12)
    V = Uu.conj().T @ Ud
    assert np.allclose(abs(V), abs(ff._ckm_from(p[6:10], tg)), atol=1e-9)
    # The SO(10) relations hold identically.
    a, b = p[12] + 1j * p[13], m["r"] - (p[12] + 1j * p[13])
    assert np.allclose(m["Mu"], a * m["Md"] + b * m["Me"])
    assert np.allclose(m["Md"], m["H"] + m["F"]) and np.allclose(m["Me"], m["H"] - 3 * m["F"])
    assert np.allclose(m["Mu"], m["r"] * (m["H"] + m["s"] * m["F"]))
    for M in (m["Mu"], m["Me"], m["MD"], m["F"]):
        assert np.allclose(M, M.T)
    # m_tau is exact; the other lepton masses are fit conditions.
    assert np.sort(np.linalg.svd(m["Me"], compute_uv=False))[2] == pytest.approx(
        tg["lepton"][2] * np.exp(p[17] * ff.SIGMA["lepton"][2]))


def test_charged_sector_fits_exactly():
    tg = ff.targets(2e16)
    c, p = ff.fit(tg, with_nu=False, n_starts=4)
    assert c < 1e-6
    m = ff.build(p, tg)
    me = np.sort(np.linalg.svd(m["Me"], compute_uv=False))
    assert np.allclose(me, tg["lepton"], rtol=1e-4)


def test_identical_branes_give_no_yukawa():
    rule = ff.identical_brane_selection_rule()
    assert rule["degeneracies"] == [3, 1, 3] or sorted(rule["degeneracies"]) == [1, 3, 3]
    assert rule["max_singlet_coupling"] < 1e-6


def test_best_generic_fit_is_reproducible_and_strained():
    tg = ff.targets(2e16)
    p = ff.BEST_FIT_GENERIC
    assert ff.chi2(p, tg) == pytest.approx(36.18, abs=0.05)
    r = ff.residuals(p, tg)
    assert r[3] < -3  # the down-quark mass is pulled more than 3 sigma low: the tension
    assert np.all(abs(r[-4:]) < 1)  # the neutrino observables themselves fit within 1 sigma
    nu = ff.neutrino_sector(p, tg)
    m1, m2, m3 = nu["light_masses_ev"]
    assert m1 < m2 < m3  # normal ordering
    assert (m2 ** 2 - m1 ** 2) / (m3 ** 2 - m1 ** 2) == pytest.approx(7.41e-5 / 2.511e-3, rel=0.05)
    assert m3 ** 2 - m1 ** 2 == pytest.approx(2.511e-3, rel=1e-6)  # the scale w is fixed by Delta m^2_31
    assert 0.05 < nu["sum_ev"] < 0.1 and nu["m_betabeta_ev"] < 0.01


def test_seesaw_scale_far_above_the_one_loop_intermediate_scale():
    tg = ff.targets(2e16)
    out = ff.seesaw_consistency(ff.BEST_FIT_GENERIC, tg)
    assert out["ratio"] > 100  # v_R >~ 3e12 GeV versus M_I ~ 1e9 GeV at one loop
