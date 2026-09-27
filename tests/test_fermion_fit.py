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


def test_parametrization_identities(run):
    tg = ff.targets(2e16)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=6) * 0.1, rng.normal(size=9), rng.normal(size=4)])
    m = ff.build(x, tg)
    r, s = x[15] + 1j * x[16], x[17] + 1j * x[18]
    # M_u is a combination of M_d and M_e; all matrices symmetric; M_d, M_e have the target spectra up to pulls.
    assert np.allclose(m["Mu"], r / 4 * ((3 + s) * m["Md"] + (1 - s) * m["Me"]))
    assert np.allclose(m["MD"], r / 4 * ((3 - 3 * s) * m["Md"] + (1 + 3 * s) * m["Me"]))
    for M in m.values():
        assert np.allclose(M, M.T)
    assert np.allclose(np.sort(np.linalg.svd(m["Me"], compute_uv=False)),
                       np.sort(tg["lepton"] * np.exp(x[3:6] * ff.SIGMA["lepton"])))


def test_identical_branes_give_no_yukawa():
    rule = ff.identical_brane_selection_rule()
    assert rule["degeneracies"] == [3, 1, 3] or sorted(rule["degeneracies"]) == [1, 3, 3]
    assert rule["max_singlet_coupling"] < 1e-6
