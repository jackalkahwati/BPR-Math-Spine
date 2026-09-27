"""Checks for doc/derivations/phase2_fermion_fit_2026-09-27.md (Phase 2).

Oracles: the SM gauge couplings from model_scales' one-loop running, known qualitative SM running (b-tau
non-unification, |V_cb| growth, |V_us| and delta stability), the round trip of the CKM parametrization, the algebraic
identities of the construction, the rank of the full model modulo U(3) (28 - 9 - 1 = 18), the exact 3-fold selection
rule, exact membership of the pinned matrices in W (projection and the reachability search of Phase 1d), a positive
control for the pinned fit, and recomputation of every stored best-fit point from scratch.
"""

import json

import numpy as np
import pytest

import bpr.fermion_fit as ff
import bpr.model_scales as ms


@pytest.fixture(scope="module")
def run():
    return ff.run_to(2e16)


@pytest.fixture(scope="module")
def tg():
    return ff.targets(2e16)


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
    assert run["delta"] == pytest.approx(ff.CKM_MZ["delta"], abs=0.01)  # nor does the CKM phase
    assert 60 < run["up"][2] < 90 and 0.2 < run["up"][1] < 0.3  # m_t, m_c at 2e16 GeV (literature: ~70-80, ~0.22-0.25)
    assert run["lepton"][2] == pytest.approx(1.7, rel=0.05)


def test_ckm_parameters_round_trip_and_rephasing_invariance():
    rng = np.random.default_rng(2)
    for delta in (0.3, 1.144, 2.5, -1.0):
        x = np.array([0.22, 0.045, 0.004, delta])
        V = ff.ckm_matrix(*x)
        P1, P2 = (np.diag(np.exp(1j * rng.uniform(0, 6, 3))) for _ in range(2))
        assert np.allclose(ff.ckm_parameters(P1 @ V @ P2)[0], x, atol=1e-12)


def test_charged_sector_construction_is_exact(tg):
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.normal(size=10) * 0.3, rng.uniform(0, 6, 2), [30.0, -20.0], [0.7], [0.4],
                        rng.uniform(0, 6, 2)])
    m = ff.build(p, tg)
    # The residuals, recomputed from the matrices, return the parameters: masses, CKM and m_tau are exact.
    r = ff.residuals(p, tg, with_nu=False)
    assert np.allclose(r[:10], p[:10], atol=1e-9) and r[12] == pytest.approx(p[15], abs=1e-9)
    # The SO(10) relations hold identically.
    a = p[12] + 1j * p[13]
    b = m["r"] - a
    assert np.allclose(m["Mu"], a * m["Md"] + b * m["Me"])
    assert np.allclose(m["Md"], m["H"] + m["F"]) and np.allclose(m["Me"], m["H"] - 3 * m["F"])
    assert np.allclose(m["Mu"], m["r"] * (m["H"] + m["s"] * m["F"]))
    assert np.allclose(m["MD"], m["r"] * (m["H"] - 3 * m["s"] * m["F"]))
    for M in (m["Mu"], m["Md"], m["Me"], m["MD"], m["F"]):
        assert np.allclose(M, M.T)


def test_parametrization_is_locally_complete(tg):
    """Rank 18 = the full model modulo U(3) and arg r; without the two M_d phases (the first version) it is 16."""
    rng = np.random.default_rng(1)
    synthetic = {"up": np.array([0.3, 0.6, 1.0]), "down": np.array([0.2, 0.5, 0.9]),
                 "lepton": np.array([0.25, 0.55, 1.1]), "ckm": np.array([0.3, 0.2, 0.1, 1.0]), "nu": tg["nu"]}
    p = np.concatenate([rng.normal(size=10) * 0.3, rng.uniform(0, 6, 2), [1.3, -0.7], [0.7], [0.2],
                        rng.uniform(0, 6, 2)])
    assert ff.parametrization_rank(p, synthetic) == 18
    assert ff.parametrization_rank(p, synthetic, frozen=(16, 17)) == 16
    assert ff.parametrization_rank(ff.BEST_FIT_GENERIC, tg) == 18


def test_charged_sector_fits_exactly(tg):
    c, p = ff.fit(tg, with_nu=False, n_starts=4)
    assert c < 1e-6
    me = np.sort(np.linalg.svd(ff.build(p, tg)["Me"], compute_uv=False))
    assert np.allclose(me, tg["lepton"], rtol=1e-4)


def test_identical_branes_give_no_yukawa():
    rule = ff.identical_brane_selection_rule()
    assert sorted(rule["degeneracies"]) == [1, 3, 3]
    assert rule["max_singlet_coupling"] < 1e-6


def _local_improvement(fun, x, args):
    """chi^2 decrease from a local re-minimization (the first version's stored point dropped from 36 to 7)."""
    from scipy.optimize import least_squares
    return float(np.sum(fun(x, *args) ** 2)) - 2 * least_squares(fun, x, args=args, max_nfev=400).cost


def test_best_generic_fit_is_a_good_local_minimum(tg):
    p = ff.BEST_FIT_GENERIC
    c = ff.chi2(p, tg)
    assert c == pytest.approx(ff.STORED_CHI2["generic"], abs=0.01) and c < 10
    assert _local_improvement(ff.residuals, p, (tg,)) < 1e-3  # a minimum in all 18 directions
    r = ff.residuals(p, tg)
    assert r[3] < -1 and abs(r[3]) == max(abs(r))  # the largest pull: m_d, about 1.5 sigma low
    assert np.all(abs(r[-4:]) < 1)  # every neutrino observable within 1 sigma
    nu = ff.neutrino_sector(ff.build(p, tg))
    m1, m2, m3 = nu["light_masses_ev"]
    assert m1 < m2 < m3  # normal ordering
    assert m3 ** 2 - m1 ** 2 == pytest.approx(2.511e-3, rel=1e-6)  # the scale w is fixed by Delta m^2_31
    assert 0.05 < nu["sum_ev"] < 0.1 and nu["m_betabeta_ev"] < 1e-3


def test_prediction_ranges_over_near_best_minima(tg):
    ranges = ff.prediction_ranges(tg)
    assert len(ranges["chi2"]) >= 2 and max(ranges["chi2"]) < ff.STORED_CHI2["generic"] + 4
    lo, hi = ranges["sum_ev"]
    assert hi - lo < 0.01  # the sum of masses is stable ...
    assert ranges["m_betabeta_ev"][1] < 2e-3  # ... and m_betabeta stays far below planned sensitivities


def test_seesaw_scale_far_above_the_one_loop_intermediate_scale(tg):
    out = ff.seesaw_consistency(ff.build(ff.BEST_FIT_GENERIC, tg))
    assert out["M_I_one_loop_gev"] == pytest.approx(ms.unify_3221()["M_I"])
    assert out["ratio"] > 100


def test_error_assumption_dependence(tg):
    """With a 10% (plus theory) m_d error the generic fit is strained: the stored point of that search."""
    sg = dict(ff.SIGMA)
    sg["down"] = np.hypot(np.array([0.1, 0.15, 0.05]), ff.THEORY_ERROR)
    assert ff.chi2(ff.BEST_FIT_GENERIC_MD10, tg, sigma=sg) == pytest.approx(ff.STORED_CHI2["generic_md10"], abs=0.01)


def test_W_is_the_kernel_of_the_condensate_within_J2():
    """W = {Y symmetric: J = 0 part zero, and annihilated by the pinning condensate chi}."""
    P, _ = ff.tetrahedral_brane_matrices()
    for Pa in P:
        assert np.allclose(ff.w_conditions(Pa), 0, atol=1e-12)
    rng = np.random.default_rng(6)
    basis = []
    for i, j in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
        E = np.zeros((3, 3), complex)
        E[i, j] = E[j, i] = 1
        basis.append(ff.w_conditions(E))
    assert np.linalg.matrix_rank(np.array(basis).T, tol=1e-10) == 2  # two independent conditions: dim W = 6 - 2 = 4
    Y = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    assert not np.allclose(ff.w_conditions(Y + Y.T), 0)


def test_pinned_points_lie_exactly_in_W(tg):
    import bpr.brane_stabilization as b
    _, pts = ff.tetrahedral_brane_matrices()
    for key, x, with_nu in (("pinned", ff.BEST_FIT_PINNED, True),
                            ("pinned_charged_only", ff.BEST_FIT_PINNED_CHARGED, False)):
        m = ff.pinned_matrices(x)
        c = ff.pinned_chi2(x, tg, with_nu)
        assert c == pytest.approx(ff.STORED_CHI2[key], abs=0.01)
        assert _local_improvement(ff.pinned_residuals, x, (tg, with_nu)) < 1e-2
        # The first version reported a chi^2 off W; here the projection onto W changes nothing ...
        assert ff.matrix_chi2(ff.projected_onto_W(m), tg, with_nu=with_nu) == pytest.approx(c, rel=1e-9)
        # ... and the independent reachability search of Phase 1d confirms membership.
        assert b.reachability_cost(m["H"], m["F"], pts, starts=5) < 1e-20
        for M in (m["H"], m["F"]):
            assert np.allclose(ff.w_conditions(M), 0, atol=1e-10 * np.linalg.norm(M))


def test_pinned_model_fails_in_the_charged_sector(tg):
    r = ff.pinned_residuals(ff.BEST_FIT_PINNED, tg)
    assert ff.STORED_CHI2["pinned"] > 100 and r[4] > 5  # m_s far too large
    assert np.all(abs(r[-4:]) < 2)  # while the neutrino observables fit
    rc = ff.pinned_residuals(ff.BEST_FIT_PINNED_CHARGED, tg, with_nu=False)
    assert ff.STORED_CHI2["pinned_charged_only"] > 50 and rc[4] > 5


def test_pinned_fit_positive_control():
    """Targets generated by a random pinned point are refitted exactly from a perturbed start."""
    rng = np.random.default_rng(4)
    x0 = np.concatenate([rng.normal(size=16), [20.0, 5.0, 0.4, 0.3]])
    obs = ff.matrix_observables(ff.pinned_matrices(x0))
    synthetic = {"up": obs["up"], "down": obs["down"], "lepton": obs["lepton"], "ckm": obs["ckm"], "nu": obs["nu"]}
    assert ff.pinned_chi2(x0, synthetic) < 1e-20
    c, _ = ff.pinned_fit(synthetic, [x0 * np.exp(0.05 * rng.normal(size=20))], hops=0)
    assert c < 1e-8


def test_generic_best_is_not_reachable_with_tetrahedral_branes(tg):
    import bpr.brane_stabilization as b
    m = ff.build(ff.BEST_FIT_GENERIC, tg)
    _, pts = ff.tetrahedral_brane_matrices()
    assert b.reachability_cost(m["H"], m["F"], pts, starts=15) > 1e-4


def test_inverted_ordering_is_excluded():
    tio = ff.targets(2e16, ordering="IO")
    assert ff.chi2(ff.BEST_FIT_GENERIC_IO, tio) == pytest.approx(ff.STORED_CHI2["generic_IO"], rel=1e-6)
    assert ff.STORED_CHI2["generic_IO"] > 1000


def test_report():
    report = ff.demonstration_report()
    json.loads(json.dumps(report, default=str))
    assert report["empirical_validation"] is False and report["limitations"] == ff.LIMITATIONS
    assert report["generic"]["parametrization_rank"] == 18
