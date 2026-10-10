"""Tripwires for the legacy GUT sector relabelled in 2026-10.

Found in the 2026-10 review of unmerged branches (the GUT audit on
claude/audit-bpr-parameters-6fSrQ) and re-checked on the current tree; see
doc/derivations/forward_gut_thresholds.md and doc/CLOSED_AND_DEPRECATED.md.
These tests pin labels and arithmetic. They assert no agreement with
experiment:

- the backward-fit "predictions" are identities, so the consistency check
  built on them is NON-DISCRIMINATING (and falls back to PASS / TENSION /
  FAIL if its two routes ever diverge, i.e. a wiring regression);
- the forward threshold factors are PHENOMENOLOGICAL (post hoc), with the
  look-elsewhere counts quoted in the docs;
- docstring scales, the ProtonDecay coupling and Super-K bound, and the
  separation of routes in alpha_em_full_chain;
- the stipulated sigma_xy shift in topological_phase_boundary.
"""

import inspect
import itertools
import math
from fractions import Fraction

import pytest

from bpr import gauge_unification as gu

P = 104761
PRIMES = (101, 104761, 1000003)


# ---------------------------------------------------------------------------
# Helpers: the forward-threshold formula of
# GaugeCouplingRunning.forward_threshold_corrections, with the group factors
# and z as free arguments, on the module's own SM couplings at M_GUT.
# ---------------------------------------------------------------------------

def _bare_and_lengths(p=P):
    g = gu.GaugeCouplingRunning(p=p)
    inv_a1, inv_a2, inv_a3, _ = g._sm_couplings_at_mgut
    return [inv_a1, inv_a2, inv_a3], p ** (1.0 / 3.0), math.log(p) / 4.0


def _closure(Y2, T2, T3, L_eff=None, p=P):
    s, N_B, L_above = _bare_and_lengths(p)
    L = L_above if L_eff is None else L_eff
    d = [N_B * 0.6 * Y2 / 3 * L / (2 * math.pi),
         N_B * T2 / 3 * L / (2 * math.pi),
         N_B * T3 / 3 * L / (2 * math.pi)]
    c = [s[i] + d[i] for i in range(3)]
    avg = sum(c) / 3
    res = max(abs(x - avg) for x in c)
    gap_code = abs(s[0] - (s[1] + s[2]) / 2)
    avg_bare = sum(s) / 3
    gap_consistent = max(abs(x - avg_bare) for x in s)
    return 1 - res / gap_code, 1 - res / gap_consistent


# ---------------------------------------------------------------------------
# Backward-fit identities and the checks that rest on them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", PRIMES)
def test_backward_fit_returns_inputs_for_any_p(p):
    g = gu.GaugeCouplingRunning(p=p)
    assert g.alpha_s_prediction == pytest.approx(gu._ALPHA_S_MZ, abs=1e-12)
    assert g.weinberg_angle_at_MZ == pytest.approx(gu._SIN2_TW, abs=1e-12)
    assert g.alpha_em_prediction == pytest.approx(gu._ALPHA_EM_MZ, rel=1e-12)
    assert g.boundary_threshold_corrections["status"].startswith("IDENTITY")


@pytest.mark.parametrize("p", PRIMES)
def test_sin2_consistency_check_is_non_discriminating(p):
    from bpr.consistency import NON_DISCRIMINATING, _check_sin2_theta_w
    c = _check_sin2_theta_w(p, 6)
    assert c["status"] == NON_DISCRIMINATING
    assert c["route_A"] == pytest.approx(0.23122, abs=1e-12)
    assert c["route_B"] == pytest.approx(0.23122, abs=1e-12)


def test_alpha_consistency_check_is_non_discriminating():
    from bpr.consistency import NON_DISCRIMINATING, _check_alpha_em
    c = _check_alpha_em(P, 6)
    if c["status"] == "SKIP":
        pytest.skip(c.get("reason", "route unavailable"))
    assert c["status"] == NON_DISCRIMINATING
    assert c["route_A"] == c["route_B"]


@pytest.mark.parametrize("factor, expected", [(1.02, "TENSION"), (1.30, "FAIL")])
def test_alpha_check_flags_a_wiring_regression(monkeypatch, factor, expected):
    # 2026-10: NON-DISCRIMINATING only while the routes agree to < 1e-9 %;
    # a diverged route B (wiring regression) falls back to PASS/TENSION/FAIL.
    import bpr.consistency as cons
    if not (cons._HAS_ALPHA and cons._HAS_CROSS):
        pytest.skip("alpha_derivation or cross_predictions unavailable")
    real = cons.full_cosmological_chain

    def shifted(p, z):
        out = dict(real(p, z))
        out["inv_alpha_predicted"] = out["inv_alpha_predicted"] * factor
        return out

    monkeypatch.setattr(cons, "full_cosmological_chain", shifted)
    c = cons._check_alpha_em(P, 6)
    assert c["status"] == expected
    assert c["status"] != cons.NON_DISCRIMINATING


@pytest.mark.parametrize("value, expected", [(0.23122 * 1.03, "TENSION"), (0.30, "FAIL")])
def test_sin2_check_flags_a_wiring_regression(monkeypatch, value, expected):
    import bpr.consistency as cons
    if not (cons._HAS_GAUGE and cons._HAS_PARTICLES):
        pytest.skip("gauge_unification or particles_matter bridge unavailable")
    real = cons.impedance_weinberg_angle

    def shifted(p):
        out = dict(real(p=p))
        out["sin2_theta_W_gauge_running"] = value
        return out

    monkeypatch.setattr(cons, "impedance_weinberg_angle", shifted)
    c = cons._check_sin2_theta_w(P, 6)
    assert c["status"] == expected
    assert c["status"] != cons.NON_DISCRIMINATING
    # the unpatched routes stay NON-DISCRIMINATING (identity to the last bit)
    monkeypatch.setattr(cons, "impedance_weinberg_angle", real)
    assert cons._check_sin2_theta_w(P, 6)["status"] == cons.NON_DISCRIMINATING


def test_v_ew_chain_uses_alpha_s_input():
    lo = gu.electroweak_scale_self_consistent(p=101)["self_consistent_chain"]
    hi = gu.electroweak_scale_self_consistent(p=P)["self_consistent_chain"]
    assert lo["alpha_s_MZ"] == hi["alpha_s_MZ"] == pytest.approx(0.1179, abs=1e-12)
    assert lo["Lambda_QCD_GeV"] == hi["Lambda_QCD_GeV"]
    assert "INPUT" in hi["status"]
    assert not hi["status"].startswith("DERIVED")


# ---------------------------------------------------------------------------
# Forward thresholds: phenomenological (post hoc)
# ---------------------------------------------------------------------------

def test_forward_thresholds_status_and_metrics():
    f = gu.GaugeCouplingRunning(p=P).forward_threshold_corrections
    assert f["status"].startswith("PHENOMENOLOGICAL")
    assert "FORWARD-DERIVED" not in f["status"].split("Previously")[0]
    assert f["fraction_closed"] == pytest.approx(0.9814, abs=1e-4)
    assert f["fraction_closed_consistent"] == pytest.approx(0.9721, abs=1e-4)
    # the helper reproduces the module's own numbers
    code, consistent = _closure(19 / 6, 1 / 7, 1 / 49)
    assert code == pytest.approx(f["fraction_closed"], abs=1e-12)
    assert consistent == pytest.approx(f["fraction_closed_consistent"], abs=1e-12)


def test_forward_threshold_look_elsewhere_counts():
    z = 6
    Ys = sorted({Fraction(z, 2), Fraction(3 * z + 1, 6), Fraction(z, 2) + Fraction(1, z),
                 Fraction(z * z + 2, 2 * z), Fraction(z, 3), Fraction(1)})
    T2s = sorted({Fraction(0), Fraction(1, z + 1), Fraction(1, z), Fraction(1, z - 1),
                  Fraction(1, 2)})
    T3s = sorted({Fraction(0), Fraction(1, (z + 1) ** 2), Fraction(1, z * z),
                  Fraction(1, z + 1)})
    rows = [_closure(float(y), float(t), float(u))[0]
            for y, t, u in itertools.product(Ys, T2s, T3s)]
    assert len(Ys) == 4 and len(rows) == 80
    assert sum(r >= 0.90 for r in rows) == 31
    assert sum(r >= 0.98 for r in rows) == 5


def test_rigidity_alone_and_t3_sign():
    assert _closure(3.0, 0.0, 0.0)[0] == pytest.approx(0.9452, abs=1e-4)
    with_t3 = _closure(19 / 6, 1 / 7, 1 / 49)[0]
    without_t3 = _closure(19 / 6, 1 / 7, 0.0)[0]
    assert without_t3 == pytest.approx(0.9888, abs=1e-4)
    assert without_t3 > with_t3  # T3 channel moves alpha_3 the wrong way
    s, _, _ = _bare_and_lengths()
    assert s[2] == max(s)        # 1/alpha_3 is already the largest


def test_y2_identity_holds_only_at_z6():
    hits = [z for z in range(1, 200)
            if Fraction(z * z + 2, 2 * z) == Fraction(3 * z + 1, 6)]
    assert hits == [6]


def test_stated_mode_spectrum_gives_much_less():
    _, N_B, L_above = _bare_and_lengths()
    N = round(N_B)
    L_spec = sum(L_above * k / N for k in range(1, N + 1)) / N
    code, consistent = _closure(19 / 6, 1 / 7, 1 / 49, L_eff=L_spec)
    assert code == pytest.approx(0.667, abs=0.002)
    assert consistent == pytest.approx(0.500, abs=0.002)


@pytest.mark.parametrize("z, expected", [(4, 0.7567), (8, 0.7918), (12, 0.3539)])
def test_formula_is_tuned_to_z6(z, expected):
    fz = _closure((3 * z + 1) / 6, 1 / (z + 1), 1 / (z + 1) ** 2)[0]
    assert fz == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Scales, ProtonDecay, docstrings
# ---------------------------------------------------------------------------

def test_gut_scale_is_6_79e17_and_docs_say_so():
    m = gu.GaugeCouplingRunning(p=P).unification_scale_GeV
    assert m == pytest.approx(6.786e17, rel=1e-3)
    assert "6.79 × 10¹⁷" in gu.__doc__
    from bpr import gut_winding_spectrum as gws
    est_doc = gws.GUTThresholdEstimate.__doc__
    # the old attribution survives only inside the dated correction note
    assert "Unification scale (BPR: M_Pl / p^{1/4} ≈ 2e16" not in est_doc
    assert "6.79e17" in est_doc


def test_gut_winding_thresholds_do_not_depend_on_m_gut():
    from bpr.gut_winding_spectrum import GUTThresholdEstimate, gut_threshold_su5_assigned
    a = GUTThresholdEstimate(M_GUT_GeV=2.0e16).threshold_delta_inv_alpha()
    b = GUTThresholdEstimate(M_GUT_GeV=6.786e17).threshold_delta_inv_alpha()
    assert a == pytest.approx(b, rel=1e-12)
    x = gut_threshold_su5_assigned(M_GUT_GeV=2.0e16)["delta_inv_alpha"]
    y = gut_threshold_su5_assigned(M_GUT_GeV=6.786e17)["delta_inv_alpha"]
    assert x == y


def test_proton_decay_flags_and_superk_bound():
    from bpr import model_scales
    assert gu._SUPER_K_TAU_YR == model_scales.SUPER_K_TAU_YR == 2.4e34
    doc = " ".join(gu.ProtonDecay.__doc__.split())
    assert "STIPULATED and NOT IMPLEMENTED" in doc
    assert "HARD-CODED" in gu.ProtonDecay.lifetime_years.__doc__
    # A lifetime of 2.0e34 yr passed the old 1.6e34 bound but fails 2.4e34.
    tau_ref = gu.ProtonDecay(p=P).lifetime_years
    m_ref = gu.ProtonDecay(p=P).M_GUT_GeV
    pd = gu.ProtonDecay(p=P, M_GUT_GeV=m_ref * (2.0e34 / tau_ref) ** 0.25)
    assert pd.lifetime_years == pytest.approx(2.0e34, rel=1e-9)
    assert pd.exceeds_superK is False


def test_three_inconsistent_unified_couplings_documented():
    from bpr.alpha_derivation import alpha_gut_from_lattice
    assert 1 / alpha_gut_from_lattice(P, 6) == pytest.approx(90.03, abs=0.01)
    assert 1 / gu.GaugeCouplingRunning(p=P).alpha_gut == pytest.approx(48.60, abs=0.01)
    src = inspect.getsource(gu.ProtonDecay.lifetime_years.fget)
    assert "alpha_gut = 1.0 / 40.0" in src


# ---------------------------------------------------------------------------
# Bridges: alpha_em_full_chain, topological_phase_boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", (101, P))
def test_alpha_full_chain_does_not_mix_scales(p):
    from bpr.bridges.particles_matter import alpha_em_full_chain
    r = alpha_em_full_chain(p=p)
    assert list(r["routes_q2_zero_independent"]) == ["substrate_formula"]
    excluded = r["routes_excluded"]
    assert excluded["gut_running_MZ_identity"]["inv_alpha"] == pytest.approx(127.952, abs=1e-9)
    assert excluded["gut_running_MZ_identity"]["scale"] == "M_Z"
    assert excluded["impedance_as_coded"]["inv_alpha"] == pytest.approx(2.0, abs=1e-6)
    assert math.isnan(r["route_mean"]) and math.isnan(r["route_spread"])
    assert r["internal_consistency"].startswith("not assessable")


def test_topological_phase_boundary_shift_is_stipulated():
    from bpr.bridges.particles_matter import topological_phase_boundary
    r = topological_phase_boundary(p=P)
    assert r["status"] == "stipulated_deviation"
    assert r["bpr_correction"] == pytest.approx(math.log(P) / (4 * math.pi * P), rel=1e-12)
    assert "STIPULATED" in topological_phase_boundary.__doc__
