"""Gates 2-4 lock-in: emergent quantum numbers, selection rules, FAIL verdicts.

These freeze the opened-envelope result so it cannot be quietly softened:
the condensate branch of BPR FAILS the glueball benchmark. If a later edit
flips a verdict without new physics (quantized lump branch, phason parity,
beyond-leading-order interactions), these tests fail.
"""
import pytest

from bpr.glueball_gates import (
    s2_multiplets,
    omega,
    pair_J0_J2_from_two_J1,
    lightest_pseudoscalar,
    spectrum_ratios,
    gamma_scan,
    open_envelope,
    report,
    _mult, _chi, _sym2, _sym3,
)


# --- emergent quantum numbers (nothing inserted by hand) --------------------

def test_so3_multiplets_emerge_with_correct_degeneracy_and_parity():
    """Cross-m clustering discovers deg = 1,3,5,7 -> J = 0..3; eps = J(J+1);
    parity = (-1)^J from the antipodal map. All emergent."""
    mult = s2_multiplets(n_theta=400, m_max=4, k_per_m=6, n_clusters=4)
    for i, m in enumerate(mult):
        assert m["degeneracy"] == 2 * i + 1
        assert m["J_emergent"] == i
        assert m["eps"] == pytest.approx(i * (i + 1), abs=0.02)
        assert m["parity"] == (-1) ** i


# --- composite selection rules (character integrals, computed) --------------

def test_two_J1_quasiparticles_give_exactly_0pp_and_2pp():
    """The nontrivial Gate-2 positive: sym^2(J=1) = {0, 2}, J=1 bose-forbidden."""
    pair = pair_J0_J2_from_two_J1()
    assert pair["J0"] == 1 and pair["J2"] == 1 and pair["J1"] == 0
    assert pair["parity"] == +1


def test_pseudoscalar_selection_rules():
    """J=0^- forbidden for <=2 quasiparticles and for the near-light triples."""
    assert _mult(_sym3(1), 0) == 0                    # (1,1,1): no J=0
    assert _mult(_sym2(2) * _chi(1), 0) == 0          # (2,2,1): no J=0
    assert _mult(_sym2(2) * _chi(3), 0) == 0          # (2,2,3): no J=0
    assert _mult(_sym3(3), 0) == 0                    # (3,3,3): no J=0
    assert _mult(_chi(2) * _chi(3) * _chi(4), 0) == 1  # (2,3,4): allowed


def test_lightest_pseudoscalar_is_2_3_4():
    ps = lightest_pseudoscalar(gamma=1.0)
    assert ps["modes"] == (2, 3, 4)


# --- the spectrum vs the sealed targets -------------------------------------

def test_tensor_ratio_never_reaches_band():
    """2++/0++ <= 1.00 for every gamma (degenerate pair or lighter single),
    never the sealed band [1.25, 1.55]."""
    for r in gamma_scan():
        assert r["r_2pp"] <= 1.0 + 1e-12


def test_pseudoscalar_ratio_far_above_band():
    """0-+/0++ >= 3.6 for every gamma, vs sealed target 1.497."""
    for r in gamma_scan():
        assert r["r_0mp"] > 3.6


def test_lightest_state_is_a_vector_below_the_scalar():
    """The spectrum's lightest excitation is a single 1^- at half M(0++) —
    a family with no counterpart in the low glueball spectrum."""
    r = spectrum_ratios(1.0)
    assert r["lightest_state"]["JPC"] == "1-+"
    assert r["lightest_state"]["E"] / r["M_0pp"] == pytest.approx(0.5, abs=1e-9)


# --- verdicts locked ---------------------------------------------------------

def test_envelope_verdicts_locked():
    env = open_envelope()
    assert env["gate2"]["verdict"] == "PARTIAL FAIL"
    assert env["gate2"]["families_0pp_2pp_emerge"] is True     # the real positive
    assert env["gate2"]["light_0mp_forbidden"] is True
    assert env["gate3"]["verdict"] == "FAIL"
    assert env["gate4"]["verdict"] == "FAIL"
    assert env["gate4"]["any_gamma_passes_2pp"] is False
    assert env["gate4"]["any_gamma_passes_0mp"] is False
    assert "FAILS" in env["overall"]
    assert len(env["loopholes"]) == 3


def test_report_states_opened_and_failed():
    txt = report()
    assert "OPENED" in txt
    assert "FAILS the glueball benchmark" in txt
    assert "Loopholes" in txt
