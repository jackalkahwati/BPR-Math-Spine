"""Lock the Z_p neutrality superselection derivation and its honest limits.

Guards: (a) the verified premises and consequences, (b) the flagged assumption
stays flagged, (c) the honest negative — quasiparticles are neutral, the light
1^- survives, and the glueball sector REMAINS CLOSED. If a later edit claims
this principle rescues the glueball sector, these tests fail.
"""
import pytest

from bpr.zp_selection_principle import (
    global_shift_invariance,
    charge_is_conserved,
    sectors_never_mix,
    physical_sector_dims,
    quasiparticle_is_neutral,
    derivation,
    consequences,
    report,
)


# --- premises and consequences (verified, not asserted) ---------------------

def test_premise1_global_shift_exactly_invariant():
    """The frozen action is exactly invariant under global phase shifts,
    including the elementary Z_p shift 2*pi/p."""
    assert global_shift_invariance() < 1e-10


def test_rule_consistent_with_dynamics():
    """[H, Q] = 0 exactly: the superselection rule survives time evolution."""
    assert charge_is_conserved() == 0.0


def test_sectors_never_mix():
    """H is exactly block-diagonal in charge — no cross-sector elements."""
    assert sectors_never_mix() == 0.0


def test_confinement_analog_and_zp_baryons():
    """Bare quanta unphysical; p-quanta composites physical; sector counting
    gives exactly dim/p physical states on the toy space."""
    d = physical_sector_dims(p_toy=3)
    assert d["bare_quantum_physical"] is False
    assert d["p_quanta_composite_physical"] is True
    assert d["physical_dim"] == d["total_dim"] // 3
    assert d["physical_dim"] == 9 and d["total_dim"] == 27


def test_quasiparticles_are_exactly_neutral():
    """Number-conserving Bogoliubov operators commute with Q exactly; bare
    creation operators carry charge 1. This is the source of the honest
    negative below."""
    qp = quasiparticle_is_neutral()
    assert qp["quasiparticle_charge_commutator"] == 0.0
    assert qp["bare_creation_carries_charge_1"] is True


# --- honesty guards ----------------------------------------------------------

def test_assumption_is_flagged_as_assumption():
    """The load-bearing premise (substrate self-containment -> redundancy)
    must stay explicitly flagged as an assumption, not silently promoted."""
    d = derivation()
    assert "ASSUMPTION" in [k for k in d if "premise_2" in k][0]
    assert "REDUNDANCY" in d["premise_2_ASSUMPTION"] or \
           "redundancy" in d["premise_2_ASSUMPTION"].lower()
    assert d["postulated_to_fix_a_spectrum"] is False


def test_glueball_sector_stays_closed():
    """THE CRITICAL GUARD: this principle does NOT remove the light 1^- (a
    neutral quasiparticle) and must not be claimed to reopen the glueball
    sector."""
    c = consequences()
    assert c["does_not_remove_light_1_minus"] is True
    assert "REMAINS CLOSED" in c["glueball_sector"]


def test_dead_ends_recorded():
    """The searched-and-rejected alternatives stay on record."""
    c = consequences()
    assert "QR grading" in c["dead_ends_searched"]
    assert "retrofit" in c["dead_ends_searched"]


def test_report_states_the_honest_negative():
    txt = report()
    assert "ASSUMPTION" in txt
    assert "NEUTRAL" in txt
    assert "SURVIVES this principle" in txt
    assert "REMAINS CLOSED" in txt
