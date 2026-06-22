"""Lock in what the phason-defect Lagrangian pins — and what it must NOT claim.

The Lagrangian fixes c and τ from structure (genuine progress) but leaves the
defect-core coupling open. These tests guard both halves so a later edit can't
upgrade the single-site HYPOTHESIS into a pretended derivation of (1/p)^7.
"""
import pytest

from bpr.phason_defect_lagrangian import (
    FREE_ENERGY_TERMS,
    energy_convention_c,
    protection_count_tau,
    bulk_exponent,
    core_scenarios,
    lagrangian_pins,
    report,
)


def test_free_energy_has_three_terms_coupling_not_structural():
    """F has phonon, phason, and coupling terms; only the coupling D is
    substrate-generated (not fixed by symmetry alone)."""
    names = {t.name for t in FREE_ENERGY_TERMS}
    assert "phonon elastic" in names
    assert "phason elastic" in names
    assert "phonon–phason coupling" in names
    coupling = [t for t in FREE_ENERGY_TERMS if "coupling" in t.name][0]
    assert coupling.fixed_by_structure is False


def test_energy_convention_pinned_to_two():
    """c = 2 is PINNED: quadratic free energy ⇒ energy ∝ |amplitude|²."""
    c = energy_convention_c()
    assert c["c"] == 2
    assert c["pinned"] is True


def test_protection_pinned_to_one():
    """τ = 1 is PINNED: the dislocation winds along a single internal direction."""
    t = protection_count_tau()
    assert t["tau"] == 1
    assert t["pinned"] is True


def test_bulk_exponent_is_six_for_nine_fold():
    """k_bulk = 2(d_⊥ − 1) = 6 for the 9-fold class (d_⊥ = 4)."""
    assert bulk_exponent(9) == 6


def test_bulk_exponent_is_two_for_rank4_classes():
    """Rank-4 classes (d_⊥ = 2) give k_bulk = 2."""
    for n in (5, 8, 12):
        assert bulk_exponent(n) == 2


def test_core_scenarios_bracket_six_to_eight():
    """The three core scalings give k ∈ {6, 7, 8} for the 9-fold class."""
    ks = sorted(s.total_k for s in core_scenarios(9))
    assert ks == [6, 7, 8]


def test_only_single_site_core_is_flagged_natural():
    """Exactly one scenario (g_core ~ 1/p ⇒ k=7) is the natural reading."""
    natural = [s for s in core_scenarios(9) if s.natural]
    assert len(natural) == 1
    assert natural[0].total_k == 7
    assert natural[0].extra_power == 1


def test_lagrangian_does_not_claim_full_derivation():
    """CRITICAL HONESTY GUARD: the Lagrangian pins c and τ but NOT the core, so
    the result must report fully_derived = False and core_pinned = False. If a
    later edit flips either, this test fails — the retrofit tripwire."""
    p = lagrangian_pins(9)
    assert p["c_pinned"] is True
    assert p["tau_pinned"] is True
    assert p["core_pinned"] is False
    assert p["fully_derived"] is False
    # the natural total is 7, but it is explicitly a hypothesis, not derived
    assert p["natural_total_k"] == 7


def test_report_states_pinned_and_hypothesis():
    txt = report(9)
    assert "PINNED" in txt
    assert "HYPOTHESIS" in txt or "hypothesis" in txt
    assert "does not close it" in txt
