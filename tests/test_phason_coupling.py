"""Tests for the two honest attacks on the 31-order phason-lift gap.

These lock in (a) the NEGATIVE mode-census result so it cannot silently regress
into a "we found a new lift channel" claim, and (b) the OPEN scale-ladder result
so the (1/p)^7 coincidence stays flagged as a retrofit risk, never a derivation.
"""
import math

import pytest

from bpr.phason_coupling import (
    P_SUBSTRATE,
    torus_homotopy_rank,
    postulate0c_mode_census,
    epsilon_scale_ladder,
    gap_attack_report,
)


# --- Route 2: mode census (must stay a clean negative) ---------------------

def test_torus_higher_homotopy_vanishes():
    """π_1(T^n) = Z^n (rank n); π_k = 0 for k=0 and k>=2. Exact."""
    for d_perp in (2, 4, 6):
        assert torus_homotopy_rank(0, d_perp) == 0       # connected
        assert torus_homotopy_rank(1, d_perp) == d_perp  # dislocations
        assert torus_homotopy_rank(2, d_perp) == 0       # no point defects
        assert torus_homotopy_rank(3, d_perp) == 0       # no textures


def test_only_one_lift_channel():
    """The census must find exactly ONE lift channel (the π_1 dislocations)."""
    c = postulate0c_mode_census(n=9)
    assert c["n_lift_channels"] == 1
    lift = [ch for ch in c["channels"] if ch.couples_to_lift][0]
    assert "dislocation" in lift.name
    assert lift.rank_of_group == c["d_perp"]


def test_census_d_perp_matches_rank_minus_two():
    """9-fold → rank φ(9)=6 → d_⊥ = 4, χ = 4."""
    c = postulate0c_mode_census(n=9)
    assert c["embedding_rank"] == 6
    assert c["d_perp"] == 4
    assert c["total_topological_charge_rank"] == 4


def test_census_verdict_is_negative():
    """The verdict must state the sector is COMPLETE — no rescue channel."""
    c = postulate0c_mode_census(n=9)
    assert "COMPLETE" in c["verdict"]
    # every non-lift channel is either perturbative or a vanishing homotopy group
    for ch in c["channels"]:
        if not ch.couples_to_lift:
            assert ch.rank_of_group == 0


# --- Route 1: scale ladder (must stay OPEN, retrofit flagged) ---------------

def test_required_power_is_about_seven():
    """ε_required corresponds to ~7 powers of 1/p (diagnostic, not a claim)."""
    r = epsilon_scale_ladder()
    assert r["power_of_inv_p_needed"] == pytest.approx(7.0, abs=0.2)


def test_no_hypothesis_is_marked_derived():
    """CRITICAL HONESTY GUARD: nothing in the ladder may claim to be derived.
    If a future edit marks the (1/p)^7 row derived without a real derivation,
    this test fails — that is the retrofit tripwire."""
    r = epsilon_scale_ladder()
    assert r["any_derived"] is False


def test_single_factor_suppressions_are_far_too_large():
    """α, 1/p, 1/p², lepton ratio all sit >>25 orders above ε_required."""
    r = epsilon_scale_ladder()
    for row in r["ladder"]:
        if row["label"] in ("ε ~ α_em", "ε ~ 1/p", "ε ~ 1/p²", "ε ~ (m_e/m_τ)"):
            assert row["orders_vs_required"] > 25


def test_inv_p7_row_is_flagged_retrofit():
    """The one numerically-close row must carry the retrofit warning."""
    r = epsilon_scale_ladder()
    row = [x for x in r["ladder"] if x["label"] == "ε ~ (1/p)^7"][0]
    assert "RETROFIT" in row["note"].upper()
    # and it is NOT marked derived
    assert row["derived"] is False
    # numerically within a couple orders of the requirement
    assert abs(row["orders_vs_required"]) < 2.0


def test_inv_p_constant():
    assert P_SUBSTRATE == 104761
    assert math.log10(1.0 / P_SUBSTRATE) == pytest.approx(-5.0202, abs=1e-3)


def test_report_mentions_both_routes_and_net():
    txt = gap_attack_report()
    assert "ROUTE 2" in txt and "ROUTE 1" in txt
    assert "NEGATIVE" in txt
    assert "OPEN" in txt
    assert "retrofit" in txt.lower()
