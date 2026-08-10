"""Lock Path B Milestone 1: gauged point-group kinematics + honesty guards.

Guards (a) the exact group theory (orthogonality, Burnside, quantum-double sum
rule), (b) the two structural results (A2 pseudoscalar charge, E_k doublets),
and (c) the honesty boundary: NO spectrum claims, milestones 2-4 OPEN, and the
module stays blind to every glueball number (mechanical grep, same discipline
as Benchmark v1).
"""
import inspect

import numpy as np
import pytest

import bpr.nonabelian_gauge_sector as ng
from bpr.nonabelian_gauge_sector import (
    ALLOWED_CLASSES,
    mul, inv, elements,
    conjugacy_classes,
    character_table,
    table_checks,
    pseudoscalar_charge,
    spin_doublets,
    anyon_content,
    milestone_status,
    report,
)


# --- group machinery is exactly right ---------------------------------------

def test_group_axioms_hold():
    """Associativity and inverses for D_n under the (k, eps) composition."""
    n = 8
    els = elements(n)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(els), size=(50, 3))
    for i, j, k in idx:
        a, b, c = els[i], els[j], els[k]
        assert mul(mul(a, b, n), c, n) == mul(a, mul(b, c, n), n)
    e = (0, 1)
    for g in els:
        assert mul(g, inv(g, n), n) == e


def test_class_and_irrep_counts_match():
    """#classes == #irreps for every allowed n (exact rep theory)."""
    for n in ALLOWED_CLASSES:
        ch = table_checks(n)
        assert ch["n_classes"] == ch["n_irreps"]


def test_character_orthogonality_and_burnside():
    for n in ALLOWED_CLASSES:
        ch = table_checks(n)
        assert ch["orthogonal"] is True
        assert ch["burnside"] is True


def test_quantum_double_sum_rule_exact():
    """sum d^2 = |G|^2 for the anyon content of D(D_n) — exact, all classes."""
    for n in ALLOWED_CLASSES:
        an = anyon_content(n)
        assert an["sum_rule_ok"] is True
        assert an["sum_d_squared"] == (2 * n) ** 2


# --- the two structural results ---------------------------------------------

def test_pseudoscalar_charge_exists_for_every_class():
    """THE HEADLINE: the A2 sign charge (rotation-invariant, reflection-odd —
    the discrete 0^- precursor) exists for every allowed class. This is what
    the Abelian scalar theory provably lacked."""
    for n in ALLOWED_CLASSES:
        ps = pseudoscalar_charge(n)
        assert ps["exists"] is True
        assert ps["rotation_invariant"] is True
        assert ps["reflection_odd"] is True


def test_spin_doublets_exist():
    """E_k doublets (discrete angular momentum +-k) exist for every class."""
    for n in ALLOWED_CLASSES:
        assert spin_doublets(n)["n_doublets"] >= 2


def test_gauged_theory_is_overwhelmingly_nonabelian():
    """Most anyon types have quantum dimension > 1 (non-Abelian content)."""
    for n in ALLOWED_CLASSES:
        an = anyon_content(n)
        assert an["n_nonabelian"] > an["n_anyon_types"] // 2


def test_d5_reference_values():
    """Spot-check against the exact hand computation for D_5:
    16 anyon types, 14 non-Abelian, sum d^2 = 100."""
    an = anyon_content(5)
    assert an["n_anyon_types"] == 16
    assert an["n_nonabelian"] == 14
    assert an["sum_d_squared"] == 100


# --- honesty guards ----------------------------------------------------------

def test_no_spectrum_claims_and_milestones_open():
    """Milestones 2-4 must remain OPEN and spectrum claims NONE until a
    dynamics is frozen. Flipping these requires real new frozen physics."""
    ms = milestone_status()
    assert ms["M2_dynamics_beyond_topological_point"].startswith("OPEN")
    assert ms["M3_flavor_sector_compatibility"].startswith("OPEN")
    assert ms["M4_sealed_benchmark_v3"].startswith("OPEN")
    assert ms["spectrum_claims"].startswith("NONE")
    assert "PROPOSED" in ms["postulate_0d_status"]


def test_module_is_blind_to_glueball_numbers():
    """MECHANICAL BLINDNESS GUARD (same discipline as Benchmark v1): the module
    source must not contain any sealed target or experimental number."""
    src = inspect.getsource(ng)
    for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370", "2359",
                   "2395", "X(23"):
        assert leaked not in src, f"gauge-sector module leaked: {leaked}"


def test_report_states_proposed_and_no_spectrum():
    txt = report()
    assert "PROPOSED" in txt
    assert "NONE" in txt
    assert "not invented" in txt
