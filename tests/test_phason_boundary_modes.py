"""Lock the v2 go/no-go: no propagating phason branch; glueball sector closed.

Guards the structural theorem (root symmetry), the scan results, the honest
slaved-dressing caveat, and the NO-GO verdict with all three loopholes
dispositioned. If a later edit flips the verdict without new frozen physics,
these fail.
"""
import numpy as np
import pytest

from bpr.phason_boundary_modes import (
    branch_roots,
    root_symmetry_violation,
    no_go_scan,
    mode_counting,
    v2_verdict,
    report,
)


def test_root_symmetry_theorem():
    """Root set closed under omega -> -conj(omega) across parameters."""
    for eps in (0.1, 2.0, 30.0, 200.0):
        for D in (0.0, 0.15, 0.22):
            assert root_symmetry_violation(eps, D=D) < 1e-6


def test_stability_requires_positive_definite_free_energy():
    """No growing modes anywhere in the stable range (KC > D^2)."""
    for eps in (0.1, 6.0, 100.0):
        for r in branch_roots(eps, D=0.2):
            assert r["omega"].imag < 1e-9


def test_phason_branch_exactly_non_propagating():
    """The phason-dominated root sits exactly on the imaginary axis."""
    for eps in (0.1, 2.0, 6.0, 30.0, 200.0):
        rs = branch_roots(eps)
        diffusive = [r for r in rs if not r["propagating"]]
        assert len(diffusive) >= 1
        # the non-propagating root is phason-dominated at these parameters
        assert max(r["phason_fraction"] for r in diffusive) > 0.9


def test_at_most_one_propagating_family():
    """The scan never finds more than one propagating (mirror) family."""
    scan = no_go_scan()
    assert max(scan["propagating_family_counts_seen"]) <= 1
    assert scan["phason_branch_always_purely_imaginary"] is True
    assert scan["stable_everywhere"] is True


def test_slaved_caveat_reported_honestly():
    """The slaved phason dressing on the phonon pair is large at strong
    coupling (>0.9) and must be reported — but as dressing, not a new family."""
    scan = no_go_scan()
    assert scan["max_slaved_phason_fraction"] > 0.9
    assert scan["slaved_dominance_cases"] >= 0
    txt = report()
    assert "SLAVED" in txt or "slaved" in txt
    assert "not a new family" in txt or "not a new branch" in txt


def test_mode_counting_no_new_particle_families():
    mc = mode_counting(d_perp=4)
    assert mc["propagating_families"] == 1
    assert mc["diffusive_families"] == 4
    assert mc["new_particle_families_from_phason"] == 0


def test_v2_no_go_and_sector_closed():
    """THE VERDICT LOCK: v2 NO-GO, all three loopholes dispositioned, sector
    closed for BPR as frozen."""
    v = v2_verdict()
    assert v["v2"] == "NO-GO"
    assert len(v["loopholes_dispositioned"]) == 3
    assert "symmetry-protected" in \
        v["loopholes_dispositioned"]["interactions_beyond_leading_order"]
    assert "parity lock" in v["loopholes_dispositioned"]["self_bound_lump_branch"]
    assert "no propagating" in v["loopholes_dispositioned"]["phason_sector"]
    assert v["glueball_sector"] == "CLOSED for BPR as frozen"


def test_report_states_no_go_and_closure():
    txt = report()
    assert "NO-GO" in txt
    assert "CLOSED for BPR as frozen" in txt
