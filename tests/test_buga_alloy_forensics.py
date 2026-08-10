"""Lock in the Buga-alloy forensic optimization — a SPHERE test, not a BPR test.

Guards two things: (a) the composition is far from every quasicrystal phase
field (the only BPR-adjacent objective), and (b) the module never claims to test
BPR. If a later edit tries to spin the metallurgy into BPR support, these fail.
"""
import pytest

from bpr.buga_alloy_forensics import (
    atomic_percent,
    electron_per_atom,
    quasicrystal_proximity,
    rare_earth_vs_mischmetal,
    forensic_report,
)


def test_bulk_is_dilute_aluminum():
    """97.5 at% Al — essentially pure aluminium, not a concentrated alloy."""
    at = atomic_percent()
    assert at["Al"] == pytest.approx(97.5, abs=0.3)
    assert sum(at.values()) == pytest.approx(100.0, abs=1e-6)


def test_electron_per_atom_far_from_quasicrystal_window():
    """e/a ≈ 2.93 sits near pure Al (3.0), far above the QC window 1.75-2.10."""
    ea = electron_per_atom()
    assert ea == pytest.approx(2.93, abs=0.05)
    assert ea > 2.1   # outside the Hume-Rothery quasicrystal window


def test_far_from_every_known_quasicrystal_phase_field():
    """KEY NEGATIVE: the composition is FAR from all known QC phase fields, so it
    is not optimized for quasicrystal formation (the only BPR-adjacent objective)."""
    for s in quasicrystal_proximity():
        assert "FAR" in s.verdict
        assert s.distance > 10.0   # tens of atomic-% away


def test_rare_earths_match_natural_mischmetal():
    """La:Ce:Nd:Pr matches natural bastnaesite/mischmetal — a terrestrial ore
    fingerprint, not engineered single-rare-earth doping."""
    s = rare_earth_vs_mischmetal()
    assert "MATCH" in s.verdict
    assert s.distance < 2.0


def test_report_disclaims_bpr_test():
    """HONESTY GUARD: the report must explicitly state it does NOT test BPR and
    must point to the flavor sector + the missing isotope measurement."""
    txt = forensic_report()
    assert "does not test BPR" in txt
    assert "ISOTOPE" in txt
    assert "flavor" in txt.lower()
    # and the bulk verdict must be terrestrial/not-quasicrystal
    assert "terrestrial secondary aluminium" in txt
    assert "NOT optimized for quasi" in txt
