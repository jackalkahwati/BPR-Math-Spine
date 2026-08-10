"""Lock Milestones 2-4: M3 PASS, M2 frozen form + exact Casimirs, M4 sealed.

Guards: (a) the exact M3 verification (charge sectors partition the spectrum —
gauging never shifts energies), (b) the flavor rep assignments, (c) the exact
Casimir values, (d) the honesty boundary — v3 stays SEALED, LO is
J^PC-degenerate, no spectrum numbers exist, and the module is blind to every
glueball target (mechanical grep).
"""
import inspect

import numpy as np
import pytest

import bpr.gauge_dynamics_m2_m3 as gd
from bpr.gauge_dynamics_m2_m3 import (
    FLAVOR_INTEGERS,
    dn_rep_of_mode,
    flavor_rep_table,
    spectrum_partition_check,
    m3_report,
    frozen_hamiltonian_spec,
    electric_casimir,
    casimir_table,
    lightest_charge_sector,
    strong_coupling_leading_order,
    benchmark_v3_status,
    report,
)


# --- M3: flavor survival -----------------------------------------------------

def test_charge_sectors_partition_spectrum_exactly():
    """THE M3 THEOREM CHECK: gauging relabels states by D_n charge but the
    sector spectra partition the full spectrum — no energy moves, for every
    allowed class."""
    for n in (5, 8, 9, 12):
        assert spectrum_partition_check(n) is True


def test_flavor_masses_unchanged():
    """Mass-formula inputs are gauge-inert; the registered LHCb pre-dictions
    are numerically unchanged by gauging."""
    m3 = m3_report()
    assert m3["any_input_gauged"] is False
    assert m3["energies_shift_under_gauging"] is False
    assert m3["lhcb_predictions_changed"] is False
    assert m3["verdict"].startswith("PASS")


def test_flavor_rep_assignments():
    """Spot-check the D_n charges of the quark modes (l mod n)."""
    assert dn_rep_of_mode(30, 5) == "A"        # 30 = 0 mod 5
    assert dn_rep_of_mode(4, 8) == "B"         # 4 = 8/2
    assert dn_rep_of_mode(24, 8) == "A"        # 24 = 0 mod 8
    assert dn_rep_of_mode(283, 9) == "E4"      # 283 = 4 mod 9
    assert dn_rep_of_mode(30, 9) == "E3"
    assert flavor_rep_table(12)["t"] == "E5"   # 283 = 7 mod 12 -> min(7,5)
    # top and charm land in the SAME rep for n=9 (both E3/E4 family checks)
    assert flavor_rep_table(9)["c"] == "E3"


def test_m3_caveats_recorded():
    """The lepton sqrt(210) wrinkle and the confinement-binding caveat must
    stay on record."""
    m3 = m3_report()
    assert any("sqrt(210)" in c for c in m3["caveats"])
    assert any("binding" in c for c in m3["caveats"])


# --- M2: frozen dynamics -----------------------------------------------------

def test_frozen_hamiltonian_has_one_coupling():
    spec = frozen_hamiltonian_spec()
    assert spec["n_couplings"] == 1
    assert "Delta_G" in spec["form"]
    assert spec["phase_location"].startswith("OPEN")


def test_exact_casimir_values():
    """Closed forms: eps(A1)=0, eps(A2)=2, eps(B1)=4, eps(B2)=6,
    eps(E_k) = 3 - 2cos(2 pi k/n) — computed from the verified char table."""
    for n in (5, 8, 9, 12):
        tab = casimir_table(n)
        assert tab["A1_trivial"] == pytest.approx(0.0, abs=1e-12)
        assert tab["A2_sign"] == pytest.approx(2.0, abs=1e-12)
        if n % 2 == 0:
            assert tab["B1"] == pytest.approx(4.0, abs=1e-12)
            assert tab["B2"] == pytest.approx(6.0, abs=1e-12)
        kmax = (n - 1) // 2 if n % 2 else n // 2 - 1
        for k in range(1, kmax + 1):
            assert tab[f"E{k}"] == pytest.approx(
                3 - 2 * np.cos(2 * np.pi * k / n), abs=1e-12)


def test_lightest_sector_structure():
    """n=5: A2 (pseudoscalar precursor) is the lightest nontrivial sector;
    n=8,9,12: E1. Recorded facts, no significance claimed."""
    assert lightest_charge_sector(5)[0] == "A2_sign"
    for n in (8, 9, 12):
        assert lightest_charge_sector(n)[0] == "E1"


def test_strong_coupling_LO_is_jpc_degenerate():
    """The M4 blocker, locked: leading order cannot split J^PC."""
    for n in (5, 8, 9, 12):
        lo = strong_coupling_leading_order(n)
        assert lo["jpc_split_at_this_order"] is False
        assert lo["glueball_LO_mass_units_invlam"] == pytest.approx(
            4 * lo["eps_min"])


# --- M4: honesty boundary ----------------------------------------------------

def test_benchmark_v3_stays_sealed():
    v3 = benchmark_v3_status()
    assert v3["status"] == "SEALED"
    assert v3["numbers_invented_here"] is False
    assert "J^PC-degenerate" in v3["reason"]


def test_module_blind_to_glueball_targets():
    """MECHANICAL BLINDNESS GUARD: no sealed target or experimental number in
    the module source."""
    src = inspect.getsource(gd)
    for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370", "2359",
                   "2395", "X(23"):
        assert leaked not in src, f"M2-M4 module leaked: {leaked}"


def test_report_honest():
    txt = report()
    assert "PASS" in txt
    assert "SEALED" in txt
    assert "No spectrum was invented" in txt
