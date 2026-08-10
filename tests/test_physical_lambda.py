"""Locks for the physical-lambda derivation (PROPOSED v0.1).

These tests lock the derivation's arithmetic, its honesty guards, and its
blindness — NOT the truth of the proposal. If the kill condition fires
(a >5x calibration shift), the module goes to the registry and these locks
are retired with it.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from bpr.nonabelian_gauge_sector import ALLOWED_CLASSES
from bpr.physical_lambda import (BETA_C_MEASURED, P_SUBSTRATE, Z_COORDINATION,
                                 alpha_inverse, beta_gauge,
                                 beta_per_link_observation, beta_thermal,
                                 convergence, phase_verdict)

MODULE = (pathlib.Path(__file__).resolve().parents[1]
          / "bpr" / "physical_lambda.py")


# ---------------------------------------------------------------------------
# Route arithmetic (exact locks)
# ---------------------------------------------------------------------------

def test_alpha_formula_matches_frozen_value():
    """The Postulate 0c formula must give 1/alpha = 137.039 for the frozen
    (p, z) — same equation as bpr/consistency.py."""
    assert alpha_inverse() == pytest.approx(137.0386, abs=5e-4)


def test_route_a_is_ln_p():
    assert beta_thermal() == pytest.approx(np.log(P_SUBSTRATE))
    assert beta_thermal() == pytest.approx(11.5594, abs=1e-3)


def test_route_b_is_inverse_4pi_alpha():
    assert beta_gauge() == pytest.approx(alpha_inverse() / (4 * np.pi))
    assert beta_gauge() == pytest.approx(10.905, abs=1e-2)


def test_routes_agree_within_ten_percent():
    c = convergence()
    assert c["agree_within_10pct"] is True
    assert c["ratio_B_over_A"] == pytest.approx(0.943, abs=5e-3)


def test_convergence_declares_its_coincidence():
    """The ln p ~ 4 pi coincidence must be stated in the output, not hidden."""
    c = convergence()
    assert "ln p" in c["why_they_agree"] and "4 pi" in c["why_they_agree"]


# ---------------------------------------------------------------------------
# The verdict and its honesty guards
# ---------------------------------------------------------------------------

def test_beta_c_covers_all_allowed_classes():
    assert set(BETA_C_MEASURED) == set(ALLOWED_CLASSES)


def test_verdict_is_deconfined_with_wide_margin():
    v = phase_verdict()
    assert v["verdict"] == "DECONFINED / TOPOLOGICAL"
    assert all(r > 4.0 for r in v["ratio_to_transition"].values())


def test_verdict_carries_kill_condition_and_status():
    v = phase_verdict()
    assert "kill_condition" in v and "calibration" in v["kill_condition"]
    assert v["status"].startswith("PROPOSED")


def test_per_link_value_is_recorded_but_disavowed():
    """ln(p)/z lands on beta_c — it must be present (recorded) AND marked as
    having no frozen provenance (not used)."""
    v = phase_verdict()
    obs = v["per_link_observation"]
    assert obs["beta_C"] == pytest.approx(np.log(P_SUBSTRATE) / Z_COORDINATION)
    assert all(0.7 < r < 1.3 for r in obs["vs_beta_c"].values())
    assert "NONE frozen" in obs["provenance"]


def test_consequence_names_both_phases():
    """The stated consequence must acknowledge the confining spectrum belongs
    to a phase the substrate does not occupy (no quiet reinterpretation)."""
    v = phase_verdict()
    assert "anyons" in v["consequence_if_stands"]
    assert "does not occupy" in v["consequence_if_stands"]


# ---------------------------------------------------------------------------
# Blindness
# ---------------------------------------------------------------------------

def test_module_blind_to_glueball_targets():
    src = MODULE.read_text()
    for leaked in ("1730", "2400", "2590", "1.387", "1.497", "2370", "2359",
                   "X(2370)", "Morningstar", "BESIII"):
        assert leaked not in src, f"physical_lambda.py leaks {leaked}"
