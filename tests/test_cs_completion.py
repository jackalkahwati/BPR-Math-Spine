"""
tests/test_cs_completion.py

Tests for the Chern-Simons UV completion claims in bpr/cs_completion.py.

Each test corresponds to a specific rigorous or semi-rigorous claim
from doc/CS_UV_COMPLETION.md.  The one open claim (coefficient of [ln p]²
equals exactly 1) is verified only to the level of "close but open".
"""

import math
import pytest

from bpr.cs_completion import (
    AlphaFormulaOrigins,
    BoundaryTheoryParams,
    HopfFibrationSummary,
    alpha_formula_cs_origin,
    compute_s2_propagator,
    hopf_fibration_summary,
    verify_anyon_field_condition,
    verify_prime_constraint,
)
from bpr.constants import P_DEFAULT, Z_DEFAULT

_GAMMA = 0.5772156649015328606
_ALPHA_INV_EXP = 137.035999084


# ---------------------------------------------------------------------------
# 1. Prime constraint: anyon field condition
# ---------------------------------------------------------------------------

class TestPrimeConstraint:
    """Z_k is a field iff k is prime — the core derived result."""

    def test_prime_level_is_field(self):
        for p in (2, 3, 5, 7, 11, 13, 97, 104761):
            r = verify_prime_constraint(p)
            assert r["zk_is_field"] is True, f"Z_{p} should be a field"
            assert r["k_is_prime"] is True, f"{p} should be prime"
            assert r["consistent"] is True

    def test_composite_level_is_not_field(self):
        for k in (4, 6, 9, 10, 15, 100):
            r = verify_prime_constraint(k)
            assert r["zk_is_field"] is False, f"Z_{k} should NOT be a field"
            assert r["k_is_prime"] is False, f"{k} should NOT be prime"
            assert r["consistent"] is True
            assert r["zero_divisor"] is not None

    def test_zero_divisor_example_z6(self):
        """Z_6 has zero divisors: 2 × 3 ≡ 0 (mod 6)."""
        r = verify_prime_constraint(6)
        m, n = r["zero_divisor"]
        assert (m * n) % 6 == 0
        assert 0 < m < 6 and 0 < n < 6

    def test_default_p_is_prime(self):
        r = verify_anyon_field_condition(P_DEFAULT)
        assert r["zk_is_field"] is True
        assert r["k_is_prime"] is True

    def test_consistency_for_small_k(self):
        """field(Z_k) == is_prime(k) for all k in 2..30."""
        for k in range(2, 31):
            r = verify_prime_constraint(k)
            assert r["consistent"] is True, f"inconsistency at k = {k}"


# ---------------------------------------------------------------------------
# 2. Hopf fibration: S³ → S²  mode count
# ---------------------------------------------------------------------------

class TestHopfFibration:
    def test_l_max_floor_sqrt(self):
        for p in (7, 97, 1009, 104761):
            hf = hopf_fibration_summary(p)
            assert hf.L_max == math.isqrt(p)

    def test_s2_modes_formula(self):
        for p in (7, 97, 1009, 104761):
            hf = hopf_fibration_summary(p)
            assert hf.s2_modes == (hf.L_max + 1) ** 2

    def test_ratio_close_to_one_for_large_p(self):
        """For p = 104761: |(L+1)²/p − 1| < 1%."""
        hf = hopf_fibration_summary(104761)
        assert hf.discrepancy_pct < 1.0, (
            f"Mode-count ratio discrepancy {hf.discrepancy_pct:.3f}% ≥ 1%"
        )

    def test_default_p_hopf(self):
        hf = hopf_fibration_summary(P_DEFAULT)
        assert hf.p == P_DEFAULT
        assert hf.L_max == 323          # ⌊√104761⌋ = 323
        assert hf.s2_modes == 104976    # 324² = 104976

    def test_ratio_numerical_value(self):
        hf = hopf_fibration_summary(P_DEFAULT)
        assert abs(hf.ratio - 104976 / 104761) < 1e-10


# ---------------------------------------------------------------------------
# 3. c=1 compact boson boundary theory
# ---------------------------------------------------------------------------

class TestBoundaryTheory:
    def test_kappa_equals_z_over_2(self):
        bt = BoundaryTheoryParams(z=6, p=P_DEFAULT)
        assert bt.kappa == pytest.approx(3.0)

    def test_compactification_radius(self):
        bt = BoundaryTheoryParams(z=6, p=P_DEFAULT)
        assert bt.compactification_radius == pytest.approx(math.sqrt(3.0))

    def test_uv_cutoff_from_p(self):
        bt = BoundaryTheoryParams(z=6, p=P_DEFAULT)
        assert bt.uv_cutoff_L == math.isqrt(P_DEFAULT)

    def test_is_c1_boson(self):
        bt = BoundaryTheoryParams()
        assert bt.is_c1_boson() is True

    def test_default_z_and_p(self):
        bt = BoundaryTheoryParams()
        assert bt.z == Z_DEFAULT
        assert bt.p == P_DEFAULT


# ---------------------------------------------------------------------------
# 4. S² propagator
# ---------------------------------------------------------------------------

class TestS2Propagator:
    def test_g_s2_exact_formula(self):
        """G = 2 H_L + 1/(L+1) − 1 at L = ⌊√p⌋."""
        p = P_DEFAULT
        L = math.isqrt(p)
        H_L = sum(1.0 / k for k in range(1, L + 1))
        expected = 2 * H_L + 1.0 / (L + 1) - 1.0

        result = compute_s2_propagator(p)
        assert result["G_s2_exact"] == pytest.approx(expected, rel=1e-12)

    def test_g_s2_asymptotic_approx(self):
        """G ≈ ln(p) + 2γ − 1 to within 1% for p = P_DEFAULT."""
        p = P_DEFAULT
        result = compute_s2_propagator(p)
        G_exact = result["G_s2_exact"]
        G_asymp = result["G_s2_asymptotic"]
        assert abs(G_exact - G_asymp) / G_exact < 0.01

    def test_g_sq_is_close_to_bpr_alpha_inv(self):
        """G_S2² within 1% of 1/α_BPR — the 0.16% gap is the open problem."""
        result = compute_s2_propagator(P_DEFAULT)
        assert result["discrepancy_pct"] < 1.0

    def test_g_sq_numerical_value(self):
        result = compute_s2_propagator(P_DEFAULT)
        assert result["G_s2_squared"] == pytest.approx(137.263, abs=0.01)

    def test_bpr_alpha_inv_close_to_experiment(self):
        result = compute_s2_propagator(P_DEFAULT)
        bpr = result["alpha_inv_bpr"]
        exp = result["alpha_inv_exp"]
        assert abs(bpr - exp) / exp < 1e-4  # BPR is 19.3 ppm off experiment

    def test_l_max_matches_hopf(self):
        """L_max in propagator agrees with Hopf fibration L_max."""
        prop = compute_s2_propagator(P_DEFAULT)
        hf = hopf_fibration_summary(P_DEFAULT)
        assert prop["L_max"] == hf.L_max

    def test_open_gap_is_positive(self):
        """G_S2² > 1/α_BPR (propagator slightly overestimates)."""
        result = compute_s2_propagator(P_DEFAULT)
        assert result["G_sq_minus_alpha_inv_bpr"] > 0


# ---------------------------------------------------------------------------
# 5. Alpha formula origins
# ---------------------------------------------------------------------------

class TestAlphaFormulaOrigins:
    def test_total_matches_bpr_formula(self):
        origins = alpha_formula_cs_origin(P_DEFAULT, Z_DEFAULT)
        ln_p = math.log(P_DEFAULT)
        z = Z_DEFAULT
        expected = ln_p**2 + z / 2.0 + _GAMMA - 1.0 / (2.0 * math.pi)
        assert origins.total == pytest.approx(expected, rel=1e-12)

    def test_z_over_2_is_3(self):
        origins = alpha_formula_cs_origin(P_DEFAULT, Z_DEFAULT)
        val, status, _ = origins.z_over_2
        assert val == pytest.approx(3.0)
        assert "derived" in status.lower()

    def test_euler_gamma_value(self):
        origins = alpha_formula_cs_origin()
        val, status, _ = origins.euler_gamma
        assert val == pytest.approx(_GAMMA, rel=1e-12)
        assert "scheme" in status.lower()

    def test_minus_1_over_2pi(self):
        origins = alpha_formula_cs_origin()
        val, status, _ = origins.minus_1_over_2pi
        assert val == pytest.approx(-1.0 / (2.0 * math.pi), rel=1e-12)
        assert "scheme" in status.lower()

    def test_ln_p_sq_status_is_open(self):
        origins = alpha_formula_cs_origin()
        _, status, _ = origins.ln_p_sq
        assert "open" in status.lower()

    def test_report_contains_key_strings(self):
        origins = alpha_formula_cs_origin()
        rpt = origins.report()
        assert "[ln p]²" in rpt
        assert "γ" in rpt
        assert "Total 1/α" in rpt
        assert "CODATA" in rpt

    def test_total_is_137_039(self):
        """BPR formula gives 1/α ≈ 137.039."""
        origins = alpha_formula_cs_origin()
        assert abs(origins.total - 137.039) < 0.001
