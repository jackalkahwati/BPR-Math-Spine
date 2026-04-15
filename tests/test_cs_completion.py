"""
tests/test_cs_completion.py

Tests for the Chern-Simons UV completion claims in bpr/cs_completion.py.

All claims are now either rigorous or scheme-identified.  The coefficient of
[ln p]² = 1 is proven via the topological entanglement entropy of U(1)_p CS:
D = √p → S_topo = (1/2)ln p → 2·S_topo = ln p exactly → [2·S_topo]² = [ln p]²
with coefficient exactly 1 (no γ corrections, since TEE is a topological invariant).
"""

import math
import pytest

from bpr.cs_completion import (
    AlphaFormulaOrigins,
    CSChiralPropagator,
    HolographicDerivation,
    TEEResolution,
    cs_chiral_derivation,
    holographic_derivation,
    verify_tee_coefficient,
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
        # γ is now derived from the CS anyon sum H_{p-1} - ln p → γ
        assert "derived" in status.lower()

    def test_minus_1_over_2pi(self):
        origins = alpha_formula_cs_origin()
        val, status, _ = origins.minus_1_over_2pi
        assert val == pytest.approx(-1.0 / (2.0 * math.pi), rel=1e-12)
        assert "scheme" in status.lower()

    def test_ln_p_sq_status_is_derived_via_tee(self):
        origins = alpha_formula_cs_origin()
        _, status, _ = origins.ln_p_sq
        assert "derived" in status.lower() or "tee" in status.lower()

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


# ---------------------------------------------------------------------------
# 6. TEE resolution: coefficient of [ln p]² = 1
# ---------------------------------------------------------------------------

class TestTEEResolution:
    """
    Coefficient = 1 proven via topological entanglement entropy.

    For U(1)_p CS: D = √p (exact) → S_topo = (1/2)ln p (exact) →
    2·S_topo = ln p (exact) → Π_EM = [ln p]² with coefficient exactly 1.
    """

    def test_total_quantum_dimension_is_sqrt_p(self):
        """D = √p for U(1)_p CS (all anyon dimensions 1)."""
        for p in (7, 97, 1009, P_DEFAULT):
            tee = TEEResolution(p=p)
            assert tee.total_quantum_dimension == pytest.approx(math.sqrt(p), rel=1e-12)

    def test_tee_is_half_ln_p(self):
        """S_topo = (1/2) ln p exactly."""
        for p in (7, 97, 1009, P_DEFAULT):
            tee = TEEResolution(p=p)
            assert tee.tee == pytest.approx(0.5 * math.log(p), rel=1e-12)

    def test_two_tee_equals_ln_p_exactly(self):
        """2·S_topo = ln p to floating-point precision."""
        for p in (7, 97, 1009, P_DEFAULT):
            tee = TEEResolution(p=p)
            assert tee.s3_partition_entropy_equals_ln_p is True
            assert abs(tee.s3_partition_entropy - math.log(p)) < 1e-12

    def test_pi_em_equals_ln_p_squared(self):
        """Π_EM = (2·S_topo)² = [ln p]² exactly."""
        for p in (7, 97, 1009, P_DEFAULT):
            tee = TEEResolution(p=p)
            assert tee.pi_em == pytest.approx(math.log(p) ** 2, rel=1e-12)

    def test_coefficient_is_exactly_1(self):
        """Coefficient of [ln p]² in Π_EM is exactly 1."""
        for p in (7, 97, 1009, P_DEFAULT):
            tee = TEEResolution(p=p)
            assert abs(tee.pi_em_coefficient - 1.0) < 1e-12

    def test_g_s2_is_wrong_object(self):
        """G_S2² − [ln p]² grows O(ln p), not zero."""
        comp = TEEResolution(p=P_DEFAULT).g_s2_comparison
        # Gap is nonzero and positive (G_S2 > ln p due to 2γ−1 correction)
        assert comp["G_s2_minus_ln_p"] > 0.14   # ≈ 2γ−1 ≈ 0.154
        assert comp["G_s2_sq_gap_from_ln_p_sq"] > 3.0  # ≈ 3.64 for p=104761
        # TEE gap is exactly zero
        assert comp["tee_sq_gap_from_ln_p_sq"] == pytest.approx(0.0, abs=1e-12)

    def test_verify_tee_coefficient_function(self):
        result = verify_tee_coefficient(P_DEFAULT)
        assert result["coefficient_is_exactly_1"] is True
        assert result["two_S_topo_equals_ln_p"] is True
        assert "PROVEN" in result["conclusion"]

    def test_gap_grows_with_p_for_g_s2_not_tee(self):
        """The O(ln p) growth of G_S2² gap confirms G_S2 is wrong object."""
        gaps = []
        for p in (1009, 9973, 104761):
            tee = TEEResolution(p=p)
            comp = tee.g_s2_comparison
            gaps.append(comp["G_s2_sq_gap_from_ln_p_sq"])
        # Gaps should grow (each larger than the previous)
        assert gaps[0] < gaps[1] < gaps[2]

    def test_tee_gap_is_zero_for_all_p(self):
        """TEE gap is zero for all primes — coefficient always exactly 1."""
        for p in (7, 97, 1009, 9973, 104761):
            tee = TEEResolution(p=p)
            assert tee.pi_em == pytest.approx(math.log(p) ** 2, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Holographic derivation: γ is derived from CS anyon sum
# ---------------------------------------------------------------------------

class TestHolographicDerivation:
    """
    The formal holographic derivation connecting Π_EM to (2·S_topo)².

    Key result: the +γ in BPR is NOT an assumption — it is the IR correction
    to the CS anyon amplitude sum H_{p-1}, since lim(H_n − ln n) = γ exactly.
    """

    def test_anyon_sum_is_harmonic_number(self):
        """A_CS = Σ_{a=1}^{p-1} 1/a = H_{p-1}."""
        for p in (7, 97, P_DEFAULT):
            holo = holographic_derivation(p)
            expected = sum(1.0 / a for a in range(1, p))
            assert holo.A_cs == pytest.approx(expected, rel=1e-12)

    def test_uv_amplitude_equals_ln_p(self):
        """UV part of A_CS = ln p exactly (= 2·S_topo)."""
        for p in (7, 97, P_DEFAULT):
            holo = holographic_derivation(p)
            assert holo.uv_amplitude == pytest.approx(math.log(p), rel=1e-12)

    def test_ir_correction_converges_to_gamma(self):
        """H_{p-1} − ln p → γ as p→∞ (by definition of γ).

        H_{n} − ln n − γ ≈ 1/(2n) asymptotically.  Here n = p−1, so the
        correct bound is 1/(2(p−1)) — slightly larger than 1/(2p) for small p.
        """
        for p in (101, 1009, 9973, P_DEFAULT):
            holo = holographic_derivation(p)
            # |IR − γ| = O(1/p): within 1/(2(p−1)) of γ (exact asymptotic)
            assert holo.ir_correction_error_from_gamma < 1.0 / (2 * (p - 1)) + 1e-9

    def test_gamma_derived_flag(self):
        """γ in BPR is derived from CS anyon sum (within O(1/p))."""
        holo = holographic_derivation(P_DEFAULT)
        assert holo.gamma_is_derived is True

    def test_pi_em_is_ln_p_squared(self):
        """Π_EM = (UV amplitude)² = [ln p]² with coefficient exactly 1."""
        for p in (7, 97, P_DEFAULT):
            holo = holographic_derivation(p)
            assert holo.pi_em == pytest.approx(math.log(p) ** 2, rel=1e-12)

    def test_bpr_formula_from_cs_vs_exact(self):
        """Reconstructed 1/α from CS H_{p-1} vs exact γ differ by O(1/p)."""
        holo = holographic_derivation(P_DEFAULT)
        diff = abs(holo.bpr_formula_from_cs - holo.bpr_formula_exact)
        assert diff < 1e-3  # O(1/p) ≈ 10⁻⁵ for p=104761

    def test_bpr_formula_from_cs_close_to_experiment(self):
        """Full BPR formula reconstructed from CS is within 19 ppm of experiment."""
        holo = holographic_derivation(P_DEFAULT)
        frac_err = abs(holo.bpr_formula_exact - _ALPHA_INV_EXP) / _ALPHA_INV_EXP
        assert frac_err < 1e-4

    def test_uv_ir_decomposition_adds_up(self):
        """UV + IR = H_{p-1} exactly."""
        holo = holographic_derivation(P_DEFAULT)
        assert holo.uv_amplitude + holo.ir_correction == pytest.approx(
            holo.A_cs, rel=1e-14
        )

    def test_ir_shrinks_with_p(self):
        """H_{p-1} − ln p − γ → 0 as p increases (O(1/p) convergence)."""
        errors = [holographic_derivation(p).ir_correction_error_from_gamma
                  for p in (101, 1009, 9973, P_DEFAULT)]
        assert errors[0] > errors[1] > errors[2] > errors[3]

    def test_derivation_summary_contains_key_results(self):
        holo = holographic_derivation(P_DEFAULT)
        summary = holo.derivation_summary()
        assert "H_{p-1}" in summary
        assert "UV part" in summary
        assert "IR corr" in summary
        assert "Π_EM" in summary
        assert "DERIVED" in summary.upper() or "coeff" in summary.lower()


# ---------------------------------------------------------------------------
# 8. CSChiralPropagator — Ward identity from the CS Lagrangian (closes last gap)
# ---------------------------------------------------------------------------

class TestCSChiralPropagator:
    """
    Tests for the CS chiral Ward identity: A_CS = Σ 1/a derived from the
    first-order CS Lagrangian (check #8 in cs_completion.py).
    """

    def test_chiral_propagator_is_one_over_a(self):
        """G_a^CS = 1/a for each mode a (first-order CS action)."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        for a in (1, 2, 3, 7, 100, 1000):
            if a < P_DEFAULT:
                assert chiral.chiral_propagator_per_mode[a - 1] == pytest.approx(1.0 / a)

    def test_second_order_propagator_is_one_over_a_sq(self):
        """G_a^Maxwell = 1/a² for each mode (contrast with CS)."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        for a in (1, 2, 3, 7, 100):
            if a < P_DEFAULT:
                assert chiral.second_order_propagator_per_mode[a - 1] == pytest.approx(1.0 / a**2)

    def test_first_order_sum_equals_harmonic_number(self):
        """Σ_{a=1}^{p-1} 1/a = H_{p-1} (first-order CS)."""
        for p in (7, 97, P_DEFAULT):
            chiral = cs_chiral_derivation(p)
            expected = sum(1.0 / a for a in range(1, p))
            assert chiral.A_cs_first_order == pytest.approx(expected, rel=1e-12)

    def test_second_order_sum_converges_to_pi_sq_over_6(self):
        """Σ 1/a² → π²/6 as p → ∞ (second-order Maxwell would give no ln p)."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        assert chiral.A_cs_second_order == pytest.approx(math.pi**2 / 6, rel=1e-4)

    def test_first_order_has_ln_p_uv(self):
        """UV part of Σ 1/a ≈ ln p (the first-order result contains ln p)."""
        for p in (97, 9973, P_DEFAULT):
            chiral = cs_chiral_derivation(p)
            assert chiral.first_order_gives_ln_p_in_uv

    def test_second_order_has_no_ln_p(self):
        """Σ 1/a² → π²/6 for ALL large p — no dependence on ln p."""
        for p in (97, 9973, P_DEFAULT):
            chiral = cs_chiral_derivation(p)
            assert chiral.second_order_gives_constant

    def test_first_vs_second_differ_significantly(self):
        """First-order sum >> second-order sum for large p (ln p >> 1)."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        # H_{p-1} ≈ 11.6 vs π²/6 ≈ 1.6 — factor ~7 difference
        assert chiral.A_cs_first_order > chiral.A_cs_second_order * 5.0

    def test_first_order_matches_holographic_derivation(self):
        """CSChiralPropagator and HolographicDerivation give identical A_CS."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        holo = holographic_derivation(P_DEFAULT)
        assert chiral.A_cs_first_order == pytest.approx(holo.A_cs, rel=1e-12)

    def test_ward_identity_closed(self):
        """The CS chiral Ward identity is closed for all tested primes."""
        for p in (7, 97, 9973, P_DEFAULT):
            chiral = cs_chiral_derivation(p)
            assert chiral.ward_identity_closed is True

    def test_comparison_report_contains_key_strings(self):
        """Report contains 'first-order', 'second-order', and 'Ward identity'."""
        chiral = cs_chiral_derivation(P_DEFAULT)
        report = chiral.comparison_report()
        assert "FIRST-ORDER" in report.upper() or "first-order" in report.lower()
        assert "Ward identity" in report
        assert "H_{p-1}" in report or "H_" in report
