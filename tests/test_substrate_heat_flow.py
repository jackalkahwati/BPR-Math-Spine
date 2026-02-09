"""
Tests for bpr.substrate_heat_flow
==================================

Covers:
  §1  T_eff computation (all three models, edge cases)
  §2  Transmission functions (flat, ohmic, resonant; shapes and bounds)
  §3  Heat current backends (landauer, ohmic, sb_proxy; thermodynamic sanity)
  §4  Null configurations (kill-switch properties)
  §5  Scaling predictions (monotonicity and consistency)
  §6  Full analysis (smoke test, report generation)
"""

import math
import numpy as np
import pytest

from bpr.substrate_heat_flow import (
    SubstrateBath,
    BoundaryCoupling,
    HeatFlowResult,
    NullConfiguration,
    TeffResult,
    compute_Teff,
    compute_heat_flow,
    compute_all_methods,
    generate_nulls_for_heat_flow,
    generate_scaling_predictions,
    heat_current_landauer,
    heat_current_ohmic_kubo,
    heat_current_sb_proxy,
    run_full_analysis,
    transmission_flat,
    transmission_ohmic,
    transmission_resonant,
    transmission_T,
    _bose_einstein,
    _default_omega_c,
    _K_B,
    _HBAR,
    _EV,
)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  T_eff computation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeTeff:
    """Tests for compute_Teff under all three models."""

    def test_entropy_mapping_basic(self):
        r = compute_Teff(J_eV=1.0, p=1e5, model="entropy_mapping")
        expected = (1.0 * _EV) / (_K_B * np.log(1e5))
        assert abs(r.T_eff - expected) / expected < 1e-10
        assert r.is_conjecture is True
        assert r.model == "entropy_mapping"

    def test_entropy_mapping_is_conjecture(self):
        r = compute_Teff(model="entropy_mapping")
        assert r.is_conjecture is True
        assert len(r.warnings) > 0
        assert "CONJECTURE" in r.warnings[0]

    def test_entropy_mapping_scales_with_J(self):
        r1 = compute_Teff(J_eV=0.5, p=1e5)
        r2 = compute_Teff(J_eV=2.0, p=1e5)
        assert r2.T_eff == pytest.approx(4.0 * r1.T_eff, rel=1e-10)

    def test_entropy_mapping_scales_inversely_with_ln_p(self):
        r1 = compute_Teff(J_eV=1.0, p=1e5)
        r2 = compute_Teff(J_eV=1.0, p=1e10)
        ratio = r1.T_eff / r2.T_eff
        expected_ratio = np.log(1e10) / np.log(1e5)
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_free_parameter_model(self):
        r = compute_Teff(model="free_parameter", Teff_override=500.0)
        assert r.T_eff == 500.0
        assert r.is_conjecture is False
        assert r.model == "free_parameter"

    def test_free_parameter_requires_override(self):
        with pytest.raises(ValueError, match="Teff_override"):
            compute_Teff(model="free_parameter")

    def test_free_parameter_rejects_zero(self):
        with pytest.raises(ValueError, match="> 0"):
            compute_Teff(model="free_parameter", Teff_override=0.0)

    def test_cosmological_model(self):
        r = compute_Teff(J_eV=1.0, model="cosmological")
        expected = (1.0 * _EV) / (_K_B * np.log(1e60))
        assert abs(r.T_eff - expected) / expected < 1e-10
        assert r.is_conjecture is True

    def test_cosmological_much_lower_than_lab(self):
        r_lab = compute_Teff(J_eV=1.0, p=1e5)
        r_cosmo = compute_Teff(J_eV=1.0, model="cosmological")
        assert r_cosmo.T_eff < r_lab.T_eff

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            compute_Teff(model="magic")

    def test_p_le_1_raises(self):
        with pytest.raises(ValueError, match="p must be > 1"):
            compute_Teff(J_eV=1.0, p=1, model="entropy_mapping")

    def test_str_representation(self):
        r = compute_Teff(J_eV=1.0, p=1e5)
        s = str(r)
        assert "K" in s
        assert "CONJECTURE" in s


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Transmission functions
# ═══════════════════════════════════════════════════════════════════════════

class TestTransmission:
    """Tests for transmission_flat, transmission_ohmic, transmission_resonant."""

    @pytest.fixture
    def coupling(self):
        return BoundaryCoupling(lambda_eff=1e-6, A=1e-4,
                                spectrum_model="ohmic")

    def test_flat_below_cutoff(self, coupling):
        omega_c = 1e14
        T = transmission_flat(1e12, coupling, omega_c)
        assert T == pytest.approx(coupling.lambda_eff)

    def test_flat_above_cutoff(self, coupling):
        omega_c = 1e14
        T = transmission_flat(2e14, coupling, omega_c)
        assert T == 0.0

    def test_flat_respects_cap(self):
        c = BoundaryCoupling(lambda_eff=2.0, transmission_cap=0.5)
        T = transmission_flat(1e12, c, 1e14)
        assert T == pytest.approx(0.5)

    def test_ohmic_peaks_at_omega_c(self, coupling):
        omega_c = 1e14
        omegas = np.logspace(10, 16, 500)
        Ts = transmission_ohmic(omegas, coupling, omega_c)
        peak_idx = np.argmax(Ts)
        # Ohmic peaks at ω = ω_c
        assert abs(omegas[peak_idx] - omega_c) / omega_c < 0.1

    def test_ohmic_zero_at_zero(self, coupling):
        T = transmission_ohmic(0.0, coupling, 1e14)
        assert T == pytest.approx(0.0)

    def test_ohmic_nonnegative(self, coupling):
        omegas = np.linspace(0, 1e16, 1000)
        Ts = transmission_ohmic(omegas, coupling, 1e14)
        assert np.all(Ts >= 0)

    def test_resonant_peaks_at_omega_res(self):
        c = BoundaryCoupling(lambda_eff=1e-6, omega_res=1e12, Q_res=1e6,
                             spectrum_model="resonant")
        omegas = np.linspace(0.99e12, 1.01e12, 1000)
        Ts = transmission_resonant(omegas, c, 1e14)
        peak_idx = np.argmax(Ts)
        assert abs(omegas[peak_idx] - 1e12) / 1e12 < 0.001

    def test_resonant_requires_params(self):
        c = BoundaryCoupling(lambda_eff=1e-6)
        with pytest.raises(ValueError, match="omega_res"):
            transmission_resonant(1e12, c, 1e14)

    def test_dispatcher_flat(self, coupling):
        c = BoundaryCoupling(lambda_eff=1e-6, spectrum_model="flat")
        T = transmission_T(1e12, c, 1e14)
        assert T == pytest.approx(c.lambda_eff)

    def test_dispatcher_unknown_raises(self, coupling):
        c = BoundaryCoupling(spectrum_model="unknown")
        with pytest.raises(ValueError, match="Unknown"):
            transmission_T(1e12, c, 1e14)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Heat current backends
# ═══════════════════════════════════════════════════════════════════════════

class TestHeatCurrents:
    """Thermodynamic sanity checks for all three backends."""

    @pytest.fixture
    def coupling(self):
        return BoundaryCoupling(lambda_eff=6e-7, A=1e-4,
                                spectrum_model="ohmic")

    # ─── Bose–Einstein helper ─────────────────────────────────────────────

    def test_bose_einstein_limit(self):
        """n → 0 for ℏω >> kT."""
        n = _bose_einstein(1e16, 4.0)
        assert n == pytest.approx(0.0, abs=1e-30)

    def test_bose_einstein_classical(self):
        """n → kT/(ℏω) − 1/2 for ℏω << kT."""
        omega = 1e8
        T = 1000.0
        n = _bose_einstein(omega, T)
        classical = _K_B * T / (_HBAR * omega)
        assert n == pytest.approx(classical, rel=0.01)

    # ─── General properties ───────────────────────────────────────────────

    def test_no_flow_when_equal_temps(self, coupling):
        """Q̇ = 0 when T_eff = T_cold (detailed balance)."""
        assert heat_current_landauer(100, 100, coupling) == 0.0
        assert heat_current_ohmic_kubo(100, 100, coupling) == 0.0
        assert heat_current_sb_proxy(100, 100, coupling) == 0.0

    def test_no_flow_when_reversed(self, coupling):
        """Q̇ = 0 when T_eff < T_cold (function clamps)."""
        assert heat_current_landauer(4, 100, coupling) == 0.0
        assert heat_current_ohmic_kubo(4, 100, coupling) == 0.0
        assert heat_current_sb_proxy(4, 100, coupling) == 0.0

    def test_positive_flow_when_hot(self, coupling):
        """Q̇ > 0 when T_eff > T_cold."""
        assert heat_current_landauer(1000, 4, coupling) > 0
        assert heat_current_ohmic_kubo(1000, 4, coupling) > 0
        assert heat_current_sb_proxy(1000, 4, coupling) > 0

    def test_flow_increases_with_delta_T(self, coupling):
        """Q̇ increases with ΔT (monotonicity)."""
        Q1 = heat_current_ohmic_kubo(500, 4, coupling)
        Q2 = heat_current_ohmic_kubo(1000, 4, coupling)
        assert Q2 > Q1

    def test_flow_increases_with_lambda_eff(self):
        """Q̇ increases with λ_eff (stronger coupling → more heat)."""
        c1 = BoundaryCoupling(lambda_eff=1e-8, A=1e-4, spectrum_model="ohmic")
        c2 = BoundaryCoupling(lambda_eff=1e-6, A=1e-4, spectrum_model="ohmic")
        Q1 = heat_current_ohmic_kubo(1000, 4, c1)
        Q2 = heat_current_ohmic_kubo(1000, 4, c2)
        assert Q2 > Q1

    # ─── Ohmic-Kubo specific ──────────────────────────────────────────────

    def test_ohmic_scales_as_deltaT_times_Teff(self, coupling):
        """Ohmic model: Q̇ ∝ ΔT × T_eff  (via ω_c = k_B T_eff / ℏ)."""
        T1, T2, Tc = 504.0, 1004.0, 4.0
        Q1 = heat_current_ohmic_kubo(T1, Tc, coupling)
        Q2 = heat_current_ohmic_kubo(T2, Tc, coupling)
        # Q̇ = α k_B (T_eff - T_cold) λ_eff (k_B T_eff / ℏ)
        # Ratio = (T2-Tc)*T2 / ((T1-Tc)*T1)
        expected_ratio = ((T2 - Tc) * T2) / ((T1 - Tc) * T1)
        assert Q2 / Q1 == pytest.approx(expected_ratio, rel=1e-6)

    def test_ohmic_linear_in_lambda_eff(self):
        """Ohmic model is linear in λ_eff."""
        c1 = BoundaryCoupling(lambda_eff=1e-7, A=1e-4, spectrum_model="ohmic")
        c2 = BoundaryCoupling(lambda_eff=2e-7, A=1e-4, spectrum_model="ohmic")
        Q1 = heat_current_ohmic_kubo(1000, 4, c1)
        Q2 = heat_current_ohmic_kubo(1000, 4, c2)
        assert Q2 / Q1 == pytest.approx(2.0, rel=1e-10)

    def test_ohmic_alpha_factor(self, coupling):
        """α prefactor scales Q̇ linearly."""
        Q1 = heat_current_ohmic_kubo(1000, 4, coupling, alpha=1.0)
        Q2 = heat_current_ohmic_kubo(1000, 4, coupling, alpha=0.5)
        assert Q2 / Q1 == pytest.approx(0.5, rel=1e-10)

    # ─── SB proxy specific ────────────────────────────────────────────────

    def test_sb_scales_with_area(self):
        """SB proxy scales linearly with A."""
        c1 = BoundaryCoupling(lambda_eff=1e-6, A=1e-4)
        c2 = BoundaryCoupling(lambda_eff=1e-6, A=2e-4)
        Q1 = heat_current_sb_proxy(1000, 4, c1)
        Q2 = heat_current_sb_proxy(1000, 4, c2)
        assert Q2 / Q1 == pytest.approx(2.0, rel=1e-10)

    def test_sb_scales_roughly_T4(self):
        """SB proxy ~ T_eff^4 for T_eff >> T_cold."""
        c = BoundaryCoupling(lambda_eff=1e-6, A=1e-4)
        Q1 = heat_current_sb_proxy(500, 4, c)
        Q2 = heat_current_sb_proxy(1000, 4, c)
        # T_cold contribution negligible → ratio should be ~ (1000/500)^4 = 16
        assert Q2 / Q1 == pytest.approx(16.0, rel=0.01)

    # ─── Backend ordering ─────────────────────────────────────────────────

    def test_sb_is_upper_bound(self, coupling):
        """SB proxy should exceed ohmic (it is the upper bound)."""
        Q_sb = heat_current_sb_proxy(1000, 4, coupling)
        Q_ohmic = heat_current_ohmic_kubo(1000, 4, coupling)
        assert Q_sb > Q_ohmic


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Null configurations
# ═══════════════════════════════════════════════════════════════════════════

class TestNulls:
    """Test that null configurations kill or reduce the signal."""

    def test_five_nulls_generated(self):
        nulls = generate_nulls_for_heat_flow()
        assert len(nulls) == 5

    def test_null1_zero_coupling_gives_zero(self):
        nulls = generate_nulls_for_heat_flow()
        null1 = nulls[0]
        assert null1.coupling.lambda_eff == 0.0
        r = compute_heat_flow(null1.bath, null1.coupling, T_cold=4.0,
                              method="ohmic")
        assert r.Qdot_W == 0.0

    def test_null3_normal_metal_much_smaller(self):
        baseline = BoundaryCoupling(lambda_eff=6e-7, A=1e-4)
        nulls = generate_nulls_for_heat_flow(baseline_coupling=baseline)
        null3 = nulls[2]
        r_base = compute_heat_flow(SubstrateBath(), baseline, 4.0, "ohmic")
        r_null = compute_heat_flow(null3.bath, null3.coupling, 4.0, "ohmic")
        assert r_null.Qdot_W < r_base.Qdot_W * 1e-5

    def test_null5_area_scaling(self):
        baseline = BoundaryCoupling(lambda_eff=6e-7, A=1e-4)
        nulls = generate_nulls_for_heat_flow(baseline_coupling=baseline)
        null5 = nulls[4]
        assert null5.coupling.A == pytest.approx(baseline.A / 2.0)

    def test_all_nulls_have_names(self):
        nulls = generate_nulls_for_heat_flow()
        for n in nulls:
            assert n.name.startswith("NULL-")
            assert len(n.description) > 10


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Scaling predictions
# ═══════════════════════════════════════════════════════════════════════════

class TestScalingPredictions:
    """Test scaling curve monotonicity and structure."""

    @pytest.fixture
    def predictions(self):
        bath = SubstrateBath(J_eV=1.0, p=1e5)
        coupling = BoundaryCoupling(lambda_eff=6e-7, A=1e-4,
                                     spectrum_model="ohmic")
        return generate_scaling_predictions(bath, coupling, T_cold=4.0,
                                            method="ohmic")

    def test_area_scaling_monotonic(self, predictions):
        y = predictions["area"]["y"]
        assert np.all(np.diff(y) >= 0)

    def test_Teff_scaling_monotonic(self, predictions):
        y = predictions["Teff"]["y"]
        assert np.all(np.diff(y) >= 0)

    def test_Q_factor_scaling_monotonic(self, predictions):
        y = predictions["Q_factor"]["y"]
        assert np.all(np.diff(y) >= 0)

    def test_predictions_have_correct_keys(self, predictions):
        assert "area" in predictions
        assert "Teff" in predictions
        assert "Q_factor" in predictions

    def test_arrays_nonempty(self, predictions):
        for key in ("area", "Teff", "Q_factor"):
            assert len(predictions[key]["x"]) > 10
            assert len(predictions[key]["y"]) > 10


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Full analysis / unified dispatcher
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeHeatFlow:
    """Test the unified compute_heat_flow dispatcher."""

    def test_returns_heat_flow_result(self):
        bath = SubstrateBath()
        coupling = BoundaryCoupling()
        r = compute_heat_flow(bath, coupling, T_cold=4.0, method="ohmic")
        assert isinstance(r, HeatFlowResult)

    def test_summary_is_string(self):
        r = compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0, "ohmic")
        s = r.summary()
        assert isinstance(s, str)
        assert "Q̇" in s

    def test_all_methods_returns_three(self):
        results = compute_all_methods(SubstrateBath(), BoundaryCoupling(), 4.0)
        assert len(results) == 3
        assert set(results.keys()) == {"landauer", "ohmic", "sb_proxy"}

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0,
                              method="antigravity")

    def test_sb_proxy_warns(self):
        r = compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0,
                              method="sb_proxy")
        assert any("upper bound" in w for w in r.warnings)

    def test_Teff_propagated_to_result(self):
        bath = SubstrateBath(J_eV=2.0, p=1e5)
        r = compute_heat_flow(bath, BoundaryCoupling(), 4.0, "ohmic")
        expected = compute_Teff(J_eV=2.0, p=1e5).T_eff
        assert r.T_eff == pytest.approx(expected)


class TestFullAnalysis:
    """Smoke test for run_full_analysis."""

    def test_returns_nonempty_string(self):
        report = run_full_analysis()
        assert isinstance(report, str)
        assert len(report) > 500

    def test_contains_all_sections(self):
        report = run_full_analysis()
        assert "THREE-BACKEND" in report
        assert "EFFECTIVE TEMPERATURE" in report
        assert "NULL CONFIGURATIONS" in report
        assert "DETECTABILITY" in report
        assert "CAVEATS" in report

    def test_accepts_custom_params(self):
        bath = SubstrateBath(J_eV=0.5, p=1e3)
        coupling = BoundaryCoupling(lambda_eff=1e-10, A=1e-6)
        report = run_full_analysis(bath, coupling, T_cold=0.1)
        assert isinstance(report, str)


# ═══════════════════════════════════════════════════════════════════════════
#  §7  SubstrateBath and BoundaryCoupling dataclass tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDataclasses:
    """Test dataclass defaults and methods."""

    def test_substrate_bath_default_Teff(self):
        b = SubstrateBath()
        r = b.T_eff()
        assert r.T_eff > 0
        assert r.model == "entropy_mapping"

    def test_substrate_bath_free_parameter(self):
        b = SubstrateBath(Teff_model="free_parameter", Teff_override=300.0)
        r = b.T_eff()
        assert r.T_eff == 300.0

    def test_boundary_coupling_defaults(self):
        c = BoundaryCoupling()
        assert c.lambda_eff == 6e-7
        assert c.A == 1e-4
        assert c.spectrum_model == "ohmic"

    def test_default_omega_c(self):
        omega_c = _default_omega_c(1000.0)
        expected = _K_B * 1000.0 / _HBAR
        assert omega_c == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Dimensional and physical consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicalConsistency:
    """Cross-check physical reasonableness of predictions."""

    def test_Teff_lab_scale_roughly_1000K(self):
        """For J=1 eV, p=10^5, T_eff should be ~1000 K."""
        r = compute_Teff(J_eV=1.0, p=1e5)
        assert 500 < r.T_eff < 2000

    def test_Teff_cosmological_positive(self):
        """Even at p=10^60, T_eff should be positive."""
        r = compute_Teff(J_eV=1.0, model="cosmological")
        assert r.T_eff > 0

    def test_ohmic_power_is_finite(self):
        """Ohmic power should be a finite, small number for default params."""
        r = compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0, "ohmic")
        assert np.isfinite(r.Qdot_W)
        assert r.Qdot_W > 0
        # Should be many orders below 1 W
        assert r.Qdot_W < 1e-3

    def test_landauer_power_is_finite(self):
        """Landauer integral should converge to a finite number."""
        r = compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0,
                              "landauer")
        assert np.isfinite(r.Qdot_W)
        assert r.Qdot_W > 0

    def test_sb_proxy_power_is_finite(self):
        r = compute_heat_flow(SubstrateBath(), BoundaryCoupling(), 4.0,
                              "sb_proxy")
        assert np.isfinite(r.Qdot_W)
        assert r.Qdot_W > 0
        # SB proxy with λ_eff ~ 10^-7 and 1 cm² should be < 1 W
        assert r.Qdot_W < 1.0

    def test_zero_coupling_zero_power_all_methods(self):
        """λ_eff = 0 → Q̇ = 0 for all backends."""
        c = BoundaryCoupling(lambda_eff=0.0, A=1e-4)
        for m in ("landauer", "ohmic", "sb_proxy"):
            r = compute_heat_flow(SubstrateBath(), c, 4.0, method=m)
            assert r.Qdot_W == pytest.approx(0.0, abs=1e-30)
