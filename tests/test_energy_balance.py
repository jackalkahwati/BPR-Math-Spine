"""
Tests for bpr.energy_balance — thermodynamic accounting module.

These tests verify:
1. Free energy functional is self-consistent (cycle closes)
2. Null configurations produce zero or negligible excess
3. Scaling predictions follow predicted power laws
4. Enhancement breakdown is transparent (no double-counting)
5. Energy accounting identity holds
"""

import numpy as np
import pytest

from bpr.energy_balance import (
    BPRCouplingParams,
    BoundaryStateCycle,
    CycleEnergyBalance,
    FreeEnergy,
    NullConfiguration,
    SuperconductorParams,
    generate_null_configurations,
    run_full_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Default Nb-on-diamond configuration."""
    return BoundaryStateCycle.default_diamond_mems()


@pytest.fixture
def default_free_energy():
    """Default free energy functional."""
    return FreeEnergy()


@pytest.fixture
def default_balance(default_config):
    """Pre-computed energy balance for default config."""
    return default_config.compute_energy_balance()


# ---------------------------------------------------------------------------
# 1. Free energy functional consistency
# ---------------------------------------------------------------------------

class TestFreeEnergyConsistency:
    """The free energy must satisfy basic thermodynamic identities."""

    def test_condensation_energy_negative_below_Tc(self, default_free_energy):
        """Condensation lowers free energy below T_c."""
        fe = default_free_energy
        assert fe.F_condensation(4.0) < 0.0
        assert fe.F_condensation(1.0) < fe.F_condensation(8.0)

    def test_condensation_energy_zero_above_Tc(self, default_free_energy):
        """No condensation energy above T_c."""
        fe = default_free_energy
        assert fe.F_condensation(10.0) == 0.0
        assert fe.F_condensation(100.0) == 0.0

    def test_condensation_energy_continuous_at_Tc(self, default_free_energy):
        """Condensation energy approaches zero as T -> T_c."""
        fe = default_free_energy
        T_c = fe.sc.T_c
        F_just_below = fe.F_condensation(T_c - 0.01)
        assert abs(F_just_below) < 1e-10  # Very small near T_c

    def test_magnetic_energy_positive(self, default_free_energy):
        """Magnetic field energy is always non-negative."""
        fe = default_free_energy
        assert fe.F_magnetic(0.0) == 0.0
        assert fe.F_magnetic(0.01) > 0.0
        assert fe.F_magnetic(0.1) > fe.F_magnetic(0.01)

    def test_vortex_energy_scales_with_count(self, default_free_energy):
        """Vortex energy is proportional to vortex count."""
        fe = default_free_energy
        assert fe.F_vortex(0) == 0.0
        E1 = fe.F_vortex(1)
        E10 = fe.F_vortex(10)
        assert abs(E10 / E1 - 10.0) < 0.01  # Linear in n

    def test_bpr_term_zero_at_W_zero(self, default_free_energy):
        """BPR anomaly term vanishes at W=0."""
        fe = default_free_energy
        assert fe.F_bpr(0, 4.0) == 0.0
        assert fe.F_bpr(0, 10.0) == 0.0

    def test_bpr_term_negative_at_nonzero_W(self, default_free_energy):
        """BPR term lowers energy at W != 0 (resonant coupling)."""
        fe = default_free_energy
        assert fe.F_bpr(1, 4.0) < 0.0
        assert fe.F_bpr(10, 4.0) < 0.0

    def test_bpr_term_scales_as_W_squared(self, default_free_energy):
        """BPR term scales as W^2."""
        fe = default_free_energy
        F1 = fe.F_bpr(1, 4.0)
        F2 = fe.F_bpr(2, 4.0)
        F10 = fe.F_bpr(10, 4.0)
        # F ~ W^2, so F(2)/F(1) ~ 4, F(10)/F(1) ~ 100
        assert abs(F2 / F1 - 4.0) < 0.01
        assert abs(F10 / F1 - 100.0) < 0.1

    def test_anomaly_matches_bpr_minus_standard(self, default_free_energy):
        """anomaly_per_state equals F_total - F_total_no_bpr."""
        fe = default_free_energy
        W, T, B = 5, 4.0, 0.01
        anomaly = fe.anomaly_per_state(W, T, B)
        diff = fe.F_total(W, T, B) - fe.F_total_no_bpr(W, T, B)
        assert abs(anomaly - diff) < 1e-30


# ---------------------------------------------------------------------------
# 2. Cycle energy balance
# ---------------------------------------------------------------------------

class TestCycleEnergyBalance:
    """The cycle must satisfy the accounting identity."""

    def test_standard_free_energy_closes(self, default_balance):
        """Standard (non-BPR) free energy change is ~zero over full cycle.

        The cycle starts and ends in the same state (W=0, T=T_high, B=0),
        so the standard free energy change must be zero.
        """
        # Allow small numerical tolerance
        assert abs(default_balance.total_standard_delta_F) < 1e-15

    def test_drive_energy_positive(self, default_balance):
        """Total drive energy input is positive (we pay to run the cycle)."""
        assert default_balance.total_drive_energy > 0

    def test_dissipation_positive(self, default_balance):
        """Total dissipation is positive (irreversible losses exist)."""
        assert default_balance.total_dissipation > 0

    def test_five_steps_in_cycle(self, default_balance):
        """Cycle has exactly 5 steps."""
        assert len(default_balance.steps) == 5

    def test_bpr_anomaly_round_trips(self, default_balance):
        """BPR anomaly term round-trips over the cycle.

        Since F_bpr(W=0) = 0 at both start and end, the total
        BPR contribution should be zero over a full cycle.
        The energy is released during vortex creation and
        absorbed during vortex destruction.
        """
        # The BPR term is negative when W>0 and zero when W=0.
        # Over a full cycle: create -> (F_bpr < 0) -> destroy -> (F_bpr = 0)
        # Total should be zero.
        assert abs(default_balance.total_bpr_delta_F) < 1e-30

    def test_summary_produces_string(self, default_balance):
        """summary() produces a non-empty string."""
        s = default_balance.summary()
        assert isinstance(s, str)
        assert len(s) > 100
        assert "CYCLE ENERGY BALANCE" in s


# ---------------------------------------------------------------------------
# 3. Null configurations
# ---------------------------------------------------------------------------

class TestNullConfigurations:
    """Null configurations must eliminate the anomaly."""

    def test_five_nulls_generated(self, default_config):
        """Exactly 5 null configurations are generated."""
        nulls = generate_null_configurations(default_config)
        assert len(nulls) == 5

    def test_null_1_no_superconductivity(self, default_config):
        """NULL-1: Normal metal shows no excess."""
        nulls = generate_null_configurations(default_config)
        null1 = nulls[0]
        balance = null1.cycle.compute_energy_balance()
        # With T_c ~ 0, there's no condensation and no BPR coupling
        assert abs(balance.total_bpr_delta_F) < 1e-30

    def test_null_2_no_vortices(self, default_config):
        """NULL-2: No vortices means W=0 throughout, no BPR signal."""
        nulls = generate_null_configurations(default_config)
        null2 = nulls[1]
        balance = null2.cycle.compute_energy_balance()
        assert abs(balance.total_bpr_delta_F) < 1e-30

    def test_null_4_bpr_off(self, default_config):
        """NULL-4: BPR coupling off gives zero anomaly."""
        nulls = generate_null_configurations(default_config)
        null4 = nulls[3]
        balance = null4.cycle.compute_energy_balance()
        assert abs(balance.total_bpr_delta_F) < 1e-30

    def test_all_nulls_have_names(self, default_config):
        """Every null has a descriptive name and kill switch."""
        nulls = generate_null_configurations(default_config)
        for null in nulls:
            assert len(null.name) > 5
            assert len(null.kill_switch) > 10


# ---------------------------------------------------------------------------
# 4. Enhancement transparency (no double-counting)
# ---------------------------------------------------------------------------

class TestEnhancementBreakdown:
    """Enhancement factors must be independently auditable."""

    def test_breakdown_has_all_factors(self):
        """Enhancement breakdown includes every factor."""
        bpr = BPRCouplingParams()
        breakdown = bpr.enhancement_breakdown()
        required_keys = [
            "base_coupling", "coherent_N2", "phonon_mode",
            "Q_factor", "derating", "ideal_product", "realistic_product",
        ]
        for key in required_keys:
            assert key in breakdown

    def test_ideal_is_product_of_factors(self):
        """Ideal coupling = base * coherent * phonon * Q (no hidden factors)."""
        bpr = BPRCouplingParams()
        expected = (bpr.lambda_bpr_base
                    * bpr.enhancement_coherent
                    * bpr.enhancement_phonon
                    * bpr.enhancement_Q)
        assert abs(bpr.lambda_ideal - expected) < 1e-60

    def test_realistic_is_ideal_times_derating(self):
        """Realistic coupling = ideal * derating."""
        bpr = BPRCouplingParams()
        assert abs(bpr.lambda_eff - bpr.lambda_ideal * bpr.derating) < 1e-60

    def test_zero_coupling_gives_zero_lambda(self):
        """If base coupling is zero, everything is zero."""
        bpr = BPRCouplingParams(lambda_bpr_base=0.0)
        assert bpr.lambda_eff == 0.0
        assert bpr.lambda_ideal == 0.0


# ---------------------------------------------------------------------------
# 5. Scaling predictions
# ---------------------------------------------------------------------------

class TestScalingPredictions:
    """Scaling laws must be internally consistent."""

    def test_excess_vs_W_has_correct_shape(self, default_config):
        """Excess vs W array has 50 elements."""
        scalings = default_config.compute_scaling_predictions()
        assert len(scalings["W_values"]) == 50

    def test_bpr_scales_quadratically_with_W(self, default_config):
        """BPR term scales as W^2."""
        scalings = default_config.compute_scaling_predictions()
        bpr = scalings["bpr_vs_W"]
        # Check ratio at W=10 vs W=5
        idx_5 = 4   # W=5 (0-indexed)
        idx_10 = 9  # W=10
        ratio = bpr[idx_10] / bpr[idx_5]
        assert abs(ratio - 4.0) < 0.1  # (10/5)^2 = 4

    def test_power_scales_linearly_with_frequency(self, default_config):
        """Power = f * Delta_E (linear in frequency)."""
        scalings = default_config.compute_scaling_predictions()
        freqs = scalings["frequencies_hz"]
        power = scalings["power_vs_freq"]
        # Check that power[i+1] / power[i] ~ freqs[i+1] / freqs[i]
        # for the non-zero part
        nonzero = power > 0
        if np.any(nonzero):
            p = power[nonzero]
            f = freqs[nonzero]
            if len(p) >= 2:
                ratio_p = p[-1] / p[0]
                ratio_f = f[-1] / f[0]
                assert abs(ratio_p / ratio_f - 1.0) < 0.1

    def test_excess_vs_area_is_monotonic(self, default_config):
        """Larger area gives larger (or equal) excess."""
        scalings = default_config.compute_scaling_predictions()
        excess = scalings["excess_vs_area"]
        # Should be monotonically increasing (larger area = more coupling)
        diffs = np.diff(excess)
        assert np.all(diffs >= -1e-30)  # Non-decreasing


# ---------------------------------------------------------------------------
# 6. Superconductor parameters
# ---------------------------------------------------------------------------

class TestSuperconductorParams:
    """Material parameters must be physically reasonable."""

    def test_condensation_energy_positive(self):
        """Condensation energy density is positive."""
        sc = SuperconductorParams()
        assert sc.condensation_energy_density > 0
        assert sc.condensation_energy > 0

    def test_vortex_self_energy_positive(self):
        """Vortex self-energy is positive."""
        sc = SuperconductorParams()
        assert sc.vortex_self_energy > 0

    def test_volume_matches_area_times_thickness(self):
        """Volume = area * thickness."""
        sc = SuperconductorParams()
        assert abs(sc.volume - sc.film_area * sc.film_thickness) < 1e-30


# ---------------------------------------------------------------------------
# 7. Full analysis report
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    """Integration test: full report generates without errors."""

    def test_full_analysis_runs(self):
        """run_full_analysis() completes and returns a string."""
        report = run_full_analysis()
        assert isinstance(report, str)
        assert "ENERGY BALANCE" in report

    def test_full_analysis_with_custom_config(self):
        """Full analysis with high-T_c config."""
        config = BoundaryStateCycle.high_tc_config()
        balance = config.compute_energy_balance()
        assert isinstance(balance.summary(), str)

    def test_predicted_excess_returns_float(self, default_config):
        """predicted_excess_per_cycle returns a number."""
        val = default_config.predicted_excess_per_cycle()
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_predicted_power_returns_float(self, default_config):
        """predicted_anomaly_power returns a non-negative number."""
        val = default_config.predicted_anomaly_power()
        assert isinstance(val, float)
        assert val >= 0.0


# ---------------------------------------------------------------------------
# 8. Accounting identity verification
# ---------------------------------------------------------------------------

class TestAccountingIdentity:
    """The non-negotiable: P_out <= P_in + P_anomaly."""

    def test_energy_conservation_without_bpr(self):
        """With BPR off, the cycle is energy-neutral or lossy."""
        bpr_off = BPRCouplingParams(
            lambda_bpr_base=0.0,
            enhancement_coherent=0.0,
            enhancement_phonon=0.0,
            enhancement_Q=0.0,
            derating=0.0,
        )
        config = BoundaryStateCycle(
            sc=SuperconductorParams(),
            bpr=bpr_off,
            T_high=12.0, T_low=4.0,
            B_vortex=0.01, W_target=10,
            f_cycle=100.0,
        )
        balance = config.compute_energy_balance()
        # With no BPR term, excess must be non-positive
        assert balance.delta_E_excess <= 0.0
        assert balance.power_anomaly == 0.0

    def test_total_standard_closes(self, default_balance):
        """Standard free energy must close over the cycle (first law)."""
        assert abs(default_balance.total_standard_delta_F) < 1e-15
