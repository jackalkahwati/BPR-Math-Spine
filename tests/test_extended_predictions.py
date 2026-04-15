"""
Tests for Extended Predictions (Predictions 1–20)
===================================================

Validates all newly-implemented physics functions: neutrino angles,
DM cross-sections, muon g-2, Hubble tension, superconductivity,
proton lifetime, GW ringing, quantum error periodicity, aging
reversal, convergent evolution, black hole entropy, and three
generations.

Run with:  pytest tests/test_extended_predictions.py -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Predictions 1–2: Neutrino mixing angles and mass splittings
# ═══════════════════════════════════════════════════════════════════

class TestNeutrinoAngles:
    def test_mixing_angles_reasonable(self):
        from bpr.neutrino import PMNSMatrix
        pmns = PMNSMatrix()
        angles = pmns.mixing_angles()
        # θ₁₂ ≈ 33.4° experimentally
        assert 30 < angles["theta12_deg"] < 40
        # θ₂₃ ≈ 49° experimentally
        assert 40 < angles["theta23_deg"] < 55
        # θ₁₃ ≈ 8.6° experimentally
        assert 7 < angles["theta13_deg"] < 10

    def test_mass_squared_differences(self):
        from bpr.neutrino import NeutrinoMassSpectrum
        ns = NeutrinoMassSpectrum()
        dm = ns.mass_squared_differences
        # Δm²₂₁ ≈ 7.5 × 10⁻⁵ eV² (experimental)
        assert 1e-6 < dm["Delta_m21_sq"] < 1e-4
        # Δm²₃₂ ≈ 2.5 × 10⁻³ eV² (experimental)
        assert 1e-4 < dm["Delta_m32_sq"] < 0.01

    def test_pmns_is_unitary(self):
        from bpr.neutrino import PMNSMatrix
        pmns = PMNSMatrix()
        assert pmns.is_unitary(tol=1e-10)


# ═══════════════════════════════════════════════════════════════════
# Prediction 3: Dark energy determines p
# ═══════════════════════════════════════════════════════════════════

class TestDarkEnergy:
    def test_p_from_lambda(self):
        from bpr.impedance import DarkEnergyDensity
        # ρ_DE ~ κ / (p L²).  With κ = M_Pl² (in Planck units, not kg²)
        # and L = R_Hubble, p ~ κ / (Λ L²).
        # Using natural units: M_Pl = 1.22e19 GeV, R_H = 4.4e26 m
        # Observed Λ ~ 2.89e-122 M_Pl⁴ → p ~ 1/Λ_natural ~ 10⁶⁰
        # The DarkEnergyDensity class uses arbitrary units; the point
        # is that p is astronomically large.
        de = DarkEnergyDensity(kappa=1.0, p=1e60, L=1.0)
        assert de.rho_DE == pytest.approx(1e-60)

    def test_dark_energy_density_positive(self):
        from bpr.impedance import DarkEnergyDensity
        de = DarkEnergyDensity()
        assert de.rho_DE > 0


# ═══════════════════════════════════════════════════════════════════
# Predictions 4–5: Decoherence T_quantum and W_crit
# ═══════════════════════════════════════════════════════════════════

class TestDecoherenceSpecific:
    def test_sub_linear_decoherence(self):
        from bpr.decoherence import decoherence_rate_with_quantum_correction
        T = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 300.0])
        gamma = decoherence_rate_with_quantum_correction(
            T, delta_Z=1.0, A_eff=1e-14, lambda_dB=1e-10, T_quantum=1.0)
        # Below T_quantum: rate grows slower than linear
        # Above T_quantum: rate grows linearly
        # Ratio gamma[cold]/gamma[hot] should be LESS than T_cold/T_hot
        ratio_rate = gamma[0] / gamma[-1]
        ratio_T = T[0] / T[-1]
        assert ratio_rate < ratio_T

    def test_c60_is_quantum(self):
        from bpr.decoherence import is_quantum, critical_winding
        # C₆₀ has high ω, low Γ → should be quantum
        W_crit = critical_winding(gamma_dec=1e6, omega_system=1e12)
        assert W_crit < 1.0  # W_crit small → easy to be quantum
        assert is_quantum(W=1.0, gamma_dec=1e6, omega_system=1e12) is True

    def test_virus_is_classical(self):
        from bpr.decoherence import is_quantum, critical_winding
        # Virus has low ω, high Γ → should be classical
        W_crit = critical_winding(gamma_dec=1e12, omega_system=1e9)
        assert W_crit > 1.0
        assert is_quantum(W=1.0, gamma_dec=1e12, omega_system=1e9) is False


# ═══════════════════════════════════════════════════════════════════
# Predictions 6–8: Scale-free tipping, qubits, KZ anesthesia
# ═══════════════════════════════════════════════════════════════════

class TestScaleFreeTipping:
    def test_critical_fraction_decreases_with_N(self):
        from bpr.collective import TippingPoint
        fc_small = TippingPoint.scale_free_critical_fraction(N=1000)
        fc_large = TippingPoint.scale_free_critical_fraction(N=1_000_000)
        assert fc_large < fc_small

    def test_vanishes_for_large_N(self):
        from bpr.collective import TippingPoint
        fc = TippingPoint.scale_free_critical_fraction(N=10**9)
        assert fc < 0.01


class TestEffectiveQubits:
    def test_qubits_increase_with_W(self):
        from bpr.complexity import TopologicalParallelism
        q1 = TopologicalParallelism(p=7, W=1.0).effective_qubits()
        q3 = TopologicalParallelism(p=7, W=3.0).effective_qubits()
        assert q3 > q1
        assert q3 == pytest.approx(3 * q1)


class TestKZAnesthesia:
    def test_fast_wakeup_more_defects(self):
        from bpr.phase_transitions import kibble_zurek_defect_density
        n_fast = kibble_zurek_defect_density(tau_quench=10.0)
        n_slow = kibble_zurek_defect_density(tau_quench=300.0)
        assert n_fast > n_slow


# ═══════════════════════════════════════════════════════════════════
# Prediction 9: Casimir fine structure
# ═══════════════════════════════════════════════════════════════════

class TestCasimirFineStructure:
    def test_fine_structure_oscillates(self):
        from bpr.memory import casimir_fine_structure
        d = np.linspace(1e-6, 5e-6, 1000)
        modulation = casimir_fine_structure(d, p=7)
        # Should have sign changes (oscillatory)
        assert np.any(modulation > 0)
        assert np.any(modulation < 0)

    def test_amplitude_scales(self):
        from bpr.memory import casimir_fine_structure
        d = np.array([1e-6])
        m1 = casimir_fine_structure(d, base_amplitude=1e-8)
        m2 = casimir_fine_structure(d, base_amplitude=2e-8)
        assert abs(m2[0]) == pytest.approx(2 * abs(m1[0]))


# ═══════════════════════════════════════════════════════════════════
# Prediction 10: DM self-interaction
# ═══════════════════════════════════════════════════════════════════

class TestDMSelfInteraction:
    def test_below_bullet_cluster_limit(self):
        from bpr.impedance import dm_self_interaction_cross_section
        sigma_m = dm_self_interaction_cross_section(W_c=1.73)
        # Bullet Cluster bound: σ/m < 1 cm²/g
        assert sigma_m < 1.0

    def test_decreases_with_Wc(self):
        from bpr.impedance import dm_self_interaction_cross_section
        s1 = dm_self_interaction_cross_section(W_c=1.0)
        s2 = dm_self_interaction_cross_section(W_c=10.0)
        assert s2 < s1


# ═══════════════════════════════════════════════════════════════════
# Prediction 11: Muon g-2
# ═══════════════════════════════════════════════════════════════════

class TestMuonG2:
    def test_positive_correction(self):
        from bpr.impedance import muon_g2_correction
        dg2 = muon_g2_correction(W_c=1.73)
        assert dg2 > 0

    def test_larger_for_small_Wc(self):
        from bpr.impedance import muon_g2_correction
        dg2_small = muon_g2_correction(W_c=1.0)
        dg2_large = muon_g2_correction(W_c=100.0)
        assert dg2_small > dg2_large


# ═══════════════════════════════════════════════════════════════════
# Prediction 12: Hubble tension
# ═══════════════════════════════════════════════════════════════════

class TestHubbleTension:
    def test_delta_H0_positive(self):
        from bpr.impedance import hubble_tension
        ht = hubble_tension(R_boundary_0=1e26)
        assert ht["delta_H0"] > 0
        assert 3.0 < ht["delta_H0"] < 10.0

    def test_CMB_lower_than_local(self):
        from bpr.impedance import hubble_tension
        ht = hubble_tension(R_boundary_0=1e26)
        assert ht["H0_CMB_effective"] < ht["H0_local"]


# ═══════════════════════════════════════════════════════════════════
# Prediction 13: Superconductivity T_c
# ═══════════════════════════════════════════════════════════════════

class TestSuperconductorTc:
    def test_stronger_coupling_higher_Tc(self):
        from bpr.phase_transitions import superconductor_tc
        # BCS: larger N(0)V → higher T_c
        Tc_weak = superconductor_tc(N0V=0.2, T_debye=300.0)
        Tc_strong = superconductor_tc(N0V=0.4, T_debye=300.0)
        assert Tc_strong > Tc_weak

    def test_moderate_coupling_has_nonzero_Tc(self):
        from bpr.phase_transitions import superconductor_tc
        Tc = superconductor_tc(N0V=0.3, T_debye=300.0)
        assert Tc > 0
        assert Tc < 300.0  # Below Debye temperature

    def test_niobium_order_of_magnitude(self):
        from bpr.phase_transitions import superconductor_tc
        # Nb: N(0)V ≈ 0.29, T_D ≈ 275 K → Tc ≈ 9.25 K
        Tc = superconductor_tc(N0V=0.29, T_debye=275.0)
        assert 3 < Tc < 20  # order-of-magnitude correct


# ═══════════════════════════════════════════════════════════════════
# Prediction 14: Proton lifetime
# ═══════════════════════════════════════════════════════════════════

class TestProtonLifetime:
    def test_exceeds_superK_bound(self):
        from bpr.impedance import proton_lifetime
        tau = proton_lifetime(p=104729)
        assert tau > 1e34  # Super-Kamiokande bound in years

    def test_small_p_shorter_lifetime(self):
        from bpr.impedance import proton_lifetime
        tau_small = proton_lifetime(p=7)
        tau_large = proton_lifetime(p=104729)
        assert tau_small < tau_large or tau_large == float("inf")


# ═══════════════════════════════════════════════════════════════════
# Prediction 15: GW memory ringing
# ═══════════════════════════════════════════════════════════════════

class TestGWMemoryRinging:
    def test_ringing_decays(self):
        from bpr.gravitational_waves import gw_memory_ringing
        times = np.linspace(0, 10, 1000)
        h = gw_memory_ringing(times, p=7, tau_ring=2.0)
        # Should decay
        assert np.max(np.abs(h[:100])) > np.max(np.abs(h[-100:]))

    def test_oscillatory(self):
        from bpr.gravitational_waves import gw_memory_ringing
        times = np.linspace(0, 1, 10000)
        h = gw_memory_ringing(times, p=7, tau_ring=100.0)
        # Should have sign changes
        assert np.any(h > 0) and np.any(h < 0)


# ═══════════════════════════════════════════════════════════════════
# Prediction 16: Quantum error periodicity
# ═══════════════════════════════════════════════════════════════════

class TestQuantumErrorPeriod:
    def test_period_positive(self):
        from bpr.memory import quantum_error_period
        T = quantum_error_period(p=104729)
        assert T > 0

    def test_envelope_longer(self):
        from bpr.memory import quantum_error_period, quantum_error_envelope_period
        T = quantum_error_period(p=104729)
        T_env = quantum_error_envelope_period(p=104729)
        assert T_env > T
        assert T_env == pytest.approx(T * 104729)


# ═══════════════════════════════════════════════════════════════════
# Prediction 17: Aging reversal
# ═══════════════════════════════════════════════════════════════════

class TestAgingReversal:
    def test_enhancement_improves_healing(self):
        from bpr.bioelectric import AgingReversalPrediction
        arp = AgingReversalPrediction(age=60, enhancement=2.0)
        assert arp.wound_healing_improvement == pytest.approx(2.0)

    def test_rejuvenation_positive(self):
        from bpr.bioelectric import AgingReversalPrediction
        arp = AgingReversalPrediction(age=60, enhancement=2.0)
        assert arp.effective_rejuvenation_years > 0

    def test_no_enhancement_no_rejuvenation(self):
        from bpr.bioelectric import AgingReversalPrediction
        arp = AgingReversalPrediction(age=60, enhancement=1.0)
        assert arp.effective_rejuvenation_years == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════
# Prediction 18: Convergent evolution
# ═══════════════════════════════════════════════════════════════════

class TestConvergentEvolution:
    def test_identical_is_one(self):
        from bpr.bioelectric import convergent_evolution_similarity
        W = np.array([1.0, 0.5, 0.3])
        assert convergent_evolution_similarity(W, W) == pytest.approx(1.0)

    def test_similar_higher_than_different(self):
        from bpr.bioelectric import convergent_evolution_similarity
        W_a = np.array([1.0, 0.8, 0.3])
        W_b = np.array([1.0, 0.75, 0.35])  # similar
        W_c = np.array([0.1, 0.1, 0.1])     # different
        sim_ab = convergent_evolution_similarity(W_a, W_b)
        sim_ac = convergent_evolution_similarity(W_a, W_c)
        assert sim_ab > sim_ac


# ═══════════════════════════════════════════════════════════════════
# Prediction 19: Black hole entropy
# ═══════════════════════════════════════════════════════════════════

class TestBlackHoleEntropy:
    def test_agrees_with_bekenstein_hawking(self):
        from bpr.black_hole import BlackHoleEntropy
        bh = BlackHoleEntropy(M_solar=1.0)
        assert bh.agreement

    def test_p_independent(self):
        from bpr.black_hole import BlackHoleEntropy
        bh7 = BlackHoleEntropy(M_solar=1.0, p=7)
        bh_big = BlackHoleEntropy(M_solar=1.0, p=104729)
        assert bh7.entropy_bpr == pytest.approx(bh_big.entropy_bpr)

    def test_entropy_scales_with_mass_squared(self):
        from bpr.black_hole import BlackHoleEntropy
        bh1 = BlackHoleEntropy(M_solar=1.0)
        bh2 = BlackHoleEntropy(M_solar=2.0)
        # S ∝ A ∝ M² → S(2M) = 4 S(M)
        assert bh2.entropy_bpr == pytest.approx(4 * bh1.entropy_bpr, rel=1e-6)

    def test_hawking_temperature_positive(self):
        from bpr.black_hole import BlackHoleEntropy
        bh = BlackHoleEntropy(M_solar=1.0)
        assert bh.hawking_temperature > 0


# ═══════════════════════════════════════════════════════════════════
# Prediction 20: Three generations
# ═══════════════════════════════════════════════════════════════════

class TestThreeGenerations:
    def test_sphere_gives_three(self):
        from bpr.neutrino import number_of_generations
        assert number_of_generations("sphere") == 3

    def test_torus_gives_three(self):
        from bpr.neutrino import number_of_generations
        assert number_of_generations("torus") == 3

    def test_genus_2_gives_five(self):
        from bpr.neutrino import number_of_generations
        assert number_of_generations("genus_2") == 5

    def test_minimum_is_three(self):
        from bpr.neutrino import number_of_generations
        assert number_of_generations("genus_0") == 3


# ═══════════════════════════════════════════════════════════════════
# Pipeline: all 91 predictions generate without error
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_all_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert len(preds) >= 80, f"Expected 80+ predictions, got {len(preds)}"

    def test_neutrino_angles_in_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert "P5.5_theta12_deg" in preds
        assert 30 < preds["P5.5_theta12_deg"] < 40

    def test_black_hole_in_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert "P19.2_BH_agrees_BekensteinHawking" in preds
        assert preds["P19.2_BH_agrees_BekensteinHawking"] == True

    def test_proton_stable(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert preds["P2.16_proton_stable"] == True


# ═══════════════════════════════════════════════════════════════════
# Dark Sector: Derived Parameters (DarkSectorParameters, DarkEnergyDensity,
#              DarkMatterProfile corrected zeros)
# ═══════════════════════════════════════════════════════════════════

class TestDarkSectorParameters:
    def test_W_c_derived(self):
        """W_c = p^{1/5} matches substrate-derived scale."""
        from bpr.impedance import DarkSectorParameters, derived_critical_winding
        dsp = DarkSectorParameters()
        assert dsp.W_c == derived_critical_winding(104761)
        # For p=104761: W_c ≈ 10.094
        assert 10.0 < dsp.W_c < 11.0

    def test_m_defect_tev_scale(self):
        """DM soliton mass m_defect = p^{2/5} × v_EW is in the TeV range."""
        from bpr.impedance import DarkSectorParameters
        dsp = DarkSectorParameters()
        assert 1000.0 < dsp.m_defect_GeV < 1e6, "m_defect should be 1–1000 TeV"

    def test_p_cosmo_order_of_magnitude(self):
        """p_cosmo = R_H/l_Pl is ~10^{61}."""
        from bpr.impedance import DarkSectorParameters
        dsp = DarkSectorParameters()
        import math
        log10_p = math.log10(dsp.p_cosmo)
        assert 60 < log10_p < 63, f"p_cosmo expected ~10^{{61}}, got 10^{{{log10_p:.1f}}}"

    def test_p_hierarchy_large(self):
        """p_cosmo/p_local is a large hierarchy (~10^{56})."""
        from bpr.impedance import DarkSectorParameters
        import math
        dsp = DarkSectorParameters()
        log10_ratio = math.log10(dsp.p_hierarchy_ratio)
        assert 50 < log10_ratio < 60

    def test_relic_abundance_positive(self):
        """Derived relic abundance (with W_c = p^{1/5}) is positive and finite."""
        import math
        from bpr.impedance import DarkSectorParameters
        dsp = DarkSectorParameters()
        Omega = dsp.relic_abundance
        assert Omega > 0
        assert math.isfinite(Omega)


class TestDarkEnergyDensityHierarchy:
    def test_p_cosmo_derived_positive(self):
        """p_cosmo_derived = R_H/l_Pl is positive."""
        from bpr.impedance import DarkEnergyDensity
        de = DarkEnergyDensity()
        assert de.p_cosmo_derived > 0

    def test_rho_DE_derived_consistent(self):
        """rho_DE_derived uses p_cosmo, not the historical default 10^{60}."""
        from bpr.impedance import DarkEnergyDensity
        de = DarkEnergyDensity()
        # p_cosmo ≈ 2.72e61, not 1e60 — ratio should be ~27
        ratio = de.rho_DE / de.rho_DE_derived
        assert 10 < ratio < 100, (
            f"rho_DE(p=1e60) / rho_DE_derived(p_cosmo) expected ~27, got {ratio:.1f}"
        )

    def test_rho_DE_positive(self):
        """Both rho_DE variants are positive."""
        from bpr.impedance import DarkEnergyDensity
        de = DarkEnergyDensity()
        assert de.rho_DE > 0
        assert de.rho_DE_derived > 0


class TestDarkMatterProfileCorrectedZeros:
    def test_rpst_zeros_are_half_riemann(self):
        """DarkMatterProfile _ZETA_ZEROS ≈ γ_n/2 (corrected RPST set)."""
        from bpr.impedance import DarkMatterProfile
        from bpr.rpst_extensions import RIEMANN_ZEROS
        dm = DarkMatterProfile()
        for i, (rpst, riemann) in enumerate(zip(dm._ZETA_ZEROS, RIEMANN_ZEROS[:10])):
            expected = riemann / 2.0
            assert abs(rpst - expected) < 1e-8, (
                f"Zero {i+1}: got {rpst}, expected {expected}"
            )

    def test_profile_positive(self):
        """Corrected-zero DM profile returns positive density."""
        from bpr.impedance import DarkMatterProfile
        import numpy as np
        dm = DarkMatterProfile()
        r = np.linspace(0.01, 10.0, 100)
        rho = dm(r)
        assert (rho > 0).all()

    def test_zeros_smaller_than_riemann(self):
        """Corrected RPST zeros are smaller than original Riemann zeros."""
        from bpr.impedance import DarkMatterProfile
        from bpr.rpst_extensions import RIEMANN_ZEROS
        dm = DarkMatterProfile()
        for rpst, riemann in zip(dm._ZETA_ZEROS, RIEMANN_ZEROS[:10]):
            assert rpst < riemann
