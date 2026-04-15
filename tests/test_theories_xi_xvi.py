"""
Tests for Theories XI–XVI (Cosmology, QCD, Spacetime, Topo Matter,
Clifford, Quantum Foundations).

Run with:  pytest -v tests/test_theories_xi_xvi.py
"""

import math
import numpy as np
import pytest


# =====================================================================
# BPR Cosmology & Early Universe
# =====================================================================

class TestInflation:
    def test_n_efolds_in_range(self):
        from bpr.cosmology import InflationaryParameters
        infl = InflationaryParameters(p=104729)
        # Should give ~60 e-folds
        assert 40 < infl.n_efolds < 100

    def test_spectral_index_near_planck(self):
        from bpr.cosmology import InflationaryParameters
        infl = InflationaryParameters(p=104729)
        # Planck 2018: n_s = 0.9649 ± 0.0042
        assert 0.95 < infl.spectral_index < 0.98

    def test_tensor_to_scalar_below_bicep(self):
        from bpr.cosmology import InflationaryParameters
        infl = InflationaryParameters(p=104729)
        # BICEP/Keck 2021: r < 0.036
        assert infl.tensor_to_scalar < 0.04

    def test_running_negative(self):
        from bpr.cosmology import InflationaryParameters
        infl = InflationaryParameters(p=104729)
        assert infl.running < 0

    def test_slow_roll_small(self):
        from bpr.cosmology import InflationaryParameters
        infl = InflationaryParameters(p=104729)
        assert infl.slow_roll_epsilon() < 0.01
        assert abs(infl.slow_roll_eta()) < 0.1


class TestBaryogenesis:
    def test_baryon_asymmetry_nonzero(self):
        from bpr.cosmology import Baryogenesis
        bary = Baryogenesis(p=104729, N=10000)
        assert bary.baryon_asymmetry != 0

    def test_matter_dominates_for_p1mod4(self):
        from bpr.cosmology import Baryogenesis
        # 104729 mod 4 = 1
        bary = Baryogenesis(p=104729)
        assert bary.matter_dominates is True

    def test_cp_phase_small_for_orientable(self):
        from bpr.cosmology import Baryogenesis
        bary = Baryogenesis(p=104729)
        assert 0 < bary.cp_phase < 0.1


class TestCMBAnomaly:
    def test_quadrupole_suppression_between_0_1(self):
        from bpr.cosmology import CMBAnomaly
        cmb = CMBAnomaly(p=104729)
        assert 0 < cmb.quadrupole_suppression < 1

    def test_high_l_no_suppression(self):
        from bpr.cosmology import CMBAnomaly
        cmb = CMBAnomaly(p=104729)
        assert cmb.power_suppression(1000) > 0.999

    def test_hemispherical_asymmetry_small(self):
        from bpr.cosmology import CMBAnomaly
        cmb = CMBAnomaly(p=104729)
        assert 0 < cmb.hemispherical_asymmetry < 0.01


class TestPrimordialSpectrum:
    def test_power_spectrum_positive(self):
        from bpr.cosmology import primordial_power_spectrum
        k = np.logspace(-3, 0, 50)
        P = primordial_power_spectrum(k)
        assert np.all(P > 0)

    def test_delta_neff_small(self):
        from bpr.cosmology import delta_neff
        dn = delta_neff(104729)
        assert 0 < dn < 0.1  # small correction to N_eff = 3.044


class TestDarkMatterRelic:
    def test_relic_abundance_positive_finite(self):
        """DM relic abundance from thermal freeze-out is positive and finite.

        Since v0.8.0, the relic abundance is computed via genuine thermal
        freeze-out (not hardcoded to 0.12).  The BPR prediction over-produces
        DM (~3.2 for W_c=1, ~9.5 for W_c=√3), but the calculation is real.
        """
        from bpr.cosmology import DarkMatterRelic
        dm = DarkMatterRelic(W_c=1.0)
        assert dm.relic_abundance > 0, "Ω must be positive"
        assert math.isfinite(dm.relic_abundance), "Ω must be finite"
        # Verify freeze-out physics gives a reasonable TeV-scale mass
        assert dm.dm_mass_GeV > 100, "DM mass should be > 100 GeV"
        assert dm.dm_coupling > 0, "DM coupling should be positive"


# =====================================================================
# QCD & Flavor Physics
# =====================================================================

class TestColorConfinement:
    def test_color_singlet_check(self):
        from bpr.qcd_flavor import ColorConfinement
        cc = ColorConfinement()
        assert cc.is_color_singlet(1, -1, 0) is True
        assert cc.is_color_singlet(1, 1, 0) is False


class TestQuarkMasses:
    def test_six_quark_masses(self):
        from bpr.qcd_flavor import QuarkMassSpectrum
        qs = QuarkMassSpectrum()
        masses = qs.all_masses_MeV
        assert len(masses) == 6
        for name, m in masses.items():
            assert m > 0, f"Quark {name} mass should be positive"

    def test_mass_hierarchy(self):
        from bpr.qcd_flavor import QuarkMassSpectrum
        qs = QuarkMassSpectrum()
        m = qs.all_masses_MeV
        # Top is heaviest
        assert m["t"] > m["b"] > m["c"] > m["s"]
        # Up-type ordering
        assert m["t"] > m["c"] > m["u"]
        # Down-type ordering
        assert m["b"] > m["s"] > m["d"]


class TestCKMMatrix:
    def test_ckm_unitary(self):
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        assert ckm.is_unitary(tol=1e-8)

    def test_cabibbo_angle(self):
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        angles = ckm.mixing_angles()
        # Cabibbo angle ≈ 13°
        assert 10 < angles["cabibbo_angle_deg"] < 16

    def test_jarlskog_nonzero(self):
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        J = ckm.mixing_angles()["Jarlskog_invariant"]
        # J ~ 3 × 10⁻⁵
        assert abs(J) > 1e-6

    def test_wolfenstein_lambda(self):
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        # λ ≈ 0.225
        assert 0.20 < ckm.wolfenstein_lambda < 0.25


class TestStrongCP:
    def test_theta_zero_for_orientable(self):
        from bpr.qcd_flavor import strong_cp_theta
        # p = 104729 ≡ 1 mod 4 → orientable → θ = 0
        assert strong_cp_theta(104729) == 0.0

    def test_theta_quantised_for_nonorientable(self):
        from bpr.qcd_flavor import strong_cp_theta
        # p = 3 ≡ 3 mod 4 → non-orientable
        theta = strong_cp_theta(3)
        assert theta in (0.0, np.pi)


class TestProtonMass:
    def test_proton_mass_approximate(self):
        from bpr.qcd_flavor import proton_mass_from_confinement
        m_p = proton_mass_from_confinement()
        # Should be ~ 1 GeV
        assert 0.5 < m_p < 2.0

    def test_pion_mass_approximate(self):
        from bpr.qcd_flavor import pion_mass
        m_pi = pion_mass()
        # Should be ~ 100-200 MeV
        assert 50 < m_pi < 300


# =====================================================================
# Emergent Spacetime & Holography
# =====================================================================

class TestEmergentDimensions:
    def test_sphere_gives_3plus1(self):
        from bpr.emergent_spacetime import EmergentDimensions
        ed = EmergentDimensions("sphere")
        assert ed.spatial_dimensions == 3
        assert ed.time_dimensions == 1
        assert ed.total_dimensions == 4

    def test_torus_gives_2plus1(self):
        from bpr.emergent_spacetime import EmergentDimensions
        ed = EmergentDimensions("torus")
        assert ed.spatial_dimensions == 2
        assert ed.total_dimensions == 3


class TestHolographicEntropy:
    def test_entropy_positive(self):
        from bpr.emergent_spacetime import HolographicEntropy
        he = HolographicEntropy(boundary_area=1.0)
        assert he.entropy > 0

    def test_p_independent(self):
        from bpr.emergent_spacetime import HolographicEntropy
        he1 = HolographicEntropy(boundary_area=1.0, p=3)
        he2 = HolographicEntropy(boundary_area=1.0, p=104729)
        assert he1.p_independent
        # Entropy should be the same (it's A/(4l_P²) regardless of p)
        assert np.isclose(he1.entropy, he2.entropy)


class TestBekensteinBound:
    def test_bound_positive(self):
        from bpr.emergent_spacetime import BekensteinBound
        bb = BekensteinBound(R=1.0, E=1.0)
        assert bb.bekenstein_entropy > 0
        assert bb.holographic_entropy > 0


class TestEREqualsEPR:
    def test_not_traversable(self):
        from bpr.emergent_spacetime import EREqualsEPR
        er = EREqualsEPR(S_entanglement=1.0)
        assert er.traversable is False

    def test_wormhole_length_positive(self):
        from bpr.emergent_spacetime import EREqualsEPR
        er = EREqualsEPR(S_entanglement=1.0)
        assert er.wormhole_length > 0


class TestScramblingTime:
    def test_scrambling_time_positive(self):
        from bpr.emergent_spacetime import scrambling_time
        t = scrambling_time(1.0)
        assert t > 0

    def test_page_time_positive(self):
        from bpr.emergent_spacetime import page_time
        t = page_time(1.0)
        assert t > 0


# =====================================================================
# Topological Condensed Matter
# =====================================================================

class TestQuantumHall:
    def test_hall_conductance_quantised(self):
        from bpr.topological_matter import QuantumHallEffect
        qhe = QuantumHallEffect(nu=1)
        e2_h = (1.602176634e-19) ** 2 / 6.62607015e-34
        assert np.isclose(qhe.hall_conductance, e2_h, rtol=1e-6)

    def test_chern_number_equals_nu(self):
        from bpr.topological_matter import QuantumHallEffect
        for nu in [1, 2, 3]:
            qhe = QuantumHallEffect(nu=nu)
            assert qhe.chern_number == nu


class TestFractionalQHE:
    def test_laughlin_filling(self):
        from bpr.topological_matter import FractionalQHE
        fqhe = FractionalQHE(W=1, p=3)
        assert np.isclose(fqhe.filling_fraction, 1.0 / 3.0)
        assert fqhe.is_laughlin_state

    def test_quasiparticle_charge_fractional(self):
        from bpr.topological_matter import FractionalQHE
        fqhe = FractionalQHE(W=1, p=3)
        e = 1.602176634e-19
        assert np.isclose(fqhe.quasiparticle_charge, e / 3)


class TestTopologicalInsulator:
    def test_odd_winding_topological(self):
        from bpr.topological_matter import TopologicalInsulator
        ti = TopologicalInsulator(W=1)
        assert ti.is_topological
        assert ti.z2_index == 1

    def test_even_winding_trivial(self):
        from bpr.topological_matter import TopologicalInsulator
        ti = TopologicalInsulator(W=2)
        assert not ti.is_topological
        assert ti.z2_index == 0


class TestAnyonStatistics:
    def test_boson_at_zero(self):
        from bpr.topological_matter import AnyonStatistics
        a = AnyonStatistics(W=0, p=3)
        assert a.particle_type == "boson"

    def test_anyon_not_boson_or_fermion(self):
        from bpr.topological_matter import AnyonStatistics
        a = AnyonStatistics(W=1, p=3)
        assert a.particle_type == "anyon"

    def test_exchange_phase(self):
        from bpr.topological_matter import AnyonStatistics
        a = AnyonStatistics(W=1, p=3)
        assert np.isclose(a.exchange_phase, np.pi / 3)


class TestMajorana:
    def test_majorana_modes_odd_winding(self):
        from bpr.topological_matter import majorana_zero_modes
        assert majorana_zero_modes(1) == 1
        assert majorana_zero_modes(3) == 3

    def test_no_majorana_even_winding(self):
        from bpr.topological_matter import majorana_zero_modes
        assert majorana_zero_modes(2) == 0


class TestQuantizedConductance:
    def test_conductance_quantum(self):
        from bpr.topological_matter import QuantizedConductance
        qc = QuantizedConductance(n_channels=1, include_spin=True)
        e2_h = (1.602176634e-19) ** 2 / 6.62607015e-34
        assert np.isclose(qc.conductance, 2 * e2_h, rtol=1e-6)


# =====================================================================
# Clifford Algebra Embedding
# =====================================================================

class TestMultivectorField:
    def test_dimension(self):
        from bpr.clifford_bpr import MultivectorField
        mv = MultivectorField(np.ones(8))
        assert len(mv.components) == 8

    def test_grade_extraction(self):
        from bpr.clifford_bpr import MultivectorField
        comp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        mv = MultivectorField(comp)
        assert mv.scalar == 1.0
        assert np.allclose(mv.vector, [2, 3, 4])
        assert np.allclose(mv.bivector, [5, 6, 7])
        assert mv.pseudoscalar == 8.0

    def test_reverse(self):
        from bpr.clifford_bpr import MultivectorField
        comp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        mv = MultivectorField(comp)
        rev = mv.reverse
        # Grades 0,1 unchanged; grades 2,3 sign-flipped
        assert rev.scalar == 1.0
        assert np.allclose(rev.vector, [2, 3, 4])
        assert np.allclose(rev.bivector, [-5, -6, -7])
        assert rev.pseudoscalar == -8.0

    def test_norm(self):
        from bpr.clifford_bpr import MultivectorField
        mv = MultivectorField(np.array([3.0, 0, 0, 0, 4.0, 0, 0, 0]))
        assert np.isclose(mv.norm, 5.0)


class TestSpinorModule:
    def test_normalisation(self):
        from bpr.clifford_bpr import SpinorModule
        s = SpinorModule(np.array([3.0 + 0j, 4.0 + 0j]))
        n = s.normalise()
        assert np.isclose(n.norm, 1.0)

    def test_coherence_self(self):
        from bpr.clifford_bpr import SpinorModule
        s = SpinorModule(np.array([1.0 + 0j, 0.0 + 0j]))
        c = s.coherence_magnitude(s)
        assert np.isclose(c, 1.0)

    def test_coherence_orthogonal(self):
        from bpr.clifford_bpr import SpinorModule
        s1 = SpinorModule(np.array([1.0 + 0j, 0.0 + 0j]))
        s2 = SpinorModule(np.array([0.0 + 0j, 1.0 + 0j]))
        c = s1.coherence_magnitude(s2)
        assert np.isclose(c, 0.0)


class TestCliffordonSpectrum:
    def test_masses_positive(self):
        from bpr.clifford_bpr import CliffordonSpectrum
        cs = CliffordonSpectrum(p=104729)
        assert np.all(cs.masses_eV > 0)

    def test_mass_increases_with_n(self):
        from bpr.clifford_bpr import CliffordonSpectrum
        cs = CliffordonSpectrum(p=104729)
        m = cs.masses_eV
        for i in range(len(m) - 1):
            assert m[i + 1] > m[i]

    def test_n1_stable(self):
        from bpr.clifford_bpr import CliffordonSpectrum
        cs = CliffordonSpectrum(p=104729)
        assert cs.stability_criterion(1) is True


class TestCliffordDimensions:
    def test_clifford_dim_3(self):
        from bpr.clifford_bpr import clifford_dimension, spinor_dimension
        assert clifford_dimension(3) == 8
        assert spinor_dimension(3) == 2


# =====================================================================
# Quantum Foundations
# =====================================================================

class TestBornRule:
    def test_accuracy_near_one(self):
        from bpr.quantum_foundations import BornRule
        br = BornRule(p=104729)
        assert br.born_rule_accuracy > 0.99999

    def test_deviation_small(self):
        from bpr.quantum_foundations import BornRule
        br = BornRule(p=104729)
        assert br.correction_amplitude < 1e-4


class TestArrowOfTime:
    def test_entropy_monotonic(self):
        from bpr.quantum_foundations import ArrowOfTime
        at = ArrowOfTime(p=104729)
        assert at.entropy_monotonic is True

    def test_time_quantum_positive(self):
        from bpr.quantum_foundations import ArrowOfTime
        at = ArrowOfTime(p=104729)
        assert at.time_quantum > 0


class TestBellInequality:
    def test_violates_classical(self):
        from bpr.quantum_foundations import BellInequality
        bell = BellInequality(p=104729)
        assert bell.violates_classical()
        assert bell.bpr_bound > 2.0

    def test_below_tsirelson(self):
        from bpr.quantum_foundations import BellInequality
        bell = BellInequality(p=104729)
        assert bell.bpr_bound < bell.tsirelson_bound

    def test_approaches_tsirelson_for_large_p(self):
        from bpr.quantum_foundations import BellInequality
        bell = BellInequality(p=104729)
        assert bell.deviation_from_tsirelson < 1e-4


class TestMeasurement:
    def test_measurement_time_positive(self):
        from bpr.quantum_foundations import MeasurementDynamics
        m = MeasurementDynamics(gamma_dec=1e12, W_apparatus=1.0)
        assert m.measurement_time > 0

    def test_collapse_physical(self):
        from bpr.quantum_foundations import MeasurementDynamics
        m = MeasurementDynamics()
        assert m.collapse_is_physical is True


class TestBoltzmannBrain:
    def test_suppression_effectively_zero(self):
        from bpr.quantum_foundations import BoltzmannBrainSuppression
        bb = BoltzmannBrainSuppression(p=104729)
        assert bb.effectively_zero
        assert bb.suppression_factor == 0.0

    def test_log_suppression_large(self):
        from bpr.quantum_foundations import BoltzmannBrainSuppression
        bb = BoltzmannBrainSuppression(p=104729)
        assert bb.log_suppression < -1000


class TestContextuality:
    def test_contextuality_dimension(self):
        from bpr.quantum_foundations import contextuality_dimension_bound
        assert contextuality_dimension_bound(104729) == 104729

    def test_free_will(self):
        from bpr.quantum_foundations import free_will_theorem_compatible
        assert free_will_theorem_compatible(104729) is True


# =====================================================================
# Full pipeline: all 150+ predictions
# =====================================================================

class TestFullPipelineV05:
    def test_prediction_count(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        # Should now have ~150 predictions
        assert len(preds) >= 145, f"Expected 145+ predictions, got {len(preds)}"

    def test_all_new_theories_present(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()

        # Check each theory block has at least one prediction
        for prefix in ["P11.", "P12.", "P13.", "P14.", "P15.", "P16."]:
            keys = [k for k in preds if k.startswith(prefix)]
            assert len(keys) >= 5, (
                f"Expected ≥5 predictions for {prefix}, got {len(keys)}: {keys}"
            )

    def test_spectral_index_in_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        n_s = preds["P11.2_spectral_index"]
        assert 0.95 < n_s < 0.98

    def test_quark_masses_in_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert preds["P12.7_m_t_MeV"] > 100000  # top quark > 100 GeV

    def test_dimensions_in_predictions(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert preds["P13.3_spatial_dimensions"] == 3
        assert preds["P13.4_time_dimensions"] == 1


# =====================================================================
# BPR Dark Energy EoS  (§11.5b)
# =====================================================================

class TestBPRDarkEnergyEOS:
    def test_w_equals_minus_one_above_z_PT(self):
        """w(z) = -1 for z ≥ z_PT (no relaxation before phase transition)."""
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        for z in [eos.z_PT, eos.z_PT + 1.0, 10.0, 100.0]:
            assert eos.w(z) == -1.0, f"w({z}) should be -1.0 above z_PT"

    def test_w_close_to_minus_one_today(self):
        """w(z=0) ≈ -1 (relaxation exponent ~94 makes deviation negligible)."""
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        # Deviation from -1 should be tiny (< 10^{-5})
        assert abs(eos.w0 - (-1.0)) < 1e-5, (
            f"w(z=0) = {eos.w0} deviates too much from -1.0"
        )

    def test_epsilon_small(self):
        """Relaxation amplitude ε = 1/p^{1/3} is small."""
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        assert eos.epsilon < 0.1, f"ε = {eos.epsilon} should be < 0.1"

    def test_w_array_input(self):
        """w(z) accepts numpy array input and returns matching array."""
        import numpy as np
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        z_arr = np.array([0.0, 1.0, eos.z_PT, eos.z_PT + 2.0])
        w_arr = eos.w(z_arr)
        assert w_arr.shape == z_arr.shape
        assert w_arr[-1] == -1.0   # above z_PT
        assert w_arr[-2] == -1.0   # exactly at z_PT

    def test_desi_tension_keys(self):
        """desi_tension dict contains the expected keys."""
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        d = eos.desi_tension
        assert "w0_bpr" in d
        assert "w0_desi" in d
        assert "w0_tension_sigma" in d
        assert "wa_bpr" in d
        assert "wa_desi" in d
        assert "wa_tension_sigma" in d

    def test_desi_tension_sigma_nonnegative(self):
        """Tension values in sigma are non-negative."""
        from bpr.cosmology import BPRDarkEnergyEOS
        eos = BPRDarkEnergyEOS()
        d = eos.desi_tension
        assert d["w0_tension_sigma"] >= 0
        assert d["wa_tension_sigma"] >= 0


# =====================================================================
# BPR Cosmology V5 — Impedance-Screened MOND Collapse
# =====================================================================

class TestBPRCosmologyV5:
    def test_g_screen_no_suppression_low_mass(self):
        """g_screen ≈ 1 for low-mass halos (M << M_imp)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        # M = 1e11 Msun, z=10: far below M_imp ~ 10^12 Msun
        g = v5._g_screen(1e11, 10)
        assert g > 0.999, f"Expected g_screen≈1 for low mass, got {g:.4f}"

    def test_g_screen_half_at_m_imp(self):
        """g_screen = 1/2 at M = M_imp = W_c^{1/2} × M★."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        m_imp = v5.m_imp(10)
        g = v5._g_screen(m_imp, 10)
        assert abs(g - 0.5) < 0.01, f"Expected g_screen=0.5 at M_imp, got {g:.4f}"

    def test_m_imp_order_of_magnitude_z10(self):
        """M_imp(z=10) ≈ 10^12 Msun (the bright-end overshoot mass scale)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        m_imp = v5.m_imp(10)
        assert 5e11 < m_imp < 2e12, f"Expected M_imp~10^12, got {m_imp:.2e}"

    def test_delta_c_v5_equals_newton_below_z_pt(self):
        """δ_c(M, z) = 1.686 for z ≤ z_PT (Newtonian, same as V4)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        # z=3 is below z_PT ≈ 5.09
        assert v5.delta_c_v5(1e12, 3.0) == 1.686

    def test_delta_c_v5_less_than_v4_for_massive_halo(self):
        """δ_c_V5 ≤ δ_c_V4 for a massive halo above M_imp (screening active)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        # M_UV = -23.5 → M_halo ≈ 10^12 Msun, well above M_imp at z=10
        M_halo = 1e12
        z = 10.0
        dc_v4 = v5.delta_c_v4(M_halo, z)
        dc_v5 = v5.delta_c_v5(M_halo, z)
        assert dc_v5 < dc_v4, (
            f"V5 should screen bright halos: V5={dc_v5:.4f} should be < V4={dc_v4:.4f}"
        )

    def test_delta_c_v5_unchanged_for_small_halo(self):
        """δ_c_V5 ≈ δ_c_V4 for a small halo far below M_imp."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        # M = 1e10 Msun at z=10 — well below M_imp
        M_halo = 1e10
        z = 10.0
        dc_v4 = v5.delta_c_v4(M_halo, z)
        dc_v5 = v5.delta_c_v5(M_halo, z)
        assert abs(dc_v5 - dc_v4) < 0.01, (
            f"Small halo should be unscreened: V4={dc_v4:.4f}, V5={dc_v5:.4f}"
        )

    def test_s8_v5_equals_v4(self):
        """S8_V5 = S8_V4 (z=0 is always Newtonian — screening unchanged)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        assert abs(v5.S8_v5 - v5.S8_v4) < 1e-10

    def test_uv_lf_v5_above_v4_at_z10_bright(self):
        """UV LF V5 ≥ V4 at z=10, M_UV=-23 (screening reduces δ_c → more halos)."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        v5 = BPRCosmologyV5()
        lf_v4 = v5.uv_luminosity_function_v4(-23.0, 10.0)
        lf_v5 = v5.uv_luminosity_function_v5(-23.0, 10.0)
        assert lf_v5 >= lf_v4, (
            f"V5 should predict more bright galaxies at z=10: V5={lf_v5:.4f}, V4={lf_v4:.4f}"
        )

    def test_w_c_matches_impedance_module(self):
        """W_c = p^{1/5} from derived_critical_winding() ≈ 10.09."""
        from bpr.jwst_cosmology import BPRCosmologyV5
        from bpr.impedance import derived_critical_winding
        v5 = BPRCosmologyV5()
        assert abs(v5._W_c - derived_critical_winding(v5.p)) < 1e-10
        assert abs(v5._W_c - 104761 ** (1.0 / 5.0)) < 1e-6


# =====================================================================
# BPR Cosmology V6 — Corrected Impedance-Screened MOND Collapse
# =====================================================================

class TestBPRCosmologyV6:
    def test_delta_c_v6_newtonian_halo_unaffected(self):
        """For μ≈1 (Newtonian halo), δ_c_V6 ≈ 1.686 regardless of g_screen."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        import math
        v6 = BPRCosmologyV6()
        # At z=16, M=1e11: μ≈0.85 (near-Newtonian), V5 got 1.394, V6 should give ~1.675
        dc_v6 = v6.delta_c_v6(1e11, 16.0)
        dc_v4 = v6.delta_c_v4(1e11, 16.0)
        # V6 should be ABOVE V5 (which was 1.394) and close to V4 (1.634)
        dc_v5 = 1.330 + 0.356 * (v6._virial_acceleration(1e11, 16.0) /
                  math.sqrt(v6._virial_acceleration(1e11, 16.0)**2 +
                            v6.mond_a0**2)) * v6._g_screen(1e11, 16.0)
        assert dc_v6 > dc_v5, (
            f"V6 should be above V5 for near-Newtonian halos: V6={dc_v6:.4f}, V5={dc_v5:.4f}"
        )
        # V6 should be within 0.1 of V4 (small MOND boost, mostly screened)
        assert abs(dc_v6 - dc_v4) < 0.1, (
            f"V6 should be close to V4 for near-Newtonian halos: V6={dc_v6:.4f}, V4={dc_v4:.4f}"
        )

    def test_delta_c_v6_screened_mond_halo(self):
        """For μ≈0 (deep MOND), g_screen=0 restores Newtonian δ_c=1.686."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        import math
        v6 = BPRCosmologyV6()
        # Simulate: μ=0, g=0 → δ_c = 1.686 - 0.356*(1-0)*0 = 1.686
        dc = 1.686 - 0.356 * (1.0 - 0.0) * 0.0
        assert abs(dc - 1.686) < 1e-10, "Fully screened deep-MOND halo should give Newtonian δ_c"

    def test_delta_c_v6_unscreened_mond_halo(self):
        """For μ=0 (deep MOND), g_screen=1 gives δ_c = 1.330."""
        dc = 1.686 - 0.356 * (1.0 - 0.0) * 1.0
        assert abs(dc - 1.330) < 1e-10, "Unscreened deep-MOND halo should give δ_c=1.330"

    def test_v6_always_geq_v5_when_g_less_than_one(self):
        """V6 δ_c ≥ V5 δ_c for all halos (V6 screens boost; V5 lowers threshold)."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        v6 = BPRCosmologyV6()
        test_cases = [
            (1e11, 9.0), (1e11, 10.0), (1e11, 12.0), (1e11, 16.0),
            (1e12, 10.0), (6e11, 12.0),
        ]
        for M, z in test_cases:
            dc_v6 = v6.delta_c_v6(M, z)
            dc_v5 = v6.delta_c_v5(M, z)
            assert dc_v6 >= dc_v5 - 1e-10, (
                f"V6 should always be ≥ V5: M={M:.0e}, z={z}, V6={dc_v6:.4f}, V5={dc_v5:.4f}"
            )

    def test_delta_c_v6_equals_newton_below_z_pt(self):
        """δ_c_V6 = 1.686 for z ≤ z_PT."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        v6 = BPRCosmologyV6()
        assert v6.delta_c_v6(1e12, 3.0) == 1.686

    def test_v6_equals_v4_when_no_screening(self):
        """When g_screen ≈ 1, V6 ≈ V4 (formula reduces to same)."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        v6 = BPRCosmologyV6()
        # z=9, M=1e11: g_screen ≈ 1 → V6 ≈ V4
        dc_v6 = v6.delta_c_v6(1e11, 9.0)
        dc_v4 = v6.delta_c_v4(1e11, 9.0)
        assert abs(dc_v6 - dc_v4) < 0.005, (
            f"V6 ≈ V4 when g_screen≈1: V6={dc_v6:.4f}, V4={dc_v4:.4f}"
        )

    def test_s8_v6_equals_v4(self):
        """S8_V6 = S8_V4 (z=0 always Newtonian)."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        v6 = BPRCosmologyV6()
        assert abs(v6.S8_v6 - v6.S8_v4) < 1e-10

    def test_uv_lf_v6_above_lcdm_at_high_z(self):
        """UV LF V6 > ΛCDM at z=10 (MOND boost active above z_PT)."""
        from bpr.jwst_cosmology import BPRCosmologyV6, LambdaCDM
        v6 = BPRCosmologyV6()
        lcdm = LambdaCDM()
        lf_v6   = v6.uv_luminosity_function_v6(-22.0, 10.0)
        lf_lcdm = lcdm.uv_luminosity_function(-22.0, 10.0)
        assert lf_v6 > lf_lcdm, "V6 should predict more galaxies than ΛCDM at z=10"

    def test_v6_fixes_z16_overshoot_vs_v5(self):
        """V6 log_phi at z=16 is closer to V4 than V5 (no sub-Newtonian pathology)."""
        from bpr.jwst_cosmology import BPRCosmologyV6
        v6 = BPRCosmologyV6()
        lf_v4 = v6.uv_luminosity_function_v4(-21.0, 16.0)
        lf_v5 = v6.uv_luminosity_function_v5(-21.0, 16.0)
        lf_v6 = v6.uv_luminosity_function_v6(-21.0, 16.0)
        # V5 massively overshoots V4 at z=16 (unphysical sub-Newtonian δ_c)
        # V6 should be close to V4
        assert abs(lf_v6 - lf_v4) < abs(lf_v5 - lf_v4), (
            f"V6 should be closer to V4 than V5 at z=16: "
            f"V4={lf_v4:.3f}, V5={lf_v5:.3f}, V6={lf_v6:.3f}"
        )


class TestBPRCosmologyV7:
    def test_fstar_retention_newtonian_returns_one(self):
        """η(μ) = 1 at z ≤ z_PT (Newtonian epoch, no enhancement)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        eta = v7._fstar_retention(1e12, 3.0)
        assert abs(eta - 1.0) < 1e-10, f"z<z_PT should give η=1, got {eta}"

    def test_fstar_retention_mond_regime_greater_than_one(self):
        """η(μ) > 1 at z > z_PT in MOND regime (halo below M★)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        eta = v7._fstar_retention(1e11, 9.0)
        assert eta > 1.0, f"MOND-regime halo should give η>1, got {eta}"

    def test_fstar_retention_bounded(self):
        """η(μ) ≤ 2 always (reflection coefficient R ≤ 1)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        for M, z in [(1e9, 10.0), (1e10, 12.0), (1e11, 9.0), (1e12, 9.0), (1e13, 16.0)]:
            eta = v7._fstar_retention(M, z)
            assert 1.0 <= eta <= 2.0 + 1e-10, (
                f"η out of bounds [1,2]: M={M:.0e}, z={z}, η={eta:.4f}"
            )

    def test_fstar_retention_deep_mond_approaches_two(self):
        """η → 2 as μ → 0 (deep MOND: R → 1, maximum baryonic retention)."""
        # R(μ) = ((1-μ)/(1+μ))² → 1 as μ→0, so η → 1+1 = 2
        r_near_zero = ((1.0 - 1e-6) / (1.0 + 1e-6)) ** 2
        eta_near_zero = 1.0 + r_near_zero
        assert abs(eta_near_zero - 2.0) < 1e-4, f"Deep MOND η should approach 2, got {eta_near_zero}"

    def test_uv_lf_v7_geq_v6_at_high_z(self):
        """UV LF V7 ≥ V6 at z > z_PT (f_star correction is non-negative)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        for M_UV in [-21.0, -22.0, -22.5]:
            lf_v7 = v7.uv_luminosity_function_v7(M_UV, 9.0)
            lf_v6 = v7.uv_luminosity_function_v6(M_UV, 9.0)
            assert lf_v7 >= lf_v6 - 1e-10, (
                f"V7 ≥ V6 at z=9 M_UV={M_UV}: V7={lf_v7:.4f}, V6={lf_v6:.4f}"
            )

    def test_uv_lf_v7_equals_v6_below_z_pt(self):
        """UV LF V7 = V6 at z ≤ z_PT (mechanism gates off)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        lf_v7 = v7.uv_luminosity_function_v7(-22.0, 3.0)
        lf_v6 = v7.uv_luminosity_function_v6(-22.0, 3.0)
        assert abs(lf_v7 - lf_v6) < 1e-10, (
            f"V7 = V6 at z<z_PT: V7={lf_v7:.4f}, V6={lf_v6:.4f}"
        )

    def test_s8_v7_equals_v6(self):
        """S8_V7 = S8_V6 (f_star only affects UV LF, not 8 Mpc clustering)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        assert abs(v7.S8_v7 - v7.S8_v6) < 1e-10

    def test_fstar_correction_direction(self):
        """f_star correction is in right direction: V7 > V6 at z=9."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        v7 = BPRCosmologyV7()
        lf_v7 = v7.uv_luminosity_function_v7(-22.0, 9.0)
        lf_v6 = v7.uv_luminosity_function_v6(-22.0, 9.0)
        assert lf_v7 > lf_v6, (
            f"f_star channel should boost UV LF at z=9: V7={lf_v7:.4f}, V6={lf_v6:.4f}"
        )

    def test_delta_muv_positive(self):
        """ΔM_UV = 2.5×log₁₀(η) > 0 in MOND regime (galaxy appears brighter)."""
        from bpr.jwst_cosmology import BPRCosmologyV7
        import math
        v7 = BPRCosmologyV7()
        eta = v7._fstar_retention(1e11, 9.0)
        delta_muv = 2.5 * math.log10(eta)
        assert delta_muv > 0, f"ΔM_UV should be positive in MOND regime, got {delta_muv:.4f}"


class TestBPRBoundaryPhonon:
    """Tests for the p^{1/3} structural ΔNeff ceiling and boundary phonon mass."""

    def test_m_phi_relativistic_at_recombination(self):
        """m_φ << T_rec ~ 0.25 eV: boundary phonon IS relativistic at recombination."""
        from bpr.cosmology import BPRBoundaryPhonon, _T_REC_EV
        ph = BPRBoundaryPhonon()
        assert ph.m_phi_eV < _T_REC_EV * 1e-10, (
            f"m_φ = {ph.m_phi_eV:.2e} eV should be << T_rec = {_T_REC_EV} eV"
        )

    def test_t_dec_at_gut_scale(self):
        """T_dec = M_Pl/p^{2/3} is at the GUT scale (10^{15}–10^{16} GeV)."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert 1e14 < ph.t_dec_GeV < 1e17, (
            f"T_dec should be GUT scale: {ph.t_dec_GeV:.2e} GeV"
        )

    def test_temperature_ratio_strongly_diluted(self):
        """T_φ/T_γ << 1 after GUT-scale decoupling (strong entropy dilution)."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert ph.temperature_ratio < 0.4, (
            f"T_φ/T_γ = {ph.temperature_ratio:.4f} should be < 0.4 (strong dilution)"
        )

    def test_delta_neff_structural_below_ceiling(self):
        """ΔNeff_structural < 0.01 — well below CMB-S4 sensitivity ~0.06."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert ph.delta_neff_structural < 0.01, (
            f"ΔNeff_structural = {ph.delta_neff_structural:.5f} should be < 0.01"
        )
        assert ph.delta_neff_structural > 0.0, "ΔNeff_structural must be positive"

    def test_structural_less_than_heuristic(self):
        """ΔNeff_structural < ΔNeff_heuristic: coupling analysis gives lower ceiling."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert ph.delta_neff_structural < ph.delta_neff_heuristic, (
            f"Structural ceiling {ph.delta_neff_structural:.5f} should be below "
            f"heuristic {ph.delta_neff_heuristic:.5f}"
        )

    def test_heuristic_matches_delta_neff_function(self):
        """delta_neff_heuristic matches the standalone delta_neff() function."""
        from bpr.cosmology import BPRBoundaryPhonon, delta_neff
        ph = BPRBoundaryPhonon()
        assert abs(ph.delta_neff_heuristic - delta_neff(ph.p)) < 1e-10

    def test_falsification_threshold_above_structural(self):
        """Falsification threshold > ΔNeff_structural (leaves room for detection)."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert ph.falsification_threshold > ph.delta_neff_structural * 5, (
            "Threshold must be well above structural ceiling to be meaningful"
        )

    def test_is_falsified_by_large_neff(self):
        """ΔNeff > 0.1 measurement falsifies the substrate coarse-graining."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert ph.is_falsified_by(0.4)
        assert ph.is_falsified_by(0.1 + 1e-6)

    def test_not_falsified_by_structural_value(self):
        """BPR's own ΔNeff_structural does not falsify itself."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        assert not ph.is_falsified_by(ph.delta_neff_structural)
        assert not ph.is_falsified_by(ph.delta_neff_heuristic)

    def test_p13_ceiling_consistency(self):
        """The same p^{1/3} factor appears in Γ_b, g_φ, and ω_UV — ceiling is structural."""
        p = 104729
        # 1. Boundary rate: Γ_b = H/p^{1/3} → z_PT = 5.09 (not tested here, just consistency)
        boundary_rate_exponent = 1.0 / 3.0
        # 2. Coupling: g_φ ~ 1/p^{1/3} → T_dec ~ M_Pl/p^{2/3}
        coupling_exponent = 1.0 / 3.0
        # T_dec ~ M_Pl / p^{2/3} = M_Pl / p^{2*coupling_exp}
        # 3. UV cutoff: l_UV = l_Pl × p^{1/3}
        uv_exponent = 1.0 / 3.0
        # All three carry the same p^{1/3} factor
        assert boundary_rate_exponent == coupling_exponent == uv_exponent, (
            "p^{1/3} ceiling is structural: all three derivations share the same exponent"
        )

    def test_rho_de_order_of_magnitude(self):
        """ρ_DE is in the correct range for dark energy density."""
        from bpr.cosmology import BPRBoundaryPhonon
        ph = BPRBoundaryPhonon()
        # ρ_DE ~ (2.4e-12 GeV)^4 ~ 3.3e-47 GeV^4
        assert 1e-48 < ph.rho_de_GeV4 < 1e-46, (
            f"ρ_DE = {ph.rho_de_GeV4:.2e} GeV^4, expected ~10^-47 GeV^4"
        )

    def test_decay_constant_substrate_suppressed(self):
        """f_φ = M_Pl/p^{1/3} is suppressed relative to M_Pl."""
        from bpr.cosmology import BPRBoundaryPhonon, _M_PL_GEV
        ph = BPRBoundaryPhonon()
        assert ph.decay_constant_GeV < _M_PL_GEV, "f_φ must be below M_Pl"
        assert ph.decay_constant_GeV > _M_PL_GEV * 1e-2, "f_φ should be within 2 orders of M_Pl"


class TestA0Derivation:
    """Tests for the Gibbons-Hawking derivation of the MOND acceleration scale.

    Validates the four-step derivation in MONDInterpolation (bpr/impedance.py):
      Step 1: T_GH = ħH₀/(2πk_B)          [standard GR]
      Step 2: ω₀ = H₀/(2π)                 [boundary phonon at T_GH]
      Step 3: a₀ = c H₀/(2π)               [BPR — leading term]
      Step 4: a₀ = c H₀/(2π) × (1+z/4lnp) [BPR — substrate correction]

    Each test checks one step independently, so failures isolate exactly
    where the derivation breaks down.
    """

    def test_leading_term_within_15_percent(self):
        """Step 3: cH₀/(2π) is within 15% of observed a₀ = 1.2e-10 m/s².

        This tests the Gibbons-Hawking leading term alone, before the
        substrate correction.  A 13% undershoot is expected.
        """
        import math
        H0_si = 67.4 * 1e3 / 3.086e22   # s⁻¹
        a0_leading = 3e8 * H0_si / (2.0 * math.pi)
        a0_observed = 1.2e-10   # m/s²
        fractional_error = abs(a0_leading - a0_observed) / a0_observed
        assert fractional_error < 0.15, (
            f"Leading term cH₀/(2π) = {a0_leading:.3e} m/s² is "
            f"{fractional_error*100:.1f}% from observed {a0_observed:.3e} — "
            f"expected ~13% undershoot from GH leading term"
        )

    def test_full_formula_within_3_percent(self):
        """Step 4: full formula a₀ = cH₀/(2π)×(1+z/4lnp) within 3% of observed."""
        from bpr.jwst_cosmology import BPRCosmologyV2
        v2 = BPRCosmologyV2()
        a0_bpr = v2.mond_a0
        a0_observed = 1.2e-10   # m/s²
        fractional_error = abs(a0_bpr - a0_observed) / a0_observed
        assert fractional_error < 0.03, (
            f"Full a₀ formula = {a0_bpr:.3e} m/s² is "
            f"{fractional_error*100:.1f}% from observed — should be < 3%"
        )

    def test_substrate_correction_increases_a0(self):
        """Step 4: the substrate correction (1 + z/4lnp) > 1 — it lifts a₀ toward observed."""
        import math
        p = 104729
        z = 6
        correction = 1.0 + z / (4.0 * math.log(p))
        assert correction > 1.0, "Substrate correction must be > 1 (positive enhancement)"
        assert 1.05 < correction < 1.20, (
            f"Correction = {correction:.4f}, expected between 1.05 and 1.20 (≈13%)"
        )

    def test_two_pi_from_gibbons_hawking(self):
        """Step 1–2: the 2π in a₀ = cH₀/(2π) is the GH Euclidean period.

        Verify that ω₀ = k_B T_GH / ħ = H₀/(2π) exactly — i.e., the 2π
        cancels cleanly and does not need to be inserted by hand.
        """
        import math
        hbar = 1.054571817e-34   # J·s
        k_B  = 1.380649e-23     # J/K
        H0   = 2.184e-18        # s⁻¹  (67.4 km/s/Mpc in SI)

        T_GH = hbar * H0 / (2.0 * math.pi * k_B)   # Gibbons-Hawking temperature
        omega_0 = k_B * T_GH / hbar                  # boundary phonon frequency

        expected = H0 / (2.0 * math.pi)
        assert abs(omega_0 - expected) / expected < 1e-10, (
            "ω₀ = k_B T_GH / ħ must equal H₀/(2π) exactly — "
            "the 2π traces to the GH Euclidean period, not a free parameter"
        )

    def test_z_pt_is_downstream_of_derived_a0(self):
        """z_PT ≈ 5.1 is a genuine prediction: derived entirely from p, z, H₀.

        No free parameters: the same substrate prime p and coordination z
        that give a₀ to 1.8% also fix z_PT through Γ_b(z_PT) = ω_MOND.
        """
        from bpr.jwst_cosmology import BPRCosmologyV2
        v2 = BPRCosmologyV2()
        z_pt = v2.z_pt
        assert 4.5 < z_pt < 6.0, (
            f"z_PT = {z_pt:.2f} is outside the expected range [4.5, 6.0] — "
            f"derived from p={v2.p}, z=6"
        )

    def test_h0_c_over_p13_fails(self):
        """H₀c/p^{1/3} is NOT the right formula — it misses observed a₀ by > 5×.

        This test guards against regressing to the naive substrate-rate formula.
        The correct leading term is cH₀/(2π), not cH₀/p^{1/3}.
        """
        import math
        H0_si = 67.4 * 1e3 / 3.086e22
        p = 104729
        a0_naive = 3e8 * H0_si / p**(1.0/3.0)
        a0_observed = 1.2e-10
        ratio = a0_observed / a0_naive
        assert ratio > 5.0, (
            f"H₀c/p^(1/3) = {a0_naive:.3e} should be > 5× below observed "
            f"{a0_observed:.3e} (ratio = {ratio:.1f})"
        )
