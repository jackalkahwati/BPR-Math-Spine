"""
Tests for Theories XI–XVI (Cosmology, QCD, Spacetime, Topo Matter,
Clifford, Quantum Foundations).

Run with:  pytest -v tests/test_theories_xi_xvi.py
"""

import math
import numpy as np
import pytest


# =====================================================================
# Theory XI: Cosmology
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
# Theory XII: QCD & Flavor
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
# Theory XIII: Emergent Spacetime
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
# Theory XIV: Topological Condensed Matter
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
# Theory XV: Clifford Algebra
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
# Theory XVI: Quantum Foundations
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
