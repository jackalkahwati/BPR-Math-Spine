"""
Tests for Theories XVII–XXI (Gauge Unification, Charged Leptons,
Nuclear Physics, Quantum Gravity Phenomenology, Quantum Chemistry).

Run with:  pytest -v tests/test_theories_xvii_xxi.py
"""

import numpy as np
import pytest


# =====================================================================
# Theory XVII: Gauge Unification
# =====================================================================

class TestGaugeCoupling:
    def test_alpha_em_at_MZ(self):
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning()
        # α₂(M_Z) ≈ 1/29
        assert 0.03 < gc.alpha2_MZ < 0.04

    def test_alpha_s_at_MZ(self):
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning()
        assert abs(gc.alpha3_MZ - 0.1179) < 0.001

    def test_gut_scale_order(self):
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning(p=104729)
        # Should be ~10¹⁶ GeV
        assert 1e14 < gc.unification_scale_GeV < 1e18

    def test_running_monotonic(self):
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning()
        # α₃ should decrease with energy (asymptotic freedom)
        a3_low = gc.alpha_i(3, 100.0)
        a3_high = gc.alpha_i(3, 1e10)
        assert a3_high < a3_low


class TestHierarchy:
    def test_hierarchy_open(self):
        from bpr.gauge_unification import HierarchyProblem
        h = HierarchyProblem(p=104729, N=10000)
        assert h.hierarchy_derived is False  # honest: open problem
        assert h.observed_ratio > 1e15

    def test_higgs_protected(self):
        from bpr.gauge_unification import HierarchyProblem
        h = HierarchyProblem()
        assert h.higgs_mass_protected is True


class TestProtonDecayGUT:
    def test_dominant_channel(self):
        from bpr.gauge_unification import ProtonDecay
        pd = ProtonDecay(p=104729)
        assert "π⁰" in pd.dominant_channel

    def test_exceeds_superK(self):
        from bpr.gauge_unification import ProtonDecay
        pd = ProtonDecay(p=104729)
        assert pd.exceeds_superK

    def test_branching_ratios_sum_reasonable(self):
        from bpr.gauge_unification import ProtonDecay
        pd = ProtonDecay()
        total = pd.branching_ratio_pi0_e + pd.branching_ratio_K_nu
        assert 0.5 < total < 1.0


class TestWeinbergAngle:
    def test_sin2tw_gut(self):
        from bpr.gauge_unification import weinberg_angle_from_boundary
        sin2tw = weinberg_angle_from_boundary("sphere")
        assert np.isclose(sin2tw, 3.0 / 8.0)


# =====================================================================
# Theory XVIII: Charged Leptons
# =====================================================================

class TestChargedLeptons:
    def test_three_masses(self):
        from bpr.charged_leptons import ChargedLeptonSpectrum
        lep = ChargedLeptonSpectrum()
        m = lep.all_masses_MeV
        assert len(m) == 3
        for name, val in m.items():
            assert val > 0

    def test_hierarchy(self):
        from bpr.charged_leptons import ChargedLeptonSpectrum
        lep = ChargedLeptonSpectrum()
        m = lep.all_masses_MeV
        assert m["tau"] > m["mu"] > m["e"]

    def test_electron_mass_order(self):
        from bpr.charged_leptons import ChargedLeptonSpectrum
        lep = ChargedLeptonSpectrum()
        m = lep.all_masses_MeV
        # Should be ~0.5 MeV
        assert 0.1 < m["e"] < 2.0

    def test_muon_mass_order(self):
        from bpr.charged_leptons import ChargedLeptonSpectrum
        lep = ChargedLeptonSpectrum()
        m = lep.all_masses_MeV
        assert 50 < m["mu"] < 200


class TestKoide:
    def test_koide_experimental(self):
        from bpr.charged_leptons import koide_parameter
        Q = koide_parameter()
        # Koide: Q ≈ 0.6667 (to 0.01% accuracy experimentally)
        assert abs(Q - 2.0 / 3.0) < 0.001

    def test_koide_predicted(self):
        from bpr.charged_leptons import koide_predicted
        assert koide_predicted() == 2.0 / 3.0


class TestLeptonUniversality:
    def test_universality_holds(self):
        from bpr.charged_leptons import LeptonUniversality
        lu = LeptonUniversality(p=104729)
        assert lu.universality_holds

    def test_R_K_near_one(self):
        from bpr.charged_leptons import LeptonUniversality
        lu = LeptonUniversality(p=104729)
        assert abs(lu.R_K_prediction - 1.0) < 0.001


# =====================================================================
# Theory XIX: Nuclear Physics
# =====================================================================

class TestMagicNumbers:
    def test_magic_numbers_correct(self):
        from bpr.nuclear_physics import magic_numbers_bpr
        magic = magic_numbers_bpr()
        assert magic == [2, 8, 20, 28, 50, 82, 126]

    def test_doubly_magic_208Pb(self):
        from bpr.nuclear_physics import is_magic
        result = is_magic(82, 126)
        assert result["doubly_magic"] is True

    def test_not_magic(self):
        from bpr.nuclear_physics import is_magic
        result = is_magic(15, 16)
        assert result["Z_magic"] is False


class TestBindingEnergy:
    def test_iron_peak(self):
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        # Fe-56: B/A ≈ 8.8 MeV (most bound)
        ba_fe = be.binding_energy_per_nucleon(56, 26)
        assert 7.5 < ba_fe < 9.5

    def test_helium_bound(self):
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        ba_he = be.binding_energy_per_nucleon(4, 2)
        assert ba_he > 0

    def test_heavy_less_bound_than_iron(self):
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        ba_fe = be.binding_energy_per_nucleon(56, 26)
        ba_u = be.binding_energy_per_nucleon(238, 92)
        assert ba_fe > ba_u


class TestNeutronStar:
    def test_max_mass(self):
        from bpr.nuclear_physics import NeutronStar
        ns = NeutronStar()
        assert 1.5 < ns.max_mass_solar < 3.0

    def test_radius(self):
        from bpr.nuclear_physics import NeutronStar
        ns = NeutronStar()
        assert 10 < ns.typical_radius_km < 15


class TestNuclearRadius:
    def test_radius_scaling(self):
        from bpr.nuclear_physics import nuclear_radius
        r12 = nuclear_radius(12)
        r208 = nuclear_radius(208)
        # R ∝ A^{1/3}
        ratio = r208 / r12
        expected = (208 / 12) ** (1.0 / 3.0)
        assert np.isclose(ratio, expected, rtol=0.01)


# =====================================================================
# Theory XX: Quantum Gravity Phenomenology
# =====================================================================

class TestModifiedDispersion:
    def test_xi1_zero(self):
        from bpr.quantum_gravity_pheno import ModifiedDispersion
        md = ModifiedDispersion()
        assert md.xi_1 == 0.0  # CPT protected

    def test_xi2_small(self):
        from bpr.quantum_gravity_pheno import ModifiedDispersion
        md = ModifiedDispersion(p=104729)
        assert 0 < md.xi_2 < 0.001

    def test_grb_delay_positive(self):
        from bpr.quantum_gravity_pheno import ModifiedDispersion
        md = ModifiedDispersion(p=104729)
        dt = md.grb_time_delay(1000.0, 1000.0)  # 1 TeV, 1 Gpc
        assert dt > 0

    def test_correction_tiny(self):
        from bpr.quantum_gravity_pheno import ModifiedDispersion
        md = ModifiedDispersion(p=104729)
        # 1 TeV photon
        corr = md.energy_correction(1000.0)
        assert corr < 1e-30


class TestGUP:
    def test_minimum_length_positive(self):
        from bpr.quantum_gravity_pheno import GeneralizedUncertainty
        gup = GeneralizedUncertainty(p=104729)
        assert gup.minimum_length > 0

    def test_minimum_length_below_lP(self):
        from bpr.quantum_gravity_pheno import GeneralizedUncertainty
        gup = GeneralizedUncertainty(p=104729)
        assert gup.minimum_length_over_lp < 1.0

    def test_beta_small(self):
        from bpr.quantum_gravity_pheno import GeneralizedUncertainty
        gup = GeneralizedUncertainty(p=104729)
        assert gup.beta < 1e-4


class TestLorentzInvariance:
    def test_within_bounds(self):
        from bpr.quantum_gravity_pheno import LorentzInvariance
        li = LorentzInvariance(p=104729)
        assert li.within_bounds

    def test_within_experimental_bounds(self):
        from bpr.quantum_gravity_pheno import LorentzInvariance
        li = LorentzInvariance(p=104729)
        # BPR prediction is just below Fermi-LAT bound: testable!
        assert li.orders_below_bound > 0


class TestHydrogenShift:
    def test_shift_tiny(self):
        from bpr.quantum_gravity_pheno import hydrogen_gravity_shift
        shift = hydrogen_gravity_shift(1)
        assert shift < 1e-40


# =====================================================================
# Theory XXI: Quantum Chemistry
# =====================================================================

class TestPeriodicTable:
    def test_noble_gases(self):
        from bpr.quantum_chemistry import noble_gas_numbers
        noble = noble_gas_numbers()
        assert noble == [2, 10, 18, 36, 54, 86, 118]

    def test_shell_capacity_s(self):
        from bpr.quantum_chemistry import shell_capacity
        assert shell_capacity(1, 0) == 2  # 1s

    def test_shell_capacity_p(self):
        from bpr.quantum_chemistry import shell_capacity
        assert shell_capacity(2, 1) == 6  # 2p

    def test_shell_capacity_d(self):
        from bpr.quantum_chemistry import shell_capacity
        assert shell_capacity(3, 2) == 10  # 3d

    def test_shell_capacity_f(self):
        from bpr.quantum_chemistry import shell_capacity
        assert shell_capacity(4, 3) == 14  # 4f


class TestChemicalBond:
    def test_bond_energy_positive(self):
        from bpr.quantum_chemistry import ChemicalBond
        bond = ChemicalBond(overlap=0.5, n_shared_modes=1)
        assert bond.bond_energy_eV > 0

    def test_double_stronger_than_single(self):
        from bpr.quantum_chemistry import ChemicalBond
        single = ChemicalBond(overlap=0.5, n_shared_modes=1)
        double = ChemicalBond(overlap=0.5, n_shared_modes=2)
        assert double.bond_energy_eV > single.bond_energy_eV

    def test_bond_type(self):
        from bpr.quantum_chemistry import ChemicalBond
        assert ChemicalBond(n_shared_modes=1).bond_type == "single"
        assert ChemicalBond(n_shared_modes=2).bond_type == "double"
        assert ChemicalBond(n_shared_modes=3).bond_type == "triple"


class TestChirality:
    def test_odd_winding_chiral(self):
        from bpr.quantum_chemistry import MolecularChirality
        mc = MolecularChirality(W_molecular=1)
        assert mc.is_chiral

    def test_even_winding_achiral(self):
        from bpr.quantum_chemistry import MolecularChirality
        mc = MolecularChirality(W_molecular=2)
        assert not mc.is_chiral


class TestHydrogenLevels:
    def test_ground_state(self):
        from bpr.quantum_chemistry import hydrogen_energy_levels
        levels = hydrogen_energy_levels(5)
        assert np.isclose(levels[0], -13.6, rtol=0.01)

    def test_levels_increase(self):
        from bpr.quantum_chemistry import hydrogen_energy_levels
        levels = hydrogen_energy_levels(5)
        # Energy should increase (become less negative) with n
        for i in range(len(levels) - 1):
            assert levels[i + 1] > levels[i]


class TestMadelung:
    def test_first_shell_1s(self):
        from bpr.quantum_chemistry import madelung_filling_order
        order = madelung_filling_order()
        assert order[0] == (1, 0)  # 1s first

    def test_cumulative_reaches_noble(self):
        from bpr.quantum_chemistry import cumulative_electrons
        shells = cumulative_electrons()
        cumulatives = [s["cumulative"] for s in shells]
        # Noble gas numbers should appear
        assert 2 in cumulatives
        assert 10 in cumulatives


# =====================================================================
# Full pipeline: all 200 predictions
# =====================================================================

class TestFullPipelineV06:
    def test_prediction_count(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert len(preds) >= 195, f"Expected 195+, got {len(preds)}"

    def test_all_new_theories_present(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        for prefix in ["P17.", "P18.", "P19.", "P20.", "P21."]:
            keys = [k for k in preds if k.startswith(prefix)]
            assert len(keys) >= 5, f"Expected ≥5 for {prefix}, got {len(keys)}"

    def test_gut_scale(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert 1e14 < preds["P17.1_GUT_scale_GeV"] < 1e18

    def test_magic_numbers(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert preds["P19.5_magic_numbers"] == [2, 8, 20, 28, 50, 82, 126]

    def test_lorentz_invariance(self):
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        assert preds["P20.8_LI_within_bounds"]
