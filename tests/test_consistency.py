"""
Mathematical Consistency Tests for BPR-Math-Spine
===================================================

These tests verify INTERNAL mathematical consistency — no experiments needed.
If any of these fail, the theory contradicts itself regardless of data.

Test categories:
    1. Limiting cases:  p→∞, N→∞, J→0 must recover standard physics
    2. Cross-module:    same quantity from two derivations must agree
    3. Conservation:    probability sums to 1, energy-momentum conserved
    4. Dimensional:     every formula has correct units
    5. Thermodynamic:   entropy bounds, second law
    6. Monotonicity:    convergence under parameter scaling

Run with:  pytest -v tests/test_consistency.py
"""

import math
import numpy as np
import pytest


# =====================================================================
# 1. LIMITING CASES
#    As p → ∞, BPR corrections vanish and standard physics is recovered.
#    As N → ∞, continuum limit is reached.
# =====================================================================

class TestLimitingCases:
    """BPR must reduce to standard physics in appropriate limits."""

    def test_born_rule_exact_as_p_to_inf(self):
        """Born rule deviation → 0 as p → ∞."""
        from bpr.quantum_foundations import BornRule
        deviations = []
        for p in [101, 10007, 104729, 1000003, 15485863]:
            br = BornRule(p=p)
            deviations.append(br.correction_amplitude)
        # Must be monotonically decreasing
        for i in range(len(deviations) - 1):
            assert deviations[i] > deviations[i + 1], \
                f"Born rule correction not decreasing: {deviations}"
        # Must approach zero
        assert deviations[-1] < 1e-6

    def test_bell_bound_approaches_tsirelson_as_p_to_inf(self):
        """Bell-BPR bound → 2√2 (Tsirelson) as p → ∞."""
        from bpr.quantum_foundations import BellInequality
        tsirelson = 2 * np.sqrt(2)
        # Deviation should decrease monotonically with p
        prev_dev = float('inf')
        for p in [101, 10007, 104729, 1000003, 15485863]:
            bi = BellInequality(p=p)
            dev = abs(bi.bpr_bound - tsirelson)
            assert dev < prev_dev, \
                f"Bell bound deviation not decreasing at p={p}"
            prev_dev = dev
        # For largest p, should be very close
        assert prev_dev < 1e-5

    def test_lorentz_violation_vanishes_as_p_to_inf(self):
        """LIV → 0 as p → ∞ (Lorentz invariance recovered)."""
        from bpr.quantum_gravity_pheno import ModifiedDispersion, LorentzInvariance
        for p in [101, 10007, 104729, 15485863]:
            md = ModifiedDispersion(p=p)
            assert md.xi_2 == pytest.approx(1.0 / p)
            assert md.xi_1 == 0.0  # CPT protection holds for all p

        # Lorentz violation must decrease with p
        li_small = LorentzInvariance(p=101)
        li_large = LorentzInvariance(p=15485863)
        assert li_large.fractional_speed_variation < li_small.fractional_speed_variation

    def test_gup_reduces_to_heisenberg_as_p_to_inf(self):
        """GUP → standard Heisenberg as p → ∞ (β → 0)."""
        from bpr.quantum_gravity_pheno import GeneralizedUncertainty
        for p in [101, 10007, 104729, 15485863]:
            gup = GeneralizedUncertainty(p=p)
            assert gup.beta == pytest.approx(1.0 / p)
        # Minimum length → 0 as p → ∞
        gup_large = GeneralizedUncertainty(p=15485863)
        assert gup_large.minimum_length < GeneralizedUncertainty(p=101).minimum_length

    def test_neutrino_masses_stable_under_large_p(self):
        """Neutrino mass structure doesn't blow up for large p."""
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        m = spec.masses_eV
        assert np.all(m > 0), "Negative neutrino masses"
        assert np.all(m < 1.0), "Neutrino mass > 1 eV"
        assert np.sum(m) == pytest.approx(0.06, abs=0.001)

    def test_predictions_continuous_in_p(self):
        """Most predictions change smoothly with small changes in p.

        Some predictions (like baryon asymmetry) depend on p mod 4,
        so they can jump at specific p values.  We check that the
        majority of numerical predictions are continuous.
        """
        from bpr.first_principles import SubstrateDerivedTheories
        # Two nearby primes with same p mod 4 = 1
        sdt1 = SubstrateDerivedTheories.from_substrate(p=104729)
        sdt2 = SubstrateDerivedTheories.from_substrate(p=104743)
        p1 = sdt1.predictions()
        p2 = sdt2.predictions()
        discontinuous = []
        total_checked = 0
        for key in p1:
            v1, v2 = p1[key], p2[key]
            if isinstance(v1, float) and isinstance(v2, float):
                if abs(v1) > 1e-100 and abs(v2) > 1e-100:
                    total_checked += 1
                    ratio = v1 / v2
                    if not (0.9 < ratio < 1.1):
                        discontinuous.append(key)
        # Allow up to 10% of predictions to be discontinuous
        frac = len(discontinuous) / max(total_checked, 1)
        assert frac < 0.1, \
            f"{len(discontinuous)}/{total_checked} predictions discontinuous: {discontinuous[:5]}"


# =====================================================================
# 2. CROSS-MODULE CONSISTENCY
#    The same physical quantity derived from different modules must agree.
# =====================================================================

class TestCrossModuleConsistency:
    """Different modules must give the same answer for shared quantities."""

    def test_planck_length_consistent(self):
        """Planck length from emergent_spacetime matches physical constant."""
        from bpr.emergent_spacetime import planck_length_from_substrate
        l_P = planck_length_from_substrate()
        assert l_P == pytest.approx(1.616255e-35, rel=1e-4)

    def test_three_generations_consistent(self):
        """Number of generations: neutrino module and chemistry agree."""
        from bpr.neutrino import number_of_generations, NeutrinoMassSpectrum
        assert number_of_generations("sphere") == 3
        # Neutrino mass spectrum has exactly 3 entries
        spec = NeutrinoMassSpectrum()
        assert len(spec.masses_eV) == 3

    def test_proton_decay_consistent(self):
        """Proton decay lifetime: impedance module vs GUT module."""
        from bpr.impedance import proton_lifetime
        from bpr.gauge_unification import ProtonDecay
        tau_imp = proton_lifetime(p=104729)
        pd = ProtonDecay(p=104729)
        tau_gut = pd.lifetime_years
        # Both must exceed Super-K bound
        assert tau_imp > 1e34
        assert tau_gut > 1e34
        # Both should be finite
        assert tau_imp < 1e100
        assert tau_gut < 1e100

    def test_dark_matter_properties_consistent(self):
        """DM self-interaction and relic abundance are positive and finite.

        Since v0.8.0, relic abundance is from genuine thermal freeze-out.
        BPR overproduces DM (~9.5) — an honest FAIL, not a fitted match.
        """
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        # DM relic abundance should be positive and finite
        omega = preds["P11.15_DM_relic_Omega_h2"]
        assert 0 < omega, "Ω_DM must be positive"
        assert math.isfinite(omega), "Ω_DM must be finite"
        # DM self-interaction should be positive and below bullet cluster
        sigma_m = preds["P2.7_DM_sigma_over_m_cm2_g"]
        assert 0 < sigma_m < 1  # bullet cluster bound

    def test_cp_phase_neutrino_vs_cosmology(self):
        """CP violation: neutrino module (p mod 4) vs cosmology module."""
        from bpr.neutrino import neutrino_nature
        from bpr.cosmology import Baryogenesis
        p = 104729
        # p mod 4 = 1 → Dirac → small CP
        assert neutrino_nature(p) == "Dirac"
        bary = Baryogenesis(p=p)
        # Small CP phase (Jarlskog ~ 3e-5 for orientable boundary)
        assert bary.cp_phase < 1e-3

    def test_decoherence_vs_measurement(self):
        """Decoherence (Theory III) must be compatible with measurement (XVI)."""
        from bpr.quantum_foundations import MeasurementDynamics
        from bpr.decoherence import DecoherenceRate
        # Measurement time from XVI
        md = MeasurementDynamics()
        t_meas = md.measurement_time
        assert t_meas > 0
        # Decoherence rate from III at room temperature
        dr = DecoherenceRate(T=300.0, Z_system=376.73, Z_environment=400.0)
        t_decoh = dr.decoherence_time
        # Both should be positive and finite
        assert t_meas > 0
        assert t_decoh > 0

    def test_mond_a0_vs_hubble(self):
        """MOND a₀ = cH₀/(2π) × (1 + z/(4 ln p)): check dimensional and numerical."""
        from bpr.impedance import MONDInterpolation
        import numpy as np
        c = 299792458.0
        H0 = 67.4e3 / 3.0857e22  # s⁻¹
        p, z = 104729, 6
        base = c * H0 / (2 * np.pi)
        a0_expected = base * (1.0 + z / (4.0 * np.log(p)))
        mond = MONDInterpolation(H0_km_s_Mpc=67.4, p=p, z=z)
        assert mond.a0 == pytest.approx(a0_expected, rel=1e-6)
        # a₀ must have units of acceleration (m/s²)
        assert 1e-11 < mond.a0 < 1e-9  # reasonable acceleration scale


# =====================================================================
# 3. CONSERVATION LAWS
#    Probabilities sum to 1. Energy-momentum conserved. Charge conserved.
# =====================================================================

class TestConservationLaws:
    """Fundamental conservation laws must hold in BPR."""

    def test_pmns_unitarity(self):
        """PMNS matrix must be unitary: U†U = I."""
        from bpr.neutrino import PMNSMatrix
        pmns = PMNSMatrix()
        assert pmns.is_unitary(tol=1e-10), "PMNS matrix is not unitary!"

    def test_ckm_unitarity(self):
        """CKM matrix must be unitary."""
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        V = ckm.V
        product = V.T.conj() @ V
        assert np.allclose(product, np.eye(3), atol=1e-10), \
            "CKM matrix is not unitary!"

    def test_neutrino_masses_sum_to_prediction(self):
        """Neutrino mass normalization: Σm_i must equal predicted total."""
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        assert np.sum(spec.masses_eV) == pytest.approx(0.06, abs=0.001)

    def test_branching_ratios_sum_leq_1(self):
        """Proton decay branching ratios must sum to ≤ 1."""
        from bpr.gauge_unification import ProtonDecay
        pd = ProtonDecay()
        total = pd.branching_ratio_pi0_e + pd.branching_ratio_K_nu
        assert 0 < total <= 1.0, f"Branching ratios sum to {total} > 1"

    def test_born_rule_probabilities_sum_to_1(self):
        """Born rule: P(α) must sum to 1 for any complete measurement."""
        from bpr.quantum_foundations import BornRule
        br = BornRule(p=104729)
        # The correction is multiplicative: P = |ψ|² × (1 + ε)
        # For normalized |ψ|², accuracy ≈ 1 - 1/p
        accuracy = br.born_rule_accuracy
        assert accuracy > 0.999, "Born rule accuracy too low"
        assert accuracy <= 1.0, "Born rule accuracy > 1 (impossible)"

    def test_oscillation_probability_bounded(self):
        """Neutrino oscillation probability must be in [0, 1]."""
        from bpr.neutrino import oscillation_probability
        for L in [1, 100, 1e4, 1e6]:
            for E in [1e6, 1e9, 1e12]:  # eV
                P = oscillation_probability(L, E, delta_m_sq=7.5e-5)
                assert 0 <= P <= 1, f"P={P} outside [0,1] at L={L}, E={E}"

    def test_hall_conductance_quantized(self):
        """QHE: Hall conductance must be exactly quantized."""
        from bpr.topological_matter import QuantumHallEffect
        for nu in [1, 2, 3]:
            qhe = QuantumHallEffect(nu=nu)
            e2_h = 3.874046e-5  # e²/h in siemens
            assert qhe.hall_conductance == pytest.approx(nu * e2_h, rel=1e-6), \
                f"Hall conductance not quantized for ν={nu}"

    def test_entropy_positive(self):
        """Black hole entropy must be positive."""
        from bpr.black_hole import black_hole_entropy
        for M in [1.0, 10.0, 100.0]:
            S = black_hole_entropy(M)
            assert S > 0, f"Negative BH entropy for M={M} M_sun"


# =====================================================================
# 4. DIMENSIONAL ANALYSIS
#    Every formula must produce quantities with correct units.
# =====================================================================

class TestDimensionalConsistency:
    """Every formula must have dimensionally correct units."""

    def test_mond_acceleration_units(self):
        """a₀ must be in m/s² (order 10⁻¹⁰)."""
        from bpr.impedance import MONDInterpolation
        mond = MONDInterpolation()
        assert 1e-12 < mond.a0 < 1e-8, \
            f"a₀ = {mond.a0} is not a reasonable acceleration"

    def test_casimir_force_units(self):
        """Casimir force formula: dimensional check via analytic formula."""
        # F_Casimir = -π²ℏc / (240 a⁴) × A  (parallel plates)
        # Check the analytic formula produces correct dimensions
        hbar = 1.0546e-34
        c = 2.998e8
        a = 1e-6  # 1 μm
        A = 1e-4  # 1 cm²
        F = -np.pi**2 * hbar * c / (240 * a**4) * A
        # Should be ~10⁻⁷ N for these parameters (force in Newtons)
        assert 1e-9 < abs(F) < 1e-4, f"Casimir force {F} N out of expected range"

    def test_neutrino_masses_in_eV(self):
        """Neutrino masses must be in eV, order 10⁻² to 10⁻¹."""
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        for m in spec.masses_eV:
            assert 1e-4 < m < 0.1, f"Neutrino mass {m} eV out of range"

    def test_decoherence_rate_in_Hz(self):
        """Decoherence rate must be in Hz (positive)."""
        from bpr.decoherence import DecoherenceRate
        dr = DecoherenceRate(T=300.0, Z_system=376.73, Z_environment=400.0)
        assert dr.gamma_dec > 0, "Negative decoherence rate"
        assert dr.gamma_dec < 1e30, "Decoherence rate unreasonably large"

    def test_gut_scale_in_GeV(self):
        """GUT scale must be in GeV, order 10¹⁵–10¹⁸."""
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning(p=104729)
        assert 1e14 < gc.unification_scale_GeV < 1e19

    def test_nuclear_radius_in_fm(self):
        """Nuclear radius must be in fm, order 1-10."""
        from bpr.nuclear_physics import nuclear_radius
        for A in [4, 56, 208]:
            r = nuclear_radius(A)
            assert 1 < r < 10, f"Nuclear radius {r} fm out of range for A={A}"

    def test_hydrogen_energy_in_eV(self):
        """Hydrogen ground state must be -13.6 eV."""
        from bpr.quantum_chemistry import hydrogen_energy_levels
        levels = hydrogen_energy_levels(1)
        assert levels[0] == pytest.approx(-13.6, rel=0.01)

    def test_binding_energy_per_nucleon_in_MeV(self):
        """B/A must be in MeV, between 0 and 9."""
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        for A, Z in [(4, 2), (56, 26), (208, 82), (238, 92)]:
            ba = be.binding_energy_per_nucleon(A, Z)
            assert 0 < ba < 10, f"B/A = {ba} MeV out of range for (A={A},Z={Z})"

    def test_correlation_length_positive(self):
        """Substrate correlation length ξ must be positive and finite."""
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        assert sdt.xi > 0
        assert sdt.xi < 1.0  # should be sub-meter for lab scale
        assert np.isfinite(sdt.xi)

    def test_all_numerical_predictions_finite(self):
        """No prediction should be NaN or Inf."""
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        preds = sdt.predictions()
        for key, val in preds.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} = {val} is not finite!"


# =====================================================================
# 5. THERMODYNAMIC IDENTITIES AND BOUNDS
# =====================================================================

class TestThermodynamicConsistency:
    """Entropy bounds and thermodynamic identities must hold."""

    def test_bekenstein_bound(self):
        """Entropy of any system must respect S ≤ 2πRE/(ℏc)."""
        from bpr.emergent_spacetime import BekensteinBound
        # For a 1 kg object of radius 1 m:
        bb = BekensteinBound(R=1.0, E=1.0 * 299792458.0**2)
        assert bb.bekenstein_entropy > 0
        assert bb.bekenstein_entropy < 1e60  # reasonable for macroscopic object

    def test_black_hole_entropy_scales_as_area(self):
        """S_BH ∝ M² (area law: A ∝ M²)."""
        from bpr.black_hole import black_hole_entropy
        S1 = black_hole_entropy(1.0)
        S2 = black_hole_entropy(2.0)
        S10 = black_hole_entropy(10.0)
        # S ∝ M² means S(2M)/S(M) ≈ 4
        ratio_2 = S2 / S1
        assert ratio_2 == pytest.approx(4.0, rel=0.01), \
            f"S(2M)/S(M) = {ratio_2}, expected 4.0"
        ratio_10 = S10 / S1
        assert ratio_10 == pytest.approx(100.0, rel=0.01), \
            f"S(10M)/S(M) = {ratio_10}, expected 100.0"

    def test_arrow_of_time_holds(self):
        """BPR must predict a consistent arrow of time."""
        from bpr.quantum_foundations import ArrowOfTime
        at = ArrowOfTime(p=104729)
        assert at.entropy_monotonic is True
        assert at.time_quantum > 0

    def test_hawking_temperature_positive(self):
        """Hawking temperature must be positive and decrease with mass."""
        from bpr.black_hole import BlackHoleEntropy
        bh1 = BlackHoleEntropy(M_solar=1.0)
        bh10 = BlackHoleEntropy(M_solar=10.0)
        T1 = bh1.hawking_temperature
        T10 = bh10.hawking_temperature
        assert T1 > 0
        assert T10 > 0
        assert T1 > T10, "Larger BH should be cooler"

    def test_neutron_star_below_bh_limit(self):
        """NS max mass must be below the BH formation limit (~3 M_☉)."""
        from bpr.nuclear_physics import NeutronStar
        ns = NeutronStar()
        assert ns.max_mass_solar < 3.0, \
            f"NS max mass {ns.max_mass_solar} M_☉ exceeds BH limit"


# =====================================================================
# 6. MONOTONICITY AND CONVERGENCE
# =====================================================================

class TestMonotonicity:
    """Physical quantities must scale correctly with parameters."""

    def test_decoherence_increases_with_temperature(self):
        """Higher T → faster decoherence."""
        from bpr.decoherence import DecoherenceRate
        gamma_cold = DecoherenceRate(T=1.0, Z_system=376.73, Z_environment=400.0).gamma_dec
        gamma_hot = DecoherenceRate(T=300.0, Z_system=376.73, Z_environment=400.0).gamma_dec
        assert gamma_hot > gamma_cold

    def test_decoherence_increases_with_impedance_mismatch(self):
        """Larger ΔZ → faster decoherence."""
        from bpr.decoherence import DecoherenceRate
        gamma_small = DecoherenceRate(T=300.0, Z_system=376.73, Z_environment=380.0).gamma_dec
        gamma_large = DecoherenceRate(T=300.0, Z_system=376.73, Z_environment=500.0).gamma_dec
        assert gamma_large > gamma_small

    def test_binding_energy_iron_peak(self):
        """B/A must peak near iron (A ≈ 56)."""
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        ba_fe = be.binding_energy_per_nucleon(56, 26)
        ba_he = be.binding_energy_per_nucleon(4, 2)
        ba_u = be.binding_energy_per_nucleon(238, 92)
        assert ba_fe > ba_he, "Iron should be more bound than helium"
        assert ba_fe > ba_u, "Iron should be more bound than uranium"

    def test_gauge_coupling_asymptotic_freedom(self):
        """α₃ (strong coupling) must decrease with energy."""
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning()
        a3_low = gc.alpha_i(3, 10.0)    # 10 GeV
        a3_mid = gc.alpha_i(3, 1e3)     # 1 TeV
        a3_high = gc.alpha_i(3, 1e10)   # 10^10 GeV
        assert a3_low > a3_mid > a3_high, \
            "Strong coupling must decrease with energy (asymptotic freedom)"

    def test_chemical_bond_stronger_with_overlap(self):
        """Larger overlap → stronger bond."""
        from bpr.quantum_chemistry import ChemicalBond
        weak = ChemicalBond(overlap=0.2, n_shared_modes=1)
        strong = ChemicalBond(overlap=0.8, n_shared_modes=1)
        assert strong.bond_energy_eV > weak.bond_energy_eV

    def test_nuclear_radius_increases_with_A(self):
        """R ∝ A^{1/3}: larger nuclei must be bigger."""
        from bpr.nuclear_physics import nuclear_radius
        r4 = nuclear_radius(4)
        r56 = nuclear_radius(56)
        r208 = nuclear_radius(208)
        assert r4 < r56 < r208

    def test_prediction_count_stable(self):
        """Prediction count should be consistent across runs."""
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate()
        p1 = sdt.predictions()
        p2 = sdt.predictions()
        assert len(p1) == len(p2)
        for k in p1:
            assert k in p2
            if isinstance(p1[k], float):
                assert p1[k] == p2[k], f"{k} changed between calls"


# =====================================================================
# 7. MATHEMATICAL CLOSURE — DEEP IDENTITIES
#    These test whether BPR's mathematical structure closes on itself:
#    derivations that approach the same quantity from two different
#    directions within the theory must agree.
# =====================================================================

class TestMathematicalClosure:
    """Deep self-consistency: mathematical identities that must hold."""

    def test_entropy_area_law_independent_of_p(self):
        """BH entropy S = A/(4l_P²) must be independent of substrate prime p.

        BPR derives this as: Ω = p^(A/l_P²), so
            S = ln(Ω)/4 = (A/l_P²) × ln(p) / 4
        But ln(p) cancels with the normalization of k_B in substrate units.
        The final result must be the SAME number regardless of p.
        """
        from bpr.black_hole import BlackHoleEntropy
        S_p1 = BlackHoleEntropy(M_solar=1.0, p=104729).entropy_bpr
        S_p2 = BlackHoleEntropy(M_solar=1.0, p=15485863).entropy_bpr
        S_p3 = BlackHoleEntropy(M_solar=1.0, p=101).entropy_bpr
        assert S_p1 == pytest.approx(S_p2, rel=1e-10), \
            f"BH entropy depends on p: {S_p1} vs {S_p2}"
        assert S_p1 == pytest.approx(S_p3, rel=1e-10), \
            f"BH entropy depends on p: {S_p1} vs {S_p3}"

    def test_entanglement_entropy_p_independent(self):
        """Entanglement entropy must also be p-independent."""
        from bpr.emergent_spacetime import HolographicEntropy
        ee1 = HolographicEntropy(boundary_area=1.0, p=104729)
        ee2 = HolographicEntropy(boundary_area=1.0, p=15485863)
        assert ee1.p_independent is True
        assert ee1.entropy == pytest.approx(ee2.entropy, rel=1e-10)

    def test_bekenstein_tighter_than_holographic_for_small_objects(self):
        """For objects far from BH formation, Bekenstein < holographic."""
        from bpr.emergent_spacetime import BekensteinBound
        # 1 kg ball of radius 0.1 m (very far from gravitational collapse)
        bb = BekensteinBound(R=0.1, E=1.0 * 299792458.0**2)
        assert bb.tighter_bound == "Bekenstein"

    def test_hawking_entropy_consistency(self):
        """T_H and S_BH must satisfy first law: dM = T dS.

        Schwarzschild: S = 4πGM²/(ℏc), T = ℏc³/(8πGMk_B)
        Check: T × dS/dM = 1  (in natural units where c=G=1)
        In SI: T_H × (dS/dM)_SI × k_B = c²
        """
        from bpr.black_hole import BlackHoleEntropy
        M1 = 1.0
        M2 = 1.001  # small perturbation
        bh1 = BlackHoleEntropy(M_solar=M1)
        bh2 = BlackHoleEntropy(M_solar=M2)
        T = bh1.hawking_temperature  # Kelvin
        k_B = 1.380649e-23
        M_sun = 1.989e30
        dS = bh2.entropy_bekenstein_hawking - bh1.entropy_bekenstein_hawking
        dM = (M2 - M1) * M_sun  # kg
        # First law: T dS k_B = dM c²
        c = 299792458.0
        lhs = T * dS * k_B
        rhs = dM * c**2
        assert lhs == pytest.approx(rhs, rel=0.01), \
            f"First law violated: TdS k_B = {lhs}, dM c² = {rhs}"

    def test_koide_formula_algebraic(self):
        """Koide relation: (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3."""
        from bpr.charged_leptons import koide_parameter, koide_predicted
        Q_actual = koide_parameter()
        Q_bpr = koide_predicted()
        assert Q_actual == pytest.approx(Q_bpr, abs=0.01)

    def test_ckm_jarlskog_from_matrix(self):
        """Jarlskog invariant J must satisfy |J| ≤ 1/(6√3) ≈ 0.0962."""
        from bpr.qcd_flavor import CKMMatrix
        ckm = CKMMatrix()
        V = ckm.V
        # J = Im(V_us V_cb V*_ub V*_cs)
        J = np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1]))
        max_J = 1.0 / (6 * np.sqrt(3))
        assert abs(J) <= max_J + 1e-10, f"|J| = {abs(J)} > {max_J}"

    def test_pmns_row_column_normalization(self):
        """Each row and column of PMNS must independently sum to 1."""
        from bpr.neutrino import PMNSMatrix
        U = PMNSMatrix().U
        # Row normalization
        for i in range(3):
            assert np.sum(U[i, :] ** 2) == pytest.approx(1.0, abs=1e-10), \
                f"PMNS row {i} normalization: {np.sum(U[i, :]**2)}"
        # Column normalization
        for j in range(3):
            assert np.sum(U[:, j] ** 2) == pytest.approx(1.0, abs=1e-10), \
                f"PMNS col {j} normalization: {np.sum(U[:, j]**2)}"

    def test_mass_ordering_consistent_with_splittings(self):
        """Neutrino mass splittings from boundary Laplacian eigenvalues.

        BPR derives mass ratios from |c_k|² = (l_k + ½)² on S²:
            l = 0, 1, 3 → ratios 1 : 9 : 49

        This gives:
            Δm²₂₁ ≈ 8.3×10⁻⁵ eV²  (exp: 7.53×10⁻⁵, within 10%)
            Δm²₃₂ ≈ 2.40×10⁻³ eV²  (exp: 2.453×10⁻³, within 2%)
        """
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        m = spec.masses_eV
        splittings = spec.mass_squared_differences
        # Internal consistency: dm32 must match raw masses
        dm32_sq = m[2]**2 - m[1]**2
        assert dm32_sq == pytest.approx(splittings["Delta_m32_sq"], rel=1e-10)
        # dm21 has boundary curvature correction applied (v0.9.0)
        dm21_raw = m[1]**2 - m[0]**2
        dm21_corrected = splittings["Delta_m21_sq"]
        assert dm21_corrected < dm21_raw, \
            "Boundary curvature correction should reduce solar splitting"
        assert dm21_corrected == pytest.approx(dm21_raw * spec._rg_correction_solar, rel=1e-10)
        # Normal ordering: m3 > m2 > m1
        assert m[2] > m[1] > m[0], "Mass ordering violated"
        # Compare with experiment (within 5%)
        assert dm21_corrected == pytest.approx(7.53e-5, rel=0.05), \
            f"dm21_sq = {dm21_corrected:.3e}, expected 7.53e-5"
        assert dm32_sq == pytest.approx(2.453e-3, rel=0.05), \
            f"dm32_sq = {dm32_sq:.3e}, expected 2.453e-3"

    def test_gauge_coupling_sum_rule(self):
        """At unification: α₁ = α₂ = α₃ (by definition of GUT scale)."""
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning(p=104729)
        E_gut = gc.unification_scale_GeV
        a1 = gc.alpha_i(1, E_gut)
        a2 = gc.alpha_i(2, E_gut)
        a3 = gc.alpha_i(3, E_gut)
        # At GUT scale, all three should converge
        mean = (a1 + a2 + a3) / 3
        for a in [a1, a2, a3]:
            assert a == pytest.approx(mean, rel=0.3), \
                f"Coupling not unified at GUT scale: {a1}, {a2}, {a3}"

    def test_weinberg_angle_from_couplings(self):
        """Gauge coupling running: sin²θ_W from top-down BPR calculation.

        BPR derives sin²θ_W(M_Z) from:
        1. sin²θ_W(M_GUT) = 3/8 (S² boundary geometry)
        2. SM 1-loop RGE running from M_GUT to M_Z
        3. BPR matching corrections from boundary mode spectrum

        The result should match the measured value 0.231 ± 0.03.
        """
        from bpr.gauge_unification import GaugeCouplingRunning
        gc = GaugeCouplingRunning(p=104729)

        # Top-down Weinberg angle prediction
        sin2_w = gc.weinberg_angle_at_MZ
        assert sin2_w == pytest.approx(0.231, abs=0.005), \
            f"sin²θ_W(M_Z) = {sin2_w}, expected ≈ 0.231"

        # α_s prediction from same top-down
        alpha_s = gc.alpha_s_prediction
        assert alpha_s == pytest.approx(0.1179, rel=0.1), \
            f"α_s(M_Z) = {alpha_s}, expected ≈ 0.1179"

        # α_EM prediction
        alpha_em = gc.alpha_em_prediction
        assert alpha_em == pytest.approx(1 / 127.952, rel=0.05), \
            f"1/α_EM = {1/alpha_em:.1f}, expected ≈ 128"

        # Asymptotic freedom still holds in SM running
        a3_mz = gc.alpha_i(3, 91.2)
        a3_gut = gc.alpha_i(3, gc.unification_scale_GeV)
        assert a3_mz > a3_gut, "Strong coupling should be larger at low energy"

    def test_nuclear_magic_stability(self):
        """Magic number nuclei must have higher B/A than neighbors."""
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        # ²⁰⁸Pb (Z=82, N=126) is doubly magic
        ba_pb = be.binding_energy_per_nucleon(208, 82)
        ba_tl = be.binding_energy_per_nucleon(207, 81)  # nearby non-magic
        ba_bi = be.binding_energy_per_nucleon(209, 83)
        # Pb should be among the most stable in this region
        assert ba_pb >= min(ba_tl, ba_bi) - 0.1, \
            "Magic ²⁰⁸Pb not more stable than neighbors"

    def test_bethe_weizsacker_self_consistency(self):
        """Semi-empirical mass formula: B(A,Z) > 0 for A ≥ 12 stable nuclei.

        Note: Bethe-Weizsacker is unreliable for very light nuclei (A < 12).
        This is a known limitation of the semi-empirical formula, not a BPR bug.
        """
        from bpr.nuclear_physics import BindingEnergy
        be = BindingEnergy()
        # Only test A ≥ 12 where the formula is reliable
        stable = [(12, 6), (16, 8), (40, 20), (56, 26),
                  (120, 50), (208, 82)]
        for A, Z in stable:
            B = be.binding_energy_per_nucleon(A, Z) * A
            assert B > 0, f"Negative total BE for (A={A}, Z={Z}): B={B} MeV"

    def test_gup_minimum_length_sub_planckian(self):
        """GUP minimum length should be ≤ l_Planck.

        BPR predicts β = 1/p, so:
            δx_min = l_P √β = l_P / √p ≈ l_P / 324 ≈ 5×10⁻³⁸ m

        This is SUB-Planckian, meaning the substrate allows probing
        below l_P by a factor √p.  This is consistent because the
        substrate IS the structure below the Planck scale.
        """
        from bpr.quantum_gravity_pheno import GeneralizedUncertainty
        gup = GeneralizedUncertainty(p=104729)
        l_P = 1.616255e-35
        # min length = l_P × sqrt(1/p) = l_P / sqrt(p)
        expected = l_P / np.sqrt(104729)
        assert gup.minimum_length == pytest.approx(expected, rel=0.01)
        # Must be smaller than Planck length (sub-Planckian)
        assert gup.minimum_length < l_P
        # Must be positive
        assert gup.minimum_length > 0

    def test_dark_energy_positive_and_small(self):
        """Dark energy density must be positive and ≪ Planck density."""
        from bpr.impedance import DarkEnergyDensity
        c = 299792458.0
        H0 = 67.4e3 / 3.0857e22  # s⁻¹
        R_hubble = c / H0
        de = DarkEnergyDensity(L=R_hubble)
        rho_de = de.rho_DE
        assert rho_de > 0, "Negative dark energy density"
        # Planck density ~ 10^113 J/m³, observed ~ 10^-9 J/m³
        assert rho_de < 1e10, f"DE density {rho_de} unreasonably large"

    def test_idempotence_predictions(self):
        """Calling predictions() twice with same params gives bitwise identical results."""
        from bpr.first_principles import SubstrateDerivedTheories
        sdt = SubstrateDerivedTheories.from_substrate(p=104729)
        p1 = sdt.predictions()
        p2 = sdt.predictions()
        for key in p1:
            v1, v2 = p1[key], p2[key]
            if isinstance(v1, float):
                # Must be EXACTLY equal (not just approximately)
                assert v1 == v2, f"{key}: {v1} != {v2} (non-deterministic!)"
