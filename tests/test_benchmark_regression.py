"""
Benchmark Regression Tests
===========================

These tests compare key BPR predictions against published experimental
measurements.  They serve as regression guards: if a code change causes
a prediction to drift away from experiment, these tests will catch it.

Unlike the consistency tests (test_consistency.py) which check internal
mathematical coherence, these tests check external accuracy.

Each test documents:
    - The BPR prediction and how it's computed
    - The experimental measurement with citation
    - The acceptable tolerance (typically generous)

Run with:  pytest -v tests/test_benchmark_regression.py
"""

import math
import numpy as np
import pytest


# =====================================================================
# Helper: generate predictions once for the module
# =====================================================================

@pytest.fixture(scope="module")
def predictions():
    """Generate the full prediction set once for all tests."""
    from bpr.first_principles import SubstrateDerivedTheories
    sdt = SubstrateDerivedTheories.from_substrate(
        p=104729, N=10000, J_eV=1.0, radius=0.01, geometry="sphere",
    )
    return sdt.predictions()


# =====================================================================
# 1. DERIVED PREDICTIONS (genuine BPR, highest priority)
#    These are the real tests of BPR's predictive power.
# =====================================================================

class TestDerivedPredictions:
    """Predictions derived from (J, p, N) with no hand-tuning."""

    def test_theta13_neutrino(self, predictions):
        """Neutrino mixing angle θ₁₃ from S² boundary geometry.

        BPR:  8.63°
        PDG:  8.54° ± 0.15°
        """
        theta13 = predictions["P5.7_theta13_deg"]
        assert theta13 == pytest.approx(8.54, abs=0.5), \
            f"θ₁₃ = {theta13}°, expected 8.54° ± 0.5°"

    def test_delta_m32_sq(self, predictions):
        """Atmospheric mass splitting |Δm²₃₂| from boundary Laplacian.

        BPR:  2.399 × 10⁻³ eV²
        PDG:  2.453 × 10⁻³ ± 0.033 × 10⁻³ eV²
        """
        dm32 = predictions["P5.9_delta_m32_sq_eV2"]
        assert dm32 == pytest.approx(2.453e-3, rel=0.05), \
            f"Δm²₃₂ = {dm32:.4e}, expected ≈ 2.453e-3"

    def test_delta_m21_sq(self, predictions):
        """Solar mass splitting Δm²₂₁ from boundary Laplacian.

        BPR:  8.27 × 10⁻⁵ eV²
        PDG:  7.53 × 10⁻⁵ ± 0.18 × 10⁻⁵ eV²
        Tolerance: 15% (currently 9.9% off — CLOSE grade)
        """
        dm21 = predictions["P5.8_delta_m21_sq_eV2"]
        assert dm21 == pytest.approx(7.53e-5, rel=0.15), \
            f"Δm²₂₁ = {dm21:.4e}, expected ≈ 7.53e-5 (within 15%)"

    def test_sum_neutrino_masses(self, predictions):
        """Sum of neutrino masses below cosmological bound.

        BPR:  0.06 eV
        Bound: < 0.12 eV (Planck 2018 + BAO)
        """
        sum_m = predictions["P5.2_sum_masses_eV"]
        assert sum_m < 0.12, f"Σm_ν = {sum_m} eV, exceeds 0.12 eV bound"
        assert sum_m > 0.0, "Sum of masses must be positive"

    def test_number_of_generations(self, predictions):
        """Three neutrino generations from genus-1 boundary topology.

        BPR:  3
        LEP:  2.984 ± 0.008 (from Z-width)
        """
        n_gen = predictions["P5.10_number_of_generations"]
        assert n_gen == 3

    def test_normal_hierarchy(self, predictions):
        """BPR predicts normal neutrino mass ordering."""
        assert predictions["P5.1_hierarchy"] == "normal"

    def test_lorentz_invariance_no_linear_violation(self, predictions):
        """No linear Lorentz invariance violation (ξ₁ = 0).

        BPR:  ξ₁ = 0 (boundary locality)
        LHAASO: E_QG,1 > 10 × M_Pl
        """
        xi1 = predictions["P20.1_LIV_xi1"]
        assert xi1 == 0.0, f"ξ₁ = {xi1}, should be exactly 0"

    def test_lorentz_violation_below_fermi_bound(self, predictions):
        """Speed-of-light deviation below Fermi-LAT bound.

        BPR:  3.38 × 10⁻²¹
        Fermi: < 6.0 × 10⁻²¹
        """
        delta_c = predictions["P20.7_LI_delta_c_over_c"]
        assert delta_c < 6.0e-21, \
            f"|δc/c| = {delta_c:.2e}, exceeds Fermi bound 6e-21"
        assert delta_c > 0, "Deviation must be positive"

    def test_gup_beta_within_bounds(self, predictions):
        """GUP parameter β = 1/p well within experimental bound.

        BPR:  9.55 × 10⁻⁶
        Bound: β < 4 × 10⁴
        """
        beta = predictions["P20.4_GUP_beta"]
        assert beta < 4e4, f"β = {beta}, exceeds bound 4e4"
        assert beta == pytest.approx(1.0 / 104729, rel=1e-3), \
            "β should equal 1/p"

    def test_spatial_dimensions(self, predictions):
        """BPR requires exactly 3 spatial dimensions."""
        assert predictions["P13.3_spatial_dimensions"] == 3

    def test_total_dimensions(self, predictions):
        """BPR requires exactly 3+1 = 4 spacetime dimensions."""
        assert predictions["P13.5_total_dimensions"] == 4

    def test_tsirelson_bound(self, predictions):
        """Bell/BPR bound matches Tsirelson's bound 2√2.

        BPR:  2.8284...
        Exact: 2√2 = 2.82842712...
        """
        bell = predictions["P16.7_bell_bpr_bound"]
        assert bell == pytest.approx(2 * math.sqrt(2), rel=1e-4), \
            f"Bell bound = {bell}, expected 2√2 = {2*math.sqrt(2)}"

    def test_proton_lifetime_exceeds_superK(self, predictions):
        """Proton lifetime exceeds Super-K bound.

        BPR:  ~10⁴³ yr (Theory XVII) and ~10⁵⁰ yr (Theory II)
        Super-K: > 2.4 × 10³⁴ yr
        """
        tau_gut = predictions["P17.8_proton_lifetime_GUT_years"]
        tau_imp = predictions["P2.15_proton_lifetime_years"]
        assert tau_gut > 2.4e34, f"τ_p(GUT) = {tau_gut:.2e} yr, below Super-K"
        assert tau_imp > 2.4e34, f"τ_p(imp) = {tau_imp:.2e} yr, below Super-K"

    def test_dm_self_interaction_below_bullet_cluster(self, predictions):
        """DM self-interaction below Bullet Cluster bound.

        BPR:  0.019 cm²/g
        Bound: < 0.6 cm²/g
        """
        sigma_m = predictions["P2.7_DM_sigma_over_m_cm2_g"]
        assert sigma_m < 0.6, f"σ/m = {sigma_m} cm²/g, exceeds 0.6"
        assert sigma_m > 0, "Self-interaction must be positive"

    def test_delta_neff_below_planck_bound(self, predictions):
        """Extra effective neutrino species below Planck bound.

        BPR:  0.038
        Bound: ΔN_eff < 0.2
        """
        delta_neff = predictions["P11.14_delta_Neff"]
        assert delta_neff < 0.2, f"ΔN_eff = {delta_neff}, exceeds 0.2"

    # ── Charged lepton masses from S² boundary modes (v0.8.0) ──

    def test_electron_mass_derived(self, predictions):
        """Electron mass from S² boundary mode l=1.

        DERIVED from l-modes (1, 14, 59), anchored to m_τ.
        BPR:  0.5104 MeV
        CODATA: 0.51099895 MeV
        Tolerance: 1% (currently 0.11% off)
        """
        m_e = predictions["P18.1_m_electron_MeV"]
        assert m_e == pytest.approx(0.5110, rel=0.01), \
            f"m_e = {m_e:.4f} MeV, expected ≈ 0.511 (within 1%)"

    def test_muon_mass_derived(self, predictions):
        """Muon mass from S² boundary mode l=14.

        DERIVED from l-modes (1, 14, 59), anchored to m_τ.
        BPR:  100.05 MeV
        CODATA: 105.658 MeV
        Tolerance: 8% (currently 5.3% off — a genuine prediction, not a fit)
        """
        m_mu = predictions["P18.2_m_muon_MeV"]
        assert m_mu == pytest.approx(105.66, rel=0.08), \
            f"m_μ = {m_mu:.2f} MeV, expected ≈ 105.66 (within 8%)"

    # ── Up-type quark masses from S² boundary modes (v0.8.0) ──

    def test_up_quark_mass_derived(self, predictions):
        """Up quark mass from S² boundary mode l=1.

        DERIVED from l-modes (1, 24, 283), anchored to m_t.
        BPR:  2.157 MeV
        PDG:  2.16 ± 0.49 MeV
        Tolerance: 5% (currently 0.1% off)
        """
        m_u = predictions["P12.2_m_u_MeV"]
        assert m_u == pytest.approx(2.16, rel=0.05), \
            f"m_u = {m_u:.4f} MeV, expected ≈ 2.16 (within 5%)"

    def test_charm_quark_mass_derived(self, predictions):
        """Charm quark mass from S² boundary mode l=24.

        DERIVED from l-modes (1, 24, 283), anchored to m_t.
        BPR:  1242 MeV
        PDG:  1270 ± 20 MeV
        Tolerance: 5% (currently 2.2% off)
        """
        m_c = predictions["P12.5_m_c_MeV"]
        assert m_c == pytest.approx(1270, rel=0.05), \
            f"m_c = {m_c:.1f} MeV, expected ≈ 1270 (within 5%)"

    # ── CKM θ₁₂ from Gatto–Sartori–Tonin (v0.8.0) ──

    def test_ckm_theta12_derived(self, predictions):
        """CKM Cabibbo angle from Gatto–Sartori–Tonin relation.

        DERIVED: sin(θ_C) = √(m_d/m_s)
        BPR:  12.92°
        PDG:  12.96° ± 0.03°
        Tolerance: 0.5° (currently 0.04° off)
        """
        theta12 = predictions["P12.8_CKM_theta12_deg"]
        assert theta12 == pytest.approx(12.96, abs=0.5), \
            f"θ₁₂ = {theta12:.2f}°, expected 12.96° ± 0.5°"

    # ── DM relic abundance from thermal freeze-out (v0.8.0) ──

    def test_dm_relic_abundance_is_derived(self, predictions):
        """DM relic abundance from thermal WIMP freeze-out.

        DERIVED: Ω h² = 3×10⁻²⁷ / ⟨σv⟩ (standard freeze-out)
        BPR:  ~9.5 (overproduces DM by ~80×)
        Planck: 0.120 ± 0.001

        This prediction FAILS experimentally, but it is now a GENUINE
        calculation from BPR parameters, not a hardcoded answer.
        The test verifies the calculation runs and gives a positive result.
        """
        omega = predictions["P11.15_DM_relic_Omega_h2"]
        assert omega > 0, "Ω_DM h² must be positive"
        assert math.isfinite(omega), "Ω_DM h² must be finite"
        # Verify it's NOT the old hardcoded 0.12 value
        assert abs(omega - 0.12) > 0.01, \
            f"Ω = {omega:.4f} — suspiciously close to old hardcoded 0.12"


# =====================================================================
# 2. FRAMEWORK PREDICTIONS (BPR formula, some experimental input)
# =====================================================================

class TestFrameworkPredictions:
    """Predictions using BPR formulas with some experimental inputs."""

    def test_mond_a0(self, predictions):
        """MOND acceleration scale a₀ = cH₀/(2π).

        BPR:  1.04 × 10⁻¹⁰ m/s²
        SPARC: 1.2 × 10⁻¹⁰ ± 0.2 × 10⁻¹⁰ m/s²
        """
        a0 = predictions["P2.2_MOND_a0"]
        assert a0 == pytest.approx(1.2e-10, rel=0.25), \
            f"a₀ = {a0:.3e}, expected ≈ 1.2e-10"

    def test_theta12_neutrino(self, predictions):
        """Neutrino θ₁₂ (FRAMEWORK — corrected from tri-bimaximal).

        BPR:  33.65°
        PDG:  33.41° ± 0.8°
        """
        theta12 = predictions["P5.5_theta12_deg"]
        assert theta12 == pytest.approx(33.41, abs=2.0), \
            f"θ₁₂ = {theta12}°, expected 33.41° ± 2°"

    def test_theta23_neutrino(self, predictions):
        """Neutrino θ₂₃ (FRAMEWORK — corrected from maximal).

        BPR:  47.6°
        PDG:  49° ± 1.3°
        """
        theta23 = predictions["P5.6_theta23_deg"]
        assert theta23 == pytest.approx(49.0, abs=3.0), \
            f"θ₂₃ = {theta23}°, expected 49° ± 3°"

    def test_baryon_asymmetry_order_of_magnitude(self, predictions):
        """Baryon-to-photon ratio η (order of magnitude).

        BPR:  3.0 × 10⁻¹⁰
        Planck: 6.12 × 10⁻¹⁰
        Tolerance: factor of 3 (currently 2× off)
        """
        eta = predictions["P11.7_baryon_asymmetry_eta"]
        assert 1e-10 < eta < 2e-9, \
            f"η = {eta:.2e}, expected ~ 6e-10 (within factor 3)"

    def test_binding_energy_Fe56(self, predictions):
        """Binding energy per nucleon for ⁵⁶Fe.

        BPR:  8.85 MeV (BW + BPR correction)
        AME:  8.790 MeV
        """
        ba = predictions["P19.7_B_per_A_Fe56_MeV"]
        assert ba == pytest.approx(8.79, rel=0.02), \
            f"B/A(Fe56) = {ba:.3f} MeV, expected 8.790"

    def test_ns_max_mass(self, predictions):
        """Neutron star maximum mass.

        BPR:  2.2 M_☉
        Obs:  2.08 ± 0.07 M_☉ (PSR J0740+6620)
        """
        mmax = predictions["P19.10_NS_max_mass_solar"]
        assert mmax == pytest.approx(2.08, rel=0.10), \
            f"M_max = {mmax:.2f} M_☉, expected ≈ 2.08"

    def test_ns_radius(self, predictions):
        """Neutron star radius for 1.4 M_☉.

        BPR:  12.4 km
        NICER: 12.35 ± 0.75 km
        """
        r = predictions["P19.11_NS_radius_km"]
        assert r == pytest.approx(12.35, abs=2.0), \
            f"R_NS = {r:.1f} km, expected 12.35 ± 2 km"


# =====================================================================
# 3. CONSISTENCY CHECKS (values from the same framework must agree)
# =====================================================================

class TestCrossModuleConsistency:
    """Same quantity computed in two different ways must agree."""

    def test_proton_lifetime_theories_agree(self, predictions):
        """Proton lifetime from Theory II and XVII should both exceed Super-K.

        These use different mechanisms (impedance vs GUT) but both
        must predict τ_p > 2.4 × 10³⁴ yr.
        """
        tau_ii = predictions["P2.15_proton_lifetime_years"]
        tau_xvii = predictions["P17.8_proton_lifetime_GUT_years"]
        # Both exceed Super-K
        assert tau_ii > 2.4e34
        assert tau_xvii > 2.4e34
        # Both are finite
        assert tau_ii < 1e100
        assert tau_xvii < 1e100

    def test_gw_speed_consistent(self, predictions):
        """GW speed deviation consistent with zero (GR limit)."""
        delta_v = predictions["P7.1_vGW_equals_c"]
        assert abs(delta_v) < 1e-14, \
            f"|v_GW/c - 1| = {delta_v}, should be ~ 0"

    def test_strong_cp_zero(self, predictions):
        """Strong CP θ = 0 for p ≡ 1 mod 4 (orientable boundary)."""
        theta = predictions["P12.12_strong_CP_theta"]
        assert theta == 0.0, f"θ_QCD = {theta}, expected 0"


# =====================================================================
# 4. EXACT VALUES (must match precisely)
# =====================================================================

class TestExactValues:
    """Predictions that must match exact theoretical or observational values."""

    def test_hall_resistance(self, predictions):
        """von Klitzing constant R_K = h/e² (exact since 2019 SI).

        BPR:  25812.807... Ω
        Exact: 25812.80745... Ω
        """
        R_K = predictions["P14.2_hall_resistance_nu1_Ohm"]
        assert R_K == pytest.approx(25812.80745, rel=1e-8), \
            f"R_K = {R_K}, expected 25812.80745"

    def test_conductance_quantum(self, predictions):
        """Conductance quantum G₀ = 2e²/h (exact since 2019 SI).

        BPR:  7.7481e-5 S
        Exact: 7.748091729e-5 S
        """
        G0 = predictions["P14.12_conductance_quantum_S"]
        assert G0 == pytest.approx(7.748091729e-5, rel=1e-8), \
            f"G₀ = {G0}, expected 7.748091729e-5"

    def test_koide_parameter(self, predictions):
        """Koide parameter Q predicted to be exactly 2/3."""
        Q_pred = predictions["P18.5_koide_BPR_prediction"]
        assert Q_pred == pytest.approx(2.0 / 3.0, rel=1e-10), \
            f"Q_BPR = {Q_pred}, expected 2/3"

    def test_magic_numbers(self, predictions):
        """Nuclear magic numbers must be [2, 8, 20, 28, 50, 82, 126]."""
        magic = predictions["P19.5_magic_numbers"]
        expected = [2, 8, 20, 28, 50, 82, 126]
        if isinstance(magic, str):
            import ast
            magic = ast.literal_eval(magic)
        assert magic == expected, f"Magic numbers: {magic}"
