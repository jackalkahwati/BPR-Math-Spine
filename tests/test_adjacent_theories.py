"""
Tests for the Ten Adjacent Theories (Al-Kahwati, Feb 2026).

Each section tests one theory module and its key equations / predictions.
Run with:  pytest tests/test_adjacent_theories.py -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Theory I : Boundary Memory Dynamics  (bpr.memory)
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryKernel:
    """§3.2  M(t,t') = exp(-|t-t'|/τ_m) cos(ω_r (t-t'))"""

    def test_kernel_at_zero_lag(self):
        from bpr.memory import memory_kernel, MemoryKernelParams
        params = MemoryKernelParams(p=7, n=1, W=0, tau_0=1.0)
        assert memory_kernel(np.array([0.0]), np.array([0.0]), params) == pytest.approx(1.0)

    def test_kernel_decays(self):
        from bpr.memory import memory_kernel, MemoryKernelParams
        params = MemoryKernelParams(p=7, n=1, W=0, tau_0=1.0)
        M0 = memory_kernel(np.array([0.0]), np.array([0.0]), params)
        M1 = memory_kernel(np.array([5.0]), np.array([0.0]), params)
        assert abs(M1) < abs(M0), "Kernel should decay with increasing lag"

    def test_topological_protection_extends_memory(self):
        from bpr.memory import MemoryKernelParams
        p0 = MemoryKernelParams(W=0, tau_0=1.0, alpha=1.0)
        p1 = MemoryKernelParams(W=3, tau_0=1.0, alpha=1.0)
        assert p1.tau_m > p0.tau_m, "Winding W>0 should extend memory timescale"

    def test_prime_harmonic_frequencies(self):
        from bpr.memory import prime_harmonic_frequencies
        freqs = prime_harmonic_frequencies(p=7, n_max=3)
        expected = 2 * np.pi * np.array([1, 2, 3]) / 7.0
        np.testing.assert_allclose(freqs, expected)

    def test_multimode_kernel_reduces_to_single(self):
        from bpr.memory import memory_kernel_multimode
        # Single mode should match dominant behaviour
        M = memory_kernel_multimode(np.array([0.0]), np.array([0.0]), p=7, n_modes=1)
        assert M == pytest.approx(1.0, abs=0.01)

    def test_consciousness_memory_timescale(self):
        from bpr.memory import consciousness_memory_timescale
        assert consciousness_memory_timescale(W=0, tau_0=1.0) == 1.0
        assert consciousness_memory_timescale(W=5, tau_0=1.0, alpha=2.0) == 25.0

    def test_boundary_memory_field_runs(self):
        from bpr.memory import BoundaryMemoryField, MemoryKernelParams
        solver = BoundaryMemoryField(kappa=1.0, n_spatial=32,
                                     kernel_params=MemoryKernelParams(p=7, W=1, tau_0=1.0))
        source = np.random.randn(32, 50) * 0.1
        phi = solver.solve(source, n_steps=50)
        assert phi.shape == (32, 50)


# ═══════════════════════════════════════════════════════════════════════════
# Theory II : Vacuum Impedance Mismatch  (bpr.impedance)
# ═══════════════════════════════════════════════════════════════════════════

class TestImpedance:

    def test_impedance_increases_with_winding(self):
        from bpr.impedance import TopologicalImpedance
        Z = TopologicalImpedance(W_c=10)
        assert Z(20) > Z(1), "Higher winding → higher impedance"

    def test_em_coupling_suppressed_for_dark_matter(self):
        from bpr.impedance import TopologicalImpedance
        Z = TopologicalImpedance(W_c=10)
        g_low = Z.em_coupling(1)
        g_high = Z.em_coupling(100)
        # g_high / g_low ≈ 0.01  (suppression factor ≈ 100×)
        assert g_high < g_low * 0.02, "High-winding solitons must be EM-dark"

    def test_dark_energy_density_positive(self):
        from bpr.impedance import DarkEnergyDensity
        de = DarkEnergyDensity()
        assert de.rho_DE > 0

    def test_mond_interpolation_limits(self):
        from bpr.impedance import MONDInterpolation
        mond = MONDInterpolation()
        # Deep-MOND: μ(x) ≈ x for x << 1
        x_small = 1e-3 * mond.a0
        assert mond.mu(x_small) == pytest.approx(x_small / mond.a0, rel=0.1)
        # Newtonian: μ(x) ≈ 1 for x >> 1
        x_large = 1e6 * mond.a0
        assert mond.mu(x_large) == pytest.approx(1.0, abs=1e-3)

    def test_rotation_curve_mond_vs_newtonian(self):
        from bpr.impedance import rotation_curve, MONDInterpolation
        mond = MONDInterpolation()
        # Use radii where a_N << a₀ (deep-MOND: v = (G M a₀)^{1/4}, independent of r)
        # a₀ ≈ 9e-10 m/s²; for M_baryon ~ 1e41 kg, a_N < a₀ above ~ 3e25 m
        M_baryon = 1e11 * 2e30  # ~ 2e41 kg
        r = np.logspace(25.5, 27, 30)  # well into deep-MOND
        v = rotation_curve(r, M_baryon=M_baryon, a0=mond.a0)
        # In deep-MOND v ∝ r^{1/4}; Keplerian would be r^{-1/2}.
        log_slope = np.polyfit(np.log10(r), np.log10(v), 1)[0]
        assert log_slope < 0.3, f"Deep-MOND slope should be ≤ 0.25; got {log_slope}"
        assert log_slope > -0.1, f"Velocity should not drop Keplerian-like; got {log_slope}"

    def test_dark_matter_profile_modulation(self):
        from bpr.impedance import DarkMatterProfile
        dm = DarkMatterProfile(modulation_amplitude=0.1)
        r = np.linspace(0.1, 10, 200)
        rho = dm(r)
        assert not np.allclose(rho, rho[0]), "Profile should vary with radius"

    def test_dominant_regime(self):
        from bpr.impedance import dominant_regime
        assert dominant_regime(100, 1) == "particle_dark_matter"
        assert dominant_regime(0.01, 1) == "modified_gravity"


# ═══════════════════════════════════════════════════════════════════════════
# Theory III : Boundary-Induced Decoherence  (bpr.decoherence)
# ═══════════════════════════════════════════════════════════════════════════

class TestDecoherence:

    def test_decoherence_rate_scales_with_T(self):
        from bpr.decoherence import decoherence_rate
        g1 = decoherence_rate(T=100, delta_Z=1.0, A_eff=1e-14, lambda_dB=1e-10)
        g2 = decoherence_rate(T=200, delta_Z=1.0, A_eff=1e-14, lambda_dB=1e-10)
        assert g2 == pytest.approx(2 * g1, rel=1e-6)

    def test_decoherence_rate_scales_with_deltaZ_squared(self):
        from bpr.decoherence import decoherence_rate
        g1 = decoherence_rate(T=300, delta_Z=1.0, A_eff=1e-14, lambda_dB=1e-10)
        g2 = decoherence_rate(T=300, delta_Z=2.0, A_eff=1e-14, lambda_dB=1e-10)
        assert g2 == pytest.approx(4 * g1, rel=1e-6)

    def test_critical_winding(self):
        from bpr.decoherence import critical_winding
        Wc = critical_winding(gamma_dec=100, omega_system=1)
        assert Wc == pytest.approx(10.0)

    def test_is_quantum(self):
        from bpr.decoherence import is_quantum
        assert is_quantum(W=20, gamma_dec=100, omega_system=1) is True
        assert is_quantum(W=5, gamma_dec=100, omega_system=1) is False

    def test_pointer_basis_from_geometry(self):
        from bpr.decoherence import PointerBasis
        grad = np.random.randn(8)
        pb = PointerBasis.from_boundary_geometry(kappa=1.0, phi_gradient=grad)
        vals, vecs = pb.compute()
        assert len(vals) == 8

    def test_decoherence_free_modes(self):
        from bpr.decoherence import decoherence_free_modes
        B = np.diag([0.0, 0.0, 1.0, 2.0])
        free = decoherence_free_modes(B)
        assert free.shape[1] == 2, "Two zero-eigenvalue modes expected"

    def test_quantum_correction_at_low_T(self):
        from bpr.decoherence import decoherence_rate_with_quantum_correction
        T = np.array([0.01, 300.0])
        rates = decoherence_rate_with_quantum_correction(
            T, delta_Z=1.0, A_eff=1e-14, lambda_dB=1e-10, T_quantum=1.0
        )
        # At T=0.01 K the quantum correction should suppress the rate
        ratio = rates[0] / rates[1]
        assert ratio < 0.01 / 300.0 * 1.5  # suppressed beyond linear scaling


# ═══════════════════════════════════════════════════════════════════════════
# Theory IV : Universal Phase Transition Taxonomy  (bpr.phase_transitions)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseTransitions:

    def test_class_b_exponents_d3(self):
        from bpr.phase_transitions import class_b_critical_exponents
        exp = class_b_critical_exponents(d=3)
        assert exp["nu"] == pytest.approx(0.8)
        assert exp["beta"] == pytest.approx(0.2)
        assert exp["gamma"] == pytest.approx(1.6)

    def test_kibble_zurek_scaling(self):
        from bpr.phase_transitions import kibble_zurek_defect_density
        n1 = kibble_zurek_defect_density(tau_quench=1.0)
        n2 = kibble_zurek_defect_density(tau_quench=10.0)
        assert n2 < n1, "Slower quench → fewer defects"

    def test_classify_transition(self):
        from bpr.phase_transitions import classify_transition, TransitionClass
        assert classify_transition("Superconductivity") == TransitionClass.C
        assert classify_transition("QCD confinement") == TransitionClass.B
        assert classify_transition("Consciousness onset") == TransitionClass.A

    def test_landau_order_parameter(self):
        from bpr.phase_transitions import landau_order_parameter
        assert landau_order_parameter(a=-1.0, b=1.0) == pytest.approx(np.sqrt(0.5))
        assert landau_order_parameter(a=1.0, b=1.0) == 0.0  # disordered


# ═══════════════════════════════════════════════════════════════════════════
# Theory V : Boundary-Mediated Neutrino Dynamics  (bpr.neutrino)
# ═══════════════════════════════════════════════════════════════════════════

class TestNeutrino:

    def test_normal_hierarchy_predicted(self):
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        assert spec.hierarchy == "normal"

    def test_sum_masses_approx_006(self):
        from bpr.neutrino import NeutrinoMassSpectrum
        spec = NeutrinoMassSpectrum()
        assert np.sum(spec.masses_eV) == pytest.approx(0.06, abs=0.001)

    def test_pmns_unitarity(self):
        from bpr.neutrino import PMNSMatrix
        pmns = PMNSMatrix()
        assert pmns.is_unitary(tol=1e-6)

    def test_neutrino_nature_dirac(self):
        from bpr.neutrino import neutrino_nature
        assert neutrino_nature(5) == "Dirac"       # 5 ≡ 1 (mod 4)
        assert neutrino_nature(7) == "Majorana"     # 7 ≡ 3 (mod 4)

    def test_oscillation_probability_range(self):
        from bpr.neutrino import oscillation_probability
        P = oscillation_probability(L=1000, E=1e9, delta_m_sq=2.5e-3)
        assert 0 <= P <= 1

    def test_sterile_neutrino_mass(self):
        from bpr.neutrino import SterileNeutrino
        sn = SterileNeutrino(kappa=1.0, R_decoupled=1e-20)
        assert sn.mass == 1e20


# ═══════════════════════════════════════════════════════════════════════════
# Theory VI : Substrate Information Geometry  (bpr.info_geometry)
# ═══════════════════════════════════════════════════════════════════════════

class TestInfoGeometry:

    def test_fisher_metric_positive_definite(self):
        from bpr.info_geometry import FisherMetric
        cov = np.eye(3) * 2.0
        fm = FisherMetric(cov_matrix=cov, sigma_phi=1.0)
        eigenvalues = np.linalg.eigvalsh(fm.g)
        assert np.all(eigenvalues > 0)

    def test_fisher_distance_zero_for_same_point(self):
        from bpr.info_geometry import FisherMetric
        fm = FisherMetric(cov_matrix=np.eye(3))
        assert fm.distance(np.zeros(3)) == 0.0

    def test_topological_cramer_rao(self):
        from bpr.info_geometry import topological_cramer_rao
        # Higher winding → tighter bound (smaller variance)
        v1 = topological_cramer_rao(N=100, F_max=1.0, W=1)
        v2 = topological_cramer_rao(N=100, F_max=1.0, W=10)
        assert v2 < v1

    def test_thermodynamic_length_positive(self):
        from bpr.info_geometry import thermodynamic_length
        g = np.eye(2)
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        L = thermodynamic_length(g, path)
        assert L == pytest.approx(2.0, abs=0.01)

    def test_geodesic_endpoints(self):
        from bpr.info_geometry import FisherMetric
        fm = FisherMetric(cov_matrix=np.eye(2))
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 1.0])
        geo = fm.geodesic(A, B, n_steps=50)
        np.testing.assert_allclose(geo[0], A)
        np.testing.assert_allclose(geo[-1], B)


# ═══════════════════════════════════════════════════════════════════════════
# Theory VII : Gravitational Wave Phenomenology  (bpr.gravitational_waves)
# ═══════════════════════════════════════════════════════════════════════════

class TestGravitationalWaves:

    def test_gw_speed_equals_c(self):
        from bpr.gravitational_waves import GWPropagation
        gw = GWPropagation()
        assert gw.dispersion == pytest.approx(0.0, abs=1e-6)

    def test_quadrupole_power_positive(self):
        from bpr.gravitational_waves import GWQuadrupole
        Q3 = np.random.randn(3, 3, 100) * 1e30
        gw = GWQuadrupole(Q_phi_dddot=Q3)
        assert gw.power > 0

    def test_gw_memory_nonzero_for_asymmetric_source(self):
        from bpr.gravitational_waves import gw_memory_displacement
        from bpr.memory import memory_kernel, MemoryKernelParams
        params = MemoryKernelParams(p=7, W=1, tau_0=1.0)
        kernel = lambda t, tp: memory_kernel(np.array([t]), np.array([tp]), params).item()
        times = np.linspace(0, 10, 200)
        dt = times[1] - times[0]
        # Asymmetric stress burst
        delta_T = np.exp(-(times - 3) ** 2) - 0.5 * np.exp(-(times - 7) ** 2)
        displacement = gw_memory_displacement(kernel, delta_T, times, dt)
        assert abs(displacement) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Theory VIII : Substrate Complexity Theory  (bpr.complexity)
# ═══════════════════════════════════════════════════════════════════════════

class TestComplexity:

    def test_topological_parallelism(self):
        from bpr.complexity import TopologicalParallelism
        tp = TopologicalParallelism(p=7, W=3)
        assert tp.n_parallel == pytest.approx(7 ** 3)

    def test_effective_qubits(self):
        from bpr.complexity import TopologicalParallelism
        tp = TopologicalParallelism(p=2, W=10)
        assert tp.effective_qubits() == pytest.approx(10.0)

    def test_complexity_bound_requires_exponential(self):
        from bpr.complexity import TopologicalComplexityBound
        cb = TopologicalComplexityBound(n=20, p=7)
        assert cb.requires_exponential_resource is True

    def test_adiabatic_gap_closes(self):
        from bpr.complexity import adiabatic_gap
        g1 = adiabatic_gap(n=5, p=7)
        g2 = adiabatic_gap(n=10, p=7)
        assert g2 < g1, "Gap should close with problem size"


# ═══════════════════════════════════════════════════════════════════════════
# Theory IX : Bioelectric Substrate Coupling  (bpr.bioelectric)
# ═══════════════════════════════════════════════════════════════════════════

class TestBioelectric:

    def test_morphogenetic_field_solves(self):
        from bpr.bioelectric import MorphogeneticField
        mf = MorphogeneticField(kappa_bio=1.0, n_cells=10,
                                V_mem=np.random.randn(10) * 50)
        phi = mf.solve_1d(L=1.0, n_grid=64)
        assert phi.shape == (64,)

    def test_cell_division_conserves_winding(self):
        from bpr.bioelectric import CellularWinding
        parent = CellularWinding(W=4.0)
        d1, d2 = parent.divide(split_ratio=0.6)
        assert d1.W + d2.W == pytest.approx(parent.W)

    def test_aberrant_winding_detected(self):
        from bpr.bioelectric import CellularWinding
        cell = CellularWinding(W=3.5)
        assert cell.is_aberrant(W_normal=1.0) is True
        assert cell.is_aberrant(W_normal=3.5) is False

    def test_bioelectric_reset(self):
        from bpr.bioelectric import CellularWinding
        cancer = CellularWinding(W=7.0, cell_type="epithelial")
        fixed = CellularWinding.bioelectric_reset(cancer, W_target=1.0)
        assert fixed.W == 1.0

    def test_aging_model_monotone_decay(self):
        from bpr.bioelectric import AgingModel
        model = AgingModel(tau_0=100, tau_aging=30)
        ages = np.array([0, 30, 60, 90])
        coh = model.coherence_time(ages)
        assert np.all(np.diff(coh) < 0), "Coherence should decrease with age"

    def test_evolution_bifurcation(self):
        from bpr.bioelectric import EvolutionTopology
        W_a = np.array([1.0, 2.0, 3.0])
        delta = np.array([0.1, 0.2, 0.3])
        s1, s2 = EvolutionTopology.bifurcation_from_ancestor(W_a, delta)
        np.testing.assert_allclose(s1, W_a + delta)
        np.testing.assert_allclose(s2, W_a - delta)


# ═══════════════════════════════════════════════════════════════════════════
# Theory X : Resonant Collective Dynamics  (bpr.collective)
# ═══════════════════════════════════════════════════════════════════════════

class TestCollective:

    def test_collective_coherence_range(self):
        from bpr.collective import CollectivePhaseField
        # Random phases → low coherence
        cpf_random = CollectivePhaseField(phases=np.random.uniform(0, 2 * np.pi, 1000))
        assert cpf_random.coherence < 0.2
        # Aligned phases → high coherence
        cpf_aligned = CollectivePhaseField(phases=np.zeros(1000))
        assert cpf_aligned.coherence == pytest.approx(1.0)

    def test_kuramoto_critical_coupling(self):
        from bpr.collective import KuramotoFlocking
        kf = KuramotoFlocking(N=100, K=1.0)
        Kc = kf.critical_coupling
        assert Kc > 0

    def test_kuramoto_simulation_runs(self):
        from bpr.collective import KuramotoFlocking
        kf = KuramotoFlocking(N=20, K=2.0, noise=0.05)
        history, coherence = kf.simulate(n_steps=100, dt=0.01)
        assert history.shape == (100, 20)
        assert len(coherence) == 100

    def test_market_price(self):
        from bpr.collective import MarketImpedance
        mi = MarketImpedance(
            valuations=np.array([100.0, 200.0]),
            impedances=np.array([1.0, 1.0]),
        )
        assert mi.market_price == pytest.approx(150.0)

    def test_tipping_point(self):
        from bpr.collective import TippingPoint
        tp = TippingPoint(mean_degree=10)
        assert tp.critical_fraction == pytest.approx(0.1)
        assert tp.has_tipped(0.15) is True
        assert tp.has_tipped(0.05) is False

    def test_cooperative_winding_defection_cost(self):
        from bpr.collective import CooperativeWinding
        cw = CooperativeWinding(individual_windings=np.array([1.0, 1.0, 1.0, 1.0]))
        cost = cw.defection_cost(0)
        assert cost > 0, "Defection should be costly in aligned group"

    def test_topological_protection(self):
        from bpr.collective import CooperativeWinding
        aligned = CooperativeWinding(individual_windings=np.ones(10))
        assert aligned.is_topologically_protected() is True
        mixed = CooperativeWinding(individual_windings=np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1.0]))
        assert mixed.is_topologically_protected() is False
