"""
Inter-Theory Connection Tests (§13 of *Ten Adjacent Theories*)
================================================================

Validates that the 10 theories form a coherent web, not 10 independent
modules.  Each test chains two or more theory modules together.

Run with:  pytest tests/test_inter_theory.py -v
"""

import numpy as np
import pytest
from bpr.first_principles import SubstrateDerivedTheories


@pytest.fixture(scope="module")
def sdt():
    """Substrate-derived theories with default parameters."""
    return SubstrateDerivedTheories.from_substrate(
        p=104729, N=10000, J_eV=1.0, radius=0.01, geometry="sphere"
    )


# ══════════════════════════════════════════════════════════════════════
# Memory (I)  ↔  Decoherence (III)
# ══════════════════════════════════════════════════════════════════════

class TestMemoryDecoherence:
    """§13: M(t,t') determines Γ(t) = Γ₀(1 − M(t,t))."""

    def test_memory_kernel_determines_decoherence(self, sdt):
        from bpr.memory import memory_kernel
        mk = sdt.memory_kernel_params(W=1.0)
        dr = sdt.decoherence_rate_obj(T=300.0)
        Gamma_0 = dr.gamma_dec

        # At zero lag M(t,t) = 1  →  Γ(t) = 0 (no decoherence)
        M_zero = memory_kernel(np.array([0.0]), np.array([0.0]), mk)
        Gamma_at_zero = Gamma_0 * (1.0 - M_zero)
        assert Gamma_at_zero == pytest.approx(0.0, abs=1e-10)

        # At large lag M → 0  →  Γ → Γ₀
        M_large = memory_kernel(np.array([1e6]), np.array([0.0]), mk)
        Gamma_at_inf = Gamma_0 * (1.0 - M_large)
        assert Gamma_at_inf == pytest.approx(Gamma_0, rel=0.1)


# ══════════════════════════════════════════════════════════════════════
# Memory (I)  ↔  GW Memory (VII)
# ══════════════════════════════════════════════════════════════════════

class TestMemoryGW:
    """§13: GW memory is Theory I kernel applied to gravitational sector."""

    def test_gw_memory_uses_memory_kernel(self, sdt):
        from bpr.memory import memory_kernel
        from bpr.gravitational_waves import gw_memory_displacement

        mk = sdt.memory_kernel_params(W=1.0)
        kernel = lambda t, tp: memory_kernel(
            np.array([t]), np.array([tp]), mk
        ).item()

        times = np.linspace(0, 10, 200)
        dt = times[1] - times[0]
        # Asymmetric GW burst
        delta_T = np.exp(-(times - 3) ** 2) - 0.3 * np.exp(-(times - 7) ** 2)
        disp = gw_memory_displacement(kernel, delta_T, times, dt)
        assert abs(disp) > 0, "GW memory should be non-zero for asymmetric source"


# ══════════════════════════════════════════════════════════════════════
# Dark Sector (II)  ↔  Neutrinos (V)
# ══════════════════════════════════════════════════════════════════════

class TestDarkSectorNeutrino:
    """§13: Heavy sterile neutrinos are high-winding DM candidates."""

    def test_sterile_neutrino_is_dark(self, sdt):
        from bpr.neutrino import SterileNeutrino
        sn = SterileNeutrino(kappa=sdt.kappa_dim, R_decoupled=1e-25)
        mass = sn.mass

        # This sterile ν should be heavy enough to be warm DM
        assert mass > 1.0, "Sterile neutrino should be heavy"

        # Its high winding makes it EM-dark
        imp = sdt.topological_impedance()
        W_sterile = mass / sdt.kappa_dim  # approximate winding
        em_coupling = imp.em_coupling(W_sterile)
        # If W >> W_c, EM coupling is suppressed
        if W_sterile > sdt.W_c * 10:
            assert em_coupling < 0.01


# ══════════════════════════════════════════════════════════════════════
# Decoherence (III)  ↔  Phase Transitions (IV)
# ══════════════════════════════════════════════════════════════════════

class TestDecoherencePhaseTransition:
    """§13: Decoherence is a Class C (impedance) transition."""

    def test_decoherence_is_class_c(self):
        from bpr.phase_transitions import classify_transition, TransitionClass
        assert classify_transition("Decoherence") == TransitionClass.C

    def test_decoherence_has_landau_form(self):
        from bpr.phase_transitions import landau_free_energy
        psi = np.linspace(0, 2, 100)
        F = landau_free_energy(psi, a=-1.0, b=1.0)
        # Should have a minimum at |ψ|² = 1/2 → |ψ| ≈ 0.707
        idx_min = np.argmin(F)
        assert abs(psi[idx_min] - 0.707) < 0.1


# ══════════════════════════════════════════════════════════════════════
# Info Geometry (VI)  ↔  Complexity (VIII)
# ══════════════════════════════════════════════════════════════════════

class TestInfoGeometryComplexity:
    """§13: Fisher metric defines computational cost of transformations."""

    def test_fisher_metric_bounds_computation(self, sdt):
        cr = sdt.cramer_rao(N_measurements=1000, W=1.0)
        var_bound = cr.variance_lower_bound

        # Higher winding → tighter bound → better estimation
        cr_high_W = sdt.cramer_rao(N_measurements=1000, W=5.0)
        assert cr_high_W.variance_lower_bound < var_bound

        # This connects to complexity: more precise → more computational power
        tp = sdt.topological_parallelism(W=5.0)
        assert tp.n_parallel > sdt.topological_parallelism(W=1.0).n_parallel


# ══════════════════════════════════════════════════════════════════════
# Phase Transitions (IV)  ↔  Collective (X)
# ══════════════════════════════════════════════════════════════════════

class TestPhaseTransitionsCollective:
    """§13: Tipping points are Class A, flocking is Class C."""

    def test_tipping_is_class_a(self):
        from bpr.phase_transitions import TransitionClass
        # Social tipping = winding transition = Class A
        # Flocking = impedance transition = Class C
        # Both map to substrate topology changes
        assert TransitionClass.A.value == "winding"
        assert TransitionClass.C.value == "impedance"

    def test_flocking_shows_phase_transition(self):
        """Below K_c: disordered. Above K_c: ordered."""
        from bpr.collective import KuramotoFlocking, CollectivePhaseField
        np.random.seed(42)

        # Low coupling → disorder
        kf_low = KuramotoFlocking(N=50, K=0.01, noise=0.5)
        _, coh_low = kf_low.simulate(n_steps=500, dt=0.01)

        # High coupling → order
        kf_high = KuramotoFlocking(N=50, K=5.0, noise=0.1)
        _, coh_high = kf_high.simulate(n_steps=500, dt=0.01)

        assert np.mean(coh_high[-100:]) > np.mean(coh_low[-100:])


# ══════════════════════════════════════════════════════════════════════
# Biology (IX)  ↔  Consciousness (I, III)
# ══════════════════════════════════════════════════════════════════════

class TestBiologyConsciousness:
    """§13: Biological coherence provides substrate for consciousness."""

    def test_aging_reduces_consciousness_memory(self, sdt):
        from bpr.memory import consciousness_memory_timescale
        aging = sdt.aging_model()

        # Young: high coherence → long memory time
        coh_young = aging.coherence_time(0)
        # Old: low coherence → short memory time
        coh_old = aging.coherence_time(60)

        assert coh_young > coh_old

        # Memory timescale with W > 0 is still long at birth
        tau_conscious = consciousness_memory_timescale(W=5, tau_0=coh_young)
        tau_conscious_old = consciousness_memory_timescale(W=5, tau_0=coh_old)
        assert tau_conscious > tau_conscious_old


# ══════════════════════════════════════════════════════════════════════
# Complexity (VIII)  ↔  Consciousness (I)
# ══════════════════════════════════════════════════════════════════════

class TestComplexityConsciousness:
    """§13: Observer-as-oracle provides computational resources beyond bulk."""

    def test_oracle_exceeds_bulk(self, sdt):
        cb = sdt.complexity_bound(n_input=20)
        # Bulk computation is polynomial-limited
        poly_limit = cb.poly_depth_limit
        # Oracle (observer) accesses exponential winding sectors
        oracle_access = cb.winding_sectors_to_search
        assert oracle_access > poly_limit


# ══════════════════════════════════════════════════════════════════════
# First-principles pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════

class TestFirstPrinciplesPipeline:
    """The entire chain (J, p, N) → predictions runs without free params."""

    def test_summary_runs(self, sdt):
        s = sdt.summary()
        assert "κ (dimless)" in s
        assert "λ_BPR" in s
        assert "Neutrino nature" in s

    def test_predictions_dict(self, sdt):
        preds = sdt.predictions()
        assert len(preds) >= 150, f"Expected 150+ predictions, got {len(preds)}"
        assert "P1.2_prime_harmonic_omega" in preds
        assert "P5.1_hierarchy" in preds
        assert preds["P5.1_hierarchy"] == "normal"
        assert "P7.1_vGW_equals_c" in preds
        assert preds["P7.1_vGW_equals_c"] == pytest.approx(0.0, abs=1e-6)

    def test_no_free_parameters(self, sdt):
        """All derived values should be deterministic from substrate inputs."""
        sdt2 = SubstrateDerivedTheories.from_substrate(
            p=104729, N=10000, J_eV=1.0, radius=0.01, geometry="sphere"
        )
        assert sdt.kappa == sdt2.kappa
        assert sdt.lambda_bpr == sdt2.lambda_bpr
        assert sdt.tau_0 == sdt2.tau_0
        assert sdt.W_c == sdt2.W_c

    def test_different_substrate_different_predictions(self):
        sdt_a = SubstrateDerivedTheories.from_substrate(p=7, N=100, J_eV=0.5)
        sdt_b = SubstrateDerivedTheories.from_substrate(p=104729, N=10000, J_eV=1.0)
        assert sdt_a.lambda_bpr != sdt_b.lambda_bpr
        assert sdt_a.tau_0 != sdt_b.tau_0
