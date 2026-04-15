"""
tests/test_boundary_topology.py

Tests for the S² uniqueness derivation in bpr/boundary_topology.py.

The core claim being tested:
  S² is the unique compact connected orientable 2-manifold with trivial
  fundamental group, derived from BPR's three necessary conditions:
  compactness, orientability, and no free holonomy parameters (π₁ = 0).
"""

import pytest
from bpr.boundary_topology import (
    CompactOrientableSurface,
    derive_boundary_topology,
    generation_count_from_topology,
    satisfies_no_free_holonomy_parameters,
    verify_s2_uniqueness,
)


# ---------------------------------------------------------------------------
# Surface properties
# ---------------------------------------------------------------------------

class TestSurfaceProperties:
    def test_s2_genus_0(self):
        s2 = CompactOrientableSurface(genus=0)
        assert s2.name == "S²"
        assert s2.euler_characteristic == 2
        assert s2.first_betti_number == 0
        assert s2.pi1_rank == 0
        assert s2.is_simply_connected is True
        assert s2.holonomy_free_parameters == 0

    def test_torus_genus_1(self):
        t2 = CompactOrientableSurface(genus=1)
        assert t2.name == "T²"
        assert t2.euler_characteristic == 0
        assert t2.first_betti_number == 2
        assert t2.pi1_rank == 2
        assert t2.is_simply_connected is False
        assert t2.holonomy_free_parameters == 2

    def test_genus_g_betti_number(self):
        for g in range(6):
            surf = CompactOrientableSurface(genus=g)
            assert surf.first_betti_number == 2 * g
            assert surf.euler_characteristic == 2 - 2 * g

    def test_only_s2_is_simply_connected(self):
        """π₁ = 0 iff g = 0."""
        for g in range(6):
            surf = CompactOrientableSurface(genus=g)
            assert surf.is_simply_connected == (g == 0)


# ---------------------------------------------------------------------------
# Three necessary conditions
# ---------------------------------------------------------------------------

class TestNecessaryConditions:
    def test_holonomy_condition_passes_only_for_s2(self):
        """b₁ = 0 iff genus = 0."""
        for g in range(6):
            surf = CompactOrientableSurface(genus=g)
            result = satisfies_no_free_holonomy_parameters(surf)
            assert result == (g == 0), (
                f"genus {g}: expected {g == 0}, got {result}"
            )

    def test_torus_fails_holonomy_condition(self):
        t2 = CompactOrientableSurface(genus=1)
        assert satisfies_no_free_holonomy_parameters(t2) is False
        assert t2.holonomy_free_parameters == 2

    def test_double_torus_fails_with_4_parameters(self):
        s2 = CompactOrientableSurface(genus=2)
        assert satisfies_no_free_holonomy_parameters(s2) is False
        assert s2.holonomy_free_parameters == 4


# ---------------------------------------------------------------------------
# S² uniqueness derivation
# ---------------------------------------------------------------------------

class TestS2Uniqueness:
    def test_exactly_one_survivor(self):
        result = derive_boundary_topology()
        assert len(result["survivors"]) == 1

    def test_survivor_is_s2(self):
        result = derive_boundary_topology()
        survivor = result["survivors"][0]
        assert survivor.genus == 0
        assert survivor.name == "S²"

    def test_all_higher_genus_eliminated(self):
        result = derive_boundary_topology(max_genus_checked=5)
        eliminated_genera = [s.genus for s, _ in result["eliminated"]]
        for g in range(1, 6):
            assert g in eliminated_genera, f"genus {g} not eliminated"

    def test_verify_s2_uniqueness_function(self):
        assert verify_s2_uniqueness() is True

    def test_derived_topology_is_s2(self):
        result = derive_boundary_topology()
        derived = result["derived_topology"]
        assert derived is not None
        assert derived.genus == 0

    def test_proof_summary_contains_key_claims(self):
        result = derive_boundary_topology()
        summary = result["proof_summary"]
        assert "S²" in summary
        assert "derived" in summary.lower()
        assert "π₁ = 0" in summary or "pi_1 = 0" in summary.lower()


# ---------------------------------------------------------------------------
# Generation count corollary
# ---------------------------------------------------------------------------

class TestGenerationCount:
    def test_s2_gives_3_generations(self):
        """The ℓ=1 eigenspace of the Laplacian on S² is 3-dimensional."""
        s2 = CompactOrientableSurface(genus=0)
        assert generation_count_from_topology(s2) == 3

    def test_torus_generation_count_undefined(self):
        """T² has no canonical generation count (metric-dependent)."""
        t2 = CompactOrientableSurface(genus=1)
        assert generation_count_from_topology(t2) is None

    def test_s2_killing_vectors_equals_generation_count(self):
        """dim Isom(S²) = dim SO(3) = 3 = generation count."""
        s2 = CompactOrientableSurface(genus=0)
        assert s2.continuous_isometry_dim == generation_count_from_topology(s2)

    def test_generation_count_matches_observed(self):
        """S² predicts 3 generations; experiment observes 3."""
        observed_generations = 3
        s2 = CompactOrientableSurface(genus=0)
        assert generation_count_from_topology(s2) == observed_generations


# ---------------------------------------------------------------------------
# Cross-checks with rest of BPR
# ---------------------------------------------------------------------------

class TestCrossChecks:
    def test_z_coordination_number_from_s2(self):
        """
        The cubic tiling of S² gives z = 6 nearest neighbours.
        This is the z = 6 used throughout BPR (verified via default constants).
        """
        from bpr.constants import Z_DEFAULT
        assert Z_DEFAULT == 6

    def test_orientability_condition_implies_p_mod_4(self):
        """
        S² orientability requires p ≡ 1 (mod 4) for the Z_p phase space.
        Verified against the default p value.
        """
        from bpr.constants import P_DEFAULT
        assert P_DEFAULT % 4 == 1, (
            f"p = {P_DEFAULT} does not satisfy p ≡ 1 (mod 4)"
        )
