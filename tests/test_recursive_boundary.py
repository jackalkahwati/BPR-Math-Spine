"""Tests for Postulate 0 (Crop-Circle Recursion) and the hexagram template."""

from __future__ import annotations

import numpy as np
import pytest

from bpr.recursive_boundary import (
    CANONICAL_DEPTH,
    HEXAGRAM_VERTEX_COUNT,
    PRINCIPAL_ANGULAR_MODE,
    HexagramTemplate,
    RecursiveBoundary,
    RecursiveHexagramTemplate,
    ScaleGenerator,
    central_node_source,
    ccr_invariant,
    default_generator,
    hexagram_template,
    phase_match_residual,
    recursive_hexagram_template,
    scaling_weight_from_casimir_delta,
)


# ---------------------------------------------------------------------------
# Scaling-generator basics
# ---------------------------------------------------------------------------

def test_generator_requires_sigma_gt_one():
    with pytest.raises(ValueError):
        ScaleGenerator(sigma=1.0, scaling_weight=0.5)


def test_universal_delta_equals_2_delta_phi():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    assert gen.universal_delta() == pytest.approx(1.37, rel=1e-12)


def test_default_generator_uses_derived_delta():
    # δ = 2 is the parameter-free value derived under Postulate 0c (QCP).
    # The earlier δ ≈ 1.37 was a fit, now superseded.
    gen = default_generator()
    assert gen.universal_delta() == pytest.approx(2.0, rel=1e-12)


def test_field_rescales_correctly_under_dilation():
    gen = ScaleGenerator(sigma=3.0, scaling_weight=0.685)
    phi_0 = 7.0
    phi_1 = gen.act_on_field(phi_0, k=1)
    phi_2 = gen.act_on_field(phi_0, k=2)
    assert phi_1 == pytest.approx(7.0 * 3.0 ** (-0.685))
    assert phi_2 == pytest.approx(7.0 * 3.0 ** (-1.370))


# ---------------------------------------------------------------------------
# Recursive-boundary spectrum cascade
# ---------------------------------------------------------------------------

def test_eigenvalue_cascade_matches_sigma_minus_two():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    rb = RecursiveBoundary(base_radius=1.0, depth=3, generator=gen)
    base = np.array([2.0, 6.0, 12.0])  # l(l+1) for l = 1, 2, 3
    cascade = rb.eigenvalue_cascade(base)
    assert cascade.shape == (3, 3)
    # Layer 0 unchanged
    np.testing.assert_allclose(cascade[0], base)
    # Layer k rescales by σ^{-2k}
    np.testing.assert_allclose(cascade[1], base * 2.0 ** (-2))
    np.testing.assert_allclose(cascade[2], base * 2.0 ** (-4))


def test_radial_ring_radii_geometric_series():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    rb = RecursiveBoundary(base_radius=1.0, depth=3, generator=gen)
    np.testing.assert_allclose(rb.radial_ring_radii(), [1.0, 2.0, 4.0])


# ---------------------------------------------------------------------------
# Phase-matching condition
# ---------------------------------------------------------------------------

def test_phase_match_residual_is_zero_when_satisfied():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    phi_k = 1.0
    phi_kplus1 = gen.act_on_field(phi_k, k=1)  # matches by construction
    assert phase_match_residual(phi_k, phi_kplus1, gen) == pytest.approx(0.0)


def test_phase_match_residual_detects_violation():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    expected = gen.act_on_field(1.0, k=1)
    violated = expected + 0.1
    assert phase_match_residual(1.0, violated, gen) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Central-node source cascade
# ---------------------------------------------------------------------------

def test_central_node_source_cascade_geometric():
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    src = central_node_source(amplitude_0=1.0, generator=gen, depth=3)
    # J_k = J_0 * σ^{-k(2 + Δ_φ)}
    expected = 2.0 ** (-(2.0 + 0.685) * np.arange(3))
    np.testing.assert_allclose(src, expected)


# ---------------------------------------------------------------------------
# Hexagram template (canonical CCR realization)
# ---------------------------------------------------------------------------

def test_hexagram_template_has_six_inner_and_outer_nodes():
    h = hexagram_template()
    assert h.n_petals == HEXAGRAM_VERTEX_COUNT == 6
    assert h.inner_orbit().shape == (6, 2)
    assert h.outer_orbit().shape == (6, 2)


def test_hexagram_inner_orbit_lies_on_circle():
    h = hexagram_template(inner_radius=1.5)
    radii = np.linalg.norm(h.inner_orbit(), axis=1)
    np.testing.assert_allclose(radii, 1.5)


def test_hexagram_outer_orbit_at_sigma_times_inner_radius():
    h = hexagram_template(inner_radius=1.0, sigma=1.8)
    radii = np.linalg.norm(h.outer_orbit(), axis=1)
    np.testing.assert_allclose(radii, 1.8)


def test_outer_ring_co_aligned_with_inner_orbit():
    """Corrected geometry (2026-05-09): outer C₆ orbit is co-aligned
    with the inner C₆ orbit (offset = 0), not staggered by π/6.
    Source: direct inspection of the reference image — the outer
    rings sit radially over the inner satellites, not between them."""
    h = hexagram_template()
    inner_angles = np.arctan2(h.inner_orbit()[:, 1], h.inner_orbit()[:, 0])
    outer_angles = np.arctan2(h.outer_orbit()[:, 1], h.outer_orbit()[:, 0])
    diff = np.mod(outer_angles[0] - inner_angles[0], 2 * np.pi)
    assert diff == pytest.approx(0.0)


def test_c6_selection_rule_allows_only_multiples_of_6():
    h = hexagram_template()
    allowed = h.allowed_angular_modes(m_max=18)
    # m ∈ {-18, -12, -6, 0, 6, 12, 18}
    expected = np.array([-18, -12, -6, 0, 6, 12, 18])
    np.testing.assert_array_equal(allowed, expected)


def test_principal_angular_mode_constant_locked_to_six():
    assert PRINCIPAL_ANGULAR_MODE == 6


def test_canonical_depth_locked_to_two():
    assert CANONICAL_DEPTH == 2


def test_overlap_circles_six_fold_arrangement_and_overlap_condition():
    """Six overlap-circle centers are equally spaced on a circle of
    radius σ·r_inner; nearest-neighbor center distance equals that
    radius (because six equally-spaced points on a circle of radius
    R have nearest-neighbor distance R). Circles overlap iff σ < 2."""
    σ = 1.5
    r_in = 1.0
    h = hexagram_template(inner_radius=r_in, sigma=σ)
    centers, shared_r = h.overlap_circles()
    expected_outer = σ * r_in
    distances = [
        np.linalg.norm(centers[i] - centers[(i + 1) % 6])
        for i in range(6)
    ]
    np.testing.assert_allclose(distances, expected_outer, rtol=1e-12)
    assert shared_r == pytest.approx(r_in)
    # Star-of-David overlap condition (visible in the reference image)
    assert 2.0 * shared_r > expected_outer


def test_layer_amplitudes_decay_geometrically():
    h = hexagram_template(inner_radius=1.0, sigma=2.0)
    amps = h.layer_amplitudes(phi_0=1.0)
    assert amps.shape == (CANONICAL_DEPTH + 1,)
    # phi_k = σ^{-k Δ_φ} φ_0 with Δ_φ = 0.685
    expected = 2.0 ** (-0.685 * np.arange(CANONICAL_DEPTH + 1))
    np.testing.assert_allclose(amps, expected)


# ---------------------------------------------------------------------------
# RecursiveHexagramTemplate (self-similar nesting)
# ---------------------------------------------------------------------------

def test_recursive_template_root_is_hexagram():
    """Level 0 of the recursive template equals a bare HexagramTemplate."""
    rt = recursive_hexagram_template(inner_radius=1.0, sigma=1.7, depth=2)
    h0 = rt.level_template(0)
    assert h0.inner_radius == pytest.approx(1.0)
    assert h0.angular_offset == pytest.approx(0.0)


def test_recursive_template_level_radii_decay_by_sigma():
    """Each recursion level shrinks the local radius by σ."""
    rt = recursive_hexagram_template(inner_radius=1.0, sigma=1.7, depth=3)
    radii = [rt.level_template(k).inner_radius for k in range(rt.depth)]
    np.testing.assert_allclose(radii, [1.0, 1.0 / 1.7, 1.0 / 1.7 ** 2])


def test_recursive_template_node_counts_match_geometry():
    """Total node count = 1 + 6 + 36 for depth=2; +216 for depth=3."""
    rt2 = recursive_hexagram_template(depth=2)
    assert rt2.total_node_count() == 1 + 6 + 36
    rt3 = recursive_hexagram_template(depth=3)
    assert rt3.total_node_count() == 1 + 6 + 36 + 216


def test_recursive_template_layer_amplitudes_geometric():
    rt = recursive_hexagram_template(inner_radius=1.0, sigma=2.0, depth=2)
    amps = rt.layer_amplitudes(phi_0=1.0)
    assert amps.shape == (3,)
    expected = 2.0 ** (-0.685 * np.arange(3))
    np.testing.assert_allclose(amps, expected)


def test_recursive_template_visible_depth_is_2():
    """Canonical recursion depth from the reference image is K = 2."""
    rt = recursive_hexagram_template()
    assert rt.depth == 2
    levels = rt.all_node_positions()
    assert len(levels) == rt.depth + 1   # K=2 → 3 levels (0, 1, 2)
    assert levels[0].shape == (1, 2)
    assert levels[1].shape == (6, 2)
    assert levels[2].shape == (36, 2)


# ---------------------------------------------------------------------------
# Casimir bridge
# ---------------------------------------------------------------------------

def test_scaling_weight_inversion():
    Δ = scaling_weight_from_casimir_delta(1.37)
    assert Δ == pytest.approx(0.685)


def test_hexagram_default_pins_universal_delta_to_published_value():
    h = hexagram_template()
    assert h.generator.universal_delta() == pytest.approx(1.37, rel=1e-12)


# ---------------------------------------------------------------------------
# CCR action invariance (smoke test)
# ---------------------------------------------------------------------------

def test_ccr_invariant_smoke():
    """A constant Lagrangian density is trivially invariant."""
    gen = ScaleGenerator(sigma=2.0, scaling_weight=0.685)
    x = np.linspace(0.1, 1.0, 8).reshape(-1, 1)
    phi = np.full(8, 0.5)

    def L(_x, _phi):
        return 1.0

    L0, L1 = ccr_invariant(L, x, phi, gen)
    assert L0 == pytest.approx(L1)
