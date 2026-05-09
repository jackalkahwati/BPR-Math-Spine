"""
Postulate 0: Crop-Circle Recursion (CCR)
=========================================

Strong-form scale-recursion postulate added to the BPR math spine.
Anchored to the Stonehenge / hexagram crop-circle template
(reference image, 2026-05-08): one central node, an inner orbit of
six satellites under C₆ symmetry, and an outer orbit of six
"central + ring" units at the vertices of six overlapping circles
(Star of David configuration).

Statement
---------
Let ``Σ_0`` be a base boundary (closed, smooth, holographic) carrying the
phase field ``φ_0 : Σ_0 → ℝ``.  CCR asserts the existence of:

(a) a **discrete rotation group** ``C_n`` (canonical n = 6) acting on
    Σ_0 with generator ``r : x ↦ R(2π/n) x``, and

(b) a **discrete scaling generator**

        s : Σ_0 → Σ_0,        x ↦ σ · x        (σ > 1),

    together with a scaling weight ``Δ_φ ∈ ℝ`` such that the boundary
    action density is invariant under

        (x, φ)  ⟶  (s · x, σ^{−Δ_φ} φ),

(c) a **finite recursion depth** K (canonical K = 2) — the visible
    cascade has two layers (inner C₆ orbit, outer hexagram orbit), not
    an infinite tower.

Consequences (taken as gospel, not derived):

1. The boundary supports a finite tower ``{Σ_k = s^k · Σ_0}`` for
   k = 0, …, K−1 carrying rescaled fields
   ``φ_k(x) = σ^{−k Δ_φ} φ_0(s^{−k} x)``.

2. A single localized source ``χ_b = J δ²(x − x₀)`` at the fixed point
   ``x₀`` of ``s`` (the **central node**) propagates outward as a
   rescaled cascade.  The nodal pattern at each layer is an orbit of
   the rotation group ``C_n`` acting on the principal angular
   eigenmode ``Y_{l m}`` with ``m = n``.

3. Stable configurations are phase-locked across layers via the
   matching condition

       φ_{k+1}(s · x) = σ^{−Δ_φ} φ_k(x)        on  ∂Σ_k,

   and across the C_n orbit by

       φ_k(r · x) = e^{2π i m / n} φ_k(x)        (m = principal mode).

4. The Casimir falsifier exponent ``δ`` (Eq 7) is **universal** — a
   property of the scaling generator, not the cavity:

       δ = 2 Δ_φ      (Casimir bilinear in φ).

5. Particles are secondary effects of the layered φ-spectrum; matter,
   energy, and metric perturbations are nested resonance configurations
   constrained by ``Σ`` geometry through Eq (3).

6. The visible "ring overlap" (Star of David / Flower of Life) at
   layer k = 1 is the locus where six rotated copies of the inner
   eigenmode interfere constructively — a phase-coupled boundary
   interaction, not a coincidence of geometry.

Implementation
--------------
This module exposes:

* ``ScaleGenerator``           – dilation operator (σ, Δ_φ)
* ``RecursiveBoundary``        – nested hierarchy {Σ_k} of finite depth K
* ``central_node_source``      – δ-source at the fixed point
* ``eigenvalue_cascade``       – λ_k(n) = σ^{−2k} λ_0(n) (Laplacian rescale)
* ``radial_ring_radii``        – r_k = σ^k · r_0  (geometric series of rings)
* ``phase_match_residual``     – numerical check of CCR matching condition
* ``universal_delta``          – δ = 2 Δ_φ  (links Eq 7 to the postulate)
* ``ccr_invariant``            – numerical check that L_bndy is invariant
                                 under (x, φ) → (σ x, σ^{−Δ_φ} φ).

The module is pure-numpy and adds no FEniCS dependency.

References
----------
Al-Kahwati (2026), *BPR-Math-Spine*: Postulate 0 supplements
Equations (1)–(7) by promoting the fractal Casimir cavity exponent to a
universal property of a substrate-level scaling generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Postulate 0 statement (importable as a string for docs / paper)
# ---------------------------------------------------------------------------

POSTULATE_0_CCR = (
    "There exists (a) a discrete rotation group C_n (canonical n = 6) "
    "and (b) a discrete scaling generator s : Σ → Σ with scale factor "
    "σ > 1 and scaling weight Δ_φ such that S_bndy[φ] is invariant "
    "under (x, φ) → (σ x, σ^{-Δ_φ} φ) and under r : x → R(2π/n) x. "
    "The boundary supports a finite tower {Σ_k = s^k · Σ_0} (canonical "
    "depth K = 2). A central source at the fixed point of s seeds a "
    "rescaled cascade whose nodal pattern at each layer is a C_n "
    "orbit of the principal angular mode Y_{l m} with m = n. The "
    "Casimir exponent δ in Eq (7) equals 2 Δ_φ — a universal property "
    "of the substrate, not of any single cavity."
)

# Canonical constants pinned to the hexagram template
PRINCIPAL_ANGULAR_MODE: int = 6   # m = n = 6  (C₆ symmetry)
CANONICAL_DEPTH: int = 2          # K = 2  (inner orbit + outer hexagram)
HEXAGRAM_VERTEX_COUNT: int = 6    # six overlapping circles


# ---------------------------------------------------------------------------
# Scaling generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleGenerator:
    """Discrete dilation acting on Σ.

    Parameters
    ----------
    sigma : float
        Scale factor σ > 1.  The generator acts as x ↦ σ x.
    scaling_weight : float
        Field scaling weight Δ_φ.  Under dilation, φ ↦ σ^{-Δ_φ} φ.
    """

    sigma: float
    scaling_weight: float

    def __post_init__(self) -> None:
        if self.sigma <= 1.0:
            raise ValueError("CCR requires σ > 1 (outward cascade).")

    # -- group action --------------------------------------------------

    def act_on_position(self, x: np.ndarray, k: int = 1) -> np.ndarray:
        """Apply s^k to a position vector (or array)."""
        return (self.sigma ** k) * np.asarray(x, dtype=float)

    def act_on_field(self, phi: float | np.ndarray, k: int = 1) -> np.ndarray:
        """Rescale a field value at depth k: φ ↦ σ^{-k Δ_φ} φ."""
        return (self.sigma ** (-k * self.scaling_weight)) * np.asarray(
            phi, dtype=float
        )

    # -- universal Casimir exponent (Eq 7) -----------------------------

    def universal_delta(self) -> float:
        """Casimir falsifier exponent δ promoted to a substrate property.

        For the BPR-Casimir bilinear in φ (Eq 7), the relative deviation
        ΔF / F ∝ (R / R_f)^{-δ} carries two factors of φ, so

            δ = 2 Δ_φ.
        """
        return 2.0 * self.scaling_weight


# ---------------------------------------------------------------------------
# Recursive boundary hierarchy
# ---------------------------------------------------------------------------

@dataclass
class RecursiveBoundary:
    """Tower {Σ_k}_{k=0..K-1} of nested boundary copies.

    Parameters
    ----------
    base_radius : float
        Radius (or characteristic length) of Σ_0.
    depth : int
        Number of layers K ≥ 1.
    generator : ScaleGenerator
        Dilation linking Σ_k to Σ_{k+1}.
    """

    base_radius: float
    depth: int
    generator: ScaleGenerator

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be ≥ 1")
        if self.base_radius <= 0.0:
            raise ValueError("base_radius must be > 0")

    # -- geometry ------------------------------------------------------

    def radial_ring_radii(self) -> np.ndarray:
        """Geometric series of ring radii r_k = σ^k · r_0."""
        return self.base_radius * (
            self.generator.sigma ** np.arange(self.depth)
        )

    # -- spectrum ------------------------------------------------------

    def eigenvalue_cascade(self, base_eigenvalues: np.ndarray) -> np.ndarray:
        """Laplacian eigenvalues at each layer.

        Under dilation, ``∇²_Σ`` rescales as σ^{-2}; therefore

            λ_k(n) = σ^{-2k} · λ_0(n).

        Returns array of shape (depth, len(base_eigenvalues)).
        """
        base = np.asarray(base_eigenvalues, dtype=float)
        sigma = self.generator.sigma
        scale = sigma ** (-2.0 * np.arange(self.depth))
        return scale[:, None] * base[None, :]

    # -- field cascade -------------------------------------------------

    def field_cascade(self, phi_0_at_x: float, x: np.ndarray) -> np.ndarray:
        """Field amplitudes φ_k(s^k x) seeded by φ_0(x) under CCR.

        Parameters
        ----------
        phi_0_at_x : float
            Reference value φ_0(x).
        x : np.ndarray
            Reference position on Σ_0.

        Returns
        -------
        cascade : np.ndarray, shape (depth,)
            Values φ_k at the s^k-image of x.
        """
        sigma = self.generator.sigma
        Δ = self.generator.scaling_weight
        return phi_0_at_x * (sigma ** (-Δ * np.arange(self.depth)))


# ---------------------------------------------------------------------------
# Central node source (gospel: "central node = primary boundary source")
# ---------------------------------------------------------------------------

def central_node_source(
    amplitude_0: float,
    generator: ScaleGenerator,
    depth: int,
) -> np.ndarray:
    """Source amplitudes χ_b^{(k)} at the fixed point of s, layer-by-layer.

    The CCR-rescaled δ-source is

        χ_b^{(k)}(x) = σ^{-k(D-1) - k Δ_φ} · J_0 · δ²(x).

    For a 2-boundary (D-1 = 2), this gives

        J_k = J_0 · σ^{-k(2 + Δ_φ)}.

    Parameters
    ----------
    amplitude_0 : float
        Bare source J_0 on Σ_0.
    generator : ScaleGenerator
    depth : int

    Returns
    -------
    np.ndarray, shape (depth,)
        Source amplitudes at each nested layer.
    """
    sigma = generator.sigma
    Δ = generator.scaling_weight
    return amplitude_0 * (sigma ** (-(2.0 + Δ) * np.arange(depth)))


# ---------------------------------------------------------------------------
# Phase matching across layers
# ---------------------------------------------------------------------------

def phase_match_residual(
    phi_k_at_x: float,
    phi_kplus1_at_sx: float,
    generator: ScaleGenerator,
) -> float:
    """Residual of the CCR matching condition

        φ_{k+1}(s · x)  =  σ^{-Δ_φ}  φ_k(x).

    Returns the absolute residual (target = 0 under the postulate).
    """
    expected = generator.act_on_field(phi_k_at_x, k=1)
    return float(abs(phi_kplus1_at_sx - expected))


# ---------------------------------------------------------------------------
# Action invariance check
# ---------------------------------------------------------------------------

def ccr_invariant(
    lagrangian_density: Callable[[np.ndarray, float], float],
    x_samples: np.ndarray,
    phi_samples: np.ndarray,
    generator: ScaleGenerator,
) -> Tuple[float, float]:
    """Numerically test invariance of L_bndy under (x, φ) → (σ x, σ^{-Δ_φ} φ).

    Parameters
    ----------
    lagrangian_density : callable
        L(x, φ) returning a scalar.  Should depend on x only through
        derivatives if invariance is expected to hold exactly.
    x_samples : np.ndarray
        Sample positions on Σ_0.
    phi_samples : np.ndarray
        Sample field values φ_0(x).
    generator : ScaleGenerator

    Returns
    -------
    (L_before, L_after) : tuple of floats
        Average Lagrangian density before and after the dilation.
        Their difference quantifies the breaking of CCR invariance.
    """
    L_before = float(
        np.mean(
            [lagrangian_density(x, phi) for x, phi in zip(x_samples, phi_samples)]
        )
    )
    x_after = generator.act_on_position(x_samples)
    phi_after = generator.act_on_field(phi_samples)
    # Action measure rescales as σ^{D-1}; for the *density* we report the
    # raw value and let the caller account for the volume factor.
    L_after = float(
        np.mean(
            [lagrangian_density(x, phi) for x, phi in zip(x_after, phi_after)]
        )
    )
    return L_before, L_after


# ---------------------------------------------------------------------------
# Bridge to Eq (7): pin Δ_φ to the published Casimir δ
# ---------------------------------------------------------------------------

# Published BPR Casimir falsifier exponent (Eq 7 in BPR_one_pager.md)
_CASIMIR_DELTA_REF: float = 1.37
_CASIMIR_DELTA_TOL: float = 0.05


def scaling_weight_from_casimir_delta(
    delta: float = _CASIMIR_DELTA_REF,
) -> float:
    """Invert δ = 2 Δ_φ to recover the substrate scaling weight."""
    return 0.5 * delta


def default_generator(sigma: float = np.e) -> ScaleGenerator:
    """Canonical CCR generator pinned to the published Casimir δ.

    Uses σ = e by default (one e-fold per layer); the choice of σ is
    free — only Δ_φ enters dimensionless predictions.
    """
    return ScaleGenerator(
        sigma=sigma,
        scaling_weight=scaling_weight_from_casimir_delta(),
    )


# ---------------------------------------------------------------------------
# Hexagram template — canonical CCR realization (6-fold, depth 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HexagramTemplate:
    """Canonical CCR realization matching the reference crop circle.

    Geometry
    --------
    * Layer 0: central node at the origin.
    * Layer 1: ``n = 6`` satellite nodes on a circle of radius ``r_1``
      at angles ``θ_j = 2π j / 6``.
    * Layer 2: ``n = 6`` "central + ring" units on a circle of radius
      ``r_2 = σ · r_1`` at angles offset by π/6 (Star of David).

    Symmetry
    --------
    * Discrete rotation group ``C_6``.
    * Optional reflection: full symmetry is ``D_6``.
    * Principal angular mode locked to ``m = 6``.

    Spectrum
    --------
    The CCR-allowed boundary modes on this template are
    ``Y_{l m}`` with ``m ∈ {0, ±6, ±12, ...}``; all other angular
    modes are forbidden by the C_6 selection rule.
    """

    inner_radius: float
    generator: ScaleGenerator
    n_petals: int = HEXAGRAM_VERTEX_COUNT
    angular_offset: float = np.pi / 6  # outer ring offset by half-step

    # -- node coordinates ---------------------------------------------

    def central_node(self) -> np.ndarray:
        """Layer-0 fixed point: (0, 0)."""
        return np.zeros(2)

    def inner_orbit(self) -> np.ndarray:
        """Layer-1 satellite positions, shape (6, 2)."""
        θ = 2 * np.pi * np.arange(self.n_petals) / self.n_petals
        return self.inner_radius * np.column_stack([np.cos(θ), np.sin(θ)])

    def outer_orbit(self) -> np.ndarray:
        """Layer-2 hexagram-vertex positions, shape (6, 2)."""
        r2 = self.generator.sigma * self.inner_radius
        θ = (2 * np.pi * np.arange(self.n_petals) / self.n_petals
             + self.angular_offset)
        return r2 * np.column_stack([np.cos(θ), np.sin(θ)])

    # -- selection rule ------------------------------------------------

    def angular_mode_allowed(self, m: int) -> bool:
        """C_n selection rule: m must be a multiple of n."""
        return (m % self.n_petals) == 0

    def allowed_angular_modes(self, m_max: int = 24) -> np.ndarray:
        """All allowed |m| ≤ m_max under the C_n selection rule."""
        return np.array(
            [m for m in range(-m_max, m_max + 1) if self.angular_mode_allowed(m)]
        )

    # -- ring-overlap (Flower of Life) loci ---------------------------

    def overlap_circles(self) -> Tuple[np.ndarray, float]:
        """Centers and shared radius of the six overlapping circles.

        Each outer-orbit point is the center of a circle of radius
        equal to the inner_radius — by construction the circles
        overlap pairwise at the C_6 fixed-point lattice (Star of
        David vertices), reproducing the reference image.

        Returns
        -------
        centers : np.ndarray, shape (6, 2)
        radius  : float
        """
        return self.outer_orbit(), self.inner_radius

    # -- field cascade -------------------------------------------------

    def layer_amplitudes(self, phi_0: float) -> np.ndarray:
        """Amplitude at each layer under CCR rescaling.

        φ_k = σ^{-k Δ_φ} · φ_0,   for k = 0, 1, 2.
        """
        Δ = self.generator.scaling_weight
        σ = self.generator.sigma
        return phi_0 * (σ ** (-Δ * np.arange(CANONICAL_DEPTH + 1)))


def hexagram_template(
    inner_radius: float = 1.0,
    sigma: float = 2.0,
    delta: float = _CASIMIR_DELTA_REF,
) -> HexagramTemplate:
    """Build the canonical hexagram-template CCR realization.

    Defaults reproduce the reference crop-circle layout:
    * 6-fold rotation
    * outer ring at 2× the inner radius
    * scaling weight pinned to δ = 1.37  ⇒  Δ_φ = 0.685
    """
    gen = ScaleGenerator(
        sigma=sigma,
        scaling_weight=scaling_weight_from_casimir_delta(delta),
    )
    return HexagramTemplate(inner_radius=inner_radius, generator=gen)


# ---------------------------------------------------------------------------
# CCR constraint action (S_CCR) — the postulate as a Lagrangian
# ---------------------------------------------------------------------------

@dataclass
class CCRAction:
    """Lagrange-multiplier action enforcing Postulate 0.

    Adds two constraint terms to the total action:

        S_CCR = ∫_Σ √|h| [
            μ_rot   · | φ − P_{C_n} φ |²
          + μ_scale · | L_s φ − σ^{−Δ_φ} φ |²
        ] d^(D-1)x

    where P_{C_n} is the C_n angular projector and L_s is the
    pull-back along the scaling generator.  In the limit
    μ_rot, μ_scale → ∞ these are hard constraints; finite μ-values
    provide a soft enforcement that integrates with the standard BPR
    field equations.
    """

    generator: ScaleGenerator
    n_rotation: int = PRINCIPAL_ANGULAR_MODE
    mu_rot: float = 1.0
    mu_scale: float = 1.0

    # -- C_n projection of an angular-mode amplitude vector ----------

    def project_c_n(self, mode_amplitudes: np.ndarray, m_indices: np.ndarray) -> np.ndarray:
        """Zero out modes whose angular index m is not divisible by n.

        Parameters
        ----------
        mode_amplitudes : np.ndarray
            Coefficients of φ in an angular-mode basis.
        m_indices : np.ndarray
            The angular index m of each coefficient (same length).
        """
        mask = (np.asarray(m_indices) % self.n_rotation) == 0
        return np.where(mask, mode_amplitudes, 0.0)

    def rotation_residual(
        self, mode_amplitudes: np.ndarray, m_indices: np.ndarray
    ) -> float:
        """| φ − P_{C_n} φ |² summed over modes."""
        diff = mode_amplitudes - self.project_c_n(mode_amplitudes, m_indices)
        return float(np.sum(np.abs(diff) ** 2))

    # -- Scaling-covariance residual ----------------------------------

    def scale_residual(
        self,
        phi_at_x: np.ndarray,
        phi_at_sx: np.ndarray,
    ) -> float:
        """| L_s φ − σ^{−Δ_φ} φ |² summed over sample points.

        Inputs are φ sampled at x and at s·x respectively.
        """
        expected = self.generator.act_on_field(phi_at_x, k=1)
        return float(np.sum(np.abs(phi_at_sx - expected) ** 2))

    # -- Total CCR action density --------------------------------------

    def lagrangian(
        self,
        mode_amplitudes: np.ndarray,
        m_indices: np.ndarray,
        phi_at_x: np.ndarray,
        phi_at_sx: np.ndarray,
    ) -> float:
        """L_CCR contribution to the action density."""
        return (
            self.mu_rot * self.rotation_residual(mode_amplitudes, m_indices)
            + self.mu_scale * self.scale_residual(phi_at_x, phi_at_sx)
        )


__all__ = [
    "POSTULATE_0_CCR",
    "PRINCIPAL_ANGULAR_MODE",
    "CANONICAL_DEPTH",
    "HEXAGRAM_VERTEX_COUNT",
    "ScaleGenerator",
    "RecursiveBoundary",
    "HexagramTemplate",
    "CCRAction",
    "central_node_source",
    "phase_match_residual",
    "ccr_invariant",
    "scaling_weight_from_casimir_delta",
    "default_generator",
    "hexagram_template",
]
