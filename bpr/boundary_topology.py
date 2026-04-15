"""
bpr/boundary_topology.py

Derives the boundary topology S² from first principles.

The central result: S² is not a structural assumption in BPR.  It is the
unique compact connected orientable 2-manifold with trivial fundamental
group.  This follows from the classification of compact surfaces combined
with three physical requirements that the BPR boundary must satisfy.

Reference: LIMITATIONS_AND_FALSIFICATION.md §12
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactOrientableSurface:
    """A compact connected orientable 2-manifold, classified by genus."""

    genus: int  # 0 = S², 1 = T², 2 = double torus, …

    @property
    def name(self) -> str:
        return {0: "S²", 1: "T²"}.get(self.genus, f"Σ_{self.genus}")

    @property
    def euler_characteristic(self) -> int:
        return 2 - 2 * self.genus

    @property
    def first_betti_number(self) -> int:
        """b₁ = rank of H¹(Σ, ℤ).  Equals 2g for orientable surfaces."""
        return 2 * self.genus

    @property
    def pi1_rank(self) -> int:
        """Rank of the abelianisation of π₁.  Equals 2g."""
        return 2 * self.genus

    @property
    def is_simply_connected(self) -> bool:
        """π₁ = 0 iff g = 0."""
        return self.genus == 0

    @property
    def holonomy_free_parameters(self) -> int:
        """
        Number of independent U(1) holonomy parameters around non-contractible
        loops.  Each independent loop contributes one continuous real parameter
        (or one Z_p-valued discrete parameter for discrete gauge group Z_p).

        This is the parameter count that violates BPR's zero-free-parameter
        requirement if genus > 0.
        """
        return self.first_betti_number

    @property
    def laplacian_first_eigenspace_dim(self) -> Optional[int]:
        """
        Multiplicity of the first non-trivial eigenvalue of the Laplacian for
        the constant-curvature (uniformisation) metric.

        - S² (g=0): round metric, ℓ=1 sector, multiplicity 3.
        - T² (g=1): flat metric, first eigenvalue multiplicity 4 (square) or
          2 (generic rectangular) — not uniquely defined for all flat tori.
        - g≥2: hyperbolic metric, multiplicity depends on Teichmüller moduli.

        Returns the value for the maximally symmetric metric, or None if it
        is not canonically determined.
        """
        if self.genus == 0:
            return 3  # SO(3)-irreducible ℓ=1 sector: m = -1, 0, +1
        return None  # depends on moduli

    @property
    def continuous_isometry_dim(self) -> int:
        """
        Dimension of the continuous isometry group for the canonical metric.

        - S²: SO(3), dim = 3
        - T²: U(1)², dim = 2
        - Σ_g (g≥2): finite group, dim = 0
        """
        if self.genus == 0:
            return 3
        if self.genus == 1:
            return 2
        return 0


# ---------------------------------------------------------------------------
# The three BPR necessary conditions
# ---------------------------------------------------------------------------

def satisfies_compactness(surface: CompactOrientableSurface) -> bool:
    """
    Condition 1: The boundary must be compact.

    A non-compact boundary has a continuous Laplacian spectrum, which would
    give a continuum of winding-mode energies rather than discrete particle
    masses.  All CompactOrientableSurface instances are compact by definition.
    """
    return True  # true by construction for this class


def satisfies_orientability(surface: CompactOrientableSurface) -> bool:
    """
    Condition 2: The boundary must be orientable.

    Global spinor fields (required for fermion generations) exist on Σ only
    if Σ has a spin structure, which requires the second Stiefel–Whitney class
    w₂(Σ) = 0.  For orientable surfaces, w₂ = 0 always.  Non-orientable
    surfaces (Klein bottle, RP²) are excluded by assumption of this class.
    """
    return True  # true by construction: class is restricted to orientable


def satisfies_no_free_holonomy_parameters(
    surface: CompactOrientableSurface,
) -> bool:
    """
    Condition 3: The boundary must have no free holonomy parameters.

    BPR claims zero free parameters in the dimensionless sector.  A gauge
    field on Σ has holonomies around each independent non-contractible loop.
    For U(1) gauge group, each loop contributes one continuous free parameter.
    For Z_p gauge group, each loop contributes one discrete parameter valued
    in Z_p — still not determined by (p, z).

    The parameter count is b₁(Σ) = 2g.  The condition is b₁ = 0, i.e. g = 0.
    """
    return surface.holonomy_free_parameters == 0


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------

def derive_boundary_topology(max_genus_checked: int = 5) -> dict:
    """
    Apply the three necessary BPR conditions to the complete list of compact
    connected orientable 2-manifolds (classified by genus).

    Returns a dict with:
        'survivors': list of surfaces passing all three conditions
        'eliminated': list of (surface, failed_condition_name) pairs
        'derived_topology': the unique surviving surface (S²), or None
        'proof_summary': human-readable string
    """
    survivors = []
    eliminated = []

    for g in range(max_genus_checked + 1):
        surf = CompactOrientableSurface(genus=g)
        if not satisfies_compactness(surf):
            eliminated.append((surf, "compactness"))
        elif not satisfies_orientability(surf):
            eliminated.append((surf, "orientability"))
        elif not satisfies_no_free_holonomy_parameters(surf):
            eliminated.append(
                (surf, f"holonomy-free (b₁ = {surf.first_betti_number})")
            )
        else:
            survivors.append(surf)

    derived = survivors[0] if len(survivors) == 1 else None

    lines = [
        "BPR boundary topology derivation",
        "=" * 40,
        "",
        "Necessary conditions:",
        "  1. Compact            → discrete mass spectrum",
        "  2. Orientable         → global spinor fields exist",
        "  3. π₁ = 0             → no free holonomy parameters",
        "",
        f"Compact orientable surfaces checked: genus 0 … {max_genus_checked}",
        "",
    ]

    for surf in survivors:
        lines.append(f"  DERIVED: {surf.name} (genus {surf.genus}) — all conditions satisfied")
    for surf, reason in eliminated:
        lines.append(
            f"  ELIMINATED: {surf.name} (genus {surf.genus}) — fails {reason}"
        )

    lines += [
        "",
        "Conclusion:",
        f"  Unique surviving topology: {derived.name if derived else 'none / ambiguous'}",
        "  By the classification of compact surfaces, S² is the only",
        "  compact connected orientable 2-manifold with π₁ = 0.",
        "  S² is derived, not assumed.",
        "",
        "Corollary (3 generations):",
        f"  First non-trivial Laplacian eigenspace on S²: dim = {CompactOrientableSurface(0).laplacian_first_eigenspace_dim}",
        "  This equals the ℓ = 1 sector (m = −1, 0, +1) of SO(3).",
        "  Identification of each sector with one fermion generation gives",
        "  exactly 3 generations — derived, not postulated.",
    ]

    return {
        "survivors": survivors,
        "eliminated": eliminated,
        "derived_topology": derived,
        "proof_summary": "\n".join(lines),
    }


def verify_s2_uniqueness() -> bool:
    """
    Return True iff S² is the unique compact orientable simply-connected
    2-manifold up to the checked genus range.  Used in test suite.
    """
    result = derive_boundary_topology()
    survivors = result["survivors"]
    return (
        len(survivors) == 1
        and survivors[0].genus == 0
        and survivors[0].name == "S²"
    )


def generation_count_from_topology(surface: CompactOrientableSurface) -> Optional[int]:
    """
    Predict the number of fermion generations from boundary topology.

    Fermion generations = multiplicity of the first non-trivial eigenspace of
    the Laplacian on the boundary, for the maximally symmetric metric.

    For S²: returns 3.
    For others: returns None (multiplicity is metric-dependent).
    """
    return surface.laplacian_first_eigenspace_dim


if __name__ == "__main__":
    result = derive_boundary_topology()
    print(result["proof_summary"])
    print()
    surf = result["derived_topology"]
    if surf:
        n_gen = generation_count_from_topology(surf)
        print(f"Fermion generations predicted by {surf.name}: {n_gen}")
        print(f"Killing vectors (continuous isometries): {surf.continuous_isometry_dim}")
        print(f"Euler characteristic: {surf.euler_characteristic}")
        print(f"Holonomy free parameters: {surf.holonomy_free_parameters}")
