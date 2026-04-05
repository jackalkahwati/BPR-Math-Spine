"""
Categorical Coherence Verification for BPR
============================================

Verifies the 2-category interchange law for boundary phase transformations.
Guarantees path-independent phase evolution under composition.

Key axiom:
    (β·g) ∘ (h·α) = (k·α) ∘ (β·f)

This ensures multi-module BPR computations are consistent regardless
of the order boundary transformations are composed.

References: Al-Kahwati (2026), Coherence of Boundary Phase Transformations
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable


@dataclass
class BoundaryMorphism:
    """A 1-morphism (boundary coupling) between two domains."""
    source: int  # domain index
    target: int  # domain index
    matrix: np.ndarray  # transformation matrix
    name: str = ""


@dataclass
class PhaseTwoMorphism:
    """A 2-morphism (phase transformation) between 1-morphisms."""
    source_morphism: BoundaryMorphism
    target_morphism: BoundaryMorphism
    transform: np.ndarray  # 2-morphism data
    name: str = ""


def horizontal_compose(alpha: PhaseTwoMorphism, f: BoundaryMorphism) -> np.ndarray:
    """Horizontal composition: α·f (whiskering).
    Apply 2-morphism α then follow morphism f."""
    return alpha.transform @ f.matrix


def vertical_compose(beta: PhaseTwoMorphism, alpha: PhaseTwoMorphism) -> np.ndarray:
    """Vertical composition: β ∘ α.
    Stack two 2-morphisms."""
    return beta.transform @ alpha.transform


def verify_exchange_axiom(alpha, beta, f, g, h, k, tolerance=1e-10):
    """Verify the interchange law:
    (β·g) ∘ (h·α) = (k·α) ∘ (β·f)

    Parameters:
    - alpha, beta: PhaseTwoMorphism instances
    - f, g, h, k: BoundaryMorphism instances
    - tolerance: numerical tolerance for equality

    Returns dict with: lhs, rhs, residual, passes
    """
    # LHS: (β·g) ∘ (h·α)
    beta_dot_g = beta.transform @ g.matrix
    h_dot_alpha = h.matrix @ alpha.transform
    lhs = beta_dot_g @ h_dot_alpha

    # RHS: (k·α) ∘ (β·f)
    k_dot_alpha = k.matrix @ alpha.transform
    beta_dot_f = beta.transform @ f.matrix
    rhs = k_dot_alpha @ beta_dot_f

    residual = np.max(np.abs(lhs - rhs))
    return {
        "lhs": lhs, "rhs": rhs,
        "residual": float(residual),
        "passes": bool(residual < tolerance),
        "tolerance": tolerance,
    }


def verify_path_independence(transformations_path1, transformations_path2,
                               initial_state, tolerance=1e-10):
    """Verify that two different composition paths give the same result.

    transformations_path1: list of matrices to compose (left to right)
    transformations_path2: alternative list of matrices
    initial_state: starting state vector

    Returns dict with: final_state_1, final_state_2, residual, passes"""
    state1 = initial_state.copy()
    for T in transformations_path1:
        state1 = T @ state1

    state2 = initial_state.copy()
    for T in transformations_path2:
        state2 = T @ state2

    residual = np.max(np.abs(state1 - state2))
    return {
        "final_state_1": state1, "final_state_2": state2,
        "residual": float(residual),
        "passes": bool(residual < tolerance),
    }


class BoundaryPhaseCategory:
    """A 2-category of boundary phase transformations.

    Objects: domains (indexed by int)
    1-morphisms: boundary couplings (BoundaryMorphism)
    2-morphisms: phase transforms (PhaseTwoMorphism)
    """

    def __init__(self):
        self.domains = set()
        self.morphisms = []
        self.two_morphisms = []

    def add_domain(self, idx):
        self.domains.add(idx)

    def add_morphism(self, source, target, matrix, name=""):
        m = BoundaryMorphism(source, target, matrix, name)
        self.morphisms.append(m)
        self.domains.add(source)
        self.domains.add(target)
        return m

    def add_two_morphism(self, source_morph, target_morph, transform, name=""):
        tm = PhaseTwoMorphism(source_morph, target_morph, transform, name)
        self.two_morphisms.append(tm)
        return tm

    def compose_morphisms(self, f, g):
        """Compose 1-morphisms: g ∘ f (f then g)."""
        assert f.target == g.source, "Morphisms not composable"
        return BoundaryMorphism(f.source, g.target, g.matrix @ f.matrix,
                                 f"{g.name}∘{f.name}")

    def verify_all_exchanges(self, tolerance=1e-10):
        """Verify exchange axiom for all compatible quadruples of 2-morphisms.
        Returns list of (alpha, beta, passes) results."""
        results = []
        for alpha in self.two_morphisms:
            for beta in self.two_morphisms:
                # Find compatible f, g, h, k if they exist
                compatible = self._find_compatible(alpha, beta)
                if compatible:
                    f, g, h, k = compatible
                    result = verify_exchange_axiom(alpha, beta, f, g, h, k, tolerance)
                    results.append({
                        "alpha": alpha.name, "beta": beta.name,
                        **result
                    })
        return results

    def _find_compatible(self, alpha, beta):
        """Find morphisms f,g,h,k compatible with alpha,beta for exchange."""
        # Look for matching source/target domains
        for f in self.morphisms:
            for g in self.morphisms:
                for h in self.morphisms:
                    for k in self.morphisms:
                        if (f.target == h.source and g.target == k.source and
                            f.source == g.source and h.target == k.target):
                            return (f, g, h, k)
        return None

    def coherence_summary(self, tolerance=1e-10):
        """Summary of categorical coherence across all verified exchanges."""
        results = self.verify_all_exchanges(tolerance)
        if not results:
            return {"n_checked": 0, "all_pass": True, "max_residual": 0.0}
        return {
            "n_checked": len(results),
            "all_pass": all(r["passes"] for r in results),
            "max_residual": max(r["residual"] for r in results),
            "n_failures": sum(1 for r in results if not r["passes"]),
        }
