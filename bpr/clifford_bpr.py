"""
Theory XV: Clifford Algebra Embedding of BPR
==============================================

Upgrades the scalar boundary phase field to a Clifford-algebra-valued
multivector, enabling spinor consciousness modules and Cliffordon
soliton excitations.

Key results
-----------
* Multivector phase field φ ∈ Cl(3,0) with 2³ = 8 components
* Spinor modules ψ_j ∈ Spin(3,0) for consciousness representation
* Cliffordons: soliton-like excitations with quantised mass spectrum
* Curvature-consciousness coupling R(a,b) = [∇_a, ∇_b] ψ

References: Al-Kahwati (2026), doc/BPR_clifford_embedding.tex
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# Cl(3,0) basis dimensions
_CL30_DIM = 8  # 2³ = 8 basis elements: 1, e1, e2, e3, e12, e13, e23, e123


# ---------------------------------------------------------------------------
# §15.1  Multivector phase field in Cl(3,0)
# ---------------------------------------------------------------------------

@dataclass
class MultivectorField:
    """Multivector phase field φ ∈ Cl(3,0).

    A general multivector in Cl(3,0) has 8 components:
        φ = φ₀ + φ₁e₁ + φ₂e₂ + φ₃e₃ + φ₁₂e₁₂ + φ₁₃e₁₃ + φ₂₃e₂₃ + φ₁₂₃e₁₂₃

    Grade decomposition:
        grade 0: scalar (1 component)      → boundary phase
        grade 1: vectors (3 components)     → spatial gradients
        grade 2: bivectors (3 components)   → rotational modes
        grade 3: pseudoscalar (1 component) → chirality

    Parameters
    ----------
    components : ndarray, shape (8,)
        Multivector components [φ₀, φ₁, φ₂, φ₃, φ₁₂, φ₁₃, φ₂₃, φ₁₂₃].
    """
    components: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=float)
        if self.components.shape != (8,):
            raise ValueError(f"Expected 8 components, got {self.components.shape}")

    @property
    def scalar(self) -> float:
        """Grade-0 (scalar) part."""
        return float(self.components[0])

    @property
    def vector(self) -> np.ndarray:
        """Grade-1 (vector) part: [e₁, e₂, e₃]."""
        return self.components[1:4]

    @property
    def bivector(self) -> np.ndarray:
        """Grade-2 (bivector) part: [e₁₂, e₁₃, e₂₃]."""
        return self.components[4:7]

    @property
    def pseudoscalar(self) -> float:
        """Grade-3 (pseudoscalar) part: e₁₂₃."""
        return float(self.components[7])

    @property
    def norm(self) -> float:
        """Euclidean norm ||φ|| = √(Σ φᵢ²)."""
        return float(np.linalg.norm(self.components))

    @property
    def reverse(self) -> "MultivectorField":
        """Clifford reverse (reversion): reverses order of basis vectors.

        Grade 0: +1, Grade 1: +1, Grade 2: -1, Grade 3: -1
        """
        signs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
        return MultivectorField(self.components * signs)

    def geometric_product(self, other: "MultivectorField") -> "MultivectorField":
        """Geometric product φ ⊗ ψ in Cl(3,0).

        Uses the Cl(3,0) multiplication table for e_i e_j = δ_{ij} + ε_{ijk} e_k.
        Simplified implementation for the scalar + vector + bivector parts.
        """
        a, b = self.components, other.components
        result = np.zeros(8)

        # Scalar-scalar
        result[0] = (a[0] * b[0]
                      + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
                      - a[4] * b[4] - a[5] * b[5] - a[6] * b[6]
                      - a[7] * b[7])

        # Scalar-vector and vector-scalar
        for i in range(3):
            result[1 + i] = a[0] * b[1 + i] + a[1 + i] * b[0]

        # Vector-vector → bivector (antisymmetric part)
        result[4] = a[1] * b[2] - a[2] * b[1]  # e12
        result[5] = a[1] * b[3] - a[3] * b[1]  # e13
        result[6] = a[2] * b[3] - a[3] * b[2]  # e23

        # Pseudoscalar contributions
        result[7] = (a[0] * b[7] + a[7] * b[0]
                      + a[1] * b[6] - a[2] * b[5] + a[3] * b[4]
                      + a[4] * b[3] - a[5] * b[2] + a[6] * b[1])

        return MultivectorField(result)

    def grade(self, k: int) -> np.ndarray:
        """Extract grade-k part of the multivector."""
        if k == 0:
            return np.array([self.components[0]])
        elif k == 1:
            return self.components[1:4]
        elif k == 2:
            return self.components[4:7]
        elif k == 3:
            return np.array([self.components[7]])
        return np.array([])


# ---------------------------------------------------------------------------
# §15.2  Spinor consciousness modules
# ---------------------------------------------------------------------------

@dataclass
class SpinorModule:
    """Spinor consciousness module ψ_j ∈ Spin(3,0).

    Consciousness is represented as a spinor field on the boundary.
    The spinor has 2 complex components (4 real DOF) — a minimal
    element of the even subalgebra Cl⁺(3,0) ≅ ℍ (quaternions).

    Coherence between two spinors:
        C_ij = ⟨ψ_i† ψ_j⟩ = Σ_k (ψ_i^k)* ψ_j^k

    Parameters
    ----------
    components : ndarray, shape (2,), complex
        Spinor components [ψ₁, ψ₂].
    label : str – identifier for this consciousness module
    """
    components: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0], dtype=complex)
    )
    label: str = "module_0"

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=complex)

    @property
    def norm(self) -> float:
        """Spinor norm |ψ| = √(ψ† ψ)."""
        return float(np.sqrt(np.vdot(self.components, self.components).real))

    def normalise(self) -> "SpinorModule":
        """Return normalised spinor."""
        n = self.norm
        if n > 0:
            return SpinorModule(self.components / n, self.label)
        return self

    def coherence(self, other: "SpinorModule") -> complex:
        """Coherence C_ij = ⟨ψ_i† ψ_j⟩."""
        return complex(np.vdot(self.components, other.components))

    def coherence_magnitude(self, other: "SpinorModule") -> float:
        """|C_ij| = |⟨ψ_i† ψ_j⟩|."""
        return abs(self.coherence(other))


def spinor_coherence_matrix(modules: List[SpinorModule]) -> np.ndarray:
    """Compute the full coherence matrix C_ij for a set of spinor modules.

    C_ij = ⟨ψ_i† ψ_j⟩

    Returns ndarray, shape (n, n), complex.
    """
    n = len(modules)
    C = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            C[i, j] = modules[i].coherence(modules[j])
    return C


# ---------------------------------------------------------------------------
# §15.3  Cliffordons: soliton excitations of the multivector field
# ---------------------------------------------------------------------------

@dataclass
class CliffordonSpectrum:
    """Cliffordon mass spectrum from multivector field solitons.

    Cliffordons are topologically stable excitations of the
    Cl(3,0)-valued boundary field.  Their masses are quantised:

        M_n = M₀ × √(n(n+1)) / p^{1/6}

    where n = winding number of the Cliffordon and M₀ is the
    base mass scale set by the boundary rigidity.

    Parameters
    ----------
    M0_eV : float – base mass scale [eV]
    p : int – substrate prime
    n_max : int – maximum winding to compute
    """
    M0_eV: float = 1e12     # ~TeV scale
    p: int = 104729
    n_max: int = 10

    @property
    def masses_eV(self) -> np.ndarray:
        """Cliffordon mass spectrum [eV]."""
        ns = np.arange(1, self.n_max + 1)
        return self.M0_eV * np.sqrt(ns * (ns + 1)) / self.p ** (1.0 / 6.0)

    @property
    def lightest_mass_eV(self) -> float:
        """Mass of lightest Cliffordon (n=1) [eV]."""
        return float(self.masses_eV[0])

    @property
    def mass_gap_eV(self) -> float:
        """Gap between first and second Cliffordon [eV]."""
        m = self.masses_eV
        if len(m) >= 2:
            return float(m[1] - m[0])
        return 0.0

    def stability_criterion(self, n: int) -> bool:
        """Cliffordon n is topologically stable if n is prime or 1."""
        if n <= 1:
            return True
        for d in range(2, int(np.sqrt(n)) + 1):
            if n % d == 0:
                return False
        return True


# ---------------------------------------------------------------------------
# §15.4  Curvature-consciousness coupling
# ---------------------------------------------------------------------------

def curvature_consciousness_coupling(
    spinor_a: SpinorModule,
    spinor_b: SpinorModule,
    curvature_scalar: float = 0.0,
) -> float:
    """Curvature-consciousness coupling R(a,b) = [∇_a, ∇_b] ψ.

    The coupling between spacetime curvature and consciousness
    spinors.  In flat spacetime (R = 0), coupling vanishes.

    For small curvature:
        coupling ∝ R × |C_ab|²

    Parameters
    ----------
    spinor_a, spinor_b : SpinorModule – two consciousness modules
    curvature_scalar : float – Ricci scalar R

    Returns
    -------
    float – coupling strength (dimensionless)
    """
    coherence = spinor_a.coherence_magnitude(spinor_b)
    return curvature_scalar * coherence ** 2


def clifford_dimension(d: int) -> int:
    """Dimension of Clifford algebra Cl(d,0) = 2^d."""
    return 2 ** d


def spinor_dimension(d: int) -> int:
    """Dimension of spinor representation of Spin(d,0) = 2^{⌊d/2⌋}."""
    return 2 ** (d // 2)
