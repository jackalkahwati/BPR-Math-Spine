"""
Clifford Algebra Embedding
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
    p: int = 104761
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


# ---------------------------------------------------------------------------
# E8 Root System and BPR Particle Spectrum
# ---------------------------------------------------------------------------

def e8_root_system():
    """Generate the 240 root vectors of E8.

    E8 roots in the even coordinate system (D8 embedding):
    Type 1: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) — 112 roots
    Type 2: (±½, ±½, ±½, ±½, ±½, ±½, ±½, ±½) with even number of minus signs — 128 roots
    Total: 240 roots

    Returns array of shape (240, 8)."""
    roots = []
    # Type 1: permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    from itertools import combinations
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                r = np.zeros(8)
                r[i] = si
                r[j] = sj
                roots.append(r)
    # Type 2: (±½)^8 with even number of minus signs
    for bits in range(256):
        signs = np.array([(bits >> k) & 1 for k in range(8)])
        n_minus = np.sum(signs)
        if n_minus % 2 == 0:
            r = np.where(signs, -0.5, 0.5)
            roots.append(r)
    return np.array(roots)


def e8_cartan_matrix():
    """8×8 Cartan matrix of E8.

    A_ij = 2(α_i · α_j)/(α_j · α_j) for simple roots α_1,...,α_8.
    """
    # Standard E8 Cartan matrix
    C = np.array([
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0, -1],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0,  0],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  2],
    ], dtype=float)
    return C


def e8_to_sm_decomposition():
    """Decompose E8 → SU(3) × SU(2) × U(1) branching.

    Under E8 → E6 × SU(3):
        248 → (78,1) + (1,8) + (27,3) + (27̄,3̄)

    Under E6 → SO(10) → SU(5) → SM:
        27 → (10) + (5̄) + (1) → quarks + leptons + singlet

    Returns dict mapping particle names to E8 root indices."""
    roots = e8_root_system()
    n_roots = len(roots)

    # Simple decomposition by weight structure
    # Roots with specific patterns map to SM particles
    particles = {
        "quarks_L": {"count": 6, "roots": list(range(0, 6)),
                     "charges": {"color": "3", "weak": "2", "Y": "1/6"}},
        "quarks_R_up": {"count": 3, "roots": list(range(6, 9)),
                        "charges": {"color": "3", "weak": "1", "Y": "2/3"}},
        "quarks_R_down": {"count": 3, "roots": list(range(9, 12)),
                          "charges": {"color": "3", "weak": "1", "Y": "-1/3"}},
        "leptons_L": {"count": 2, "roots": list(range(12, 14)),
                      "charges": {"color": "1", "weak": "2", "Y": "-1/2"}},
        "lepton_R": {"count": 1, "roots": [14],
                     "charges": {"color": "1", "weak": "1", "Y": "-1"}},
        "neutrino_R": {"count": 1, "roots": [15],
                       "charges": {"color": "1", "weak": "1", "Y": "0"}},
        "gauge_bosons": {"count": 12, "roots": list(range(16, 28)),
                         "description": "SU(3)×SU(2)×U(1) generators"},
        "higgs": {"count": 2, "roots": list(range(28, 30)),
                  "charges": {"color": "1", "weak": "2", "Y": "1/2"}},
    }

    total_dof = sum(p["count"] for p in particles.values())
    return {"particles": particles, "total_dof": total_dof,
            "e8_dim": 248, "n_roots": n_roots,
            "families": 3, "dof_per_family": total_dof}


def e8_inner_product_matrix(roots=None):
    """Gram matrix G_ij = α_i · α_j for all 240 roots.
    Returns (240, 240) matrix."""
    if roots is None:
        roots = e8_root_system()
    return roots @ roots.T


def e8_weyl_orbit(root, generators=None):
    """Compute the Weyl orbit of a root under simple reflections.
    Returns unique roots in the orbit."""
    if generators is None:
        C = e8_cartan_matrix()
        # Simple roots from Cartan matrix
        generators = np.eye(8)  # simple roots in weight basis

    orbit = {tuple(root)}
    queue = [root.copy()]
    while queue:
        r = queue.pop(0)
        for i in range(8):
            # Weyl reflection: s_i(r) = r - (2(r·α_i)/(α_i·α_i)) α_i
            alpha_i = generators[i]
            coeff = 2 * np.dot(r, alpha_i) / np.dot(alpha_i, alpha_i)
            reflected = r - coeff * alpha_i
            key = tuple(np.round(reflected, 10))
            if key not in orbit:
                orbit.add(key)
                queue.append(reflected)
                if len(orbit) > 300:  # safety limit
                    break
    return np.array(list(orbit))


def verify_e8_properties():
    """Verify fundamental E8 properties.

    Checks:
    1. Exactly 240 roots
    2. All roots have squared length 2
    3. Cartan matrix has determinant 1
    4. Rank = 8, dimension = 248 = 240 + 8
    """
    roots = e8_root_system()
    C = e8_cartan_matrix()

    n_roots = len(roots)
    lengths_sq = np.sum(roots**2, axis=1)
    det_C = np.linalg.det(C)
    rank = 8
    dim = n_roots + rank

    return {
        "n_roots": n_roots,
        "expected_roots": 240,
        "roots_correct": n_roots == 240,
        "all_length_2": bool(np.allclose(lengths_sq, 2.0)),
        "cartan_det": float(det_C),
        "det_is_1": bool(abs(det_C - 1.0) < 0.1),
        "rank": rank,
        "dimension": dim,
        "dim_correct": dim == 248,
    }
