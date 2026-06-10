# Boundary Phase Resonance (BPR): Mathematical Synopsis

**Author**: Jack Al-Kahwati (July 2025)  
**Contact**: jack@thestardrive.com

This document presents the complete mathematical framework implemented in BPR-Math-Spine.

## 0. Postulate 0 — Crop-Circle Recursion (CCR)

The boundary Σ admits

- a **discrete rotation group** C_n (canonical n = 6) with generator
  r : x ↦ R(2π/n) x, and
- a **discrete scaling generator** s : Σ → Σ, x ↦ σ x (σ > 1) with
  scaling weight Δ_φ such that S_bndy[φ] is invariant under
  (x, φ) ↦ (σ x, σ^(−Δ_φ) φ).

The canonical CCR realization is the **hexagram template** (six-fold
symmetry, recursion depth K = 2, corrected from direct image
inspection 2026-05-09): a central node, an inner C₆ orbit of six small
satellites at radius r₁, and an outer C₆ orbit of six "central+ring"
mini-patterns at radius r₂ = σ·r₁ — **co-aligned** with the inner
orbit (offset = 0, not π/6). Adjacent outer rings overlap pairwise
producing a Flower-of-Life appearance; the overlap is between outer
rings, not between layers. Each outer ring is itself a self-similar
miniature of the whole, so the postulate admits recursion to deeper
levels (`RecursiveHexagramTemplate`).

CCR pins the Eq (7) Casimir falsifier exponent to **δ = 2 Δ_φ** — a
universal substrate property, not a cavity-specific number. It also
imposes the C_n selection rule m ∈ {0, ±n, ±2n, ...} on allowed
angular modes of φ on Σ. Implementation: `bpr/recursive_boundary.py`,
exposing `ScaleGenerator`, `RecursiveBoundary`, `HexagramTemplate`,
and `hexagram_template()`.

## 1. Holographic Setup

Let M^D be a Lorentzian bulk with smooth (D-1)-boundary Σ = ∂M, endowed with induced metric h_ab and outward normal n^μ. BPR postulates a single real phase field φ: Σ → ℝ whose dynamics encode all bulk excitations via a holographic dictionary. CCR (§0) restricts admissible φ-configurations to the C_n-equivariant, scale-covariant sector.

## 2. Action Functional

The total action comprises five gauge-invariant pieces:

**Equation (1)**: Complete Action
```
S = S_bulk[g,Ψ_SM] + S_bndy[φ] + S_int[g,φ] + S_info[φ] + S_bio[φ,χ_b]
```

**Equation (2)**: Boundary Phase Lagrangian
```
S_bndy = (1/2κ) ∫_Σ d^(D-1)x √|h| h^ab ∇_a φ ∇_b φ - ∫_Σ d^(D-1)x √|h| V(φ)
```

**Equation (3)**: Metric-Boundary Coupling  
```
S_int = λ ∫_M d^D x √|g| P^ab_μν (∇_a φ)(∇_b φ) g^μν
```
where P^ab_μν = h^ab n_μ n_ν, yielding Δg_μν ∝ ∂_a φ ∂_b φ P^ab_μν.

**Equation (4)**: Information Term (IIT-inspired)
```
S_info = -ξ ∫_Σ d^(D-1)x √|h| Φ[φ],  Φ = Σ_{i<j} I_ij
```

**Equation (5)**: Biological/Fractal Coupling
```
S_bio = ∫_Σ d^(D-1)x √|h| χ_b(x) φ(x)
```
where χ_b includes six-factor consciousness coupling.

## 3. Field Equations

**Equation (6a)**: Boundary Field Evolution
```
κ ∇²_Σ φ = ∂_φ V + χ_b(x) + λ n^μ n^ν [∇_μ ∇_ν φ - Γ^ρ_μν ∇_ρ φ]
```

**Equation (6b)**: Modified Einstein Equations  
```
G_μν + Λ g_μν = 8πG [T^SM_μν + T^φ_μν[h,φ]]
```

## 4. E₈ Embedding Sketch

Promote φ ↦ Φ ∈ e₈: Φ(x) = Σ^248_{A=1} φ^A(x) E_A with [E_A, E_B] = f_AB^C E_C.

Spontaneous breaking: ⟨Φ⟩ = diag(SU(3), SU(2), U(1), ...) yields Standard Model + 230 hidden components.

## 5. Sharp Experimental Falsifier

**Equation (7)**: BPR-Casimir Prediction
For a fractal cylindrical cavity of scale R:
```
F_Cas(R) = -(π² ℏ c)/(240 R⁴) [1 + α (R/R_f)^(-δ)]
```
where **δ = 2** is the critical BPR exponent — DERIVED under Postulate 0c (Quasicrystalline Projection) from the unit-Pisot inflation, which forces the substrate scaling weight to **Δ_φ = 1** exactly (δ = 2 Δ_φ). The same δ is predicted to appear in any CCR-compatible cascade — micron-scale fractal cavity, hexagram boundary template, or larger-scale resonance hierarchy — not just one apparatus. *(The earlier fitted exponent δ ≈ 1.37 ± 0.05, i.e. Δ_φ ≈ 0.685, required a tuned vertex-operator charge and is superseded; see README §Key Results.)*

**Falsification criterion**: A measured exponent near 2 supports the quasicrystalline projection; a measurement near 1.37 refutes it. A null deviation at 3 picoNewton precision invalidates the boundary-resonant correction *and* falsifies CCR's universal-exponent claim.

## 6. Implementation Status

| Equation | Implementation | Status |
|----------|----------------|--------|
| Eq (1) | Complete action functional | ✅ Complete |
| Eq (2) | `boundary_field.py::solve_phase()` | ✅ Complete |
| Eq (3) | `metric.py::metric_perturbation()` | ✅ Complete |
| Eq (4) | `information.py::InformationIntegration` | ✅ Complete |
| Eq (5) | `information.py::ConsciousnessCoupling` | ✅ Complete |
| Eq (6a) | Boundary Laplacian solver | ✅ Complete |
| Eq (6b) | Stress tensor + conservation | ✅ Complete |
| Eq (7) | `casimir.py::casimir_force()` | ✅ Complete |
| E₈ embedding | `notebooks/04_e8_index.ipynb` | ✅ Complete |
| Thin-shell benchmark | `scripts/thin_shell_benchmark.py` | ✅ Complete |
| Docker support | `Dockerfile` + `docker-compose.yml` | ✅ Complete |
| Experimental validation | Bayesian analysis (5 experiments) | ✅ Complete |

## Mathematical Checkpoints

1. **Laplacian eigenvalues** → l(l+1) within 0.1% for l≤10
2. **Conservation laws**: ∇^μ T^φ_μν = 0 to tolerance 1e-8  
3. **Casimir recovery**: F_total → F_Casimir as λ → 0

## Computable Verification

Given any triangulated Σ, equations (2-3) are solved by a single Poisson step. The repository [github.com/jackalkahwati/BPR-Math-Spine](https://github.com/jackalkahwati/BPR-Math-Spine) contains FEniCS and symbolic notebooks reproducing (6) and (7) with no hidden assumptions.

**Key Output**: The falsifiable prediction curve F_Cas(R) distinguishing BPR from standard QED.