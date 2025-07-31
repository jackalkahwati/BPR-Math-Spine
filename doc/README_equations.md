# BPR Mathematical Equations Reference

This document lists the key equations implemented in BPR-Math-Spine, referencing the one-page synopsis.

## Core Equations

### Equation (3): Metric Perturbation
```
Δg_μν = λ φ(x_∂) [geometric coupling terms]
```
- **File**: `bpr/metric.py` 
- **Function**: `metric_perturbation()`
- **Description**: Coupling between boundary phase field and spacetime metric

### Equation (6a): Boundary Laplacian
```
κ ∇²_Σ φ = f
```
- **File**: `bpr/boundary_field.py`
- **Function**: `solve_phase()`
- **Description**: Phase field equation on boundary surface Σ

### Equation (6b): Geometric Coupling
```
[Specific form of boundary-metric coupling]
```
- **File**: `bpr/metric.py`
- **Function**: `_compute_delta_g_matrix()`
- **Description**: Geometric coupling terms in metric perturbation

### Equation (7): BPR-Casimir Prediction
```
F_total = F_Casimir + ΔF_BPR(φ, λ, R)
```
- **File**: `bpr/casimir.py`
- **Function**: `casimir_force()`
- **Description**: **Main falsifiable prediction** - Casimir force with BPR corrections

## Mathematical Checkpoints

### Checkpoint 1: Laplacian Eigenvalues
- **Criterion**: λ_l converges to l(l+1) within 0.1% for l≤10
- **Test**: `tests/test_boundary.py::test_mathematical_checkpoint_1`
- **Purpose**: Verify boundary field solver accuracy

### Checkpoint 2: Energy-Momentum Conservation
- **Criterion**: ∇^μ T^φ_μν = 0 to tolerance 1e-8
- **Test**: `tests/test_metric.py::test_mathematical_checkpoint_2`
- **Purpose**: Verify stress tensor calculations

### Checkpoint 3: Casimir Recovery
- **Criterion**: F_total → F_Casimir as λ → 0
- **Test**: `tests/test_casimir.py::test_mathematical_checkpoint_3`
- **Purpose**: Verify theory reduces to standard QED

## Implementation Notes

1. **Boundary Surface**: Unit sphere (R=1) is default geometry
2. **Coordinate Systems**: Cartesian, spherical, cylindrical supported
3. **Mesh Resolution**: Adaptive based on radius scale
4. **Numerical Methods**: FEniCS for FEM, SymPy for symbolic math

## Physical Constants

```python
# From bpr/casimir.py
EPSILON_0 = 8.854187817e-12  # F/m
MU_0 = 4e-7 * pi             # H/m  
CASIMIR_PREFACTOR = pi**2 * hbar * c / 240
```

## Key Dimensionless Parameters

- **Coupling strength**: λ ~ 1e-3 (typical)
- **Mesh size**: h ~ 0.1 (default)
- **Radius range**: 0.2-5.0 μm (for experiments)

## Verification Strategy

1. **Analytical**: Compare with known solutions (sphere eigenvalues)
2. **Numerical**: Convergence tests with mesh refinement
3. **Physical**: Check conservation laws and limiting behavior
4. **Computational**: Cross-validation with independent implementations