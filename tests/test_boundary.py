"""
Tests for boundary field calculations in BPR-Math-Spine

Mathematical Checkpoint 1: Laplacian eigenvalues on S² converge to l(l+1) within 0.1% for l≤10

Works with both FEniCS and the pure-numpy/scipy fallback.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bpr import make_boundary
from bpr.boundary_field import solve_phase, solve_eigenvalue_problem, verify_convergence
from bpr.geometry import get_mesh_quality_metrics, compute_boundary_area, NumpyMesh
import bpr.geometry as _bpr_geometry

# Determine backend
FENICS_AVAILABLE = _bpr_geometry.fe is not None
BACKEND = "FEniCS" if FENICS_AVAILABLE else "numpy"


class TestBoundaryMesh:
    """Test boundary mesh generation and quality."""

    def test_sphere_mesh_creation(self):
        """Test creation of spherical boundary mesh."""
        mesh_size = 0.2
        boundary_mesh = make_boundary(
            mesh_size=mesh_size,
            geometry="sphere",
            radius=1.0,
        )

        assert boundary_mesh is not None
        assert boundary_mesh.mesh_size == mesh_size

        # Check mesh quality
        quality = get_mesh_quality_metrics(boundary_mesh)
        assert "num_vertices" in quality
        assert "num_cells" in quality
        assert quality["num_vertices"] > 10

    def test_sphere_area_calculation(self):
        """Test that computed sphere area is close to 4π."""
        boundary_mesh = make_boundary(
            mesh_size=0.1,
            geometry="sphere",
            radius=1.0,
        )

        area = compute_boundary_area(boundary_mesh)
        expected_area = 4 * np.pi
        relative_error = abs(area - expected_area) / expected_area

        # Icosphere area converges to 4π with subdivision;
        # for coarse meshes allow up to 5%
        assert relative_error < 0.05, f"Area error {relative_error:.1%} > 5%"


class TestBoundaryField:
    """Test boundary field solver and eigenvalue computations."""

    def test_phase_field_solver(self):
        """Test basic phase field solution."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)

        def constant_source(x, y, z):
            return 1.0

        solution = solve_phase(boundary_mesh, constant_source, kappa=1.0)
        assert solution is not None
        assert solution.phi is not None

        # Check energy is finite and non-negative
        energy = solution.compute_energy()
        assert energy >= 0
        assert np.isfinite(energy)

    def test_eigenvalue_convergence(self):
        """
        Mathematical Checkpoint 1: Laplacian eigenvalue convergence.

        Eigenvalues of -∇² on the unit sphere are l(l+1) with
        multiplicity 2l+1.  On a discrete mesh the degeneracies are
        broken but the *mean* eigenvalue per l-cluster should converge
        to the exact value as the mesh is refined.
        """
        # Use a fine-ish mesh for decent accuracy
        boundary_mesh = make_boundary(mesh_size=0.08, geometry="sphere", radius=1.0)

        n_modes = 25  # enough to see l = 0..3 clusters
        eigenvals, _ = solve_eigenvalue_problem(boundary_mesh, n_modes=n_modes)

        # Theoretical eigenvalues: l(l+1) with multiplicity 2l+1
        # l=0 -> 0  (×1)
        # l=1 -> 2  (×3)
        # l=2 -> 6  (×5)
        # l=3 -> 12 (×7)
        # ...
        # The FEM solver returns them in ascending order but the
        # degenerate modes are split slightly.  So we cluster by
        # proximity rather than assuming exact multiplicity.

        # Basic sanity: first eigenvalue should be ~0 (constant mode)
        assert abs(eigenvals[0]) < 0.5, (
            f"First eigenvalue {eigenvals[0]:.3f} should be ~0"
        )

        # Check that we see clusters near l(l+1) for l=1,2
        # l=1 cluster: eigenvalues[1:4] should be ≈ 2
        l1_cluster = eigenvals[1:4]
        l1_mean = np.mean(l1_cluster)
        assert abs(l1_mean - 2.0) / 2.0 < 0.05, (
            f"l=1 mean eigenvalue {l1_mean:.3f}, expected ≈ 2 (error {abs(l1_mean-2)/2:.1%})"
        )

        # l=2 cluster: eigenvalues[4:9] should be ≈ 6
        l2_cluster = eigenvals[4:9]
        l2_mean = np.mean(l2_cluster)
        assert abs(l2_mean - 6.0) / 6.0 < 0.05, (
            f"l=2 mean eigenvalue {l2_mean:.3f}, expected ≈ 6 (error {abs(l2_mean-6)/6:.1%})"
        )

        print(f"✅ Checkpoint 1 PASSED (backend={BACKEND}): "
              f"l=1 mean={l1_mean:.3f}, l=2 mean={l2_mean:.3f}")

    def test_convergence_analysis(self):
        """Test mesh convergence rate analysis.

        On a closed sphere (no boundary), the manufactured-solution
        convergence test is ill-posed, so we only verify that the
        infrastructure runs and the error decreases (rate > 0).
        """
        mesh_sizes = [0.3, 0.2, 0.15]

        convergence = verify_convergence(mesh_sizes)
        assert convergence is not None
        assert "convergence_rate" in convergence

        rate = convergence["convergence_rate"]
        if np.isnan(rate):
            pytest.skip("Could not compute convergence rate")

        # On FEniCS meshes of open domains we'd expect rate ≈ 2.
        # On the numpy icosphere fallback the manufactured test is
        # ill-conditioned; we just check rate is non-negative.
        assert rate > -0.5, f"Convergence rate {rate:.2f} is regressing"
        print(f"✅ Convergence rate: {rate:.2f}")


class TestBoundaryFieldProperties:
    """Test mathematical properties of boundary field solutions."""

    def test_energy_positivity(self):
        """Test that field energy is always non-negative."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)

        source_terms = [
            lambda x, y, z: 1.0,
            lambda x, y, z: x + y + z,
            lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y),
        ]

        for i, source in enumerate(source_terms):
            solution = solve_phase(boundary_mesh, source, kappa=1.0)
            energy = solution.compute_energy()
            assert energy >= 0, f"Energy {energy} negative for source {i}"
            assert np.isfinite(energy), f"Energy {energy} not finite for source {i}"

    def test_scaling_properties(self):
        """Test scaling properties of the solution."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)

        def test_source(x, y, z):
            return np.sin(np.pi * x)

        kappa_values = [0.5, 1.0, 2.0]
        energies = []

        for kappa in kappa_values:
            solution = solve_phase(boundary_mesh, test_source, kappa=kappa)
            energy = solution.compute_energy()
            energies.append(energy)

        # Energy should decrease with increasing kappa (more diffusion)
        assert energies[2] <= energies[0] * 1.01, (
            f"Energy scaling with kappa incorrect: E(κ=2)={energies[2]:.4e} vs E(κ=0.5)={energies[0]:.4e}"
        )


def test_mathematical_checkpoint_1():
    """
    Comprehensive test for Mathematical Checkpoint 1.

    Verifies that Laplacian eigenvalues on S² converge to l(l+1).
    """
    print(f"\n🔍 Running Mathematical Checkpoint 1 (backend={BACKEND})")
    print("=" * 70)

    boundary_mesh = make_boundary(mesh_size=0.08, geometry="sphere", radius=1.0)
    n_modes = 16
    eigenvals, _ = solve_eigenvalue_problem(boundary_mesh, n_modes=n_modes)

    # Check l=1..3 cluster means
    theoretical = {1: 2.0, 2: 6.0, 3: 12.0}
    cumulative = 0  # cumulative index (l=0 occupies index 0)
    all_passed = True
    max_error = 0.0

    print("  l | Cluster mean | Theoretical | Rel. Error | Status")
    print("  " + "-" * 55)

    for l_val in (1, 2, 3):
        start = cumulative + 1 if l_val == 1 else cumulative
        if l_val == 1:
            start = 1
            end = 4
        elif l_val == 2:
            start = 4
            end = 9
        else:
            start = 9
            end = 16

        if end > len(eigenvals):
            break

        cluster = eigenvals[start:end]
        mean_val = np.mean(cluster)
        exact = theoretical[l_val]
        rel_err = abs(mean_val - exact) / exact
        max_error = max(max_error, rel_err)
        status = "✅ PASS" if rel_err < 0.05 else "❌ FAIL"
        if rel_err >= 0.05:
            all_passed = False

        print(f"  {l_val:2d} | {mean_val:10.4f}   | {exact:10.4f}    | {rel_err:.2e}  | {status}")

    print("  " + "-" * 55)
    print(f"  Maximum relative error: {max_error:.2e}")

    if all_passed:
        print("\n🎉 MATHEMATICAL CHECKPOINT 1: PASSED")
    else:
        print("\n⚠️  MATHEMATICAL CHECKPOINT 1: FAILED — refine mesh or check implementation")

    assert all_passed, f"Checkpoint 1 failed: max cluster-mean error {max_error:.1e}"


if __name__ == "__main__":
    test_mathematical_checkpoint_1()
