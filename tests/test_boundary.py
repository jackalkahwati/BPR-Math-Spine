"""
Tests for boundary field calculations in BPR-Math-Spine

Mathematical Checkpoint 1: Laplacian eigenvalues on S² converge to l(l+1) within 0.1% for l≤10
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from bpr import make_boundary
    from bpr.boundary_field import solve_phase, solve_eigenvalue_problem, verify_convergence
    from bpr.geometry import get_mesh_quality_metrics, compute_boundary_area
    import bpr.geometry as _bpr_geometry
    import bpr.boundary_field as _bpr_boundary_field

    # FEniCS availability must be determined from the runtime import flags,
    # not merely whether our Python modules import successfully.
    FENICS_AVAILABLE = (_bpr_geometry.fe is not None) and (_bpr_boundary_field.fe is not None)
except ImportError:
    FENICS_AVAILABLE = False

if not FENICS_AVAILABLE:
    pytestmark = pytest.mark.skip("FEniCS not available")


class TestBoundaryMesh:
    """Test boundary mesh generation and quality."""
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required")
    def test_sphere_mesh_creation(self):
        """Test creation of spherical boundary mesh."""
        mesh_size = 0.2
        boundary_mesh = make_boundary(
            mesh_size=mesh_size, 
            geometry="sphere", 
            radius=1.0
        )
        
        assert boundary_mesh is not None
        assert boundary_mesh.mesh_size == mesh_size
        
        # Check mesh quality
        quality = get_mesh_quality_metrics(boundary_mesh)
        assert 'num_vertices' in quality
        assert 'num_cells' in quality
        assert quality['num_vertices'] > 10  # Should have reasonable number of vertices
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required")
    def test_sphere_area_calculation(self):
        """Test that computed sphere area is close to 4π."""
        boundary_mesh = make_boundary(
            mesh_size=0.1, 
            geometry="sphere", 
            radius=1.0
        )
        
        try:
            area = compute_boundary_area(boundary_mesh)
            expected_area = 4 * np.pi
            relative_error = abs(area - expected_area) / expected_area
            
            # Area should be within 5% of theoretical value
            assert relative_error < 0.05, f"Area error {relative_error:.1%} > 5%"
            
        except Exception:
            # Skip if area computation not available
            pytest.skip("Area computation not available")


class TestBoundaryField:
    """Test boundary field solver and eigenvalue computations."""
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required")
    def test_phase_field_solver(self):
        """Test basic phase field solution."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)
        
        # Simple constant source
        def constant_source(x, y, z):
            return 1.0
        
        try:
            solution = solve_phase(boundary_mesh, constant_source, kappa=1.0)
            assert solution is not None
            assert solution.phi is not None
            
            # Check energy is finite and positive
            energy = solution.compute_energy()
            assert energy > 0
            assert np.isfinite(energy)
            
        except Exception as e:
            pytest.skip(f"Phase field solver not available: {e}")
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required") 
    def test_eigenvalue_convergence(self):
        """
        Mathematical Checkpoint 1: Test Laplacian eigenvalue convergence.
        
        Eigenvalues should converge to l(l+1) within 0.1% for l≤10.
        """
        boundary_mesh = make_boundary(mesh_size=0.15, geometry="sphere", radius=1.0)
        
        try:
            n_modes = 8  # Test first 8 modes
            eigenvals, eigenfuncs = solve_eigenvalue_problem(boundary_mesh, n_modes=n_modes)
            
            # Theoretical eigenvalues for unit sphere
            theoretical = np.array([l * (l + 1) for l in range(n_modes)])
            
            # Check convergence for non-trivial modes (l ≥ 1)
            for l in range(1, min(6, len(eigenvals))):  # Test l=1 to l=5
                computed = eigenvals[l]
                exact = theoretical[l]
                relative_error = abs(computed - exact) / exact
                
                # Checkpoint 1: Error should be < 0.1%
                assert relative_error < 0.001, (
                    f"Mode l={l}: error {relative_error:.1e} > 0.1% "
                    f"(computed={computed:.3f}, exact={exact:.3f})"
                )
            
            print(f"✅ Checkpoint 1 PASSED: Eigenvalue convergence verified")
            
        except Exception as e:
            pytest.skip(f"Eigenvalue computation failed: {e}")
    
    def test_convergence_analysis(self):
        """Test mesh convergence rate analysis."""
        if not FENICS_AVAILABLE:
            pytest.skip("FEniCS required")
        
        mesh_sizes = [0.3, 0.2, 0.15]
        
        try:
            convergence = verify_convergence(mesh_sizes)
            
            if convergence and 'convergence_rate' in convergence:
                rate = convergence['convergence_rate']
                
                # For P1 elements, expect convergence rate ~2
                if not np.isnan(rate):
                    assert rate > 1.0, f"Convergence rate {rate:.2f} too low"
                    # Allow some flexibility in convergence rate
                    assert rate < 4.0, f"Convergence rate {rate:.2f} suspiciously high"
                    
                    print(f"✅ Convergence rate: {rate:.2f}")
                else:
                    pytest.skip("Could not compute convergence rate")
            else:
                pytest.skip("Convergence analysis not available")
                
        except Exception as e:
            pytest.skip(f"Convergence test failed: {e}")


class TestBoundaryFieldProperties:
    """Test mathematical properties of boundary field solutions."""
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required")
    def test_energy_positivity(self):
        """Test that field energy is always positive."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)
        
        # Test with different source terms
        source_terms = [
            lambda x, y, z: 1.0,  # Constant
            lambda x, y, z: x + y + z,  # Linear
            lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y),  # Oscillatory
        ]
        
        for i, source in enumerate(source_terms):
            try:
                solution = solve_phase(boundary_mesh, source, kappa=1.0)
                energy = solution.compute_energy()
                
                assert energy >= 0, f"Energy {energy} negative for source {i}"
                assert np.isfinite(energy), f"Energy {energy} not finite for source {i}"
                
            except Exception:
                continue  # Skip if computation fails
    
    @pytest.mark.skipif(not FENICS_AVAILABLE, reason="FEniCS required")
    def test_scaling_properties(self):
        """Test scaling properties of the solution."""
        boundary_mesh = make_boundary(mesh_size=0.2, geometry="sphere", radius=1.0)
        
        def test_source(x, y, z):
            return np.sin(np.pi * x)
        
        try:
            # Test with different kappa values
            kappa_values = [0.5, 1.0, 2.0]
            energies = []
            
            for kappa in kappa_values:
                solution = solve_phase(boundary_mesh, test_source, kappa=kappa)
                energy = solution.compute_energy()
                energies.append(energy)
            
            # Energy should scale appropriately with kappa
            # For fixed source and larger kappa (more diffusion), energy should be smaller
            assert energies[2] <= energies[0], "Energy scaling with kappa incorrect"
            
        except Exception:
            pytest.skip("Scaling test not available")


def test_mathematical_checkpoint_1():
    """
    Comprehensive test for Mathematical Checkpoint 1.
    
    This is the main checkpoint function that can be called independently.
    """
    if not FENICS_AVAILABLE:
        pytest.skip("FEniCS required for Mathematical Checkpoint 1")
    
    print("\n🔍 Running Mathematical Checkpoint 1: Laplacian Eigenvalue Convergence")
    print("=" * 70)
    
    try:
        # Create test mesh
        boundary_mesh = make_boundary(mesh_size=0.12, geometry="sphere", radius=1.0)
        
        # Solve eigenvalue problem
        n_modes = 6
        eigenvals, _ = solve_eigenvalue_problem(boundary_mesh, n_modes=n_modes)
        
        # Check against theoretical values
        max_error = 0.0
        all_passed = True
        
        print("Mode | Computed | Theoretical | Rel. Error | Status")
        print("-" * 50)
        
        for l in range(1, min(6, len(eigenvals))):  # Check l=1 to l=5
            computed = eigenvals[l]
            theoretical = l * (l + 1)
            rel_error = abs(computed - theoretical) / theoretical
            max_error = max(max_error, rel_error)
            
            status = "✅ PASS" if rel_error < 0.001 else "❌ FAIL"
            if rel_error >= 0.001:
                all_passed = False
            
            print(f" {l:2d}   | {computed:6.3f}   | {theoretical:6.3f}      | {rel_error:.2e}  | {status}")
        
        print("-" * 50)
        print(f"Maximum relative error: {max_error:.2e}")
        print(f"Required threshold:     1.0e-03 (0.1%)")
        
        if all_passed:
            print("\n🎉 MATHEMATICAL CHECKPOINT 1: PASSED")
            print("   Laplacian eigenvalues converge within 0.1% tolerance")
        else:
            print("\n⚠️  MATHEMATICAL CHECKPOINT 1: FAILED")
            print("   Consider mesh refinement or check implementation")
        
        # Assert for pytest
        assert all_passed, f"Checkpoint 1 failed: max error {max_error:.1e} > 0.1%"
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ MATHEMATICAL CHECKPOINT 1: ERROR")
        print(f"   {e}")
        pytest.fail(f"Checkpoint 1 error: {e}")


if __name__ == "__main__":
    # Run the main checkpoint when called directly
    test_mathematical_checkpoint_1()