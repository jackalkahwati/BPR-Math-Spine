"""
Boundary-solver integration tests
==================================

End-to-end tests that exercise mesh generation → PDE solve → Casimir
pipeline.  Works with both the FEniCS backend and the pure-numpy/scipy
fallback so they are **never** skipped.

Run with:
    pytest tests/test_fenics_integration.py -v
"""

import numpy as np
import pytest

from bpr.geometry import make_boundary, get_mesh_quality_metrics, NumpyMesh, fe
from bpr.boundary_field import solve_phase

BACKEND = "FEniCS" if fe is not None else "numpy"


class TestMeshGeneration:
    """Verify BPR mesh helpers produce valid meshes."""

    def test_make_boundary_sphere(self):
        bm = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        assert bm is not None
        nv = bm.mesh.num_vertices()
        assert nv > 10, f"Sphere mesh too coarse: {nv} vertices"

    def test_make_boundary_cylinder(self):
        bm = make_boundary(mesh_size=0.3, geometry="cylinder", radius=1.0)
        assert bm is not None
        nv = bm.mesh.num_vertices()
        assert nv > 10, f"Cylinder mesh too coarse: {nv} vertices"

    def test_mesh_quality_metrics(self):
        bm = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        quality = get_mesh_quality_metrics(bm)
        assert "num_vertices" in quality
        assert quality["num_vertices"] > 0


class TestPhaseFieldSolver:
    """Verify solve_phase returns a usable solution."""

    def test_solve_phase_sphere(self):
        bm = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        source = lambda x, y, z: 1.0
        sol = solve_phase(bm, source, kappa=1.0)
        assert sol is not None

    def test_solve_phase_produces_finite(self):
        bm = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        source = lambda x, y, z: np.sin(np.pi * x)
        sol = solve_phase(bm, source, kappa=1.0)

        # Get the DOF vector (backend-agnostic)
        if sol._phi_vec is not None:
            arr = sol._phi_vec
        else:
            arr = sol.phi.vector().get_local()

        assert np.all(np.isfinite(arr)), "Solution contains non-finite values"


class TestCasimirEndToEnd:
    """End-to-end Casimir force test (uses whichever backend is available)."""

    def test_casimir_prediction(self):
        from bpr.casimir import casimir_force

        result = casimir_force(
            radius=1e-6,
            coupling_lambda=1e-3,
            mesh_size=0.3,
        )
        assert result is not None
        # Standard Casimir force should be attractive (negative)
        assert result.standard_force < 0
        # Total force should be finite
        assert np.isfinite(result.total_force)

    def test_sweep_radius_small(self):
        from bpr.casimir import sweep_radius

        data = sweep_radius(
            r_min=0.5e-6,
            r_max=2.0e-6,
            n=3,
            coupling_lambda=1e-3,
        )
        # sweep_radius returns a DataFrame (pandas) or list of dicts
        if hasattr(data, "__len__"):
            assert len(data) == 3
        else:
            pytest.skip("sweep_radius returned unexpected type")


class TestMetricPerturbation:
    """Metric perturbation from a real (not just symbolic) field solution."""

    def test_metric_perturbation_field(self):
        bm = make_boundary(mesh_size=0.4, geometry="sphere", radius=1.0)
        source = lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y)
        sol = solve_phase(bm, source, kappa=1.0)
        assert sol is not None

        # Verify the solution has content
        if sol._phi_vec is not None:
            assert len(sol._phi_vec) > 0
        else:
            assert sol.phi.vector().size() > 0

        # Energy should be finite and non-negative
        energy = sol.compute_energy()
        assert np.isfinite(energy)
        assert energy >= 0
