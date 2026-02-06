"""
FEniCS Integration Tests
========================

These tests require FEniCS (legacy ``dolfin`` or FEniCSX ``dolfinx``).
They are **skipped** automatically when FEniCS is not installed, so
they are safe to run locally.  The Docker CI image includes FEniCS.

Run with:
    pytest tests/test_fenics_integration.py -v

Inside Docker:
    docker-compose run --rm --profile testing bpr-test
"""

import numpy as np
import pytest

# ── Skip entire module when FEniCS is absent ──
fenics = pytest.importorskip("fenics", reason="FEniCS not installed")


class TestMeshGeneration:
    """Verify BPR mesh helpers produce valid FEniCS meshes."""

    def test_make_boundary_sphere(self):
        from bpr.geometry import make_boundary
        mesh = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        assert mesh is not None
        nv = mesh.num_vertices()
        assert nv > 10, f"Sphere mesh too coarse: {nv} vertices"

    def test_make_boundary_cylinder(self):
        from bpr.geometry import make_boundary
        mesh = make_boundary(mesh_size=0.3, geometry="cylinder", radius=1.0, length=2.0)
        assert mesh is not None

    def test_mesh_quality_metrics(self):
        from bpr.geometry import make_boundary, get_mesh_quality_metrics
        mesh = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        quality = get_mesh_quality_metrics(mesh)
        assert "num_vertices" in quality
        assert quality["num_vertices"] > 0


class TestPhaseFieldSolver:
    """Verify solve_phase returns a FEniCS Function."""

    def test_solve_phase_sphere(self):
        from bpr.geometry import make_boundary
        from bpr.boundary_field import solve_phase
        mesh = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        phi = solve_phase(mesh, kappa=1.0)
        assert phi is not None

    def test_solve_phase_produces_finite(self):
        from bpr.geometry import make_boundary
        from bpr.boundary_field import solve_phase
        mesh = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        phi = solve_phase(mesh, kappa=1.0)
        arr = phi.vector().get_local()
        assert np.all(np.isfinite(arr))


class TestCasimirWithFEniCS:
    """End-to-end Casimir test requiring FEniCS for mesh generation."""

    def test_casimir_prediction(self):
        from bpr.casimir import casimir_force
        result = casimir_force(
            distance=1e-6,
            plate_area=1e-4,
            lambda_bpr=1e-12,
        )
        assert result is not None
        # Force should be attractive (negative)
        assert result.force < 0

    def test_sweep_radius(self):
        from bpr.casimir import sweep_radius
        distances = np.linspace(0.5e-6, 5e-6, 5)
        results = sweep_radius(distances, plate_area=1e-4, lambda_bpr=1e-12)
        assert len(results) == 5
        # Force magnitude should decrease with distance
        forces = [r.force for r in results]
        assert abs(forces[0]) >= abs(forces[-1])


class TestMetricPerturbation:
    """Metric perturbation on a real FEniCS mesh (not just symbolic)."""

    def test_metric_perturbation_field(self):
        from bpr.geometry import make_boundary
        from bpr.boundary_field import solve_phase
        mesh = make_boundary(mesh_size=0.4, geometry="sphere", radius=1.0)
        phi = solve_phase(mesh, kappa=1.0)
        # We can at least verify the field was computed
        assert phi is not None
        assert phi.vector().size() > 0
