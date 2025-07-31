"""
Geometry module for BPR-Math-Spine

Handles triangulation of boundary surfaces (sphere, cylinder) and FEniCS mesh utilities.
Implements the geometric foundation for solving boundary value problems.
"""

import numpy as np
try:
    import fenics as fe
    import mshr
except ImportError:
    # Fallback for systems without FEniCS
    fe = None
    mshr = None


class BoundaryMesh:
    """Container for boundary mesh data and metadata."""
    
    def __init__(self, mesh, boundary_markers=None, mesh_size=None):
        self.mesh = mesh
        self.boundary_markers = boundary_markers
        self.mesh_size = mesh_size
        self.dimension = mesh.topology().dim() if hasattr(mesh, 'topology') else 2


def make_boundary(mesh_size=0.1, geometry="sphere", radius=1.0, **kwargs):
    """
    Build a triangulated boundary surface.
    
    Parameters
    ----------
    mesh_size : float
        Characteristic mesh size parameter
    geometry : str
        Type of boundary: "sphere", "cylinder", "torus"
    radius : float
        Characteristic radius of the geometry
    **kwargs : dict
        Additional geometry-specific parameters
        
    Returns
    -------
    BoundaryMesh
        Container with mesh and associated data
        
    Examples
    --------
    >>> mesh = make_boundary(mesh_size=0.05, geometry="sphere", radius=1.0)
    >>> print(f"Mesh has {mesh.mesh.num_vertices()} vertices")
    """
    if fe is None:
        raise ImportError("FEniCS not available. Install with: conda install fenics")
        
    if geometry == "sphere":
        return _make_sphere_mesh(mesh_size, radius, **kwargs)
    elif geometry == "cylinder":
        return _make_cylinder_mesh(mesh_size, radius, **kwargs)
    elif geometry == "torus":
        return _make_torus_mesh(mesh_size, radius, **kwargs)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _make_sphere_mesh(mesh_size, radius, center=(0, 0, 0)):
    """Create a triangulated sphere using mshr."""
    if mshr is None:
        # Fallback: create unit sphere manually
        return _make_unit_sphere_manual(mesh_size, radius, center)
    
    # Use mshr for more robust mesh generation
    center_point = fe.Point(*center)
    sphere = mshr.Sphere(center_point, radius)
    
    # Generate mesh with specified resolution
    resolution = int(20 / mesh_size)  # Adaptive resolution
    mesh = mshr.generate_mesh(sphere, resolution)
    
    return BoundaryMesh(mesh, mesh_size=mesh_size)


def _make_unit_sphere_manual(mesh_size, radius, center):
    """Manual sphere mesh generation as fallback."""
    # Create icosphere-based triangulation
    n_refinements = max(1, int(-np.log2(mesh_size)))
    
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Scale and translate
    vertices = vertices * radius + np.array(center)
    
    # Basic icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    
    # Create FEniCS mesh manually
    if fe is not None:
        mesh = fe.Mesh()
        editor = fe.MeshEditor()
        editor.open(mesh, "triangle", 2, 3)
        
        editor.init_vertices(len(vertices))
        for i, vertex in enumerate(vertices):
            editor.add_vertex(i, vertex)
            
        editor.init_cells(len(faces))
        for i, face in enumerate(faces):
            editor.add_cell(i, face)
            
        editor.close()
        return BoundaryMesh(mesh, mesh_size=mesh_size)
    else:
        # Return raw data if FEniCS unavailable
        return BoundaryMesh({"vertices": vertices, "faces": faces}, mesh_size=mesh_size)


def _make_cylinder_mesh(mesh_size, radius, height=2.0, center=(0, 0, 0)):
    """Create a triangulated cylinder."""
    if mshr is None:
        raise NotImplementedError("Cylinder mesh requires mshr")
    
    # Create cylinder geometry
    center_bottom = fe.Point(center[0], center[1], center[2] - height/2)
    center_top = fe.Point(center[0], center[1], center[2] + height/2)
    cylinder = mshr.Cylinder(center_bottom, center_top, radius, radius)
    
    resolution = int(15 / mesh_size)
    mesh = mshr.generate_mesh(cylinder, resolution)
    
    return BoundaryMesh(mesh, mesh_size=mesh_size)


def _make_torus_mesh(mesh_size, radius, minor_radius=0.3, center=(0, 0, 0)):
    """Create a triangulated torus (placeholder for future implementation)."""
    raise NotImplementedError("Torus mesh generation not yet implemented")


def compute_boundary_area(boundary_mesh):
    """
    Compute the surface area of a boundary mesh.
    
    Parameters
    ----------
    boundary_mesh : BoundaryMesh
        The mesh to compute area for
        
    Returns
    -------
    float
        Total surface area
    """
    if fe is None:
        raise ImportError("FEniCS required for area computation")
        
    mesh = boundary_mesh.mesh
    return fe.assemble(fe.Constant(1.0) * fe.dx(mesh))


def get_mesh_quality_metrics(boundary_mesh):
    """
    Compute mesh quality metrics.
    
    Parameters
    ----------
    boundary_mesh : BoundaryMesh
        The mesh to analyze
        
    Returns
    -------
    dict
        Dictionary containing quality metrics
    """
    if fe is None or not hasattr(boundary_mesh.mesh, 'num_vertices'):
        return {"error": "FEniCS mesh required"}
    
    mesh = boundary_mesh.mesh
    
    metrics = {
        "num_vertices": mesh.num_vertices(),
        "num_cells": mesh.num_cells(),
        "dimension": mesh.topology().dim(),
        "mesh_size": boundary_mesh.mesh_size
    }
    
    # Compute mesh size statistics
    if hasattr(mesh, 'hmax'):
        metrics["h_max"] = mesh.hmax()
        metrics["h_min"] = mesh.hmin()
    
    return metrics