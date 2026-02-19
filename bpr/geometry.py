"""
Geometry module for BPR-Math-Spine

Handles triangulation of boundary surfaces (sphere, cylinder) and mesh utilities.
Implements the geometric foundation for solving boundary value problems.

Supports two backends:
  - FEniCS (if installed): full PDE-grade meshing via mshr
  - NumPy fallback: icosphere subdivision + surface FEM assembly
"""

import numpy as np
from scipy import sparse

try:
    import fenics as fe
    import mshr
except ImportError:
    fe = None
    mshr = None


# ═══════════════════════════════════════════════════════════════════════
#  NumPy mesh (always available — no FEniCS needed)
# ═══════════════════════════════════════════════════════════════════════

class NumpyMesh:
    """Lightweight triangulated surface mesh using pure numpy.

    Provides the same API surface that FEniCS meshes expose so that
    downstream code (tests, boundary_field, casimir) works unchanged.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self._topology_dim = 2  # surface triangles
        self._geom_dim = vertices.shape[1]

    # ── FEniCS-compatible API ──────────────────────────────────────────

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_cells(self) -> int:
        return len(self.faces)

    def coordinates(self) -> np.ndarray:
        return self.vertices

    def cells(self) -> np.ndarray:
        return self.faces

    def topology(self):
        return self  # duck-type: topology().dim()

    def dim(self) -> int:
        return self._topology_dim

    def hmax(self) -> float:
        return float(np.max(self._edge_lengths()))

    def hmin(self) -> float:
        return float(np.min(self._edge_lengths()))

    # ── helpers ────────────────────────────────────────────────────────

    def _edge_lengths(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        e0 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
        e1 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
        e2 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
        return np.concatenate([e0, e1, e2])

    def triangle_areas(self) -> np.ndarray:
        """Area of every triangle (vectorised)."""
        v = self.vertices
        f = self.faces
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        cross = np.cross(e1, e2)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def total_area(self) -> float:
        return float(np.sum(self.triangle_areas()))

    # ── surface FEM matrices (P1 elements) ─────────────────────────────

    def assemble_stiffness(self) -> sparse.csr_matrix:
        """Stiffness matrix K: K_ij = Σ_T ∫_T ∇N_i · ∇N_j dA.

        Fully vectorised — no Python loops over triangles.
        """
        nv = self.num_vertices()
        v = self.vertices
        f = self.faces
        nf = len(f)

        p0 = v[f[:, 0]]  # (nf, 3)
        p1 = v[f[:, 1]]
        p2 = v[f[:, 2]]

        # Edge vectors opposite each local vertex
        e0 = p2 - p1  # opposite vertex 0
        e1 = p0 - p2  # opposite vertex 1
        e2 = p1 - p0  # opposite vertex 2

        normal = np.cross(e2, -e1)  # (nf, 3)
        area2 = np.linalg.norm(normal, axis=1, keepdims=True)  # (nf, 1)
        area2_flat = area2.ravel()
        good = area2_flat > 1e-30
        n_hat = np.zeros_like(normal)
        n_hat[good] = normal[good] / area2[good]
        area = 0.5 * area2_flat  # (nf,)

        # Gradients: ∇N_i = (n̂ × e_opp_i) / (2A)
        inv_2A = np.zeros(nf)
        inv_2A[good] = 1.0 / (2.0 * area[good])

        grad0 = np.cross(n_hat, e0) * inv_2A[:, None]  # (nf, 3)
        grad1 = np.cross(n_hat, e1) * inv_2A[:, None]
        grad2 = np.cross(n_hat, e2) * inv_2A[:, None]

        grads = np.stack([grad0, grad1, grad2], axis=1)  # (nf, 3, 3)

        # K_local[i,j] = area * dot(grad_i, grad_j)
        # Compute all 9 dot products at once:  (nf, 3, 3)
        K_local = np.einsum('nik,njk->nij', grads, grads) * area[:, None, None]

        # Assemble into global sparse matrix
        row_idx = f[:, :, None].repeat(3, axis=2).reshape(-1)
        col_idx = f[:, None, :].repeat(3, axis=1).reshape(-1)
        vals = K_local.reshape(-1)

        return sparse.csr_matrix(
            (vals, (row_idx, col_idx)), shape=(nv, nv)
        )

    def assemble_mass(self) -> sparse.csr_matrix:
        """Consistent mass matrix M: M_ij = Σ_T ∫_T N_i N_j dA.

        Fully vectorised — no Python loops over triangles.
        """
        nv = self.num_vertices()
        f = self.faces
        nf = len(f)
        areas = self.triangle_areas()  # (nf,)

        # Local mass matrix: diag = A/6, off-diag = A/12
        # Build (nf, 3, 3)
        local = np.full((nf, 3, 3), 1.0 / 12.0)
        local[:, 0, 0] = 1.0 / 6.0
        local[:, 1, 1] = 1.0 / 6.0
        local[:, 2, 2] = 1.0 / 6.0
        local *= areas[:, None, None]

        row_idx = f[:, :, None].repeat(3, axis=2).reshape(-1)
        col_idx = f[:, None, :].repeat(3, axis=1).reshape(-1)
        vals = local.reshape(-1)

        return sparse.csr_matrix(
            (vals, (row_idx, col_idx)), shape=(nv, nv)
        )

    def assemble_load(self, source_func) -> np.ndarray:
        """Load vector b_i = Σ_T ∫_T f(x) N_i dA.

        Uses vertex-quadrature (exact for P0 source, approximate for smooth):
            b_i += (A/3) * f(x_i) for each triangle containing vertex i.

        The source_func evaluation is vectorised where possible; falls
        back to a loop only if the callable doesn't support arrays.
        """
        nv = self.num_vertices()
        v = self.vertices
        f = self.faces
        areas = self.triangle_areas()  # (nf,)

        # Evaluate source at all vertices
        try:
            fvals = source_func(v[:, 0], v[:, 1], v[:, 2])
            fvals = np.asarray(fvals, dtype=float)
            if fvals.ndim == 0:
                # scalar broadcast (e.g. constant source returning 1.0)
                fvals = np.full(nv, float(fvals))
            elif fvals.shape[0] != nv:
                raise ValueError("shape mismatch")
        except Exception:
            # Scalar fallback
            fvals = np.array([
                source_func(vi[0], vi[1], vi[2] if len(vi) > 2 else 0.0)
                for vi in v
            ], dtype=float)

        # b_i += (A/3) * f(x_i) for every triangle containing vertex i
        contrib = (areas[:, None] / 3.0) * fvals[f]  # (nf, 3)
        b = np.zeros(nv)
        np.add.at(b, f.ravel(), contrib.ravel())
        return b


# ═══════════════════════════════════════════════════════════════════════
#  Icosphere generation (pure numpy)
# ═══════════════════════════════════════════════════════════════════════

def _icosphere(n_subdivisions: int, radius: float = 1.0,
               center=(0.0, 0.0, 0.0)) -> NumpyMesh:
    """Create an icosphere by subdividing an icosahedron.

    Parameters
    ----------
    n_subdivisions : int  (0 = 12 verts, 1 = 42, 2 = 162, 3 = 642, ...)
    radius : float
    center : tuple

    Returns
    -------
    NumpyMesh
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)

    # Subdivide
    for _ in range(n_subdivisions):
        verts, faces = _subdivide(verts, faces)

    # Project onto sphere, scale, translate
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts * radius + np.array(center, dtype=float)

    return NumpyMesh(verts, faces)


def _subdivide(verts, faces):
    """One step of Loop-style subdivision (midpoint, project to sphere later)."""
    edge_midpoints = {}
    new_verts = list(verts)

    def _mid(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_midpoints:
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)  # project to unit sphere
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
        return edge_midpoints[key]

    new_faces = []
    for tri in faces:
        a, b, c_ = tri
        ab = _mid(a, b)
        bc = _mid(b, c_)
        ca = _mid(c_, a)
        new_faces.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c_, ca, bc],
            [ab, bc, ca],
        ])

    return np.array(new_verts, dtype=float), np.array(new_faces, dtype=int)


# ═══════════════════════════════════════════════════════════════════════
#  BoundaryMesh container (works with both backends)
# ═══════════════════════════════════════════════════════════════════════

class BoundaryMesh:
    """Container for boundary mesh data and metadata."""

    def __init__(self, mesh, boundary_markers=None, mesh_size=None):
        self.mesh = mesh
        self.boundary_markers = boundary_markers
        self.mesh_size = mesh_size
        if hasattr(mesh, 'topology'):
            self.dimension = mesh.topology().dim()
        else:
            self.dimension = 2


# ═══════════════════════════════════════════════════════════════════════
#  make_boundary — public entry point
# ═══════════════════════════════════════════════════════════════════════

def make_boundary(mesh_size=0.1, geometry="sphere", radius=1.0, **kwargs):
    """
    Build a triangulated boundary surface.

    Uses FEniCS/mshr if available; otherwise falls back to a pure-numpy
    icosphere triangulation with scipy FEM support.
    """
    if geometry == "sphere":
        return _make_sphere_mesh(mesh_size, radius, **kwargs)
    elif geometry == "cylinder":
        return _make_cylinder_mesh(mesh_size, radius, **kwargs)
    elif geometry == "torus":
        return _make_torus_mesh(mesh_size, radius, **kwargs)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def _make_sphere_mesh(mesh_size, radius, center=(0, 0, 0)):
    """Create a triangulated sphere."""
    if fe is not None and mshr is not None:
        # Full FEniCS path
        center_point = fe.Point(*center)
        sphere = mshr.Sphere(center_point, radius)
        resolution = int(20 / mesh_size)
        mesh = mshr.generate_mesh(sphere, resolution)
        return BoundaryMesh(mesh, mesh_size=mesh_size)

    # ── numpy fallback ──
    n_sub = max(1, int(-np.log2(mesh_size) + 0.5))
    n_sub = min(n_sub, 6)  # cap at 6 (163842 verts) for sanity
    mesh = _icosphere(n_sub, radius=radius, center=center)
    return BoundaryMesh(mesh, mesh_size=mesh_size)


def _make_cylinder_mesh(mesh_size, radius, height=2.0, center=(0, 0, 0), **kwargs):
    """Create a triangulated cylinder."""
    if fe is not None and mshr is not None:
        center_bottom = fe.Point(center[0], center[1], center[2] - height / 2)
        center_top = fe.Point(center[0], center[1], center[2] + height / 2)
        cylinder = mshr.Cylinder(center_bottom, center_top, radius, radius)
        resolution = int(15 / mesh_size)
        mesh = mshr.generate_mesh(cylinder, resolution)
        return BoundaryMesh(mesh, mesh_size=mesh_size)

    # ── numpy fallback: ring-based cylinder surface ──
    n_ring = max(8, int(2 * np.pi * radius / mesh_size))
    n_height = max(2, int(height / mesh_size))

    cx, cy, cz = center

    # Vertices: lateral rings + top/bottom cap centres
    verts = []
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    zs = np.linspace(cz - height / 2, cz + height / 2, n_height + 1)

    for z in zs:
        for t in theta:
            verts.append([cx + radius * np.cos(t),
                          cy + radius * np.sin(t), z])

    # Cap centres
    bot_center_idx = len(verts)
    verts.append([cx, cy, cz - height / 2])
    top_center_idx = len(verts)
    verts.append([cx, cy, cz + height / 2])

    verts = np.array(verts, dtype=float)
    faces = []

    # Lateral faces (quads split into two triangles)
    for i in range(n_height):
        for j in range(n_ring):
            j_next = (j + 1) % n_ring
            a = i * n_ring + j
            b = i * n_ring + j_next
            c = (i + 1) * n_ring + j
            d = (i + 1) * n_ring + j_next
            faces.append([a, b, d])
            faces.append([a, d, c])

    # Bottom cap
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        faces.append([bot_center_idx, j_next, j])

    # Top cap
    top_ring_start = n_height * n_ring
    for j in range(n_ring):
        j_next = (j + 1) % n_ring
        faces.append([top_center_idx,
                      top_ring_start + j,
                      top_ring_start + j_next])

    mesh = NumpyMesh(verts, np.array(faces, dtype=int))
    return BoundaryMesh(mesh, mesh_size=mesh_size)


def _make_torus_mesh(mesh_size, radius, minor_radius=0.3, center=(0, 0, 0)):
    """Create a triangulated torus (placeholder)."""
    raise NotImplementedError("Torus mesh generation not yet implemented")


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def compute_boundary_area(boundary_mesh):
    """Compute surface area of a boundary mesh."""
    mesh = boundary_mesh.mesh
    if isinstance(mesh, NumpyMesh):
        return mesh.total_area()
    if fe is not None:
        return fe.assemble(fe.Constant(1.0) * fe.dx(mesh))
    raise ImportError("Cannot compute area without FEniCS or NumpyMesh")


def get_mesh_quality_metrics(boundary_mesh):
    """Compute mesh quality metrics."""
    mesh = boundary_mesh.mesh

    if isinstance(mesh, NumpyMesh):
        metrics = {
            "num_vertices": mesh.num_vertices(),
            "num_cells": mesh.num_cells(),
            "dimension": mesh.dim(),
            "mesh_size": boundary_mesh.mesh_size,
            "h_max": mesh.hmax(),
            "h_min": mesh.hmin(),
        }
        return metrics

    if fe is not None and hasattr(mesh, 'num_vertices'):
        metrics = {
            "num_vertices": mesh.num_vertices(),
            "num_cells": mesh.num_cells(),
            "dimension": mesh.topology().dim(),
            "mesh_size": boundary_mesh.mesh_size,
        }
        if hasattr(mesh, 'hmax'):
            metrics["h_max"] = mesh.hmax()
            metrics["h_min"] = mesh.hmin()
        return metrics

    return {"error": "Unrecognised mesh type"}
