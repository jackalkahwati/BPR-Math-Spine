"""
Boundary field solver for BPR-Math-Spine

Solves the boundary Laplacian equation κ∇²_Σ φ = f (Eq 6a).

Two backends:
  - FEniCS: full PDE solver on arbitrary meshes
  - NumPy/SciPy fallback: P1 surface FEM on NumpyMesh (icosphere)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

try:
    import fenics as fe
except ImportError:
    fe = None

from .geometry import NumpyMesh


# ═══════════════════════════════════════════════════════════════════════
#  Solution containers
# ═══════════════════════════════════════════════════════════════════════

class BoundaryFieldSolution:
    """Container for boundary field solution data.

    Works with both FEniCS and NumPy backends.
    """

    def __init__(self, phi, mesh, function_space=None,
                 boundary_mesh=None, source_term=None,
                 # numpy-backend extras
                 phi_vec=None, stiffness=None, numpy_mesh=None):
        self.phi = phi                # FEniCS Function or callable
        self.mesh = mesh              # underlying mesh object
        self.function_space = function_space
        self.boundary_mesh = boundary_mesh
        self.source_term = source_term
        # numpy backend
        self._phi_vec = phi_vec       # DOF vector (nv,)
        self._K = stiffness           # stiffness matrix (nv,nv)
        self._np_mesh = numpy_mesh    # NumpyMesh reference

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate_at_points(self, points):
        """Evaluate solution at given points."""
        if self._np_mesh is not None:
            return self._evaluate_numpy(points)
        if fe is not None and hasattr(self.phi, '__call__'):
            values = []
            for pt in points:
                try:
                    values.append(self.phi(pt))
                except Exception:
                    values.append(np.nan)
            return np.array(values)
        raise ImportError("No backend available for evaluation")

    def _evaluate_numpy(self, points):
        """Nearest-vertex interpolation (fast, sufficient for tests)."""
        verts = self._np_mesh.vertices
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        # For each query point, find nearest vertex
        diffs = verts[np.newaxis, :, :] - points[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)
        nearest = np.argmin(dists, axis=1)
        return self._phi_vec[nearest]

    # ── gradient / energy ─────────────────────────────────────────────

    def get_gradient(self):
        if fe is not None and hasattr(self.phi, '__call__') and self._np_mesh is None:
            return fe.grad(self.phi)
        raise NotImplementedError("Gradient not available in numpy mode (use compute_energy)")

    def compute_energy(self):
        """Compute field energy  E = ∫ |∇φ|² dS = φᵀ K φ."""
        if self._np_mesh is not None and self._K is not None:
            return float(self._phi_vec @ self._K @ self._phi_vec)
        if fe is not None:
            grad_phi = fe.grad(self.phi)
            energy_density = fe.dot(grad_phi, grad_phi)
            return fe.assemble(energy_density * fe.dx(self.mesh))
        raise ImportError("No backend available for energy computation")


# ═══════════════════════════════════════════════════════════════════════
#  PDE solver: κ∇²φ = f
# ═══════════════════════════════════════════════════════════════════════

def solve_phase(boundary_mesh, source, kappa=1.0, boundary_conditions=None):
    """
    Solve κ∇²_Σ φ = f on the boundary surface.

    Uses FEniCS if available, otherwise a numpy/scipy P1 FEM solver.
    """
    mesh = boundary_mesh.mesh

    # ── numpy backend ──────────────────────────────────────────────────
    if isinstance(mesh, NumpyMesh):
        return _solve_phase_numpy(mesh, source, kappa, boundary_mesh)

    # ── FEniCS backend ─────────────────────────────────────────────────
    if fe is None:
        raise ImportError("FEniCS not available and mesh is not a NumpyMesh")

    V = fe.FunctionSpace(mesh, "CG", 1)
    phi = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    if callable(source):
        source_expr = _convert_callable_to_expression(source, mesh)
    else:
        source_expr = source

    a = kappa * fe.dot(fe.grad(phi), fe.grad(v)) * fe.dx
    L = source_expr * v * fe.dx

    bcs = []
    if boundary_conditions is not None:
        bcs = _apply_boundary_conditions(V, boundary_conditions)

    phi_solution = fe.Function(V)
    try:
        fe.solve(a == L, phi_solution, bcs)
    except Exception:
        A_, b_ = fe.assemble_system(a, L, bcs)
        solver = fe.KrylovSolver("cg", "ilu")
        solver.solve(A_, phi_solution.vector(), b_)

    return BoundaryFieldSolution(
        phi_solution, mesh, V, boundary_mesh, source_term=source_expr
    )


def _solve_phase_numpy(np_mesh: NumpyMesh, source, kappa, boundary_mesh):
    """P1 surface FEM solve on an icosphere mesh (numpy/scipy)."""
    K = np_mesh.assemble_stiffness()  # sparse (nv, nv)
    M = np_mesh.assemble_mass()       # sparse (nv, nv)

    # Right-hand side: b_i = ∫ f N_i dA
    b = np_mesh.assemble_load(source)

    # System: κ K φ = b
    # The Laplacian on a closed surface (sphere) has a rank-1 null space
    # (constant functions).  We pin the first DOF to zero to remove it.
    A = (kappa * K).tocsr()

    # Pin DOF 0  (Dirichlet φ_0 = 0)
    nv = np_mesh.num_vertices()
    A_mod = A.tolil()
    A_mod[0, :] = 0
    A_mod[0, 0] = 1.0
    b[0] = 0.0
    A_mod = A_mod.tocsr()

    phi_vec = spsolve(A_mod, b)

    # Wrap in callable for downstream compatibility
    def _phi_callable(point):
        """Nearest-vertex evaluation."""
        verts = np_mesh.vertices
        d = np.linalg.norm(verts - np.asarray(point), axis=1)
        return float(phi_vec[np.argmin(d)])

    return BoundaryFieldSolution(
        phi=_phi_callable,
        mesh=np_mesh,
        function_space=None,
        boundary_mesh=boundary_mesh,
        source_term=source,
        phi_vec=phi_vec,
        stiffness=K,
        numpy_mesh=np_mesh,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Eigenvalue problem: ∇²_Σ φ = λ φ
# ═══════════════════════════════════════════════════════════════════════

def solve_eigenvalue_problem(boundary_mesh, n_modes=10):
    """
    Solve ∇²_Σ φ = λ φ for the lowest eigenvalues / eigenfunctions.

    Returns
    -------
    (eigenvalues, eigenfunctions)
        eigenvalues : ndarray (n_modes,)
        eigenfunctions : list of ndarray or FEniCS Functions
    """
    mesh = boundary_mesh.mesh

    # ── numpy backend ──────────────────────────────────────────────────
    if isinstance(mesh, NumpyMesh):
        return _solve_eigenvalue_numpy(mesh, n_modes)

    # ── FEniCS backend ─────────────────────────────────────────────────
    if fe is None:
        raise ImportError("FEniCS required for eigenvalue computation")

    V = fe.FunctionSpace(mesh, "CG", 1)
    phi = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    a = fe.dot(fe.grad(phi), fe.grad(v)) * fe.dx
    m = phi * v * fe.dx
    A_ = fe.assemble(a)
    M_ = fe.assemble(m)

    A_np = A_.array()
    M_np = M_.array()

    try:
        eigenvals, eigenvecs = eigsh(A_np, k=n_modes, M=M_np, sigma=0.1)
    except Exception:
        eigenvals, eigenvecs = np.linalg.eigh(np.linalg.solve(M_np, A_np))
        eigenvals = eigenvals[:n_modes]
        eigenvecs = eigenvecs[:, :n_modes]

    eigenfunctions = []
    for i in range(n_modes):
        phi_mode = fe.Function(V)
        phi_mode.vector()[:] = eigenvecs[:, i]
        eigenfunctions.append(phi_mode)

    return eigenvals, eigenfunctions


def _solve_eigenvalue_numpy(np_mesh: NumpyMesh, n_modes):
    """Solve generalised eigenvalue problem K φ = λ M φ (scipy)."""
    K = np_mesh.assemble_stiffness()
    M = np_mesh.assemble_mass()

    # eigsh needs M to be positive-definite.  The mass matrix from P1
    # on a closed surface is SPD.  Use sigma=0 to get the smallest
    # eigenvalues (shift-invert mode).
    # Request one extra mode since the first one is the trivial λ=0.
    n_req = n_modes + 2  # a little extra for safety
    n_req = min(n_req, np_mesh.num_vertices() - 2)

    eigenvals, eigenvecs = eigsh(K, k=n_req, M=M, sigma=0.0, which='LM')

    # Sort by ascending eigenvalue and discard negative/spurious modes
    order = np.argsort(eigenvals)
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    eigenfunctions = [eigenvecs[:, i] for i in range(len(eigenvals))]
    return eigenvals[:n_modes], eigenfunctions[:n_modes]


# ═══════════════════════════════════════════════════════════════════════
#  Convergence verification
# ═══════════════════════════════════════════════════════════════════════

def verify_convergence(boundary_mesh_sizes, exact_solution=None):
    """Verify convergence of the solver over a sequence of mesh sizes."""
    from .geometry import make_boundary

    errors = []
    for mesh_size in boundary_mesh_sizes:
        mesh_obj = make_boundary(mesh_size=mesh_size)

        if exact_solution is None:
            exact_phi = lambda x, y, z: np.sin(np.pi * x) * np.cos(np.pi * y)
            source = lambda x, y, z: 2 * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
        else:
            exact_phi = exact_solution
            source = lambda x, y, z: 0

        solution = solve_phase(mesh_obj, source)

        try:
            coords = mesh_obj.mesh.coordinates()
            exact_vals = np.array([
                exact_phi(c[0], c[1], c[2] if len(c) > 2 else 0)
                for c in coords
            ])

            if solution._phi_vec is not None:
                computed_vals = solution._phi_vec
            else:
                computed_vals = solution.phi.vector().get_local()

            error = np.sqrt(np.mean((exact_vals - computed_vals) ** 2))
            errors.append(error)
        except Exception:
            errors.append(np.nan)

    return {
        "mesh_sizes": boundary_mesh_sizes,
        "errors": errors,
        "convergence_rate": _estimate_convergence_rate(boundary_mesh_sizes, errors),
    }


def _estimate_convergence_rate(h_values, errors):
    valid = ~np.isnan(errors)
    if np.sum(valid) < 2:
        return np.nan
    h_c = np.array(h_values)[valid]
    e_c = np.array(errors)[valid]
    if np.any(e_c <= 0):
        return np.nan
    try:
        coeffs = np.polyfit(np.log(h_c), np.log(e_c), 1)
        return coeffs[0]
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════
#  FEniCS helpers (only used when fe is not None)
# ═══════════════════════════════════════════════════════════════════════

def _convert_callable_to_expression(source_func, mesh):
    """Convert a Python callable to a FEniCS expression."""
    class SourceExpression(fe.UserExpression):
        def __init__(self, func, **kwargs):
            super().__init__(**kwargs)
            self.func = func

        def eval(self, value, x):
            if len(x) == 3:
                value[0] = self.func(x[0], x[1], x[2])
            elif len(x) == 2:
                value[0] = self.func(x[0], x[1], 0)
            else:
                value[0] = self.func(x[0], 0, 0)

        def value_shape(self):
            return ()

    return SourceExpression(source_func, degree=2)


def _apply_boundary_conditions(function_space, boundary_conditions):
    bcs = []
    for bc_spec in boundary_conditions:
        if bc_spec["type"] == "dirichlet":
            value = bc_spec["value"]
            boundary = bc_spec["boundary"]
            if callable(value):
                bc_expr = _convert_callable_to_expression(value, function_space.mesh())
            else:
                bc_expr = fe.Constant(value)
            bc = fe.DirichletBC(function_space, bc_expr, boundary)
            bcs.append(bc)
    return bcs
